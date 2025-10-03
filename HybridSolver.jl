"""
HybridSolver.jl - Hybrid systems solver with ForwardDiff support.
Implements hybrid time domain (t,j) with ordering (t,j) ≼ (s,k) iff t+j ≤ s+k.
"""

module HybridSolver

using OrdinaryDiffEq, Plots, Statistics, LaTeXStrings, LinearAlgebra
import Plots: plot, plot!
import Base: <, <=, ==, isless, show
import CommonSolve: solve

# Default plotting style
default(
    fontfamily="Computer Modern",
    linewidth=2, markersize=4, markerstrokewidth=1,
    grid=true, gridstyle=:dash, gridalpha=0.75, gridcolor=:lightgray,
    framestyle=:box, tick_direction=:in, minorticks=false,
    left_margin=4Plots.mm, bottom_margin=2Plots.mm,
    right_margin=4Plots.mm, top_margin=2Plots.mm,
    titlefontsize=16, guidefontsize=14, tickfontsize=10, legendfontsize=14,
    camera=(30, 30), size=(800, 600)
)

export HybridSystem, HybridTime, HybridState, HybridSolution,
    HybridSolverConfig, FlowSegment, JumpTransition,
    flow_set, jump_set, flow_map!, jump_map,
    solve, extract_flow_segments, extract_jump_transitions,
    get_state, plot, vline!, vspan!, hline!, legend!, current_hybrid_plot, HybridPlot

################################################################################
#### TYPE DEFINITIONS
################################################################################

abstract type HybridSystem end

flow_set(sys::HybridSystem, x, t, j) = error("flow_set not implemented for $(typeof(sys))")
jump_set(sys::HybridSystem, x, t, j) = error("jump_set not implemented for $(typeof(sys))")
flow_map!(sys::HybridSystem, dx, x, p) = error("flow_map! not implemented for $(typeof(sys))")
jump_map(sys::HybridSystem, x, p) = error("jump_map not implemented for $(typeof(sys))")

struct HybridTime{T<:Real}
    t::T
    j::Int
end

HybridTime(t::Real, j::Int) = HybridTime{typeof(t)}(t, j)
Base.isless(h1::HybridTime, h2::HybridTime) = h1.t + h1.j < h2.t + h2.j
Base.:(==)(h1::HybridTime, h2::HybridTime) = h1.t == h2.t && h1.j == h2.j
Base.show(io::IO, h::HybridTime) = print(io, "($(h.t), $(h.j))")

struct HybridState{T}
    time::HybridTime
    state::Vector{T}
end

Base.show(io::IO, hs::HybridState) = print(io, "$(hs.time) → $(hs.state)")

struct HybridSolution{T}
    points::Vector{HybridState{T}}
    jump_indices::Vector{Int}
    initial_condition::Vector{T}
    final_time::T
    total_jumps::Int
    termination_condition::Symbol
end

struct FlowSegment{T}
    j::Int
    t_start::T
    t_end::T
    states::Vector{Vector{T}}
    times::Vector{T}
end

struct JumpTransition{T}
    j::Int
    tau::T
    x_minus::Vector{T}
    x_plus::Vector{T}
end

Base.@kwdef struct HybridSolverConfig
    # Integration parameters
    solver = Tsit5()
    dt::Float64 = 1e-3
    dtmax::Float64 = 1e-1
    abstol::Float64 = 1e-6
    reltol::Float64 = 1e-4

    # Hybrid system behavior
    jump_priority::Bool = true  # If both flow and jump sets are satisfied, prioritize jumps
    boundary_search_steps::Int = 5

    # System behavior
    verbose_exit::Bool = false      # Print termination reason
    verbose::Bool = false
end

################################################################################
#### CORE SOLVING
################################################################################

function _check_state_status(sys, x, t, j, jump_priority)
    if any(isnan, x) || any(isinf, x)
        return :invalid_state
    end

    inside_flow = flow_set(sys, x, t, j)
    inside_jump = jump_set(sys, x, t, j)

    if !inside_flow && !inside_jump
        return :outside_domain
    elseif inside_flow && (!inside_jump || !jump_priority)
        return :should_flow
    elseif inside_jump && (!inside_flow || jump_priority)
        return :should_jump
    else
        return :outside_domain
    end
end


"""
solve(sys, x0, max_hybrid_time; config, p) - Evolve until t+j ≤ max_hybrid_time.
solve(sys, x0, t_max, j_max; config, p) - Evolve until t ≤ t_max or j ≤ j_max.
"""
function solve(sys::HybridSystem, x0::Vector{<:AbstractFloat}, max_hybrid_time::Real; config::HybridSolverConfig=HybridSolverConfig(), p=nothing)
    any(isnan, x0) ? error("Input cannot have NaN values") :
    t_max = Float64(max_hybrid_time)
    j_max = floor(Int, max_hybrid_time)
    # Array sizing with safety buffer
    max_flow_steps = ceil(Int, t_max / config.dtmax)
    max_jump_steps = j_max + 1000  # Add buffer for safety
    max_total_points = max_flow_steps + max_jump_steps + 1000  # Buffer for ForwardDiff

    # Determine element type - use parameter type for ForwardDiff compatibility
    T = (p === nothing) ? eltype(x0) : promote_type(eltype(x0), eltype(p))
    times_array = zeros(T, max_total_points)
    states_array = zeros(T, length(x0), max_total_points)
    jumps_array = zeros(Int, max_total_points)
    jump_indices_array = zeros(Int, max_jump_steps)

    # Initialize evolution state
    j_current = [0]
    point_count = [1]
    jump_count = [0]

    # Helper function to save a point
    function save_point!(state, is_jump=false)
        point_count[1] += 1

        # Bounds check to prevent array overflow
        if point_count[1] > max_total_points
            error("Array overflow: simulation exceeded $(max_total_points) points.")
        end

        times_array[point_count[1]] = integrator.t
        states_array[:, point_count[1]] = state
        jumps_array[point_count[1]] = j_current[1]

        if is_jump
            jump_count[1] += 1
            if jump_count[1] > max_jump_steps
                error("Jump array overflow: simulation exceeded $(max_jump_steps) jumps.")
            end
            jump_indices_array[jump_count[1]] = point_count[1]
        end
    end

    function ode_func!(dx, x, params, t)
        flow_map!(sys, dx, x, params)
    end


    # Create integrator - ForwardDiff compatibility via parameter passing
    tspan = (0.0, t_max)
    if p === nothing
        prob = ODEProblem(ode_func!, x0, tspan)
    else
        prob = ODEProblem(ode_func!, x0, tspan, p)
    end


    # Initialize integrator with callback
    integrator = init(prob, config.solver;
        dt=config.dt, dtmax=config.dtmax,
        abstol=config.abstol, reltol=config.reltol,
        save_everystep=false
    )

    # Add initial point
    times_array[1] = 0.0
    states_array[:, 1] = x0
    jumps_array[1] = 0

    config.verbose && println("Starting hybrid solution from x₀ = $x0")

    # Track termination condition
    termination_condition = :max_limits_reached

    # Main evolution loop
    while integrator.t + j_current[1] <= max_hybrid_time
        # Check current state using consolidated function
        state_status = _check_state_status(sys, integrator.u, integrator.t, j_current[1], config.jump_priority)

        if state_status in [:outside_domain, :invalid_state]
            # Handle termination cases
            if state_status == :invalid_state
                termination_condition = :invalid_state
            else  # :outside_domain
                termination_condition = :outside_domain
            end
            if config.verbose_exit || config.verbose
                println("Terminating: $state_status at t=$(integrator.t), j=$(j_current[1])")
            end
            break

        elseif state_status == :should_jump
            # Apply jump map if j + 1 does not exceeded max_time
            if integrator.t + j_current[1] + 1 <= max_hybrid_time
                x_new = jump_map(sys, integrator.u, p)
                integrator.u[1:end] = x_new
                j_current[1] += 1
                save_point!(x_new, true)
            else
                if config.verbose_exit || config.verbose
                    println("Terminating: $state_status at t=$(integrator.t), j=$(j_current[1])."*
                                "Solution can jump but would exceeded maximum allowed time.")
                end
                termination_condition = :max_limits_reached
                break
            end

        elseif state_status == :should_flow
            # Store previous state for boundary overshoot detection
            prev_t = integrator.t
            prev_x = copy(integrator.u)

            # Take integration step and capture result
            step!(integrator)

            # Check for non-progression
            if integrator.t == prev_t
                dx = similar(integrator.u)
                flow_map!(sys, dx, integrator.u, p)
                flow_magnitude = norm(dx)

                @warn "Solver not progressing at t=$(integrator.t), j=$(j_current[1]), |f(x,t,j)|=$(round(flow_magnitude, digits=3)). Setting final state to NaN and terminating."

                # Set state to NaN and terminate
                integrator.u .= NaN
                save_point!(integrator.u)
                termination_condition = :solver_not_progressing
                break
            end

            # Store the ODE solution
            ode_solution_x = copy(integrator.u)
            ode_solution_t = integrator.t

            # Check if we overshot boundary and need fallback
            ode_state_status = _check_state_status(sys, ode_solution_x, ode_solution_t, j_current[1], config.jump_priority)
            boundary_crossed = (ode_state_status != :should_flow)

            if boundary_crossed && integrator.t > prev_t
                # We crossed boundary but callback didn't stop us - use Euler fallback
                dt_ode = ode_solution_t - prev_t
                x_corrected, t_corrected = _euler_boundary_search(sys, prev_x, prev_t, ode_solution_x, ode_solution_t, dt_ode, j_current[1], p,
                    config.jump_priority, config.boundary_search_steps)

                # Update integrator state to corrected boundary point
                integrator.t = t_corrected
                integrator.u[1:end] = x_corrected
            end

            # Check for infinity/NaN states
            if any(isnan.(integrator.u)) || any(isinf.(integrator.u))
                if config.verbose
                    println("Invalid state detected at t=$(integrator.t), j=$(j_current[1]): NaN or Inf values")
                end
                termination_condition = :invalid_state
                break
            end

            # Save flow state
            save_point!(integrator.u)

        else
            # Outside both flow and jump sets - terminate
            termination_condition = :outside_domain
            if config.verbose_exit || config.verbose
                println("Terminating: outside domain at t=$(integrator.t), j=$(j_current[1])")
            end
            break
        end
    end

    # Convert to final trajectory structure
    n_points = point_count[1]
    points = Vector{HybridState{T}}(undef, n_points)

    # Resize arrays and create points
    resize!(times_array, n_points)
    resize!(jumps_array, n_points)

    for i in 1:n_points
        points[i] = HybridState(
            HybridTime(times_array[i], jumps_array[i]),
            states_array[:, i]
        )
    end

    n_jumps = jump_count[1]
    jump_indices = jump_indices_array[1:n_jumps]

    # Convert initial condition to same type as solution points
    x0_converted = convert(Vector{T}, copy(x0))
    final_time_converted = convert(T, integrator.t)  # Convert to same type T for ForwardDiff compatibility
    solution = HybridSolution(
        points, jump_indices, x0_converted,
        final_time_converted, j_current[1], termination_condition
    )

    if config.verbose
        termination_msg = if termination_condition == :max_limits_reached
            "reached time/jump limits"
        elseif termination_condition == :outside_domain
            "outside flow/jump domain"
        else
            string(termination_condition)
        end
        println("Solution complete: $(j_current[1]) jumps, final time $(round(integrator.t, digits=3)), terminated: $termination_msg")
    end
    return solution
end

"""
Core hybrid system solver implementation with separate time and jump limits
"""
function solve(sys::HybridSystem, x0::Vector{<:AbstractFloat}, t_max::Real, j_max::Int; config::HybridSolverConfig=HybridSolverConfig(), p=nothing)
    max_hybrid_time = t_max + j_max
    return solve(sys, x0, max_hybrid_time; config=config, p=p)
end

""" Bisection search for boundary crossing when ODE overshoots """
function _euler_boundary_search(sys::HybridSystem,
    x_prev, t_prev,
    x_ode, t_ode, dt_ode,
    j, p, jump_priority::Bool,
    boundary_search_steps::Int)

    # Helper function to check if we should still flow from a state
    function should_flow_from(x, t)
        in_flow = flow_set(sys, x, t, j)
        in_jump = jump_set(sys, x, t, j)
        return in_flow && (!jump_priority || !in_jump)
    end

    # Bisection search for boundary crossing point
    dt_left = zero(dt_ode)  # Zero step always stays in flow set (ForwardDiff compatible)
    dt_right = dt_ode      
    x_boundary = x_ode  # Default boundary point
    t_boundary = t_ode

    for _ in 1:boundary_search_steps  #
        dt_mid = (dt_left + dt_right) / 2

        # Check convergence - stop when interval is small enough
        if abs(dt_right - dt_left) < 1e-12
            break
        end

        # Euler step from previous good state to midpoint
        dx = similar(x_prev)
        flow_map!(sys, dx, x_prev, p)
        x_test = x_prev + dt_mid * dx
        t_test = t_prev + dt_mid

        if should_flow_from(x_test, t_test)
            # Still in flow set, boundary is to the right
            dt_left = dt_mid
        else
            # Exited flow set, boundary is to the left - store this point
            dt_right = dt_mid
            x_boundary = x_test
            t_boundary = t_test
        end
    end

    # If we couldn't find any good step size (all steps overshoot), 
    # boundary defaults to original ODE solution (already set above)

    return x_boundary, t_boundary
end

################################################################################
#### SOLUTION ANALYSIS AND EXTRACTION
################################################################################

"""
Parse a query string like "t<20" or "j>=5" 
"""
function parse_query(query::String)
    # Match patterns like "t<20", "j >= 5", "t <= 20", etc.
    m = match(r"([tj])\s*([<>=]+)\s*(\d+(?:\.\d+)?)", query)
    if m === nothing
        error("Invalid query format: $query. Expected format like 't<20' or 'j >= 5'")
    end

    var = Symbol(m.captures[1])
    op_str = m.captures[2]
    val = parse(Float64, m.captures[3])

    # Convert string operator to function
    if op_str == "<"
        op = <
    elseif op_str == "<="
        op = <=
    elseif op_str == ">"
        op = >
    elseif op_str == ">="
        op = >=
    elseif op_str == "=="
        op = ==
    else
        error("Unsupported operator: $op_str")
    end

    return var, op, val
end

"""
Index into a hybrid solution using a string query
"""
function Base.getindex(sol::HybridSolution, query::String)
    var, op, val = parse_query(query)

    if var == :t
        domain = [p.time.t for p in sol.points]
    else  # var == :j
        domain = [p.time.j for p in sol.points]
    end

    mask = broadcast(op, domain, val)
    return Base.getindex(sol, mask)
end

function Base.getindex(sol::HybridSolution, mask::BitVector)
    filtered_points = sol.points[mask]

    # Filter jump indices efficiently: map original indices to new indices
    old_to_new_idx = cumsum(mask)  # Maps old index to new index
    new_jump_indices = Int[]

    for old_jump_idx in sol.jump_indices
        if old_jump_idx <= length(mask) && mask[old_jump_idx]  # If this jump point is included
            push!(new_jump_indices, old_to_new_idx[old_jump_idx])
        end
    end

    return HybridSolution(
        filtered_points, new_jump_indices, sol.initial_condition,
        length(filtered_points) > 0 ? filtered_points[end].time.t : 0.0,
        length(filtered_points) > 0 ? filtered_points[end].time.j : 0,
        :filtered
    )
end

"""
Get state at specific hybrid time (with interpolation if needed)
"""
function get_state(traj::HybridSolution, target::HybridTime)
    for (i, point) in enumerate(traj.points)
        if point.time == target
            return point.state
        elseif point.time.j == target.j && point.time.t >= target.t
            if i > 1 && traj.points[i-1].time.j == target.j
                prev_point = traj.points[i-1]
                dt = point.time.t - prev_point.time.t
                if dt > 0
                    α = (target.t - prev_point.time.t) / dt
                    return prev_point.state + α * (point.state - prev_point.state)
                end
            end
            return point.state
        end
    end
    error("Hybrid time point $target not found in trajectory")
end

"""
Extract flow segments from hybrid solution
"""
function extract_flow_segments(traj::HybridSolution{T}) where T
    segments = FlowSegment{T}[]

    if isempty(traj.points)
        return segments
    end

    current_j = traj.points[1].time.j
    segment_start_idx = 1

    for i in 2:lastindex(traj.points)
        point = traj.points[i]
        if point.time.j != current_j
            end_idx = i - 1
            if end_idx >= segment_start_idx
                segment_points = view(traj.points, segment_start_idx:end_idx)
                states = [p.state for p in segment_points]
                times = [p.time.t for p in segment_points]

                push!(segments, FlowSegment(
                    current_j,
                    first(times), last(times),
                    states, times
                ))
            end

            current_j = point.time.j
            segment_start_idx = i
        end
    end

    # Handle final segment
    if segment_start_idx <= lastindex(traj.points)
        final_points = view(traj.points, segment_start_idx:lastindex(traj.points))
        states = [p.state for p in final_points]
        times = [p.time.t for p in final_points]

        push!(segments, FlowSegment(
            current_j,
            first(times), last(times),
            states, times
        ))
    end

    return segments
end

"""
Extract jump transitions from hybrid solution  
"""
function extract_jump_transitions(traj::HybridSolution{T}) where T
    transitions = JumpTransition{T}[]

    for jump_idx in traj.jump_indices
        if 1 < jump_idx <= lastindex(traj.points)
            pre_jump = traj.points[jump_idx-1]
            post_jump = traj.points[jump_idx]

            push!(transitions, JumpTransition(
                post_jump.time.j,
                post_jump.time.t,
                pre_jump.state,
                post_jump.state
            ))
        end
    end

    return transitions
end

################################################################################
#### PLOTTING FUNCTIONALITY
################################################################################

"""
Custom plot wrapper for hybrid solutions that maintains hybrid-specific metadata
and supports all hybrid plot types with plot! functionality.
"""
struct HybridPlot
    plot_obj  # Union{Plot, Tuple{Plot,Plot}, Plot3D} - underlying plot(s)
    plot_type::Symbol  # :hybrid, :time_only, :jumps_only, :hybrid3d
    title::Union{String,Nothing}
    ylabel::String
end

# Global variable to track current HybridPlot
const CURRENT_HYBRID_PLOT = Ref{Union{HybridPlot,Nothing}}(nothing)

"""
Get the current HybridPlot (similar to Plots.jl's current())
"""
current_hybrid_plot() = CURRENT_HYBRID_PLOT[]

"""
Set the current HybridPlot (similar to Plots.jl's current())
"""
function current_hybrid_plot!(hplot::HybridPlot)
    CURRENT_HYBRID_PLOT[] = hplot
    return hplot
end

"""
Create a new HybridPlot with the specified type and initial styling.
"""
function HybridPlot(type::Symbol; title=nothing, ylabel=L"$x$", kwargs...)
    if type ∉ [:hybrid, :time_only, :jumps_only, :hybrid3d]
        error("HybridPlot type must be :hybrid, :time_only, :jumps_only, or :hybrid3d")
    end

    if type == :hybrid
        # Create dual subplot structure
        plt_vs_t = plot(xlabel=L"$t$", ylabel=ylabel, title=""; kwargs...)
        plt_vs_j = plot(xlabel=L"$j$", ylabel="", title=""; kwargs...)
        plot_obj = (plt_vs_t, plt_vs_j)
        # Title is handled at layout level for hybrid plots
        plot_title = title
    elseif type == :time_only
        plot_title = title === nothing ? L"$x$ vs time" : title
        plot_obj = plot(xlabel=L"$t$", ylabel=ylabel, title=plot_title; kwargs...)
    elseif type == :jumps_only
        plot_title = title === nothing ? L"$x$ vs jumps" : title
        plot_obj = plot(xlabel=L"$j$", ylabel=ylabel, title=plot_title; kwargs...)
    elseif type == :hybrid3d
        plot_title = title === nothing ? L"$x$ in hybrid time" : title
        plot_obj = plot3d(xlabel=L"$t$", ylabel=L"$j$", zlabel=ylabel, title=plot_title; kwargs...)
    end

    return HybridPlot(plot_obj, type, plot_title, ylabel)
end

"""
Main plotting interface for hybrid solutions.
"""
function plot(sol::HybridSolution, f=nothing;
    flow_color=:steelblue,
    jump_color=:firebrick,
    markersize=4,
    linewidth=2,
    type=:hybrid,
    title=nothing,
    ylabel=L"$x$",
    kwargs...)

    # Create HybridPlot and add solution to it
    hplot = HybridPlot(type; title=title, ylabel=ylabel, kwargs...)
    plot!(hplot, sol, f; flow_color=flow_color, jump_color=jump_color,
        markersize=markersize, linewidth=linewidth)

    # Set as current HybridPlot
    current_hybrid_plot!(hplot)

    # Return the underlying plot object(s) for backward compatibility
    if type == :hybrid
        plt_vs_t, plt_vs_j = hplot.plot_obj
        combined_title = hplot.title === nothing ? "" : hplot.title
        return plot(plt_vs_t, plt_vs_j, layout=(1, 2), size=(800, 400), plot_title=combined_title)
    else
        return hplot.plot_obj
    end
end

"""
Add a hybrid solution to an existing HybridPlot.
"""
function plot!(hplot::HybridPlot, sol::HybridSolution, f=nothing;
    flow_color=:steelblue,
    jump_color=:firebrick,
    markersize=4,
    linewidth=2,
    kwargs...)

    if f === nothing
        f = x -> x
        default_label = L"$x$"
    else
        default_label = L"$f(x)$"
    end

    # Dispatch based on hybrid plot type
    if hplot.plot_type == :hybrid3d
        # Add zlabel only if not already specified
        final_kwargs = haskey(kwargs, :zlabel) ? kwargs : (kwargs..., zlabel=default_label)
        return _plot_3d_function_to_hybrid!(hplot, sol, f;
            flow_color, jump_color, markersize, linewidth, final_kwargs...)
    else
        # Add ylabel only if not already specified  
        final_kwargs = haskey(kwargs, :ylabel) ? kwargs : (kwargs..., ylabel=default_label)
        return _plot_2d_function_to_hybrid!(hplot, sol, f;
            flow_color, jump_color, markersize, linewidth, final_kwargs...)
    end
end

"""
Add a hybrid solution to the current HybridPlot (similar to Plots.jl's plot!).
"""
function plot!(sol::HybridSolution, f=nothing; kwargs...)
    current_plot = current_hybrid_plot()
    if current_plot === nothing
        error("No current HybridPlot exists. Use plot(sol, f; kwargs...) first to create a new plot, then plot!(sol, f; kwargs...) to add to it.")
    end

    return plot!(current_plot, sol, f; kwargs...)
end

"""
Display a HybridPlot, handling the appropriate layout for different plot types.
"""
function Base.display(hplot::HybridPlot)
    if hplot.plot_type == :hybrid
        plt_vs_t, plt_vs_j = hplot.plot_obj
        combined_title = hplot.title === nothing ? "" : hplot.title
        combined_plot = plot(plt_vs_t, plt_vs_j, layout=(1, 2), size=(800, 400), plot_title=combined_title)
        display(combined_plot)
    else
        display(hplot.plot_obj)
    end
end

"""
Helper function to add 2D hybrid solution to HybridPlot
"""
function _plot_2d_function_to_hybrid!(hplot::HybridPlot, sol::HybridSolution, f;
    flow_color, jump_color, markersize, linewidth, kwargs...)

    # Check that function returns scalar values for 2D plotting
    if !isempty(sol.points)
        test_val = f(sol.points[1].state)
        if !isa(test_val, Real)
            output_dim_f = size(test_val)
            if length(output_dim_f) > 1 && output_dim_f[2] > 1
                error("Only plotting for vector-valued functions supported.")
            end
            ndims = output_dim_f[1]
        else
            ndims = 1
        end
    else
        error("Your solution is empty. Nothing to plot.")
    end

    times = [p.time.t for p in sol.points]
    jumps = [p.time.j for p in sol.points]
    jump_indices = sol.jump_indices

    # Handle color arrays for multi-dimensional functions
    if flow_color isa Vector && length(flow_color) != ndims
        error("In plot!(::HybridPlot, ::HybridSolution, f): flow_color vector length ($(length(flow_color))) must match function output dimension ($ndims).")
    end
    if jump_color isa Vector && length(jump_color) != ndims
        error("In plot!(::HybridPlot, ::HybridSolution, f): jump_color vector length ($(length(jump_color))) must match function output dimension ($ndims).")
    end

    flow_colors = flow_color isa Vector ? flow_color : fill(flow_color, ndims)
    jump_colors = jump_color isa Vector ? jump_color : fill(jump_color, ndims)

    # Get plot objects based on hybrid plot type
    if hplot.plot_type == :hybrid
        plt_vs_t, plt_vs_j = hplot.plot_obj
    elseif hplot.plot_type == :time_only
        plt_vs_t = hplot.plot_obj
        plt_vs_j = nothing
    elseif hplot.plot_type == :jumps_only
        plt_vs_t = nothing
        plt_vs_j = hplot.plot_obj
    end

    # Add data to appropriate plots
    for i in 1:ndims
        values = [f(p.state)[i] for p in sol.points]

        # Plot vs continuous time (if needed)
        if plt_vs_t !== nothing
            _plot_flow_segments!(plt_vs_t, times, values, jump_indices;
                flow_color=flow_colors[i], linewidth=linewidth)
            _plot_jump_transitions!(plt_vs_t, times, values, jump_indices;
                jump_color=jump_colors[i], markersize=markersize, linewidth=linewidth)
        end

        # Plot vs discrete jumps (if needed)  
        if plt_vs_j !== nothing
            _plot_flow_segments!(plt_vs_j, jumps, values, jump_indices;
                flow_color=flow_colors[i], linewidth=linewidth)
            _plot_jump_transitions!(plt_vs_j, jumps, values, jump_indices;
                jump_color=jump_colors[i], markersize=markersize, linewidth=linewidth)
        end
    end

    return hplot
end

"""
Helper function to add 3D hybrid solution to HybridPlot
"""
function _plot_3d_function_to_hybrid!(hp::HybridPlot, sol::HybridSolution, f;
    flow_color, jump_color, markersize, linewidth, kwargs...)

    # Check that function returns scalar values for 3D plotting
    if !isempty(sol.points)
        test_val = f(sol.points[1].state)
        if !isa(test_val, Real)
            error("Function must return scalar values for 3D hybrid plotting")
        end
    else
        error("Your solution is empty. Nothing to plot.")
    end

    times = [p.time.t for p in sol.points]
    jumps = [p.time.j for p in sol.points]
    values = [f(p.state) for p in sol.points]
    jump_indices = sol.jump_indices

    plt3d = hp.plot_obj

    # Plot flow segments in 3D
    start_idx = 1
    for jump_idx in jump_indices
        end_idx = jump_idx - 1
        if end_idx >= start_idx && end_idx >= 1
            plot!(plt3d, times[start_idx:end_idx], jumps[start_idx:end_idx], values[start_idx:end_idx],
                color=flow_color, linewidth=linewidth, label="")
        end
        start_idx = jump_idx + 1
    end

    # Plot final segment after last jump
    if start_idx <= length(times)
        plot!(plt3d, times[start_idx:end], jumps[start_idx:end], values[start_idx:end],
            color=flow_color, linewidth=linewidth, label="")
    end

    # Plot jump transitions in 3D
    for jump_idx in jump_indices
        pre_jump_idx = jump_idx - 1
        post_jump_idx = jump_idx

        if pre_jump_idx >= 1 && post_jump_idx <= length(times)
            plot!(plt3d, [times[pre_jump_idx], times[post_jump_idx]],
                [jumps[pre_jump_idx], jumps[post_jump_idx]],
                [values[pre_jump_idx], values[post_jump_idx]],
                color=jump_color, linestyle=:dash, linewidth=linewidth, label="")

            scatter!(plt3d, [times[pre_jump_idx], times[post_jump_idx]],
                [jumps[pre_jump_idx], jumps[post_jump_idx]],
                [values[pre_jump_idx], values[post_jump_idx]],
                color=jump_color, markersize=markersize, label="")
        end
    end

    return hp
end

"""
Helper: Plot flow segments in 2D
"""
function _plot_flow_segments!(plt, times, values, jump_indices;
    flow_color, linewidth)
    start_idx = 1
    for jump_idx in jump_indices
        end_idx = jump_idx - 1
        # Only plot if we have a valid range and it's not empty
        if end_idx >= start_idx && end_idx >= 1
            plot!(plt, times[start_idx:end_idx], values[start_idx:end_idx],
                color=flow_color, linewidth=linewidth, label="")
        end
        start_idx = jump_idx + 1  # Start after the jump point
    end

    # Plot final segment after last jump (if any points remain)
    if start_idx <= length(times)
        plot!(plt, times[start_idx:end], values[start_idx:end],
            color=flow_color, linewidth=linewidth, label="")
    end
end

"""
Helper: Plot jump transitions in 2D
"""
function _plot_jump_transitions!(plt, times, values, jump_indices;
    jump_color, markersize, linewidth)
    for jump_idx in jump_indices
        pre_jump_idx = jump_idx - 1  # Point just before the jump
        post_jump_idx = jump_idx     # Point just after the jump

        # Plot jump transition if both pre and post indices are valid
        if pre_jump_idx >= 1 && post_jump_idx <= length(times)
            plot!(plt, [times[pre_jump_idx], times[post_jump_idx]],
                [values[pre_jump_idx], values[post_jump_idx]],
                color=jump_color, linestyle=:dash, linewidth=linewidth, label="")

            scatter!(plt, [times[pre_jump_idx], times[post_jump_idx]],
                [values[pre_jump_idx], values[post_jump_idx]],
                color=jump_color, markersize=markersize, label="")
        end
    end
end

################################################################################
#### ADDITIONAL PLOTTING COMMANDS FOR HYBRID PLOTS
################################################################################

""" Add vertical lines to HybridPlot. Supports tuple specification for :hybrid type. """
function vline!(hplot::HybridPlot, x_vals; kwargs...)

    # Check for tuple input validation - handle both single tuple and vector of tuples
    is_single_tuple = x_vals isa Tuple && length(x_vals) == 2
    is_vector_of_tuples = x_vals isa Vector && length(x_vals) > 0 && x_vals[1] isa Tuple
    is_tuple_input = is_single_tuple || is_vector_of_tuples

    if is_tuple_input
        # Validate plot type
        if hplot.plot_type != :hybrid
            throw(ArgumentError("Tuple specification (val1, val2) or [(val1, val2), ...] is only valid for :hybrid plot type, got :$(hplot.plot_type)"))
        end
        # For vector input, validate all elements are tuples
        if is_vector_of_tuples && !all(x -> x isa Tuple, x_vals)
            throw(ArgumentError("Mixed input types: if first element is a tuple, all elements must be tuples"))
        end
    end

    if hplot.plot_type == :hybrid
        plt_vs_t, plt_vs_j = hplot.plot_obj

        if is_tuple_input
            # Per-subplot specification: (time_val, jump_val) or [(time_val, jump_val), ...]
            time_vals = []
            jump_vals = []

            # Handle both single tuple and vector of tuples
            tuples_to_process = is_single_tuple ? [x_vals] : x_vals

            for (t_val, j_val) in tuples_to_process
                if t_val !== nothing
                    push!(time_vals, t_val)
                end
                if j_val !== nothing
                    push!(jump_vals, j_val)
                end
            end

            # Add lines to respective subplots (with label="" to avoid legend entries)
            if !isempty(time_vals)
                Plots.vline!(plt_vs_t, time_vals; label="", kwargs...)
            end
            if !isempty(jump_vals)
                Plots.vline!(plt_vs_j, jump_vals; label="", kwargs...)
            end
        else
            # Uniform specification: same lines on both subplots
            x_array = x_vals isa Number ? [x_vals] : x_vals
            Plots.vline!(plt_vs_t, x_array; label="", kwargs...)
            Plots.vline!(plt_vs_j, x_array; label="", kwargs...)
        end

    elseif hplot.plot_type in [:time_only, :jumps_only]
        x_array = x_vals isa Number ? [x_vals] : x_vals
        Plots.vline!(hplot.plot_obj, x_array; kwargs...)

    elseif hplot.plot_type == :hybrid3d
        @warn "vline! is not supported for :hybrid3d plots due to GR backend limitations with transparency"
    end

    return hplot
end

function vline!(x_vals; kwargs...)
    current_plot = current_hybrid_plot()
    if current_plot === nothing
        error("No current HybridPlot exists. Use plot(sol; kwargs...) first to create a plot.")
    end
    return vline!(current_plot, x_vals; kwargs...)
end



function vspan!(hplot::HybridPlot, x_ranges; kwargs...)
    # Check for tuple input validation
    is_single_tuple = x_ranges isa Tuple && length(x_ranges) == 2
    is_vector_of_tuples = x_ranges isa Vector && length(x_ranges) > 0 && x_ranges[1] isa Tuple
    is_tuple_input = is_single_tuple || is_vector_of_tuples

    if is_tuple_input
        # Validate plot type
        if hplot.plot_type != :hybrid
            throw(ArgumentError("Tuple specification is only valid for :hybrid plot type, got :$(hplot.plot_type)"))
        end
        if is_vector_of_tuples && !all(x -> x isa Tuple, x_ranges)
            throw(ArgumentError("Mixed input types: if first element is a tuple, all elements must be tuples"))
        end
    end

    if hplot.plot_type == :hybrid
        plt_vs_t, plt_vs_j = hplot.plot_obj

        if is_tuple_input
            # Per-subplot specification: (time_range, jump_range) or [(time_range, jump_range), ...]
            tuples_to_process = is_single_tuple ? [x_ranges] : x_ranges

            for (t_range, j_range) in tuples_to_process
                if t_range !== nothing && length(t_range) == 2
                    Plots.vspan!(plt_vs_t, [t_range]; label="", kwargs...)
                end
                if j_range !== nothing && length(j_range) == 2
                    Plots.vspan!(plt_vs_j, [j_range]; label="", kwargs...)
                end
            end
        else
            # Uniform specification: same span on both subplots
            range_array = x_ranges isa Vector && length(x_ranges) == 2 ? [x_ranges] : x_ranges
            Plots.vspan!(plt_vs_t, range_array; label="", kwargs...)
            Plots.vspan!(plt_vs_j, range_array; label="", kwargs...)
        end

    elseif hplot.plot_type in [:time_only, :jumps_only]
        range_array = x_ranges isa Vector && length(x_ranges) == 2 ? [x_ranges] : x_ranges
        Plots.vspan!(hplot.plot_obj, range_array; kwargs...)

    elseif hplot.plot_type == :hybrid3d
        @warn "vspan! is not supported for :hybrid3d plots"
    end

    return hplot
end

function vspan!(x_ranges; kwargs...)
    current_plot = current_hybrid_plot()
    if current_plot === nothing
        error("No current HybridPlot exists. Use plot(sol; kwargs...) first to create a plot.")
    end
    return vspan!(current_plot, x_ranges; kwargs...)
end

""" Add horizontal lines to HybridPlot. Supports tuple specification for :hybrid type. """
function hline!(hplot::HybridPlot, y_vals; kwargs...)

    # Check for tuple input validation - handle both single tuple and vector of tuples
    is_single_tuple = y_vals isa Tuple && length(y_vals) == 2
    is_vector_of_tuples = y_vals isa Vector && length(y_vals) > 0 && y_vals[1] isa Tuple
    is_tuple_input = is_single_tuple || is_vector_of_tuples

    if is_tuple_input
        # Validate plot type
        if hplot.plot_type != :hybrid
            throw(ArgumentError("Tuple specification (val1, val2) or [(val1, val2), ...] is only valid for :hybrid plot type, got :$(hplot.plot_type)"))
        end
        # For vector input, validate all elements are tuples
        if is_vector_of_tuples && !all(y -> y isa Tuple, y_vals)
            throw(ArgumentError("Mixed input types: if first element is a tuple, all elements must be tuples"))
        end
    end

    if hplot.plot_type == :hybrid
        plt_vs_t, plt_vs_j = hplot.plot_obj

        if is_tuple_input
            # Per-subplot specification: (time_val, jump_val) or [(time_val, jump_val), ...]
            time_vals = []
            jump_vals = []

            # Handle both single tuple and vector of tuples
            tuples_to_process = is_single_tuple ? [y_vals] : y_vals

            for (t_val, j_val) in tuples_to_process
                if t_val !== nothing
                    push!(time_vals, t_val)
                end
                if j_val !== nothing
                    push!(jump_vals, j_val)
                end
            end

            # Add lines to respective subplots (with label="" to avoid legend entries)
            if !isempty(time_vals)
                Plots.hline!(plt_vs_t, time_vals; label="", kwargs...)
            end
            if !isempty(jump_vals)
                Plots.hline!(plt_vs_j, jump_vals; label="", kwargs...)
            end
        else
            # Uniform specification: same lines on both subplots
            y_array = y_vals isa Number ? [y_vals] : y_vals
            Plots.hline!(plt_vs_t, y_array; label="", kwargs...)
            Plots.hline!(plt_vs_j, y_array; label="", kwargs...)
        end

    elseif hplot.plot_type in [:time_only, :jumps_only]
        y_array = y_vals isa Number ? [y_vals] : y_vals
        Plots.hline!(hplot.plot_obj, y_array; kwargs...)

    elseif hplot.plot_type == :hybrid3d
        @warn "hline! is not supported for :hybrid3d plots due to GR backend limitations with transparency"
    end

    return hplot
end

function hline!(y_vals; kwargs...)
    current_plot = current_hybrid_plot()
    if current_plot === nothing
        error("No current HybridPlot exists. Use plot(sol; kwargs...) first to create a plot.")
    end
    return hline!(current_plot, y_vals; kwargs...)
end

end # module HybridSolver
