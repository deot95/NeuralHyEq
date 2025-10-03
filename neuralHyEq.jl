"""
Neural network training for hybrid systems.
"""

println("\n" * "="^60)
println("PREPARING FOR TRAINING PROCESS")
println("="^60)
println("Including HybridSolver.jl module...")

include("HybridSolver.jl")

println("Loading packages and defining hybrid systems...")


using .HybridSolver
using SimpleChains, StaticArrays, ForwardDiff,
    OptimizationOptimisers, Plots, Optimization, LinearAlgebra, LaTeXStrings,
    Statistics, HDF5, Dates
const hy = HybridSolver

################################################################################
#### NEURAL HYBRID ARCHITECTURE
################################################################################

struct NeuralHybridSystem{T} <: HybridSystem
    nn_flow::SimpleChain
    nn_jump::SimpleChain
    flow_param_length::Int
end

NeuralHybridSystem(nn_flow::SimpleChain, nn_jump::SimpleChain, flow_param_length::Int) =
    NeuralHybridSystem{Float64}(nn_flow, nn_jump, flow_param_length)

hy.flow_set(sys::NeuralHybridSystem, x, t, j) = true
hy.jump_set(sys::NeuralHybridSystem, x, t, j) = true

function extract_parameters(sys::NeuralHybridSystem, θ)
    θ_flow = θ[1:sys.flow_param_length]
    θ_jump = θ[sys.flow_param_length+1:end]
    return θ_flow, θ_jump
end

function hy.flow_map!(sys::NeuralHybridSystem, dx, x, θ)
    θ_flow, _ = extract_parameters(sys, θ)
    nn_output = sys.nn_flow(x, θ_flow)
    dx[1:end] = nn_output
end

function hy.jump_map(sys::NeuralHybridSystem, x, θ)
    _, θ_jump = extract_parameters(sys, θ)
    nn_output = sys.nn_jump(x, θ_jump)
    return collect(nn_output)
end
function create_neural_hybrid_system(state_dim::Int;
    flow_nodes=24, jump_nodes=24)
    # Create neural networks
    nn_flow = SimpleChain(static(state_dim),
        TurboDense{true}(relu, static(flow_nodes)),
        TurboDense{true}(identity, static(state_dim)))

    nn_jump = SimpleChain(static(state_dim),
        TurboDense{true}(relu, static(jump_nodes)),
        TurboDense{true}(identity, static(state_dim)))

    # Initialize parameters - avoid Float64.() to allow ForwardDiff dual numbers
    θ_flow_initial = Array(SimpleChains.init_params(nn_flow))
    θ_jump_initial = Array(SimpleChains.init_params(nn_jump))
    θ_initial = vcat(θ_flow_initial, θ_jump_initial)

    neural_sys = NeuralHybridSystem(nn_flow, nn_jump, length(θ_flow_initial))

    return neural_sys, θ_initial
end

################################################################################
#### STANDALONE NEURAL SYSTEM
################################################################################

""" Neural system using true flow/jump sets for validation """
struct StandaloneNeuralSystem{T} <: HybridSystem
    true_system::HybridSystem
    nn_flow::SimpleChain
    nn_jump::SimpleChain
    flow_param_length::Int
end

StandaloneNeuralSystem(true_sys::HybridSystem, nn_flow::SimpleChain, nn_jump::SimpleChain, flow_param_length::Int) =
    StandaloneNeuralSystem{Float64}(true_sys, nn_flow, nn_jump, flow_param_length)

hy.flow_set(sys::StandaloneNeuralSystem, x, t, j) = hy.flow_set(sys.true_system, x, t, j)
hy.jump_set(sys::StandaloneNeuralSystem, x, t, j) = hy.jump_set(sys.true_system, x, t, j)

function extract_parameters(sys::StandaloneNeuralSystem, θ)
    θ_flow = θ[1:sys.flow_param_length]
    θ_jump = θ[sys.flow_param_length+1:end]
    return θ_flow, θ_jump
end

function hy.flow_map!(sys::StandaloneNeuralSystem, dx, x, θ)
    θ_flow, _ = extract_parameters(sys, θ)
    nn_output = sys.nn_flow(x, θ_flow)
    dx[1:end] = nn_output
end

function hy.jump_map(sys::StandaloneNeuralSystem, x, θ)
    _, θ_jump = extract_parameters(sys, θ)
    nn_output = sys.nn_jump(x, θ_jump)
    return collect(nn_output)
end

function create_standalone_neural_system(true_system::HybridSystem, neural_sys::NeuralHybridSystem)
    return StandaloneNeuralSystem(
        true_system,
        neural_sys.nn_flow,
        neural_sys.nn_jump,
        neural_sys.flow_param_length
    )
end
function validate_neural_system(true_system::HybridSystem, neural_sys::NeuralHybridSystem,
    trained_params::Vector{Float64}, x0::Vector{Float64}, max_time::Float64)
    # Create standalone neural system with true flow/jump sets
    standalone_neural = create_standalone_neural_system(true_system, neural_sys)

    # Solve both systems
    true_solution = solve(true_system, x0, max_time; config=HybridSolverConfig(verbose=false))
    neural_solution = solve(standalone_neural, x0, max_time; config=HybridSolverConfig(verbose=false), p=trained_params)

    println("Validation Results:")
    println("  True system: $(length(true_solution.points)) points, $(true_solution.total_jumps) jumps")
    println("  Neural system: $(length(neural_solution.points)) points, $(neural_solution.total_jumps) jumps")

    return true_solution, neural_solution
end

################################################################################
####  AUGMENTED SYSTEM
################################################################################

""" Augmented system evolving true and neural systems in tandem """
struct AugmentedHybridSystem{T} <: HybridSystem
    true_system::HybridSystem           # True system for structure
    neural_system::NeuralHybridSystem   # Neural system for dynamics
    state_dim::Int                      # Dimension of original state (n)
end

AugmentedHybridSystem(true_sys::HybridSystem, neural_sys::NeuralHybridSystem, state_dim::Int) =
    AugmentedHybridSystem{Float64}(true_sys, neural_sys, state_dim)

# Flow and jump sets based only on true system part
hy.flow_set(aug_sys::AugmentedHybridSystem, x, t, j) = hy.flow_set(aug_sys.true_system, x[1:aug_sys.state_dim], t, j)
hy.jump_set(aug_sys::AugmentedHybridSystem, x, t, j) = hy.jump_set(aug_sys.true_system, x[1:aug_sys.state_dim], t, j)

"""
Helper functions for augmented state manipulation
"""
get_true_part(x, state_dim::Int) = x[1:state_dim]
get_neural_part(x, state_dim::Int) = x[state_dim+1:2*state_dim]
combine_states(x_true, x_neural) = vcat(x_true, x_neural)

"""
Flow map for augmented system: evolve both systems in parallel
"""
function hy.flow_map!(aug_sys::AugmentedHybridSystem, dx, x, θ)
    n = aug_sys.state_dim

    # Extract parts
    x_true = get_true_part(x, n)
    x_neural = get_neural_part(x, n)

    # Extract derivatives
    dx_true = view(dx, 1:n)
    dx_neural = view(dx, n+1:2n)

    # Evolve both systems
    hy.flow_map!(aug_sys.true_system, dx_true, x_true, nothing)
    hy.flow_map!(aug_sys.neural_system, dx_neural, x_neural, θ)
end

"""
Jump map for augmented system: apply jump to both systems
"""
function hy.jump_map(aug_sys::AugmentedHybridSystem, x, θ)
    n = aug_sys.state_dim

    # Extract parts
    x_true = get_true_part(x, n)
    x_neural = get_neural_part(x, n)

    # Apply jumps
    x_true_jumped = hy.jump_map(aug_sys.true_system, x_true, nothing)
    x_neural_jumped = hy.jump_map(aug_sys.neural_system, x_neural, θ)

    return combine_states(x_true_jumped, x_neural_jumped)
end

"""
Extract trajectories from augmented solution
"""
function extract_true_trajectory(aug_solution::HybridSolution, state_dim::Int)
    true_points = HybridState{eltype(aug_solution.points[1].state[1:state_dim])}[]

    for point in aug_solution.points
        true_state = get_true_part(point.state, state_dim)
        push!(true_points, HybridState(point.time, true_state))
    end

    return HybridSolution(
        true_points,
        aug_solution.jump_indices,
        get_true_part(aug_solution.initial_condition, state_dim),
        aug_solution.final_time,
        aug_solution.total_jumps,
        aug_solution.termination_condition
    )
end

function extract_neural_trajectory(aug_solution::HybridSolution, state_dim::Int)
    neural_points = HybridState{eltype(aug_solution.points[1].state[1:state_dim])}[]

    for point in aug_solution.points
        neural_state = get_neural_part(point.state, state_dim)
        push!(neural_points, HybridState(point.time, neural_state))
    end

    return HybridSolution(
        neural_points,
        aug_solution.jump_indices,
        get_neural_part(aug_solution.initial_condition, state_dim),
        aug_solution.final_time,
        aug_solution.total_jumps,
        aug_solution.termination_condition
    )
end

################################################################################
#### SCALAR FUNCTIONS FOR LYAPUNOV CONSTRAINTS
################################################################################

abstract type ScalarFunction end

""" Bouncing ball energy: V(x) = (1/2)*m*vy² + m*g*y """
struct BouncingBallEnergy <: ScalarFunction
    m::Float64
    g::Float64
end

BouncingBallEnergy(; m=1.0, g=9.81) = BouncingBallEnergy(m, g)

function evaluate_V(V_func::BouncingBallEnergy, x)
    kinetic = 0.5 * V_func.m * x[2]^2
    potential = V_func.m * V_func.g * max(0.0, x[1])
    return kinetic + potential
end

function gradient_V(V_func::BouncingBallEnergy, x)
    return [V_func.m * V_func.g, V_func.m * x[2]]
end

struct GenericScalarFunction <: ScalarFunction
    func::Function
    grad_func::Union{Function,Nothing}
end

function evaluate_V(V_func::GenericScalarFunction, x)
    return V_func.func(x)
end

function gradient_V(V_func::GenericScalarFunction, x)
    if V_func.grad_func !== nothing
        return V_func.grad_func(x)
    else
        return ForwardDiff.gradient(V_func.func, x)
    end
end

function compute_Vdot(V_func::ScalarFunction, x, dx)
    ∇V = gradient_V(V_func, x)
    return dot(∇V, dx)
end

function compute_delta_V(V_func::ScalarFunction, x_minus, x_plus)
    return evaluate_V(V_func, x_plus) - evaluate_V(V_func, x_minus)
end

################################################################################
#### TRAINING CONFIGURATION
################################################################################

Base.@kwdef struct TrainingConfig
    # Optimization parameters
    max_iters::Int = 200
    optimizer_lr::Float64 = 0.01

    # Loss weights
    flow_weight::Float64 = 1.0
    jump_weight::Float64 = 5.0

    # Energy-based costs
    vdot_weight::Float64 = 0.1               # Weight for V̇ penalty during flows
    delta_v_weight::Float64 = 0.1            # Weight for ΔV penalty during jumps

    # Parameter regularization
    regularization_weight::Float64 = 1e-4    # Regularization weight
    regularization_norm::Int = 2             # Regularization norm (1 or 2)

    # Randomized initial conditions
    use_random_init::Bool = false
    random_batch_size::Int = 4
    random_update_frequency::Int = 25
    random_start_iteration::Int = 100

    # Monitoring and output
    show_plots::Bool = true
    plot_frequency::Int = 10
    progress_frequency::Int = 1
    short_ma_window::Int = 20
    long_ma_window::Int = 250

    # Saving
    save_params::Bool = true
    filename::String = "hybrid_training_results"
end



"""
Setup initial conditions batch for training with optional randomization
"""
function setup_initial_conditions_batch(x0, config::TrainingConfig, random_ic_generator::Function=nothing)
    x0_batch = [x0]  # Always include original
    last_update_iteration = 0

    if config.use_random_init
        if random_ic_generator === nothing
            error("Random IC generator function required when use_random_init=true")
        end
        for _ in 2:config.random_batch_size
            push!(x0_batch, random_ic_generator())
        end
        println("Generated batch of $(length(x0_batch)) initial conditions for training")
    end

    return x0_batch, last_update_iteration
end

"""
Create enhanced hybrid loss function with continuous, discrete, energy, and regularization components
"""
function create_hybrid_loss_function(aug_sys::AugmentedHybridSystem, x0_batch, max_hybrid_time, iteration_ref,
    last_update_iteration_ref, solver_config::HybridSolverConfig=HybridSolverConfig(verbose=false), random_ic_generator::Function=nothing,
    V_func::Union{ScalarFunction,Nothing}=nothing)
    function loss_func(θ, p)
        # Extract α and β from p parameter
        α = p.flow_weight   # Continuous weight
        β = p.jump_weight   # Discrete weight

        # Determine if we should use randomized initial conditions based on schedule
        should_use_random = p.use_random_init && (iteration_ref[1] >= p.random_start_iteration)

        if should_use_random
            if random_ic_generator === nothing
                error("Random IC generator function required when use_random_init=true")
            end
            # Update batch periodically based on optimization iterations
            if iteration_ref[1] >= last_update_iteration_ref[1] + p.random_update_frequency
                last_update_iteration_ref[1] = iteration_ref[1]
                # Regenerate random conditions (keep original at index 1)
                for i in 2:lastindex(x0_batch)
                    x0_batch[i] = random_ic_generator()
                end
            end

            # Cycle through batch based on iteration number
            batch_idx = ((iteration_ref[1] - p.random_start_iteration) % p.random_batch_size) + 1
            current_x0 = x0_batch[batch_idx]
        else
            current_x0 = x0_batch[1]  # Use original x0
        end

        x0_aug = combine_states(current_x0, current_x0)
        aug_trajectory = solve(aug_sys, x0_aug, max_hybrid_time; config=solver_config, p=θ)
        true_traj = extract_true_trajectory(aug_trajectory, aug_sys.state_dim)
        neural_traj = extract_neural_trajectory(aug_trajectory, aug_sys.state_dim)
        neural_flow_segments = extract_flow_segments(neural_traj)
        true_flow_segments = extract_flow_segments(true_traj)

        # Extract jump transitions for discrete loss
        neural_jump_transitions = extract_jump_transitions(neural_traj)
        true_jump_transitions = extract_jump_transitions(true_traj)

        # Continuous loss
        continuous_loss = 0.0
        for (true_seg, neural_seg) in zip(true_flow_segments, neural_flow_segments)
            # Vectorized computation per segment (more ForwardDiff-friendly)
            true_seg_matrix = reduce(hcat, true_seg.states)     # Convert to matrix
            neural_seg_matrix = reduce(hcat, neural_seg.states) # Convert to matrix
            continuous_loss += sum(abs2, neural_seg_matrix - true_seg_matrix)
        end

        # Discrete loss
        discrete_loss = 0.0
        for (neural_jump, true_jump) in zip(neural_jump_transitions, true_jump_transitions)
            # Direct neural jump map evaluation (more meaningful than transition comparison)
            discrete_loss += sum(abs2, (neural_jump.x_plus - neural_jump.x_minus) - (true_jump.x_plus - true_jump.x_minus))
        end

        # Energy-based costs (V̇ and ΔV penalties) 
        energy_flow_loss = 0.0
        energy_jump_loss = 0.0

        if V_func !== nothing
            # Compute V̇ costs during flow segments for neural trajectory
            for neural_seg in neural_flow_segments
                for i in 1:(length(neural_seg.states)-1)
                    x_current = neural_seg.states[i]
                    # Compute flow derivative at this point
                    dx = similar(x_current)
                    hy.flow_map!(aug_sys.neural_system, dx, x_current, θ)
                    # Compute V̇ = ∇V⋅f
                    vdot = compute_Vdot(V_func, x_current, dx)
                    energy_flow_loss += abs(vdot)  # Penalize V̇
                end
            end

            # Compute ΔV costs during jump transitions for neural trajectory  
            for neural_jump in neural_jump_transitions
                delta_v = compute_delta_V(V_func, neural_jump.x_minus, neural_jump.x_plus)
                energy_jump_loss += abs(delta_v)  # Penalize ΔV
            end
        end

        # Parameter regularization
        reg_loss = 0.0
        if p.regularization_weight > 0
            if p.regularization_norm == 1
                reg_loss = sum(abs, θ)  # L1 norm
            else
                reg_loss = sum(abs2, θ)  # L2 norm
            end
            reg_loss *= p.regularization_weight
        end

        # Combined hybrid loss with all components
        total_loss = α * continuous_loss + β * discrete_loss +
                     p.vdot_weight * energy_flow_loss + p.delta_v_weight * energy_jump_loss +
                     reg_loss
        return total_loss
    end

    return loss_func
end

"""
Create training callback with progress tracking, plotting, and parameter snapshots
"""
function create_training_callback(θ_initial, aug_sys, x0, max_hybrid_time, config::TrainingConfig, iteration_ref)
    cost_history = Float64[]
    parameter_history = Vector{Float64}[]  # Full parameter history for gif generation
    best_cost = Inf
    best_params = copy(θ_initial)

    function callback(state, cost)
        push!(cost_history, cost)

        # Update iteration counter for loss function scheduling
        iteration_ref[1] = state.iter

        # Track best model
        if cost < best_cost
            best_cost = cost
            best_params = copy(state.u)
        end

        # Save parameter history (every iteration)
        push!(parameter_history, copy(state.u))

        # Progress reporting with moving averages
        progress_msg = "Iter $(state.iter): Cost = $(round(cost, sigdigits=4)), Best = $(round(best_cost, sigdigits=4))"

        # Add short-time moving average
        if length(cost_history) >= config.short_ma_window
            short_costs = cost_history[end-(config.short_ma_window-1):end]
            short_avg = sum(short_costs) / config.short_ma_window
            progress_msg *= ", MA$(config.short_ma_window) = $(round(short_avg, sigdigits=4))"
        end

        # Add long-time moving average
        if length(cost_history) >= config.long_ma_window
            long_costs = cost_history[end-(config.long_ma_window-1):end]
            long_avg = sum(long_costs) / config.long_ma_window
            progress_msg *= ", MA$(config.long_ma_window) = $(round(long_avg, sigdigits=4))"
        end

        # Check if switching to random mode
        if config.use_random_init && state.iter == config.random_start_iteration
            println("Switching to randomized initial conditions at iteration $(state.iter)")
        end

        println(progress_msg)

        # Real-time plotting 
        if config.show_plots && state.iter % config.plot_frequency == 0 && state.iter > 0
            try
                # Generate current prediction
                current_aug_solution = solve(aug_sys, combine_states(x0, x0), max_hybrid_time; config=HybridSolverConfig(verbose=false), p=state.u)

                # Create comparison plots
                plt_height = plot(current_aug_solution, x -> [x[1]; x[3]], ylabel="Height (m)",
                    flow_color=[:steelblue, :orange], jump_color=[:navy, :coral], label=["True" "Neural"])
                plt_velocity = plot(current_aug_solution, x -> [x[2]; x[4]], ylabel="Velocity (m/s)",
                    flow_color=[:steelblue, :orange], jump_color=[:navy, :coral], label=["True" "Neural"])

                # Cost evolution plot with both instantaneous and moving average
                plt_cost = plot(title="Cost Evolution", xlabel="Iteration", ylabel="Cost (log scale)", yscale=:log10)

                # Plot instantaneous cost with transparency
                plot!(plt_cost, 1:length(cost_history), cost_history,
                    linewidth=1, alpha=0.3, label="Instantaneous Cost", color=:crimson)

                # Add moving averages for cleaner visualization  
                if length(cost_history) >= config.short_ma_window
                    short_ma = [sum(cost_history[max(1, i - (config.short_ma_window - 1)):i]) / min(i, config.short_ma_window)
                                for i in eachindex(cost_history)]
                    plot!(plt_cost, 1:length(short_ma), short_ma,
                        linewidth=3, color=:steelblue, label="MA$(config.short_ma_window)")
                end

                if length(cost_history) >= config.long_ma_window
                    long_ma = [sum(cost_history[max(1, i - (config.long_ma_window - 1)):i]) / min(i, config.long_ma_window)
                               for i in eachindex(cost_history)]
                    plot!(plt_cost, 1:length(long_ma), long_ma,
                        linewidth=3, color=:darkgreen, label="MA$(config.long_ma_window)")
                end

                # Combined plot with more space for cost function
                plt_combined = plot(plt_height, plt_velocity, plt_cost,
                    layout=@layout([a; b; c{0.4h}]), size=(900, 1000))
                display(plt_combined)
            catch e
                println("Plotting failed at iteration $(state.iter): $e")
            end
        end

        return false
    end

    return callback, () -> (best_cost, best_params, parameter_history, cost_history)
end

"""
Save training results to HDF5 file for later analysis and plotting
"""
function save_training_results_hdf5(results::Dict, filename::String)
    # Create data directory relative to this script file if it doesn't exist
    data_dir = joinpath(dirname(@__FILE__), "neural_hybrid_data")
    if !isdir(data_dir)
        mkdir(data_dir)
        println("Created data directory: $data_dir")
    end

    # Create timestamp for unique filenames
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    filepath = joinpath(data_dir, "$(filename)_$(timestamp).h5")

    h5open(filepath, "w") do file
        # Save metadata
        file["approach"] = results["approach"]
        file["final_cost"] = results["final_cost"]
        file["best_cost"] = results["best_cost"]
        file["time_to_train"] = results["time_to_train"]
        file["timestamp"] = timestamp
        file["x0_batch"] = reduce(hcat, results["x0_batch"])

        # Save cost history
        file["cost_history"] = results["cost_history"]

        # Save parameters
        file["initial_params"] = results["initial_params"]
        file["final_params"] = results["final_params"]
        file["best_params"] = results["best_params"]

        # Save parameter history
        if haskey(results, "parameter_history")
            history_group = create_group(file, "parameter_history")
            for (i, params) in enumerate(results["parameter_history"])
                history_group[string(i)] = params
            end
        end

        # Save training configuration
        if haskey(results, "config")
            config = results["config"]
            config_group = create_group(file, "config")
            config_group["max_iters"] = config.max_iters
            config_group["optimizer_lr"] = config.optimizer_lr
            config_group["flow_weight"] = config.flow_weight
            config_group["jump_weight"] = config.jump_weight
            config_group["use_random_init"] = config.use_random_init
        end
    end

    println("Training results saved to: $filepath")
    return filepath
end

"""
Load training results from HDF5 file
"""
function load_training_results_hdf5(filepath::String)
    if !isfile(filepath)
        error("File not found: $filepath")
    end

    results = Dict{String,Any}()

    h5open(filepath, "r") do file
        # Load metadata
        results["approach"] = read(file, "approach")
        results["final_cost"] = read(file, "final_cost")
        results["best_cost"] = read(file, "best_cost")
        results["time_to_train"] = read(file, "time_to_train")
        results["timestamp"] = read(file, "timestamp")
        results["x0_batch"] = read(file, "x0_batch")

        # Load cost history
        results["cost_history"] = read(file, "cost_history")

        # Load parameters
        results["initial_params"] = read(file, "initial_params")
        results["final_params"] = read(file, "final_params")
        results["best_params"] = read(file, "best_params")

        # Load parameter history
        if haskey(file, "parameter_history")
            history = Vector{Float64}[]
            history_group = file["parameter_history"]
            # Read in order
            indices = sort([parse(Int, key) for key in keys(history_group)])
            for i in indices
                push!(history, read(history_group, string(i)))
            end
            results["parameter_history"] = history
        end

        # Load training configuration
        if haskey(file, "config")
            config_data = Dict{String,Any}()
            config_group = file["config"]
            for key in keys(config_group)
                config_data[key] = read(config_group, key)
            end
            results["config"] = config_data
        end
    end

    println("Loaded training results from: $filepath")
    println("  - Approach: $(results["approach"])")
    println("  - Best cost: $(round(results["best_cost"], sigdigits=4))")
    println("  - Training time: $(round(results["time_to_train"], digits=2)) seconds ($(round(results["time_to_train"]/60, digits=2)) minutes)")
    println("  - Cost history: $(length(results["cost_history"])) points")
    if haskey(results, "parameter_history")
        println("  - Parameter history: $(length(results["parameter_history"])) points")
    end

    return results
end

"""
    train_neural_hybrid_equation(true_system, x0, max_hybrid_time; config=TrainingConfig(), random_ic_generator=nothing)

Interface to train a neural hybrid equation.
Returns the trained neural system and parameters.

Arguments:
- true_system: The true hybrid system to approximate
- x0: Initial condition
- max_hybrid_time: Maximum simulation time
- config: Training configuration
- random_ic_generator: Function that generates random initial conditions (required if config.use_random_init=true)
"""
function train_neural_hybrid_equation(true_system::HybridSystem, x0, max_hybrid_time::Real;
    config::TrainingConfig=TrainingConfig(), solver_config::HybridSolverConfig=HybridSolverConfig(verbose=false),
    random_ic_generator::Function=nothing, V_func::Union{ScalarFunction,Nothing}=nothing)

    state_dim = length(x0)

    # Create neural system
    neural_sys, θ_initial = create_neural_hybrid_system(state_dim)

    # Create augmented system
    aug_sys = AugmentedHybridSystem(true_system, neural_sys, state_dim)

    # Shared iteration counter for scheduling
    iteration_ref = [0]

    # Setup initial conditions batch
    x0_batch, last_update_iteration = setup_initial_conditions_batch(x0, config, random_ic_generator)
    last_update_iteration_ref = [last_update_iteration]

    # Create modular components
    loss_func = create_hybrid_loss_function(aug_sys, x0_batch, max_hybrid_time, iteration_ref, last_update_iteration_ref,
        solver_config, random_ic_generator, V_func)
    callback, get_best_results = create_training_callback(θ_initial, aug_sys, x0, max_hybrid_time, config, iteration_ref)

    # Setup optimization problem with config passed as p parameter
    prob = Optimization.OptimizationProblem(
        Optimization.OptimizationFunction(loss_func, Optimization.AutoForwardDiff()),
        θ_initial,
        config  # Pass config as p parameter containing α and β
    )

    # Run optimization with timing
    println("Starting optimization...")
    time_to_train = @elapsed result = Optimization.solve(prob, Adam(config.optimizer_lr), callback=callback, maxiters=config.max_iters)

    println("\n" * "="^60)
    println("TRAINING COMPLETED")
    println("="^60)
    println("Training time: $(round(time_to_train, digits=2)) seconds ($(round(time_to_train/60, digits=2)) minutes)")
    println("="^60)

    # Get best results and all training data
    best_cost, best_params, parameter_history, cost_history = get_best_results()

    # Create comprehensive results dictionary
    results = Dict(
        "approach" => "modular_hybrid_training",
        "cost_history" => cost_history,
        "initial_params" => θ_initial,
        "x0_batch" => x0_batch,
        "final_params" => result.u,
        "best_params" => best_params,
        "parameter_history" => parameter_history,
        "final_cost" => cost_history[end],
        "best_cost" => best_cost,
        "converged" => length(cost_history) < config.max_iters,
        "time_to_train" => time_to_train,
        "max_flow_steps" => max_hybrid_time,
        "config" => config
    )

    # Save results if requested
    if config.save_params
        filepath = save_training_results_hdf5(results, config.filename)
        println("Training results saved")
        println("  - Cost history: $(length(cost_history)) points")
        println("  - Parameter history: $(length(parameter_history)) points")
    end

    # Return final augmented solution for plotting using best parameters
    final_aug_solution = solve(aug_sys, combine_states(x0, x0), max_hybrid_time; config=HybridSolverConfig(verbose=false), p=best_params)
    return best_params, aug_sys, final_aug_solution, results
end

"""
Simple example: Train neural approximation of bouncing ball
"""

################################################################################
#### BOUNCING BALL SYSTEM
################################################################################

"""
BouncingBall <: HybridSystem

A hybrid dynamical system representing a ball bouncing under gravity with energy loss.

Hybrid dynamics:
- Flow set: C = {x ∈ ℝ² : x1 ≥ 0} (ball above ground)
- Jump set: D = {x ∈ ℝ² : x1 ≤ 0, x2 ≤ 0} (ball hits ground with downward velocity)
- Flow map: f(x) = [x2, -g] (free fall dynamics)
- Jump map: G(x) = [x1, -e*x2] (bilipschitz elastic collision)
"""
struct BouncingBall <: HybridSystem
    g::Float64           # Gravity acceleration
    e::Float64           # Restitution coefficient
end

"""
Generate random initial conditions for
"""
function generate_random_ic_bouncing_ball()
    x10 = 0.5 + 9.5 * rand()      # Height: 0.5-8m
    x20 = -3.0 + 6.0 * rand()    # Vertical velocity: -3 to 3 m/s
    return [x10, x20]
end

"""
    hy.flow_set(sys::BouncingBall, x, t, j) -> Bool

Flow set C: defines when continuous dynamics can happen.
"""
function hy.flow_set(sys::BouncingBall, x, t, j)
    return x[1] >= 0 || x[2] >= 0  # y ≥ 0 or vy ≥ 0
end

"""
    hy.jump_set(sys::BouncingBall, x, t, j) -> Bool

Jump set D: defines when jumps can occur.
"""
function hy.jump_set(sys::BouncingBall, x, t, j)
    return x[1] <= 0 && x[2] <= 0  # y ≤ 0 and vy ≤ 0
end

"""
    hy.flow_map!(sys::BouncingBall, dx, x, p)

Flow map f(x): continuous dynamics during flow.
"""
function hy.flow_map!(sys::BouncingBall, dx, x, p)
    dx[1] = x[2]          # dy/dt = vy
    dx[2] = -sys.g        # dvy/dt = -g
end

"""
    hy.jump_map(sys::BouncingBall, x, p) -> Vector

Jump map G(x): discrete state update during jumps.
"""
function hy.jump_map(sys::BouncingBall, x, p)
    return [x[1], -sys.e * x[2]]    # [y⁺, vy⁺] = [y⁻, -e*vy⁻]
end

# Initial Comparison
true_ball = BouncingBall(9.81, 0.8)
V_ball_energy = BouncingBallEnergy(m=1.0, g=true_ball.g)
x0 = [3.5, 0.0]
neural_sys, θ_initial = create_neural_hybrid_system(length(x0))
aug_sys = AugmentedHybridSystem(true_ball, neural_sys, length(x0))

solver_config = HybridSolverConfig(verbose=false, jump_priority=false)
x0_aug = combine_states(x0, x0)

max_hybrid_time_training = 10.0
aug_sol = solve(aug_sys, x0_aug, max_hybrid_time_training; config=solver_config, p=θ_initial)
plt_height = plot(aug_sol, x -> [x[1]; x[3]], flow_color=[:steelblue, :orange],
    jump_color=[:navy, :coral], label=["True" "Neural"], ylabel="Height")
plt_velocity = plot(aug_sol, x -> [x[2]; x[4]], flow_color=[:steelblue, :orange],
    jump_color=[:navy, :coral], label=["True" "Neural"], ylabel="Velocity")
plt_comparison = plot(plt_height, plt_velocity, layout=(2, 1), size=(900, 600))



################################################################################
#### TRAINING CALL
################################################################################
println("\n" * "="^60)
println("TRAINING STARTING")
println("="^60)

# Create energy function for bouncing ball

# Train neural approximation with enhanced loss function
max_iters_training = 10000
println("Training neural hybrid system with energy constraints\n for $max_iters_training steps...")

config = TrainingConfig(max_iters=max_iters_training, optimizer_lr=5e-4,
    plot_frequency=20, use_random_init=true,
    random_update_frequency=25, random_start_iteration=1,
    random_batch_size=8, flow_weight=1.0, jump_weight=2.0,
    vdot_weight=0, delta_v_weight=0, regularization_weight=0.1)
max_hybrid_time = 10. # t + j ≤ T_max + J_max
trained_params, aug_sys, final_aug_solution, results = train_neural_hybrid_equation(true_ball, x0, max_hybrid_time; config=config, solver_config, random_ic_generator=generate_random_ic_bouncing_ball, V_func=V_ball_energy)

# Plot the final augmented solution
println("\nPlotting trained system...")
plt_height = plot(final_aug_solution, x -> [x[1]; x[3]], type=:time_only, title="Height - Trained", ylabel="Height (m)",
    flow_color=[:steelblue, :orange], jump_color=[:navy, :coral], label=["True" "Neural"])
plt_velocity = plot(final_aug_solution, x -> [x[2]; x[4]], type=:time_only, title="Velocity - Trained", ylabel="Velocity (m/s)",
    flow_color=[:steelblue, :orange], jump_color=[:navy, :coral], label=["True" "Neural"])

plt_comparison = plot(plt_height, plt_velocity, layout=(2, 1), size=(900, 600))
display(plt_comparison)

################################################################################
#### VALIDATION AND ANIMATION SECTION
################################################################################

println("\n" * "="^60)
println("LOADING DATA AND VALIDATION")
println("="^60)

# Find and load the latest training data file
data_dir = joinpath(dirname(@__FILE__), "neural_hybrid_data")
if isdir(data_dir)
    # Get all .h5 files that match the training results pattern
    h5_files = filter(f -> endswith(f, ".h5") && startswith(f, "hybrid_training_results_"), readdir(data_dir))

    if !isempty(h5_files)
        # Sort by modification time to get the latest file
        h5_paths = [joinpath(data_dir, f) for f in h5_files]
        latest_file = h5_paths[argmax(mtime.(h5_paths))]
        data_file = latest_file

        # Get the modification time for display (convert from UTC to local time)
        save_time_utc = unix2datetime(mtime(data_file))
        # Calculate timezone offset by comparing local now with UTC now
        local_now = Dates.now()
        utc_now = Dates.unix2datetime(time())  # time() gives UTC timestamp
        local_offset = local_now - utc_now
        save_time_local = save_time_utc + local_offset

        println("Loading latest data saved on $(Dates.format(save_time_local, "U d, yyyy \\at HH:MM:SS")): $(basename(data_file))")
    else
        error("No training results files found in $data_dir")
    end
else
    error("Data directory not found: $data_dir")
end

loaded_results = load_training_results_hdf5(data_file)
best_trained_params = loaded_results["best_params"]
cost_history = loaded_results["cost_history"]
parameter_history = loaded_results["parameter_history"]
x0_batch = loaded_results["x0_batch"]

# Create standalone neural system with true flow/jump sets
standalone_neural = create_standalone_neural_system(true_ball, neural_sys)

# Use one of the random initial conditions but with longer time domain
test_x0 = [10., 3.]
max_hybrid_time_training = 10.0
validation_time = 20.0  # Longer than training time for extrapolation test

println("Test initial condition: [$(round.(test_x0, digits=2))]")
println("Training time domain: $max_hybrid_time_training s")
println("Validation time domain: $validation_time s")

# Solve both systems
solver_config_test = HybridSolverConfig(verbose=false, dtmax=5e-3)
true_solution_val = solve(true_ball, test_x0, validation_time; config=solver_config_test)
neural_solution_val = solve(standalone_neural, test_x0, validation_time; config=solver_config_test, p=parameter_history[end])

plot(true_solution_val, x -> x[1], flow_color=[:steelblue], jump_color=[:navy])
plot!(neural_solution_val, x -> x[1], flow_color=[:orange], jump_color=[:coral])

# Create individual plots and combine them for superimposed visualization
plt_true_height = plot(true_solution_val, x -> x[1], flow_color=:steelblue, jump_color=:navy,
    title="Height Validation - Extrapolation Test", ylabel="Height (m)")
plt_neural_height = plot!(neural_solution_val, x -> x[1], flow_color=:orange, jump_color=:coral,
    linestyle=:dash)

# Add vertical line to show training domain boundary
trueBallTestSol = solve(true_ball, test_x0, max_hybrid_time_training; config=solver_config_test)
trueBallTestSol.final_time
trueBallTestSol.total_jumps

tmax, jmax = (trueBallTestSol.final_time, trueBallTestSol.total_jumps)
hy.vline!((tmax, jmax), color=:red, linestyle=:dot, linewidth=2, label="")


# Add legend manually
plot!([], [], color=:steelblue, linewidth=2, label="True System")
plot!([], [], color=:orange, linewidth=2, linestyle=:dash, label="Neural System")
plot!([], [], color=:red, linestyle=:dot, linewidth=2, label="Training Domain")

println("\n2) OPTIMIZATION HISTORY ANIMATION")
println("-"^40)

# Create animation of optimization history
println("Creating optimization animation with $(length(parameter_history)) frames...")

anim = Animation()
step_size = max(1, div(length(parameter_history), 100))  # Limit to ~100 frames for reasonable file size

# Add initial freeze frames (show first frame for 2 seconds at 5fps = 10 frames)
initial_params = parameter_history[1]
initial_solution = solve(aug_sys, combine_states(x0, x0), validation_time; config=HybridSolverConfig(verbose=false), p=initial_params)

plt_height_initial = plot(initial_solution, x -> [x[1]; x[3]], flow_color=[:steelblue, :orange], jump_color=[:navy, :coral],
    title="Training Progress - Initial (Iteration 0)", ylabel="Height (m)")
plot!([], [], color=:steelblue, linewidth=2, label="True")
plot!([], [], color=:orange, linewidth=2, label="Neural")

plt_velocity_initial = plot(initial_solution, x -> [x[2]; x[4]], flow_color=[:steelblue, :orange], jump_color=[:navy, :coral],
    title="", ylabel="Velocity (m/s)")
plot!([], [], color=:steelblue, linewidth=2, label="True")
plot!([], [], color=:orange, linewidth=2, label="Neural")

plt_cost_initial = plot(title="", xlabel="Iteration", ylabel="Cost (log scale)", yscale=:log10)
plot!(1:1, cost_history[1:1], linewidth=2, color=:crimson, label="Cost")

plt_initial = plot(plt_height_initial, plt_velocity_initial, plt_cost_initial,
    layout=@layout([a; b; c{0.4h}]), size=(900, 1000),
    top_margin=2Plots.mm, bottom_margin=2Plots.mm)

# Add 10 freeze frames (2 seconds at 5fps)
for _ in 1:10
    frame(anim, plt_initial)
end



for (i, params) in enumerate(parameter_history[1:step_size:end])
    # Solve augmented system with current parameters over validation time domain
    current_aug_solution = solve(aug_sys, combine_states(test_x0, test_x0), validation_time; config=HybridSolverConfig(verbose=false), p=params)

    # Create individual plots and combine them (same as in callback)
    plt_height_anim = plot(current_aug_solution, x -> [x[1]; x[3]], flow_color=[:steelblue, :orange], jump_color=[:navy, :coral],
        title="Training Progress - Iteration $(i*step_size)", ylabel="Height (m)")

    # Add vertical line at training time boundary
    hy.vline!((tmax, jmax), color=:red, linestyle=:dash, linewidth=2, label="")

    plot!([], [], color=:steelblue, linewidth=2, label="True")
    plot!([], [], color=:orange, linewidth=2, label="Neural")

    # Create velocity plots
    plt_velocity_anim = plot(current_aug_solution, x -> [x[2]; x[4]], flow_color=[:steelblue, :orange], jump_color=[:navy, :coral],
        title="", ylabel="Velocity (m/s)")

    # Add vertical line at training time boundary  
    hy.vline!((tmax, jmax), color=:red, linestyle=:dash, linewidth=2, label="")

    plot!([], [], color=:steelblue, linewidth=2, label="True")
    plot!([], [], color=:orange, linewidth=2, label="Neural")

    # Cost evolution plot (same as in callback)
    current_costs = cost_history[1:min(i * step_size, end)]
    plt_cost_anim = plot(title="", xlabel="Iteration", ylabel="Cost (log scale)", yscale=:log10)

    # Plot instantaneous cost with transparency
    plot!(1:length(current_costs), current_costs,
        linewidth=1, alpha=0.3, label="Instantaneous Cost", color=:crimson)

    # Add moving averages for cleaner visualization
    short_ma_window = 20
    long_ma_window = 250

    if length(current_costs) >= short_ma_window
        short_ma = [sum(current_costs[max(1, j - (short_ma_window - 1)):j]) / min(j, short_ma_window)
                    for j in eachindex(current_costs)]
        plot!(plt_cost_anim, 1:length(short_ma), short_ma,
            linewidth=3, color=:steelblue, label="MA$(short_ma_window)")
    end

    if length(current_costs) >= long_ma_window
        long_ma = [sum(current_costs[max(1, j - (long_ma_window - 1)):j]) / min(j, long_ma_window)
                   for j in eachindex(current_costs)]
        plot!(plt_cost_anim, 1:length(long_ma), long_ma,
            linewidth=3, color=:darkgreen, label="MA$(long_ma_window)")
    end

    # Combined plot
    plt_combined_anim = plot(plt_height_anim, plt_velocity_anim, plt_cost_anim,
        layout=@layout([a; b; c{0.4h}]), size=(900, 1000),
        top_margin=2Plots.mm, bottom_margin=2Plots.mm)

    frame(anim, plt_combined_anim)

    if i % 10 == 0
        println("  Frame $i/$(div(length(parameter_history), step_size))")
    end
end

# Save animation
data_path = joinpath(dirname(@__FILE__),
    "neural_hybrid_data")

animation_file = joinpath(data_path, "training_animation_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).gif")
gif(anim, animation_file, fps=5)
println("Animation saved to: $animation_file")


################################################################################
#### Paper Plot: Hint = (C×R, f×f̂, D×R, g×ĝ)
################################################################################
println("\n" * "="^60)
println("PAPER PLOT")
println("="^60)

figs_dir = joinpath(dirname(@__FILE__), "figures")
mkpath(figs_dir)
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

n_iterations = length(parameter_history)
snapshot_indices = [1, n_iterations]
snapshot_labels = ["Initial", "Final"]

tmax, jmax = (trueBallTestSol.final_time, trueBallTestSol.total_jumps)

true_color = :dodgerblue
neural_color = :darkorange
true_jump = :navy
neural_jump = :orangered


all_traj_plots = []

for (snap_idx, (iter_idx, label)) in enumerate(zip(snapshot_indices, snapshot_labels))    
    params = parameter_history[iter_idx]
    current_solution = solve(aug_sys, combine_states(test_x0, test_x0),
                            validation_time;
                            config=HybridSolverConfig(verbose=false),
                            p=params)
    
    p_x1 = hy.HybridPlot(:hybrid)
    hy.vline!(p_x1, (tmax, jmax), style=:dash, color=:black, alpha=0.7)
    plot!(p_x1, current_solution, x -> [x[1]; x[3]],
          flow_color=[true_color, neural_color],
          jump_color=[true_jump, neural_jump],
          markersize=3,
          ylabel=L"x_1",
          legend=false,
          title=label)
    
    push!(all_traj_plots, p_x1.plot_obj...)
end

# Set y-labels with larger font for t plots, hide labels for j plots
plot!(all_traj_plots[1], ylabel=L"x_1", guidefontsize=12)
plot!(all_traj_plots[2], ylabel="", yformatter=_->"")  # Keep ticks, hide labels
plot!(all_traj_plots[3], ylabel=L"x_1", guidefontsize=12)
plot!(all_traj_plots[4], ylabel="", yformatter=_->"")  # Keep ticks, hide labels

# Add x-labels
plot!(all_traj_plots[1], xlabel=L"t")
plot!(all_traj_plots[2], xlabel=L"j")
plot!(all_traj_plots[3], xlabel=L"t")
plot!(all_traj_plots[4], xlabel=L"j")

plt_cost = plot(xlabel="Iteration", ylabel="Loss",
                yscale=:log10, legend=:topright,
                xrotation=0,
                bottom_margin=2Plots.mm)

plot!(plt_cost, 1:length(cost_history), cost_history,
    linewidth=3, alpha=0.2, label="Loss", color=:gray)

short_ma_window = 20
long_ma_window = 250

if length(cost_history) >= short_ma_window
    short_ma = [sum(cost_history[max(1, j - (short_ma_window - 1)):j]) / min(j, short_ma_window)
                for j in eachindex(cost_history)]
    plot!(plt_cost, 1:length(short_ma), short_ma,
        linewidth=3, color=:mediumseagreen, label="MA$short_ma_window")
end

if length(cost_history) >= long_ma_window
    long_ma = [sum(cost_history[max(1, j - (long_ma_window - 1)):j]) / min(j, long_ma_window)
               for j in eachindex(cost_history)]
    plot!(plt_cost, 1:length(long_ma), long_ma,
        linewidth=3, color=:mediumpurple, label="MA$long_ma_window")
end

for idx in snapshot_indices
    Plots.vline!(plt_cost, [idx], linestyle=:dash, color=:black, alpha=0.3, linewidth=1, label="")
end


plt_legend = plot(xlims=(0,1), ylims=(0,1), 
                 framestyle=:none, 
                 legend=:inside,
                 legend_position=:top,
                 legend_columns=3,
                 grid=false,
                 showaxis=false,
                 ticks=false)
plot!(plt_legend, [], [], color=true_color, linewidth=3, label="Nominal")
plot!(plt_legend, [], [], color=neural_color, linewidth=3, label="Neural")
plot!(plt_legend, [], [], color=:black, linestyle=:dash, linewidth=1.5, label="Training Limit")


final_plot = plot(
    plt_legend,
    all_traj_plots...,
    plt_cost,
    layout=@layout([
        a{0.04h}
        grid(1, 4){0.45h}
        b{0.35h}
    ]),
    size=(1400, 550),
    link=:none,
    left_margin=4Plots.mm,
    right_margin=2Plots.mm,
    bottom_margin=4Plots.mm,
    top_margin=1Plots.mm
)

# Link y-axes pairwise: t plots together, j plots together
plot!(final_plot, subplot=3, ylink=1)  # final_t links to initial_t
plot!(final_plot, subplot=4, ylink=2)  # final_j links to initial_j

figure_file = joinpath(figs_dir, "combined_final_$timestamp.pdf")
savefig(final_plot, figure_file)

println("\nFigure saved to: $figure_file")





loaded_results = load_training_results_hdf5(data_file)
best_trained_params = loaded_results["best_params"]
cost_history = loaded_results["cost_history"]
parameter_history = loaded_results["parameter_history"]
x0_batch = loaded_results["x0_batch"]

println("\n1) VALIDATION WITH EXTRAPOLATION TEST")
println("-"^40)

# Create standalone neural system with true flow/jump sets
standalone_neural = create_standalone_neural_system(true_ball, neural_sys)

# Use one of the random initial conditions but with longer time domain
test_x0 = [10., 3.]
max_hybrid_time_training = 10.0
validation_time = 20.0  # Longer than training time for extrapolation test

println("Test initial condition: [$(round.(test_x0, digits=2))]")
println("Training time domain: $max_hybrid_time_training s")
println("Validation time domain: $validation_time s")

# Solve both systems
solver_config_test = HybridSolverConfig(verbose=false, dtmax=5e-3)
true_solution_val = solve(true_ball, test_x0, validation_time; config=solver_config_test)
neural_solution_val = solve(standalone_neural, test_x0, validation_time; config=solver_config_test, p=parameter_history[end])

plot(true_solution_val, x -> x[1], flow_color=[:steelblue], jump_color=[:navy])
plot!(neural_solution_val, x -> x[1], flow_color=[:orange], jump_color=[:coral])

# Create individual plots and combine them for superimposed visualization
plt_true_height = plot(true_solution_val, x -> x[1], flow_color=:steelblue, jump_color=:navy,
    title="Height Validation - Extrapolation Test", ylabel="Height (m)")
plt_neural_height = plot!(neural_solution_val, x -> x[1], flow_color=:orange, jump_color=:coral,
    linestyle=:dash)

# Add vertical line to show training domain boundary
trueBallTestSol = solve(true_ball, test_x0, max_hybrid_time_training; config=solver_config_test)
trueBallTestSol.final_time
trueBallTestSol.total_jumps

tmax, jmax = (trueBallTestSol.final_time, trueBallTestSol.total_jumps)
hy.vline!((tmax, jmax), color=:red, linestyle=:dot, linewidth=2, label="")


# Add legend manually
plot!([], [], color=:steelblue, linewidth=2, label="True System")
plot!([], [], color=:orange, linewidth=2, linestyle=:dash, label="Neural System")
plot!([], [], color=:red, linestyle=:dot, linewidth=2, label="Training Domain")

println("\n2) OPTIMIZATION HISTORY ANIMATION")
println("-"^40)

# Create animation of optimization history
println("Creating optimization animation with $(length(parameter_history)) frames...")

anim = Animation()
step_size = max(1, div(length(parameter_history), 100))  # Limit to ~100 frames for reasonable file size

# Add initial freeze frames (show first frame for 2 seconds at 5fps = 10 frames)
initial_params = parameter_history[1]
initial_solution = solve(aug_sys, combine_states(x0, x0), validation_time; config=HybridSolverConfig(verbose=false), p=initial_params)

plt_height_initial = plot(initial_solution, x -> [x[1]; x[3]], flow_color=[:steelblue, :orange], jump_color=[:navy, :coral],
    title="Training Progress - Initial (Iteration 0)", ylabel="Height (m)")
plot!([], [], color=:steelblue, linewidth=2, label="True")
plot!([], [], color=:orange, linewidth=2, label="Neural")

plt_velocity_initial = plot(initial_solution, x -> [x[2]; x[4]], flow_color=[:steelblue, :orange], jump_color=[:navy, :coral],
    title="", ylabel="Velocity (m/s)")
plot!([], [], color=:steelblue, linewidth=2, label="True")
plot!([], [], color=:orange, linewidth=2, label="Neural")

plt_cost_initial = plot(title="", xlabel="Iteration", ylabel="Cost (log scale)", yscale=:log10)
plot!(1:1, cost_history[1:1], linewidth=2, color=:crimson, label="Cost")

plt_initial = plot(plt_height_initial, plt_velocity_initial, plt_cost_initial,
    layout=@layout([a; b; c{0.4h}]), size=(900, 1000),
    top_margin=2Plots.mm, bottom_margin=2Plots.mm)

# Add 10 freeze frames (2 seconds at 5fps)
for _ in 1:10
    frame(anim, plt_initial)
end



for (i, params) in enumerate(parameter_history[1:step_size:end])
    # Solve augmented system with current parameters over validation time domain
    current_aug_solution = solve(aug_sys, combine_states(test_x0, test_x0), validation_time; config=HybridSolverConfig(verbose=false), p=params)

    # Create individual plots and combine them (same as in callback)
    plt_height_anim = plot(current_aug_solution, x -> [x[1]; x[3]], flow_color=[:steelblue, :orange], jump_color=[:navy, :coral],
        title="Training Progress - Iteration $(i*step_size)", ylabel="Height (m)")

    # Add vertical line at training time boundary
    hy.vline!((tmax, jmax), color=:red, linestyle=:dash, linewidth=2, label="")

    plot!([], [], color=:steelblue, linewidth=2, label="True")
    plot!([], [], color=:orange, linewidth=2, label="Neural")

    # Create velocity plots
    plt_velocity_anim = plot(current_aug_solution, x -> [x[2]; x[4]], flow_color=[:steelblue, :orange], jump_color=[:navy, :coral],
        title="", ylabel="Velocity (m/s)")

    # Add vertical line at training time boundary  
    hy.vline!((tmax, jmax), color=:red, linestyle=:dash, linewidth=2, label="")

    plot!([], [], color=:steelblue, linewidth=2, label="True")
    plot!([], [], color=:orange, linewidth=2, label="Neural")

    # Cost evolution plot (same as in callback)
    current_costs = cost_history[1:min(i * step_size, end)]
    plt_cost_anim = plot(title="", xlabel="Iteration", ylabel="Cost (log scale)", yscale=:log10)

    # Plot instantaneous cost with transparency
    plot!(1:length(current_costs), current_costs,
        linewidth=1, alpha=0.3, label="Instantaneous Cost", color=:crimson)

    # Add moving averages for cleaner visualization
    short_ma_window = 20
    long_ma_window = 250

    if length(current_costs) >= short_ma_window
        short_ma = [sum(current_costs[max(1, j - (short_ma_window - 1)):j]) / min(j, short_ma_window)
                    for j in eachindex(current_costs)]
        plot!(plt_cost_anim, 1:length(short_ma), short_ma,
            linewidth=3, color=:steelblue, label="MA$(short_ma_window)")
    end

    if length(current_costs) >= long_ma_window
        long_ma = [sum(current_costs[max(1, j - (long_ma_window - 1)):j]) / min(j, long_ma_window)
                   for j in eachindex(current_costs)]
        plot!(plt_cost_anim, 1:length(long_ma), long_ma,
            linewidth=3, color=:darkgreen, label="MA$(long_ma_window)")
    end

    # Combined plot
    plt_combined_anim = plot(plt_height_anim, plt_velocity_anim, plt_cost_anim,
        layout=@layout([a; b; c{0.4h}]), size=(900, 1000),
        top_margin=2Plots.mm, bottom_margin=2Plots.mm)

    frame(anim, plt_combined_anim)

    if i % 10 == 0
        println("  Frame $i/$(div(length(parameter_history), step_size))")
    end
end

# Save animation
data_path = joinpath(dirname(@__FILE__),
    "neural_hybrid_data")

animation_file = joinpath(data_path, "training_animation_$(Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")).gif")
gif(anim, animation_file, fps=5)
println("Animation saved to: $animation_file")


################################################################################
#### Paper Plot: Ĥ_int = (C×C,f×f̂, D×D, g×ĝ)
################################################################################
println("\n" * "="^60)
println("PAPER PLOT")
println("="^60)

figs_dir = joinpath(dirname(@__FILE__), "figures")
mkpath(figs_dir)
timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

# Create standalone neural system with true flow/jump sets
standalone_neural = create_standalone_neural_system(true_ball, neural_sys)

n_iterations = length(parameter_history)
snapshot_indices = [1, n_iterations]
snapshot_labels = ["Initial", "Final"]

tmax, jmax = (trueBallTestSol.final_time, trueBallTestSol.total_jumps)

true_color = :dodgerblue
neural_color = :darkorange
true_jump = :navy
neural_jump = :orangered

println("Testing standalone neural system with true sets...")

all_traj_plots = []

for (snap_idx, (iter_idx, label)) in enumerate(zip(snapshot_indices, snapshot_labels))    
    params = parameter_history[iter_idx]
    
    # Solve true system
    true_solution = solve(true_ball, test_x0, validation_time;
                         config=HybridSolverConfig(verbose=false))
    
    # Solve standalone neural system with true sets
    neural_solution = solve(standalone_neural, test_x0, validation_time;
                           config=HybridSolverConfig(verbose=false),
                           p=params)
    
    p_x1 = hy.HybridPlot(:hybrid)
    hy.vline!(p_x1, (tmax, jmax), style=:dash, color=:black, alpha=0.7)
    
    # Plot true system
    plot!(p_x1, true_solution, x -> x[1],
          flow_color=true_color,
          jump_color=true_jump,
          markersize=3,
          ylabel=L"x_1",
          legend=false,
          title=label)
    
    # Plot neural system
    plot!(p_x1, neural_solution, x -> x[1],
          flow_color=neural_color,
          jump_color=neural_jump,
          markersize=3,
          legend=false)
    
    push!(all_traj_plots, p_x1.plot_obj...)
end

# Set y-labels with larger font for t plots, hide labels for j plots
plot!(all_traj_plots[1], ylabel=L"x_1", guidefontsize=12)
plot!(all_traj_plots[2], ylabel="", yformatter=_->"")
plot!(all_traj_plots[3], ylabel=L"x_1", guidefontsize=12)
plot!(all_traj_plots[4], ylabel="", yformatter=_->"")

# Add x-labels
plot!(all_traj_plots[1], xlabel=L"t")
plot!(all_traj_plots[2], xlabel=L"j")
plot!(all_traj_plots[3], xlabel=L"t")
plot!(all_traj_plots[4], xlabel=L"j")

println("Creating cost history plot...")

plt_cost = plot(xlabel="Iteration", ylabel="Loss",
                yscale=:log10, legend=:topright,
                xrotation=0,
                bottom_margin=2Plots.mm)

plot!(plt_cost, 1:length(cost_history), cost_history,
    linewidth=3, alpha=0.2, label="Loss", color=:gray)

short_ma_window = 20
long_ma_window = 250

if length(cost_history) >= short_ma_window
    short_ma = [sum(cost_history[max(1, j - (short_ma_window - 1)):j]) / min(j, short_ma_window)
                for j in eachindex(cost_history)]
    plot!(plt_cost, 1:length(short_ma), short_ma,
        linewidth=3, color=:mediumseagreen, label="MA$short_ma_window")
end

if length(cost_history) >= long_ma_window
    long_ma = [sum(cost_history[max(1, j - (long_ma_window - 1)):j]) / min(j, long_ma_window)
               for j in eachindex(cost_history)]
    plot!(plt_cost, 1:length(long_ma), long_ma,
        linewidth=3, color=:mediumpurple, label="MA$long_ma_window")
end

for idx in snapshot_indices
    Plots.vline!(plt_cost, [idx], linestyle=:dash, color=:black, alpha=0.3, linewidth=1, label="")
end

println("Creating legend...")

plt_legend = plot(xlims=(0,1), ylims=(0,1), 
                 framestyle=:none, 
                 legend=:inside,
                 legend_position=:top,
                 legend_columns=3,
                 grid=false,
                 showaxis=false,
                 ticks=false)
plot!(plt_legend, [], [], color=true_color, linewidth=3, label="Nominal")
plot!(plt_legend, [], [], color=neural_color, linewidth=3, label="Neural")
plot!(plt_legend, [], [], color=:black, linestyle=:dash, linewidth=1.5, label="Training Limit")

println("Assembling final figure...")

final_plot = plot(
    plt_legend,
    all_traj_plots...,
    plt_cost,
    layout=@layout([
        a{0.04h}
        grid(1, 4){0.45h}
        b{0.35h}
    ]),
    size=(1400, 550),
    link=:none,
    left_margin=4Plots.mm,
    right_margin=2Plots.mm,
    bottom_margin=4Plots.mm,
    top_margin=1Plots.mm
)

# Link y-axes pairwise: t plots together, j plots together
plot!(final_plot, subplot=3, ylink=1)
plot!(final_plot, subplot=4, ylink=2)

figure_file = joinpath(figs_dir, "combined_final_$timestamp.pdf")
savefig(final_plot, figure_file)

println("\nFigure saved to: $figure_file")
println("="^60)