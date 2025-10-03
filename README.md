# NeuralHyEq - Submitted to ACC 2026

Neural approximation of hybrid dynamical systems using differentiable hybrid equations solvers.

## Overview

This repository contains the implementation code associated with the paper:

> **Title**: Approximation of Hybrid Dynamical Systems via Neural Hybrid Equations
> **Authors**: Daniel E. Ochoa, Ricardo G. Sanfelice
> **Submitted to**: American Control Conference (ACC) 2026

The code demonstrates training neural networks to approximate hybrid system dynamics by evolving nominal and neural solutions in tandem with gradient-based optimization through hybrid time.

## Training Animation

![Training Progress](neural_hybrid_data/training_animation_2025-09-29_22-48-19.gif)

The animation shows the neural network learning to approximate a bouncing ball system, with the cost decreasing over training iterations.

## Code Structure

- **[HybridSolver.jl](HybridSolver.jl)** - Julia-based hybrid equations solver with ForwardDiff.jl compatibility for automatic differentiation through hybrid trajectories
- **[neuralHyEq.jl](neuralHyEq.jl)** - Complete neural hybrid system training framework with bouncing ball example

## Connection to Paper

This implementation corresponds to the training methodology described in the ACC 2026 submission. As mentioned in the paper:

> "We employ a loss function θ↦ℒ(θ) that measures the error between nominal sample solutions and neural solutions, and find suitable parameters by optimizing ℒ via gradient-based methods. For this, we use a Julia-based hybrid equations solver with autodifferentiable numerical integration, which enables backpropagation of gradients through the continuous-time and discrete-time evolution of ℋₐᶿ."

The **Julia-based hybrid equations solver** referenced in the paper is `HybridSolver.jl`, which implements proper hybrid time domain evolution with ForwardDiff support.

The interconnected system ℋ_int mentioned in the paper is implemented as `AugmentedHybridSystem` in [neuralHyEq.jl](neuralHyEq.jl), which evolves both the true system ℋ and the neural approximation ℋₐᶿ simultaneously to enable gradient computation.

Repository URL referenced in paper footnote: https://bit.ly/neuralHyEQCode

## Requirements

### Julia Version
- Julia 1.9 or higher

### Dependencies
The following Julia packages are required:

**Core Solver:**
- `OrdinaryDiffEq.jl` - High-performance ODE solvers
- `ForwardDiff.jl` - Automatic differentiation for gradient computation
- `Plots.jl` - Visualization

**Neural Training:**
- `SimpleChains.jl` - Fast neural network implementation
- `Optimization.jl` - Optimization framework
- `OptimizationOptimisers.jl` - Adam and other optimizers
- `HDF5.jl` - Data persistence
- `LaTeXStrings.jl` - LaTeX formatting in plots
- `Statistics.jl`, `LinearAlgebra.jl`, `Dates.jl` - Standard library utilities

### Installation

```julia
using Pkg
Pkg.add(["OrdinaryDiffEq", "ForwardDiff", "Plots", "SimpleChains",
         "Optimization", "OptimizationOptimisers", "HDF5", "LaTeXStrings"])
```

## Usage

Run the complete training example:
```julia
include("neuralHyEq.jl")
```

This will:
1. Train a neural hybrid system to approximate bouncing ball dynamics
2. Generate training progress animations
3. Save results to `neural_hybrid_data/`
4. Create publication-quality figures in `figures/`
