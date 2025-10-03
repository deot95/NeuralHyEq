## Overview

This repository contains the implementation code associated with the paper:

> **Title**: Approximation of Hybrid Dynamical Systems via Neural Hybrid Equations
> 
> **Authors**: Daniel E. Ochoa, Ricardo G. Sanfelice
> 
> **Submitted to**: American Control Conference (ACC) 2026

The code demonstrates training neural networks to approximate hybrid system dynamics by evolving nominal and neural solutions in tandem with gradient-based optimization through hybrid time. The following animation shows the neural network learning to approximate a bouncing ball system, with the cost decreasing over training iterations.

![Training Progress](neural_hybrid_data/training_animation_2025-09-29_22-48-19.gif)

As mentioned in the paper:

> "We employ a loss function ℒ that measures the error between nominal sample solutions and neural solutions, and find suitable parameters by optimizing ℒ via gradient-based methods. For this, we use a Julia-based hybrid equations solver with autodifferentiable numerical integration, which enables backpropagation of gradients through the continuous-time and discrete-time evolution of ℋₐ."

The **Julia-based hybrid equations solver** referenced in the paper is `HybridSolver.jl` in [HybridSolver.jl](HybridSolver.jl), which implements suitable hybrid time domain evolution with ForwardDiff support.

The interconnected system ℋ_int mentioned in the paper is implemented as `AugmentedHybridSystem` in [neuralHyEq.jl](neuralHyEq.jl), which evolves both the true system ℋ and the neural approximation ℋₐ simultaneously to ensure identital hybrid time domains.

## Requirements

### Julia Version
- Julia 1.11 

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

## Usage

From the repository directory, run:
```bash
julia neuralHyEq.jl
```

This will:
1. Train a neural hybrid system to approximate bouncing ball dynamics
2. Generate training progress animations
3. Save numerical results to `neural_hybrid_data/`
4. Create and save figures in `figures/`

## Disclaimer

This code is provided for **academic and research purposes only**. It is a reference implementation accompanying the ACC 2026 submission and is not intended for production use. The code is provided "as-is" without warranty of any kind, and **no ongoing support or maintenance will be provided**.