# Tutorials

Step-by-step tutorials for learning Prismo through practical examples.

These tutorials are designed to be followed in order, building your understanding progressively.

## Tutorial Overview

```{toctree}
:maxdepth: 1

basic_simulation
waveguide_coupling
sparameters
optimization
```

## What You'll Learn

### 1. Basic Simulation

Build your first complete FDTD simulation from scratch. Learn about:

- Setting up a simulation domain
- Adding sources and monitors
- Running simulations
- Visualizing results

### 2. Waveguide Coupling

Design and simulate waveguide structures. Topics include:

- Creating waveguide geometries
- Using the mode solver
- Mode injection with ModeSource
- Analyzing coupling efficiency

### 3. S-Parameters

Extract S-parameters from two-port devices. You'll learn:

- Setting up port monitors
- Mode decomposition
- Calculating reflection (S11) and transmission (S21)
- Frequency-domain analysis

### 4. Parameter Optimization

Optimize device parameters using sweeps. Covers:

- Parameter sweep setup
- Performance metrics
- Finding optimal designs
- Visualization of design space

## Prerequisites

Before starting these tutorials, you should:

- Have Prismo installed (see {doc}`../user_guide/installation`)
- Be familiar with basic Python and NumPy
- Understand basic electromagnetics concepts

## Tutorial Format

Each tutorial includes:

- **Learning objectives** - What you'll accomplish
- **Code examples** - Fully working, copy-paste ready code
- **Explanations** - Why each step matters
- **Exercises** - Try it yourself challenges
- **Solutions** - Reference implementations

## Getting Help

If you get stuck:

1. Check the {doc}`../user_guide/index` for concept explanations
2. Review the {doc}`../api/index` for detailed API information
3. Look at the {doc}`../examples/index` for more code samples
4. Open an issue on [GitHub](https://github.com/rithulkamesh/prismo/issues)

Ready to start? Begin with {doc}`basic_simulation`!
