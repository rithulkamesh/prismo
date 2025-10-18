# Tutorial 4: Parameter Optimization

**Time**: 40 minutes  
**Difficulty**: Advanced  
**Prerequisites**: Tutorials 1-3

## Learning Objectives

- Perform parameter sweeps
- Optimize device performance
- Visualize design space
- Find optimal operating points

## The Challenge: Optimize a Taper

We'll optimize a waveguide taper to minimize insertion loss.

## Approach

We'll sweep two parameters:

1. **Taper length** (5-20 μm)
2. **Taper shape** (linear vs exponential vs polynomial)

## Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from prismo.optimization import ParameterSweep

# Define parameters to sweep
lengths = np.linspace(5e-6, 20e-6, 8)  # 8 different lengths
shapes = ['linear', 'exponential', 'polynomial']

# Setup parameter sweep
sweep = ParameterSweep(
    parameters={
        'length': lengths,
        'shape': shapes,
    },
    metric='insertion_loss',  # What to optimize
)

# Define simulation function
def simulate_taper(length, shape):
    """Run simulation for given taper parameters."""

    # Create taper with specified parameters
    taper = create_taper(length=length, shape=shape)

    # Setup and run simulation
    sim = setup_simulation(taper)
    sim.run(100e-15)

    # Extract insertion loss
    s_params = extract_s_parameters(sim)
    insertion_loss = -20 * np.log10(np.abs(s_params['S21']))

    return insertion_loss[0]  # At design frequency

# Run sweep
results = sweep.run(simulate_taper, parallel=True, n_jobs=4)

# Find optimal parameters
optimal = sweep.get_optimal()
print(f"Optimal parameters:")
print(f"  Length: {optimal['length']*1e6:.2f} μm")
print(f"  Shape: {optimal['shape']}")
print(f"  Insertion Loss: {optimal['metric']:.3f} dB")

# Visualize design space
sweep.plot_2d(
    x='length',
    y='shape',
    metric='insertion_loss',
    cmap='viridis_r',  # Reverse so dark = better
)
plt.savefig('tutorial4_optimization.png', dpi=150)
plt.show()
```

## Manual Parameter Sweep

For more control, implement sweeps manually:

```python
# Parameter ranges
lengths = np.linspace(5e-6, 20e-6, 10)
widths_in = [2e-6, 3e-6, 4e-6]

# Results storage
results = np.zeros((len(widths_in), len(lengths)))

# Sweep
for i, width_in in enumerate(widths_in):
    for j, length in enumerate(lengths):
        # Run simulation
        il = simulate_taper_config(width_in, length)
        results[i, j] = il

        print(f"  w={width_in*1e6:.1f}μm, L={length*1e6:.1f}μm: IL={il:.3f}dB")

# Plot heatmap
plt.figure(figsize=(10, 6))
plt.imshow(
    results,
    aspect='auto',
    origin='lower',
    extent=[lengths[0]*1e6, lengths[-1]*1e6, 0, len(widths_in)],
    cmap='RdYlGn_r',  # Red = bad, Green = good
)
plt.colorbar(label='Insertion Loss (dB)')
plt.xlabel('Taper Length (μm)')
plt.ylabel('Input Width Configuration')
plt.yticks(range(len(widths_in)), [f'{w*1e6:.1f} μm' for w in widths_in])
plt.title('Taper Optimization: Insertion Loss')
plt.tight_layout()
plt.savefig('manual_sweep.png', dpi=150)
plt.show()

# Find optimal
min_idx = np.unravel_index(np.argmin(results), results.shape)
optimal_width = widths_in[min_idx[0]]
optimal_length = lengths[min_idx[1]]
optimal_loss = results[min_idx]

print(f"\nOptimal design:")
print(f"  Width: {optimal_width*1e6:.1f} μm")
print(f"  Length: {optimal_length*1e6:.1f} μm")
print(f"  Loss: {optimal_loss:.3f} dB")
```

## Gradient-Free Optimization

For complex optimization, use scipy:

```python
from scipy.optimize import minimize

def objective_function(params):
    """Objective to minimize."""
    length, width_ratio = params

    # Constraints
    if length < 5e-6 or length > 25e-6:
        return 1000.0  # Penalty
    if width_ratio < 1.5 or width_ratio > 4.0:
        return 1000.0

    # Run simulation
    il = simulate_taper_config(length=length, width_ratio=width_ratio)

    return il  # Minimize insertion loss

# Initial guess
x0 = [10e-6, 2.5]  # [length, width_ratio]

# Optimize
result = minimize(
    objective_function,
    x0,
    method='Nelder-Mead',  # Gradient-free
    options={'maxiter': 50, 'disp': True}
)

print(f"\nOptimized parameters:")
print(f"  Length: {result.x[0]*1e6:.2f} μm")
print(f"  Width ratio: {result.x[1]:.2f}")
print(f"  Final IL: {result.fun:.3f} dB")
```

## Multi-Objective Optimization

Optimize for multiple goals:

```python
def multi_objective(params):
    """Optimize insertion loss AND bandwidth."""

    # Run simulation
    sim_results = simulate_full(params)

    insertion_loss = sim_results['IL']
    bandwidth_3db = sim_results['BW']

    # Combined objective (weighted sum)
    weight_il = 0.7
    weight_bw = 0.3

    # Normalize and combine
    objective = weight_il * insertion_loss - weight_bw * bandwidth_3db

    return objective, {'IL': insertion_loss, 'BW': bandwidth_3db}

# Run optimization with multiple objectives
# ... (use Pareto optimization or weighted sum)
```

## Visualization Techniques

### 1. Contour Plot

```python
# Create 2D grid
L = np.linspace(5e-6, 20e-6, 50)
W = np.linspace(2e-6, 4e-6, 50)
LL, WW = np.meshgrid(L, W)

# Evaluate objective (use fast surrogate model)
Z = np.array([[objective_function([l, w]) for l in L] for w in W])

# Plot
plt.figure(figsize=(10, 8))
contour = plt.contour(LL*1e6, WW*1e6, Z, levels=20, cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)
plt.xlabel('Length (μm)')
plt.ylabel('Width (μm)')
plt.title('Optimization Landscape')
plt.colorbar(label='Insertion Loss (dB)')
plt.tight_layout()
plt.savefig('contour_plot.png', dpi=150)
plt.show()
```

### 2. Parallel Coordinates

```python
import pandas as pd
from pandas.plotting import parallel_coordinates

# Create DataFrame of results
df = pd.DataFrame({
    'Length': lengths_tested,
    'Width': widths_tested,
    'IL': insertion_losses,
    'BW': bandwidths,
    'Quality': ['Good' if il < 0.5 else 'Bad' for il in insertion_losses]
})

# Plot
plt.figure(figsize=(12, 6))
parallel_coordinates(df, 'Quality', color=['red', 'green'])
plt.ylabel('Normalized Value')
plt.title('Parameter Space Exploration')
plt.tight_layout()
plt.savefig('parallel_coords.png', dpi=150)
plt.show()
```

## Best Practices

1. **Start coarse, refine later**: Use wide spacing initially, zoom in on promising regions
2. **Check convergence**: Re-run optimal point to verify repeatability
3. **Use surrogate models**: For expensive simulations, fit a fast model
4. **Validate physically**: Ensure optimal parameters make physical sense
5. **Consider manufacturing**: Include fabrication tolerances

## Performance Tips

```python
# 1. Parallel execution
from multiprocessing import Pool

def parallel_sweep(params_list):
    with Pool(processes=8) as pool:
        results = pool.map(simulate_single, params_list)
    return results

# 2. Caching results
import pickle

def cached_simulation(params):
    cache_file = f"cache_{hash(tuple(params))}.pkl"

    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    result = run_simulation(params)

    with open(cache_file, 'wb') as f:
        pickle.dump(result, f)

    return result

# 3. Adaptive sampling
# Focus computational effort on interesting regions
# (implement using Gaussian Processes or similar)
```

## Complete Example: Ring Resonator Optimization

```python
# Optimize ring resonator for maximum Q-factor
def optimize_ring():
    # Parameters to optimize
    radii = np.linspace(5e-6, 15e-6, 20)
    gaps = np.linspace(100e-9, 500e-9, 20)

    best_q = 0
    best_params = None

    for radius in radii:
        for gap in gaps:
            # Simulate
            q_factor = simulate_ring(radius=radius, gap=gap)

            if q_factor > best_q:
                best_q = q_factor
                best_params = (radius, gap)

    print(f"Optimal design:")
    print(f"  Radius: {best_params[0]*1e6:.2f} μm")
    print(f"  Gap: {best_params[1]*1e9:.0f} nm")
    print(f"  Q-factor: {best_q:.0f}")

    return best_params

optimal = optimize_ring()
```

## Exercises

1. Optimize a Y-branch splitter for equal power splitting
2. Find the shortest taper with < 0.1 dB loss
3. Optimize a grating coupler for maximum efficiency
4. Multi-objective: minimize loss AND maximize bandwidth

## Summary

**You've learned:**

- Parameter sweep techniques
- Single and multi-objective optimization
- Visualization of design spaces
- Performance optimization strategies

**Key Takeaway**: Systematic optimization reveals non-intuitive designs that outperform intuition!

## Further Reading

- Scipy optimization documentation
- {doc}`../user_guide/advanced` - Advanced optimization techniques
- {doc}`../api/optimization` - Optimization API reference
