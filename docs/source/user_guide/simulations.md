# Simulations

Learn how to set up and configure FDTD simulations in Prismo.

## The Simulation Class

The `Simulation` class is the main interface for creating and running FDTD simulations:

```python
from prismo import Simulation

sim = Simulation(
    size=(5.0e-6, 3.0e-6, 0.0),         # Domain size (Lx, Ly, Lz) in meters
    resolution=20e6,                     # Grid resolution (points/meter)
    boundary_conditions="pml",           # Boundary type
    pml_layers=10,                       # PML thickness
    courant_factor=0.9,                  # CFL safety factor
)
```

## Grid Setup

### Domain Size

Specify the physical dimensions of your computational domain:

```python
# 2D simulation (Lz = 0)
sim = Simulation(size=(5.0e-6, 3.0e-6, 0.0), ...)

# 3D simulation
sim = Simulation(size=(5.0e-6, 3.0e-6, 2.0e-6), ...)
```

**Units**: Always in meters
- Use `e-6` for micrometers
- Use `e-9` for nanometers

### Grid Resolution

The resolution determines spatial discretization:

```python
# Uniform resolution (same in all directions)
resolution=20e6  # 20 points per meter = 0.05 µm spacing

# Anisotropic resolution (different per axis)
resolution=(40e6, 40e6, 20e6)  # (res_x, res_y, res_z)
```

**Guidelines:**
- **Minimum**: 20 points per wavelength
- **Recommended**: 30-40 points per wavelength
- **High accuracy**: 50+ points per wavelength

**Example calculation:**
```python
wavelength = 1.55e-6  # 1.55 µm
points_per_wavelength = 40
resolution = points_per_wavelength / wavelength  # 25.8e6 points/m
```

### Grid Properties

Access grid information:

```python
print(f"Grid dimensions: {sim.grid.dimensions}")  # (Nx, Ny, Nz)
print(f"Grid spacing: {sim.grid.spacing}")        # (dx, dy, dz)
print(f"Time step: {sim.dt:.3e} s")               # Courant-limited dt
```

## Boundary Conditions

### PML (Perfectly Matched Layer)

Absorbing boundaries that prevent reflections (default):

```python
sim = Simulation(
    size=(5.0e-6, 3.0e-6, 0.0),
    resolution=20e6,
    boundary_conditions="pml",
    pml_layers=10,  # Number of grid points
)
```

**PML Parameters:**
- `pml_layers`: Thickness in grid points (typically 8-15)
- Thicker PML → better absorption, larger domain

**Performance:**
- Reflection coefficient: < -40 dB (typical)
- Computational overhead: ~15-20% for typical domains

### Periodic Boundaries

For simulating periodic structures:

```python
sim = Simulation(
    size=(5.0e-6, 3.0e-6, 0.0),
    resolution=20e6,
    boundary_conditions="periodic",
)
```

**Use cases:**
- Photonic crystals
- Gratings
- Metamaterials

### Reflecting Boundaries

Perfect electric conductor (PEC) boundaries:

```python
sim = Simulation(
    size=(5.0e-6, 3.0e-6, 0.0),
    resolution=20e6,
    boundary_conditions="reflecting",
)
```

**Use cases:**
- Metal-walled cavities
- Waveguide ports
- Testing and validation

## Time Stepping

### Automatic Time Step

The time step is automatically calculated from the Courant stability condition:

```python
dt = courant_factor * min(dx, dy, dz) / (c * sqrt(dimensionality))
```

where `c` is the speed of light.

**Courant Factor:**
- Default: 0.9 (safe margin)
- Range: 0 < courant_factor < 1
- Higher → faster simulation, less stable
- Lower → more stable, slower

### Manual Time Step Control

Override automatic calculation (advanced):

```python
from prismo.core import FDTDSolver

# Get the grid
grid = sim.grid

# Calculate custom time step
dt_custom = 0.5 * grid.get_time_step(0.9)

# Create solver with custom dt
solver = FDTDSolver(grid, dt_custom)
```

⚠️ **Warning**: Manual time steps must satisfy Courant condition or simulation will be unstable!

### Running Simulations

#### Run for Fixed Time

```python
sim_time = 100e-15  # 100 femtoseconds
sim.run(sim_time)
```

#### Run for Number of Steps

```python
num_steps = 1000
for step in range(num_steps):
    sim.step()  # Single time step
```

#### Run with Progress Callback

```python
def progress_callback(step, total_steps, sim_time, elapsed_time):
    if step % 100 == 0:
        progress = step / total_steps * 100
        print(f"Progress: {progress:.1f}% - Sim time: {sim_time*1e15:.2f} fs")

sim.run(100e-15, progress_callback=progress_callback)
```

## Simulation Workflow

### Complete Example

```python
from prismo import Simulation, TFSFSource, FieldMonitor
import numpy as np

# 1. Create simulation
sim = Simulation(
    size=(4.0e-6, 3.0e-6, 0.0),
    resolution=30e6,
    boundary_conditions="pml",
    pml_layers=12,
)

# 2. Add sources
source = TFSFSource(
    center=(2.0e-6, 1.5e-6, 0.0),
    size=(2.5e-6, 2.0e-6, 0.0),
    direction="+x",
    polarization="y",
    frequency=193.4e12,
    pulse=True,
    pulse_width=20e-15,
)
sim.add_source(source)

# 3. Add monitors
monitor = FieldMonitor(
    center=(2.0e-6, 1.5e-6, 0.0),
    size=(3.5e-6, 2.5e-6, 0.0),
    components=["Ey", "Hz"],
    time_domain=True,
    frequencies=[193.4e12],
)
sim.add_monitor(monitor)

# 4. Run simulation
def progress(step, total, sim_time, elapsed):
    if step % 200 == 0:
        print(f"Step {step}/{total}: {sim_time*1e15:.1f} fs")

sim_time = 200e-15
sim.run(sim_time, progress_callback=progress)

# 5. Analyze results
time_points, ey_data = monitor.get_time_data("Ey")
print(f"Captured {len(time_points)} time steps")

# Peak field
peak = np.max(np.abs(ey_data))
print(f"Peak field: {peak:.2e} V/m")

# Field energy
energy = np.sum(ey_data**2)
print(f"Total energy: {energy:.2e}")
```

## Advanced Configuration

### Custom Grid Specification

For more control over grid setup:

```python
from prismo.core import GridSpec, YeeGrid

# Create custom grid specification
grid_spec = GridSpec(
    size=(5.0e-6, 3.0e-6, 0.0),
    resolution=(40e6, 40e6, 0.0),
    boundary_layers=15,
)

# Create Yee grid
grid = YeeGrid(grid_spec)

# Use in simulation
from prismo.core import FDTDSolver
dt = grid.get_time_step(0.9)
solver = FDTDSolver(grid, dt)
```

### Simulation State Management

```python
# Get current state
print(f"Current time: {sim.current_time:.3e} s")
print(f"Step count: {sim.step_count}")

# Reset simulation
sim.reset()  # Clears fields, resets time

# Check if simulation is running
if sim.step_count > 0:
    print("Simulation has been run")
```

## Performance Optimization

### Resolution vs. Accuracy Trade-off

```python
# Fast but less accurate
sim_fast = Simulation(size=(5.0e-6, 3.0e-6, 0.0), resolution=15e6)

# Balanced (recommended)
sim_balanced = Simulation(size=(5.0e-6, 3.0e-6, 0.0), resolution=30e6)

# High accuracy but slow
sim_accurate = Simulation(size=(5.0e-6, 3.0e-6, 0.0), resolution=60e6)
```

**Complexity:**
- Computational cost ∝ (resolution)^(dimensionality) × time_steps
- Memory usage ∝ (resolution)^(dimensionality)

### Domain Size Optimization

Keep domain as small as possible:

```python
# Too large (wastes computation)
sim_large = Simulation(size=(10.0e-6, 10.0e-6, 0.0), ...)

# Optimized (just enough space)
sim_optimized = Simulation(size=(5.0e-6, 3.0e-6, 0.0), ...)
```

**Guidelines:**
- Include structures of interest + margin
- Account for PML layers (2 × pml_thickness)
- Leave space for field evolution

### Memory Estimation

```python
# Estimate memory usage
nx, ny, nz = sim.grid.dimensions
bytes_per_field = 8  # float64

# 6 field components (Ex, Ey, Ez, Hx, Hy, Hz)
memory_fields = nx * ny * nz * 6 * bytes_per_field

# Plus material arrays, monitor storage, etc.
print(f"Field storage: {memory_fields / 1e9:.2f} GB")
```

## Numerical Stability

### Courant Condition

The time step must satisfy:

```
c * dt * sqrt(1/dx² + 1/dy² + 1/dz²) < 1
```

Prismo automatically ensures this is satisfied.

### Checking Stability

```python
# Get Courant number (should be < 1)
courant = sim.grid.get_courant_number(sim.dt)
print(f"Courant number: {courant:.3f}")

if courant >= 1.0:
    print("⚠️ Warning: Unstable time step!")
```

### Diagnosing Instabilities

If you see NaN or exponentially growing fields:

1. **Check Courant condition**
2. **Reduce courant_factor**
3. **Check source amplitudes** (not too strong)
4. **Increase PML thickness**
5. **Verify grid resolution** (sufficient points per wavelength)

## Best Practices

### 1. Start with Low Resolution

Prototype with coarse grids, then refine:

```python
# Phase 1: Test setup
sim_test = Simulation(size=(5.0e-6, 3.0e-6, 0.0), resolution=15e6)
sim_test.run(50e-15)  # Quick test

# Phase 2: Production run
sim_prod = Simulation(size=(5.0e-6, 3.0e-6, 0.0), resolution=40e6)
sim_prod.run(200e-15)  # Full simulation
```

### 2. Monitor Convergence

Run with increasing resolution until results converge:

```python
resolutions = [20e6, 30e6, 40e6, 50e6]
results = []

for res in resolutions:
    sim = Simulation(size=(5.0e-6, 3.0e-6, 0.0), resolution=res)
    # ... add sources, monitors
    sim.run(100e-15)
    # ... collect results
    results.append(result)

# Check convergence
```

### 3. Save Simulation Parameters

```python
import json

params = {
    'size': list(sim.size),
    'resolution': sim.resolution,
    'boundary_conditions': sim.boundary_conditions,
    'dt': sim.dt,
    'courant_factor': sim.courant_factor,
}

with open('sim_params.json', 'w') as f:
    json.dump(params, f, indent=2)
```

## Troubleshooting

### Common Issues

**Problem**: Simulation is too slow
- **Solution**: Reduce resolution or domain size

**Problem**: Fields become NaN
- **Solution**: Check Courant condition, reduce courant_factor

**Problem**: Unexpected reflections
- **Solution**: Increase PML layers, move sources away from boundaries

**Problem**: Not enough detail in results
- **Solution**: Increase grid resolution

## Next Steps

- Learn about [Sources and Monitors](sources_monitors.md)
- Explore [Materials](materials.md) for dielectric structures
- Check [Examples](../examples/index.md) for complete simulations
- Read about [Validation](validation.md) for accuracy checks
