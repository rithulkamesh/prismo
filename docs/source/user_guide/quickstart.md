# Quick Start

This guide will get you running your first FDTD simulation in minutes.

## Your First Simulation

Let's create a simple 2D plane wave simulation using the TFSF (Total-Field/Scattered-Field) formulation:

```python
from prismo import Simulation, TFSFSource, FieldMonitor

# Create a 2D simulation domain
sim = Simulation(
    size=(2.0e-6, 2.0e-6, 0.0),     # 2µm × 2µm (note: in meters!)
    resolution=40e6,                 # 40 points per meter (40 ppµm)
    boundary_conditions="pml",       # Perfectly Matched Layer boundaries
    pml_layers=10,                   # 10 grid points for PML
)

# Add a plane wave source
source = TFSFSource(
    center=(1.0e-6, 1.0e-6, 0.0),   # Center of domain
    size=(1.0e-6, 1.0e-6, 0.0),     # TFSF region size
    direction="+x",                  # Propagate in +x direction
    polarization="y",                # E-field polarized in y
    frequency=150e12,                # 150 THz (2µm wavelength)
    pulse=False,                     # Continuous wave
    amplitude=1.0,
)
sim.add_source(source)

# Add a field monitor to record data
monitor = FieldMonitor(
    center=(1.0e-6, 1.0e-6, 0.0),
    size=(1.8e-6, 1.8e-6, 0.0),     # Monitor region
    components=["Ey"],               # Record Ey component
    time_domain=True,
)
sim.add_monitor(monitor)

# Run the simulation
sim_time = 50e-15  # 50 femtoseconds
sim.run(sim_time)

# Get and visualize results
time_points, ey_data = monitor.get_time_data("Ey")
print(f"Captured {len(time_points)} time steps")
print(f"Field shape: {ey_data.shape}")
```

## Understanding the Output

The simulation will:
1. Create a computational grid with PML absorbing boundaries
2. Inject a plane wave using TFSF formulation (artifact-free)
3. Record the Ey field component at every time step
4. Return time-domain data as NumPy arrays

The `ey_data` array has shape `(time_steps, ny, nx)` where:
- `time_steps`: Number of time steps recorded
- `ny, nx`: Spatial grid dimensions

## Basic Visualization

Add visualization to see the field evolution:

```python
import matplotlib.pyplot as plt
import numpy as np

# Plot the final field distribution
plt.figure(figsize=(10, 8))
vmax = np.max(np.abs(ey_data)) * 0.8

plt.imshow(
    ey_data[-1],  # Last time step
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax,
    origin="lower",
    extent=[0, 2.0, 0, 2.0],  # Physical dimensions in µm
)

plt.colorbar(label="Ey (V/m)")
plt.xlabel("x (µm)")
plt.ylabel("y (µm)")
plt.title("Electric Field at Final Time")
plt.tight_layout()
plt.show()
```

## Key Concepts

### Units
Prismo uses **SI units** throughout:
- **Length**: meters (m)
- **Time**: seconds (s)
- **Frequency**: Hertz (Hz)
- **Fields**: V/m (electric), A/m (magnetic)

💡 **Tip**: For convenience, use scientific notation:
- `1e-6` for micrometers (µm)
- `1e-15` for femtoseconds (fs)
- `1e12` for THz

### Grid Resolution
The `resolution` parameter defines spatial discretization:
- Higher resolution → better accuracy, longer computation
- Rule of thumb: Use at least **20 points per wavelength**
- For λ = 2µm at 40 ppµm: 80 points per wavelength ✓

### Boundary Conditions
- **PML (Perfectly Matched Layer)**: Absorbing boundaries (default)
- Prevents reflections from domain edges
- Typically use 8-12 PML layers

### Time Step
The time step `dt` is automatically calculated based on the Courant stability condition:

```python
print(f"Time step: {sim.dt:.3e} seconds")
```

## Common Patterns

### Adding Multiple Sources

```python
# Add multiple dipole sources
from prismo import ElectricDipole

dipole1 = ElectricDipole(
    position=(0.5e-6, 1.0e-6, 0.0),
    polarization="y",
    frequency=150e12,
    pulse=True,
    pulse_width=10e-15,
)
sim.add_source(dipole1)

dipole2 = ElectricDipole(
    position=(1.5e-6, 1.0e-6, 0.0),
    polarization="y",
    frequency=150e12,
    pulse=True,
    pulse_width=10e-15,
)
sim.add_source(dipole2)
```

### Frequency-Domain Monitoring

```python
monitor = FieldMonitor(
    center=(1.0e-6, 1.0e-6, 0.0),
    size=(1.8e-6, 1.8e-6, 0.0),
    components=["Ey", "Ez"],
    time_domain=True,
    frequencies=[150e12, 200e12],  # Monitor at specific frequencies
)
sim.add_monitor(monitor)

# After simulation
freq_field = monitor.get_frequency_data("Ey", 150e12)
```

### Progress Monitoring

```python
def progress_callback(step, total_steps, sim_time, elapsed_time):
    if step % 100 == 0:
        print(f"Progress: {step}/{total_steps} ({step/total_steps*100:.1f}%)")

sim.run(sim_time, progress_callback=progress_callback)
```

## Complete Example Scripts

Check out the example scripts in the `examples/` directory:

- **`tfsf_plane_wave.py`**: TFSF plane wave demonstration
- **`basic_waveguide.py`**: Gaussian beam in waveguide
- **`plane_wave_validation.py`**: Validation against analytical solutions

Run them with:
```bash
python examples/tfsf_plane_wave.py
```

## Next Steps

- Learn about [Sources and Monitors](sources_monitors.md) in detail
- Explore [Simulation Setup](simulations.md) for advanced options
- Check out [Examples](../examples/index.md) for more complex simulations
- Read about [Validation](validation.md) to verify your results

## Troubleshooting

### Simulation Takes Too Long
- Reduce grid resolution
- Decrease simulation time
- Use smaller computational domain

### Instabilities or NaN Values
- Check Courant condition (should be < 1.0)
- Ensure sources aren't too strong
- Verify PML boundaries are thick enough

### Memory Issues
- Reduce grid size or resolution
- Limit monitor regions
- Use selective component monitoring

For more help, see the [FAQ](faq.md) or [open an issue](https://github.com/rithulkamesh/prismo/issues).
