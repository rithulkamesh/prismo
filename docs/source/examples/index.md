# Examples

This page provides detailed explanations of the example scripts included with Prismo.

## Overview

The `examples/` directory contains complete, runnable demonstrations of Prismo's capabilities:

- **`tfsf_plane_wave.py`**: TFSF plane wave propagation
- **`basic_waveguide.py`**: Gaussian beam in a waveguide
- **`plane_wave_validation.py`**: Plane wave validation

All examples can be run directly:
```bash
cd examples
python tfsf_plane_wave.py
```

---

## TFSF Plane Wave Example

**File**: `examples/tfsf_plane_wave.py`

This example demonstrates the Total-Field/Scattered-Field (TFSF) formulation for clean plane wave injection.

### What It Does

1. Creates a 2D computational domain with PML boundaries
2. Injects a plane wave using TFSF (artifact-free)
3. Records the field evolution
4. Visualizes the field distribution and analyzes the data

### Key Code

```python
from prismo import Simulation, TFSFSource, FieldMonitor

# Create 2µm × 2µm simulation
sim = Simulation(
    size=(2.0e-6, 2.0e-6, 0.0),
    resolution=40e6,  # 40 points per meter
    boundary_conditions="pml",
    pml_layers=10,
)

# Add TFSF source
source = TFSFSource(
    center=(1.0e-6, 1.0e-6, 0.0),
    size=(1.0e-6, 1.0e-6, 0.0),  # TFSF region
    direction="+x",               # Propagate in +x
    polarization="y",             # Ey polarization
    frequency=150e12,             # 150 THz (2µm wavelength)
    pulse=False,                  # Continuous wave
)
sim.add_source(source)

# Run for 5 periods
periods = 5
sim_time = periods / source.frequency
sim.run(sim_time)
```

### Output

The example generates two visualizations:

1. **`tfsf_plane_wave.png`**: 2D field distribution showing:
   - Plane wave propagation
   - TFSF boundary (dashed line)
   - Total-field region (inside)
   - Scattered-field region (outside)

2. **`tfsf_analysis.png`**: Detailed analysis showing:
   - Spatial profile along propagation direction
   - Temporal evolution at center point

### Learning Points

- **TFSF formulation**: Clean plane wave without artifacts
- **Boundary visualization**: See the separation of total and scattered fields
- **Field analysis**: Spatial and temporal field behavior

### Modifications to Try

```python
# 1. Change wavelength
source.frequency = 100e12  # 3µm wavelength

# 2. Use pulsed excitation
source.pulse = True
source.pulse_width = 20e-15  # 20 fs pulse

# 3. Change propagation direction
source.direction = "+y"
source.polarization = "x"

# 4. Increase resolution for better accuracy
sim.resolution = 60e6  # 60 points per meter
```

---

## Basic Waveguide Example

**File**: `examples/basic_waveguide.py`

Demonstrates Gaussian beam propagation in a simple waveguide geometry.

### What It Does

1. Creates a 5µm × 3µm computational domain
2. Excites a Gaussian beam at the input
3. Records field evolution over time
4. Creates animated visualization of propagation

### Key Code

```python
from prismo import Simulation, GaussianBeamSource, FieldMonitor

# Create waveguide simulation
sim = Simulation(
    size=(5.0e-6, 3.0e-6, 0.0),
    resolution=20e6,
)

# Add Gaussian beam source
source = GaussianBeamSource(
    center=(1.0e-6, 1.5e-6, 0.0),
    size=(0.0, 1.0e-6, 0.0),     # Line source
    direction="x",
    polarization="y",
    frequency=193.4e12,           # 1550 nm
    beam_waist=0.5e-6,           # 500 nm waist
    pulse=True,
    pulse_width=10e-15,          # 10 fs
)
sim.add_source(source)

# Monitor large region
monitor = FieldMonitor(
    center=(2.5e-6, 1.5e-6, 0.0),
    size=(4.5e-6, 2.5e-6, 0.0),
    components=["Ey"],
    time_domain=True,
)
sim.add_monitor(monitor)

# Run for 100 fs
sim.run(100e-15)
```

### Output

Creates visualization showing:
- Final field distribution
- Gaussian beam profile
- Beam propagation and spreading

### Learning Points

- **Gaussian beam properties**: Beam waist, Rayleigh range
- **Pulsed vs. continuous excitation**
- **Field monitoring** over large regions

### Modifications to Try

```python
# 1. Tighter focusing
source.beam_waist = 0.3e-6  # 300 nm waist

# 2. Different wavelength
source.frequency = 200e12  # 1500 nm

# 3. Continuous wave instead of pulse
source.pulse = False

# 4. Multiple sources
source2 = GaussianBeamSource(
    center=(4.0e-6, 1.5e-6, 0.0),
    direction="-x",  # Counter-propagating
    ...
)
sim.add_source(source2)
```

---

## Plane Wave Validation Example

**File**: `examples/plane_wave_validation.py`

Validates basic FDTD implementation using simple plane wave propagation.

### What It Does

1. Creates a basic 2D simulation
2. Injections a continuous wave plane wave
3. Records field at multiple positions
4. Creates animation of propagation

### Key Code

```python
from prismo import Simulation, PlaneWaveSource, FieldMonitor

# Create simulation
sim = Simulation(
    size=(2.0e-6, 2.0e-6, 0.0),
    resolution=20e6,
)

# Basic plane wave source
source = PlaneWaveSource(
    center=(0.5e-6, 1.0e-6, 0.0),
    size=(1.5e-6, 0.0, 0.0),     # Line source
    direction="+y",
    polarization="z",             # Ez (TM mode)
    frequency=193.4e12,
    pulse=False,
)
sim.add_source(source)

# Run for 5 periods
period = 1 / source.frequency
sim.run(5 * period)
```

### Output

Animation showing plane wave propagation through the domain.

### Learning Points

- **Basic plane wave source** (compare to TFSF)
- **TM vs. TE polarization**
- **Periodic field patterns**

---

## Common Patterns

### Pattern 1: Parameter Sweep

```python
# Sweep over wavelengths
wavelengths = [1.3e-6, 1.55e-6, 2.0e-6]

for wavelength in wavelengths:
    freq = 3e8 / wavelength
    
    sim = Simulation(size=(5.0e-6, 3.0e-6, 0.0), resolution=30e6)
    source = TFSFSource(..., frequency=freq)
    sim.add_source(source)
    
    monitor = FieldMonitor(...)
    sim.add_monitor(monitor)
    
    sim.run(100e-15)
    
    # Save results
    time_points, data = monitor.get_time_data("Ey")
    np.save(f'results_{wavelength*1e9:.0f}nm.npy', data)
```

### Pattern 2: Convergence Study

```python
# Test different resolutions
resolutions = [20e6, 30e6, 40e6, 50e6]
results = []

for res in resolutions:
    sim = Simulation(size=(3.0e-6, 2.0e-6, 0.0), resolution=res)
    # ... setup and run
    
    # Extract key metric
    peak_field = np.max(np.abs(field_data))
    results.append(peak_field)

# Plot convergence
import matplotlib.pyplot as plt
plt.plot(resolutions, results, 'o-')
plt.xlabel('Resolution (points/m)')
plt.ylabel('Peak Field (V/m)')
plt.title('Convergence Study')
plt.show()
```

### Pattern 3: Animation Creation

```python
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# After simulation
time_points, field_data = monitor.get_time_data("Ey")

fig, ax = plt.subplots(figsize=(10, 8))
vmax = np.max(np.abs(field_data))

im = ax.imshow(field_data[0], cmap='RdBu_r', 
               vmin=-vmax, vmax=vmax,
               origin='lower')
plt.colorbar(im, label='Ey (V/m)')

def update(frame):
    im.set_data(field_data[frame])
    ax.set_title(f'Time: {time_points[frame]*1e15:.1f} fs')
    return [im]

anim = FuncAnimation(fig, update, frames=len(time_points),
                     interval=50, blit=True)

anim.save('animation.mp4', writer='ffmpeg', fps=20)
plt.show()
```

---

## Running Examples

### Basic Execution

```bash
cd examples
python tfsf_plane_wave.py
```

### With Custom Parameters

Modify the example files directly, or use command-line arguments (if implemented):

```python
import sys

if len(sys.argv) > 1:
    frequency = float(sys.argv[1])  # THz
else:
    frequency = 150e12

source = TFSFSource(..., frequency=frequency*1e12)
```

Then run:
```bash
python tfsf_plane_wave.py 200  # 200 Hz
```

### Batch Processing

Create a script to run multiple examples:

```bash
#!/bin/bash
for example in tfsf_plane_wave.py basic_waveguide.py plane_wave_validation.py
do
    echo "Running $example..."
    python $example
done
```

---

## Next Steps

- Modify the examples to explore different parameters
- Combine techniques from multiple examples
- Create your own simulations based on these templates
- Check the [User Guide](../user_guide/index.md) for detailed explanations
- See [API Reference](../api/index.md) for all available options
