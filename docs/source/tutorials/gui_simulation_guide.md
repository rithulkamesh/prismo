# GUI Simulation Guide: Creating Shapes and Running Simulations

**Time**: 45 minutes  
**Difficulty**: Intermediate  
**Prerequisites**: Basic Python, familiarity with electromagnetic simulations

## Learning Objectives

By the end of this tutorial, you will:

- ✓ Understand the Prismo GUI interface (Lumerical-style)
- ✓ Create a new simulation using the GUI
- ✓ Add geometric shapes (Box, Sphere, Cylinder, Polygon)
- ✓ Assign materials to shapes
- ✓ Add sources (PlaneWave, GaussianBeam, ModeSource)
- ✓ Add monitors (FieldMonitor, FluxMonitor)
- ✓ Use the 3D viewport with slice planes
- ✓ Run simulations and view results

## Overview

This guide will walk you through creating a complete simulation using Prismo's GUI. We'll build a waveguide structure with proper materials, sources, and monitors - similar to how you would use Lumerical FDTD Solutions.

The Prismo GUI provides:

- **3D Viewport**: Interactive 3D visualization of your simulation geometry
- **Slice Planes**: Cut through your geometry to inspect interior structures (XY, XZ, YZ planes)
- **Shape Management**: Create and modify geometric objects with materials
- **Simulation Control**: Run, stop, and reset simulations with progress monitoring

## Step 1: Launch the GUI

Start the Prismo GUI from the command line:

```bash
prismo gui
```

Or from Python:

```python
from prismo.gui import MainWindow

window = MainWindow()
window.show()
```

The main window will open with:

- **Left Panel**: 3D Viewport with slice plane controls
- **Right Panel**: Property plotter and analysis tools
- **Top Menu**: File operations, simulation control, view options

## Step 2: Create a New Simulation

When you first launch the GUI, a default simulation is created automatically. To create a new simulation:

1. Click **File → New Simulation**, or
2. Click the **New** button in the toolbar

This creates a fresh simulation with default parameters:
- Size: 10µm × 10µm × 1µm
- Resolution: 20 points per micrometer
- Boundary conditions: PML (Perfectly Matched Layer)

## Step 3: Understanding the 3D Viewport

The 3D viewport displays your simulation geometry. To open the interactive 3D window:

1. Click **"Open 3D View"** button in the viewport panel
2. A separate window will open with the 3D visualization

The viewport supports:

- **Mouse Interaction**: 
  - Left-click drag: Rotate view
  - Right-click drag: Pan view
  - Scroll wheel: Zoom in/out
- **Slice Planes**: Cut through geometry to see cross-sections
- **Reset Camera**: Return to default viewing angle

## Step 4: Using Slice Planes

Slice planes let you inspect the interior of your geometry, just like in Lumerical.

### Enable Slice Planes

In the "Slice Planes" section:

1. **XY Plane** (horizontal slice): Check the box to enable
2. Use the **Z Position** slider to move the slice up/down
3. **XZ Plane** (vertical along y): Enable and adjust Y Position
4. **YZ Plane** (vertical along x): Enable and adjust X Position

You can enable multiple slice planes simultaneously to see different cross-sections.

Example: To see the middle of a waveguide:

```python
# In the GUI, enable XY slice plane
# Set Z Position to 0.5e-6 (middle of 1µm thick structure)
```

## Step 5: Creating Geometric Shapes

### Adding Shapes Programmatically

While the GUI provides visualization, shapes are typically added via Python code. Here's how to create common shapes:

#### Box (Rectangular Structure)

```python
from prismo import Simulation, Box, Material

# Create simulation
sim = Simulation(
    size=(10.0e-6, 5.0e-6, 1.0e-6),
    resolution=20.0e6,
)

# Define material
silicon = Material(name="Si", epsilon_r=12.0)

# Create a waveguide box
waveguide = Box(
    material=silicon,
    center=(5.0e-6, 2.5e-6, 0.5e-6),  # Center position
    size=(8.0e-6, 0.5e-6, 0.22e-6),   # Length, width, height
)

# Add to simulation
sim.add_shape(waveguide)

# Update GUI viewport (if using GUI)
# viewport.sync_with_simulation(sim)
```

#### Sphere

```python
from prismo import Sphere

# Create a spherical structure
sphere = Sphere(
    material=silicon,
    center=(5.0e-6, 2.5e-6, 0.5e-6),
    radius=0.5e-6,
)

sim.add_shape(sphere)
```

#### Cylinder

```python
from prismo import Cylinder

# Create a cylindrical waveguide
cylinder = Cylinder(
    material=silicon,
    center=(5.0e-6, 2.5e-6, 0.5e-6),
    radius=0.25e-6,
    height=1.0e-6,
    axis="z",  # Orientation: "x", "y", or "z"
)

sim.add_shape(cylinder)
```

#### Polygon (Custom 2D Shape)

```python
from prismo import Polygon
import numpy as np

# Define polygon vertices (2D, will be extruded)
vertices = np.array([
    [1.0e-6, 1.0e-6],
    [3.0e-6, 1.0e-6],
    [3.0e-6, 2.0e-6],
    [2.0e-6, 3.0e-6],
    [1.0e-6, 2.0e-6],
])

polygon = Polygon(
    material=silicon,
    vertices=vertices,
    z_min=0.0,      # Bottom of extrusion
    z_max=0.22e-6,  # Top of extrusion
)

sim.add_shape(polygon)
```

#### Multiple Materials

You can add multiple shapes with different materials:

```python
from prismo import Material

# Define materials
silicon = Material(name="Si", epsilon_r=12.0)
oxide = Material(name="SiO2", epsilon_r=2.25)

# Substrate
substrate = Box(
    material=oxide,
    center=(5.0e-6, 2.5e-6, 0.0),
    size=(10.0e-6, 5.0e-6, 0.5e-6),
)

# Waveguide on top
waveguide = Box(
    material=silicon,
    center=(5.0e-6, 2.5e-6, 0.61e-6),
    size=(8.0e-6, 0.5e-6, 0.22e-6),
)

sim.add_shape(substrate)
sim.add_shape(waveguide)
```

## Step 6: Adding Sources

Sources excite electromagnetic fields in your simulation.

### Plane Wave Source

```python
from prismo import PlaneWaveSource

# Create a plane wave source
source = PlaneWaveSource(
    center=(1.0e-6, 2.5e-6, 0.5e-6),
    size=(0.0, 5.0e-6, 1.0e-6),  # Cross-sectional area
    direction="x",                # Propagation direction
    polarization="y",             # E-field polarization
    frequency=193.4e12,           # 1550 nm (193.4 THz)
    amplitude=1.0,
)

sim.add_source(source)
```

### Gaussian Beam Source

```python
from prismo import GaussianBeamSource

# Create a Gaussian beam
source = GaussianBeamSource(
    center=(1.0e-6, 2.5e-6, 0.5e-6),
    size=(0.0, 1.0e-6, 0.0),     # Line source
    direction="x",
    polarization="y",
    frequency=193.4e12,
    beam_waist=0.5e-6,           # Beam waist radius
    pulse=True,                  # Use Gaussian pulse
    pulse_width=10e-15,          # Pulse width (10 fs)
    amplitude=1.0,
)

sim.add_source(source)
```

### Mode Source (Waveguide Mode Launcher)

```python
from prismo import ModeSource

# Create a mode source that launches a waveguide mode
source = ModeSource(
    center=(1.0e-6, 2.5e-6, 0.5e-6),
    size=(0.0, 1.0e-6, 0.22e-6),  # Cross-section
    direction="x",
    mode_index=0,                 # Fundamental mode
    frequency=193.4e12,
    amplitude=1.0,
)

sim.add_source(source)
```

## Step 7: Adding Monitors

Monitors record field data during the simulation.

### Field Monitor

Record electric and magnetic fields:

```python
from prismo import FieldMonitor

# Create a field monitor
monitor = FieldMonitor(
    center=(5.0e-6, 2.5e-6, 0.5e-6),
    size=(8.0e-6, 3.0e-6, 0.22e-6),
    components=["Ey", "Hz"],      # Components to record
    time_domain=True,             # Record time-domain data
    frequencies=[193.4e12],       # Also do DFT at this frequency
    name="field_monitor",
)

sim.add_monitor(monitor)
```

**Monitor Types by Size:**

- **Point Monitor**: `size=(0, 0, 0)` - Single location
- **Line Monitor**: `size=(Lx, 0, 0)` - 1D profile
- **Plane Monitor**: `size=(Lx, Ly, 0)` - 2D slice
- **Volume Monitor**: `size=(Lx, Ly, Lz)` - Full 3D region

### Flux Monitor

Record power flow:

```python
from prismo import FluxMonitor

# Monitor power transmission
flux_monitor = FluxMonitor(
    center=(9.0e-6, 2.5e-6, 0.5e-6),
    size=(0.0, 1.0e-6, 0.22e-6),
    direction="x",                # Direction of power flow
    frequencies=[193.4e12],
    name="transmission",
)

sim.add_monitor(flux_monitor)
```

### Multiple Monitors

You can add multiple monitors to track different regions:

```python
# Input monitor
input_monitor = FieldMonitor(
    center=(1.0e-6, 2.5e-6, 0.5e-6),
    size=(0.0, 1.0e-6, 0.22e-6),
    components="all",
    frequencies=[193.4e12],
    name="input",
)

# Output monitor
output_monitor = FieldMonitor(
    center=(9.0e-6, 2.5e-6, 0.5e-6),
    size=(0.0, 1.0e-6, 0.22e-6),
    components="all",
    frequencies=[193.4e12],
    name="output",
)

sim.add_monitor(input_monitor)
sim.add_monitor(output_monitor)
```

## Step 8: Setting Simulation Parameters

Before running, configure simulation parameters:

### Simulation Size and Resolution

```python
sim = Simulation(
    size=(10.0e-6, 5.0e-6, 1.0e-6),  # Physical size (m)
    resolution=20.0e6,                # Points per meter
    boundary_conditions="pml",        # PML absorbing boundaries
    pml_layers=10,                    # Number of PML layers
    courant_factor=0.9,               # Time step safety factor
)
```

**Resolution Guidelines:**

- Higher resolution = more accurate but slower
- Typical: 20-50 points per wavelength
- For 1550 nm light: ~20-50 points per micrometer

### Boundary Conditions

- **PML**: Absorbing boundaries (recommended)
- **Periodic**: Periodic boundary conditions
- **Reflecting**: Perfect electric/magnetic conductors

## Step 9: Viewing Geometry in the GUI

After adding shapes, sources, and monitors:

1. **Sync Viewport**: The GUI automatically updates when you sync:

```python
# In your code, after adding shapes
viewport.sync_with_simulation(sim)
```

2. **Open 3D View**: Click "Open 3D View" to see:
   - **Shapes**: Colored by material (permittivity-based colors)
   - **Sources**: Green wireframe boxes or red spheres
   - **Monitors**: Yellow wireframe boxes

3. **Use Slice Planes**: 
   - Enable XY plane to see horizontal cross-section
   - Adjust Z position to slice through different heights
   - Enable multiple planes for different views

## Step 10: Running the Simulation

### From Code

```python
import time

# Define progress callback
def progress(step, total_steps, sim_time, elapsed_time):
    if step % 100 == 0:
        print(f"Step {step}/{total_steps} ({step/total_steps*100:.1f}%)")

# Run simulation
sim_time = 100e-15  # 100 femtoseconds
sim.run(sim_time, progress_callback=progress)

print(f"Simulation completed in {sim.current_time*1e15:.1f} fs")
```

### From GUI

1. Click **Simulation → Run** or the **Run** button
2. Progress will be shown in the status bar
3. Click **Stop** to interrupt the simulation

## Step 11: Viewing Results

After the simulation completes:

### Retrieve Field Data

```python
# Get time-domain field data
time_points, field_data = monitor.get_time_data("Ey")

# Get frequency-domain data
field_at_freq = monitor.get_frequency_data("Ey", frequency=193.4e12)

# Plot with matplotlib
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.imshow(field_data[-1], cmap="RdBu_r", origin="lower")
plt.colorbar(label="Ey (V/m)")
plt.xlabel("x (grid points)")
plt.ylabel("y (grid points)")
plt.title("Electric Field Ey at Final Time")
plt.show()
```

### Retrieve Flux Data

```python
# Get power transmission
transmission = flux_monitor.get_frequency_domain_power(frequency=193.4e12)
print(f"Transmission: {transmission:.4f}")
```

## Complete Example

Here's a complete example combining everything:

```python
from prismo import (
    Simulation, Box, Material, 
    GaussianBeamSource, FieldMonitor, FluxMonitor
)

# Create simulation
sim = Simulation(
    size=(10.0e-6, 5.0e-6, 1.0e-6),
    resolution=20.0e6,
)

# Materials
silicon = Material(name="Si", epsilon_r=12.0)
oxide = Material(name="SiO2", epsilon_r=2.25)

# Substrate
substrate = Box(
    material=oxide,
    center=(5.0e-6, 2.5e-6, 0.0),
    size=(10.0e-6, 5.0e-6, 0.5e-6),
)

# Waveguide
waveguide = Box(
    material=silicon,
    center=(5.0e-6, 2.5e-6, 0.61e-6),
    size=(8.0e-6, 0.5e-6, 0.22e-6),
)

sim.add_shape(substrate)
sim.add_shape(waveguide)

# Source
freq = 193.4e12  # 1550 nm
source = GaussianBeamSource(
    center=(1.0e-6, 2.5e-6, 0.5e-6),
    size=(0.0, 1.0e-6, 0.0),
    direction="x",
    polarization="y",
    frequency=freq,
    beam_waist=0.5e-6,
    pulse=True,
    pulse_width=10e-15,
)

sim.add_source(source)

# Monitors
field_monitor = FieldMonitor(
    center=(5.0e-6, 2.5e-6, 0.5e-6),
    size=(8.0e-6, 3.0e-6, 0.22e-6),
    components=["Ey"],
    time_domain=True,
    frequencies=[freq],
)

flux_monitor = FluxMonitor(
    center=(9.0e-6, 2.5e-6, 0.5e-6),
    size=(0.0, 1.0e-6, 0.22e-6),
    direction="x",
    frequencies=[freq],
)

sim.add_monitor(field_monitor)
sim.add_monitor(flux_monitor)

# Run simulation
sim.run(100e-15)

# View results
time_points, field_data = field_monitor.get_time_data("Ey")
print(f"Field monitor recorded {len(time_points)} time steps")
```

## Tips and Best Practices

1. **Start Simple**: Begin with a single shape and source to verify setup

2. **Use Appropriate Resolution**: 
   - Too low: Inaccurate results
   - Too high: Slow simulation
   - Rule of thumb: 20-30 points per wavelength

3. **Monitor Placement**: 
   - Place monitors away from sources to avoid artifacts
   - Use multiple monitors to track field evolution

4. **Slice Planes**: 
   - Use slice planes to verify geometry is correct
   - Check material boundaries before running

5. **Material Properties**: 
   - Verify epsilon_r values match your materials
   - Use library materials when possible: `prismo.get_material("Si")`

6. **Simulation Time**: 
   - Run long enough for fields to propagate and settle
   - Typical: 10-100 femtoseconds for photonic structures

## Troubleshooting

**Problem**: Viewport shows nothing
- **Solution**: Ensure you've added shapes and synced with simulation

**Problem**: Slice planes don't appear
- **Solution**: Make sure "Open 3D View" is clicked and slice plane checkbox is enabled

**Problem**: Simulation is very slow
- **Solution**: Reduce resolution or simulation size

**Problem**: Fields look incorrect
- **Solution**: Check material properties, source frequency, and monitor placement

## Next Steps

Now that you understand the basics:

- Explore advanced features in the [Advanced Guide](../user_guide/advanced.md)
- Learn about [Mode Analysis](../user_guide/mode_analysis.md)
- See [Optimization Techniques](../tutorials/optimization.md)
- Check out [S-Parameter Extraction](../tutorials/sparameters.md)

## Additional Resources

- [User Guide](../user_guide/index.md)
- [API Reference](../api/index.md)
- [Examples](../../examples/)
- [Documentation Home](../index.md)

