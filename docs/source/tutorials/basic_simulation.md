# Tutorial 1: Your First FDTD Simulation

**Time**: 30 minutes  
**Difficulty**: Beginner  
**Prerequisites**: Python, NumPy basics

## Learning Objectives

By the end of this tutorial, you will:

- ✓ Set up a complete 2D FDTD simulation
- ✓ Add a point source and monitors
- ✓ Run the simulation and extract data
- ✓ Visualize electromagnetic fields
- ✓ Create an animation of field evolution

## Overview

We'll simulate a simple dipole radiating in free space and observe the electromagnetic waves spreading outward.

```{image} ../_static/tutorial1_preview.png
:alt: Tutorial preview
:width: 600px
:align: center
```

## Step 1: Import and Setup

Let's start by importing the necessary modules:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import Prismo components
from prismo import Simulation
from prismo.sources import ElectricDipole
from prismo.sources.waveform import GaussianPulse
from prismo.monitors import FieldMonitor

print("✓ Imports successful")
```

## Step 2: Define Simulation Parameters

Define physical parameters for our simulation:

```python
# Physical parameters
wavelength = 1.55e-6  # 1.55 μm (telecom wavelength)
frequency = 299792458.0 / wavelength  # c / λ

# Simulation domain
domain_size = 10 * wavelength  # 10λ × 10λ domain
sim_time = 50e-15  # 50 femtoseconds

# Grid resolution
points_per_wavelength = 30  # Good balance of accuracy and speed
resolution = points_per_wavelength / wavelength

print(f"Wavelength: {wavelength*1e6:.2f} μm")
print(f"Frequency: {frequency/1e12:.2f} THz")
print(f"Domain size: {domain_size*1e6:.2f} μm × {domain_size*1e6:.2f} μm")
print(f"Resolution: {points_per_wavelength} points/λ")
```

**Why these values?**

- **Wavelength 1.55 μm**: Standard telecom wavelength, commonly used in photonics
- **30 points/wavelength**: Provides good accuracy without excessive computation
- **Domain size 10λ**: Large enough to see wave propagation

## Step 3: Create the Simulation

```python
sim = Simulation(
    size=(domain_size, domain_size, 0.0),  # 2D simulation (z=0)
    resolution=resolution,
    boundary_conditions='pml',  # Absorbing boundaries
    pml_layers=10,              # 10-layer PML
    is_3d=False,                # 2D simulation
)

print(f"✓ Simulation created")
print(f"  Grid size: {sim.grid.dimensions}")
print(f"  Time step: {sim.dt*1e18:.3f} attoseconds")
```

**Understanding the parameters**:

- `size=(Lx, Ly, Lz)`: Physical dimensions in meters
- `resolution`: Grid points per meter
- `boundary_conditions='pml'`: Perfectly Matched Layer absorbs outgoing waves
- `pml_layers=10`: Standard thickness for good absorption

## Step 4: Add a Source

Create a Gaussian pulse source:

```python
# Create Gaussian pulse waveform
pulse = GaussianPulse(
    frequency=frequency,
    width=10e-15,  # 10 fs pulse width
)

# Create electric dipole source at center
source = ElectricDipole(
    position=(domain_size/2, domain_size/2, 0.0),  # Center of domain
    polarization='z',  # Out-of-plane polarization
    waveform=pulse,
    amplitude=1.0,
)

sim.add_source(source)

print(f"✓ Source added at center")
```

**Source types**:

- `ElectricDipole`: Point source (what we're using)
- `GaussianBeam`: Focused beam
- `PlaneWave`: Plane wave (use TFSFSource)
- `ModeSource`: Waveguide mode

## Step 5: Add Monitors

Add monitors to record the electromagnetic fields:

```python
# Full-field monitor (records entire domain)
full_field_monitor = FieldMonitor(
    center=(domain_size/2, domain_size/2, 0.0),
    size=(domain_size*0.9, domain_size*0.9, 0.0),  # Slightly smaller than domain
    components=['Ez'],  # We only need Ez for visualization
    time_domain=True,
    name='full_field',
)

# Point monitor (records time series at one location)
point_monitor = FieldMonitor(
    center=(domain_size*0.75, domain_size/2, 0.0),  # To the right of source
    size=(0.0, 0.0, 0.0),  # Single point
    components=['Ez', 'Hx', 'Hy'],
    time_domain=True,
    name='point_probe',
)

sim.add_monitor(full_field_monitor)
sim.add_monitor(point_monitor)

print(f"✓ Monitors added")
print(f"  Full-field monitor: {full_field_monitor.name}")
print(f"  Point monitor: {point_monitor.name}")
```

**Why two monitors?**

- **Full-field**: See the entire wave pattern
- **Point monitor**: Analyze the wave at a specific location

## Step 6: Run the Simulation

```python
print("\nRunning simulation...")

# Progress callback
def progress_callback(step, total_steps, sim_time, elapsed_time):
    if step % 100 == 0:
        percent = (step / total_steps) * 100
        print(f"  Progress: {percent:.1f}% (step {step}/{total_steps})", end='\r')

# Run simulation
sim.run(
    total_time=sim_time,
    progress_callback=progress_callback,
)

print(f"\n✓ Simulation complete!")
```

**What's happening?**
During the simulation, Prismo:

1. Solves Maxwell's equations on the grid
2. Updates E and H fields at each time step
3. Applies boundary conditions (PML absorption)
4. Records data at monitors

## Step 7: Extract and Plot Results

### Plot 1: Snapshot of Field Distribution

```python
# Get full field data
times, Ez_full = full_field_monitor.get_time_data('Ez')

# Plot final snapshot
plt.figure(figsize=(10, 8))

# Get field at final time
Ez_final = Ez_full[-1, :, :]

# Plot
vmax = np.max(np.abs(Ez_final)) * 0.8
plt.imshow(
    Ez_final.T,
    extent=[0, domain_size*1e6, 0, domain_size*1e6],
    origin='lower',
    cmap='RdBu_r',
    vmin=-vmax,
    vmax=vmax,
)

plt.colorbar(label='Ez (V/m)')
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')
plt.title(f'Electric Field Ez at t = {times[-1]*1e15:.1f} fs')
plt.tight_layout()
plt.savefig('tutorial1_snapshot.png', dpi=150)
plt.show()

print("✓ Snapshot saved as 'tutorial1_snapshot.png'")
```

**What you should see:**

- Circular waves emanating from the center
- Wave pattern with ~1.55 μm spacing (one wavelength)
- Fields decaying toward edges (absorbed by PML)

### Plot 2: Time Series at Point

```python
# Get point monitor data
times_point, Ez_point = point_monitor.get_time_data('Ez')
_, Hx_point = point_monitor.get_time_data('Hx')

# Plot time evolution
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Ez vs time
axes[0].plot(times_point*1e15, Ez_point[:, 0, 0], 'b-', linewidth=1.5)
axes[0].set_ylabel('Ez (V/m)')
axes[0].set_title('Electric Field Time Series')
axes[0].grid(True, alpha=0.3)

# Hx vs time
axes[1].plot(times_point*1e15, Hx_point[:, 0, 0], 'r-', linewidth=1.5)
axes[1].set_xlabel('Time (fs)')
axes[1].set_ylabel('Hx (A/m)')
axes[1].set_title('Magnetic Field Time Series')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tutorial1_timeseries.png', dpi=150)
plt.show()

print("✓ Time series saved as 'tutorial1_timeseries.png'")
```

**What you should see:**

- Gaussian pulse arriving at the point
- Oscillations at the carrier frequency
- E and H fields 90° out of phase

### Plot 3: Animation of Field Evolution

```python
# Create animation
fig, ax = plt.subplots(figsize=(8, 8))

# Determine color scale
vmax = np.max(np.abs(Ez_full)) * 0.8

def update_frame(frame):
    ax.clear()

    im = ax.imshow(
        Ez_full[frame, :, :].T,
        extent=[0, domain_size*1e6, 0, domain_size*1e6],
        origin='lower',
        cmap='RdBu_r',
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xlabel('x (μm)')
    ax.set_ylabel('y (μm)')
    ax.set_title(f'Electric Field Ez at t = {times[frame]*1e15:.1f} fs')

    return [im]

# Create animation (every 5th frame for speed)
frames = range(0, len(times), 5)
anim = FuncAnimation(
    fig,
    update_frame,
    frames=frames,
    interval=50,  # ms between frames
    blit=True,
)

# Save animation
anim.save('tutorial1_animation.gif', writer='pillow', fps=20)
plt.close()

print("✓ Animation saved as 'tutorial1_animation.gif'")
```

## Step 8: Analysis and Validation

Let's verify our simulation makes physical sense:

```python
# Check Courant condition
c = 299792458.0
dx = 1.0 / resolution
dt = sim.dt
courant = c * dt * np.sqrt(1/dx**2 + 1/dx**2)

print(f"\nSimulation Validation:")
print(f"  Courant number: {courant:.4f} (should be < 1.0)")

# Check wavelength in simulation
# Find period from time series
from scipy.signal import find_peaks

peaks, _ = find_peaks(Ez_point[:, 0, 0])
if len(peaks) > 1:
    periods = np.diff(times_point[peaks])
    avg_period = np.mean(periods)
    measured_freq = 1.0 / avg_period

    print(f"  Expected frequency: {frequency/1e12:.3f} THz")
    print(f"  Measured frequency: {measured_freq/1e12:.3f} THz")
    print(f"  Error: {abs(measured_freq - frequency)/frequency*100:.2f}%")

# Check energy conservation (for lossless free space)
energy_density = 0.5 * 8.854187817e-12 * Ez_full**2  # ε₀|E|²/2
total_energy = np.sum(energy_density, axis=(1, 2))

print(f"  Energy variation: {(np.max(total_energy) - np.min(total_energy))/np.mean(total_energy)*100:.2f}%")
```

**Expected results:**

- Courant number < 1.0 (typically ~0.5-0.7)
- Frequency error < 1%
- Energy variation < 10% (some loss to PML is expected)

## Complete Code

Here's the complete tutorial code in one block:

```python
# Complete Tutorial 1 Code
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from prismo import Simulation
from prismo.sources import ElectricDipole
from prismo.sources.waveform import GaussianPulse
from prismo.monitors import FieldMonitor

# Parameters
wavelength = 1.55e-6
frequency = 299792458.0 / wavelength
domain_size = 10 * wavelength
sim_time = 50e-15
resolution = 30 / wavelength

# Create simulation
sim = Simulation(
    size=(domain_size, domain_size, 0.0),
    resolution=resolution,
    boundary_conditions='pml',
    pml_layers=10,
)

# Add source
pulse = GaussianPulse(frequency=frequency, width=10e-15)
source = ElectricDipole(
    position=(domain_size/2, domain_size/2, 0.0),
    polarization='z',
    waveform=pulse,
    amplitude=1.0,
)
sim.add_source(source)

# Add monitors
full_monitor = FieldMonitor(
    center=(domain_size/2, domain_size/2, 0.0),
    size=(domain_size*0.9, domain_size*0.9, 0.0),
    components=['Ez'],
    time_domain=True,
)
sim.add_monitor(full_monitor)

# Run
sim.run(sim_time)

# Visualize
times, Ez_data = full_monitor.get_time_data('Ez')
plt.figure(figsize=(10, 8))
vmax = np.max(np.abs(Ez_data[-1])) * 0.8
plt.imshow(Ez_data[-1].T, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
plt.colorbar(label='Ez (V/m)')
plt.title('Electromagnetic Wave from Dipole Source')
plt.tight_layout()
plt.savefig('tutorial1_result.png', dpi=150)
plt.show()

print("✓ Tutorial complete!")
```

## Exercises

Try these modifications to deepen your understanding:

1. **Change the wavelength** to 0.5 μm (visible light) and observe the difference
2. **Add a second source** at a different location and observe interference
3. **Try different polarizations** ('x', 'y', or 'z')
4. **Measure wave speed** by calculating distance/time for the wave to reach the point monitor
5. **Experiment with resolution** (try 15, 30, 60 points/wavelength) and compare accuracy

## Troubleshooting

**Simulation takes too long:**

- Reduce `domain_size` to 5λ
- Decrease `points_per_wavelength` to 20
- Reduce `sim_time` to 30e-15

**No visible fields:**

- Check that PML is not too thick (try 8 layers)
- Ensure monitor covers the source location
- Verify time is long enough for waves to appear

**Instabilities (NaN values):**

- This usually means Courant condition is violated
- Prismo automatically satisfies this, but check `sim.dt`

## Next Steps

Now that you've completed your first simulation, you're ready for:

- {doc}`waveguide_coupling` - Learn to simulate waveguides with mode sources
- {doc}`../user_guide/sources_monitors` - Explore all source and monitor types
- {doc}`../user_guide/visualization` - Advanced visualization techniques

Congratulations on completing Tutorial 1! 🎉
