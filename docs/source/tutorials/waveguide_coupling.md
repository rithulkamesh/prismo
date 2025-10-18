# Tutorial 2: Waveguide Mode Coupling

**Time**: 45 minutes  
**Difficulty**: Intermediate  
**Prerequisites**: Tutorial 1, basic waveguide theory

## Learning Objectives

- Design a silicon waveguide structure
- Use ModeSolver to find guided modes
- Inject modes with ModeSource
- Analyze mode propagation and coupling

## The Problem

We'll simulate light coupling from a wide waveguide (3 μm) to a narrow waveguide (1 μm) through a taper.

## Step 1: Design the Waveguide Structure

```python
import numpy as np
from prismo import Simulation
from prismo.geometry import Box
from prismo.materials import Material

# Parameters
wavelength = 1.55e-6
width_input = 3e-6  # Wide waveguide
width_output = 1e-6  # Narrow waveguide
taper_length = 10e-6

# Create simulation
sim = Simulation(
    size=(20e-6, 10e-6, 0.0),
    resolution=40e6,  # 40 points per micron
    boundary_conditions='pml',
    pml_layers=10,
)

# Add waveguide core (Silicon, n=3.48)
si = Material(epsilon=3.48**2)

# Input waveguide
input_wg = Box(
    center=(-5e-6, 0.0, 0.0),
    size=(5e-6, width_input, 0.22e-6),
    material=si,
)

# Taper
taper = Taper(
    start=(-2.5e-6, 0.0, 0.0),
    end=(7.5e-6, 0.0, 0.0),
    width_start=width_input,
    width_end=width_output,
    height=0.22e-6,
    material=si,
)

# Output waveguide
output_wg = Box(
    center=(12.5e-6, 0.0, 0.0),
    size=(5e-6, width_output, 0.22e-6),
    material=si,
)

sim.add_structure(input_wg)
sim.add_structure(taper)
sim.add_structure(output_wg)

print("✓ Waveguide structure created")
```

## Step 2: Solve for Input Mode

```python
from prismo.modes.solver import ModeSolver

# Create cross-section for mode solving
ny = 200
y = np.linspace(-5e-6, 5e-6, ny)
z = np.array([0.0])  # 2D mode solver

# Build permittivity profile at input
Y, Z = np.meshgrid(y, z)
epsilon_input = np.ones((ny, 1)) * 1.45**2  # SiO2 cladding
core_mask = np.abs(Y[:, 0]) < width_input/2
epsilon_input[core_mask, 0] = 3.48**2  # Si core

# Solve for mode
mode_solver = ModeSolver(
    wavelength=wavelength,
    x=y,  # Transverse coordinate
    y=z,
    epsilon=epsilon_input,
)

modes = mode_solver.solve(num_modes=1, mode_type='TE')
fundamental_mode = modes[0]

print(f"✓ Fundamental mode: neff = {fundamental_mode.neff.real:.4f}")
```

## Step 3: Create Mode Source

```python
from prismo.sources.mode import ModeSource
from prismo.sources.waveform import GaussianPulse

# Waveform
pulse = GaussianPulse(
    frequency=299792458.0 / wavelength,
    width=20e-15,
)

# Mode source at input
mode_source = ModeSource(
    center=(-8e-6, 0.0, 0.0),  # Start of input waveguide
    size=(0.0, 8e-6, 0.0),  # Source plane
    mode=fundamental_mode,
    direction='+x',  # Propagate in +x
    waveform=pulse,
    amplitude=1.0,
)

sim.add_source(mode_source)

print("✓ Mode source added")
```

## Step 4: Add Monitors

```python
from prismo.monitors import FieldMonitor, FluxMonitor

# Field monitor to visualize propagation
field_monitor = FieldMonitor(
    center=(2.5e-6, 0.0, 0.0),
    size=(18e-6, 8e-6, 0.0),
    components=['Ey'],  # TE mode has Ey dominant
    time_domain=True,
)

# Flux monitor at output
output_flux = FluxMonitor(
    center=(15e-6, 0.0, 0.0),
    size=(0.0, 3e-6, 0.0),  # Vertical line
    direction='x',
    frequencies=[299792458.0 / wavelength],
)

sim.add_monitor(field_monitor)
sim.add_monitor(output_flux)

print("✓ Monitors added")
```

## Step 5: Run and Analyze

```python
# Run simulation
sim.run(100e-15)

# Get field distribution
times, Ey_data = field_monitor.get_time_data('Ey')

# Plot final field
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.imshow(
    Ey_data[-1, :, :].T,
    extent=[-8, 18, -4, 4],  # μm
    origin='lower',
    cmap='RdBu_r',
    aspect='auto',
)
plt.colorbar(label='Ey (V/m)')
plt.xlabel('x (μm)')
plt.ylabel('y (μm)')
plt.title('Mode Propagation Through Taper')

# Overlay waveguide structure
plt.axhline(width_input/2*1e6, x=0.0, color='k', linestyle='--', alpha=0.3)
plt.axhline(-width_input/2*1e6, x=0.0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('tutorial2_propagation.png', dpi=150)
plt.show()

# Calculate coupling efficiency
output_power = output_flux.get_flux()
input_power = 1.0  # Normalized

coupling_efficiency = output_power / input_power
coupling_db = 10 * np.log10(coupling_efficiency)

print(f"\nResults:")
print(f"  Coupling efficiency: {coupling_efficiency*100:.2f}%")
print(f"  Coupling loss: {-coupling_db:.2f} dB")
```

## Expected Results

For a well-designed taper:

- **Coupling efficiency**: > 95%
- **Loss**: < 0.2 dB
- **Field pattern**: Smooth transition from wide to narrow mode

## Exercises

1. Try different taper lengths (5 μm, 15 μm, 20 μm)
2. Simulate an abrupt transition (no taper) and compare loss
3. Add a second mode and observe higher-order mode excitation
4. Design a mode converter (different mode orders)

## Key Takeaways

- Mode sources enable realistic waveguide simulations
- ModeSolver provides accurate mode profiles
- Tapers reduce coupling loss between different waveguide widths
- Flux monitors measure power transmission

## Next Tutorial

Ready for {doc}`sparameters` - Learn to extract full S-parameter matrices!
