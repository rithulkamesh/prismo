# Waveguide Mode Analysis

## Overview

Prismo includes a 2D eigenmode solver for analyzing guided modes in waveguide structures. This is essential for:

- Understanding waveguide properties
- Injecting specific modes
- Decomposing fields into modes
- Calculating mode-based S-parameters

## Basic Mode Solving

### Define Waveguide Structure

```python
import numpy as np
from prismo import ModeSolver

# Define cross-section coordinates
x = np.linspace(-2e-6, 2e-6, 100)  # 4 μm width
y = np.linspace(-1e-6, 1e-6, 80)    # 2 μm height

# Create permittivity profile
# Example: 500nm × 220nm Silicon waveguide on SiO2
epsilon = np.ones((len(x), len(y))) * 1.44**2  # SiO2 background

wg_width = 0.5e-6
wg_height = 0.22e-6

for i, xi in enumerate(x):
    for j, yj in enumerate(y):
        # Silicon core (centered, on bottom)
        if abs(xi) < wg_width/2 and -wg_height < yj < 0:
            epsilon[i, j] = 11.68  # Silicon at 1550nm
```

### Solve for Modes

```python
# Create mode solver
mode_solver = ModeSolver(
    wavelength=1.55e-6,  # Operating wavelength
    x=x,
    y=y,
    epsilon=epsilon
)

# Solve for TE modes
modes = mode_solver.solve(
    num_modes=3,
    mode_type='TE'
)

# Access fundamental mode
fundamental = mode_solver.get_mode(0)
```

### Mode Properties

```python
# Effective index
neff = fundamental.neff.real
print(f"Effective index: {neff:.4f}")

# Mode fields (Ex, Ey, Ez, Hx, Hy, Hz)
Ex = fundamental.Ex
Hy = fundamental.Hy

# Coordinates
x_coords = fundamental.x
y_coords = fundamental.y
```

## Mode Types

### TE Modes

Transverse Electric (Ez = 0, Hz dominant):

```python
te_modes = mode_solver.solve(num_modes=2, mode_type='TE')
```

### TM Modes

Transverse Magnetic (Hz = 0, Ez dominant):

```python
tm_modes = mode_solver.solve(num_modes=2, mode_type='TM')
```

### Vectorial Modes

Full vector modes (all components):

```python
vector_modes = mode_solver.solve(num_modes=2, mode_type='vector')
```

## Using Modes in Simulations

### Mode Sources

Inject a specific mode into the simulation:

```python
from prismo import ModeSource
from prismo.sources.waveform import GaussianPulse

# Create waveform
waveform = GaussianPulse(
    frequency=193e12,
    pulse_width=10e-15
)

# Create mode source
mode_source = ModeSource(
    center=(0, 0, 0),
    size=(0, 2e-6, 0),
    mode=fundamental,
    direction='+x',
    waveform=waveform,
    amplitude=1.0
)

sim.add_source(mode_source)
```

### Mode Expansion Monitors

Decompose simulation fields into mode coefficients:

```python
from prismo import ModeExpansionMonitor

# Create monitor
mode_monitor = ModeExpansionMonitor(
    center=(8e-6, 0, 0),
    size=(0, 2e-6, 0),
    modes=modes,  # Use calculated modes
    direction='x',
    frequencies=frequencies.tolist()
)

sim.add_monitor(mode_monitor)
sim.run(time=50e-15)

# Get mode coefficients
for i, mode in enumerate(modes):
    coeff = mode_monitor.get_mode_coefficient(i, domain='frequency')
    power = mode_monitor.get_mode_power(i)
    print(f"Mode {i}: Power = {power:.3f}")
```

## Mode-Based S-Parameters

Extract S-parameters using mode expansion:

```python
# Get mode coefficients at each port
forward, backward = mode_monitor.separate_forward_backward(
    mode_index=0,
    frequency_index=5
)

# Calculate S-parameters
s_analyzer = prismo.SParameterAnalyzer(num_ports=2, frequencies=frequencies)

s_analyzer.add_mode_data(
    port_index=1,
    excitation_port=0,
    mode_coefficients={'forward': forward_array, 'backward': backward_array}
)
```

## Tips

- **Resolution**: Use at least 20 points per wavelength in the material
- **Domain size**: Extend beyond waveguide to capture evanescent fields
- **Boundary conditions**: Mode solver assumes zero fields at boundaries
- **Convergence**: Increase grid resolution if modes look incorrect
