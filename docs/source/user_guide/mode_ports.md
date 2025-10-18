# Mode Ports

Mode ports are powerful tools for analyzing waveguide devices. They allow you to inject specific waveguide modes and extract mode coefficients to calculate S-parameters.

## Overview

A mode port combines:

- **Mode injection**: Launch a specific waveguide mode into your simulation
- **Mode extraction**: Decompose simulation fields into mode amplitudes
- **S-parameters**: Calculate reflection and transmission coefficients

```{mermaid}
graph LR
    A[Mode Solver] --> B[Mode Port 1<br/>Input]
    B --> C[Waveguide Device]
    C --> D[Mode Port 2<br/>Output]
    D --> E[S-Parameters<br/>S11, S21, S22, S12]
```

## Basic Workflow

### 1. Solve for Waveguide Modes

First, use the `ModeSolver` to calculate the waveguide modes:

```python
from prismo.modes.solver import ModeSolver
import numpy as np

# Create waveguide cross-section
nx, ny = 100, 100
x = np.linspace(-2e-6, 2e-6, nx)
y = np.linspace(-2e-6, 2e-6, ny)

# Define permittivity (waveguide structure)
X, Y = np.meshgrid(x, y, indexing='ij')
epsilon = np.ones((nx, ny)) * 1.45**2  # SiO2 cladding

# Add waveguide core
core_width = 0.5e-6
core_mask = (np.abs(X) < core_width/2) & (np.abs(Y) < core_width/2)
epsilon[core_mask] = 3.48**2  # Silicon core

# Solve for modes
wavelength = 1.55e-6
solver = ModeSolver(wavelength, x, y, epsilon)
modes = solver.solve(num_modes=2, mode_type='TE')

print(f"Fundamental mode: neff = {modes[0].neff.real:.4f}")
```

### 2. Create Mode Source

Inject a mode into your simulation:

```python
from prismo.sources.mode import ModeSource
from prismo.sources.waveform import GaussianPulse

# Create waveform
waveform = GaussianPulse(
    frequency=193.5e12,  # 1.55 μm
    width=20e-15,        # 20 fs pulse
)

# Create mode source
mode_source = ModeSource(
    center=(0.0, 0.0, 1e-6),       # Position in simulation
    size=(4e-6, 4e-6, 0.0),        # Source plane size
    mode=modes[0],                  # Fundamental mode
    direction='+z',                 # Propagation direction
    waveform=waveform,
    amplitude=1.0,
)

sim.add_source(mode_source)
```

### 3. Add Mode Monitors

Extract mode coefficients:

```python
from prismo.monitors.mode_monitor import ModeExpansionMonitor

# Input monitor (for reflection S11)
input_monitor = ModeExpansionMonitor(
    center=(0.0, 0.0, 0.5e-6),
    size=(4e-6, 4e-6, 0.0),
    modes=modes,
    direction='z',
    frequencies=[193.5e12],  # Monitor at source frequency
    name='input_port'
)

# Output monitor (for transmission S21)
output_monitor = ModeExpansionMonitor(
    center=(0.0, 0.0, 10e-6),
    size=(4e-6, 4e-6, 0.0),
    modes=modes,
    direction='z',
    frequencies=[193.5e12],
    name='output_port'
)

sim.add_monitor(input_monitor)
sim.add_monitor(output_monitor)
```

### 4. Run Simulation and Extract S-Parameters

```python
# Run simulation
sim.run(total_time=100e-15)

# Get S-parameters
s11 = input_monitor.compute_s_parameters(source_mode_index=0)['S_11']
s21 = output_monitor.compute_s_parameters(source_mode_index=0)['S_11']

print(f"|S11| = {abs(s11[0]):.4f}  (reflection)")
print(f"|S21| = {abs(s21[0]):.4f}  (transmission)")
print(f"Loss = {-20*np.log10(abs(s21[0])):.2f} dB")
```

## Advanced Features

### Multi-Mode Ports

Handle multiple modes simultaneously:

```python
# Solve for more modes
modes = solver.solve(num_modes=4, mode_type='TE')

# Monitor all modes
monitor = ModeExpansionMonitor(
    center=(0.0, 0.0, 5e-6),
    size=(4e-6, 4e-6, 0.0),
    modes=modes,  # All 4 modes
    direction='z',
    frequencies=[193.5e12],
)

# Extract coefficients for each mode
for mode_idx in range(len(modes)):
    coeff = monitor.get_mode_coefficient(mode_idx)
    power = monitor.get_mode_power(mode_idx, frequency_index=0)
    print(f"Mode {mode_idx}: Power = {power:.6f}")
```

### Forward/Backward Separation

Separate forward and backward propagating modes:

```python
from prismo.utils.mode_matching import separate_forward_backward

# Use two monitors separated by known distance
distance = 5e-6

# Get coefficients at both monitors
coeff_left = monitor1.get_mode_coefficient(0)
coeff_right = monitor2.get_mode_coefficient(0)

# Separate directions
a_fwd, a_bwd = separate_forward_backward(
    coeff_left[-1],  # Final time step
    coeff_right[-1],
    modes[0].neff,
    distance,
    wavelength
)

print(f"Forward amplitude: {abs(a_fwd):.4f}")
print(f"Backward amplitude: {abs(a_bwd):.4f}")
```

### Mode Normalization

Normalize modes to specific power:

```python
from prismo.utils.mode_matching import normalize_mode_to_power

# Normalize mode to 1 mW
mode_normalized = normalize_mode_to_power(
    modes[0],
    target_power=1e-3,  # 1 mW
    direction='z',
    dx=x[1] - x[0],
    dy=y[1] - y[0],
)

# Use normalized mode in source
mode_source = ModeSource(
    center=(0.0, 0.0, 1e-6),
    size=(4e-6, 4e-6, 0.0),
    mode=mode_normalized,
    direction='+z',
    waveform=waveform,
)
```

## Using ModePort Boundary Condition

The `ModePort` class provides an integrated boundary condition:

```python
from prismo.boundaries.mode_port import ModePort, ModePortConfig

# Create input port configuration
input_config = ModePortConfig(
    center=(0.0, 0.0, 0.5e-6),
    size=(4e-6, 4e-6, 0.0),
    direction='+z',
    modes=modes,
    inject=True,  # This port injects modes
)

input_port = ModePort(input_config, name='port1')

# Create output port (extraction only)
output_config = ModePortConfig(
    center=(0.0, 0.0, 10e-6),
    size=(4e-6, 4e-6, 0.0),
    direction='+z',
    modes=modes,
    inject=False,  # Only extract, don't inject
)

output_port = ModePort(output_config, name='port2')

# Initialize ports (after creating simulation grid)
input_port.initialize(sim.grid)
output_port.initialize(sim.grid)

# During simulation loop, inject and extract
for time_step in range(num_steps):
    # Inject at input port
    input_port.inject_fields(fields, time, dt, mode_amplitudes=[1.0])

    # Extract at output port
    coeffs = output_port.extract_mode_coefficients(fields, time)
```

## Validation and Best Practices

### Check Mode Orthogonality

Ensure modes are properly orthogonal:

```python
from prismo.utils.mode_matching import check_mode_orthogonality

orthogonality = check_mode_orthogonality(
    modes[0], modes[1],
    direction='z',
    dx=x[1] - x[0],
    dy=y[1] - y[0],
)

print(f"Orthogonality: {orthogonality:.6f}")
# Should be close to 0 for orthogonal modes
```

### Resolution Requirements

- Use at least **30 points per wavelength** in the mode solver
- Mode source plane should cover the **entire mode profile** (include evanescent tails)
- Monitor planes should be placed in regions with **uniform waveguide cross-section**

### Common Pitfalls

1. **Mode mismatch**: Ensure mode solver grid matches simulation transverse grid
2. **Insufficient mode decay**: Place monitors where evanescent fields have decayed
3. **Reflections**: Place monitors away from discontinuities to avoid spurious reflections
4. **Frequency mismatch**: Mode effective index varies with frequency

## Complete Example

See the complete working example in `examples/mode_port_demo.py` which demonstrates:

- Waveguide design
- Mode solving
- Mode injection
- S-parameter extraction
- Visualization

## See Also

- {doc}`mode_analysis` - Mode solver details
- {doc}`sources_monitors` - General source and monitor concepts
- {doc}`../api/modes` - Mode solver API reference
- {doc}`../api/monitors` - Monitor API reference
- {doc}`../examples/index` - More examples
