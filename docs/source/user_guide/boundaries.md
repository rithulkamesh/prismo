# Boundary Conditions

Boundary conditions define how electromagnetic fields behave at the edges of your simulation domain. Proper boundary conditions are crucial for accurate FDTD simulations.

## Overview

Prismo supports several types of boundary conditions:

1. **PML (Perfectly Matched Layer)** - Absorbing boundaries (most common)
2. **Periodic Boundaries** - For periodic structures
3. **Perfect Electric Conductor (PEC)** - Metallic boundaries
4. **Perfect Magnetic Conductor (PMC)** - Magnetic walls
5. **Mode Ports** - Waveguide mode injection/extraction

## PML (Perfectly Matched Layer)

PML boundaries absorb outgoing waves with minimal reflection. They're the default choice for most simulations.

### Basic Usage

```python
from prismo import Simulation

sim = Simulation(
    size=(10e-6, 10e-6, 0.0),
    resolution=40e6,
    boundary_conditions='pml',  # Use PML
    pml_layers=10,               # 10 grid points thick
)
```

### How PML Works

PML creates an artificial absorbing layer around your simulation domain:

```
┌────────────────────────────────┐
│ PML Layer (absorbing)          │
│  ┌──────────────────────────┐  │
│  │                          │  │
│  │  Simulation Domain       │  │
│  │  (physical region)       │  │
│  │                          │  │
│  └──────────────────────────┘  │
│                                │
└────────────────────────────────┘
```

### Advanced PML Configuration

Fine-tune PML performance:

```python
from prismo.boundaries import CPML, PMLParams

# Custom PML parameters
pml_params = PMLParams(
    layers=12,           # Number of PML layers
    sigma_max=1.5,       # Maximum conductivity
    kappa_max=15.0,      # Maximum kappa value
    alpha_max=0.05,      # Maximum alpha (for better absorption at grazing incidence)
    polynomial_order=3,  # Polynomial grading order
)

# Apply to simulation
sim = Simulation(
    size=(10e-6, 10e-6, 0.0),
    resolution=40e6,
    boundary_conditions='pml',
    pml_params=pml_params,
)
```

### PML Best Practices

**Thickness**:

- Minimum: 8 layers
- Recommended: 10-12 layers
- For high-angle reflections: 15-20 layers

**When PML works well**:

- Normal or near-normal incidence
- Broadband sources
- Open boundaries (radiation problems)

**When to use more layers**:

- Grazing incidence (waves nearly parallel to boundary)
- Very low reflectivity requirements (< -40 dB)
- Lossy materials near boundaries

### Verifying PML Performance

Check for reflections:

```python
# Add a field monitor near the boundary
edge_monitor = FieldMonitor(
    center=(sim.size[0]*0.45, 0.0, 0.0),  # Near edge
    size=(0.1e-6, sim.size[1], 0.0),
    components=['Ex', 'Ey', 'Ez'],
    time_domain=True,
)
sim.add_monitor(edge_monitor)

# After simulation, check for reflections
field_data = edge_monitor.get_time_data('Ex')
if np.max(np.abs(field_data[-100:])) > 0.01 * np.max(np.abs(field_data)):
    print("Warning: Significant reflections from PML")
```

## Periodic Boundaries

For structures that repeat in space (photonic crystals, gratings, etc.):

```python
sim = Simulation(
    size=(10e-6, 10e-6, 0.0),
    resolution=40e6,
    boundary_conditions={
        'x': 'periodic',  # Periodic in x
        'y': 'pml',       # PML in y
    },
)
```

### Bloch Boundaries

For angled incidence in periodic structures:

```python
# Periodic boundary with Bloch wave vector
kx = 2 * np.pi / (10e-6)  # Wave vector in x

sim = Simulation(
    size=(10e-6, 10e-6, 0.0),
    resolution=40e6,
    boundary_conditions={
        'x': ('bloch', kx),  # Bloch periodic
        'y': 'pml',
    },
)
```

### Requirements for Periodic Boundaries

1. **Symmetry**: Structure must actually be periodic
2. **Source placement**: Sources must respect periodicity
3. **Grid alignment**: Period must be integer number of grid cells

## Perfect Conductors

### PEC (Perfect Electric Conductor)

Metallic boundaries where tangential E = 0:

```python
sim = Simulation(
    size=(10e-6, 10e-6, 0.0),
    resolution=40e6,
    boundary_conditions={
        'x': 'pec',  # Metal walls in x
        'y': 'pec',  # Metal walls in y
    },
)
```

**Use cases**:

- Metallic cavities
- Waveguides with metal walls
- Symmetry planes for TE polarization

### PMC (Perfect Magnetic Conductor)

Magnetic walls where tangential H = 0:

```python
sim = Simulation(
    size=(10e-6, 10e-6, 0.0),
    resolution=40e6,
    boundary_conditions={
        'x': 'pmc',  # Magnetic walls
        'y': 'pml',
    },
)
```

**Use cases**:

- Symmetry planes for TM polarization
- Certain theoretical models

## Symmetry Boundaries

Exploit symmetry to reduce computational domain:

### Example: Symmetric Waveguide

```python
# Only simulate half the structure
sim = Simulation(
    size=(10e-6, 5e-6, 0.0),  # Half height
    resolution=40e6,
    boundary_conditions={
        'x': 'pml',
        'y_min': 'pec',  # Symmetry plane at y=0
        'y_max': 'pml',
    },
)
```

```
Full structure:        Simulated half:
┌──────────┐          ┌──────────┐
│          │          │          │
│   Core   │          │   Core   │
│          │          ├══════════┤ ← PEC symmetry plane
│   Core   │
│          │
└──────────┘
```

## Mixed Boundaries

Different boundaries on different sides:

```python
sim = Simulation(
    size=(10e-6, 10e-6, 5e-6),
    resolution=40e6,
    is_3d=True,
    boundary_conditions={
        'x_min': 'pml',
        'x_max': 'pml',
        'y_min': 'pml',
        'y_max': 'pml',
        'z_min': 'pec',      # Metal substrate
        'z_max': 'pml',      # Open top
    },
)
```

## Choosing the Right Boundary

| Scenario             | Recommended Boundary          |
| -------------------- | ----------------------------- |
| Open space radiation | PML                           |
| Waveguide ports      | Mode ports                    |
| Photonic crystal     | Periodic or Bloch             |
| Metal cavity         | PEC                           |
| Symmetric structure  | PEC/PMC (symmetry)            |
| Substrate simulation | PEC (bottom), PML (sides/top) |

## Common Issues and Solutions

### High Reflections from PML

**Problem**: Fields reflecting back into simulation domain

**Solutions**:

1. Increase PML layers (try 15-20)
2. Adjust PML parameters (increase `sigma_max`)
3. Ensure sources are not too close to PML
4. Check for evanescent waves reaching PML

```python
# Better PML for grazing incidence
pml_params = PMLParams(
    layers=20,
    sigma_max=2.0,
    alpha_max=0.1,  # Helps with grazing incidence
)
```

### Periodic Boundary Phase Errors

**Problem**: Incorrect results with periodic boundaries

**Solutions**:

1. Verify structure is truly periodic
2. Check grid alignment with period
3. Ensure source respects periodicity

```python
# Verify period is integer grid cells
period = 5e-6
dx = 1.0 / sim.resolution
cells_per_period = period / dx
assert abs(cells_per_period - round(cells_per_period)) < 1e-6, \
    "Period must be integer number of cells"
```

### PEC/PMC Polarization Mismatch

**Problem**: Using wrong symmetry boundary for polarization

**Solution**: Match boundary to polarization

For **TE polarization** (E tangential to symmetry plane):

- Use **PEC** symmetry

For **TM polarization** (H tangential to symmetry plane):

- Use **PMC** symmetry

## Performance Considerations

### Computational Cost

Boundary condition performance (fastest to slowest):

1. PEC/PMC (trivial, no extra cost)
2. Periodic (minimal extra cost)
3. PML (moderate, ~10-20% overhead)

### Memory Usage

- PEC/PMC: No extra memory
- Periodic: No extra memory
- PML: Requires extra field storage (~10-20% more memory)

## Validation Tests

### PML Reflection Test

```python
def test_pml_reflectivity(sim, source_amplitude=1.0):
    """Measure PML reflectivity."""
    # Place monitor at simulation center
    center_monitor = FieldMonitor(
        center=(0.0, 0.0, 0.0),
        size=(0.1e-6, 0.1e-6, 0.0),
        components=['Ex'],
        time_domain=True,
    )

    # Run simulation
    sim.run(100e-15)

    # Get field at late times (after reflections would arrive)
    Ex_data = center_monitor.get_time_data('Ex')[0]
    late_time_field = Ex_data[-100:]

    # Reflectivity
    reflectivity = np.max(np.abs(late_time_field)) / source_amplitude
    reflectivity_db = 20 * np.log10(reflectivity)

    print(f"PML reflectivity: {reflectivity_db:.1f} dB")

    return reflectivity_db
```

Good PML: < -40 dB
Acceptable: -30 to -40 dB
Poor: > -30 dB

## See Also

- {doc}`mode_ports` - Mode port boundaries
- {doc}`simulations` - Simulation setup
- {doc}`../api/boundaries` - Boundary API reference
- {doc}`validation` - How to validate boundary conditions
