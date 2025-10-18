# Architecture

This document describes Prismo's architecture, design patterns, and code organization.

## Overview

Prismo follows a modular architecture with clear separation of concerns:

```
┌────────────────────────────────────────────────┐
│              User Interface                     │
│         (Simulation, convenience APIs)          │
└────────────────────────────────────────────────┘
                      │
┌────────────────────────────────────────────────┐
│          Core FDTD Engine                       │
│    (Solver, Grid, Fields, Time-stepping)        │
└────────────────────────────────────────────────┘
                      │
┌─────────────┬───────────────┬──────────────────┐
│  Sources    │   Monitors    │    Boundaries    │
│  (Excite)   │   (Measure)   │    (Absorb/BC)   │
└─────────────┴───────────────┴──────────────────┘
                      │
┌─────────────┬───────────────┬──────────────────┐
│  Materials  │   Geometry    │     Backends     │
│  (ε, μ, σ)  │   (Shapes)    │   (CPU/GPU)      │
└─────────────┴───────────────┴──────────────────┘
```

## Directory Structure

```
prismo/
├── core/               # Core FDTD engine
│   ├── fields.py      # Field storage and access
│   ├── grid.py        # Yee grid implementation
│   ├── solver.py      # Main FDTD solver
│   └── simulation.py  # High-level simulation interface
├── sources/           # Field sources
│   ├── base.py        # Abstract source base class
│   ├── point.py       # Point sources
│   ├── plane_wave.py  # Plane wave sources
│   ├── mode.py        # Mode sources
│   └── waveform.py    # Temporal waveforms
├── monitors/          # Field monitors
│   ├── base.py        # Abstract monitor base class
│   ├── field.py       # Field monitors
│   ├── flux.py        # Flux/power monitors
│   └── mode_monitor.py # Mode expansion monitors
├── boundaries/        # Boundary conditions
│   ├── pml.py         # PML absorbing boundaries
│   └── mode_port.py   # Mode port boundaries
├── materials/         # Material models
│   ├── dispersion.py  # Dispersive materials
│   ├── library.py     # Material library
│   └── tensor.py      # Anisotropic materials
├── modes/             # Mode solver
│   └── solver.py      # Eigenmode solver
├── geometry/          # Geometric primitives
│   └── shapes.py      # Box, sphere, cylinder, etc.
├── backends/          # Computational backends
│   ├── numpy_backend.py  # NumPy (CPU)
│   └── cupy_backend.py   # CuPy (GPU)
├── io/                # Input/output
│   └── exporters/     # Data export formats
├── utils/             # Utilities
│   └── mode_matching.py # Mode overlap integrals
└── visualization/     # Plotting helpers
```

## Design Patterns

### 1. Backend Abstraction

All numerical operations go through a backend abstraction:

```python
from prismo.backends import Backend, get_backend

class Solver:
    def __init__(self, backend='numpy'):
        self.backend = get_backend(backend)

    def update_e_fields(self):
        # Backend-agnostic code
        curl_h = self.backend.curl_h(self.fields.Hx, self.fields.Hy, self.fields.Hz)
        self.fields.Ex += self.backend.multiply(self.dt, curl_h.x)
```

**Benefits:**

- Single codebase for CPU/GPU
- Easy to add new backends
- Performance portability

### 2. Component Pattern

Sources, monitors, and boundaries follow a consistent interface:

```python
class Component(ABC):
    def initialize(self, grid: YeeGrid) -> None:
        """Initialize component on grid."""
        pass

    @abstractmethod
    def update(self, fields: Fields, time: float, dt: float) -> None:
        """Update component at each time step."""
        pass
```

**Example:**

```python
class Source(Component):
    def update(self, fields, time, dt):
        # Add source contribution to fields
        pass

class Monitor(Component):
    def update(self, fields, time, dt):
        # Record field values
        pass
```

### 3. Yee Grid

The Yee grid staggers E and H fields in space and time:

```
E fields: (i, j+½, k+½)  Ex
          (i+½, j, k+½)  Ey
          (i+½, j+½, k)  Ez

H fields: (i+½, j, k)    Hx
          (i, j+½, k)    Hy
          (i, j, k+½)    Hz
```

Implementation:

```python
class YeeGrid:
    def __init__(self, dimensions, spacing):
        self.nx, self.ny, self.nz = dimensions
        self.dx, self.dy, self.dz = spacing

    def get_component_indices(self, component, region):
        """Get grid indices for a field component."""
        # Handle staggering offsets
        pass
```

### 4. Field Storage

Fields are stored contiguously for cache efficiency:

```python
class ElectromagneticFields:
    def __init__(self, grid, backend):
        self.Ex = backend.zeros((grid.nx, grid.ny, grid.nz))
        self.Ey = backend.zeros((grid.nx, grid.ny, grid.nz))
        self.Ez = backend.zeros((grid.nx, grid.ny, grid.nz))
        self.Hx = backend.zeros((grid.nx, grid.ny, grid.nz))
        self.Hy = backend.zeros((grid.nx, grid.ny, grid.nz))
        self.Hz = backend.zeros((grid.nx, grid.ny, grid.nz))
```

## Key Algorithms

### FDTD Update Equations

**E-field update:**

```
E^{n+1} = E^n + (Δt/ε) ∇ × H^{n+½}
```

**H-field update:**

```
H^{n+½} = H^{n-½} + (Δt/μ) ∇ × E^n
```

**Implementation:**

```python
def step(self):
    # Update H from E (curl operator)
    self.update_h_fields()

    # Apply H boundary conditions
    self.apply_h_boundaries()

    # Update E from H
    self.update_e_fields()

    # Apply E boundary conditions
    self.apply_e_boundaries()

    # Add sources
    self.apply_sources()

    # Record monitors
    self.update_monitors()
```

### PML Implementation

Convolutional PML (CPML) for absorbing boundaries:

```python
class CPML:
    def __init__(self, thickness, grid):
        # Compute PML conductivity profile
        self.sigma = self._compute_sigma_profile(thickness)

        # Auxiliary fields for convolution
        self.Psi_Ex = zeros_like(...)

    def update_e_fields(self, Ex, Hy, Hz):
        # Update auxiliary fields
        self.Psi_Ex = self.b * self.Psi_Ex + self.a * (dHz_dy - dHy_dz)

        # Update E with PML correction
        Ex += self.dt / self.epsilon * self.Psi_Ex
```

### Mode Solver

Eigenvalue problem for waveguide modes:

```python
def solve_modes(self):
    # Build operators
    A = self._build_curl_curl_operator()  # ∇ × ∇ ×
    M = self._build_material_operator()   # ε

    # Solve: (A + k₀²M) E = β² E
    eigenvalues, eigenvectors = sparse_eig(A + k0**2 * M)

    # Extract modes
    beta = sqrt(eigenvalues)
    neff = beta / k0

    return modes
```

## Performance Considerations

### Memory Layout

Fields are stored in C-contiguous order for cache efficiency:

```python
# Good: Contiguous access
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            Ex[i, j, k] = ...  # Fast

# Bad: Strided access
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            Ex[i, j, k] = ...  # Slower
```

### Vectorization

Use NumPy/CuPy vectorization:

```python
# Good: Vectorized
Ex[1:-1, 1:-1, 1:-1] += dt / eps * (
    (Hz[1:-1, 2:, 1:-1] - Hz[1:-1, :-2, 1:-1]) / (2*dy) -
    (Hy[1:-1, 1:-1, 2:] - Hy[1:-1, 1:-1, :-2]) / (2*dz)
)

# Bad: Explicit loops
for i in range(1, nx-1):
    for j in range(1, ny-1):
        for k in range(1, nz-1):
            Ex[i,j,k] += ...  # Much slower
```

### GPU Considerations

- **Batch operations**: Keep data on GPU, minimize transfers
- **Kernel fusion**: Combine operations to reduce kernel launches
- **Memory coalescing**: Access memory in aligned, contiguous patterns

## Extension Points

### Adding a New Source

1. Inherit from `Source` base class
2. Implement `update_fields()` method
3. Register in `sources/__init__.py`

```python
from prismo.sources.base import Source

class MySource(Source):
    def update_fields(self, fields, time, dt):
        # Compute source value
        value = self.compute_amplitude(time)

        # Add to fields
        fields.Ez[self.indices] += value
```

### Adding a New Material Model

1. Inherit from `Material`
2. Implement `update_polarization()` for dispersive materials
3. Add to material library

```python
from prismo.materials.dispersion import DispersiveMaterial

class MyMaterial(DispersiveMaterial):
    def update_polarization(self, E, P, time, dt):
        # Update polarization based on material model
        # E.g., Drude, Lorentz, Debye, etc.
        pass
```

## Testing Strategy

```python
# Unit tests: Test individual components
def test_gaussian_pulse():
    pulse = GaussianPulse(frequency=1e14, width=1e-15)
    assert pulse.value(0.0) == 1.0
    assert abs(pulse.value(5e-15)) < 0.01

# Integration tests: Test component interactions
def test_source_monitor_integration():
    sim = Simulation(...)
    sim.add_source(source)
    sim.add_monitor(monitor)
    sim.run(...)
    assert monitor.has_data()

# Validation tests: Compare to analytical/reference solutions
def test_plane_wave_propagation():
    sim = setup_plane_wave_sim()
    sim.run(...)
    error = compare_to_analytical()
    assert error < 0.01
```

## See Also

- {doc}`contributing` - How to contribute code
- {doc}`testing` - Testing guidelines
- {doc}`benchmarks` - Performance benchmarks
- {doc}`../api/index` - API reference
