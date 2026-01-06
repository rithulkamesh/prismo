# Solvers API

Prismo provides a unified solver interface that supports multiple electromagnetic simulation methods. All solvers implement the same base interface, allowing you to switch between different solvers transparently.

## Solver Interface

```{eval-rst}
.. automodule:: prismo.solvers.base
   :members:
   :undoc-members:
   :show-inheritance:
```

## Base Classes

### SolverBase

```{eval-rst}
.. autoclass:: prismo.solvers.base.SolverBase
   :members:
   :undoc-members:
```

### TimeDomainSolver

Base class for time-domain solvers (FDTD, MEEP). These solvers advance fields in time using time-stepping.

```{eval-rst}
.. autoclass:: prismo.solvers.base.TimeDomainSolver
   :members:
   :undoc-members:
```

### FrequencyDomainSolver

Base class for frequency-domain solvers (FEM). These solvers solve for fields at specific frequencies.

```{eval-rst}
.. autoclass:: prismo.solvers.base.FrequencyDomainSolver
   :members:
   :undoc-members:
```

## Solver Implementations

### FDTDSolver

Native FDTD (Finite-Difference Time-Domain) solver implementation. This is the default solver and provides high performance with GPU acceleration support.

```{eval-rst}
.. autoclass:: prismo.core.solver.FDTDSolver
   :members:
   :undoc-members:
```

**Example Usage:**

```python
from prismo.core import Simulation

# Create simulation with FDTD solver (default)
sim = Simulation(
    size=(10e-6, 5e-6, 0),
    resolution=50e6,
    solver_type="fdtd"  # This is the default
)

# Or create solver directly
from prismo.core.solver import FDTDSolver
from prismo.core.grid import YeeGrid, GridSpec

grid_spec = GridSpec(size=(10e-6, 5e-6, 0), resolution=50e6)
grid = YeeGrid(grid_spec)
solver = FDTDSolver(grid)
```

### MEEPSolver

Wrapper for MIT MEEP FDTD solver. This provides access to MEEP's advanced features while maintaining compatibility with Prismo's interface.

**Note**: MEEP is an optional dependency. Install with:
```bash
conda install -c conda-forge pymeeus meep
```

```{eval-rst}
.. autoclass:: prismo.solvers.meep_solver.MEEPSolver
   :members:
   :undoc-members:
```

**Example Usage:**

```python
from prismo.core import Simulation

# Create simulation with MEEP solver
sim = Simulation(
    size=(10e-6, 5e-6, 0),
    resolution=50e6,
    solver_type="meep"  # Use MEEP solver
)
```

**Note**: The MEEP solver wrapper is currently a placeholder implementation. Full integration with MEEP's geometry and material systems is planned for future releases.

### FEMSolver

Finite Element Method (FEM) solver using FEniCS/dolfinx for frequency-domain and eigenvalue electromagnetic problems.

**Note**: FEniCS is an optional dependency and is not available on PyPI. Install with:
```bash
conda install -c conda-forge fenics-dolfinx
```

Or install from source: https://fenicsproject.org/download/

```{eval-rst}
.. autoclass:: prismo.solvers.fem_solver.FEMSolver
   :members:
   :undoc-members:
```

**Example Usage:**

```python
from prismo.core import Simulation

# Create simulation with FEM solver
sim = Simulation(
    size=(10e-6, 5e-6, 0),
    resolution=50e6,
    solver_type="fem"  # Use FEM solver
)

# For frequency-domain problems
fields = sim.solver.solve(frequency=193e12)  # 1550 nm

# For eigenvalue problems (e.g., waveguide modes)
eigenvalues, eigenmodes = sim.solver.solve_eigenvalue(num_modes=3)
```

**Note**: The FEM solver is currently a placeholder implementation. Full integration with FEniCS mesh generation, weak form assembly, and boundary conditions is planned for future releases.

## Choosing a Solver

- **FDTD (default)**: Best for time-domain simulations, transient analysis, and broadband frequency sweeps. Supports GPU acceleration.
- **MEEP**: Useful when you need MEEP's specific features or want to compare results with MEEP simulations.
- **FEM**: Best for frequency-domain problems, eigenvalue analysis (waveguide modes), and problems requiring unstructured meshes.

## Solver Selection in Simulation

The solver type is specified when creating a `Simulation`:

```python
sim = Simulation(
    size=(10e-6, 5e-6, 0),
    resolution=50e6,
    solver_type="fdtd"  # Options: "fdtd", "meep", "fem"
)
```

All solvers provide the same interface, so you can switch between them without changing your simulation code (except for frequency-domain specific methods in FEM solver).

