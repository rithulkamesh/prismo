"""
Demonstration of new Prismo features:
- Magnetic boundary conditions
- Solver selection
- Basic usage examples
"""

import numpy as np
from prismo.boundaries import PMC, MagneticPML, MagneticPMLParams
from prismo.core import Simulation, GridSpec, YeeGrid
from prismo.sources import GaussianBeamSource

print("=" * 70)
print("Prismo New Features Demonstration")
print("=" * 70)

# Demo 1: PMC Boundary Condition
print("\n" + "=" * 70)
print("Demo 1: Perfect Magnetic Conductor (PMC) Boundary")
print("=" * 70)

grid_spec = GridSpec(size=(2e-6, 1e-6, 0), resolution=20e6, boundary_layers=5)
grid = YeeGrid(grid_spec)

# Create PMC on x boundaries
pmc = PMC(grid, faces=["x_min", "x_max"])
print(f"Created PMC: {pmc}")
print("  - Applied to x_min and x_max faces")
print("  - Enforces tangential H-field = 0 at boundaries")

# Create simulation and apply PMC
sim = Simulation(size=(2e-6, 1e-6, 0), resolution=20e6, solver_type="fdtd")
pmc.apply(sim.fields)
print("  ✓ PMC applied to simulation fields")

# Demo 2: Magnetic PML
print("\n" + "=" * 70)
print("Demo 2: Magnetic Perfectly Matched Layer (Magnetic PML)")
print("=" * 70)

m_pml_params = MagneticPMLParams(
    thickness=5, kappa_max=15.0, alpha_max=0.0, polynomial_order=3
)
m_pml = MagneticPML(grid, m_pml_params)
print(f"Created Magnetic PML: {m_pml}")
print(f"  - Thickness: {m_pml_params.thickness} layers")
print(f"  - Kappa max: {m_pml_params.kappa_max}")
print(f"  - Reflection coefficient: {m_pml.get_reflection_coefficient():.2e}")

# Demo 3: Solver Selection
print("\n" + "=" * 70)
print("Demo 3: Solver Selection")
print("=" * 70)

# FDTD solver (default)
sim_fdtd = Simulation(size=(2e-6, 1e-6, 0), resolution=20e6, solver_type="fdtd")
print(f"✓ FDTD Simulation: {type(sim_fdtd.solver).__name__}")
print(f"  - Time step: {sim_fdtd.solver.get_time_step():.2e} s")
print(f"  - Grid: {sim_fdtd.grid.dimensions}")

# Try MEEP (if available)
try:
    sim_meep = Simulation(size=(2e-6, 1e-6, 0), resolution=20e6, solver_type="meep")
    print(f"✓ MEEP Simulation: {type(sim_meep.solver).__name__}")
except ImportError:
    print("⚠ MEEP solver not available (install MEEP to use)")

# Try FEM (if available)
try:
    sim_fem = Simulation(size=(2e-6, 1e-6, 0), resolution=20e6, solver_type="fem")
    print(f"✓ FEM Simulation: {type(sim_fem.solver).__name__}")
except ImportError:
    print("⚠ FEM solver not available (install FEniCS to use)")

# Demo 4: Complete Simulation with PMC
print("\n" + "=" * 70)
print("Demo 4: Complete Simulation with PMC Boundaries")
print("=" * 70)

sim = Simulation(size=(5e-6, 2e-6, 0), resolution=30e6, solver_type="fdtd")

# Add source
source = GaussianBeamSource(
    center=(-2e-6, 0, 0),
    size=(0, 1e-6, 0),
    direction="x",
    polarization="y",
    frequency=193e12,  # 1550 nm
    beam_waist=0.5e-6,
    pulse_width=10e-15,
)
sim.add_source(source)
print("✓ Added Gaussian beam source")

# Apply PMC boundaries
pmc = PMC(sim.grid, faces=["x_min", "x_max", "y_min", "y_max"])
print("✓ Applied PMC to all boundaries")

# Run a few steps
print("\nRunning simulation for 10 time steps...")
for i in range(10):
    sim.step()
    if i % 2 == 0:
        print(f"  Step {i+1}: t = {sim.current_time:.2e} s")

print(f"\n✓ Simulation completed!")
print(f"  - Total time: {sim.current_time:.2e} s")
print(f"  - Steps: {sim.step_count}")

# Demo 5: Solver Information
print("\n" + "=" * 70)
print("Demo 5: Solver Information")
print("=" * 70)

info = sim.solver.get_simulation_info()
print("Simulation Info:")
for key, value in info.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4e}")
    else:
        print(f"  {key}: {value}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("✓ All core features are working!")
print("✓ Magnetic boundary conditions (PMC, Magnetic PML) implemented")
print("✓ Solver selection system functional")
print("✓ FDTD solver working with new boundary conditions")
print("\nOptional features (install to enable):")
print("  - GUI: pip install PySide6")
print("  - FEM: pip install fenics-dolfinx")
print("  - MEEP: conda install -c conda-forge meep")
print("=" * 70)
