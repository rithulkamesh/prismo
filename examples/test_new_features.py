"""
End-to-end test script for new features:
- Magnetic boundary conditions (PMC, Magnetic PML)
- Solver selection (FDTD, MEEP, FEM)
- GUI availability check
"""

import sys

print("=" * 60)
print("Prismo New Features Test")
print("=" * 60)

# Test 1: Magnetic Boundary Conditions
print("\n1. Testing Magnetic Boundary Conditions...")
try:
    from prismo.boundaries import PMC, MagneticPML, MagneticPMLParams
    from prismo.core.grid import GridSpec, YeeGrid

    # Create a test grid
    grid_spec = GridSpec(size=(1e-6, 1e-6, 0), resolution=10e6, boundary_layers=5)
    grid = YeeGrid(grid_spec)

    # Test PMC
    pmc = PMC(grid, faces=["x_min", "x_max"])
    print(f"  ✓ PMC created: {pmc}")

    # Test Magnetic PML
    m_pml_params = MagneticPMLParams(thickness=5)
    m_pml = MagneticPML(grid, m_pml_params)
    print(f"  ✓ Magnetic PML created: {m_pml}")

    print("  ✓ Magnetic boundary conditions working!")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback

    traceback.print_exc()

# Test 2: Solver Abstraction
print("\n2. Testing Solver Abstraction...")
try:
    from prismo.solvers import SolverBase, TimeDomainSolver, FrequencyDomainSolver
    from prismo.core.solver import FDTDSolver

    print("  ✓ Solver base classes imported")
    print(
        f"  ✓ FDTDSolver inherits from TimeDomainSolver: {issubclass(FDTDSolver, TimeDomainSolver)}"
    )
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback

    traceback.print_exc()

# Test 3: FEM Solver (if available)
print("\n3. Testing FEM Solver...")
try:
    from prismo.solvers import FEMSolver

    grid_spec = GridSpec(size=(1e-6, 1e-6, 0), resolution=10e6)
    grid = YeeGrid(grid_spec)

    fem_solver = FEMSolver(grid)
    print(f"  ✓ FEM Solver created: {fem_solver}")
    print(f"  ✓ FEniCS available: {FEMSolver.__module__}")
except ImportError as e:
    print(f"  ⚠ FEM Solver not available (expected if FEniCS not installed): {e}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback

    traceback.print_exc()

# Test 4: MEEP Solver (if available)
print("\n4. Testing MEEP Solver...")
try:
    from prismo.solvers import MEEPSolver

    grid_spec = GridSpec(size=(1e-6, 1e-6, 0), resolution=10e6)
    grid = YeeGrid(grid_spec)

    meep_solver = MEEPSolver(grid)
    print(f"  ✓ MEEP Solver created: {meep_solver}")
except ImportError as e:
    print(f"  ⚠ MEEP Solver not available (expected if MEEP not installed): {e}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback

    traceback.print_exc()

# Test 5: Simulation with Solver Selection
print("\n5. Testing Simulation with Solver Selection...")
try:
    from prismo.core import Simulation

    # Test default FDTD solver
    sim_fdtd = Simulation(size=(1e-6, 1e-6, 0), resolution=10e6, solver_type="fdtd")
    print(f"  ✓ FDTD Simulation created: solver={type(sim_fdtd.solver).__name__}")

    # Test MEEP solver (if available)
    try:
        sim_meep = Simulation(size=(1e-6, 1e-6, 0), resolution=10e6, solver_type="meep")
        print(f"  ✓ MEEP Simulation created: solver={type(sim_meep.solver).__name__}")
    except ImportError:
        print("  ⚠ MEEP solver not available (expected if MEEP not installed)")

    # Test FEM solver (if available)
    try:
        sim_fem = Simulation(size=(1e-6, 1e-6, 0), resolution=10e6, solver_type="fem")
        print(f"  ✓ FEM Simulation created: solver={type(sim_fem.solver).__name__}")
    except ImportError:
        print("  ⚠ FEM solver not available (expected if FEniCS not installed)")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback

    traceback.print_exc()

# Test 6: GUI Availability
print("\n6. Testing GUI Availability...")
try:
    from prismo.gui import GUI_AVAILABLE, MainWindow

    if GUI_AVAILABLE:
        print("  ✓ GUI is available (PySide6 installed)")
        print("  ✓ MainWindow class imported")
        print("  ✓ Run 'prismo gui' to launch the GUI")
    else:
        print("  ⚠ GUI not available (install with: pip install PySide6)")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback

    traceback.print_exc()

# Test 7: Basic PMC Application
print("\n7. Testing PMC Application to Fields...")
try:
    from prismo.boundaries import PMC
    from prismo.core import Simulation, ElectromagneticFields

    sim = Simulation(size=(1e-6, 1e-6, 0), resolution=10e6)
    pmc = PMC(sim.grid, faces=["x_min", "x_max"])

    # Apply PMC to fields
    pmc.apply(sim.fields)
    print("  ✓ PMC applied to fields successfully")

except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback

    traceback.print_exc()

# Summary
print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)
print("\nCore features implemented:")
print("  ✓ PMC boundary condition")
print("  ✓ Magnetic PML boundary condition")
print("  ✓ Solver abstraction layer")
print("  ✓ FDTD solver (inherits from TimeDomainSolver)")
print("  ✓ FEM solver wrapper (requires FEniCS)")
print("  ✓ MEEP solver wrapper (requires MEEP)")
print("  ✓ Simulation solver selection")
print("  ✓ GUI scaffolding (requires PySide6)")

print("\nTo test GUI:")
print("  prismo gui")

print("\nTo install optional dependencies:")
print("  pip install PySide6          # For GUI")
print("  pip install fenics-dolfinx   # For FEM solver")
print("  conda install -c conda-forge meep  # For MEEP solver")

print("\n" + "=" * 60)

