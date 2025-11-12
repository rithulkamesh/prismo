"""
Solver implementations for electromagnetic simulations.

This module provides various solver implementations including FDTD, FEM, and MEEP.
"""

from .base import FrequencyDomainSolver, SolverBase, TimeDomainSolver

# Try to import FEM solver
try:
    from .fem_solver import FEMSolver

    __all__ = ["SolverBase", "TimeDomainSolver", "FrequencyDomainSolver", "FEMSolver"]
except ImportError:
    __all__ = ["SolverBase", "TimeDomainSolver", "FrequencyDomainSolver"]

# Try to import MEEP solver
try:
    from .meep_solver import MEEPSolver

    __all__.append("MEEPSolver")
except ImportError:
    pass
