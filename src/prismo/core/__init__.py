"""
Core FDTD solver components.

This module contains the fundamental classes and algorithms for the
Finite-Difference Time-Domain electromagnetic solver.
"""

from .grid import YeeGrid, GridSpec
from .fields import ElectromagneticFields
from .solver import MaxwellUpdater, FDTDSolver
from .simulation import Simulation

# Core components will be implemented here:
# - Grid: Yee grid implementation ✓
# - Fields: Electromagnetic field storage and manipulation ✓
# - Solver: Main FDTD time-stepping engine ✓
# - Simulation: High-level simulation orchestration ✓

__all__ = [
    "YeeGrid",
    "GridSpec",
    "ElectromagneticFields",
    "MaxwellUpdater",
    "FDTDSolver",
    "Simulation",
]
