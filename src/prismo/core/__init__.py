"""
Core FDTD solver components.

This module contains the fundamental classes and algorithms for the
Finite-Difference Time-Domain electromagnetic solver.
"""

from .fields import ElectromagneticFields
from .grid import GridSpec, YeeGrid
from .simulation import Simulation
from .solver import FDTDSolver, MaxwellUpdater

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
