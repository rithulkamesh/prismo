"""
Boundary conditions for FDTD simulations.

This module provides various boundary condition implementations including
PML absorbing boundaries, periodic boundaries, and perfect conductors.
"""

from .pml import CPML, PMLParams

__all__ = [
    "CPML",
    "PMLParams",
]
