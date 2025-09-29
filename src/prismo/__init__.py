"""
Prismo: A high-performance Python-based FDTD solver for waveguide photonics.

This package provides tools for electromagnetic simulation using the
Finite-Difference Time-Domain (FDTD) method, specifically designed
for photonic waveguide applications.
"""

from . import (
    boundaries,
    core,
    geometry,
    materials,
    monitors,
    solvers,
    sources,
    utils,
    visualization,
)

# Version information
__version__ = "0.1.0-dev"
__author__ = "Rithul Kamesh"
__email__ = "rithul@example.com"

# Public API exports
__all__ = [
    "boundaries",
    "core",
    "geometry",
    "materials",
    "monitors",
    "solvers",
    "sources",
    "utils",
    "visualization",
]
