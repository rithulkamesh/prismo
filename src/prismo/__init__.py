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

# Import key classes for top-level access
from .core.simulation import Simulation
from .core.grid import YeeGrid, GridSpec
from .sources.base import Source
from .sources.point import ElectricDipole, MagneticDipole
from .sources.gaussian import GaussianBeamSource
from .sources.plane_wave import PlaneWaveSource
from .sources.tfsf import TFSFSource
from .monitors.field import FieldMonitor

# Version information
__version__ = "0.1.0-dev"
__author__ = "Rithul Kamesh"
__email__ = "rithul@example.com"

# Public API exports
__all__ = [
    # Modules
    "boundaries",
    "core",
    "geometry",
    "materials",
    "monitors",
    "solvers",
    "sources",
    "utils",
    "visualization",
    # Classes
    "Simulation",
    "YeeGrid",
    "GridSpec",
    "Source",
    "ElectricDipole",
    "MagneticDipole",
    "GaussianBeamSource",
    "PlaneWaveSource",
    "TFSFSource",
    "FieldMonitor",
]
