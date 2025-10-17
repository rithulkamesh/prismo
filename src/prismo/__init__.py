"""
Prismo: A high-performance Python-based FDTD solver for waveguide photonics.

This package provides tools for electromagnetic simulation using the
Finite-Difference Time-Domain (FDTD) method, specifically designed
for photonic waveguide applications.
"""

from . import (
    analysis,
    backends,
    boundaries,
    core,
    geometry,
    io,
    materials,
    modes,
    monitors,
    solvers,
    sources,
    utils,
    visualization,
)

# Import key classes for top-level access
from .core.simulation import Simulation
from .core.grid import YeeGrid, GridSpec
from .core.solver import FDTDSolver, MaxwellUpdater
from .core.fields import ElectromagneticFields

# Backends
from .backends import Backend, get_backend, set_backend, list_available_backends

# Sources
from .sources.base import Source
from .sources.point import ElectricDipole, MagneticDipole
from .sources.gaussian import GaussianBeamSource
from .sources.plane_wave import PlaneWaveSource
from .sources.tfsf import TFSFSource
from .sources.mode import ModeSource, ModeLauncher

# Monitors
from .monitors.field import FieldMonitor
from .monitors.dft import DFTMonitor
from .monitors.flux import FluxMonitor
from .monitors.mode_monitor import ModeExpansionMonitor

# Boundaries
from .boundaries.pml import CPML, PMLParams

# Materials
from .materials.dispersion import (
    DispersiveMaterial,
    LorentzMaterial,
    DrudeMaterial,
    DebyeMaterial,
    SellmeierMaterial,
    LorentzPole,
    DrudePole,
    DebyePole,
)
from .materials.library import get_material, list_materials, add_material
from .materials.ade import ADESolver, ADEManager
from .materials.tensor import (
    TensorMaterial,
    TensorComponents,
    AnisotropicUpdater,
    create_uniaxial_material,
    create_biaxial_material,
)

# Geometry
from .geometry import (
    Shape,
    Material,
    Box,
    Sphere,
    Cylinder,
    Polygon,
    CustomShape,
    GeometryGroup,
)

# Mode solver
from .modes import ModeSolver, WaveguideMode

# Analysis
from .analysis import SParameterAnalyzer, export_touchstone

# Data export
from .io import CSVExporter, ParquetExporter

# Lumerical import
from .io.lumerical import (
    FSPParser,
    FSPProject,
    LumericalMaterialDB,
    import_lumerical_material,
)

# Optimization
from .optimization import ParameterSweep, SweepParameter

# Version information
__version__ = "0.1.0-dev"
__author__ = "Rithul Kamesh"
__email__ = "rithul@example.com"

# Public API exports
__all__ = [
    # Modules
    "analysis",
    "backends",
    "boundaries",
    "core",
    "geometry",
    "io",
    "materials",
    "modes",
    "monitors",
    "solvers",
    "sources",
    "utils",
    "visualization",
    # Core classes
    "Simulation",
    "YeeGrid",
    "GridSpec",
    "FDTDSolver",
    "MaxwellUpdater",
    "ElectromagneticFields",
    # Backends
    "Backend",
    "get_backend",
    "set_backend",
    "list_available_backends",
    # Sources
    "Source",
    "ElectricDipole",
    "MagneticDipole",
    "GaussianBeamSource",
    "PlaneWaveSource",
    "TFSFSource",
    "ModeSource",
    "ModeLauncher",
    # Monitors
    "FieldMonitor",
    "DFTMonitor",
    "FluxMonitor",
    "ModeExpansionMonitor",
    # Boundaries
    "CPML",
    "PMLParams",
    # Materials
    "DispersiveMaterial",
    "LorentzMaterial",
    "DrudeMaterial",
    "DebyeMaterial",
    "SellmeierMaterial",
    "LorentzPole",
    "DrudePole",
    "DebyePole",
    "get_material",
    "list_materials",
    "add_material",
    "ADESolver",
    "ADEManager",
    "TensorMaterial",
    "TensorComponents",
    "AnisotropicUpdater",
    "create_uniaxial_material",
    "create_biaxial_material",
    # Geometry
    "Shape",
    "Material",
    "Box",
    "Sphere",
    "Cylinder",
    "Polygon",
    "CustomShape",
    "GeometryGroup",
    # Mode solver
    "ModeSolver",
    "WaveguideMode",
    # Analysis
    "SParameterAnalyzer",
    "export_touchstone",
    # Data export
    "CSVExporter",
    "ParquetExporter",
    # Lumerical import
    "FSPParser",
    "FSPProject",
    "LumericalMaterialDB",
    "import_lumerical_material",
    # Optimization
    "ParameterSweep",
    "SweepParameter",
]
