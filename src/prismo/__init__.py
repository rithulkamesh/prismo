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

# Analysis
from .analysis import SParameterAnalyzer, export_touchstone

# Backends
from .backends import Backend, get_backend, list_available_backends, set_backend

# Boundaries
from .boundaries.magnetic_pml import MagneticPML, MagneticPMLParams
from .boundaries.pmc import PMC
from .boundaries.pml import CPML, PMLParams
from .core.fields import ElectromagneticFields
from .core.grid import GridSpec, YeeGrid

# Import key classes for top-level access
from .core.simulation import Simulation
from .core.solver import FDTDSolver, MaxwellUpdater

# Geometry
from .geometry import (
    Box,
    CustomShape,
    Cylinder,
    GeometryGroup,
    Material,
    Polygon,
    Shape,
    Sphere,
)

# Data export
from .io import CSVExporter, ParquetExporter

# Lumerical import
from .io.lumerical import (
    FSPParser,
    FSPProject,
    LumericalMaterialDB,
    import_lumerical_material,
)
from .materials.ade import ADEManager, ADESolver

# Materials
from .materials.dispersion import (
    DebyeMaterial,
    DebyePole,
    DispersiveMaterial,
    DrudeMaterial,
    DrudePole,
    LorentzMaterial,
    LorentzPole,
    SellmeierMaterial,
)
from .materials.library import add_material, get_material, list_materials
from .materials.tensor import (
    AnisotropicUpdater,
    TensorComponents,
    TensorMaterial,
    create_biaxial_material,
    create_uniaxial_material,
)

# Mode solver
from .modes import ModeSolver, WaveguideMode
from .monitors.dft import DFTMonitor

# Monitors
from .monitors.field import FieldMonitor
from .monitors.flux import FluxMonitor
from .monitors.mode_monitor import ModeExpansionMonitor

# Optimization
from .optimization import ParameterSweep, SweepParameter

# Sources
from .sources.base import Source
from .sources.gaussian import GaussianBeamSource
from .sources.mode import ModeLauncher, ModeSource
from .sources.plane_wave import PlaneWaveSource
from .sources.point import ElectricDipole, MagneticDipole
from .sources.tfsf import TFSFSource

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
    "PMC",
    "MagneticPML",
    "MagneticPMLParams",
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
