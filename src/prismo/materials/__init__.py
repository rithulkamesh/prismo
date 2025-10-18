"""
Material models and material library for FDTD simulations.

This module provides material definitions, dispersion models, and a library
of commonly used optical and photonic materials.
"""

from .ade import ADEManager, ADESolver
from .dispersion import (
    DebyeMaterial,
    DebyePole,
    DispersiveMaterial,
    DrudeMaterial,
    DrudePole,
    LorentzMaterial,
    LorentzPole,
    SellmeierMaterial,
)
from .library import (
    MaterialLibrary,
    add_material,
    get_material,
    list_materials,
)
from .tensor import (
    AnisotropicUpdater,
    TensorComponents,
    TensorMaterial,
    create_biaxial_material,
    create_uniaxial_material,
)

__all__ = [
    # Dispersion models
    "DispersiveMaterial",
    "LorentzMaterial",
    "DrudeMaterial",
    "DebyeMaterial",
    "SellmeierMaterial",
    "LorentzPole",
    "DrudePole",
    "DebyePole",
    # Library functions
    "MaterialLibrary",
    "get_material",
    "list_materials",
    "add_material",
    # ADE solver
    "ADESolver",
    "ADEManager",
    # Tensor materials
    "TensorMaterial",
    "TensorComponents",
    "AnisotropicUpdater",
    "create_uniaxial_material",
    "create_biaxial_material",
]
