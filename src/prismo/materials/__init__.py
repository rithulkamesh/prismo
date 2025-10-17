"""
Material models and material library for FDTD simulations.

This module provides material definitions, dispersion models, and a library
of commonly used optical and photonic materials.
"""

from .dispersion import (
    DispersiveMaterial,
    LorentzMaterial,
    DrudeMaterial,
    DebyeMaterial,
    SellmeierMaterial,
    LorentzPole,
    DrudePole,
    DebyePole,
)
from .library import (
    MaterialLibrary,
    get_material,
    list_materials,
    add_material,
)
from .ade import ADESolver, ADEManager
from .tensor import (
    TensorMaterial,
    TensorComponents,
    AnisotropicUpdater,
    create_uniaxial_material,
    create_biaxial_material,
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
