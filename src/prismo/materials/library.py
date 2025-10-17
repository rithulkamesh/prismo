"""
Material library with predefined optical and photonic materials.

This module provides a library of commonly used materials for photonic
simulations, including their dispersion models.
"""

from typing import Dict, Optional
import numpy as np

from .dispersion import (
    DispersiveMaterial,
    LorentzMaterial,
    DrudeMaterial,
    SellmeierMaterial,
    LorentzPole,
)


class MaterialLibrary:
    """
    Library of predefined materials for photonic simulations.

    Provides access to commonly used materials with their dispersion models
    fitted from experimental data.
    """

    def __init__(self):
        self._materials: Dict[str, DispersiveMaterial] = {}
        self._register_default_materials()

    def _register_default_materials(self) -> None:
        """Register all default materials in the library."""
        # Silicon (Si) - Lorentz model around 1550nm
        self._materials["Si"] = self._create_silicon()
        self._materials["Silicon"] = self._materials["Si"]

        # Silicon Dioxide (SiO2) - Sellmeier model
        self._materials["SiO2"] = self._create_silica()
        self._materials["Silica"] = self._materials["SiO2"]

        # Silicon Nitride (Si3N4)
        self._materials["Si3N4"] = self._create_silicon_nitride()
        self._materials["SiN"] = self._materials["Si3N4"]

        # Gold (Au) - Drude model
        self._materials["Au"] = self._create_gold()
        self._materials["Gold"] = self._materials["Au"]

        # Silver (Ag) - Drude model
        self._materials["Ag"] = self._create_silver()
        self._materials["Silver"] = self._materials["Ag"]

        # Aluminum (Al) - Drude model
        self._materials["Al"] = self._create_aluminum()
        self._materials["Aluminum"] = self._materials["Al"]

        # Indium Tin Oxide (ITO)
        self._materials["ITO"] = self._create_ito()

    def _create_silicon(self) -> LorentzMaterial:
        """
        Create Silicon (Si) material model.

        Parameters fitted for telecommunication wavelengths (1200-1600nm).
        Reference: n ≈ 3.48 at 1550nm
        """
        # Simplified single-pole Lorentz model
        # For more accuracy, use multi-pole model or Sellmeier
        epsilon_inf = 11.68  # ε_∞

        # Single pole approximation near telecom band
        poles = [
            LorentzPole(
                omega_0=2 * np.pi * 3e8 / 1.2e-6,  # ~1.2 μm resonance
                delta_epsilon=1.0,
                gamma=1e13,  # Damping
            )
        ]

        return LorentzMaterial(epsilon_inf=epsilon_inf, poles=poles, name="Silicon")

    def _create_silica(self) -> SellmeierMaterial:
        """
        Create Silicon Dioxide (SiO2/Silica) material model.

        Uses Sellmeier equation with standard coefficients.
        Valid from 0.21 to 6.7 μm.
        """
        # Sellmeier coefficients for fused silica
        B_coeffs = [0.6961663, 0.4079426, 0.8974794]
        C_coeffs = [0.0684043**2, 0.1162414**2, 9.896161**2]  # in μm²

        return SellmeierMaterial(B_coeffs=B_coeffs, C_coeffs=C_coeffs, name="Silica")

    def _create_silicon_nitride(self) -> SellmeierMaterial:
        """
        Create Silicon Nitride (Si3N4) material model.

        Uses Sellmeier equation.
        Reference: n ≈ 2.0 at 1550nm
        """
        # Sellmeier coefficients for Si3N4
        B_coeffs = [3.0249, 40314.0]
        C_coeffs = [0.1353406**2, 1239.842**2]  # in μm²

        return SellmeierMaterial(
            B_coeffs=B_coeffs, C_coeffs=C_coeffs, name="Silicon Nitride"
        )

    def _create_gold(self) -> DrudeMaterial:
        """
        Create Gold (Au) material model.

        Drude model for gold in the visible to near-IR range.
        Parameters from Johnson and Christy (1972).
        """
        epsilon_inf = 9.84  # High-frequency permittivity

        # Plasma frequency: ~2.175 PHz (corresponds to ~138 nm)
        omega_p = 2 * np.pi * 2.175e15  # rad/s

        # Collision frequency: ~6.5 THz
        gamma = 2 * np.pi * 6.5e12  # rad/s

        return DrudeMaterial(
            epsilon_inf=epsilon_inf, omega_p=omega_p, gamma=gamma, name="Gold"
        )

    def _create_silver(self) -> DrudeMaterial:
        """
        Create Silver (Ag) material model.

        Drude model for silver in the visible to near-IR range.
        """
        epsilon_inf = 3.7
        omega_p = 2 * np.pi * 2.2e15  # rad/s (~136 nm)
        gamma = 2 * np.pi * 4.35e12  # rad/s

        return DrudeMaterial(
            epsilon_inf=epsilon_inf, omega_p=omega_p, gamma=gamma, name="Silver"
        )

    def _create_aluminum(self) -> DrudeMaterial:
        """
        Create Aluminum (Al) material model.

        Drude model for aluminum.
        """
        epsilon_inf = 1.0
        omega_p = 2 * np.pi * 3.55e15  # rad/s
        gamma = 2 * np.pi * 1.94e14  # rad/s

        return DrudeMaterial(
            epsilon_inf=epsilon_inf, omega_p=omega_p, gamma=gamma, name="Aluminum"
        )

    def _create_ito(self) -> DrudeMaterial:
        """
        Create Indium Tin Oxide (ITO) material model.

        Drude model for ITO (transparent conductive oxide).
        """
        epsilon_inf = 3.9
        omega_p = 2 * np.pi * 4.0e14  # rad/s (tunable with doping)
        gamma = 2 * np.pi * 2.0e13  # rad/s

        return DrudeMaterial(
            epsilon_inf=epsilon_inf, omega_p=omega_p, gamma=gamma, name="ITO"
        )

    def get(self, name: str) -> Optional[DispersiveMaterial]:
        """
        Get a material by name.

        Parameters
        ----------
        name : str
            Material name (case-insensitive).

        Returns
        -------
        DispersiveMaterial or None
            Material object if found, None otherwise.
        """
        # Try exact match first
        if name in self._materials:
            return self._materials[name]

        # Try case-insensitive match
        name_lower = name.lower()
        for key, material in self._materials.items():
            if key.lower() == name_lower:
                return material

        return None

    def add(self, name: str, material: DispersiveMaterial) -> None:
        """
        Add a custom material to the library.

        Parameters
        ----------
        name : str
            Material name.
        material : DispersiveMaterial
            Material object.
        """
        self._materials[name] = material

    def list_materials(self) -> list:
        """
        List all available materials.

        Returns
        -------
        list
            List of material names.
        """
        # Return unique material objects (not aliases)
        unique_materials = {}
        for name, mat in self._materials.items():
            mat_id = id(mat)
            if mat_id not in unique_materials:
                unique_materials[mat_id] = name

        return sorted(unique_materials.values())

    def __contains__(self, name: str) -> bool:
        """Check if material exists in library."""
        return self.get(name) is not None

    def __getitem__(self, name: str) -> DispersiveMaterial:
        """Get material using subscript notation."""
        material = self.get(name)
        if material is None:
            available = ", ".join(self.list_materials())
            raise KeyError(
                f"Material '{name}' not found in library. "
                f"Available materials: {available}"
            )
        return material


# Global material library instance
_library = MaterialLibrary()


def get_material(name: str) -> DispersiveMaterial:
    """
    Get a material from the global library.

    Parameters
    ----------
    name : str
        Material name.

    Returns
    -------
    DispersiveMaterial
        Material object.

    Examples
    --------
    >>> si = get_material('Si')
    >>> sio2 = get_material('SiO2')
    >>> au = get_material('Gold')
    """
    return _library[name]


def list_materials() -> list:
    """
    List all available materials in the global library.

    Returns
    -------
    list
        List of material names.
    """
    return _library.list_materials()


def add_material(name: str, material: DispersiveMaterial) -> None:
    """
    Add a custom material to the global library.

    Parameters
    ----------
    name : str
        Material name.
    material : DispersiveMaterial
        Material object.
    """
    _library.add(name, material)
