"""
Lumerical material database import.

This module provides functionality to import material definitions from
Lumerical's material database files.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import json
import xml.etree.ElementTree as ET

from prismo.materials.dispersion import (
    DispersiveMaterial,
    LorentzMaterial,
    DrudeMaterial,
    LorentzPole,
)


class LumericalMaterialDB:
    """
    Parser for Lumerical material database files.

    Lumerical stores material data in various formats including:
    - Text files with (n, k) data
    - XML files with dispersion model parameters
    - MDF (Material Data Format) files

    Parameters
    ----------
    db_path : Path
        Path to Lumerical material database directory or file.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.materials: Dict[str, Any] = {}

    def load_material(self, material_name: str) -> Optional[DispersiveMaterial]:
        """
        Load a material from the database.

        Parameters
        ----------
        material_name : str
            Name of material to load.

        Returns
        -------
        DispersiveMaterial or None
            Material object if found.
        """
        # Search for material file
        # Lumerical uses various extensions: .txt, .xml, .mdf

        for ext in [".txt", ".xml", ".mdf", ".json"]:
            material_file = self.db_path / f"{material_name}{ext}"
            if material_file.exists():
                return self._parse_material_file(material_file)

        return None

    def _parse_material_file(self, file_path: Path) -> Optional[DispersiveMaterial]:
        """Parse a material data file."""
        if file_path.suffix == ".txt":
            return self._parse_nk_data(file_path)
        elif file_path.suffix == ".xml":
            return self._parse_xml_material(file_path)
        elif file_path.suffix == ".json":
            return self._parse_json_material(file_path)
        else:
            return None

    def _parse_nk_data(self, file_path: Path) -> Optional[DispersiveMaterial]:
        """
        Parse (n, k) data file.

        Format: wavelength(nm)  n  k
        """
        try:
            # Read data
            data = np.loadtxt(file_path, skiprows=1)
            wavelength_nm = data[:, 0]
            n = data[:, 1]
            k = data[:, 2] if data.shape[1] > 2 else np.zeros_like(n)

            # Convert to permittivity
            epsilon = (n + 1j * k) ** 2

            # Fit to Lorentz model (simplified - should use proper fitting)
            # For now, use constant permittivity at a reference wavelength
            ref_idx = len(wavelength_nm) // 2  # Middle wavelength
            epsilon_static = epsilon[ref_idx].real

            # Create simple material (no dispersion for now)
            # Full implementation would fit to Lorentz poles
            material_name = file_path.stem

            return LorentzMaterial(
                epsilon_inf=epsilon_static,
                poles=[],  # Would fit poles from n-k data
                name=material_name,
            )

        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None

    def _parse_xml_material(self, file_path: Path) -> Optional[DispersiveMaterial]:
        """Parse XML material definition."""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Extract dispersion model parameters
            # Lumerical XML format varies - this is simplified

            model_type = (
                root.find(".//model").get("type")
                if root.find(".//model") is not None
                else None
            )

            if model_type == "Lorentz":
                return self._parse_lorentz_xml(root)
            elif model_type == "Drude":
                return self._parse_drude_xml(root)
            else:
                return None

        except Exception as e:
            print(f"Error parsing XML {file_path}: {e}")
            return None

    def _parse_lorentz_xml(self, root: ET.Element) -> Optional[LorentzMaterial]:
        """Parse Lorentz model from XML."""
        try:
            epsilon_inf = float(root.find(".//epsilon_inf").text)

            poles = []
            for pole_elem in root.findall(".//pole"):
                pole = LorentzPole(
                    omega_0=float(pole_elem.find("omega_0").text),
                    delta_epsilon=float(pole_elem.find("delta_epsilon").text),
                    gamma=float(pole_elem.find("gamma").text),
                )
                poles.append(pole)

            return LorentzMaterial(
                epsilon_inf=epsilon_inf, poles=poles, name=root.get("name", "imported")
            )
        except:
            return None

    def _parse_drude_xml(self, root: ET.Element) -> Optional[DrudeMaterial]:
        """Parse Drude model from XML."""
        try:
            epsilon_inf = float(root.find(".//epsilon_inf").text)
            omega_p = float(root.find(".//omega_p").text)
            gamma = float(root.find(".//gamma").text)

            return DrudeMaterial(
                epsilon_inf=epsilon_inf,
                omega_p=omega_p,
                gamma=gamma,
                name=root.get("name", "imported"),
            )
        except:
            return None

    def _parse_json_material(self, file_path: Path) -> Optional[DispersiveMaterial]:
        """Parse JSON material definition."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            model_type = data.get("type", "")

            if model_type == "Lorentz":
                poles = [
                    LorentzPole(**pole_data) for pole_data in data.get("poles", [])
                ]
                return LorentzMaterial(
                    epsilon_inf=data["epsilon_inf"],
                    poles=poles,
                    name=data.get("name", "imported"),
                )
            elif model_type == "Drude":
                return DrudeMaterial(
                    epsilon_inf=data["epsilon_inf"],
                    omega_p=data["omega_p"],
                    gamma=data["gamma"],
                    name=data.get("name", "imported"),
                )

        except Exception as e:
            print(f"Error parsing JSON {file_path}: {e}")

        return None


def import_lumerical_material(
    file_path: Path, material_name: Optional[str] = None
) -> Optional[DispersiveMaterial]:
    """
    Import a single material from a Lumerical file.

    Parameters
    ----------
    file_path : Path
        Path to material data file.
    material_name : str, optional
        Material name to use. If None, uses filename.

    Returns
    -------
    DispersiveMaterial or None
        Imported material.

    Examples
    --------
    >>> mat = import_lumerical_material('path/to/Silicon.txt')
    >>> prismo.add_material('Silicon_imported', mat)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Material file not found: {file_path}")

    # Use database parser
    db = LumericalMaterialDB(file_path.parent)
    material = db._parse_material_file(file_path)

    if material and material_name:
        material.name = material_name

    return material


def convert_lumerical_to_prismo_units(lumerical_value: float, quantity: str) -> float:
    """
    Convert Lumerical units to Prismo (SI) units.

    Parameters
    ----------
    lumerical_value : float
        Value in Lumerical units.
    quantity : str
        Physical quantity ('length', 'frequency', 'power', etc.).

    Returns
    -------
    float
        Value in SI units.
    """
    # Lumerical often uses μm for length, THz for frequency
    conversion = {
        "length": 1e-6,  # μm → m
        "frequency": 1e12,  # THz → Hz
        "wavelength": 1e-9,  # nm → m
        "power": 1e-3,  # mW → W
    }

    return lumerical_value * conversion.get(quantity, 1.0)
