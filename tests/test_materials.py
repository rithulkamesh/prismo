"""
Tests for material models and dispersion.
"""

import pytest
import numpy as np

from prismo.materials import (
    LorentzMaterial,
    DrudeMaterial,
    DebyeMaterial,
    SellmeierMaterial,
    LorentzPole,
    get_material,
    list_materials,
)


class TestDispersionModels:
    """Test dispersion model implementations."""

    def test_lorentz_material(self):
        """Test Lorentz material."""
        mat = LorentzMaterial(
            epsilon_inf=2.0,
            poles=[
                LorentzPole(
                    omega_0=2 * np.pi * 1e15,
                    delta_epsilon=1.0,
                    gamma=1e13,
                )
            ],
            name="TestLorentz",
        )

        # Test permittivity calculation
        omega = 2 * np.pi * 1e15
        eps = mat.permittivity(omega)

        assert isinstance(eps, complex)
        assert eps.real > 0

    def test_drude_material(self):
        """Test Drude material."""
        mat = DrudeMaterial(
            epsilon_inf=1.0, omega_p=2 * np.pi * 1e15, gamma=1e13, name="TestDrude"
        )

        omega = 2 * np.pi * 500e12
        eps = mat.permittivity(omega)

        assert isinstance(eps, complex)

    def test_ade_coefficients(self):
        """Test ADE coefficient generation."""
        mat = LorentzMaterial(
            epsilon_inf=2.0,
            poles=[LorentzPole(omega_0=1e15, delta_epsilon=1.0, gamma=1e13)],
        )

        dt = 1e-17
        coeffs = mat.get_ade_coefficients(dt)

        assert "poles" in coeffs
        assert "epsilon_inf" in coeffs
        assert len(coeffs["poles"]) == 1


class TestMaterialLibrary:
    """Test material library functionality."""

    def test_list_materials(self):
        """Test listing materials."""
        materials = list_materials()
        assert isinstance(materials, list)
        assert len(materials) > 0
        assert "Si" in materials

    def test_get_material(self):
        """Test getting material from library."""
        si = get_material("Si")
        assert si is not None
        assert si.name == "Silicon"

    def test_silicon_refractive_index(self):
        """Test Silicon refractive index at 1550nm."""
        si = get_material("Si")

        # 1550 nm wavelength
        wavelength = 1.55e-6
        omega = 2 * np.pi * 299792458.0 / wavelength

        n = si.refractive_index(omega)

        # Silicon n ≈ 3.48 at 1550nm
        assert 3.4 < n.real < 3.6

    def test_gold_drude_model(self):
        """Test Gold Drude model."""
        au = get_material("Au")

        # Check it's a Drude material
        assert isinstance(au, DrudeMaterial)
        assert au.epsilon_inf > 0
        assert au.omega_p > 0
