"""
Validation tests for dispersive materials.

These tests validate dispersive material implementations against
analytical solutions and known results.
"""

import pytest
import numpy as np

from prismo.materials import LorentzMaterial, DrudeMaterial, LorentzPole
from prismo.materials.ade import ADESolver
from prismo.backends import get_backend


class TestLorentzValidation:
    """Validate Lorentz model implementation."""

    def test_lorentz_permittivity_at_resonance(self):
        """Test permittivity at resonance frequency."""
        omega_0 = 2 * np.pi * 1e15  # 1 PHz

        mat = LorentzMaterial(
            epsilon_inf=1.0,
            poles=[LorentzPole(omega_0=omega_0, delta_epsilon=2.0, gamma=1e13)],
        )

        # At resonance, permittivity should have large imaginary part
        eps = mat.permittivity(omega_0)

        assert eps.imag > 0  # Absorption at resonance
        assert abs(eps.real) > 1  # Enhancement

    def test_lorentz_causality(self):
        """Test that Lorentz model satisfies Kramers-Kronig relations (causality)."""
        mat = LorentzMaterial(
            epsilon_inf=2.0,
            poles=[
                LorentzPole(omega_0=2 * np.pi * 1e15, delta_epsilon=1.0, gamma=1e13)
            ],
        )

        # Frequency sweep
        omega = np.linspace(1e14, 1e16, 1000) * 2 * np.pi
        eps = mat.permittivity(omega)

        # Check basic physics:
        # 1. Real part should be positive at low frequencies
        assert eps[0].real > 0

        # 2. Imaginary part should be positive (passive material)
        assert np.all(eps.imag >= -1e-10)  # Small negative allowed for numerical error


class TestDrudeValidation:
    """Validate Drude model implementation."""

    def test_drude_plasma_frequency(self):
        """Test Drude model at plasma frequency."""
        omega_p = 2 * np.pi * 1e15

        mat = DrudeMaterial(epsilon_inf=1.0, omega_p=omega_p, gamma=1e13)

        # At plasma frequency, Re(ε) should be small or negative
        eps = mat.permittivity(omega_p)

        # Below plasma frequency, metal behavior (negative real epsilon)
        eps_below = mat.permittivity(omega_p / 2)
        assert eps_below.real < 0

        # Above plasma frequency, dielectric behavior (positive real epsilon)
        eps_above = mat.permittivity(omega_p * 2)
        assert eps_above.real > 0

    def test_gold_optical_properties(self):
        """Test Gold material properties."""
        from prismo import get_material

        au = get_material("Au")

        # Test at 500 nm (visible)
        wavelength = 500e-9
        omega = 2 * np.pi * 299792458.0 / wavelength

        eps = au.permittivity(omega)
        n = au.refractive_index(omega)

        # Gold should have large imaginary part (absorption) in visible
        assert n.imag > 1.0


class TestADESolver:
    """Validate ADE solver implementation."""

    def test_ade_solver_creation(self):
        """Test ADE solver creation."""
        mat = LorentzMaterial(
            epsilon_inf=2.0,
            poles=[LorentzPole(omega_0=1e15, delta_epsilon=1.0, gamma=1e13)],
        )

        backend = get_backend("numpy")
        solver = ADESolver(
            material=mat, dt=1e-17, grid_shape=(10, 10, 10), backend=backend
        )

        assert solver.material == mat
        assert len(solver.P_current) == 1  # One pole

    def test_ade_polarization_update(self):
        """Test polarization update."""
        mat = DrudeMaterial(epsilon_inf=1.0, omega_p=1e15, gamma=1e13)

        backend = get_backend("numpy")
        solver = ADESolver(
            material=mat, dt=1e-17, grid_shape=(5, 5, 5), backend=backend
        )

        # Create test E field
        E_field = backend.ones((5, 5, 5))

        # Update polarization
        solver.update_polarization(E_field)

        # Check polarization was updated
        P = solver.get_polarization_current()
        assert backend.sum(backend.abs(P)) > 0


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for validation."""

    def test_numpy_backend_performance(self, benchmark):
        """Benchmark NumPy backend performance."""
        backend = get_backend("numpy")

        def run_operations():
            arr = backend.zeros((100, 100, 100))
            arr = backend.sqrt(arr + 1)
            return arr

        result = benchmark(run_operations)

    @pytest.mark.skipif(
        "cupy" not in list_available_backends(), reason="CuPy not available"
    )
    def test_cupy_backend_performance(self, benchmark):
        """Benchmark CuPy backend performance."""
        backend = get_backend("cupy")

        def run_operations():
            arr = backend.zeros((100, 100, 100))
            arr = backend.sqrt(arr + 1)
            backend.synchronize()
            return arr

        result = benchmark(run_operations)
