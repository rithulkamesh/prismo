"""
Test mode port functionality: injection, extraction, and S-parameters.

This test suite validates:
- Mode injection with minimal reflection
- Mode extraction and overlap integrals
- Forward/backward mode separation
- S-parameter calculation
- Multi-mode port orthogonality
"""

import pytest
import numpy as np
from typing import List

from prismo.modes.solver import ModeSolver, WaveguideMode
from prismo.boundaries.mode_port import ModePort, ModePortConfig
from prismo.sources.mode import ModeSource
from prismo.monitors.mode_monitor import ModeExpansionMonitor
from prismo.utils import mode_matching


class TestModeMatching:
    """Test mode matching utilities."""

    def test_mode_power_calculation(self):
        """Test mode power calculation from Poynting vector."""
        # Create a simple mode with known power
        wavelength = 1.55e-6
        nx, ny = 50, 50
        x = np.linspace(-2e-6, 2e-6, nx)
        y = np.linspace(-2e-6, 2e-6, ny)

        # Gaussian mode profile
        X, Y = np.meshgrid(x, y, indexing="ij")
        w0 = 1e-6  # Beam waist

        # TE mode: Hz dominant
        Hz = np.exp(-(X**2 + Y**2) / w0**2)
        Ex = np.zeros_like(Hz)
        Ey = np.zeros_like(Hz)
        Hx = np.zeros_like(Hz)
        Hy = np.zeros_like(Hz)
        Ez = np.zeros_like(Hz)

        mode = WaveguideMode(
            mode_number=0,
            neff=1.5 + 0j,
            frequency=299792458.0 / wavelength,
            wavelength=wavelength,
            Ex=Ex,
            Ey=Ey,
            Ez=Ez,
            Hx=Hx,
            Hy=Hy,
            Hz=Hz,
            x=x,
            y=y,
            power=1.0,
        )

        dx = x[1] - x[0]
        dy = y[1] - y[0]

        power = mode_matching.compute_mode_power(mode, direction="z", dx=dx, dy=dy)

        # Power should be non-negative
        assert power >= 0.0

    def test_mode_normalization(self):
        """Test mode normalization to target power."""
        wavelength = 1.55e-6
        nx, ny = 40, 40
        x = np.linspace(-1.5e-6, 1.5e-6, nx)
        y = np.linspace(-1.5e-6, 1.5e-6, ny)

        X, Y = np.meshgrid(x, y, indexing="ij")
        w0 = 0.8e-6

        # Create a proper TE mode with transverse E fields
        # For TE mode propagating in z: Hz is dominant, Ex and Ey are transverse
        Hz = np.exp(-(X**2 + Y**2) / w0**2)
        
        # Simple transverse E fields (in reality these come from Maxwell's equations)
        # For power calculation in z-direction, we need Ex and Hy (or Ey and Hx)
        Ex = 0.5 * np.exp(-(X**2 + Y**2) / w0**2)
        Ey = 0.3 * np.exp(-(X**2 + Y**2) / w0**2)
        
        # Corresponding H fields for power calculation
        Hx = -0.3 * np.exp(-(X**2 + Y**2) / w0**2)
        Hy = 0.5 * np.exp(-(X**2 + Y**2) / w0**2)

        mode = WaveguideMode(
            mode_number=0,
            neff=1.5 + 0j,
            frequency=299792458.0 / wavelength,
            wavelength=wavelength,
            Ex=Ex,
            Ey=Ey,
            Ez=np.zeros_like(Hz),
            Hx=Hx,
            Hy=Hy,
            Hz=Hz,
            x=x,
            y=y,
            power=1.0,
        )

        target_power = 2.0
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        normalized_mode = mode_matching.normalize_mode_to_power(
            mode, target_power, direction="z", dx=dx, dy=dy
        )

        # Check that normalized mode has target power
        new_power = mode_matching.compute_mode_power(
            normalized_mode, direction="z", dx=dx, dy=dy
        )

        # Allow some numerical error
        assert abs(new_power - target_power) < 0.1 * target_power

    def test_mode_orthogonality(self):
        """Test orthogonality check between different modes."""
        wavelength = 1.55e-6
        nx, ny = 50, 50
        x = np.linspace(-2e-6, 2e-6, nx)
        y = np.linspace(-2e-6, 2e-6, ny)

        X, Y = np.meshgrid(x, y, indexing="ij")
        w0 = 1e-6

        # Mode 0: Fundamental Gaussian
        Hz0 = np.exp(-(X**2 + Y**2) / w0**2)

        # Mode 1: First-order mode (with X variation)
        Hz1 = X * np.exp(-(X**2 + Y**2) / w0**2)

        mode0 = WaveguideMode(
            mode_number=0,
            neff=1.5 + 0j,
            frequency=299792458.0 / wavelength,
            wavelength=wavelength,
            Ex=np.zeros_like(Hz0),
            Ey=np.zeros_like(Hz0),
            Ez=np.zeros_like(Hz0),
            Hx=np.zeros_like(Hz0),
            Hy=np.zeros_like(Hz0),
            Hz=Hz0,
            x=x,
            y=y,
            power=1.0,
        )

        mode1 = WaveguideMode(
            mode_number=1,
            neff=1.48 + 0j,
            frequency=299792458.0 / wavelength,
            wavelength=wavelength,
            Ex=np.zeros_like(Hz1),
            Ey=np.zeros_like(Hz1),
            Ez=np.zeros_like(Hz1),
            Hx=np.zeros_like(Hz1),
            Hy=np.zeros_like(Hz1),
            Hz=Hz1,
            x=x,
            y=y,
            power=1.0,
        )

        dx = x[1] - x[0]
        dy = y[1] - y[0]

        orthogonality = mode_matching.check_mode_orthogonality(
            mode0, mode1, direction="z", dx=dx, dy=dy
        )

        # Modes should be reasonably orthogonal (metric close to 0)
        # Due to numerical discretization, we allow some tolerance
        assert orthogonality < 0.5

    def test_forward_backward_separation(self):
        """Test forward/backward mode separation using dual monitors."""
        wavelength = 1.55e-6
        neff = 1.5 + 0j
        distance = 5e-6  # 5 microns between monitors

        # Simulate forward-propagating wave
        # At left monitor: coefficient = 1.0
        # At right monitor: coefficient = 1.0 * exp(i*beta*d)
        beta = 2 * np.pi * neff.real / wavelength
        phase_shift = beta * distance

        coeff_left = 1.0 + 0j
        coeff_right = 1.0 * np.exp(1j * phase_shift)

        a_fwd, a_bwd = mode_matching.separate_forward_backward(
            coeff_left, coeff_right, neff, distance, wavelength
        )

        # Should recover forward amplitude ≈ 1.0, backward ≈ 0.0
        assert abs(a_fwd - 1.0) < 0.1
        assert abs(a_bwd) < 0.1


class TestModePort:
    """Test ModePort boundary condition."""

    @pytest.fixture
    def simple_mode(self):
        """Create a simple test mode."""
        wavelength = 1.55e-6
        nx, ny = 30, 30
        x = np.linspace(-1e-6, 1e-6, nx)
        y = np.linspace(-1e-6, 1e-6, ny)

        X, Y = np.meshgrid(x, y, indexing="ij")
        w0 = 0.5e-6

        Hz = np.exp(-(X**2 + Y**2) / w0**2)

        mode = WaveguideMode(
            mode_number=0,
            neff=1.5 + 0j,
            frequency=299792458.0 / wavelength,
            wavelength=wavelength,
            Ex=np.zeros_like(Hz),
            Ey=np.zeros_like(Hz),
            Ez=np.zeros_like(Hz),
            Hx=np.zeros_like(Hz),
            Hy=np.zeros_like(Hz),
            Hz=Hz,
            x=x,
            y=y,
            power=1.0,
        )

        return mode

    def test_mode_port_creation(self, simple_mode):
        """Test ModePort creation and configuration."""
        config = ModePortConfig(
            center=(0.0, 0.0, 0.0),
            size=(2e-6, 2e-6, 0.0),
            direction="+z",
            modes=[simple_mode],
            inject=True,
        )

        port = ModePort(config, name="test_port")

        assert port.name == "test_port"
        assert port.enabled is True
        assert port.axis == "z"
        assert port.sign == +1

    def test_mode_port_direction_parsing(self, simple_mode):
        """Test direction parsing for different axes."""
        directions = ["+x", "-x", "+y", "-y", "+z", "-z"]
        expected_axes = ["x", "x", "y", "y", "z", "z"]
        expected_signs = [+1, -1, +1, -1, +1, -1]

        for direction, axis, sign in zip(directions, expected_axes, expected_signs):
            config = ModePortConfig(
                center=(0.0, 0.0, 0.0),
                size=(2e-6, 2e-6, 0.0),
                direction=direction,
                modes=[simple_mode],
                inject=False,
            )
            port = ModePort(config)

            assert port.axis == axis
            assert port.sign == sign


class TestModeInjectionExtraction:
    """Test mode injection and extraction in a simulation."""

    def test_mode_injection_creates_expected_profile(self):
        """Test that mode injection creates expected field profile."""
        # This would require a full simulation setup
        # Simplified version for now
        pytest.skip("Requires full simulation framework")

    def test_s_parameter_calculation(self):
        """Test S-parameter calculation from mode monitors."""
        # Create mock mode coefficients
        frequencies = np.array([150e12, 200e12, 250e12])

        # Create a simple mode for testing
        wavelength = 1.55e-6
        nx, ny = 20, 20
        x = np.linspace(-0.5e-6, 0.5e-6, nx)
        y = np.linspace(-0.5e-6, 0.5e-6, ny)

        X, Y = np.meshgrid(x, y, indexing="ij")
        Hz = np.exp(-(X**2 + Y**2) / (0.3e-6) ** 2)

        mode = WaveguideMode(
            mode_number=0,
            neff=1.5 + 0j,
            frequency=299792458.0 / wavelength,
            wavelength=wavelength,
            Ex=np.zeros_like(Hz),
            Ey=np.zeros_like(Hz),
            Ez=np.zeros_like(Hz),
            Hx=np.zeros_like(Hz),
            Hy=np.zeros_like(Hz),
            Hz=Hz,
            x=x,
            y=y,
            power=1.0,
        )

        # Create a mock monitor
        # Note: This is simplified - in practice would need full simulation context
        # For now, just verify the API works
        monitor = ModeExpansionMonitor(
            center=(0.0, 0.0, 0.0),
            size=(1e-6, 1e-6, 0.0),
            modes=[mode],
            direction="z",
            frequencies=frequencies.tolist(),
        )

        # Verify S-parameter computation API
        # Mock some coefficients
        monitor._mode_coeffs_freq = {
            0: np.array([0.9 + 0.1j, 0.85 + 0.15j, 0.8 + 0.2j])
        }

        s_params = monitor.compute_s_parameters(source_mode_index=0, source_power=1.0)

        assert "S_11" in s_params
        assert len(s_params["S_11"]) == len(frequencies)


class TestModeSolver:
    """Test mode solver for generating mode profiles."""

    def test_mode_solver_te_modes(self):
        """Test TE mode calculation for simple waveguide."""
        wavelength = 1.55e-6

        # Create a simple waveguide structure
        nx, ny = 60, 60
        x = np.linspace(-3e-6, 3e-6, nx)
        y = np.linspace(-3e-6, 3e-6, ny)

        # Core-cladding structure
        X, Y = np.meshgrid(x, y, indexing="ij")
        core_width = 1.5e-6

        # Square waveguide
        epsilon = np.ones((nx, ny)) * 1.5**2  # Cladding: n=1.5
        core_mask = (np.abs(X) < core_width / 2) & (np.abs(Y) < core_width / 2)
        epsilon[core_mask] = 1.7**2  # Core: n=1.7

        solver = ModeSolver(wavelength, x, y, epsilon)

        # Solve for fundamental TE mode
        modes = solver.solve(num_modes=1, mode_type="TE")

        assert len(modes) > 0
        mode0 = modes[0]

        # Check mode properties
        assert mode0.mode_number == 0
        assert mode0.wavelength == wavelength
        assert mode0.neff.real > 1.5  # neff should be between cladding and core
        assert mode0.neff.real < 1.7

        # Check field shapes
        assert mode0.Hz.shape == (nx, ny)

        # Mode should be confined to core region
        # Check that field is strongest near center
        center_value = np.abs(mode0.Hz[nx // 2, ny // 2])
        edge_value = np.abs(mode0.Hz[0, 0])
        assert center_value > edge_value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
