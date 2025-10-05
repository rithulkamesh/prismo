"""
Tests for the source implementations.

This module tests various source implementations, including point sources,
plane wave sources, and Gaussian beam sources.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from prismo.core.grid import YeeGrid, GridSpec
from prismo.core.fields import ElectromagneticFields
from prismo.sources.waveform import (
    Waveform,
    ContinuousWave,
    GaussianPulse,
    RickerWavelet,
    CustomWaveform,
)
from prismo.sources.base import Source
from prismo.sources.point import PointSource, ElectricDipole, MagneticDipole
from prismo.sources.gaussian import GaussianBeamSource
from prismo.sources.plane_wave import PlaneWaveSource


class TestWaveforms:
    """Test cases for waveform classes."""

    def test_continuous_wave(self):
        """Test the ContinuousWave waveform."""
        freq = 1e9  # 1 GHz
        amplitude = 2.0
        phase = np.pi / 4

        cw = ContinuousWave(freq, amplitude, phase)

        # Test single time point
        t = 1e-9  # 1 ns
        expected = amplitude * np.sin(2 * np.pi * freq * t + phase)
        assert_allclose(cw(t), expected)

        # Test array of time points
        t_array = np.array([1e-9, 2e-9, 3e-9])
        expected_array = amplitude * np.sin(2 * np.pi * freq * t_array + phase)
        assert_allclose(cw(t_array), expected_array)

    def test_gaussian_pulse(self):
        """Test the GaussianPulse waveform."""
        freq = 1e9  # 1 GHz
        pulse_width = 1e-9  # 1 ns
        amplitude = 2.0
        phase = 0.0
        delay = 5e-9  # 5 ns

        pulse = GaussianPulse(freq, pulse_width, amplitude, phase, delay)

        # Test single time point at the peak
        t = delay
        tau = 0  # At peak, (t - delay) / pulse_width = 0
        expected = amplitude * np.exp(-0.5 * tau * tau) * np.sin(2 * np.pi * freq * t)
        assert_allclose(pulse(t), expected)

        # Test array of time points
        t_array = np.array([delay - pulse_width, delay, delay + pulse_width])
        tau_array = (t_array - delay) / pulse_width
        expected_array = (
            amplitude
            * np.exp(-0.5 * tau_array * tau_array)
            * np.sin(2 * np.pi * freq * t_array)
        )
        assert_allclose(pulse(t_array), expected_array)

    def test_ricker_wavelet(self):
        """Test the RickerWavelet waveform."""
        freq = 1e9  # 1 GHz
        amplitude = 2.0

        ricker = RickerWavelet(freq, amplitude)

        # Test at peak time
        t = ricker.delay
        tau = 0  # At peak
        expected = amplitude
        assert_allclose(ricker(t), expected)

        # Test at time where function should be zero
        # This occurs when tau^2 = 0.5
        tau_squared = 0.5
        # Calculate the time that gives this tau value
        t_zero = ricker.delay + np.sqrt(tau_squared) / (np.pi * freq)
        expected_zero = amplitude * (1.0 - 2.0 * 0.5) * np.exp(-0.5)
        assert_allclose(ricker(t_zero), expected_zero, atol=1e-14)

    def test_custom_waveform(self):
        """Test the CustomWaveform class."""

        def my_func(t):
            return np.sin(t) * np.cos(2 * t)

        amplitude = 3.0
        custom = CustomWaveform(my_func, amplitude)

        # Test single time point
        t = 1.0
        expected = amplitude * my_func(t)
        assert_allclose(custom(t), expected)

        # Test array of time points
        t_array = np.array([0.0, 0.5, 1.0])
        expected_array = amplitude * np.array([my_func(t) for t in t_array])
        assert_allclose(custom(t_array), expected_array)


class TestPointSource:
    """Test cases for point source implementations."""

    @pytest.fixture
    def grid_2d(self):
        """Create a simple 2D grid for testing."""
        spec = GridSpec(
            size=(1.0, 1.0, 0.0),  # 1µm x 1µm 2D grid
            resolution=10.0,  # 10 points per µm
        )
        return YeeGrid(spec)

    @pytest.fixture
    def grid_3d(self):
        """Create a simple 3D grid for testing."""
        spec = GridSpec(
            size=(1.0, 1.0, 1.0),  # 1µm x 1µm x 1µm 3D grid
            resolution=10.0,  # 10 points per µm
        )
        return YeeGrid(spec)

    def test_point_source_2d(self, grid_2d):
        """Test a point source in a 2D grid."""
        # Create a simple sine wave waveform
        waveform = ContinuousWave(frequency=2e14, amplitude=1.0)

        # Create a point source at the center
        source = PointSource(
            position=(0.5, 0.5, 0.0), component="Ez", waveform=waveform
        )

        # Initialize source and fields
        source.initialize(grid_2d)
        fields = ElectromagneticFields(grid_2d)

        # Update fields at a non-zero time (sine wave is zero at t=0)
        source.update_fields(fields, time=1e-16, dt=1e-16)

        # The Ez component should have a non-zero value at the source position
        assert np.max(np.abs(fields["Ez"])) > 0.0

    def test_electric_dipole_3d(self, grid_3d):
        """Test an electric dipole in a 3D grid."""
        # Create an electric dipole at the center
        source = ElectricDipole(
            position=(0.5, 0.5, 0.5),
            polarization="z",
            frequency=2e14,
            pulse=True,
            pulse_width=1e-14,
            amplitude=1.0,
        )

        # Initialize source and fields
        source.initialize(grid_3d)
        fields = ElectromagneticFields(grid_3d)

        # Update fields at peak time
        source.update_fields(fields, time=source.waveform.delay, dt=1e-16)

        # The Ez component should have a non-zero value at the source position
        assert np.max(np.abs(fields["Ez"])) > 0.0

    def test_magnetic_dipole_3d(self, grid_3d):
        """Test a magnetic dipole in a 3D grid."""
        # Create a magnetic dipole at the center
        source = MagneticDipole(
            position=(0.5, 0.5, 0.5),
            polarization="z",
            frequency=2e14,
            pulse=True,
            pulse_width=1e-14,
            amplitude=1.0,
        )

        # Initialize source and fields
        source.initialize(grid_3d)
        fields = ElectromagneticFields(grid_3d)

        # Update fields at peak time
        source.update_fields(fields, time=source.waveform.delay, dt=1e-16)

        # The Hz component should have a non-zero value at the source position
        assert np.max(np.abs(fields["Hz"])) > 0.0


class TestPlaneWaveSource:
    """Test cases for plane wave source implementation."""

    @pytest.fixture
    def grid_2d(self):
        """Create a simple 2D grid for testing."""
        spec = GridSpec(
            size=(1.0, 1.0, 0.0),  # 1µm x 1µm 2D grid
            resolution=10.0,  # 10 points per µm
        )
        return YeeGrid(spec)

    def test_plane_wave_x_polarization_y(self, grid_2d):
        """Test a y-propagating plane wave with x-polarization."""
        # Create a plane wave source
        source = PlaneWaveSource(
            center=(0.5, 0.1, 0.0),
            size=(0.8, 0.0, 0.0),  # Line source along x
            direction="+y",
            polarization="x",
            frequency=2e14,
            pulse=False,
            amplitude=1.0,
        )

        # Initialize source and fields
        source.initialize(grid_2d)
        fields = ElectromagneticFields(grid_2d)

        # Update fields at a non-zero time (sine wave is zero at t=0)
        source.update_fields(fields, time=1e-16, dt=1e-16)

        # The Ex component should have non-zero values along the source line
        assert np.max(np.abs(fields["Ex"])) > 0.0

        # The Hz component should also have non-zero values
        assert np.max(np.abs(fields["Hz"])) > 0.0

    def test_plane_wave_polarization_validation(self):
        """Test that polarization perpendicular to direction is enforced."""
        # Should raise an error when polarization is the same as direction
        with pytest.raises(ValueError):
            PlaneWaveSource(
                center=(0.5, 0.5, 0.0),
                size=(1.0, 0.0, 0.0),
                direction="x",
                polarization="x",
                frequency=2e14,
            )


class TestGaussianBeamSource:
    """Test cases for Gaussian beam source implementation."""

    @pytest.fixture
    def grid_2d(self):
        """Create a simple 2D grid for testing."""
        spec = GridSpec(
            size=(1.0, 1.0, 0.0),  # 1µm x 1µm 2D grid
            resolution=20.0,  # 20 points per µm
        )
        return YeeGrid(spec)

    def test_gaussian_beam_x_propagation(self, grid_2d):
        """Test a Gaussian beam propagating in the x-direction."""
        # Create a Gaussian beam source
        source = GaussianBeamSource(
            center=(0.1, 0.5, 0.0),
            size=(0.0, 0.8, 0.0),  # Line source along y
            direction="x",
            polarization="y",
            frequency=2e14,
            beam_waist=0.2e-6,  # 200 nm
            pulse=False,
            amplitude=1.0,
        )

        # Initialize source and fields
        source.initialize(grid_2d)
        fields = ElectromagneticFields(grid_2d)

        # Update fields at a non-zero time (sine wave is zero at t=0)
        source.update_fields(fields, time=1e-16, dt=1e-16)

        # The Ey component should have non-zero values with a Gaussian profile
        assert np.max(np.abs(fields["Ey"])) > 0.0

        # Check that the Gaussian profile is centered properly
        # Find the index of maximum field value
        max_idx = np.unravel_index(np.argmax(np.abs(fields["Ey"])), fields["Ey"].shape)

        # The maximum should be near the center y-position
        assert abs(max_idx[1] - grid_2d.Ny // 2) <= 1
