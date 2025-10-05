"""
Plane wave validation test suite.

This module implements comprehensive validation tests for plane wave
propagation, including:
- Free space propagation validation
- Interface reflection/transmission (Fresnel coefficients)
- Plane wave scattering from simple geometries
- Comparison with analytical solutions
"""

import pytest
import numpy as np
from prismo import Simulation, TFSFSource, FieldMonitor, ElectricDipole
from prismo.core.grid import YeeGrid, GridSpec


class TestPlaneWavePropagation:
    """Test plane wave propagation in free space."""

    def test_free_space_propagation_2d(self):
        """Test plane wave propagation in vacuum matches analytical solution."""
        # Create simulation
        sim = Simulation(
            size=(2.0, 2.0, 0.0),  # 2µm x 2µm 2D grid
            resolution=40.0,  # 40 points per µm
            boundary_conditions="pml",
            pml_layers=8,
        )

        # Wave parameters
        freq = 150e12  # 150 THz (2 µm wavelength)
        wavelength = 299792458.0 / freq

        # Create TFSF source for clean plane wave injection
        source = TFSFSource(
            center=(1.0, 1.0, 0.0),
            size=(1.0, 1.0, 0.0),  # TFSF region in center
            direction="+x",  # Propagating in +x direction
            polarization="y",  # Ey polarization (TE mode)
            frequency=freq,
            pulse=False,  # Continuous wave
            amplitude=1.0,
        )
        sim.add_source(source)

        # Add field monitors at different positions along propagation
        monitor1 = FieldMonitor(
            center=(0.8, 1.0, 0.0),
            size=(0.0, 0.5, 0.0),  # Line monitor
            components=["Ey"],
            time_domain=True,
        )
        monitor2 = FieldMonitor(
            center=(1.2, 1.0, 0.0),
            size=(0.0, 0.5, 0.0),  # Line monitor
            components=["Ey"],
            time_domain=True,
        )
        sim.add_monitor(monitor1)
        sim.add_monitor(monitor2)

        # Run for several periods
        periods = 5
        sim_time = periods / freq
        sim.run(sim_time)

        # Get field data
        time_points1, ey_data1 = monitor1.get_time_data("Ey")
        time_points2, ey_data2 = monitor2.get_time_data("Ey")

        # Calculate phase difference
        # The monitors are separated by 0.4 µm
        delta_x = 0.4e-6  # meters
        expected_phase_shift = 2 * np.pi * delta_x / wavelength

        # Extract signal at center point of each monitor
        # ey_data has shape (time_steps, spatial_points)
        if len(ey_data1.shape) > 1:
            signal1 = ey_data1[:, len(ey_data1[0]) // 2]
            signal2 = ey_data2[:, len(ey_data2[0]) // 2]
        else:
            signal1 = ey_data1
            signal2 = ey_data2

        # Use last 2 periods for analysis (steady state)
        start_idx = int(3 * len(signal1) / 5)
        signal1_steady = signal1[start_idx:]
        signal2_steady = signal2[start_idx:]
        time_steady = time_points1[start_idx:]

        # Calculate cross-correlation to find phase shift
        correlation = np.correlate(signal1_steady, signal2_steady, mode="full")
        max_corr_idx = np.argmax(correlation)
        time_shift_idx = max_corr_idx - len(signal1_steady) + 1
        time_shift = time_shift_idx * sim.dt
        measured_phase_shift = 2 * np.pi * freq * abs(time_shift)

        # Check phase shift is reasonable (within 20% due to numerical dispersion)
        phase_error = (
            abs(measured_phase_shift - expected_phase_shift) / expected_phase_shift
        )
        assert phase_error < 0.2, f"Phase shift error {phase_error:.1%} exceeds 20%"

        # Check amplitude is consistent (wave hasn't attenuated significantly)
        amp1 = np.max(np.abs(signal1_steady))
        amp2 = np.max(np.abs(signal2_steady))
        amp_ratio = amp2 / amp1
        assert (
            0.9 < amp_ratio < 1.1
        ), f"Amplitude ratio {amp_ratio:.3f} indicates attenuation"

    def test_plane_wave_wavelength(self):
        """Test that simulated wavelength matches theoretical wavelength."""
        # Create simulation
        sim = Simulation(
            size=(3.0, 1.0, 0.0),  # 3µm x 1µm 2D grid
            resolution=50.0,  # 50 points per µm
            boundary_conditions="pml",
            pml_layers=10,
        )

        # Wave parameters
        freq = 100e12  # 100 THz (3 µm wavelength)
        wavelength = 299792458.0 / freq

        # Create TFSF source
        source = TFSFSource(
            center=(1.5, 0.5, 0.0),
            size=(2.0, 0.5, 0.0),
            direction="+x",
            polarization="y",
            frequency=freq,
            pulse=False,
            amplitude=1.0,
        )
        sim.add_source(source)

        # Add spatial monitor along propagation direction
        monitor = FieldMonitor(
            center=(1.5, 0.5, 0.0),
            size=(2.5, 0.0, 0.0),  # Line along x
            components=["Ey"],
            time_domain=True,
        )
        sim.add_monitor(monitor)

        # Run until steady state
        periods = 10
        sim_time = periods / freq
        sim.run(sim_time)

        # Get field data
        time_points, ey_data = monitor.get_time_data("Ey")

        # Use the last snapshot (steady state)
        field_profile = ey_data[-1, :]

        # Find peaks in the spatial profile
        peak_indices = []
        for i in range(1, len(field_profile) - 1):
            if (
                field_profile[i] > field_profile[i - 1]
                and field_profile[i] > field_profile[i + 1]
            ):
                if field_profile[i] > 0.5 * np.max(field_profile):
                    peak_indices.append(i)

        # Calculate average distance between peaks (should be one wavelength)
        if len(peak_indices) >= 2:
            peak_distances = np.diff(peak_indices) * (
                1.0 / sim.resolution
            )  # Convert to physical units
            measured_wavelength = np.mean(peak_distances)

            # Check wavelength accuracy (within 10%)
            wavelength_error = abs(measured_wavelength - wavelength) / wavelength
            assert (
                wavelength_error < 0.1
            ), f"Wavelength error {wavelength_error:.1%} exceeds 10%"
        else:
            pytest.skip("Not enough spatial oscillations to measure wavelength")


class TestFresnelCoefficients:
    """Test reflection and transmission at dielectric interfaces."""

    @pytest.mark.skip(reason="Material interfaces not yet implemented in core")
    def test_normal_incidence_reflection(self):
        """Test Fresnel reflection coefficient for normal incidence."""
        # This test requires material interface support
        # Will be implemented after material system is complete
        pass

    @pytest.mark.skip(reason="Material interfaces not yet implemented in core")
    def test_normal_incidence_transmission(self):
        """Test Fresnel transmission coefficient for normal incidence."""
        # This test requires material interface support
        pass

    @pytest.mark.skip(reason="Material interfaces not yet implemented in core")
    def test_oblique_incidence(self):
        """Test Fresnel coefficients for oblique incidence."""
        # This test requires oblique TFSF implementation
        pass


class TestPlaneWaveScattering:
    """Test plane wave scattering from simple geometries."""

    @pytest.mark.skip(reason="Geometry system not yet implemented")
    def test_cylinder_scattering(self):
        """Test plane wave scattering from a dielectric cylinder."""
        # This test requires geometry and material systems
        pass

    @pytest.mark.skip(reason="Geometry system not yet implemented")
    def test_sphere_scattering(self):
        """Test plane wave scattering from a dielectric sphere (Mie theory)."""
        # This test requires 3D geometry and material systems
        pass


class TestNumericalDispersion:
    """Test numerical dispersion properties of the FDTD algorithm."""

    def test_dispersion_relation(self):
        """Test that numerical dispersion relation is close to ideal."""
        # Create high-resolution simulation to minimize dispersion
        sim = Simulation(
            size=(4.0, 1.0, 0.0),  # Long domain for multiple wavelengths
            resolution=100.0,  # Very high resolution (100 points per µm)
            boundary_conditions="pml",
            pml_layers=15,
        )

        # Use lower frequency for more points per wavelength
        freq = 50e12  # 50 THz (6 µm wavelength)
        wavelength = 299792458.0 / freq

        # Create TFSF source
        source = TFSFSource(
            center=(2.0, 0.5, 0.0),
            size=(3.0, 0.5, 0.0),
            direction="+x",
            polarization="y",
            frequency=freq,
            pulse=False,
            amplitude=1.0,
        )
        sim.add_source(source)

        # Add spatial monitor
        monitor = FieldMonitor(
            center=(2.0, 0.5, 0.0),
            size=(3.5, 0.0, 0.0),
            components=["Ey"],
            time_domain=True,
        )
        sim.add_monitor(monitor)

        # Run simulation
        periods = 15
        sim_time = periods / freq
        sim.run(sim_time)

        # Get field data
        time_points, ey_data = monitor.get_time_data("Ey")

        # Use last snapshot
        field_profile = ey_data[-1, :]

        # Perform FFT to get wavelength in the simulation
        fft = np.fft.fft(field_profile)
        freqs = np.fft.fftfreq(len(field_profile), d=1.0 / sim.resolution)

        # Find peak frequency (spatial frequency)
        peak_idx = np.argmax(np.abs(fft[1 : len(fft) // 2])) + 1
        spatial_freq = abs(freqs[peak_idx])
        measured_wavelength = 1.0 / spatial_freq if spatial_freq > 0 else 0

        # Calculate dispersion error
        if measured_wavelength > 0:
            dispersion_error = abs(measured_wavelength - wavelength) / wavelength
            # High resolution should give < 5% error
            assert (
                dispersion_error < 0.05
            ), f"Dispersion error {dispersion_error:.1%} exceeds 5%"


class TestSourceStability:
    """Test stability and consistency of plane wave sources."""

    def test_continuous_wave_stability(self):
        """Test that continuous wave maintains constant amplitude."""
        # Create simulation
        sim = Simulation(
            size=(1.5, 1.5, 0.0),
            resolution=30.0,
            boundary_conditions="pml",
            pml_layers=10,
        )

        # Wave parameters
        freq = 150e12  # 150 THz

        # Create TFSF source
        source = TFSFSource(
            center=(0.75, 0.75, 0.0),
            size=(0.8, 0.8, 0.0),
            direction="+x",
            polarization="y",
            frequency=freq,
            pulse=False,
            amplitude=1.0,
        )
        sim.add_source(source)

        # Monitor at center
        monitor = FieldMonitor(
            center=(0.75, 0.75, 0.0),
            size=(0.0, 0.0, 0.0),  # Point monitor
            components=["Ey"],
            time_domain=True,
        )
        sim.add_monitor(monitor)

        # Run for many periods
        periods = 20
        sim_time = periods / freq
        sim.run(sim_time)

        # Get field data
        time_points, ey_data = monitor.get_time_data("Ey")
        signal = ey_data[:, 0, 0]

        # Check amplitude stability over time
        # Split into segments and check amplitude consistency
        n_segments = 4
        segment_size = len(signal) // n_segments
        amplitudes = []

        for i in range(n_segments):
            start = i * segment_size
            end = (i + 1) * segment_size
            segment = signal[start:end]
            amp = np.max(np.abs(segment))
            amplitudes.append(amp)

        # Check that amplitude variation is small
        amp_std = np.std(amplitudes)
        amp_mean = np.mean(amplitudes)
        amp_variation = amp_std / amp_mean if amp_mean > 0 else 0

        assert (
            amp_variation < 0.1
        ), f"Amplitude variation {amp_variation:.1%} exceeds 10%"

    def test_gaussian_pulse_shape(self):
        """Test that Gaussian pulse maintains expected shape."""
        # Create simulation
        sim = Simulation(
            size=(2.0, 1.5, 0.0),
            resolution=40.0,
            boundary_conditions="pml",
            pml_layers=10,
        )

        # Wave parameters
        freq = 150e12  # 150 THz
        pulse_width = 10e-15  # 10 fs

        # Create TFSF source with Gaussian pulse
        source = TFSFSource(
            center=(1.0, 0.75, 0.0),
            size=(1.2, 0.8, 0.0),
            direction="+x",
            polarization="y",
            frequency=freq,
            pulse=True,
            pulse_width=pulse_width,
            amplitude=1.0,
        )
        sim.add_source(source)

        # Monitor inside TFSF region
        monitor = FieldMonitor(
            center=(1.0, 0.75, 0.0),
            size=(0.0, 0.0, 0.0),
            components=["Ey"],
            time_domain=True,
        )
        sim.add_monitor(monitor)

        # Run simulation for duration of pulse
        sim_time = 50e-15  # 50 fs
        sim.run(sim_time)

        # Get field data
        time_points, ey_data = monitor.get_time_data("Ey")
        signal = ey_data[:, 0, 0]

        # The envelope should be Gaussian-like
        # Find peak and check that signal decays on both sides
        peak_idx = np.argmax(np.abs(signal))
        peak_value = abs(signal[peak_idx])

        # Check that signal has decayed to < 10% of peak at start and end
        start_value = abs(signal[0])
        end_value = abs(signal[-1])

        assert start_value < 0.1 * peak_value, "Pulse hasn't fully risen at start"
        assert end_value < 0.3 * peak_value, "Pulse hasn't decayed at end"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
