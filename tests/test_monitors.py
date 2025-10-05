"""
Tests for the monitor implementations.

This module tests various monitor implementations, including field monitors
for time-domain and frequency-domain analysis.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from prismo.core.grid import YeeGrid, GridSpec
from prismo.core.fields import ElectromagneticFields
from prismo.monitors.base import Monitor
from prismo.monitors.field import FieldMonitor


class TestFieldMonitor:
    """Test cases for field monitor implementation."""

    @pytest.fixture
    def grid_2d(self):
        """Create a simple 2D grid for testing."""
        spec = GridSpec(
            size=(1.0, 1.0, 0.0),  # 1µm x 1µm 2D grid
            resolution=10.0,  # 10 points per µm
        )
        return YeeGrid(spec)

    @pytest.fixture
    def fields_2d(self, grid_2d):
        """Create fields with a simple pattern for testing."""
        fields = ElectromagneticFields(grid_2d)

        # Create a simple sinusoidal pattern in Ez
        for i in range(grid_2d.Nx):
            for j in range(grid_2d.Ny):
                x = i / grid_2d.Nx
                y = j / grid_2d.Ny
                fields["Ez"][i, j, 0] = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

        return fields

    def test_time_domain_monitor(self, grid_2d, fields_2d):
        """Test a time-domain field monitor."""
        # Create a field monitor at the center
        monitor = FieldMonitor(
            center=(0.5, 0.5, 0.0),
            size=(0.2, 0.2, 0.0),
            components=["Ez"],
            time_domain=True,
        )

        # Initialize monitor
        monitor.initialize(grid_2d)

        # Record multiple time steps
        num_steps = 5
        dt = 1e-15  # 1 fs
        for t in range(num_steps):
            time = t * dt
            monitor.update(fields_2d, time, dt)

        # Get time data
        time_points, ez_data = monitor.get_time_data("Ez")

        # Check that we have the correct number of time points
        assert len(time_points) == num_steps
        assert ez_data.shape[0] == num_steps

        # Check that time points are correct
        assert_allclose(time_points, np.arange(num_steps) * dt)

    def test_frequency_domain_monitor(self, grid_2d, fields_2d):
        """Test a frequency-domain field monitor."""
        # Create a field monitor with frequency analysis
        frequencies = [1.0e14, 2.0e14]  # 100 THz, 200 THz
        monitor = FieldMonitor(
            center=(0.5, 0.5, 0.0),
            size=(0.2, 0.2, 0.0),
            components=["Ez"],
            frequencies=frequencies,
            time_domain=False,
        )

        # Initialize monitor
        monitor.initialize(grid_2d)

        # Record multiple time steps with oscillating field
        num_steps = 20
        dt = 1e-15  # 1 fs
        for t in range(num_steps):
            time = t * dt
            # Update field with time-dependent amplitude to simulate oscillation
            fields_2d["Ez"] *= np.cos(2 * np.pi * 1.0e14 * time)

            monitor.update(fields_2d, time, dt)

        # Get frequency data
        freq_data = monitor.get_frequency_data("Ez", 1.0e14)

        # Check that we have non-zero frequency data
        assert np.max(np.abs(freq_data)) > 0.0

    def test_power_flow_calculation(self, grid_2d):
        """Test power flow calculation."""
        # Create a field monitor
        monitor = FieldMonitor(
            center=(0.5, 0.5, 0.0),
            size=(0.2, 0.2, 0.0),
            components=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
            time_domain=True,
            frequencies=[1.0e14],
        )

        # Initialize monitor and fields
        monitor.initialize(grid_2d)
        fields = ElectromagneticFields(grid_2d)

        # Create a plane wave pattern: Ex and Hz for propagation in y-direction
        for i in range(grid_2d.Nx):
            for j in range(grid_2d.Ny):
                fields["Ex"][i, j, 0] = np.sin(2 * np.pi * j / grid_2d.Ny)
                fields["Hz"][i, j, 0] = 0.01 * np.sin(2 * np.pi * j / grid_2d.Ny)

        # Record multiple time steps
        num_steps = 5
        dt = 1e-15  # 1 fs
        for t in range(num_steps):
            time = t * dt
            monitor.update(fields, time, dt)

        # Get power flow in time domain
        time_points, power_flow = monitor.get_power_flow()

        # Check that power flow is non-zero
        assert np.max(np.abs(power_flow)) > 0.0

        # Get power flow in frequency domain
        freq_power = monitor.get_power_flow(1.0e14)

        # Check that frequency-domain power flow is non-zero
        assert np.max(np.abs(freq_power)) > 0.0

    def test_monitor_component_validation(self):
        """Test validation of monitor field components."""
        # Invalid component should raise an error
        with pytest.raises(ValueError):
            FieldMonitor(
                center=(0.5, 0.5, 0.0),
                size=(0.2, 0.2, 0.0),
                components=["Ex", "InvalidComponent"],
            )
