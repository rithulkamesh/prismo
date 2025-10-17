"""
Tests for advanced monitors (DFT, flux, mode expansion).
"""

import pytest
import numpy as np

from prismo import DFTMonitor, FluxMonitor
from prismo.core.grid import YeeGrid, GridSpec


class TestDFTMonitor:
    """Test DFT monitor functionality."""

    def test_dft_monitor_creation(self):
        """Test DFT monitor creation."""
        frequencies = [190e12, 193e12, 200e12]

        monitor = DFTMonitor(
            center=(0, 0, 0),
            size=(0, 1e-6, 0),
            frequencies=frequencies,
        )

        assert len(monitor.frequencies) == 3
        assert len(monitor.omega) == 3

    def test_dft_monitor_initialization(self):
        """Test DFT monitor initialization on grid."""
        grid_spec = GridSpec(
            size=(5e-6, 3e-6, 0),
            resolution=10e6,
        )
        grid = YeeGrid(grid_spec)

        monitor = DFTMonitor(
            center=(0, 0, 0),
            size=(0, 1e-6, 0),
            frequencies=[193e12],
            components=["Ex", "Ey"],
        )

        monitor.initialize(grid)

        assert "Ex" in monitor._dft_data
        assert "Ey" in monitor._dft_data


class TestFluxMonitor:
    """Test flux monitor functionality."""

    def test_flux_monitor_creation(self):
        """Test flux monitor creation."""
        monitor = FluxMonitor(
            center=(0, 0, 0),
            size=(0, 1e-6, 0),
            direction="x",
            frequencies=[193e12],
        )

        assert monitor.direction == "x"
        assert monitor.frequencies is not None

    def test_flux_monitor_initialization(self):
        """Test flux monitor initialization."""
        grid_spec = GridSpec(
            size=(5e-6, 3e-6, 0),
            resolution=10e6,
        )
        grid = YeeGrid(grid_spec)

        monitor = FluxMonitor(
            center=(0, 0, 0),
            size=(0, 1e-6, 0),
            direction="x",
        )

        monitor.initialize(grid)

        # Check storage is initialized
        assert monitor._power_flow_history == []
        assert monitor._time_history == []
