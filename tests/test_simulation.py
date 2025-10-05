"""
Tests for the high-level Simulation class.

This module tests the Simulation class that orchestrates the entire
FDTD simulation process, including grid creation, source specification,
and monitor placement.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from prismo.core.simulation import Simulation
from prismo.sources.point import PointSource, ElectricDipole
from prismo.sources.waveform import ContinuousWave, GaussianPulse
from prismo.monitors.field import FieldMonitor


class TestSimulation:
    """Test cases for the Simulation class."""

    def test_simulation_initialization(self):
        """Test simulation initialization."""
        # Create a simple 2D simulation
        sim = Simulation(
            size=(1.0, 1.0, 0.0),
            resolution=10.0,
        )

        # Check that core components are properly initialized
        assert sim.grid is not None
        assert sim.fields is not None
        assert sim.solver is not None
        assert sim.dt > 0.0

        # Check that simulation state is initialized
        assert sim.step_count == 0
        assert sim.current_time == 0.0

    def test_add_source(self):
        """Test adding a source to the simulation."""
        # Create a simulation
        sim = Simulation(
            size=(1.0, 1.0, 0.0),
            resolution=10.0,
        )

        # Create a simple source
        waveform = ContinuousWave(frequency=2e14, amplitude=1.0)
        source = PointSource(
            position=(0.5, 0.5, 0.0), component="Ez", waveform=waveform
        )

        # Add source to simulation
        sim.add_source(source)

        # Check that source was added
        assert len(sim.sources) == 1
        assert sim.sources[0] is source

        # Check that source was initialized
        assert source._grid is sim.grid

    def test_add_monitor(self):
        """Test adding a monitor to the simulation."""
        # Create a simulation
        sim = Simulation(
            size=(1.0, 1.0, 0.0),
            resolution=10.0,
        )

        # Create a monitor
        monitor = FieldMonitor(
            center=(0.5, 0.5, 0.0), size=(0.2, 0.2, 0.0), components=["Ez"]
        )

        # Add monitor to simulation
        sim.add_monitor(monitor)

        # Check that monitor was added
        assert len(sim.monitors) == 1
        assert sim.monitors[0] is monitor

        # Check that monitor was initialized
        assert monitor._grid is sim.grid

    def test_simulation_step(self):
        """Test a single simulation step."""
        # Create a simulation
        sim = Simulation(
            size=(1.0, 1.0, 0.0),
            resolution=10.0,
        )

        # Create and add a source
        source = ElectricDipole(
            position=(0.5, 0.5, 0.0),
            polarization="z",
            frequency=2e14,
            pulse=False,
            amplitude=1.0,
        )
        sim.add_source(source)

        # Create and add a monitor
        monitor = FieldMonitor(
            center=(0.5, 0.5, 0.0), size=(0.2, 0.2, 0.0), components=["Ez"]
        )
        sim.add_monitor(monitor)

        # Run a single step
        sim.step()

        # Check that simulation state was updated
        assert sim.step_count == 1
        assert sim.current_time == sim.dt

        # Check that fields were updated (non-zero somewhere)
        assert np.max(np.abs(sim.fields["Ez"])) > 0.0

        # Check that monitor recorded data
        time_points, field_data = monitor.get_time_data("Ez")
        assert len(time_points) == 1
        assert field_data.shape[0] == 1

    def test_simulation_run(self):
        """Test running a simulation for multiple steps."""
        # Create a simulation
        sim = Simulation(
            size=(1.0, 1.0, 0.0),
            resolution=10.0,
        )

        # Create and add a source
        source = ElectricDipole(
            position=(0.5, 0.5, 0.0),
            polarization="z",
            frequency=2e14,
            pulse=False,
            amplitude=1.0,
        )
        sim.add_source(source)

        # Create and add a monitor
        monitor = FieldMonitor(
            center=(0.5, 0.5, 0.0), size=(0.2, 0.2, 0.0), components=["Ez"]
        )
        sim.add_monitor(monitor)

        # Run the simulation for 10 time steps
        run_time = 10 * sim.dt
        sim.run(run_time)

        # Check that simulation ran for the correct number of steps
        assert sim.step_count == 10
        assert_allclose(sim.current_time, run_time)

        # Check that monitor recorded data for all steps
        time_points, field_data = monitor.get_time_data("Ez")
        assert len(time_points) == 10
        assert field_data.shape[0] == 10

    def test_get_field_data(self):
        """Test retrieving field data from a monitor."""
        # Create a simulation
        sim = Simulation(
            size=(1.0, 1.0, 0.0),
            resolution=10.0,
        )

        # Create and add a source
        source = ElectricDipole(
            position=(0.5, 0.5, 0.0),
            polarization="z",
            frequency=2e14,
            pulse=False,
            amplitude=1.0,
        )
        sim.add_source(source)

        # Create and add a monitor
        monitor = FieldMonitor(
            center=(0.5, 0.5, 0.0), size=(0.2, 0.2, 0.0), components=["Ez"]
        )
        sim.add_monitor(monitor)

        # Run the simulation for a few steps
        sim.run(5 * sim.dt)

        # Get field data from the monitor
        field_data = sim.get_field_data(monitor, "Ez")

        # Check that field data has the correct shape
        assert field_data.shape[0] == 5  # 5 time steps
