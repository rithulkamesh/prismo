"""
High-level simulation interface for FDTD simulations.

This module provides the main Simulation class that orchestrates the
entire FDTD simulation process, including grid creation, material definition,
source specification, and monitor placement.
"""

from typing import Tuple, Dict, List, Optional, Union, Literal, Set, Any
import numpy as np
import time as time_module

from prismo.core.grid import YeeGrid, GridSpec
from prismo.core.fields import ElectromagneticFields
from prismo.core.solver import FDTDSolver, MaxwellUpdater
from prismo.sources.base import Source
from prismo.monitors.base import Monitor
from prismo.monitors.field import FieldMonitor


class Simulation:
    """
    High-level simulation interface for FDTD simulations.

    This class orchestrates the entire FDTD simulation process, providing
    a simple interface for specifying geometry, materials, sources, and monitors.

    Parameters
    ----------
    size : Tuple[float, float, float]
        Physical size of the simulation domain (Lx, Ly, Lz) in meters.
        For 2D simulations, set Lz=0.
    resolution : float or Tuple[float, float, float]
        Grid resolution in points per meter. If scalar, same resolution
        is used in all dimensions. If tuple, (res_x, res_y, res_z).
    boundary_conditions : str, optional
        Type of boundary conditions, default="pml".
        Options: "pml", "periodic", "reflecting".
    pml_layers : int, optional
        Number of PML layers for absorbing boundaries, default=10.
    courant_factor : float, optional
        Safety factor for time step calculation, default=0.9.
    """

    def __init__(
        self,
        size: Tuple[float, float, float],
        resolution: Union[float, Tuple[float, float, float]],
        boundary_conditions: str = "pml",
        pml_layers: int = 10,
        courant_factor: float = 0.9,
    ):
        # Create grid
        self.grid_spec = GridSpec(
            size=size,
            resolution=resolution,
            boundary_layers=pml_layers,
        )
        self.grid = YeeGrid(self.grid_spec)

        # Store parameters
        self.size = size
        self.resolution = resolution
        self.boundary_conditions = boundary_conditions
        self.courant_factor = courant_factor

        # Create fields
        self.fields = ElectromagneticFields(self.grid)

        # Set up solver
        self.dt = self.grid.get_time_step(courant_factor)
        self.solver = FDTDSolver(self.grid, self.dt)

        # Storage for sources and monitors
        self.sources: List[Source] = []
        self.monitors: List[Monitor] = []

        # Simulation state
        self.step_count = 0
        self.current_time = 0.0

    def add_source(self, source: Source) -> None:
        """
        Add a source to the simulation.

        Parameters
        ----------
        source : Source
            The source to add.
        """
        source.initialize(self.grid)
        self.sources.append(source)

    def add_monitor(self, monitor: Monitor) -> None:
        """
        Add a monitor to the simulation.

        Parameters
        ----------
        monitor : Monitor
            The monitor to add.
        """
        monitor.initialize(self.grid)
        self.monitors.append(monitor)

    def run(
        self,
        time: float,
        progress_callback: Optional[callable] = None,
        progress_interval: int = 10,
    ) -> None:
        """
        Run the simulation for a specified amount of time.

        Parameters
        ----------
        time : float
            Simulation time in seconds.
        progress_callback : callable, optional
            Function to call to report progress.
        progress_interval : int, optional
            Interval in time steps for progress updates, default=10.
        """
        # Calculate number of time steps
        steps = int(np.ceil(time / self.dt))

        # Start timer
        start_time = time_module.time()

        # Run simulation
        for i in range(steps):
            self.step()

            # Report progress
            if progress_callback is not None and i % progress_interval == 0:
                progress_callback(
                    i, steps, self.current_time, time_module.time() - start_time
                )

        # Final progress report
        if progress_callback is not None:
            progress_callback(
                steps, steps, self.current_time, time_module.time() - start_time
            )

    def step(self) -> None:
        """
        Run a single time step of the simulation.
        """
        # Step FDTD solver (update H fields to n+1/2, then E fields to n+1)
        self.solver.step(self.fields)

        # Update simulation state (time is now n+1)
        self.step_count += 1
        self.current_time += self.dt

        # Update sources at current time (n+1, matching E-field time)
        for source in self.sources:
            source.update_fields(self.fields, self.current_time, self.dt)

        # Update monitors
        for monitor in self.monitors:
            monitor.update(self.fields, self.current_time, self.dt)

    def get_field_data(self, monitor: FieldMonitor, component: str) -> np.ndarray:
        """
        Get field data from a monitor.

        Parameters
        ----------
        monitor : FieldMonitor
            The monitor from which to retrieve data.
        component : str
            The field component to retrieve.

        Returns
        -------
        numpy.ndarray
            Field data for the specified component.
        """
        if monitor not in self.monitors:
            raise ValueError("Monitor not found in this simulation")

        # Get time domain data
        time_points, field_data = monitor.get_time_data(component)

        return field_data

    def get_frequency_data(
        self, monitor: FieldMonitor, component: str, frequency: float
    ) -> np.ndarray:
        """
        Get frequency-domain field data from a monitor.

        Parameters
        ----------
        monitor : FieldMonitor
            The monitor from which to retrieve data.
        component : str
            The field component to retrieve.
        frequency : float
            The frequency in Hz for which to retrieve data.

        Returns
        -------
        numpy.ndarray
            Complex-valued field data at the specified frequency.
        """
        if monitor not in self.monitors:
            raise ValueError("Monitor not found in this simulation")

        return monitor.get_frequency_data(component, frequency)

    def get_transmission(
        self, monitor: FieldMonitor, frequency: Optional[float] = None
    ) -> float:
        """
        Calculate power transmission through a monitor.

        Parameters
        ----------
        monitor : FieldMonitor
            The monitor for which to calculate transmission.
        frequency : float, optional
            Frequency in Hz for frequency-domain calculation.

        Returns
        -------
        float
            Normalized power transmission through the monitor.
        """
        if monitor not in self.monitors:
            raise ValueError("Monitor not found in this simulation")

        if frequency is None:
            # Time-domain calculation
            time_points, power_flow = monitor.get_power_flow()

            # Average over time steps
            avg_power = np.mean(power_flow)
        else:
            # Frequency-domain calculation
            power_flow = monitor.get_power_flow(frequency)

            # Average over spatial points
            avg_power = np.mean(power_flow)

        # Return normalized power (in future, normalize to source power)
        return avg_power
