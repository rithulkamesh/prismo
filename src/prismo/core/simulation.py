"""
High-level simulation interface for FDTD simulations.

This module provides the main Simulation class that orchestrates the
entire FDTD simulation process, including grid creation, material definition,
source specification, and monitor placement.
"""

import time as time_module
from typing import Optional, Union

import numpy as np

from prismo.core.fields import ElectromagneticFields
from prismo.core.grid import GridSpec, YeeGrid
from prismo.core.solver import FDTDSolver
from prismo.geometry.shapes import Shape
from prismo.monitors.base import Monitor
from prismo.monitors.field import FieldMonitor
from prismo.solvers.base import SolverBase, TimeDomainSolver
from prismo.sources.base import Source


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
    solver_type : str, optional
        Type of solver to use. Options: "fdtd" (default), "meep", "fem".
        "fdtd" uses the native FDTD solver, "meep" uses MIT MEEP,
        "fem" uses the FEM solver (frequency-domain only).
    """

    def __init__(
        self,
        size: tuple[float, float, float],
        resolution: Union[float, tuple[float, float, float]],
        boundary_conditions: str = "pml",
        pml_layers: int = 10,
        courant_factor: float = 0.9,
        solver_type: str = "fdtd",
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
        self.solver_type = solver_type.lower()

        # Storage for sources and monitors
        self.sources: list[Source] = []
        self.monitors: list[Monitor] = []
        
        # Storage for geometry shapes
        self.shapes: list[Shape] = []

        # Material arrays (will be computed from shapes)
        self.material_arrays: Optional[dict] = None

        # Create fields
        self.fields = ElectromagneticFields(self.grid)

        # Set up solver using factory pattern (will be updated when shapes are added)
        self.solver = self._create_solver(courant_factor)
        self.dt = self.solver.get_time_step()

        # Simulation state
        self.step_count = 0
        self.current_time = 0.0

    def _create_solver(self, courant_factor: float) -> SolverBase:
        """
        Create solver based on solver_type parameter.

        Parameters
        ----------
        courant_factor : float
            Safety factor for time step calculation.

        Returns
        -------
        SolverBase
            The created solver instance.
        """
        if self.solver_type == "fdtd":
            dt = self.grid.get_time_step(courant_factor)
            # Use material arrays if they've been computed
            return FDTDSolver(self.grid, dt, material_arrays=self.material_arrays)
        elif self.solver_type == "meep":
            # Try to import and use MEEP solver
            try:
                from prismo.solvers.meep_solver import MEEPSolver

                return MEEPSolver(self.grid)
            except ImportError:
                raise ImportError(
                    "MEEP solver requested but MEEP is not available. "
                    "Install with: conda install -c conda-forge pymeeus meep"
                )
        elif self.solver_type == "fem":
            # Try to import and use FEM solver
            try:
                from prismo.solvers.fem_solver import FEMSolver

                return FEMSolver(self.grid)
            except ImportError:
                raise ImportError(
                    "FEM solver requested but FEniCS is not available.\n"
                    "Install with: conda install -c conda-forge fenics-dolfinx\n"
                    "or from source: https://fenicsproject.org/download/"
                )
        else:
            raise ValueError(
                f"Unknown solver type '{self.solver_type}'. "
                "Options: 'fdtd', 'meep', 'fem'"
            )

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

    def add_shape(self, shape: Shape) -> None:
        """
        Add a geometric shape to the simulation.

        Parameters
        ----------
        shape : Shape
            The geometric shape to add.
        """
        self.shapes.append(shape)
        # Recompute material arrays when shapes are added
        self._update_material_arrays()

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

    def _update_material_arrays(self) -> None:
        """
        Rasterize all shapes to create material property arrays.
        
        This method converts geometric shapes into material property arrays
        (eps_rel, mu_rel) that are used by the FDTD solver.
        """
        if len(self.shapes) == 0:
            # No shapes, use vacuum (default)
            self.material_arrays = None
            return
        
        # Get grid dimensions
        nx, ny, nz = self.grid.dimensions
        
        # Initialize material arrays with vacuum (eps_r=1, mu_r=1)
        eps_rel = np.ones((nx, ny, nz), dtype=np.float64)
        mu_rel = np.ones((nx, ny, nz), dtype=np.float64)
        
        # Get grid coordinates
        dx, dy, dz = self.grid.spacing
        origin = self.grid.origin
        
        # Create coordinate arrays
        x = origin[0] + np.arange(nx) * dx
        y = origin[1] + np.arange(ny) * dy
        if self.grid.is_2d:
            z = np.array([0.0])
        else:
            z = origin[2] + np.arange(nz) * dz
        
        # Rasterize each shape and apply its material properties
        for shape in self.shapes:
            # Rasterize shape to get boolean mask
            mask = shape.rasterize(x, y, z if not self.grid.is_2d else None)
            
            # Ensure mask has correct shape
            if self.grid.is_2d and len(mask.shape) == 2:
                # Expand to 3D for consistency
                mask = mask[:, :, np.newaxis]
            
            # Apply material properties where shape is present
            # Use the shape's material epsilon_r and mu_r
            eps_rel[mask] = shape.material.epsilon_r
            mu_rel[mask] = shape.material.mu_r
        
        # Store material arrays
        self.material_arrays = {
            "eps_rel": eps_rel,
            "mu_rel": mu_rel,
            "sigma_e": np.zeros((nx, ny, nz), dtype=np.float64),
            "sigma_m": np.zeros((nx, ny, nz), dtype=np.float64),
        }
        
        # Recreate solver with new material arrays
        if self.solver_type == "fdtd":
            # Preserve current time step
            if hasattr(self, 'solver') and self.solver is not None:
                dt = self.solver.get_time_step()
            else:
                dt = self.grid.get_time_step(self.courant_factor)
            
            # Create new solver with material arrays
            # The solver will create its own fields, but we'll sync them
            new_solver = FDTDSolver(self.grid, dt, material_arrays=self.material_arrays)
            
            # Sync fields: copy field values from old fields to new solver's fields
            if hasattr(self, 'fields') and self.fields is not None:
                # Copy field values to new solver's fields
                new_solver.fields.Ex[:] = self.fields.Ex
                new_solver.fields.Ey[:] = self.fields.Ey
                new_solver.fields.Ez[:] = self.fields.Ez
                new_solver.fields.Hx[:] = self.fields.Hx
                new_solver.fields.Hy[:] = self.fields.Hy
                new_solver.fields.Hz[:] = self.fields.Hz
                # Update reference
                self.fields = new_solver.fields
            
            self.solver = new_solver
            self.dt = self.solver.get_time_step()

    def step(self) -> None:
        """
        Run a single time step of the simulation.
        """
        # Step solver (works for time-domain solvers)
        if isinstance(self.solver, TimeDomainSolver):
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
        else:
            raise NotImplementedError(
                f"step() not implemented for solver type {type(self.solver)}. "
                "Use solve() for frequency-domain solvers."
            )

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
