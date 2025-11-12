"""
MIT MEEP FDTD solver integration.

This module provides a wrapper for MIT MEEP, allowing it to be used as an
alternative FDTD solver with the same interface as the native FDTD solver.
"""

from typing import Optional, Union

import numpy as np

from prismo.backends import Backend, get_backend
from prismo.core.fields import ElectromagneticFields
from prismo.core.grid import YeeGrid
from prismo.solvers.base import TimeDomainSolver

# Try to import MEEP
try:
    import meep as mp

    MEEP_AVAILABLE = True
except ImportError:
    MEEP_AVAILABLE = False
    mp = None


class MEEPSolver(TimeDomainSolver):
    """
    MIT MEEP FDTD solver wrapper.

    This class wraps MIT MEEP's FDTD solver, providing the same interface
    as the native FDTDSolver for seamless integration.

    Parameters
    ----------
    grid : YeeGrid
        The simulation grid.
    material_arrays : dict, optional
        Material property arrays (ε, μ, σ).
    backend : Backend, optional
        Computational backend to use (MEEP handles its own backend).
    """

    def __init__(
        self,
        grid: YeeGrid,
        material_arrays: Optional[dict] = None,
        backend: Optional[Union[str, Backend]] = None,
    ):
        if not MEEP_AVAILABLE:
            raise ImportError(
                "MEEP is not available. Install with: "
                "conda install -c conda-forge pymeeus meep or "
                "pip install meep (requires compiled MEEP library)"
            )

        super().__init__(grid)

        # Initialize backend (MEEP uses its own, but we track for compatibility)
        if isinstance(backend, str):
            self.backend = get_backend(backend)
        elif isinstance(backend, Backend):
            self.backend = backend
        elif backend is None:
            self.backend = get_backend()
        else:
            raise TypeError("backend must be a Backend instance or string name")

        self.material_arrays = material_arrays or {}

        # Physical constants
        self.eps0 = 8.854187817e-12  # Vacuum permittivity (F/m)
        self.mu0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)
        self.c = 299792458.0  # Speed of light (m/s)

        # MEEP simulation object (will be created when needed)
        self.meep_sim = None
        self._setup_meep_simulation()

        # Initialize fields
        self.fields = ElectromagneticFields(grid, backend=self.backend)

        # Time step (MEEP calculates this automatically)
        self.dt = self.grid.suggest_time_step(safety_factor=0.9)

    def _setup_meep_simulation(self) -> None:
        """Set up MEEP simulation from grid and materials."""
        if not MEEP_AVAILABLE:
            return

        # Convert Prismo grid to MEEP geometry
        nx, ny, nz = self.grid.dimensions
        dx, dy, dz = self.grid.spacing

        # MEEP cell size (physical dimensions)
        Lx, Ly, Lz = self.grid.spec.size

        # Create MEEP cell
        if self.grid.is_2d:
            cell = mp.Vector3(Lx, Ly, 0)
        else:
            cell = mp.Vector3(Lx, Ly, Lz)

        # Set up MEEP geometry (materials)
        geometry = []
        # TODO: Convert Prismo materials to MEEP materials
        # This would iterate through geometry and create MEEP geometric objects

        # Set up sources (empty for now, will be added via add_source)
        sources = []

        # Set up PML boundaries
        pml_layers = [mp.PML(self.grid.spec.boundary_layers * max(dx, dy, dz))]

        # Create MEEP simulation
        # Note: MEEP uses different resolution definition (pixels per unit length)
        resolution = (
            min(self.grid.spec.resolution)
            if isinstance(self.grid.spec.resolution, tuple)
            else self.grid.spec.resolution
        )

        self.meep_sim = mp.Simulation(
            cell_size=cell,
            boundary_layers=pml_layers,
            geometry=geometry,
            sources=sources,
            resolution=resolution,
        )

    def step(self, fields: Optional[ElectromagneticFields] = None) -> None:
        """
        Perform a single MEEP time step.

        Parameters
        ----------
        fields : ElectromagneticFields, optional
            Fields to update. If None, uses internal fields.
        """
        if not MEEP_AVAILABLE:
            raise ImportError("MEEP is required for MEEPSolver")

        if self.meep_sim is None:
            self._setup_meep_simulation()

        # Run one MEEP step
        self.meep_sim.run(until=1)  # Run for 1 time unit (MEEP's internal unit)

        # Update time and step count
        self.time += self.dt
        self.step_count += 1

        # Extract fields from MEEP to Prismo format
        if fields is not None:
            self._extract_fields_to_prismo(fields)
        else:
            self._extract_fields_to_prismo(self.fields)

    def _extract_fields_to_prismo(self, fields: ElectromagneticFields) -> None:
        """Extract fields from MEEP simulation to Prismo field format."""
        if self.meep_sim is None:
            return

        # Get field arrays from MEEP
        # MEEP stores fields differently, need to interpolate to Yee grid
        # This is a simplified placeholder - full implementation would:
        # 1. Get field slices from MEEP at appropriate times
        # 2. Interpolate to Prismo Yee grid positions
        # 3. Handle field component staggering

        # For now, this is a placeholder
        # Full implementation requires understanding MEEP's field storage
        pass

    def run(self, total_time: float, callback: Optional[callable] = None) -> None:
        """
        Run MEEP simulation for specified time.

        Parameters
        ----------
        total_time : float
            Total simulation time in seconds.
        callback : callable, optional
            Function called after each time step.
        """
        if not MEEP_AVAILABLE:
            raise ImportError("MEEP is required for MEEPSolver")

        if self.meep_sim is None:
            self._setup_meep_simulation()

        dt = self.get_time_step()
        num_steps = int(np.ceil(total_time / dt))

        for step in range(num_steps):
            self.step()

            if callback is not None:
                callback(self, step)

    def run_steps(self, num_steps: int, callback: Optional[callable] = None) -> None:
        """
        Run MEEP simulation for specified number of steps.

        Parameters
        ----------
        num_steps : int
            Number of time steps to run.
        callback : callable, optional
            Function called after each time step.
        """
        if not MEEP_AVAILABLE:
            raise ImportError("MEEP is required for MEEPSolver")

        if self.meep_sim is None:
            self._setup_meep_simulation()

        for step in range(num_steps):
            self.step()

            if callback is not None:
                callback(self, step)

    def get_fields(self) -> ElectromagneticFields:
        """Get the current electromagnetic fields."""
        # Extract fields from MEEP if needed
        self._extract_fields_to_prismo(self.fields)
        return self.fields

    def reset(self) -> None:
        """Reset simulation to initial state."""
        if self.meep_sim is not None:
            # MEEP doesn't have a direct reset, need to recreate simulation
            self._setup_meep_simulation()

        self.fields.zero_fields()
        self.time = 0.0
        self.step_count = 0

    def get_time_step(self) -> float:
        """
        Get the time step used by MEEP.

        Returns
        -------
        float
            Time step in seconds.
        """
        # MEEP calculates dt automatically based on grid
        # We use the same calculation as our grid
        return self.dt

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MEEPSolver(t={self.time:.2e}s, steps={self.step_count}, "
            f"grid={self.grid.dimensions}, MEEP={MEEP_AVAILABLE})"
        )
