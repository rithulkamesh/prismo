"""
Abstract base class for electromagnetic solvers.

This module defines the unified interface that all solvers (FDTD, FEM, MEEP)
must implement, allowing solver-agnostic simulation orchestration.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from prismo.core.fields import ElectromagneticFields
from prismo.core.grid import YeeGrid


class SolverBase(ABC):
    """
    Abstract base class for all electromagnetic solvers.

    This class defines the unified interface that all solvers must implement,
    allowing the Simulation class to work with any solver type transparently.
    """

    def __init__(self, grid: YeeGrid):
        """
        Initialize the solver.

        Parameters
        ----------
        grid : YeeGrid
            The simulation grid.
        """
        self.grid = grid
        self.time = 0.0
        self.step_count = 0

    @abstractmethod
    def step(self, fields: Optional[ElectromagneticFields] = None) -> None:
        """
        Perform a single time step.

        Parameters
        ----------
        fields : ElectromagneticFields, optional
            Fields to update. If None, uses internal fields.
        """
        pass

    @abstractmethod
    def get_fields(self) -> ElectromagneticFields:
        """
        Get the current electromagnetic fields.

        Returns
        -------
        ElectromagneticFields
            Current field state.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset simulation to initial state."""
        pass

    @abstractmethod
    def get_time_step(self) -> float:
        """
        Get the time step used by the solver.

        Returns
        -------
        float
            Time step in seconds.
        """
        pass

    def get_simulation_info(self) -> dict:
        """
        Get information about the current simulation state.

        Returns
        -------
        dict
            Dictionary with simulation information.
        """
        return {
            "time": self.time,
            "step_count": self.step_count,
            "dt": self.get_time_step(),
            "grid_dimensions": self.grid.dimensions,
            "is_2d": self.grid.is_2d,
        }

    def __repr__(self) -> str:
        """String representation."""
        info = self.get_simulation_info()
        return (
            f"{self.__class__.__name__}(t={info['time']:.2e}s, "
            f"steps={info['step_count']})"
        )


class TimeDomainSolver(SolverBase):
    """
    Base class for time-domain solvers (FDTD, MEEP).

    Time-domain solvers advance fields in time using time-stepping.
    """

    @abstractmethod
    def run(self, total_time: float, callback: Optional[callable] = None) -> None:
        """
        Run simulation for specified time.

        Parameters
        ----------
        total_time : float
            Total simulation time in seconds.
        callback : callable, optional
            Function called after each time step.
        """
        pass

    @abstractmethod
    def run_steps(self, num_steps: int, callback: Optional[callable] = None) -> None:
        """
        Run simulation for specified number of steps.

        Parameters
        ----------
        num_steps : int
            Number of time steps to run.
        callback : callable, optional
            Function called after each time step.
        """
        pass


class FrequencyDomainSolver(SolverBase):
    """
    Base class for frequency-domain solvers (FEM).

    Frequency-domain solvers solve for fields at specific frequencies.
    """

    @abstractmethod
    def solve(self, frequency: float) -> ElectromagneticFields:
        """
        Solve for fields at a specific frequency.

        Parameters
        ----------
        frequency : float
            Frequency in Hz.

        Returns
        -------
        ElectromagneticFields
            Solution fields at the specified frequency.
        """
        pass

    @abstractmethod
    def solve_eigenvalue(self, num_modes: int = 1) -> tuple[np.ndarray, list]:
        """
        Solve eigenvalue problem (e.g., for waveguide modes).

        Parameters
        ----------
        num_modes : int
            Number of modes to compute.

        Returns
        -------
        tuple
            (eigenvalues, eigenmodes) where eigenmodes is a list of fields.
        """
        pass
