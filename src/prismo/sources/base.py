"""
Base classes for electromagnetic field sources.

This module defines the base Source class and related utilities for
implementing various types of electromagnetic sources in FDTD simulations.
"""

from typing import Tuple, Dict, Optional, Union, Literal, Protocol
import numpy as np
from abc import ABC, abstractmethod

from prismo.core.grid import YeeGrid
from prismo.core.fields import ElectromagneticFields


class Source(ABC):
    """
    Abstract base class for all electromagnetic sources.

    Parameters
    ----------
    center : Tuple[float, float, float]
        Physical coordinates of the source center (x, y, z) in meters.
    size : Tuple[float, float, float]
        Physical dimensions of the source region (Lx, Ly, Lz) in meters.
        For point sources, use (0, 0, 0).
    name : str, optional
        Name of the source for identification.
    enabled : bool, optional
        Flag to enable/disable the source, default=True.
    """

    def __init__(
        self,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        name: Optional[str] = None,
        enabled: bool = True,
    ):
        self.center = center
        self.size = size
        self.name = name or f"{self.__class__.__name__}_{id(self)}"
        self.enabled = enabled

        # Set during initialization by the simulation
        self._grid: Optional[YeeGrid] = None
        self._source_region: Dict[str, np.ndarray] = {}

    def initialize(self, grid: YeeGrid) -> None:
        """
        Initialize the source on a specific grid.

        Parameters
        ----------
        grid : YeeGrid
            The grid on which to initialize the source.
        """
        self._grid = grid
        self._setup_source_region()

    def _setup_source_region(self) -> None:
        """
        Set up the source region in grid coordinates.

        This method computes the grid indices where the source is applied.
        """
        if self._grid is None:
            raise RuntimeError("Source must be initialized with a grid first")

        # Convert physical coordinates to grid indices
        x_min, x_max, y_min, y_max, z_min, z_max = self._compute_source_region()

        # Store source region indices for each field component
        # Each source will implement how to use these indices
        self._source_region = {
            "Ex": self._grid.get_component_indices(
                "Ex", x_min, x_max, y_min, y_max, z_min, z_max
            ),
            "Ey": self._grid.get_component_indices(
                "Ey", x_min, x_max, y_min, y_max, z_min, z_max
            ),
            "Ez": self._grid.get_component_indices(
                "Ez", x_min, x_max, y_min, y_max, z_min, z_max
            ),
            "Hx": self._grid.get_component_indices(
                "Hx", x_min, x_max, y_min, y_max, z_min, z_max
            ),
            "Hy": self._grid.get_component_indices(
                "Hy", x_min, x_max, y_min, y_max, z_min, z_max
            ),
            "Hz": self._grid.get_component_indices(
                "Hz", x_min, x_max, y_min, y_max, z_min, z_max
            ),
        }

    def _compute_source_region(self) -> Tuple[int, int, int, int, int, int]:
        """
        Compute the grid index bounds for the source region.

        Returns
        -------
        Tuple[int, int, int, int, int, int]
            Grid index bounds (x_min, x_max, y_min, y_max, z_min, z_max).
        """
        if self._grid is None:
            raise RuntimeError("Source must be initialized with a grid first")

        # Get physical half-sizes
        half_Lx = self.size[0] / 2
        half_Ly = self.size[1] / 2
        half_Lz = self.size[2] / 2

        # Convert physical coordinates to grid indices
        x_min, y_min, z_min = self._grid.point_to_index(
            (
                self.center[0] - half_Lx,
                self.center[1] - half_Ly,
                self.center[2] - half_Lz,
            )
        )

        x_max, y_max, z_max = self._grid.point_to_index(
            (
                self.center[0] + half_Lx,
                self.center[1] + half_Ly,
                self.center[2] + half_Lz,
            )
        )

        # Ensure we have at least one cell for point sources
        if x_min == x_max:
            x_max = x_min + 1
        if y_min == y_max:
            y_max = y_min + 1
        if z_min == z_max and self._grid.is_3d:
            z_max = z_min + 1

        return x_min, x_max, y_min, y_max, z_min, z_max

    @abstractmethod
    def update_fields(
        self, fields: ElectromagneticFields, time: float, dt: float
    ) -> None:
        """
        Update electromagnetic fields with source contribution.

        Parameters
        ----------
        fields : ElectromagneticFields
            Electromagnetic fields to update.
        time : float
            Current simulation time in seconds.
        dt : float
            Time step in seconds.
        """
        pass

    def disable(self) -> None:
        """Disable the source."""
        self.enabled = False

    def enable(self) -> None:
        """Enable the source."""
        self.enabled = True
