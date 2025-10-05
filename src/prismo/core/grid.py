"""
Yee grid implementation for FDTD simulations.

This module implements the Yee grid, which is the fundamental discretization
scheme for FDTD electromagnetic simulations. The Yee grid staggers the electric
and magnetic field components in space to ensure second-order accuracy.
"""

from typing import Tuple, Union, Optional, Literal
import numpy as np
from dataclasses import dataclass


@dataclass
class GridSpec:
    """
    Specification for a Yee grid.

    Parameters
    ----------
    size : tuple of float
        Physical size of the simulation domain (Lx, Ly, Lz) in meters.
        For 2D simulations, set Lz=0.
    resolution : float or tuple of float
        Grid resolution in points per meter. If scalar, same resolution
        is used in all dimensions. If tuple, (res_x, res_y, res_z).
    boundary_layers : int, optional
        Number of boundary layers for PML, default=10.
    """

    size: Tuple[float, float, float]
    resolution: Union[float, Tuple[float, float, float]]
    boundary_layers: int = 10

    def __post_init__(self):
        """Validate and process grid specification."""
        # Ensure size is positive
        if any(s < 0 for s in self.size):
            raise ValueError("Grid size components must be non-negative")

        # Process resolution
        if isinstance(self.resolution, (int, float)):
            self.resolution = (self.resolution, self.resolution, self.resolution)
        elif len(self.resolution) != 3:
            raise ValueError("Resolution must be scalar or 3-tuple")

        # Validate resolution
        if any(r <= 0 for r in self.resolution):
            raise ValueError("Resolution must be positive")


class YeeGrid:
    """
    Yee grid for FDTD simulations.

    The Yee grid staggers electric and magnetic field components in space
    and time to achieve second-order accuracy in the FDTD method.

    Field component locations on the Yee cell:
    - Ex: faces perpendicular to x-axis (center of y-z faces)
    - Ey: faces perpendicular to y-axis (center of x-z faces)
    - Ez: faces perpendicular to z-axis (center of x-y faces)
    - Hx: edges parallel to x-axis (center of cell edges)
    - Hy: edges parallel to y-axis (center of cell edges)
    - Hz: edges parallel to z-axis (center of cell edges)

    Parameters
    ----------
    spec : GridSpec
        Grid specification containing size, resolution, and boundary info.
    """

    def __init__(self, spec: GridSpec):
        self.spec = spec
        self._setup_grid()

    def _setup_grid(self) -> None:
        """Set up the grid dimensions and spacing."""
        # Physical domain size
        self.Lx, self.Ly, self.Lz = self.spec.size

        # Grid resolution
        self.res_x, self.res_y, self.res_z = self.spec.resolution

        # Calculate grid spacing (dx = 1/resolution)
        self.dx = 1.0 / self.res_x if self.res_x > 0 else 0.0
        self.dy = 1.0 / self.res_y if self.res_y > 0 else 0.0
        self.dz = 1.0 / self.res_z if self.res_z > 0 else 0.0

        # Determine simulation dimensionality
        self.is_2d = self.Lz == 0.0
        self.is_3d = not self.is_2d

        # Calculate number of grid points
        self.Nx = int(np.ceil(self.Lx * self.res_x)) if self.Lx > 0 else 1
        self.Ny = int(np.ceil(self.Ly * self.res_y)) if self.Ly > 0 else 1
        self.Nz = (
            int(np.ceil(self.Lz * self.res_z)) if self.Lz > 0 and not self.is_2d else 1
        )

        # For 2D simulations, force Nz = 1
        if self.is_2d:
            self.Nz = 1
            self.dz = 0.0

        # Add boundary layers
        self.pml_layers = self.spec.boundary_layers
        self.Nx_total = self.Nx + 2 * self.pml_layers
        self.Ny_total = self.Ny + 2 * self.pml_layers
        self.Nz_total = self.Nz + (2 * self.pml_layers if not self.is_2d else 0)

        # Grid origin (bottom-left-back corner)
        self.origin = np.array(
            [
                -self.pml_layers * self.dx,
                -self.pml_layers * self.dy,
                -self.pml_layers * self.dz if not self.is_2d else 0.0,
            ]
        )

    @property
    def dimensions(self) -> Tuple[int, int, int]:
        """Total grid dimensions including PML layers."""
        return (self.Nx_total, self.Ny_total, self.Nz_total)

    @property
    def physical_dimensions(self) -> Tuple[int, int, int]:
        """Physical grid dimensions (excluding PML layers)."""
        return (self.Nx, self.Ny, self.Nz)

    @property
    def spacing(self) -> Tuple[float, float, float]:
        """Grid spacing in each dimension."""
        return (self.dx, self.dy, self.dz)

    def get_field_shape(
        self, component: Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    ) -> Tuple[int, ...]:
        """
        Get the array shape for a specific field component.

        Parameters
        ----------
        component : str
            Field component name ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz').

        Returns
        -------
        tuple of int
            Array shape for the field component.
        """
        # Base dimensions
        nx, ny, nz = self.dimensions

        # Yee grid staggering - each component has slightly different dimensions
        if component == "Ex":
            shape = (nx, ny - 1, nz - 1) if not self.is_2d else (nx, ny - 1)
        elif component == "Ey":
            shape = (nx - 1, ny, nz - 1) if not self.is_2d else (nx - 1, ny)
        elif component == "Ez":
            shape = (nx - 1, ny - 1, nz) if not self.is_2d else (nx - 1, ny - 1)
        elif component == "Hx":
            shape = (nx - 1, ny, nz) if not self.is_2d else (nx - 1, ny)
        elif component == "Hy":
            shape = (nx, ny - 1, nz) if not self.is_2d else (nx, ny - 1)
        elif component == "Hz":
            shape = (nx, ny, nz - 1) if not self.is_2d else (nx, ny)
        else:
            raise ValueError(f"Unknown field component: {component}")

        # For 2D, ensure we return 2D shapes
        if self.is_2d and len(shape) == 3:
            shape = shape[:2]

        return shape

    def get_coordinates(
        self, component: Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    ) -> Tuple[np.ndarray, ...]:
        """
        Get coordinate arrays for a specific field component.

        Parameters
        ----------
        component : str
            Field component name.

        Returns
        -------
        tuple of ndarray
            Coordinate arrays (x, y) for 2D or (x, y, z) for 3D.
        """
        # Generate base coordinate arrays
        x = self.origin[0] + np.arange(self.Nx_total) * self.dx
        y = self.origin[1] + np.arange(self.Ny_total) * self.dy
        z = self.origin[2] + np.arange(self.Nz_total) * self.dz

        # Apply Yee grid offsets (field components are offset by half grid spacing)
        if component in ["Ey", "Ez", "Hy", "Hz"]:
            x = x[:-1] + self.dx / 2  # Offset in x
        if component in ["Ex", "Ez", "Hx", "Hz"]:
            y = y[:-1] + self.dy / 2  # Offset in y
        if component in ["Ex", "Ey", "Hx", "Hy"] and not self.is_2d:
            z = z[:-1] + self.dz / 2  # Offset in z

        if self.is_2d:
            return x, y
        else:
            return x, y, z

    def is_inside_pml(self, i: int, j: int, k: int = 0) -> bool:
        """
        Check if grid indices are inside PML boundary layers.

        Parameters
        ----------
        i, j, k : int
            Grid indices.

        Returns
        -------
        bool
            True if inside PML region.
        """
        pml = self.pml_layers

        # Check x boundaries
        if i < pml or i >= self.Nx_total - pml:
            return True

        # Check y boundaries
        if j < pml or j >= self.Ny_total - pml:
            return True

        # Check z boundaries (only for 3D)
        if not self.is_2d and (k < pml or k >= self.Nz_total - pml):
            return True

        return False

    def get_physical_indices(self) -> Tuple[slice, slice, slice]:
        """
        Get slices for the physical domain (excluding PML layers).

        Returns
        -------
        tuple of slice
            Slices for (x, y, z) physical domain.
        """
        pml = self.pml_layers
        x_slice = slice(pml, self.Nx_total - pml)
        y_slice = slice(pml, self.Ny_total - pml)
        z_slice = slice(pml, self.Nz_total - pml) if not self.is_2d else slice(None)

        return x_slice, y_slice, z_slice

    def get_courant_number(self, dt: float) -> float:
        """
        Calculate the Courant number for numerical stability.

        The Courant number S = c * dt / dx should be < 1 for stability.
        For 3D: S = c * dt * sqrt(1/dx² + 1/dy² + 1/dz²)
        For 2D: S = c * dt * sqrt(1/dx² + 1/dy²)

        Parameters
        ----------
        dt : float
            Time step in seconds.

        Returns
        -------
        float
            Courant number.
        """
        c = 299792458.0  # Speed of light in vacuum (m/s)

        if self.is_2d:
            inv_dx_sq = (1 / self.dx) ** 2 + (1 / self.dy) ** 2
        else:
            inv_dx_sq = (1 / self.dx) ** 2 + (1 / self.dy) ** 2 + (1 / self.dz) ** 2

        courant = c * dt * np.sqrt(inv_dx_sq)
        return courant

    def get_time_step(self, safety_factor: float = 0.9) -> float:
        """
        Calculate a stable time step based on the Courant condition.

        Alias for suggest_time_step for backward compatibility.

        Parameters
        ----------
        safety_factor : float, optional
            Safety factor (< 1) for stability margin, default=0.9.

        Returns
        -------
        float
            Stable time step in seconds.
        """
        return self.suggest_time_step(safety_factor)

    def suggest_time_step(self, safety_factor: float = 0.9) -> float:
        """
        Suggest a stable time step based on Courant condition.

        Parameters
        ----------
        safety_factor : float, optional
            Safety factor (< 1) for stability margin, default=0.9.

        Returns
        -------
        float
            Suggested time step in seconds.
        """
        c = 299792458.0  # Speed of light in vacuum (m/s)

        if self.is_2d:
            inv_dx_sq = (1 / self.dx) ** 2 + (1 / self.dy) ** 2
        else:
            inv_dx_sq = (1 / self.dx) ** 2 + (1 / self.dy) ** 2 + (1 / self.dz) ** 2

        dt_max = safety_factor / (c * np.sqrt(inv_dx_sq))
        return dt_max

    def point_to_index(self, point: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """
        Convert a physical point to grid indices.

        Parameters
        ----------
        point : tuple of float
            Physical coordinates (x, y, z) in meters.

        Returns
        -------
        tuple of int
            Grid indices (i, j, k).
        """
        x, y, z = point

        # Calculate indices
        i = int(round((x - self.origin[0]) / self.dx))
        j = int(round((y - self.origin[1]) / self.dy))
        k = int(round((z - self.origin[2]) / self.dz)) if self.is_3d else 0

        # Clamp to grid dimensions
        i = max(0, min(i, self.Nx - 1))
        j = max(0, min(j, self.Ny - 1))
        k = max(0, min(k, self.Nz - 1))

        return (i, j, k)

    def index_to_coord(
        self, dim: int, indices: Union[int, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Convert grid indices to physical coordinates along a specific dimension.

        Parameters
        ----------
        dim : int
            Dimension to convert (0=x, 1=y, 2=z).
        indices : int or ndarray
            Grid indices to convert.

        Returns
        -------
        float or ndarray
            Physical coordinates in meters.
        """
        if dim == 0:
            return self.origin[0] + indices * self.dx
        elif dim == 1:
            return self.origin[1] + indices * self.dy
        elif dim == 2:
            return self.origin[2] + indices * self.dz
        else:
            raise ValueError(f"Invalid dimension: {dim}. Must be 0, 1, or 2.")

    def get_component_indices(
        self,
        component: str,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        z_min: int,
        z_max: int,
    ) -> tuple:
        """
        Get the grid indices for a field component within a specified region.

        Parameters
        ----------
        component : str
            Field component ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz").
        x_min, x_max, y_min, y_max, z_min, z_max : int
            Grid index bounds for the region.

        Returns
        -------
        tuple
            Tuple of slices or indices for the component grid.
        """
        # Ensure bounds are within grid dimensions
        x_min = max(0, min(x_min, self.Nx - 1))
        x_max = max(1, min(x_max, self.Nx))
        y_min = max(0, min(y_min, self.Ny - 1))
        y_max = max(1, min(y_max, self.Ny))

        if self.is_3d:
            z_min = max(0, min(z_min, self.Nz - 1))
            z_max = max(1, min(z_max, self.Nz))
        else:
            z_min, z_max = 0, 1

        # Get the field shape for the component
        shape = self.get_field_shape(component)

        # Create indices based on component position
        # Field components may be staggered by half a cell
        # For 2D grids, shape only has 2 dimensions
        if self.is_3d:
            if component == "Ex":
                # Ex has indices (i, j+1/2, k+1/2)
                return np.ix_(
                    range(x_min, x_max),
                    range(y_min, min(y_max, shape[1])),
                    range(z_min, min(z_max, shape[2])),
                )
            elif component == "Ey":
                # Ey has indices (i+1/2, j, k+1/2)
                return np.ix_(
                    range(x_min, min(x_max, shape[0])),
                    range(y_min, y_max),
                    range(z_min, min(z_max, shape[2])),
                )
            elif component == "Ez":
                # Ez has indices (i+1/2, j+1/2, k)
                return np.ix_(
                    range(x_min, min(x_max, shape[0])),
                    range(y_min, min(y_max, shape[1])),
                    range(z_min, z_max),
                )
            elif component == "Hx":
                # Hx has indices (i+1/2, j, k)
                return np.ix_(
                    range(x_min, min(x_max, shape[0])),
                    range(y_min, y_max),
                    range(z_min, z_max),
                )
            elif component == "Hy":
                # Hy has indices (i, j+1/2, k)
                return np.ix_(
                    range(x_min, x_max),
                    range(y_min, min(y_max, shape[1])),
                    range(z_min, z_max),
                )
            elif component == "Hz":
                # Hz has indices (i, j, k+1/2)
                return np.ix_(
                    range(x_min, x_max),
                    range(y_min, y_max),
                    range(z_min, min(z_max, shape[2])),
                )
            else:
                raise ValueError(f"Invalid field component: {component}")
        else:
            # 2D case - shape only has 2 dimensions
            if component == "Ex":
                # Ex has indices (i, j+1/2)
                return np.ix_(
                    range(x_min, x_max),
                    range(y_min, min(y_max, shape[1])),
                )
            elif component == "Ey":
                # Ey has indices (i+1/2, j)
                return np.ix_(
                    range(x_min, min(x_max, shape[0])),
                    range(y_min, y_max),
                )
            elif component == "Ez":
                # Ez has indices (i+1/2, j+1/2)
                return np.ix_(
                    range(x_min, min(x_max, shape[0])),
                    range(y_min, min(y_max, shape[1])),
                )
            elif component == "Hx":
                # Hx has indices (i+1/2, j)
                return np.ix_(
                    range(x_min, min(x_max, shape[0])),
                    range(y_min, y_max),
                )
            elif component == "Hy":
                # Hy has indices (i, j+1/2)
                return np.ix_(
                    range(x_min, x_max),
                    range(y_min, min(y_max, shape[1])),
                )
            elif component == "Hz":
                # Hz has indices (i, j)
                return np.ix_(
                    range(x_min, x_max),
                    range(y_min, y_max),
                )
            else:
                raise ValueError(f"Invalid field component: {component}")

        # Should not reach here
        raise ValueError(f"Invalid field component: {component}")

    def __repr__(self) -> str:
        """String representation of the grid."""
        dim_str = "2D" if self.is_2d else "3D"
        return (
            f"YeeGrid({dim_str}, shape={self.dimensions}, "
            f"spacing=({self.dx:.2e}, {self.dy:.2e}, {self.dz:.2e}), "
            f"PML={self.pml_layers})"
        )
