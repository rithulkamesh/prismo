"""
Electromagnetic field storage and manipulation for FDTD simulations.

This module implements the Fields class which stores and manipulates the
electromagnetic field components (Ex, Ey, Ez, Hx, Hy, Hz) on the Yee grid.
"""

from typing import Tuple, Optional, Union, Literal, Dict, Any
import numpy as np
from .grid import YeeGrid
from prismo.backends import Backend, get_backend


FieldComponent = Literal["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
FieldType = Literal["E", "H"]


class ElectromagneticFields:
    """
    Storage and manipulation of electromagnetic fields on a Yee grid.

    This class manages the six electromagnetic field components (Ex, Ey, Ez, Hx, Hy, Hz)
    with proper array shapes for the staggered Yee grid. It provides methods for
    field initialization, access, and manipulation.

    Parameters
    ----------
    grid : YeeGrid
        The Yee grid on which fields are defined.
    dtype : numpy.dtype, optional
        Data type for field arrays, default=np.float64.
    """

    def __init__(
        self,
        grid: YeeGrid,
        dtype: Any = None,
        backend: Optional[Union[str, Backend]] = None,
    ):
        self.grid = grid

        # Initialize backend
        if isinstance(backend, str):
            self.backend = get_backend(backend)
        elif isinstance(backend, Backend):
            self.backend = backend
        elif backend is None:
            self.backend = get_backend()  # Auto-select
        else:
            raise TypeError("backend must be a Backend instance or string name")

        # Set dtype
        if dtype is None:
            self.dtype = self.backend.float64
        else:
            self.dtype = dtype

        # Initialize field component arrays
        self._fields: Dict[FieldComponent, Any] = {}
        self._initialize_fields()

    def _initialize_fields(self) -> None:
        """Initialize all electromagnetic field component arrays."""
        components = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

        for component in components:
            shape = self.grid.get_field_shape(component)
            self._fields[component] = self.backend.zeros(shape, dtype=self.dtype)

    def __getitem__(self, component: FieldComponent) -> Any:
        """
        Access field component by name.

        Parameters
        ----------
        component : str
            Field component name ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz').

        Returns
        -------
        array
            Field component array (backend-specific).
        """
        if component not in self._fields:
            raise KeyError(f"Unknown field component: {component}")
        return self._fields[component]

    def __setitem__(self, component: FieldComponent, value: Union[Any, float]) -> None:
        """
        Set field component values.

        Parameters
        ----------
        component : str
            Field component name.
        value : array or float
            New field values. If scalar, all elements are set to this value.
        """
        if component not in self._fields:
            raise KeyError(f"Unknown field component: {component}")

        if np.isscalar(value):
            self._fields[component].fill(value)
        else:
            value_arr = self.backend.asarray(value)
            if value_arr.shape != self._fields[component].shape:
                raise ValueError(
                    f"Shape mismatch for {component}: "
                    f"expected {self._fields[component].shape}, "
                    f"got {value_arr.shape}"
                )
            self._fields[component][:] = value_arr

    @property
    def Ex(self) -> np.ndarray:
        """Electric field x-component."""
        return self._fields["Ex"]

    @property
    def Ey(self) -> np.ndarray:
        """Electric field y-component."""
        return self._fields["Ey"]

    @property
    def Ez(self) -> np.ndarray:
        """Electric field z-component."""
        return self._fields["Ez"]

    @property
    def Hx(self) -> np.ndarray:
        """Magnetic field x-component."""
        return self._fields["Hx"]

    @property
    def Hy(self) -> np.ndarray:
        """Magnetic field y-component."""
        return self._fields["Hy"]

    @property
    def Hz(self) -> np.ndarray:
        """Magnetic field z-component."""
        return self._fields["Hz"]

    def get_electric_field_components(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all electric field components.

        Returns
        -------
        tuple of ndarray
            (Ex, Ey, Ez) arrays.
        """
        return self.Ex, self.Ey, self.Ez

    def get_magnetic_field_components(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all magnetic field components.

        Returns
        -------
        tuple of ndarray
            (Hx, Hy, Hz) arrays.
        """
        return self.Hx, self.Hy, self.Hz

    def get_field_type_components(
        self, field_type: FieldType
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all components of a specific field type.

        Parameters
        ----------
        field_type : str
            Field type ('E' or 'H').

        Returns
        -------
        tuple of ndarray
            Field components (x, y, z).
        """
        if field_type == "E":
            return self.get_electric_field_components()
        elif field_type == "H":
            return self.get_magnetic_field_components()
        else:
            raise ValueError(f"Unknown field type: {field_type}")

    def zero_fields(self, field_type: Optional[FieldType] = None) -> None:
        """
        Zero out field components.

        Parameters
        ----------
        field_type : str, optional
            Field type to zero ('E', 'H', or None for all fields).
        """
        if field_type is None:
            components = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
        elif field_type == "E":
            components = ["Ex", "Ey", "Ez"]
        elif field_type == "H":
            components = ["Hx", "Hy", "Hz"]
        else:
            raise ValueError(f"Unknown field type: {field_type}")

        for component in components:
            self._fields[component].fill(0.0)

    def copy_fields_from(self, other: "ElectromagneticFields") -> None:
        """
        Copy field values from another Fields object.

        Parameters
        ----------
        other : ElectromagneticFields
            Source fields to copy from.
        """
        if not isinstance(other, ElectromagneticFields):
            raise TypeError("Can only copy from another ElectromagneticFields object")

        # Check grid compatibility
        if self.grid.dimensions != other.grid.dimensions:
            raise ValueError("Grid dimensions must match for field copying")

        for component in self._fields:
            if self._fields[component].shape != other._fields[component].shape:
                raise ValueError(f"Shape mismatch for component {component}")
            self._fields[component][:] = other._fields[component]

    def get_field_energy(
        self,
        field_type: Optional[FieldType] = None,
        region: Optional[Tuple[slice, ...]] = None,
    ) -> float:
        """
        Calculate field energy in a specified region.

        Energy density: u = (1/2) * (ε₀|E|² + μ₀|H|²)

        Parameters
        ----------
        field_type : str, optional
            Field type ('E', 'H', or None for total energy).
        region : tuple of slice, optional
            Spatial region to calculate energy over. If None, uses entire domain.

        Returns
        -------
        float
            Field energy in Joules.
        """
        # Physical constants
        eps0 = 8.854187817e-12  # Vacuum permittivity (F/m)
        mu0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)

        # Get grid spacing for volume element
        dx, dy, dz = self.grid.spacing
        dV = dx * dy * dz if not self.grid.is_2d else dx * dy

        total_energy = 0.0

        if field_type is None or field_type == "E":
            # Electric field energy
            for component in ["Ex", "Ey", "Ez"]:
                field = self._fields[component]
                if region is not None:
                    field = field[region]
                total_energy += 0.5 * eps0 * np.sum(field**2) * dV

        if field_type is None or field_type == "H":
            # Magnetic field energy
            for component in ["Hx", "Hy", "Hz"]:
                field = self._fields[component]
                if region is not None:
                    field = field[region]
                total_energy += 0.5 * mu0 * np.sum(field**2) * dV

        return total_energy

    def get_field_magnitude(
        self, field_type: FieldType, region: Optional[Tuple[slice, ...]] = None
    ) -> np.ndarray:
        """
        Calculate field magnitude |F| = sqrt(Fx² + Fy² + Fz²).

        Parameters
        ----------
        field_type : str
            Field type ('E' or 'H').
        region : tuple of slice, optional
            Spatial region. If None, uses entire domain.

        Returns
        -------
        ndarray
            Field magnitude array.
        """
        fx, fy, fz = self.get_field_type_components(field_type)

        if region is not None:
            fx = fx[region]
            fy = fy[region]
            fz = fz[region]

        # For proper magnitude calculation, we need to interpolate to common grid points
        # This is a simplified version - for production use, proper interpolation is needed
        min_shape = tuple(min(s) for s in zip(fx.shape, fy.shape, fz.shape))

        fx_interp = (
            fx[: min_shape[0], : min_shape[1]]
            if len(min_shape) >= 2
            else fx[: min_shape[0]]
        )
        fy_interp = (
            fy[: min_shape[0], : min_shape[1]]
            if len(min_shape) >= 2
            else fy[: min_shape[0]]
        )
        fz_interp = (
            fz[: min_shape[0], : min_shape[1]]
            if len(min_shape) >= 2
            else fz[: min_shape[0]]
        )

        if len(min_shape) == 3:
            fx_interp = fx_interp[:, :, : min_shape[2]]
            fy_interp = fy_interp[:, :, : min_shape[2]]
            fz_interp = fz_interp[:, :, : min_shape[2]]

        magnitude = np.sqrt(fx_interp**2 + fy_interp**2 + fz_interp**2)
        return magnitude

    def apply_boundary_conditions(self, boundary_type: str = "pec") -> None:
        """
        Apply boundary conditions to fields.

        Parameters
        ----------
        boundary_type : str
            Type of boundary condition ('pec' for perfect electric conductor).
        """
        if boundary_type.lower() == "pec":
            # Perfect Electric Conductor: tangential E = 0 at boundaries
            # This is a simplified implementation

            # Zero tangential E fields at x boundaries
            self.Ey[0, :] = 0  # Left boundary
            self.Ey[-1, :] = 0  # Right boundary
            self.Ez[0, :] = 0
            self.Ez[-1, :] = 0

            # Zero tangential E fields at y boundaries
            self.Ex[:, 0] = 0  # Bottom boundary
            self.Ex[:, -1] = 0  # Top boundary
            self.Ez[:, 0] = 0
            self.Ez[:, -1] = 0

            # For 3D, also handle z boundaries
            if not self.grid.is_2d:
                self.Ex[:, :, 0] = 0
                self.Ex[:, :, -1] = 0
                self.Ey[:, :, 0] = 0
                self.Ey[:, :, -1] = 0
        else:
            raise NotImplementedError(
                f"Boundary condition '{boundary_type}' not implemented"
            )

    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage information for the field arrays.

        Returns
        -------
        dict
            Memory usage information.
        """
        total_bytes = 0
        component_info = {}

        for component, field in self._fields.items():
            bytes_used = field.nbytes
            total_bytes += bytes_used
            component_info[component] = {
                "shape": field.shape,
                "dtype": str(field.dtype),
                "bytes": bytes_used,
                "megabytes": bytes_used / (1024**2),
            }

        return {
            "total_bytes": total_bytes,
            "total_megabytes": total_bytes / (1024**2),
            "total_gigabytes": total_bytes / (1024**3),
            "components": component_info,
        }

    def __repr__(self) -> str:
        """String representation of the fields."""
        mem_info = self.get_memory_usage()
        return (
            f"ElectromagneticFields(grid={self.grid.dimensions}, "
            f"dtype={self.dtype}, memory={mem_info['total_megabytes']:.1f}MB)"
        )

    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.__repr__()
