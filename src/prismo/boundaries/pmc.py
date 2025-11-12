"""
Perfect Magnetic Conductor (PMC) boundary conditions.

This module implements PMC boundary conditions where the tangential component
of the magnetic field is zero at the boundary. This is the magnetic dual of
Perfect Electric Conductor (PEC) boundaries.

PMC enforces: n × H = 0 at boundaries (tangential H = 0)

For the Yee grid:
- Hx is zero at y and z boundaries
- Hy is zero at x and z boundaries
- Hz is zero at x and y boundaries
"""

from typing import Literal, Optional, Union

import numpy as np

from prismo.backends import Backend, get_backend
from prismo.core.fields import ElectromagneticFields
from prismo.core.grid import YeeGrid


class PMC:
    """
    Perfect Magnetic Conductor (PMC) boundary condition.

    PMC boundaries enforce that the tangential component of the magnetic field
    is zero, effectively creating a perfect magnetic wall that reflects
    magnetic fields.

    Parameters
    ----------
    grid : YeeGrid
        The simulation grid.
    faces : list of str, optional
        List of faces to apply PMC to. Options: 'x_min', 'x_max', 'y_min',
        'y_max', 'z_min', 'z_max'. If None, applies to all faces.
    backend : Backend, optional
        Computational backend to use.
    """

    def __init__(
        self,
        grid: YeeGrid,
        faces: Optional[list[str]] = None,
        backend: Optional[Union[str, Backend]] = None,
    ):
        self.grid = grid

        # Initialize backend
        if isinstance(backend, str):
            self.backend = get_backend(backend)
        elif isinstance(backend, Backend):
            self.backend = backend
        elif backend is None:
            self.backend = get_backend()
        else:
            raise TypeError("backend must be a Backend instance or string name")

        # Default to all faces if not specified
        if faces is None:
            faces = ["x_min", "x_max", "y_min", "y_max"]
            if grid.is_3d:
                faces.extend(["z_min", "z_max"])

        self.faces = faces
        self._validate_faces()

    def _validate_faces(self) -> None:
        """Validate that specified faces are valid."""
        valid_faces = ["x_min", "x_max", "y_min", "y_max"]
        if self.grid.is_3d:
            valid_faces.extend(["z_min", "z_max"])

        for face in self.faces:
            if face not in valid_faces:
                raise ValueError(f"Invalid face '{face}'. Valid faces: {valid_faces}")

    def apply(self, fields: ElectromagneticFields) -> None:
        """
        Apply PMC boundary conditions to magnetic fields.

        Parameters
        ----------
        fields : ElectromagneticFields
            Electromagnetic fields to apply PMC to.
        """
        nx, ny, nz = self.grid.dimensions

        # Apply PMC at x boundaries (x_min and x_max)
        # At x boundaries, tangential H fields are Hy and Hz
        if "x_min" in self.faces:
            # Left boundary (x = 0): zero Hy and Hz
            if self.grid.is_2d:
                # 2D: Hy is at (i, j+1/2), zero at i=0
                if fields.Hy.shape[0] > 0:
                    fields.Hy[0, :] = 0
            else:
                # 3D: Hy is at (i, j+1/2, k), zero at i=0
                if fields.Hy.shape[0] > 0:
                    fields.Hy[0, :, :] = 0
                # Hz is at (i, j, k+1/2), zero at i=0
                if fields.Hz.shape[0] > 0:
                    fields.Hz[0, :, :] = 0

        if "x_max" in self.faces:
            # Right boundary (x = nx-1): zero Hy and Hz
            if self.grid.is_2d:
                if fields.Hy.shape[0] > 0:
                    fields.Hy[-1, :] = 0
            else:
                if fields.Hy.shape[0] > 0:
                    fields.Hy[-1, :, :] = 0
                if fields.Hz.shape[0] > 0:
                    fields.Hz[-1, :, :] = 0

        # Apply PMC at y boundaries (y_min and y_max)
        # At y boundaries, tangential H fields are Hx and Hz
        if "y_min" in self.faces:
            # Bottom boundary (y = 0): zero Hx and Hz
            if self.grid.is_2d:
                # 2D: Hx is at (i+1/2, j), zero at j=0
                if fields.Hx.shape[1] > 0:
                    fields.Hx[:, 0] = 0
            else:
                # 3D: Hx is at (i+1/2, j, k), zero at j=0
                if fields.Hx.shape[1] > 0:
                    fields.Hx[:, 0, :] = 0
                # Hz is at (i, j, k+1/2), zero at j=0
                if fields.Hz.shape[1] > 0:
                    fields.Hz[:, 0, :] = 0

        if "y_max" in self.faces:
            # Top boundary (y = ny-1): zero Hx and Hz
            if self.grid.is_2d:
                if fields.Hx.shape[1] > 0:
                    fields.Hx[:, -1] = 0
            else:
                if fields.Hx.shape[1] > 0:
                    fields.Hx[:, -1, :] = 0
                if fields.Hz.shape[1] > 0:
                    fields.Hz[:, -1, :] = 0

        # Apply PMC at z boundaries (z_min and z_max) - only for 3D
        if self.grid.is_3d:
            # At z boundaries, tangential H fields are Hx and Hy
            if "z_min" in self.faces:
                # Back boundary (z = 0): zero Hx and Hy
                if fields.Hx.shape[2] > 0:
                    fields.Hx[:, :, 0] = 0
                if fields.Hy.shape[2] > 0:
                    fields.Hy[:, :, 0] = 0

            if "z_max" in self.faces:
                # Front boundary (z = nz-1): zero Hx and Hy
                if fields.Hx.shape[2] > 0:
                    fields.Hx[:, :, -1] = 0
                if fields.Hy.shape[2] > 0:
                    fields.Hy[:, :, -1] = 0

    def __repr__(self) -> str:
        """String representation."""
        faces_str = ", ".join(self.faces)
        return f"PMC(faces=[{faces_str}], grid={self.grid.dimensions})"
