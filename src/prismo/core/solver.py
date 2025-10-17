"""
Maxwell equation updates for FDTD simulations.

This module implements the core Maxwell equation updates using finite differences
on the Yee grid. The updates follow Faraday's law and Ampère's law:

∂H/∂t = -(1/μ₀) ∇ × E
∂E/∂t = (1/ε₀) ∇ × H

The curl operations are discretized using finite differences on the staggered
Yee grid to maintain second-order accuracy.
"""

from typing import Tuple, Optional, Union
import numpy as np
from .grid import YeeGrid
from .fields import ElectromagneticFields
from prismo.backends import Backend, get_backend


class MaxwellUpdater:
    """
    Core Maxwell equation updater for FDTD simulations.

    This class implements the discrete curl operations and time updates
    for electromagnetic fields on the Yee grid.

    Parameters
    ----------
    grid : YeeGrid
        The computational grid.
    dt : float
        Time step in seconds.
    material_arrays : dict, optional
        Material property arrays (ε, μ, σ). If None, vacuum properties are used.
    """

    def __init__(
        self,
        grid: YeeGrid,
        dt: float,
        material_arrays: Optional[dict] = None,
        backend: Optional[Union[str, Backend]] = None,
    ):
        self.grid = grid
        self.dt = dt

        # Initialize backend
        if isinstance(backend, str):
            self.backend = get_backend(backend)
        elif isinstance(backend, Backend):
            self.backend = backend
        elif backend is None:
            self.backend = get_backend()  # Auto-select
        else:
            raise TypeError("backend must be a Backend instance or string name")

        # Physical constants
        self.eps0 = 8.854187817e-12  # Vacuum permittivity (F/m)
        self.mu0 = 4 * self.backend.pi * 1e-7  # Vacuum permeability (H/m)
        self.c = 299792458.0  # Speed of light (m/s)

        # Validate Courant condition
        courant = self.grid.get_courant_number(dt)
        if courant >= 1.0:
            raise ValueError(
                f"Time step dt={dt:.2e} violates Courant condition (S={courant:.3f} >= 1)"
            )

        # Initialize material arrays (vacuum by default)
        self._initialize_materials(material_arrays)

        # Precompute update coefficients
        self._compute_update_coefficients()

    def _initialize_materials(self, material_arrays: Optional[dict]) -> None:
        """Initialize material property arrays."""
        # Get grid dimensions
        nx, ny, nz = self.grid.dimensions

        if material_arrays is None:
            # Vacuum everywhere
            self.eps_rel = self.backend.ones(
                (nx, ny, nz), dtype=self.backend.float64
            )  # Relative permittivity
            self.mu_rel = self.backend.ones(
                (nx, ny, nz), dtype=self.backend.float64
            )  # Relative permeability
            self.sigma_e = self.backend.zeros(
                (nx, ny, nz), dtype=self.backend.float64
            )  # Electric conductivity
            self.sigma_m = self.backend.zeros(
                (nx, ny, nz), dtype=self.backend.float64
            )  # Magnetic conductivity (usually 0)
        else:
            # Use provided material arrays (convert to backend arrays)
            self.eps_rel = self.backend.asarray(
                material_arrays.get("eps_rel", np.ones((nx, ny, nz)))
            )
            self.mu_rel = self.backend.asarray(
                material_arrays.get("mu_rel", np.ones((nx, ny, nz)))
            )
            self.sigma_e = self.backend.asarray(
                material_arrays.get("sigma_e", np.zeros((nx, ny, nz)))
            )
            self.sigma_m = self.backend.asarray(
                material_arrays.get("sigma_m", np.zeros((nx, ny, nz)))
            )

    def _compute_update_coefficients(self) -> None:
        """Precompute update coefficients for E and H field updates."""
        # For E-field update: E^(n+1) = Ca * E^n + Cb * curl(H^(n+1/2))
        # Ca and Cb account for material properties and conductivity

        # E-field coefficients (with conductivity)
        eps = self.eps0 * self.eps_rel
        sigma_dt_2eps = self.sigma_e * self.dt / (2 * eps)

        self.Ca = (1 - sigma_dt_2eps) / (1 + sigma_dt_2eps)
        self.Cb = (self.dt / eps) / (1 + sigma_dt_2eps)

        # H-field coefficients (usually no magnetic conductivity)
        mu = self.mu0 * self.mu_rel
        sigma_m_dt_2mu = self.sigma_m * self.dt / (2 * mu)

        self.Da = (1 - sigma_m_dt_2mu) / (1 + sigma_m_dt_2mu)
        self.Db = (self.dt / mu) / (1 + sigma_m_dt_2mu)

        # Grid spacing for curl operations
        self.dx, self.dy, self.dz = self.grid.spacing

    def update_magnetic_fields(self, fields: ElectromagneticFields) -> None:
        """
        Update magnetic field components using Faraday's law.

        ∂H/∂t = -(1/μ) ∇ × E

        Parameters
        ----------
        fields : ElectromagneticFields
            Electromagnetic fields to update.
        """
        if self.grid.is_2d:
            self._update_h_fields_2d(fields)
        else:
            self._update_h_fields_3d(fields)

    def update_electric_fields(self, fields: ElectromagneticFields) -> None:
        """
        Update electric field components using Ampère's law.

        ∂E/∂t = (1/ε) ∇ × H

        Parameters
        ----------
        fields : ElectromagneticFields
            Electromagnetic fields to update.
        """
        if self.grid.is_2d:
            self._update_e_fields_2d(fields)
        else:
            self._update_e_fields_3d(fields)

    def _update_h_fields_3d(self, fields: ElectromagneticFields) -> None:
        """Update H-field components for 3D case."""
        Ex, Ey, Ez = fields.Ex, fields.Ey, fields.Ez
        Hx, Hy, Hz = fields.Hx, fields.Hy, fields.Hz

        # Hx update: ∂Hx/∂t = -(1/μ) * (∂Ez/∂y - ∂Ey/∂z)
        # Hx is at (i+1/2, j, k), shape (nx-1, ny, nz)
        # Ez is at (i+1/2, j+1/2, k), shape (nx-1, ny-1, nz)
        # Ey is at (i+1/2, j, k+1/2), shape (nx-1, ny, nz-1)

        # For ∂Ez/∂y: need Ez at j and j-1, both exist for j=0 to ny-1
        curl_ez_y = (Ez[:, 1:, :] - Ez[:, :-1, :]) / self.dy  # Shape: (nx-1, ny-1, nz)
        # For ∂Ey/∂z: need Ey at k and k-1, both exist for k=0 to nz-1
        curl_ey_z = (Ey[:, :, 1:] - Ey[:, :, :-1]) / self.dz  # Shape: (nx-1, ny, nz-1)

        # To match Hx shape (nx-1, ny, nz), we need to:
        # - Take first ny-1 elements of curl_ey_z in y-direction: curl_ey_z[:, :-1, :]
        # - Take first nz-1 elements of curl_ez_y in z-direction: curl_ez_y[:, :, :-1]
        # But actually, curl_ez_y already has shape (nx-1, ny-1, nz)
        # and curl_ey_z has shape (nx-1, ny, nz-1)
        # We need both to have shape (nx-1, ny, nz)
        # So pad or trim appropriately

        # Actually, let's match to the smallest common shape
        ny_common = min(Hx.shape[1], curl_ez_y.shape[1], curl_ey_z.shape[1])
        nz_common = min(Hx.shape[2], curl_ez_y.shape[2], curl_ey_z.shape[2])

        da_hx = self._interpolate_to_hx_points(self.Da)[
            : Hx.shape[0], :ny_common, :nz_common
        ]
        db_hx = self._interpolate_to_hx_points(self.Db)[
            : Hx.shape[0], :ny_common, :nz_common
        ]

        Hx[:, :ny_common, :nz_common] = da_hx * Hx[
            :, :ny_common, :nz_common
        ] - db_hx * (
            curl_ez_y[:, :ny_common, :nz_common] - curl_ey_z[:, :ny_common, :nz_common]
        )

        # Hy update: ∂Hy/∂t = -(1/μ) * (∂Ex/∂z - ∂Ez/∂x)
        # Hy is at (i, j+1/2, k), shape (nx, ny-1, nz)
        # Ex is at (i, j+1/2, k+1/2), shape (nx, ny-1, nz-1)
        # Ez is at (i+1/2, j+1/2, k), shape (nx-1, ny-1, nz)

        curl_ex_z = (Ex[:, :, 1:] - Ex[:, :, :-1]) / self.dz  # Shape: (nx, ny-1, nz-1)
        curl_ez_x = (Ez[1:, :, :] - Ez[:-1, :, :]) / self.dx  # Shape: (nx-1, ny-1, nz)

        nx_common = min(Hy.shape[0], curl_ex_z.shape[0], curl_ez_x.shape[0])
        nz_common = min(Hy.shape[2], curl_ex_z.shape[2], curl_ez_x.shape[2])

        da_hy = self._interpolate_to_hy_points(self.Da)[
            :nx_common, : Hy.shape[1], :nz_common
        ]
        db_hy = self._interpolate_to_hy_points(self.Db)[
            :nx_common, : Hy.shape[1], :nz_common
        ]

        Hy[:nx_common, :, :nz_common] = da_hy * Hy[
            :nx_common, :, :nz_common
        ] - db_hy * (
            curl_ex_z[:nx_common, :, :nz_common] - curl_ez_x[:nx_common, :, :nz_common]
        )

        # Hz update: ∂Hz/∂t = -(1/μ) * (∂Ey/∂x - ∂Ex/∂y)
        # Hz is at (i, j, k+1/2), shape (nx, ny, nz-1)
        # Ey is at (i+1/2, j, k+1/2), shape (nx-1, ny, nz-1)
        # Ex is at (i, j+1/2, k+1/2), shape (nx, ny-1, nz-1)

        curl_ey_x = (Ey[1:, :, :] - Ey[:-1, :, :]) / self.dx  # Shape: (nx-1, ny, nz-1)
        curl_ex_y = (Ex[:, 1:, :] - Ex[:, :-1, :]) / self.dy  # Shape: (nx, ny-1, nz-1)

        nx_common = min(Hz.shape[0], curl_ey_x.shape[0], curl_ex_y.shape[0])
        ny_common = min(Hz.shape[1], curl_ey_x.shape[1], curl_ex_y.shape[1])

        da_hz = self._interpolate_to_hz_points(self.Da)[
            :nx_common, :ny_common, : Hz.shape[2]
        ]
        db_hz = self._interpolate_to_hz_points(self.Db)[
            :nx_common, :ny_common, : Hz.shape[2]
        ]

        Hz[:nx_common, :ny_common, :] = da_hz * Hz[
            :nx_common, :ny_common, :
        ] - db_hz * (
            curl_ey_x[:nx_common, :ny_common, :] - curl_ex_y[:nx_common, :ny_common, :]
        )

    def _update_e_fields_3d(self, fields: ElectromagneticFields) -> None:
        """Update E-field components for 3D case."""
        Ex, Ey, Ez = fields.Ex, fields.Ey, fields.Ez
        Hx, Hy, Hz = fields.Hx, fields.Hy, fields.Hz

        # Ex update: ∂Ex/∂t = (1/ε) * (∂Hz/∂y - ∂Hy/∂z)
        # Ex is at (i, j+1/2, k+1/2), shape (nx, ny-1, nz-1)
        # Hz is at (i, j, k+1/2), shape (nx, ny, nz-1)
        # Hy is at (i, j+1/2, k), shape (nx, ny-1, nz)

        curl_hz_y = (Hz[:, 1:, :] - Hz[:, :-1, :]) / self.dy  # Shape: (nx, ny-1, nz-1)
        curl_hy_z = (Hy[:, :, 1:] - Hy[:, :, :-1]) / self.dz  # Shape: (nx, ny-1, nz-1)

        ca_ex = self._interpolate_to_ex_points(self.Ca)[
            : Ex.shape[0], : Ex.shape[1], : Ex.shape[2]
        ]
        cb_ex = self._interpolate_to_ex_points(self.Cb)[
            : Ex.shape[0], : Ex.shape[1], : Ex.shape[2]
        ]

        Ex[:, :, :] = ca_ex * Ex + cb_ex * (curl_hz_y - curl_hy_z)

        # Ey update: ∂Ey/∂t = (1/ε) * (∂Hx/∂z - ∂Hz/∂x)
        # Ey is at (i+1/2, j, k+1/2), shape (nx-1, ny, nz-1)
        # Hx is at (i+1/2, j, k), shape (nx-1, ny, nz)
        # Hz is at (i, j, k+1/2), shape (nx, ny, nz-1)

        curl_hx_z = (Hx[:, :, 1:] - Hx[:, :, :-1]) / self.dz  # Shape: (nx-1, ny, nz-1)
        curl_hz_x = (Hz[1:, :, :] - Hz[:-1, :, :]) / self.dx  # Shape: (nx-1, ny, nz-1)

        ca_ey = self._interpolate_to_ey_points(self.Ca)[
            : Ey.shape[0], : Ey.shape[1], : Ey.shape[2]
        ]
        cb_ey = self._interpolate_to_ey_points(self.Cb)[
            : Ey.shape[0], : Ey.shape[1], : Ey.shape[2]
        ]

        Ey[:, :, :] = ca_ey * Ey + cb_ey * (curl_hx_z - curl_hz_x)

        # Ez update: ∂Ez/∂t = (1/ε) * (∂Hy/∂x - ∂Hx/∂y)
        # Ez is at (i+1/2, j+1/2, k), shape (nx-1, ny-1, nz)
        # Hy is at (i, j+1/2, k), shape (nx, ny-1, nz)
        # Hx is at (i+1/2, j, k), shape (nx-1, ny, nz)

        curl_hy_x = (Hy[1:, :, :] - Hy[:-1, :, :]) / self.dx  # Shape: (nx-1, ny-1, nz)
        curl_hx_y = (Hx[:, 1:, :] - Hx[:, :-1, :]) / self.dy  # Shape: (nx-1, ny-1, nz)

        ca_ez = self._interpolate_to_ez_points(self.Ca)[
            : Ez.shape[0], : Ez.shape[1], : Ez.shape[2]
        ]
        cb_ez = self._interpolate_to_ez_points(self.Cb)[
            : Ez.shape[0], : Ez.shape[1], : Ez.shape[2]
        ]

        Ez[:, :, :] = ca_ez * Ez + cb_ez * (curl_hy_x - curl_hx_y)

    def _update_h_fields_2d(self, fields: ElectromagneticFields) -> None:
        """Update H-field components for 2D case."""
        # In 2D we support two modes:
        # TE mode: Ez, Hx, Hy (Ez is dominant, H fields in-plane)
        # TM mode: Hz, Ex, Ey (Hz is dominant, E fields in-plane)

        Ez = fields.Ez
        Hx, Hy = fields.Hx, fields.Hy

        # TM mode (Ez dominant): Update Hx and Hy from Ez
        if np.max(np.abs(Ez)) > 0:
            # Hx update: ∂Hx/∂t = -(1/μ) * (-∂Ez/∂y) = (1/μ) * ∂Ez/∂y
            # Hx is at (i+1/2, j), Ez is at (i+1/2, j+1/2)
            # ∂Ez/∂y at Hx position: (Ez[i,j+1/2] - Ez[i,j-1/2])/dy
            nx_hx, ny_hx = Hx.shape
            nx_ez, ny_ez = Ez.shape

            if nx_ez >= nx_hx and ny_ez > 0:
                curl_ez_y = (
                    Ez[:nx_hx, 1:] - Ez[:nx_hx, :-1]
                ) / self.dy  # Shape: (nx_hx, ny_ez-1)

                da_hx = self._interpolate_to_hx_points_2d(self.Da[:, :, 0])
                db_hx = self._interpolate_to_hx_points_2d(self.Db[:, :, 0])

                # Apply update only where curl is defined
                ny_curl = min(ny_hx, curl_ez_y.shape[1])
                Hx[:, :ny_curl] = (
                    da_hx[:nx_hx, :ny_curl] * Hx[:, :ny_curl]
                    + db_hx[:nx_hx, :ny_curl] * curl_ez_y[:, :ny_curl]
                )

            # Hy update: ∂Hy/∂t = -(1/μ) * (∂Ez/∂x)
            # Hy is at (i, j+1/2), Ez is at (i+1/2, j+1/2)
            # ∂Ez/∂x at Hy position: (Ez[i+1/2,j] - Ez[i-1/2,j])/dx
            nx_hy, ny_hy = Hy.shape

            if nx_ez > 0 and ny_ez >= ny_hy:
                curl_ez_x = (
                    Ez[1:, :ny_hy] - Ez[:-1, :ny_hy]
                ) / self.dx  # Shape: (nx_ez-1, ny_hy)

                da_hy = self._interpolate_to_hy_points_2d(self.Da[:, :, 0])
                db_hy = self._interpolate_to_hy_points_2d(self.Db[:, :, 0])

                # Apply update only where curl is defined
                nx_curl = min(nx_hy, curl_ez_x.shape[0])
                Hy[:nx_curl, :] = (
                    da_hy[:nx_curl, :ny_hy] * Hy[:nx_curl, :]
                    - db_hy[:nx_curl, :ny_hy] * curl_ez_x[:nx_curl, :]
                )

        # TE mode (Hz dominant): Update Hz from Ex, Ey
        Hz = fields.Hz
        Ex, Ey = fields.Ex, fields.Ey

        if np.max(np.abs(Ex)) > 0 or np.max(np.abs(Ey)) > 0:
            # Hz update: ∂Hz/∂t = -(1/μ) * (∂Ey/∂x - ∂Ex/∂y)
            # Hz is at (i, j), Ex is at (i, j+1/2), Ey is at (i+1/2, j)

            curl_ey_x = np.zeros_like(Hz)
            if Ey.shape[0] > 0:
                # ∂Ey/∂x: (Ey[i+1/2,j] - Ey[i-1/2,j])/dx
                curl_ey_x[:-1, :] = (Ey[1:, :] - Ey[:-1, :]) / self.dx

            curl_ex_y = np.zeros_like(Hz)
            if Ex.shape[1] > 0:
                # ∂Ex/∂y: (Ex[i,j+1/2] - Ex[i,j-1/2])/dy
                curl_ex_y[:, :-1] = (Ex[:, 1:] - Ex[:, :-1]) / self.dy

            da_hz = self._interpolate_to_hz_points_2d(self.Da[:, :, 0])
            db_hz = self._interpolate_to_hz_points_2d(self.Db[:, :, 0])

            nx_hz, ny_hz = Hz.shape
            Hz[:, :] = da_hz[:nx_hz, :ny_hz] * Hz - db_hz[:nx_hz, :ny_hz] * (
                curl_ey_x - curl_ex_y
            )

    def _update_e_fields_2d(self, fields: ElectromagneticFields) -> None:
        """Update E-field components for 2D case."""
        # TM mode (Ez dominant): Update Ez from Hx, Hy
        Ez = fields.Ez
        Hx, Hy = fields.Hx, fields.Hy

        if Ez.shape[0] > 0 and Ez.shape[1] > 0:
            # Ez update: ∂Ez/∂t = (1/ε) * (∂Hy/∂x - ∂Hx/∂y)
            # Ez is at (i+1/2, j+1/2), Hx is at (i+1/2, j), Hy is at (i, j+1/2)

            curl_hy_x = (Hy[1:, :] - Hy[:-1, :]) / self.dx  # Shape: (nx-1, ny)
            curl_hx_y = (Hx[:, 1:] - Hx[:, :-1]) / self.dy  # Shape: (nx, ny-1)

            # Both curls need to match Ez shape (nx-1, ny-1)
            # Take appropriate slices
            ca_ez = self._interpolate_to_ez_points_2d(self.Ca[:, :, 0])
            cb_ez = self._interpolate_to_ez_points_2d(self.Cb[:, :, 0])

            nx_ez, ny_ez = Ez.shape
            ca_ez = ca_ez[:nx_ez, :ny_ez]
            cb_ez = cb_ez[:nx_ez, :ny_ez]

            Ez[:, :] = ca_ez * Ez + cb_ez * (
                curl_hy_x[:nx_ez, :ny_ez] - curl_hx_y[:nx_ez, :ny_ez]
            )

        # TE mode (Hz dominant): Update Ex, Ey from Hz
        Hz = fields.Hz
        Ex, Ey = fields.Ex, fields.Ey

        if Hz.shape[0] > 0 and Hz.shape[1] > 0:
            # Ex update: ∂Ex/∂t = (1/ε) * ∂Hz/∂y
            # Ex is at (i, j+1/2), Hz is at (i, j)
            if Ex.shape[0] > 0 and Ex.shape[1] > 0:
                curl_hz_y = (Hz[:, 1:] - Hz[:, :-1]) / self.dy  # Shape matches Ex

                ca_ex = self._interpolate_to_ex_points_2d(self.Ca[:, :, 0])
                cb_ex = self._interpolate_to_ex_points_2d(self.Cb[:, :, 0])

                nx_ex, ny_ex = Ex.shape
                ca_ex = ca_ex[:nx_ex, :ny_ex]
                cb_ex = cb_ex[:nx_ex, :ny_ex]

                Ex[:, :] = ca_ex * Ex + cb_ex * curl_hz_y

            # Ey update: ∂Ey/∂t = -(1/ε) * ∂Hz/∂x
            # Ey is at (i+1/2, j), Hz is at (i, j)
            if Ey.shape[0] > 0 and Ey.shape[1] > 0:
                curl_hz_x = (Hz[1:, :] - Hz[:-1, :]) / self.dx  # Shape matches Ey

                ca_ey = self._interpolate_to_ey_points_2d(self.Ca[:, :, 0])
                cb_ey = self._interpolate_to_ey_points_2d(self.Cb[:, :, 0])

                nx_ey, ny_ey = Ey.shape
                ca_ey = ca_ey[:nx_ey, :ny_ey]
                cb_ey = cb_ey[:nx_ey, :ny_ey]

                Ey[:, :] = ca_ey * Ey - cb_ey * curl_hz_x

    def _interpolate_to_ex_points(self, array: np.ndarray) -> np.ndarray:
        """Interpolate material properties to Ex field points."""
        # Ex is at (i, j+1/2, k+1/2) - average over y and z
        return 0.25 * (
            array[:, :-1, :-1]
            + array[:, 1:, :-1]
            + array[:, :-1, 1:]
            + array[:, 1:, 1:]
        )

    def _interpolate_to_ey_points(self, array: np.ndarray) -> np.ndarray:
        """Interpolate material properties to Ey field points."""
        # Ey is at (i+1/2, j, k+1/2) - average over x and z
        return 0.25 * (
            array[:-1, :, :-1]
            + array[1:, :, :-1]
            + array[:-1, :, 1:]
            + array[1:, :, 1:]
        )

    def _interpolate_to_ez_points(self, array: np.ndarray) -> np.ndarray:
        """Interpolate material properties to Ez field points."""
        # Ez is at (i+1/2, j+1/2, k) - average over x and y
        return 0.25 * (
            array[:-1, :-1, :]
            + array[1:, :-1, :]
            + array[:-1, 1:, :]
            + array[1:, 1:, :]
        )

    def _interpolate_to_hx_points(self, array: np.ndarray) -> np.ndarray:
        """Interpolate material properties to Hx field points."""
        # Hx is at (i+1/2, j, k) - average over x
        return 0.5 * (array[:-1, :, :] + array[1:, :, :])

    def _interpolate_to_hy_points(self, array: np.ndarray) -> np.ndarray:
        """Interpolate material properties to Hy field points."""
        # Hy is at (i, j+1/2, k) - average over y
        return 0.5 * (array[:, :-1, :] + array[:, 1:, :])

    def _interpolate_to_hz_points(self, array: np.ndarray) -> np.ndarray:
        """Interpolate material properties to Hz field points."""
        # Hz is at (i, j, k+1/2) - average over z
        return 0.5 * (array[:, :, :-1] + array[:, :, 1:])

    def _interpolate_to_ex_points_2d(self, array: np.ndarray) -> np.ndarray:
        """Interpolate material properties to Ex field points (2D)."""
        # Ex is at (i, j+1/2) - average over y
        return 0.5 * (array[:, :-1] + array[:, 1:])

    def _interpolate_to_ey_points_2d(self, array: np.ndarray) -> np.ndarray:
        """Interpolate material properties to Ey field points (2D)."""
        # Ey is at (i+1/2, j) - average over x
        return 0.5 * (array[:-1, :] + array[1:, :])

    def _interpolate_to_ez_points_2d(self, array: np.ndarray) -> np.ndarray:
        """Interpolate material properties to Ez field points (2D)."""
        # Ez is at (i+1/2, j+1/2) - average over both x and y
        return 0.25 * (
            array[:-1, :-1] + array[1:, :-1] + array[:-1, 1:] + array[1:, 1:]
        )

    def _interpolate_to_hz_points_2d(self, array: np.ndarray) -> np.ndarray:
        """Interpolate material properties to Hz field points (2D)."""
        # Hz is at (i, j) - no averaging needed for cell centers
        return array

    def _interpolate_to_hx_points_2d(self, array: np.ndarray) -> np.ndarray:
        """Interpolate material properties to Hx field points (2D)."""
        # Hx is at (i+1/2, j) - average over x
        return 0.5 * (array[:-1, :] + array[1:, :])

    def _interpolate_to_hy_points_2d(self, array: np.ndarray) -> np.ndarray:
        """Interpolate material properties to Hy field points (2D)."""
        # Hy is at (i, j+1/2) - average over y
        return 0.5 * (array[:, :-1] + array[:, 1:])

    def step(self, fields: ElectromagneticFields) -> None:
        """
        Perform one complete FDTD time step.

        The leap-frog time stepping follows:
        1. Update H-fields using E-fields at time n
        2. Update E-fields using H-fields at time n+1/2

        Parameters
        ----------
        fields : ElectromagneticFields
            Electromagnetic fields to advance in time.
        """
        # Update H-fields (time n -> n+1/2)
        self.update_magnetic_fields(fields)

        # Update E-fields (time n -> n+1)
        self.update_electric_fields(fields)

    def get_time_step(self) -> float:
        """Get the time step."""
        return self.dt

    def get_courant_number(self) -> float:
        """Get the Courant number for this updater."""
        return self.grid.get_courant_number(self.dt)

    def __repr__(self) -> str:
        """String representation."""
        courant = self.get_courant_number()
        return (
            f"MaxwellUpdater(dt={self.dt:.2e}s, "
            f"Courant={courant:.3f}, grid={self.grid.dimensions})"
        )


class FDTDSolver:
    """
    High-level FDTD solver combining grid, fields, and updater.

    This class provides a convenient interface for running FDTD simulations
    with automatic time stepping and field management.

    Parameters
    ----------
    grid : YeeGrid
        Computational grid.
    dt : float, optional
        Time step. If None, automatically computed for stability.
    material_arrays : dict, optional
        Material property arrays.
    """

    def __init__(
        self,
        grid: YeeGrid,
        dt: Optional[float] = None,
        material_arrays: Optional[dict] = None,
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

        # Auto-determine time step if not provided
        if dt is None:
            dt = grid.suggest_time_step(safety_factor=0.95)

        self.fields = ElectromagneticFields(grid, backend=self.backend)
        self.updater = MaxwellUpdater(grid, dt, material_arrays, backend=self.backend)

        self.time = 0.0
        self.step_count = 0

    def run(self, total_time: float, callback: Optional[callable] = None) -> None:
        """
        Run FDTD simulation for specified time.

        Parameters
        ----------
        total_time : float
            Total simulation time in seconds.
        callback : callable, optional
            Function called after each time step: callback(solver, step_num).
        """
        dt = self.updater.get_time_step()
        num_steps = int(np.ceil(total_time / dt))

        for step in range(num_steps):
            # Perform one FDTD step
            self.updater.step(self.fields)

            # Update time tracking
            self.time += dt
            self.step_count += 1

            # Call user callback if provided
            if callback is not None:
                callback(self, step)

    def run_steps(self, num_steps: int, callback: Optional[callable] = None) -> None:
        """
        Run FDTD simulation for specified number of steps.

        Parameters
        ----------
        num_steps : int
            Number of time steps to run.
        callback : callable, optional
            Function called after each time step.
        """
        dt = self.updater.get_time_step()

        for step in range(num_steps):
            self.updater.step(self.fields)
            self.time += dt
            self.step_count += 1

            if callback is not None:
                callback(self, step)

    def step(self, fields: Optional[ElectromagneticFields] = None) -> None:
        """
        Perform a single FDTD time step.

        Parameters
        ----------
        fields : ElectromagneticFields, optional
            Fields to update. If None, uses internal fields.
        """
        if fields is None:
            fields = self.fields

        self.updater.step(fields)
        self.time += self.updater.get_time_step()
        self.step_count += 1

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self.fields.zero_fields()
        self.time = 0.0
        self.step_count = 0

    def get_simulation_info(self) -> dict:
        """Get information about the current simulation state."""
        return {
            "time": self.time,
            "step_count": self.step_count,
            "dt": self.updater.get_time_step(),
            "courant_number": self.updater.get_courant_number(),
            "grid_dimensions": self.grid.dimensions,
            "is_2d": self.grid.is_2d,
            "field_energy": self.fields.get_field_energy(),
        }

    def __repr__(self) -> str:
        """String representation."""
        info = self.get_simulation_info()
        return (
            f"FDTDSolver(t={info['time']:.2e}s, steps={info['step_count']}, "
            f"E_total={info['field_energy']:.2e}J)"
        )
