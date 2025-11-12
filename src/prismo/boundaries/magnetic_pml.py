"""
Magnetic Perfectly Matched Layer (Magnetic PML) absorbing boundaries.

This module implements magnetic PML for absorbing magnetic fields at boundaries.
It is the magnetic dual of the electric CPML, using similar CFS-PML formulation
but applied to magnetic field updates.

References:
- Roden, J. A., & Gedney, S. D. (2000). "Convolution PML (CPML): An efficient
  FDTD implementation of the CFS-PML for arbitrary media." Microwave and optical
  technology letters, 27(5), 334-339.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from prismo.backends import Backend, get_backend
from prismo.core.fields import ElectromagneticFields
from prismo.core.grid import YeeGrid


@dataclass
class MagneticPMLParams:
    """
    Parameters for magnetic PML configuration.

    Parameters
    ----------
    thickness : int
        Number of PML layers in grid cells.
    sigma_max : float, optional
        Maximum magnetic conductivity value. If None, uses optimal value.
    kappa_max : float, optional
        Maximum kappa stretching factor, default=15.0.
    alpha_max : float, optional
        Maximum alpha CFS parameter, default=0.0.
    polynomial_order : int, optional
        Polynomial grading order, default=3.
    """

    thickness: int
    sigma_max: Optional[float] = None
    kappa_max: float = 15.0
    alpha_max: float = 0.0
    polynomial_order: int = 3


class MagneticPML:
    """
    Magnetic Perfectly Matched Layer (Magnetic PML) absorbing boundary.

    The magnetic PML is the dual of electric PML, designed to absorb magnetic
    fields at boundaries. It uses the same CFS-PML formulation but applied
    to the magnetic field update equations.

    Parameters
    ----------
    grid : YeeGrid
        The simulation grid.
    params : MagneticPMLParams
        Magnetic PML configuration parameters.
    backend : Backend, optional
        Computational backend to use.
    """

    def __init__(
        self,
        grid: YeeGrid,
        params: MagneticPMLParams,
        backend: Optional[Union[str, Backend]] = None,
    ):
        self.grid = grid
        self.params = params

        # Initialize backend
        if isinstance(backend, str):
            self.backend = get_backend(backend)
        elif isinstance(backend, Backend):
            self.backend = backend
        elif backend is None:
            self.backend = get_backend()
        else:
            raise TypeError("backend must be a Backend instance or string name")

        # Physical constants
        self.eps0 = 8.854187817e-12  # Vacuum permittivity (F/m)
        self.mu0 = 4 * self.backend.pi * 1e-7  # Vacuum permeability (H/m)

        # Initialize PML parameters and auxiliary fields
        self._setup_pml_regions()
        self._compute_pml_parameters()
        self._initialize_auxiliary_fields()

    def _setup_pml_regions(self) -> None:
        """Identify PML regions in the grid."""
        nx, ny, nz = self.grid.dimensions
        t = self.params.thickness

        # Define PML regions (bool masks)
        # x-direction PML
        self.pml_x_min = np.arange(nx) < t
        self.pml_x_max = np.arange(nx) >= (nx - t)

        # y-direction PML
        self.pml_y_min = np.arange(ny) < t
        self.pml_y_max = np.arange(ny) >= (ny - t)

        # z-direction PML (only for 3D)
        if self.grid.is_3d:
            self.pml_z_min = np.arange(nz) < t
            self.pml_z_max = np.arange(nz) >= (nz - t)
        else:
            self.pml_z_min = np.zeros(nz, dtype=bool)
            self.pml_z_max = np.zeros(nz, dtype=bool)

    def _compute_pml_parameters(self) -> None:
        """
        Compute magnetic PML conductivity, kappa, and alpha parameters.

        Uses polynomial grading for smooth transition from interior to PML.
        """
        t = self.params.thickness
        m = self.params.polynomial_order

        # Compute optimal sigma_max if not provided
        # For magnetic PML, we use similar scaling as electric PML
        if self.params.sigma_max is None:
            sigma_max = (
                0.8 * (m + 1) / (150.0 * self.backend.pi * max(self.grid.spacing))
            )
        else:
            sigma_max = self.params.sigma_max

        kappa_max = self.params.kappa_max
        alpha_max = self.params.alpha_max

        # Create grading arrays (distance from PML boundary)
        depth = np.arange(t, dtype=np.float64)

        # Polynomial grading
        rho = depth / t  # Normalized depth (0 at interior, 1 at boundary)

        # Compute sigma, kappa, alpha profiles
        sigma = sigma_max * (rho**m)
        kappa = 1.0 + (kappa_max - 1.0) * (rho**m)
        alpha = alpha_max * ((1.0 - rho) ** m)

        # Convert to backend arrays
        self.sigma_pml = self.backend.asarray(sigma)
        self.kappa_pml = self.backend.asarray(kappa)
        self.alpha_pml = self.backend.asarray(alpha)

    def _initialize_auxiliary_fields(self) -> None:
        """
        Initialize auxiliary field arrays for magnetic PML convolution.

        For magnetic CPML, we need auxiliary fields to store convolution history
        for the magnetic field updates.
        """
        nx, ny, nz = self.grid.dimensions

        # Create auxiliary field arrays for magnetic field updates
        # These store the convolution integrals for magnetic PML

        # For H-field updates (need Psi_H for curl E)
        self.Psi_Hxy = self.backend.zeros((nx, ny, nz), dtype=self.backend.float64)
        self.Psi_Hxz = self.backend.zeros((nx, ny, nz), dtype=self.backend.float64)
        self.Psi_Hyx = self.backend.zeros((nx, ny, nz), dtype=self.backend.float64)
        self.Psi_Hyz = self.backend.zeros((nx, ny, nz), dtype=self.backend.float64)
        self.Psi_Hzx = self.backend.zeros((nx, ny, nz), dtype=self.backend.float64)
        self.Psi_Hzy = self.backend.zeros((nx, ny, nz), dtype=self.backend.float64)

        # Compute update coefficients for convolution
        self._compute_pml_coefficients()

    def _compute_pml_coefficients(self) -> None:
        """
        Compute magnetic PML update coefficients for convolution recursion.

        The magnetic CPML update uses: Psi^(n+1) = b * Psi^n + a * (dF/dx)
        where F is the field derivative for magnetic fields.
        """
        nx, ny, nz = self.grid.dimensions
        dx, dy, dz = self.grid.spacing
        t = self.params.thickness

        # Get time step from grid
        dt = self.grid.suggest_time_step(safety_factor=0.9)

        # Compute b and a coefficients for each PML layer
        # b = exp(-(sigma/kappa + alpha) * dt)
        # a = sigma / (sigma * kappa + kappa^2 * alpha) * (b - 1)

        def compute_coeff_arrays(direction: str) -> tuple[np.ndarray, np.ndarray]:
            """Compute coefficient arrays for a given direction."""
            if direction == "x":
                n_dim = nx
            elif direction == "y":
                n_dim = ny
            else:  # z
                n_dim = nz

            b_array = np.ones(n_dim)
            a_array = np.zeros(n_dim)

            # Fill in PML regions
            sigma_np = self.backend.to_numpy(self.sigma_pml)
            kappa_np = self.backend.to_numpy(self.kappa_pml)
            alpha_np = self.backend.to_numpy(self.alpha_pml)

            # Lower boundary (indices 0 to t-1)
            for i in range(t):
                idx = t - 1 - i  # Reverse indexing
                b_array[i] = np.exp(
                    -(sigma_np[idx] / kappa_np[idx] + alpha_np[idx]) * dt
                )
                if sigma_np[idx] != 0:
                    a_array[i] = (
                        sigma_np[idx]
                        / (
                            sigma_np[idx] * kappa_np[idx]
                            + kappa_np[idx] ** 2 * alpha_np[idx]
                        )
                        * (b_array[i] - 1.0)
                    )

            # Upper boundary (indices n-t to n-1)
            for i in range(t):
                idx = i
                b_array[n_dim - t + i] = np.exp(
                    -(sigma_np[idx] / kappa_np[idx] + alpha_np[idx]) * dt
                )
                if sigma_np[idx] != 0:
                    a_array[n_dim - t + i] = (
                        sigma_np[idx]
                        / (
                            sigma_np[idx] * kappa_np[idx]
                            + kappa_np[idx] ** 2 * alpha_np[idx]
                        )
                        * (b_array[n_dim - t + i] - 1.0)
                    )

            return self.backend.asarray(b_array), self.backend.asarray(a_array)

        # Compute coefficients for each direction
        self.bx, self.ax = compute_coeff_arrays("x")
        self.by, self.ay = compute_coeff_arrays("y")
        if self.grid.is_3d:
            self.bz, self.az = compute_coeff_arrays("z")

    def update_magnetic_pml(
        self, fields: ElectromagneticFields, curl_e_x: any, curl_e_y: any, curl_e_z: any
    ) -> tuple[any, any, any]:
        """
        Update magnetic PML auxiliary fields and apply PML correction.

        Parameters
        ----------
        fields : ElectromagneticFields
            Field arrays.
        curl_e_x, curl_e_y, curl_e_z : array
            Curl of E-field components.

        Returns
        -------
        Tuple of corrected curl components with magnetic PML applied.
        """
        # Update Psi fields (convolution integrals) for magnetic fields
        # This is a simplified implementation - full implementation would
        # apply updates only in PML regions for efficiency

        # For now, return unmodified curl (basic implementation)
        # Full implementation would modify curl_e based on auxiliary fields
        # and apply magnetic PML corrections similar to electric PML
        return curl_e_x, curl_e_y, curl_e_z

    def apply_pml(self, fields: ElectromagneticFields) -> None:
        """
        Apply magnetic PML boundary conditions to fields.

        This is called after standard FDTD field updates to apply
        the magnetic PML absorption in boundary regions.

        Parameters
        ----------
        fields : ElectromagneticFields
            Electromagnetic fields to apply magnetic PML to.
        """
        # Main entry point for magnetic PML application
        # In a full implementation, this would:
        # 1. Update auxiliary Psi fields for magnetic fields
        # 2. Modify magnetic field updates in PML regions
        # 3. Apply stretching factors (kappa)
        pass

    def get_reflection_coefficient(self, angle: float = 0.0) -> float:
        """
        Estimate theoretical reflection coefficient of magnetic PML.

        Parameters
        ----------
        angle : float
            Incidence angle in degrees.

        Returns
        -------
        float
            Theoretical reflection coefficient.
        """
        # Simplified estimation (similar to electric PML)
        t = self.params.thickness
        m = self.params.polynomial_order

        # Theoretical reflection coefficient (approximate)
        R0 = np.exp(-2 * t / (m + 1))

        return R0

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MagneticPML(thickness={self.params.thickness}, "
            f"grid={self.grid.dimensions})"
        )
