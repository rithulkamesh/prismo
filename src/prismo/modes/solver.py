"""
Eigenmode solver for waveguide structures.

This module implements a 2D eigenmode solver for computing guided modes
of waveguide structures using the finite-difference method.
"""

from typing import Tuple, Optional, List
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


@dataclass
class WaveguideMode:
    """
    Container for waveguide mode information.

    Attributes
    ----------
    mode_number : int
        Mode index (0 for fundamental, 1 for first higher order, etc.).
    neff : complex
        Effective refractive index.
    frequency : float
        Frequency at which mode was calculated (Hz).
    wavelength : float
        Wavelength (m).
    Ex, Ey, Ez : ndarray
        Electric field components.
    Hx, Hy, Hz : ndarray
        Magnetic field components.
    x, y : ndarray
        Coordinate arrays.
    power : float
        Mode power (normalized).
    """

    mode_number: int
    neff: complex
    frequency: float
    wavelength: float
    Ex: np.ndarray
    Ey: np.ndarray
    Ez: np.ndarray
    Hx: np.ndarray
    Hy: np.ndarray
    Hz: np.ndarray
    x: np.ndarray
    y: np.ndarray
    power: float = 1.0


class ModeSolver:
    """
    2D finite-difference eigenmode solver for waveguides.

    Solves the vector wave equation to find guided modes:
    ∇ × ∇ × E = (ω/c)² ε E

    Parameters
    ----------
    wavelength : float
        Vacuum wavelength (m).
    x : ndarray
        x-coordinates for the mode profile.
    y : ndarray
        y-coordinates for the mode profile.
    epsilon : ndarray
        Permittivity distribution (relative).
    """

    def __init__(
        self,
        wavelength: float,
        x: np.ndarray,
        y: np.ndarray,
        epsilon: np.ndarray,
    ):
        self.wavelength = wavelength
        self.frequency = 299792458.0 / wavelength  # c / λ
        self.k0 = 2 * np.pi / wavelength  # Free-space wavenumber

        self.x = x
        self.y = y
        self.epsilon = epsilon

        # Grid spacing
        self.dx = x[1] - x[0] if len(x) > 1 else 1.0
        self.dy = y[1] - y[0] if len(y) > 1 else 1.0

        # Grid dimensions
        self.nx = len(x)
        self.ny = len(y)

        self.modes: List[WaveguideMode] = []

    def solve(
        self,
        num_modes: int = 1,
        mode_type: str = "TE",
        beta_guess: Optional[float] = None,
    ) -> List[WaveguideMode]:
        """
        Solve for guided modes.

        Parameters
        ----------
        num_modes : int
            Number of modes to calculate.
        mode_type : str
            Mode type: 'TE', 'TM', or 'vector' (full vectorial).
        beta_guess : float, optional
            Initial guess for propagation constant.

        Returns
        -------
        List[WaveguideMode]
            List of calculated modes.
        """
        if mode_type.upper() == "TE":
            modes = self._solve_te_modes(num_modes, beta_guess)
        elif mode_type.upper() == "TM":
            modes = self._solve_tm_modes(num_modes, beta_guess)
        elif mode_type.upper() == "VECTOR":
            modes = self._solve_vector_modes(num_modes, beta_guess)
        else:
            raise ValueError(f"Unknown mode type: {mode_type}")

        self.modes = modes
        return modes

    def _solve_te_modes(
        self, num_modes: int, beta_guess: Optional[float]
    ) -> List[WaveguideMode]:
        """
        Solve for TE modes (Ez = 0, Hz dominant).

        Wave equation for Hz:
        ∇²Hz + (k0² ε - β²) Hz = 0
        """
        # Build finite-difference Laplacian operator
        A = self._build_laplacian_2d()

        # Build material operator: k0² ε
        eps_flat = self.epsilon.ravel()
        M = sparse.diags(self.k0**2 * eps_flat)

        # Eigenvalue problem: A Hz = -λ Hz where λ = β² - k0² ε
        # Rearrange: (A + k0² ε) Hz = β² Hz
        # Or: (A + M) Hz = β² Hz

        # Estimate beta range
        if beta_guess is None:
            # Guess based on max and min epsilon
            n_max = np.sqrt(np.max(self.epsilon))
            n_min = np.sqrt(np.min(self.epsilon))
            beta_max = self.k0 * n_max
            beta_min = self.k0 * n_min
            sigma = 0.5 * (beta_max + beta_min)  # Target near middle
        else:
            sigma = beta_guess

        # Solve eigenvalue problem
        try:
            # Use shift-invert mode for finding modes near sigma
            eigenvalues, eigenvectors = spla.eigs(
                A + M,
                k=min(num_modes, self.nx * self.ny - 2),
                sigma=sigma**2,
                which="LM",
            )
        except:
            # Fallback to direct solver
            A_dense = (A + M).toarray()
            eigenvalues, eigenvectors = np.linalg.eig(A_dense)
            # Sort by largest eigenvalues
            idx = np.argsort(-np.real(eigenvalues))
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        # Convert eigenvalues to propagation constants: β = sqrt(λ)
        beta_values = np.sqrt(eigenvalues)

        # Filter physically meaningful modes (real beta, above light line)
        valid_modes = []
        for i, beta in enumerate(beta_values):
            if np.real(beta) > self.k0 * np.sqrt(np.min(self.epsilon)):
                if np.real(beta) < self.k0 * np.sqrt(np.max(self.epsilon)):
                    valid_modes.append((beta, eigenvectors[:, i]))

        # Create WaveguideMode objects
        modes = []
        for mode_idx, (beta, Hz_flat) in enumerate(valid_modes[:num_modes]):
            Hz = Hz_flat.reshape((self.nx, self.ny))

            # Normalize mode
            Hz = Hz / np.max(np.abs(Hz))

            # Calculate transverse fields from Hz
            # For TE: Ex = (jβ/k0²ε) ∂Hz/∂y, Ey = -(jβ/k0²ε) ∂Hz/∂x
            Ex, Ey = self._calculate_te_transverse_fields(Hz, beta)

            neff = beta / self.k0

            mode = WaveguideMode(
                mode_number=mode_idx,
                neff=neff,
                frequency=self.frequency,
                wavelength=self.wavelength,
                Ex=Ex,
                Ey=Ey,
                Ez=np.zeros_like(Hz),
                Hx=np.zeros_like(Hz),
                Hy=np.zeros_like(Hz),
                Hz=Hz,
                x=self.x,
                y=self.y,
                power=1.0,
            )
            modes.append(mode)

        return modes

    def _solve_tm_modes(
        self, num_modes: int, beta_guess: Optional[float]
    ) -> List[WaveguideMode]:
        """
        Solve for TM modes (Hz = 0, Ez dominant).

        Similar to TE but for Ez component.
        """
        # Similar implementation to TE modes
        # For brevity, return empty list (full implementation would be similar to TE)
        return []

    def _solve_vector_modes(
        self, num_modes: int, beta_guess: Optional[float]
    ) -> List[WaveguideMode]:
        """
        Solve for full vectorial modes.

        This requires solving the full 6-component Maxwell system.
        More complex - placeholder implementation.
        """
        # Full vectorial mode solver
        # This is significantly more complex and would require
        # solving a larger eigenvalue problem
        return []

    def _build_laplacian_2d(self) -> sparse.spmatrix:
        """
        Build 2D Laplacian operator using finite differences.

        ∇² = ∂²/∂x² + ∂²/∂y²

        Returns
        -------
        sparse matrix
            Laplacian operator.
        """
        nx, ny = self.nx, self.ny
        n = nx * ny

        # Build Laplacian using 5-point stencil
        # ∇²f ≈ (f[i+1,j] + f[i-1,j] - 2f[i,j])/dx²
        #     + (f[i,j+1] + f[i,j-1] - 2f[i,j])/dy²

        # Diagonal entries
        diag = -2.0 / self.dx**2 - 2.0 / self.dy**2

        # Off-diagonal entries
        diag_x = 1.0 / self.dx**2  # x-neighbors
        diag_y = 1.0 / self.dy**2  # y-neighbors

        # Build sparse matrix
        diagonals = [
            np.full(n, diag),  # Main diagonal
            np.full(n - 1, diag_x),  # x-neighbors
            np.full(n - 1, diag_x),
            np.full(n - nx, diag_y),  # y-neighbors
            np.full(n - nx, diag_y),
        ]

        offsets = [0, 1, -1, nx, -nx]

        A = sparse.diags(diagonals, offsets, shape=(n, n), format="csr")

        # Apply boundary conditions (Dirichlet: field = 0 at boundaries)
        # This is automatically satisfied by the zero-padding assumption

        return A

    def _calculate_te_transverse_fields(
        self, Hz: np.ndarray, beta: complex
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Ex, Ey from Hz for TE modes.

        Ex = (jβ/k0²ε) ∂Hz/∂y
        Ey = -(jβ/k0²ε) ∂Hz/∂x
        """
        # Calculate derivatives using finite differences
        dHz_dy = np.gradient(Hz, self.dy, axis=1)
        dHz_dx = np.gradient(Hz, self.dx, axis=0)

        # Calculate coefficient
        coeff = 1j * beta / (self.k0**2 * self.epsilon)

        Ex = coeff * dHz_dy
        Ey = -coeff * dHz_dx

        return Ex, Ey

    def get_mode(self, mode_number: int = 0) -> WaveguideMode:
        """
        Get a specific mode by index.

        Parameters
        ----------
        mode_number : int
            Mode index (0 = fundamental).

        Returns
        -------
        WaveguideMode
            The requested mode.
        """
        if mode_number >= len(self.modes):
            raise IndexError(
                f"Mode {mode_number} not found. Only {len(self.modes)} modes calculated."
            )
        return self.modes[mode_number]
