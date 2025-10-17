"""
Anisotropic tensor materials for FDTD simulations.

This module implements materials with tensor permittivity and permeability,
supporting diagonal, symmetric, and full anisotropic tensors.
"""

from typing import Tuple, Optional, Union, Literal
import numpy as np
from dataclasses import dataclass

from prismo.backends import Backend, get_backend


@dataclass
class TensorComponents:
    """
    Components of a 3x3 material tensor.

    For diagonal tensors, only xx, yy, zz are needed.
    For symmetric tensors, also need xy, xz, yz.
    For full tensors, all 9 components.

    Attributes
    ----------
    xx, yy, zz : float or array
        Diagonal components.
    xy, xz, yz : float or array, optional
        Off-diagonal components (symmetric).
    yx, zx, zy : float or array, optional
        Off-diagonal components (full tensor).
    """

    # Diagonal
    xx: Union[float, np.ndarray]
    yy: Union[float, np.ndarray]
    zz: Union[float, np.ndarray]

    # Off-diagonal (symmetric part)
    xy: Union[float, np.ndarray] = 0.0
    xz: Union[float, np.ndarray] = 0.0
    yz: Union[float, np.ndarray] = 0.0

    # Off-diagonal (asymmetric part)
    yx: Optional[Union[float, np.ndarray]] = None
    zx: Optional[Union[float, np.ndarray]] = None
    zy: Optional[Union[float, np.ndarray]] = None

    def is_diagonal(self) -> bool:
        """Check if tensor is diagonal."""
        return (
            np.all(self.xy == 0)
            and np.all(self.xz == 0)
            and np.all(self.yz == 0)
            and (self.yx is None or np.all(self.yx == 0))
            and (self.zx is None or np.all(self.zx == 0))
            and (self.zy is None or np.all(self.zy == 0))
        )

    def is_symmetric(self) -> bool:
        """Check if tensor is symmetric."""
        if self.yx is None and self.zx is None and self.zy is None:
            return True

        return (
            np.allclose(self.xy, self.yx if self.yx is not None else self.xy)
            and np.allclose(self.xz, self.zx if self.zx is not None else self.xz)
            and np.allclose(self.yz, self.zy if self.zy is not None else self.yz)
        )

    def to_full_matrix(self, backend: Backend) -> np.ndarray:
        """
        Convert to full 3x3 matrix representation.

        Returns
        -------
        ndarray
            Tensor as 3x3 matrix or (..., 3, 3) array for spatially varying.
        """
        # Handle symmetric case
        yx = self.yx if self.yx is not None else self.xy
        zx = self.zx if self.zx is not None else self.xz
        zy = self.zy if self.zy is not None else self.yz

        # Create matrix
        if np.isscalar(self.xx):
            # Scalar tensor
            tensor = np.array(
                [[self.xx, self.xy, self.xz], [yx, self.yy, self.yz], [zx, zy, self.zz]]
            )
        else:
            # Spatially varying tensor
            shape = np.asarray(self.xx).shape
            tensor = np.zeros(shape + (3, 3))

            tensor[..., 0, 0] = self.xx
            tensor[..., 1, 1] = self.yy
            tensor[..., 2, 2] = self.zz
            tensor[..., 0, 1] = self.xy
            tensor[..., 0, 2] = self.xz
            tensor[..., 1, 2] = self.yz
            tensor[..., 1, 0] = yx
            tensor[..., 2, 0] = zx
            tensor[..., 2, 1] = zy

        return backend.asarray(tensor)


class TensorMaterial:
    """
    Material with anisotropic tensor properties.

    Supports materials with tensor permittivity and/or permeability:
    D = ε₀ [ε] · E
    B = μ₀ [μ] · H

    where [ε] and [μ] are 3×3 tensors.

    Parameters
    ----------
    epsilon : TensorComponents
        Permittivity tensor components (relative).
    mu : TensorComponents, optional
        Permeability tensor components (relative). Default is isotropic μ=1.
    name : str, optional
        Material name.
    backend : Backend, optional
        Computational backend.
    """

    def __init__(
        self,
        epsilon: TensorComponents,
        mu: Optional[TensorComponents] = None,
        name: str = "",
        backend: Optional[Union[str, Backend]] = None,
    ):
        self.epsilon = epsilon

        if mu is None:
            # Isotropic permeability
            mu = TensorComponents(xx=1.0, yy=1.0, zz=1.0)
        self.mu = mu

        self.name = name or "AnisotropicMaterial"

        # Initialize backend
        if isinstance(backend, str):
            self.backend = get_backend(backend)
        elif isinstance(backend, Backend):
            self.backend = backend
        elif backend is None:
            self.backend = get_backend()
        else:
            raise TypeError("backend must be a Backend instance or string name")

        # Convert tensors to backend arrays
        self.epsilon_tensor = epsilon.to_full_matrix(self.backend)
        self.mu_tensor = mu.to_full_matrix(self.backend)

        # Detect tensor type for optimization
        self.is_diagonal = epsilon.is_diagonal() and mu.is_diagonal()
        self.is_symmetric = epsilon.is_symmetric() and mu.is_symmetric()

    def apply_to_e_field(self, E: Tuple[any, any, any]) -> Tuple[any, any, any]:
        """
        Apply permittivity tensor to E field: D = ε₀ [ε] · E

        Parameters
        ----------
        E : Tuple of arrays
            Electric field components (Ex, Ey, Ez).

        Returns
        -------
        Tuple of arrays
            D field components (Dx, Dy, Dz).
        """
        Ex, Ey, Ez = E

        if self.is_diagonal:
            # Optimized for diagonal tensors
            Dx = self.epsilon.xx * Ex
            Dy = self.epsilon.yy * Ey
            Dz = self.epsilon.zz * Ez
        else:
            # Full tensor multiplication
            # D = [ε] · E
            Dx = (
                self.epsilon_tensor[..., 0, 0] * Ex
                + self.epsilon_tensor[..., 0, 1] * Ey
                + self.epsilon_tensor[..., 0, 2] * Ez
            )
            Dy = (
                self.epsilon_tensor[..., 1, 0] * Ex
                + self.epsilon_tensor[..., 1, 1] * Ey
                + self.epsilon_tensor[..., 1, 2] * Ez
            )
            Dz = (
                self.epsilon_tensor[..., 2, 0] * Ex
                + self.epsilon_tensor[..., 2, 1] * Ey
                + self.epsilon_tensor[..., 2, 2] * Ez
            )

        return Dx, Dy, Dz

    def apply_to_h_field(self, H: Tuple[any, any, any]) -> Tuple[any, any, any]:
        """
        Apply permeability tensor to H field: B = μ₀ [μ] · H

        Parameters
        ----------
        H : Tuple of arrays
            Magnetic field components (Hx, Hy, Hz).

        Returns
        -------
        Tuple of arrays
            B field components (Bx, By, Bz).
        """
        Hx, Hy, Hz = H

        if self.is_diagonal:
            # Optimized for diagonal tensors
            Bx = self.mu.xx * Hx
            By = self.mu.yy * Hy
            Bz = self.mu.zz * Hz
        else:
            # Full tensor multiplication
            Bx = (
                self.mu_tensor[..., 0, 0] * Hx
                + self.mu_tensor[..., 0, 1] * Hy
                + self.mu_tensor[..., 0, 2] * Hz
            )
            By = (
                self.mu_tensor[..., 1, 0] * Hx
                + self.mu_tensor[..., 1, 1] * Hy
                + self.mu_tensor[..., 1, 2] * Hz
            )
            Bz = (
                self.mu_tensor[..., 2, 0] * Hx
                + self.mu_tensor[..., 2, 1] * Hy
                + self.mu_tensor[..., 2, 2] * Hz
            )

        return Bx, By, Bz

    def get_inverse_epsilon(self) -> np.ndarray:
        """
        Get inverse of permittivity tensor: [ε]^(-1)

        Used in E-field updates: E = [ε]^(-1) · D

        Returns
        -------
        ndarray
            Inverse permittivity tensor.
        """
        if self.is_diagonal:
            # For diagonal, inverse is simple
            inv_eps = np.zeros_like(self.epsilon_tensor)
            inv_eps[..., 0, 0] = 1.0 / self.epsilon.xx
            inv_eps[..., 1, 1] = 1.0 / self.epsilon.yy
            inv_eps[..., 2, 2] = 1.0 / self.epsilon.zz
            return inv_eps
        else:
            # Full matrix inversion
            if self.epsilon_tensor.ndim == 2:
                return np.linalg.inv(self.epsilon_tensor)
            else:
                # Spatially varying - invert at each point
                shape = self.epsilon_tensor.shape[:-2]
                inv_eps = np.zeros_like(self.epsilon_tensor)

                # This is inefficient - better to use specialized algorithms
                it = np.nditer(np.zeros(shape), flags=["multi_index"])
                for _ in it:
                    idx = it.multi_index
                    inv_eps[idx] = np.linalg.inv(self.epsilon_tensor[idx])

                return inv_eps

    def get_inverse_mu(self) -> np.ndarray:
        """
        Get inverse of permeability tensor: [μ]^(-1)

        Returns
        -------
        ndarray
            Inverse permeability tensor.
        """
        if self.is_diagonal:
            inv_mu = np.zeros_like(self.mu_tensor)
            inv_mu[..., 0, 0] = 1.0 / self.mu.xx
            inv_mu[..., 1, 1] = 1.0 / self.mu.yy
            inv_mu[..., 2, 2] = 1.0 / self.mu.zz
            return inv_mu
        else:
            if self.mu_tensor.ndim == 2:
                return np.linalg.inv(self.mu_tensor)
            else:
                shape = self.mu_tensor.shape[:-2]
                inv_mu = np.zeros_like(self.mu_tensor)

                it = np.nditer(np.zeros(shape), flags=["multi_index"])
                for _ in it:
                    idx = it.multi_index
                    inv_mu[idx] = np.linalg.inv(self.mu_tensor[idx])

                return inv_mu


def create_uniaxial_material(
    n_ordinary: float,
    n_extraordinary: float,
    optic_axis: Literal["x", "y", "z"] = "z",
    name: str = "",
) -> TensorMaterial:
    """
    Create a uniaxial material (e.g., liquid crystal).

    Uniaxial materials have different refractive indices parallel and
    perpendicular to the optic axis.

    Parameters
    ----------
    n_ordinary : float
        Ordinary refractive index (perpendicular to optic axis).
    n_extraordinary : float
        Extraordinary refractive index (parallel to optic axis).
    optic_axis : str
        Optic axis direction ('x', 'y', or 'z').
    name : str
        Material name.

    Returns
    -------
    TensorMaterial
        Uniaxial material with diagonal tensor.
    """
    eps_o = n_ordinary**2
    eps_e = n_extraordinary**2

    if optic_axis == "x":
        epsilon = TensorComponents(xx=eps_e, yy=eps_o, zz=eps_o)
    elif optic_axis == "y":
        epsilon = TensorComponents(xx=eps_o, yy=eps_e, zz=eps_o)
    else:  # z
        epsilon = TensorComponents(xx=eps_o, yy=eps_o, zz=eps_e)

    return TensorMaterial(
        epsilon=epsilon,
        name=name or f"Uniaxial_no{n_ordinary:.2f}_ne{n_extraordinary:.2f}",
    )


def create_biaxial_material(
    nx: float, ny: float, nz: float, name: str = ""
) -> TensorMaterial:
    """
    Create a biaxial material.

    Biaxial materials have three different principal refractive indices.

    Parameters
    ----------
    nx, ny, nz : float
        Refractive indices along x, y, z axes.
    name : str
        Material name.

    Returns
    -------
    TensorMaterial
        Biaxial material with diagonal tensor.
    """
    epsilon = TensorComponents(xx=nx**2, yy=ny**2, zz=nz**2)

    return TensorMaterial(
        epsilon=epsilon, name=name or f"Biaxial_nx{nx:.2f}_ny{ny:.2f}_nz{nz:.2f}"
    )


def create_rotated_tensor(
    base_tensor: TensorComponents,
    rotation_angles: Tuple[float, float, float],
    convention: str = "xyz",
) -> TensorComponents:
    """
    Rotate a material tensor by Euler angles.

    Useful for tilted anisotropic materials.

    Parameters
    ----------
    base_tensor : TensorComponents
        Original tensor in principal axes.
    rotation_angles : Tuple[float, float, float]
        Euler angles (in radians) for rotation.
    convention : str
        Euler angle convention ('xyz', 'zxz', etc.).

    Returns
    -------
    TensorComponents
        Rotated tensor.
    """
    from scipy.spatial.transform import Rotation

    # Create rotation matrix
    R = Rotation.from_euler(convention, rotation_angles).as_matrix()

    # Convert base tensor to matrix
    backend = get_backend()
    T_base = base_tensor.to_full_matrix(backend)

    # Apply rotation: T_rot = R · T_base · R^T
    T_rotated = R @ T_base @ R.T

    # Extract components
    rotated = TensorComponents(
        xx=T_rotated[0, 0],
        yy=T_rotated[1, 1],
        zz=T_rotated[2, 2],
        xy=T_rotated[0, 1],
        xz=T_rotated[0, 2],
        yz=T_rotated[1, 2],
        yx=T_rotated[1, 0],
        zx=T_rotated[2, 0],
        zy=T_rotated[2, 1],
    )

    return rotated


class AnisotropicUpdater:
    """
    Field updater for anisotropic materials.

    Handles FDTD field updates with tensor materials using:
    ∂D/∂t = ∇ × H,  where D = ε₀ [ε] · E
    ∂B/∂t = -∇ × E, where B = μ₀ [μ] · H

    Parameters
    ----------
    tensor_material : TensorMaterial
        Anisotropic material definition.
    dt : float
        Time step.
    backend : Backend, optional
        Computational backend.
    """

    def __init__(
        self,
        tensor_material: TensorMaterial,
        dt: float,
        backend: Optional[Union[str, Backend]] = None,
    ):
        self.material = tensor_material
        self.dt = dt

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
        self.eps0 = 8.854187817e-12
        self.mu0 = 4 * self.backend.pi * 1e-7

        # Precompute inverse tensors for field updates
        self.inv_epsilon = tensor_material.get_inverse_epsilon()
        self.inv_mu = tensor_material.get_inverse_mu()

    def update_e_from_curl_h(
        self, E: Tuple[any, any, any], curl_H: Tuple[any, any, any]
    ) -> Tuple[any, any, any]:
        """
        Update E-field with anisotropic permittivity.

        E^(n+1) = E^n + dt * [ε]^(-1) · (∇ × H) / ε₀

        Parameters
        ----------
        E : Tuple of arrays
            Current E-field (Ex, Ey, Ez).
        curl_H : Tuple of arrays
            Curl of H-field.

        Returns
        -------
        Tuple of arrays
            Updated E-field.
        """
        Ex, Ey, Ez = E
        curl_Hx, curl_Hy, curl_Hz = curl_H

        # D update: ∂D/∂t = ∇ × H
        # Then: E = [ε]^(-1) · D / ε₀

        if self.material.is_diagonal:
            # Optimized for diagonal tensors
            Ex_new = Ex + (self.dt / self.eps0) * curl_Hx / self.material.epsilon.xx
            Ey_new = Ey + (self.dt / self.eps0) * curl_Hy / self.material.epsilon.yy
            Ez_new = Ez + (self.dt / self.eps0) * curl_Hz / self.material.epsilon.zz
        else:
            # Full tensor update
            # dE = dt * [ε]^(-1) · (∇ × H) / ε₀
            dEx = (self.dt / self.eps0) * (
                self.inv_epsilon[..., 0, 0] * curl_Hx
                + self.inv_epsilon[..., 0, 1] * curl_Hy
                + self.inv_epsilon[..., 0, 2] * curl_Hz
            )
            dEy = (self.dt / self.eps0) * (
                self.inv_epsilon[..., 1, 0] * curl_Hx
                + self.inv_epsilon[..., 1, 1] * curl_Hy
                + self.inv_epsilon[..., 1, 2] * curl_Hz
            )
            dEz = (self.dt / self.eps0) * (
                self.inv_epsilon[..., 2, 0] * curl_Hx
                + self.inv_epsilon[..., 2, 1] * curl_Hy
                + self.inv_epsilon[..., 2, 2] * curl_Hz
            )

            Ex_new = Ex + dEx
            Ey_new = Ey + dEy
            Ez_new = Ez + dEz

        return Ex_new, Ey_new, Ez_new

    def update_h_from_curl_e(
        self, H: Tuple[any, any, any], curl_E: Tuple[any, any, any]
    ) -> Tuple[any, any, any]:
        """
        Update H-field with anisotropic permeability.

        H^(n+1) = H^n - dt * [μ]^(-1) · (∇ × E) / μ₀

        Parameters
        ----------
        H : Tuple of arrays
            Current H-field (Hx, Hy, Hz).
        curl_E : Tuple of arrays
            Curl of E-field.

        Returns
        -------
        Tuple of arrays
            Updated H-field.
        """
        Hx, Hy, Hz = H
        curl_Ex, curl_Ey, curl_Ez = curl_E

        if self.material.is_diagonal:
            # Optimized for diagonal tensors
            Hx_new = Hx - (self.dt / self.mu0) * curl_Ex / self.material.mu.xx
            Hy_new = Hy - (self.dt / self.mu0) * curl_Ey / self.material.mu.yy
            Hz_new = Hz - (self.dt / self.mu0) * curl_Ez / self.material.mu.zz
        else:
            # Full tensor update
            dHx = -(self.dt / self.mu0) * (
                self.inv_mu[..., 0, 0] * curl_Ex
                + self.inv_mu[..., 0, 1] * curl_Ey
                + self.inv_mu[..., 0, 2] * curl_Ez
            )
            dHy = -(self.dt / self.mu0) * (
                self.inv_mu[..., 1, 0] * curl_Ex
                + self.inv_mu[..., 1, 1] * curl_Ey
                + self.inv_mu[..., 1, 2] * curl_Ez
            )
            dHz = -(self.dt / self.mu0) * (
                self.inv_mu[..., 2, 0] * curl_Ex
                + self.inv_mu[..., 2, 1] * curl_Ey
                + self.inv_mu[..., 2, 2] * curl_Ez
            )

            Hx_new = Hx + dHx
            Hy_new = Hy + dHy
            Hz_new = Hz + dHz

        return Hx_new, Hy_new, Hz_new


def average_tensors_subpixel(
    tensor1: TensorComponents,
    tensor2: TensorComponents,
    fraction: float,
    averaging: Literal["arithmetic", "harmonic", "geometric"] = "arithmetic",
) -> TensorComponents:
    """
    Average two material tensors for subpixel smoothing.

    Improves accuracy at material interfaces by properly averaging
    material properties.

    Parameters
    ----------
    tensor1, tensor2 : TensorComponents
        Tensors to average.
    fraction : float
        Fraction of tensor2 (0 = pure tensor1, 1 = pure tensor2).
    averaging : str
        Averaging method: 'arithmetic', 'harmonic', or 'geometric'.

    Returns
    -------
    TensorComponents
        Averaged tensor.
    """

    def avg_component(c1, c2):
        """Average a single component."""
        if averaging == "arithmetic":
            return (1 - fraction) * c1 + fraction * c2
        elif averaging == "harmonic":
            return 1.0 / ((1 - fraction) / c1 + fraction / c2)
        elif averaging == "geometric":
            return c1 ** (1 - fraction) * c2**fraction
        else:
            raise ValueError(
                "averaging must be 'arithmetic', 'harmonic', or 'geometric'"
            )

    return TensorComponents(
        xx=avg_component(tensor1.xx, tensor2.xx),
        yy=avg_component(tensor1.yy, tensor2.yy),
        zz=avg_component(tensor1.zz, tensor2.zz),
        xy=avg_component(tensor1.xy, tensor2.xy),
        xz=avg_component(tensor1.xz, tensor2.xz),
        yz=avg_component(tensor1.yz, tensor2.yz),
    )
