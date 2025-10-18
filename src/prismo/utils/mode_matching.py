"""
Mode matching utilities for mode port analysis.

This module provides utilities for mode overlap integrals, normalization,
forward/backward mode separation, and phase calibration.
"""

from typing import Tuple, Optional, List
import numpy as np
from dataclasses import dataclass

from prismo.modes.solver import WaveguideMode


@dataclass
class ModeCoefficients:
    """
    Container for mode expansion coefficients.

    Attributes
    ----------
    forward : complex
        Forward-propagating mode amplitude.
    backward : complex
        Backward-propagating mode amplitude.
    total : complex
        Total mode amplitude.
    power_forward : float
        Forward-propagating power.
    power_backward : float
        Backward-propagating power.
    """

    forward: complex
    backward: complex
    total: complex
    power_forward: float
    power_backward: float


def compute_mode_overlap(
    Ex_sim: np.ndarray,
    Ey_sim: np.ndarray,
    Ez_sim: np.ndarray,
    Hx_sim: np.ndarray,
    Hy_sim: np.ndarray,
    Hz_sim: np.ndarray,
    mode: WaveguideMode,
    direction: str = "z",
    dx: float = 1.0,
    dy: float = 1.0,
) -> complex:
    """
    Compute overlap integral between simulation fields and a mode.

    The overlap integral for mode expansion is:
    a_m = ∫∫ (E_sim × H_mode* + E_mode* × H_sim) · n dA / (2 * P_mode)

    where P_mode is the mode normalization (power).

    Parameters
    ----------
    Ex_sim, Ey_sim, Ez_sim : ndarray
        Electric field components from simulation.
    Hx_sim, Hy_sim, Hz_sim : ndarray
        Magnetic field components from simulation.
    mode : WaveguideMode
        Waveguide mode to compute overlap with.
    direction : str
        Normal direction of the plane ('x', 'y', or 'z').
    dx, dy : float
        Grid spacing in transverse directions.

    Returns
    -------
    complex
        Mode overlap coefficient.

    Notes
    -----
    The mode fields must be on the same grid as the simulation fields.
    Use interpolation if necessary before calling this function.
    """
    # Get mode field components
    Ex_mode = mode.Ex
    Ey_mode = mode.Ey
    Ez_mode = mode.Ez
    Hx_mode = mode.Hx
    Hy_mode = mode.Hy
    Hz_mode = mode.Hz

    # Ensure fields are on same grid (broadcast if needed)
    shape = Ex_sim.shape
    if Ex_mode.shape != shape:
        # Simple resize - in practice should use proper interpolation
        Ex_mode = _resize_to_shape(Ex_mode, shape)
        Ey_mode = _resize_to_shape(Ey_mode, shape)
        Ez_mode = _resize_to_shape(Ez_mode, shape)
        Hx_mode = _resize_to_shape(Hx_mode, shape)
        Hy_mode = _resize_to_shape(Hy_mode, shape)
        Hz_mode = _resize_to_shape(Hz_mode, shape)

    # Compute Poynting vector components: S = E × H*
    # Direction determines which component we integrate

    if direction.lower() == "x":
        # S_x = Ey * Hz* - Ez * Hy*
        S_sim = Ey_sim * np.conj(Hz_mode) - Ez_sim * np.conj(Hy_mode)
        S_mode = Ey_mode * np.conj(Hz_sim) - Ez_mode * np.conj(Hy_sim)
    elif direction.lower() == "y":
        # S_y = Ez * Hx* - Ex * Hz*
        S_sim = Ez_sim * np.conj(Hx_mode) - Ex_sim * np.conj(Hz_mode)
        S_mode = Ez_mode * np.conj(Hx_sim) - Ex_mode * np.conj(Hz_sim)
    else:  # z direction
        # S_z = Ex * Hy* - Ey * Hx*
        S_sim = Ex_sim * np.conj(Hy_mode) - Ey_sim * np.conj(Hx_mode)
        S_mode = Ex_mode * np.conj(Hy_sim) - Ey_mode * np.conj(Hx_sim)

    # Compute overlap integral using reciprocity relation
    # a = 0.5 * ∫∫ (S_sim + S_mode) dA
    overlap = 0.5 * np.sum(S_sim + S_mode) * dx * dy

    # Normalize by mode power
    mode_power = compute_mode_power(mode, direction, dx, dy)

    if abs(mode_power) > 1e-20:
        coefficient = overlap / mode_power
    else:
        coefficient = 0.0

    return complex(coefficient)


def compute_mode_power(
    mode: WaveguideMode,
    direction: str = "z",
    dx: float = 1.0,
    dy: float = 1.0,
) -> float:
    """
    Compute mode power using Poynting vector integration.

    P = (1/2) Re[∫∫ E × H* · n dA]

    Parameters
    ----------
    mode : WaveguideMode
        Waveguide mode.
    direction : str
        Propagation direction ('x', 'y', or 'z').
    dx, dy : float
        Grid spacing.

    Returns
    -------
    float
        Mode power (normalized).
    """
    Ex, Ey, Ez = mode.Ex, mode.Ey, mode.Ez
    Hx, Hy, Hz = mode.Hx, mode.Hy, mode.Hz

    if direction.lower() == "x":
        S = Ey * np.conj(Hz) - Ez * np.conj(Hy)
    elif direction.lower() == "y":
        S = Ez * np.conj(Hx) - Ex * np.conj(Hz)
    else:  # z
        S = Ex * np.conj(Hy) - Ey * np.conj(Hx)

    power = 0.5 * np.real(np.sum(S)) * dx * dy

    return float(abs(power))


def normalize_mode_to_power(
    mode: WaveguideMode,
    target_power: float = 1.0,
    direction: str = "z",
    dx: float = 1.0,
    dy: float = 1.0,
) -> WaveguideMode:
    """
    Normalize mode fields to achieve target power.

    Parameters
    ----------
    mode : WaveguideMode
        Original mode.
    target_power : float
        Target power (default=1.0).
    direction : str
        Propagation direction.
    dx, dy : float
        Grid spacing.

    Returns
    -------
    WaveguideMode
        Normalized mode with scaled fields.
    """
    current_power = compute_mode_power(mode, direction, dx, dy)

    if abs(current_power) > 1e-20:
        scale_factor = np.sqrt(target_power / current_power)
    else:
        scale_factor = 1.0

    from prismo.modes.solver import WaveguideMode

    normalized_mode = WaveguideMode(
        mode_number=mode.mode_number,
        neff=mode.neff,
        frequency=mode.frequency,
        wavelength=mode.wavelength,
        Ex=mode.Ex * scale_factor,
        Ey=mode.Ey * scale_factor,
        Ez=mode.Ez * scale_factor,
        Hx=mode.Hx * scale_factor,
        Hy=mode.Hy * scale_factor,
        Hz=mode.Hz * scale_factor,
        x=mode.x,
        y=mode.y,
        power=target_power,
    )

    return normalized_mode


def separate_forward_backward(
    coefficient_left: complex,
    coefficient_right: complex,
    neff: complex,
    distance: float,
    wavelength: float,
) -> Tuple[complex, complex]:
    """
    Separate forward and backward propagating amplitudes using dual monitors.

    Uses phase information from two monitors separated by a known distance
    to separate forward and backward propagating mode amplitudes.

    Parameters
    ----------
    coefficient_left : complex
        Mode coefficient at left monitor.
    coefficient_right : complex
        Mode coefficient at right monitor.
    neff : complex
        Effective index of the mode.
    distance : float
        Distance between monitors.
    wavelength : float
        Wavelength.

    Returns
    -------
    Tuple[complex, complex]
        (forward_amplitude, backward_amplitude)

    Notes
    -----
    This method assumes:
    - Left monitor sees: a_L = a_fwd * exp(i*β*0) + a_bwd * exp(-i*β*0)
    - Right monitor sees: a_R = a_fwd * exp(i*β*d) + a_bwd * exp(-i*β*d)

    Solving this system gives the forward/backward amplitudes.
    """
    # Propagation constant
    beta = 2 * np.pi * neff.real / wavelength

    # Phase accumulated over distance
    phi = beta * distance

    # System of equations:
    # a_L = a_f + a_b
    # a_R = a_f * exp(i*phi) + a_b * exp(-i*phi)

    # Solve for a_f and a_b
    exp_phi = np.exp(1j * phi)
    exp_mphi = np.exp(-1j * phi)

    # Determinant
    det = exp_phi - exp_mphi

    if abs(det) > 1e-10:
        a_forward = (coefficient_right - coefficient_left * exp_mphi) / det
        a_backward = (coefficient_left * exp_phi - coefficient_right) / det
    else:
        # Monitors too close or wavelength mismatch
        # Fall back to simple assumption
        a_forward = coefficient_right
        a_backward = 0.0

    return a_forward, a_backward


def check_mode_orthogonality(
    mode1: WaveguideMode,
    mode2: WaveguideMode,
    direction: str = "z",
    dx: float = 1.0,
    dy: float = 1.0,
) -> float:
    """
    Check orthogonality between two modes.

    Modes should be orthogonal: ∫∫ (E1 × H2*) · n dA ≈ 0 for m ≠ n

    Parameters
    ----------
    mode1, mode2 : WaveguideMode
        Modes to check.
    direction : str
        Propagation direction.
    dx, dy : float
        Grid spacing.

    Returns
    -------
    float
        Orthogonality metric (0 = perfectly orthogonal).
    """
    Ex1, Ey1, Ez1 = mode1.Ex, mode1.Ey, mode1.Ez
    Hx1, Hy1, Hz1 = mode1.Hx, mode1.Hy, mode1.Hz

    Ex2, Ey2, Ez2 = mode2.Ex, mode2.Ey, mode2.Ez
    Hx2, Hy2, Hz2 = mode2.Hx, mode2.Hy, mode2.Hz

    # Ensure same shape
    if Ex1.shape != Ex2.shape:
        Ex2 = _resize_to_shape(Ex2, Ex1.shape)
        Ey2 = _resize_to_shape(Ey2, Ex1.shape)
        Ez2 = _resize_to_shape(Ez2, Ex1.shape)
        Hx2 = _resize_to_shape(Hx2, Ex1.shape)
        Hy2 = _resize_to_shape(Hy2, Ex1.shape)
        Hz2 = _resize_to_shape(Hz2, Ex1.shape)

    if direction.lower() == "x":
        S = Ey1 * np.conj(Hz2) - Ez1 * np.conj(Hy2)
    elif direction.lower() == "y":
        S = Ez1 * np.conj(Hx2) - Ex1 * np.conj(Hz2)
    else:
        S = Ex1 * np.conj(Hy2) - Ey1 * np.conj(Hx2)

    overlap = abs(np.sum(S) * dx * dy)

    # Normalize by mode powers
    P1 = compute_mode_power(mode1, direction, dx, dy)
    P2 = compute_mode_power(mode2, direction, dx, dy)

    if P1 > 1e-20 and P2 > 1e-20:
        orthogonality = overlap / np.sqrt(P1 * P2)
    else:
        orthogonality = 0.0

    return float(orthogonality)


def interpolate_mode_to_grid(
    mode: WaveguideMode,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> WaveguideMode:
    """
    Interpolate mode fields to a new grid.

    Parameters
    ----------
    mode : WaveguideMode
        Original mode on source grid.
    x_new, y_new : ndarray
        New grid coordinates.

    Returns
    -------
    WaveguideMode
        Mode interpolated to new grid.
    """
    from scipy.interpolate import RegularGridInterpolator

    # Original grid
    x_old = mode.x
    y_old = mode.y

    # Create interpolators for each component
    def interpolate_component(field_data):
        if field_data.ndim == 2:
            # Split real and imaginary parts
            interp_real = RegularGridInterpolator(
                (x_old, y_old),
                field_data.real,
                bounds_error=False,
                fill_value=0.0,
            )
            interp_imag = RegularGridInterpolator(
                (x_old, y_old),
                field_data.imag,
                bounds_error=False,
                fill_value=0.0,
            )

            # Create meshgrid for new coordinates
            X_new, Y_new = np.meshgrid(x_new, y_new, indexing="ij")
            points = np.column_stack([X_new.ravel(), Y_new.ravel()])

            # Interpolate
            field_real = interp_real(points).reshape(X_new.shape)
            field_imag = interp_imag(points).reshape(X_new.shape)

            return field_real + 1j * field_imag
        else:
            # 1D or 0D - just return zeros of appropriate shape
            return np.zeros((len(x_new), len(y_new)), dtype=complex)

    Ex_new = interpolate_component(mode.Ex)
    Ey_new = interpolate_component(mode.Ey)
    Ez_new = interpolate_component(mode.Ez)
    Hx_new = interpolate_component(mode.Hx)
    Hy_new = interpolate_component(mode.Hy)
    Hz_new = interpolate_component(mode.Hz)

    from prismo.modes.solver import WaveguideMode

    interpolated_mode = WaveguideMode(
        mode_number=mode.mode_number,
        neff=mode.neff,
        frequency=mode.frequency,
        wavelength=mode.wavelength,
        Ex=Ex_new,
        Ey=Ey_new,
        Ez=Ez_new,
        Hx=Hx_new,
        Hy=Hy_new,
        Hz=Hz_new,
        x=x_new,
        y=y_new,
        power=mode.power,
    )

    return interpolated_mode


def _resize_to_shape(array: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Resize array to target shape using simple interpolation.

    Parameters
    ----------
    array : ndarray
        Input array.
    target_shape : tuple
        Target shape.

    Returns
    -------
    ndarray
        Resized array.
    """
    from scipy.ndimage import zoom

    if array.shape == target_shape:
        return array

    # Calculate zoom factors
    zoom_factors = tuple(t / s for t, s in zip(target_shape, array.shape))

    # Handle complex arrays
    if np.iscomplexobj(array):
        real_resized = zoom(array.real, zoom_factors, order=1)
        imag_resized = zoom(array.imag, zoom_factors, order=1)
        return real_resized + 1j * imag_resized
    else:
        return zoom(array, zoom_factors, order=1)
