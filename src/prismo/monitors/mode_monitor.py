"""
Mode expansion monitors for decomposing fields into waveguide modes.

This module implements monitors that decompose electromagnetic fields into
waveguide mode coefficients using overlap integrals.
"""

from typing import List, Tuple, Dict, Optional, Union
import numpy as np

from prismo.core.grid import YeeGrid
from prismo.core.fields import ElectromagneticFields
from prismo.monitors.base import Monitor
from prismo.modes.solver import WaveguideMode
from prismo.backends import Backend, get_backend


class ModeExpansionMonitor(Monitor):
    """
    Mode expansion monitor for decomposing fields into mode coefficients.

    Computes overlap integrals between simulation fields and waveguide modes
    to extract forward and backward propagating mode amplitudes.

    The overlap integral is:
    a_m = ∫∫ (E_sim × H_mode* - E_mode* × H_sim) · n dA

    Parameters
    ----------
    center : Tuple[float, float, float]
        Physical coordinates of monitor center.
    size : Tuple[float, float, float]
        Physical dimensions of monitor.
    modes : List[WaveguideMode]
        List of modes to decompose into.
    direction : str
        Normal direction of monitor plane ('x', 'y', or 'z').
    frequencies : List[float], optional
        Frequencies for frequency-domain decomposition.
    name : str, optional
        Monitor name.
    backend : Backend, optional
        Computational backend.
    """

    def __init__(
        self,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        modes: List[WaveguideMode],
        direction: str = "x",
        frequencies: Optional[List[float]] = None,
        name: Optional[str] = None,
        backend: Optional[Union[str, Backend]] = None,
    ):
        super().__init__(center, size, name)

        self.modes = modes
        self.direction = direction.lower()
        self.frequencies = frequencies

        if self.direction not in ["x", "y", "z"]:
            raise ValueError("direction must be 'x', 'y', or 'z'")

        # Initialize backend
        if isinstance(backend, str):
            self.backend = get_backend(backend)
        elif isinstance(backend, Backend):
            self.backend = backend
        elif backend is None:
            self.backend = get_backend()
        else:
            raise TypeError("backend must be a Backend instance or string name")

        # Storage for mode coefficients
        self._mode_coeffs_time: Dict[int, List[complex]] = {
            i: [] for i in range(len(modes))
        }
        self._time_points: List[float] = []

        # Frequency-domain storage
        if frequencies is not None:
            self.omega = 2 * np.pi * np.array(frequencies)
            self._mode_coeffs_freq: Dict[int, np.ndarray] = {
                i: np.zeros(len(frequencies), dtype=complex) for i in range(len(modes))
            }

        # Mode profiles interpolated to monitor grid
        self._interpolated_modes: List[Dict[str, np.ndarray]] = []

    def initialize(self, grid: YeeGrid) -> None:
        """Initialize the mode expansion monitor on the grid."""
        super().initialize(grid)

        # Interpolate mode profiles to monitor plane
        self._interpolate_modes_to_monitor()

        # Compute mode normalization factors
        self._compute_normalization()

    def _interpolate_modes_to_monitor(self) -> None:
        """
        Interpolate mode profiles to the monitor plane grid.

        The modes may be defined on a different grid than the simulation,
        so we need to interpolate.
        """
        from scipy.interpolate import RegularGridInterpolator

        for mode in self.modes:
            # Create interpolators for mode fields
            mode_interp = {}

            # Simplified: assume modes are on same grid as monitor
            # Full implementation would interpolate properly
            mode_interp["Ex"] = mode.Ex
            mode_interp["Ey"] = mode.Ey
            mode_interp["Ez"] = mode.Ez
            mode_interp["Hx"] = mode.Hx
            mode_interp["Hy"] = mode.Hy
            mode_interp["Hz"] = mode.Hz

            self._interpolated_modes.append(mode_interp)

    def _compute_normalization(self) -> None:
        """
        Compute mode normalization factors.

        Modes should be normalized such that:
        ∫∫ (E_m × H_m*) · n dA = 1
        """
        # Placeholder for normalization computation
        self._mode_norms = np.ones(len(self.modes))

    def update(self, fields: ElectromagneticFields, time: float, dt: float) -> None:
        """
        Update mode expansion with current fields.

        Computes overlap integrals and stores mode coefficients.

        Parameters
        ----------
        fields : ElectromagneticFields
            Current electromagnetic fields.
        time : float
            Current simulation time.
        dt : float
            Time step.
        """
        # Compute mode coefficients at current time
        coeffs = self._compute_mode_coefficients(fields)

        # Store time-domain coefficients
        for mode_idx, coeff in enumerate(coeffs):
            self._mode_coeffs_time[mode_idx].append(coeff)

        self._time_points.append(time)

        # Update frequency-domain DFT if needed
        if self.frequencies is not None:
            for mode_idx, coeff in enumerate(coeffs):
                for freq_idx, omega in enumerate(self.omega):
                    phase = np.exp(-1j * omega * time)
                    self._mode_coeffs_freq[mode_idx][freq_idx] += coeff * phase * dt

    def _compute_mode_coefficients(
        self, fields: ElectromagneticFields
    ) -> List[complex]:
        """
        Compute mode coefficients using overlap integrals.

        For each mode m:
        a_m = ∫∫ (E_sim × H_mode* + E_mode* × H_sim) · n dA

        Returns
        -------
        List[complex]
            Mode coefficients for each mode.
        """
        # Extract field components at monitor plane
        Ex_sim = self._extract_field(fields, "Ex")
        Ey_sim = self._extract_field(fields, "Ey")
        Ez_sim = self._extract_field(fields, "Ez")
        Hx_sim = self._extract_field(fields, "Hx")
        Hy_sim = self._extract_field(fields, "Hy")
        Hz_sim = self._extract_field(fields, "Hz")

        coeffs = []

        for mode_idx, mode_fields in enumerate(self._interpolated_modes):
            # Get mode fields
            Ex_mode = mode_fields["Ex"]
            Ey_mode = mode_fields["Ey"]
            Ez_mode = mode_fields["Ez"]
            Hx_mode = mode_fields["Hx"]
            Hy_mode = mode_fields["Hy"]
            Hz_mode = mode_fields["Hz"]

            # Compute overlap integral (simplified - assumes 2D)
            # Full implementation would properly handle 3D and different directions

            # Power flux: S = E × H*
            # Component normal to monitor
            if self.direction == "x":
                # S_x = Ey * Hz* - Ez * Hy*
                S_sim = Ey_sim * np.conj(Hz_mode) - Ez_sim * np.conj(Hy_mode)
                S_mode = Ey_mode * np.conj(Hz_sim) - Ez_mode * np.conj(Hy_sim)
            elif self.direction == "y":
                S_sim = Ez_sim * np.conj(Hx_mode) - Ex_sim * np.conj(Hz_mode)
                S_mode = Ez_mode * np.conj(Hx_sim) - Ex_mode * np.conj(Hz_sim)
            else:  # z
                S_sim = Ex_sim * np.conj(Hy_mode) - Ey_sim * np.conj(Hx_mode)
                S_mode = Ex_mode * np.conj(Hy_sim) - Ey_mode * np.conj(Hx_sim)

            # Overlap integral
            overlap = 0.5 * np.sum(S_sim + S_mode)

            # Normalize
            coeff = overlap / self._mode_norms[mode_idx]
            coeffs.append(complex(coeff))

        return coeffs

    def _extract_field(
        self, fields: ElectromagneticFields, component: str
    ) -> np.ndarray:
        """Extract field component at monitor plane."""
        field = fields[component]

        # Convert to numpy
        if hasattr(fields, "backend"):
            field_np = fields.backend.to_numpy(field)
        else:
            field_np = np.asarray(field)

        # Extract monitor region (placeholder)
        if field_np.ndim >= 2:
            return field_np[:10, :10]
        return field_np[:10]

    def get_mode_coefficient(
        self, mode_index: int, domain: str = "time"
    ) -> Union[np.ndarray, List[complex]]:
        """
        Get mode coefficient time series or frequency spectrum.

        Parameters
        ----------
        mode_index : int
            Mode index.
        domain : str
            'time' or 'frequency'.

        Returns
        -------
        array or list
            Mode coefficients.
        """
        if domain == "time":
            return self._mode_coeffs_time[mode_index]
        elif domain == "frequency":
            if self.frequencies is None:
                raise RuntimeError("No frequencies specified")
            return self._mode_coeffs_freq[mode_index]
        else:
            raise ValueError("domain must be 'time' or 'frequency'")

    def separate_forward_backward(
        self, mode_index: int, frequency_index: int
    ) -> Tuple[complex, complex]:
        """
        Separate forward and backward propagating mode amplitudes.

        Uses phase information to separate directions.

        Parameters
        ----------
        mode_index : int
            Mode index.
        frequency_index : int
            Frequency index.

        Returns
        -------
        Tuple[complex, complex]
            (forward_amplitude, backward_amplitude)
        """
        if self.frequencies is None:
            raise RuntimeError("Frequency-domain decomposition required")

        # Get mode coefficient at this frequency
        coeff = self._mode_coeffs_freq[mode_index][frequency_index]

        # Simplified separation (assumes single direction)
        # Full implementation would analyze phase vs position
        forward = coeff
        backward = 0.0

        return forward, backward

    def get_mode_power(
        self, mode_index: int, frequency_index: Optional[int] = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate mode power.

        Parameters
        ----------
        mode_index : int
            Mode index.
        frequency_index : int, optional
            Frequency index for frequency-domain power.

        Returns
        -------
        float or array
            Mode power.
        """
        if frequency_index is not None:
            # Frequency-domain power
            coeff = self._mode_coeffs_freq[mode_index][frequency_index]
            return float(np.abs(coeff) ** 2)
        else:
            # Time-domain power (average)
            coeffs = np.array(self._mode_coeffs_time[mode_index])
            power = np.abs(coeffs) ** 2
            return np.mean(power)

    def get_mode_transmission(
        self, mode_index: int, reference_power: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate mode transmission vs frequency.

        Parameters
        ----------
        mode_index : int
            Mode index.
        reference_power : float, optional
            Reference power for normalization.

        Returns
        -------
        ndarray
            Transmission coefficient vs frequency.
        """
        if self.frequencies is None:
            raise RuntimeError("Frequency-domain analysis required")

        # Get power for this mode at all frequencies
        power = np.abs(self._mode_coeffs_freq[mode_index]) ** 2

        if reference_power is not None:
            return power / reference_power
        else:
            return power / np.max(power) if np.max(power) > 0 else power
