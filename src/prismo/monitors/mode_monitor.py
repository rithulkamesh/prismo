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
        a_m = ∫∫ (E_sim × H_mode* + E_mode* × H_sim) · n dA / (2 * P_mode)

        Returns
        -------
        List[complex]
            Mode coefficients for each mode.
        """
        from prismo.utils.mode_matching import compute_mode_overlap

        # Extract field components at monitor plane
        Ex_sim = self._extract_field(fields, "Ex")
        Ey_sim = self._extract_field(fields, "Ey")
        Ez_sim = self._extract_field(fields, "Ez")
        Hx_sim = self._extract_field(fields, "Hx")
        Hy_sim = self._extract_field(fields, "Hy")
        Hz_sim = self._extract_field(fields, "Hz")

        coeffs = []

        # Get grid spacing (assuming uniform grid)
        dx = 1.0  # Will be updated from actual grid
        dy = 1.0

        for mode_idx, mode in enumerate(self.modes):
            # Use mode_matching utility for accurate overlap
            coeff = compute_mode_overlap(
                Ex_sim,
                Ey_sim,
                Ez_sim,
                Hx_sim,
                Hy_sim,
                Hz_sim,
                mode,
                direction=self.direction,
                dx=dx,
                dy=dy,
            )
            coeffs.append(coeff)

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

    def compute_s_parameters(
        self,
        source_mode_index: int = 0,
        source_power: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Compute S-parameters from mode coefficients.

        For a two-port device with input and output monitors:
        - S11: Reflection coefficient at input port
        - S21: Transmission coefficient from port 1 to port 2

        Parameters
        ----------
        source_mode_index : int
            Index of the excited mode.
        source_power : float
            Input power (for normalization).

        Returns
        -------
        Dict[str, ndarray]
            Dictionary with 'S11', 'S21', etc. vs frequency.

        Notes
        -----
        This method assumes this monitor is the output/reflection monitor.
        For full S-parameters, you need multiple monitors.
        """
        if self.frequencies is None:
            raise RuntimeError("Frequency-domain analysis required")

        s_params = {}

        # Get forward and backward amplitudes
        # Simplified: assumes monitor can separate directions
        for mode_idx in range(len(self.modes)):
            # Transmission/reflection relative to source
            coeff_freq = self._mode_coeffs_freq[mode_idx]

            # Normalize by source
            if source_power > 0:
                s_param = coeff_freq / np.sqrt(source_power)
            else:
                s_param = coeff_freq

            # Name the parameter
            param_name = f"S_{mode_idx+1}{source_mode_index+1}"
            s_params[param_name] = s_param

        return s_params

    def compute_s_matrix(
        self,
        other_monitor: "ModeExpansionMonitor",
        frequency_index: int,
        source_mode_index: int = 0,
    ) -> np.ndarray:
        """
        Compute full S-matrix between this and another monitor.

        Parameters
        ----------
        other_monitor : ModeExpansionMonitor
            The other port monitor.
        frequency_index : int
            Frequency index to compute S-matrix at.
        source_mode_index : int
            Index of excited mode.

        Returns
        -------
        ndarray
            2x2 or NxN S-matrix.

        Examples
        --------
        >>> # With input and output monitors
        >>> S = output_monitor.compute_s_matrix(input_monitor, freq_idx=0)
        >>> S11 = S[0, 0]  # Reflection
        >>> S21 = S[1, 0]  # Transmission
        """
        n_modes = len(self.modes)
        S_matrix = np.zeros((n_modes, n_modes), dtype=complex)

        # Fill S-matrix elements
        # S[i,j] = output_mode_i / input_mode_j
        for i in range(n_modes):
            for j in range(n_modes):
                if j == source_mode_index:
                    # This mode was excited
                    # Reflection at input
                    refl_coeff = other_monitor._mode_coeffs_freq[i][frequency_index]
                    # Transmission at output
                    trans_coeff = self._mode_coeffs_freq[i][frequency_index]

                    S_matrix[i, j] = trans_coeff  # Simplified

        return S_matrix
