"""
Flux monitors for calculating power flow through surfaces.

This module implements monitors for computing electromagnetic power flow
(Poynting vector) through specified surfaces in the simulation domain.
"""

from typing import Tuple, List, Optional, Union, Literal
import numpy as np

from prismo.core.grid import YeeGrid
from prismo.core.fields import ElectromagneticFields
from prismo.monitors.base import Monitor
from prismo.backends import Backend, get_backend


class FluxMonitor(Monitor):
    """
    Flux monitor for computing power flow through a surface.

    Computes the Poynting vector S = E × H and integrates over a surface
    to determine power flow. Can compute both time-domain and frequency-domain flux.

    Parameters
    ----------
    center : Tuple[float, float, float]
        Physical coordinates of monitor center.
    size : Tuple[float, float, float]
        Physical dimensions of monitor surface.
    direction : str
        Normal direction of flux surface ('x', 'y', or 'z').
    name : str, optional
        Monitor name.
    frequencies : List[float], optional
        Frequencies for frequency-domain flux computation.
    backend : Backend, optional
        Computational backend.
    """

    def __init__(
        self,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        direction: Literal["x", "y", "z"],
        name: Optional[str] = None,
        frequencies: Optional[List[float]] = None,
        backend: Optional[Union[str, Backend]] = None,
    ):
        super().__init__(center, size, name)

        self.direction = direction.lower()
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

        # Time-domain power flow storage
        self._power_flow_history: List[float] = []
        self._time_history: List[float] = []

        # Frequency-domain setup
        self.frequencies = frequencies
        if frequencies is not None:
            self.omega = 2 * np.pi * np.array(frequencies)
            self._dft_ex = None  # Will be initialized
            self._dft_ey = None
            self._dft_ez = None
            self._dft_hx = None
            self._dft_hy = None
            self._dft_hz = None

        # Physical constants
        self.eps0 = 8.854187817e-12
        self.mu0 = 4 * self.backend.pi * 1e-7

    def initialize(self, grid: YeeGrid) -> None:
        """Initialize the flux monitor on the grid."""
        super().initialize(grid)

        # Initialize frequency-domain DFT arrays if needed
        if self.frequencies is not None:
            shape = self._get_surface_shape()
            n_freq = len(self.frequencies)

            # Initialize complex DFT arrays for all field components
            self._dft_ex = np.zeros((n_freq,) + shape, dtype=np.complex128)
            self._dft_ey = np.zeros((n_freq,) + shape, dtype=np.complex128)
            self._dft_ez = np.zeros((n_freq,) + shape, dtype=np.complex128)
            self._dft_hx = np.zeros((n_freq,) + shape, dtype=np.complex128)
            self._dft_hy = np.zeros((n_freq,) + shape, dtype=np.complex128)
            self._dft_hz = np.zeros((n_freq,) + shape, dtype=np.complex128)

    def _get_surface_shape(self) -> Tuple[int, ...]:
        """Get the shape of the flux surface."""
        # Placeholder - should extract from monitor region
        return (10, 10)

    def update(self, fields: ElectromagneticFields, time: float, dt: float) -> None:
        """
        Update flux monitor with current fields.

        Computes instantaneous power flow and updates DFT if frequencies are specified.

        Parameters
        ----------
        fields : ElectromagneticFields
            Current electromagnetic fields.
        time : float
            Current simulation time.
        dt : float
            Time step.
        """
        # Compute instantaneous power flow
        power = self._compute_power_flow(fields)
        self._power_flow_history.append(power)
        self._time_history.append(time)

        # Update frequency-domain DFT if needed
        if self.frequencies is not None:
            self._update_dft(fields, time, dt)

    def _compute_power_flow(self, fields: ElectromagneticFields) -> float:
        """
        Compute instantaneous power flow through the surface.

        Power = ∫∫ S·n dA where S = E × H is the Poynting vector.
        """
        # Extract field components at the flux surface
        # For simplicity, assume fields are at the same spatial points
        # In practice, need to interpolate to common points

        Ex = self._extract_field_component(fields, "Ex")
        Ey = self._extract_field_component(fields, "Ey")
        Ez = self._extract_field_component(fields, "Ez")
        Hx = self._extract_field_component(fields, "Hx")
        Hy = self._extract_field_component(fields, "Hy")
        Hz = self._extract_field_component(fields, "Hz")

        # Compute Poynting vector components: S = E × H
        Sx = Ey * Hz - Ez * Hy
        Sy = Ez * Hx - Ex * Hz
        Sz = Ex * Hy - Ey * Hx

        # Select component normal to surface
        if self.direction == "x":
            S_normal = Sx
        elif self.direction == "y":
            S_normal = Sy
        else:  # z
            S_normal = Sz

        # Integrate over surface
        # Get grid spacing for surface area element
        dx, dy, dz = self._grid.spacing

        if self.direction == "x":
            dA = dy * dz
        elif self.direction == "y":
            dA = dx * dz
        else:  # z
            dA = dx * dy

        # Sum power flow
        power = np.sum(S_normal) * dA

        return float(power)

    def _extract_field_component(
        self, fields: ElectromagneticFields, component: str
    ) -> np.ndarray:
        """Extract field component from monitor region."""
        field = fields[component]

        # Convert to numpy if needed
        if hasattr(fields, "backend"):
            field_np = fields.backend.to_numpy(field)
        else:
            field_np = np.asarray(field)

        # Extract from monitor region (placeholder)
        if field_np.ndim >= 2:
            return field_np[:10, :10]
        return field_np[:10]

    def _update_dft(
        self, fields: ElectromagneticFields, time: float, dt: float
    ) -> None:
        """Update frequency-domain DFT accumulation."""
        # Extract field components
        Ex = self._extract_field_component(fields, "Ex")
        Ey = self._extract_field_component(fields, "Ey")
        Ez = self._extract_field_component(fields, "Ez")
        Hx = self._extract_field_component(fields, "Hx")
        Hy = self._extract_field_component(fields, "Hy")
        Hz = self._extract_field_component(fields, "Hz")

        # Update DFT for each frequency
        for i, omega in enumerate(self.omega):
            phase = np.exp(-1j * omega * time)

            self._dft_ex[i] += Ex * phase * dt
            self._dft_ey[i] += Ey * phase * dt
            self._dft_ez[i] += Ez * phase * dt
            self._dft_hx[i] += Hx * phase * dt
            self._dft_hy[i] += Hy * phase * dt
            self._dft_hz[i] += Hz * phase * dt

    def get_time_domain_power(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get time-domain power flow history.

        Returns
        -------
        Tuple[ndarray, ndarray]
            (time_array, power_array)
        """
        return np.array(self._time_history), np.array(self._power_flow_history)

    def get_frequency_domain_power(
        self, frequency_index: Optional[int] = None
    ) -> Union[float, np.ndarray]:
        """
        Get frequency-domain power flow.

        Computes power from frequency-domain fields:
        P(ω) = (1/2) Re[∫∫ (E × H*) · n dA]

        Parameters
        ----------
        frequency_index : int, optional
            Frequency index. If None, returns array for all frequencies.

        Returns
        -------
        float or ndarray
            Power flow at specified frequency(ies).
        """
        if self.frequencies is None:
            raise RuntimeError("No frequencies specified for this monitor")

        dx, dy, dz = self._grid.spacing
        if self.direction == "x":
            dA = dy * dz
        elif self.direction == "y":
            dA = dx * dz
        else:
            dA = dx * dy

        def compute_power_at_freq(idx: int) -> float:
            """Compute power for a single frequency."""
            # Get field DFTs
            Ex = self._dft_ex[idx]
            Ey = self._dft_ey[idx]
            Ez = self._dft_ez[idx]
            Hx = self._dft_hx[idx]
            Hy = self._dft_hy[idx]
            Hz = self._dft_hz[idx]

            # Compute Poynting vector: S = (1/2) * Re[E × H*]
            Sx = 0.5 * np.real(Ey * np.conj(Hz) - Ez * np.conj(Hy))
            Sy = 0.5 * np.real(Ez * np.conj(Hx) - Ex * np.conj(Hz))
            Sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))

            # Select component normal to surface
            if self.direction == "x":
                S_normal = Sx
            elif self.direction == "y":
                S_normal = Sy
            else:
                S_normal = Sz

            # Integrate
            power = np.sum(S_normal) * dA
            return float(power)

        if frequency_index is None:
            # Return array for all frequencies
            return np.array(
                [compute_power_at_freq(i) for i in range(len(self.frequencies))]
            )
        else:
            return compute_power_at_freq(frequency_index)

    def get_transmission(
        self, reference_power: Optional[Union[float, np.ndarray]] = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate transmission (normalized power).

        Parameters
        ----------
        reference_power : float or ndarray, optional
            Reference power for normalization.

        Returns
        -------
        float or ndarray
            Transmission coefficient(s).
        """
        if self.frequencies is not None:
            power = self.get_frequency_domain_power()
        else:
            _, power_history = self.get_time_domain_power()
            power = np.mean(power_history) if len(power_history) > 0 else 0.0

        if reference_power is not None:
            return power / reference_power
        else:
            return power
