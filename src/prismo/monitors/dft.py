"""
Discrete Fourier Transform (DFT) monitors for frequency-domain field analysis.

This module implements on-the-fly DFT computation during time-domain simulations,
allowing efficient extraction of frequency-domain fields without storing all time steps.
"""

from typing import List, Tuple, Dict, Optional, Union
import numpy as np

from prismo.core.grid import YeeGrid
from prismo.core.fields import ElectromagneticFields, FieldComponent
from prismo.monitors.base import Monitor
from prismo.backends import Backend, get_backend


class DFTMonitor(Monitor):
    """
    Discrete Fourier Transform monitor for frequency-domain analysis.

    This monitor computes frequency-domain fields on-the-fly during time-domain
    simulation using:
    F(ω) = ∫ f(t) * exp(-jωt) dt

    Discretized as:
    F(ω) ≈ Σ f(n*dt) * exp(-jω*n*dt) * dt

    Parameters
    ----------
    center : Tuple[float, float, float]
        Physical coordinates of monitor center (x, y, z) in meters.
    size : Tuple[float, float, float]
        Physical dimensions of monitor (Lx, Ly, Lz) in meters.
    frequencies : List[float]
        List of frequencies to monitor (Hz).
    components : List[str], optional
        Field components to record. Default is all E components.
    name : str, optional
        Monitor name.
    backend : Backend, optional
        Computational backend to use.
    """

    def __init__(
        self,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        frequencies: List[float],
        components: Optional[List[str]] = None,
        name: Optional[str] = None,
        backend: Optional[Union[str, Backend]] = None,
    ):
        super().__init__(center, size, name)

        self.frequencies = np.array(frequencies)
        self.omega = 2 * np.pi * self.frequencies  # Angular frequencies

        if components is None:
            self.components = ["Ex", "Ey", "Ez"]
        else:
            self.components = components

        # Initialize backend
        if isinstance(backend, str):
            self.backend = get_backend(backend)
        elif isinstance(backend, Backend):
            self.backend = backend
        elif backend is None:
            self.backend = get_backend()
        else:
            raise TypeError("backend must be a Backend instance or string name")

        # Storage for DFT accumulation (complex-valued)
        self._dft_data: Dict[str, np.ndarray] = {}
        self._time_steps = 0
        self._dt = None

    def initialize(self, grid: YeeGrid) -> None:
        """Initialize the DFT monitor on the grid."""
        super().initialize(grid)

        # Initialize DFT accumulation arrays
        for component in self.components:
            # Get shape for this component in the monitor region
            shape = self._get_component_shape(component)

            # Create complex array: (n_frequencies, *spatial_dims)
            dft_shape = (len(self.frequencies),) + shape
            self._dft_data[component] = np.zeros(dft_shape, dtype=np.complex128)

    def _get_component_shape(self, component: str) -> Tuple[int, ...]:
        """Get the spatial shape for a field component in the monitor region."""
        # This returns the shape of the field in the monitored region
        # For simplicity, we'll use the grid shape
        # In practice, this would be determined by the monitor region slices

        if self._grid is None:
            raise RuntimeError("Monitor must be initialized first")

        # Get the component indices
        indices = self._monitor_region[component]

        # For now, assume a simple point or line monitor
        # Full implementation would extract proper shapes from indices
        return (10, 10)  # Placeholder

    def update(self, fields: ElectromagneticFields, time: float, dt: float) -> None:
        """
        Update DFT accumulation with current field values.

        Parameters
        ----------
        fields : ElectromagneticFields
            Current electromagnetic fields.
        time : float
            Current simulation time (s).
        dt : float
            Time step (s).
        """
        if self._dt is None:
            self._dt = dt

        # For each frequency, accumulate: F(ω) += f(t) * exp(-jωt) * dt
        for component in self.components:
            # Get current field values in monitor region
            field_data = self._extract_field_data(fields, component)

            # Update DFT for each frequency
            for freq_idx, omega in enumerate(self.omega):
                # Compute phase factor: exp(-j * omega * time)
                phase = np.exp(-1j * omega * time)

                # Accumulate: DFT += field * phase * dt
                self._dft_data[component][freq_idx] += field_data * phase * dt

        self._time_steps += 1

    def _extract_field_data(
        self, fields: ElectromagneticFields, component: str
    ) -> np.ndarray:
        """Extract field data from the monitor region."""
        # Get field component
        field = fields[component]

        # Extract data from monitor region
        # For now, return a simplified extraction
        # Full implementation would use proper slicing from _monitor_region

        # Convert to numpy if using GPU backend
        if hasattr(fields, "backend"):
            field_np = fields.backend.to_numpy(field)
        else:
            field_np = np.asarray(field)

        # Return a subset (placeholder - should use monitor region)
        if field_np.ndim >= 2:
            return field_np[:10, :10]
        else:
            return field_np[:10]

    def get_frequency_data(
        self, component: str, frequency_index: Optional[int] = None
    ) -> Union[np.ndarray, Dict[float, np.ndarray]]:
        """
        Get frequency-domain field data.

        Parameters
        ----------
        component : str
            Field component ('Ex', 'Ey', 'Ez', etc.).
        frequency_index : int, optional
            Index of frequency to retrieve. If None, returns all frequencies.

        Returns
        -------
        ndarray or dict
            Complex frequency-domain field data.
            If frequency_index is None, returns dict mapping frequency to data.
        """
        if component not in self._dft_data:
            raise ValueError(f"Component {component} not monitored")

        if frequency_index is None:
            # Return dict of all frequencies
            result = {}
            for i, freq in enumerate(self.frequencies):
                result[freq] = self._dft_data[component][i]
            return result
        else:
            return self._dft_data[component][frequency_index]

    def get_intensity(self, component: str, frequency_index: int) -> np.ndarray:
        """
        Get intensity (|E|²) at a specific frequency.

        Parameters
        ----------
        component : str
            Field component.
        frequency_index : int
            Frequency index.

        Returns
        -------
        ndarray
            Field intensity.
        """
        field = self.get_frequency_data(component, frequency_index)
        return np.abs(field) ** 2

    def get_power_spectrum(self, component: str, normalize: bool = True) -> np.ndarray:
        """
        Get power spectrum vs frequency.

        Parameters
        ----------
        component : str
            Field component.
        normalize : bool
            Whether to normalize to maximum value.

        Returns
        -------
        ndarray
            Power spectrum with shape (n_frequencies,).
        """
        spectrum = np.zeros(len(self.frequencies))

        for i in range(len(self.frequencies)):
            intensity = self.get_intensity(component, i)
            spectrum[i] = np.sum(intensity)

        if normalize and np.max(spectrum) > 0:
            spectrum = spectrum / np.max(spectrum)

        return spectrum

    def get_transmission_spectrum(
        self, reference_power: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Calculate transmission spectrum (normalized power flow).

        Parameters
        ----------
        reference_power : ndarray, optional
            Reference power for normalization. If None, normalizes to max.

        Returns
        -------
        ndarray
            Transmission spectrum.
        """
        # Calculate total power for each frequency
        power = np.zeros(len(self.frequencies))

        for i in range(len(self.frequencies)):
            # Sum power from all E-field components
            for comp in ["Ex", "Ey", "Ez"]:
                if comp in self._dft_data:
                    intensity = self.get_intensity(comp, i)
                    power[i] += np.sum(intensity)

        # Normalize
        if reference_power is not None:
            transmission = power / reference_power
        elif np.max(power) > 0:
            transmission = power / np.max(power)
        else:
            transmission = power

        return transmission
