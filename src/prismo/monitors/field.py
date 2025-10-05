"""
Field monitor implementation for FDTD simulations.

This module implements field monitors for recording electromagnetic field
data during FDTD simulations, with options for time-domain and frequency-domain
analysis.
"""

from typing import Tuple, Dict, List, Optional, Union, Literal, Set
import numpy as np
from dataclasses import dataclass, field

from prismo.core.grid import YeeGrid
from prismo.core.fields import ElectromagneticFields, FieldComponent
from prismo.monitors.base import Monitor


class FieldMonitor(Monitor):
    """
    Monitor for recording electromagnetic field data.

    Parameters
    ----------
    center : Tuple[float, float, float]
        Physical coordinates of the monitor center (x, y, z) in meters.
    size : Tuple[float, float, float]
        Physical dimensions of the monitor region (Lx, Ly, Lz) in meters.
    components : List[str] or str, optional
        Field components to record, default="all".
    name : str, optional
        Name of the monitor for identification.
    time_domain : bool, optional
        Whether to record time-domain data, default=True.
    frequencies : List[float], optional
        Frequencies to record in the frequency domain, in Hz.
    """

    def __init__(
        self,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        components: Union[List[FieldComponent], str] = "all",
        name: Optional[str] = None,
        time_domain: bool = True,
        frequencies: Optional[List[float]] = None,
    ):
        super().__init__(center=center, size=size, name=name)

        # Set components to record
        all_components = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
        if components == "all":
            self.components = all_components
        elif components == "E":
            self.components = ["Ex", "Ey", "Ez"]
        elif components == "H":
            self.components = ["Hx", "Hy", "Hz"]
        else:
            # Check that all components are valid
            for comp in components:
                if comp not in all_components:
                    raise ValueError(f"Invalid field component: {comp}")
            self.components = components

        self.time_domain = time_domain
        self.frequencies = frequencies if frequencies is not None else []

        # Data storage
        self._time_data: Dict[str, List[np.ndarray]] = {}
        self._time_points: List[float] = []
        self._freq_data: Dict[str, Dict[float, np.ndarray]] = {}

        # Initialize time domain storage if enabled
        if self.time_domain:
            for comp in self.components:
                self._time_data[comp] = []

        # Initialize frequency domain storage if frequencies provided
        if self.frequencies:
            for comp in self.components:
                self._freq_data[comp] = {}
                for freq in self.frequencies:
                    self._freq_data[comp][freq] = None

        # Will store component shapes for data initialization
        self._component_shapes: Dict[str, Tuple[int, ...]] = {}

    def initialize(self, grid: YeeGrid) -> None:
        """
        Initialize the field monitor on a specific grid.

        Parameters
        ----------
        grid : YeeGrid
            The grid on which to initialize the monitor.
        """
        super().initialize(grid)

        # Store the shape of each component for initializing frequency domain data
        for comp in self.components:
            indices = self._monitor_region[comp]
            shape = tuple(idx.size for idx in indices)
            self._component_shapes[comp] = shape

        # Initialize frequency domain data arrays
        if self.frequencies:
            for comp in self.components:
                shape = self._component_shapes[comp]
                for freq in self.frequencies:
                    self._freq_data[comp][freq] = np.zeros(shape, dtype=np.complex128)

    def update(self, fields: ElectromagneticFields, time: float, dt: float) -> None:
        """
        Record field data at the current time step.

        Parameters
        ----------
        fields : ElectromagneticFields
            Electromagnetic fields to record.
        time : float
            Current simulation time in seconds.
        dt : float
            Time step in seconds.
        """
        # Record time point
        if self.time_domain:
            self._time_points.append(time)

        # Record field data for each component
        for comp in self.components:
            # Get monitor region indices and field data
            indices = self._monitor_region[comp]
            field_data = fields[comp][indices].copy()

            # Store time domain data
            if self.time_domain:
                self._time_data[comp].append(field_data)

            # Update frequency domain data using DFT
            if self.frequencies:
                for freq in self.frequencies:
                    omega = 2 * np.pi * freq
                    complex_phase = np.exp(-1j * omega * time)
                    self._freq_data[comp][freq] += field_data * complex_phase * dt

    def get_time_data(self, component: FieldComponent) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get time-domain data for a specific field component.

        Parameters
        ----------
        component : str
            Field component to retrieve ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz").

        Returns
        -------
        tuple of numpy.ndarray
            Tuple containing (time_points, field_data).
        """
        if not self.time_domain:
            raise ValueError("Time domain data not enabled for this monitor")

        if component not in self.components:
            raise ValueError(f"Component {component} not recorded by this monitor")

        # Convert list of arrays to a single array
        # Shape: (time_steps, x, y, z)
        field_data = np.array(self._time_data[component])
        time_points = np.array(self._time_points)

        return time_points, field_data

    def get_frequency_data(
        self, component: FieldComponent, frequency: float
    ) -> np.ndarray:
        """
        Get frequency-domain data for a specific field component and frequency.

        Parameters
        ----------
        component : str
            Field component to retrieve ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz").
        frequency : float
            Frequency in Hz to retrieve.

        Returns
        -------
        numpy.ndarray
            Complex-valued field data at the specified frequency.
        """
        if not self.frequencies:
            raise ValueError("Frequency domain data not enabled for this monitor")

        if component not in self.components:
            raise ValueError(f"Component {component} not recorded by this monitor")

        if frequency not in self.frequencies:
            raise ValueError(f"Frequency {frequency} not recorded by this monitor")

        return self._freq_data[component][frequency]

    def get_power_flow(
        self, frequency: Optional[float] = None
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate Poynting vector (power flow) through the monitor.

        Parameters
        ----------
        frequency : float, optional
            Frequency in Hz for frequency-domain calculation.
            If None, calculate time-domain Poynting vector.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            If frequency is None, returns tuple (time_points, poynting_vector).
            Otherwise, returns poynting_vector at the specified frequency.
        """
        if frequency is None:
            # Calculate time-domain Poynting vector
            if not self.time_domain:
                raise ValueError("Time domain data not enabled for this monitor")

            # Check that we have E and H components
            e_components = [comp for comp in self.components if comp.startswith("E")]
            h_components = [comp for comp in self.components if comp.startswith("H")]

            if not (e_components and h_components):
                raise ValueError(
                    "Need both E and H components to calculate Poynting vector"
                )

            # Calculate Poynting vector components
            poynting_vector = 0
            pairs = [
                ("Ey", "Hz"),
                ("Ez", "Hy"),
                ("Ez", "Hx"),
                ("Ex", "Hz"),
                ("Ex", "Hy"),
                ("Ey", "Hx"),
            ]
            signs = [1, -1, 1, -1, 1, -1]  # Signs for E × H

            # For each available pair, add contribution to Poynting vector
            for (e_comp, h_comp), sign in zip(pairs, signs):
                if e_comp in self.components and h_comp in self.components:
                    _, e_data = self.get_time_data(e_comp)
                    _, h_data = self.get_time_data(h_comp)
                    poynting_vector += sign * e_data * h_data

            return np.array(self._time_points), poynting_vector

        else:
            # Calculate frequency-domain Poynting vector
            if frequency not in self.frequencies:
                raise ValueError(f"Frequency {frequency} not recorded by this monitor")

            # Check that we have E and H components
            e_components = [comp for comp in self.components if comp.startswith("E")]
            h_components = [comp for comp in self.components if comp.startswith("H")]

            if not (e_components and h_components):
                raise ValueError(
                    "Need both E and H components to calculate Poynting vector"
                )

            # Calculate Poynting vector components
            poynting_vector = 0
            pairs = [
                ("Ey", "Hz"),
                ("Ez", "Hy"),
                ("Ez", "Hx"),
                ("Ex", "Hz"),
                ("Ex", "Hy"),
                ("Ey", "Hx"),
            ]
            signs = [1, -1, 1, -1, 1, -1]  # Signs for E × H

            # For each available pair, add contribution to Poynting vector
            for (e_comp, h_comp), sign in zip(pairs, signs):
                if e_comp in self.components and h_comp in self.components:
                    e_data = self.get_frequency_data(e_comp, frequency)
                    h_data = self.get_frequency_data(h_comp, frequency)
                    poynting_vector += 0.5 * sign * np.real(e_data * np.conj(h_data))

            return poynting_vector
