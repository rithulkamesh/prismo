"""
Point sources for FDTD simulations.

This module implements electric and magnetic dipole sources located at a single
point (or small region) within the simulation domain.
"""

from typing import Tuple, Dict, Optional, Union, Literal
import numpy as np

from prismo.core.grid import YeeGrid
from prismo.core.fields import ElectromagneticFields, FieldComponent
from prismo.sources.base import Source
from prismo.sources.waveform import Waveform, GaussianPulse, ContinuousWave


class PointSource(Source):
    """
    Point dipole source for exciting electromagnetic fields.

    Parameters
    ----------
    position : Tuple[float, float, float]
        Physical coordinates of the source (x, y, z) in meters.
    component : str
        Field component to excite ('Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz').
    waveform : Waveform
        Time-dependent waveform for the source.
    name : str, optional
        Name of the source for identification.
    enabled : bool, optional
        Flag to enable/disable the source, default=True.
    """

    def __init__(
        self,
        position: Tuple[float, float, float],
        component: FieldComponent,
        waveform: Waveform,
        name: Optional[str] = None,
        enabled: bool = True,
    ):
        # Point sources have zero size
        super().__init__(center=position, size=(0, 0, 0), name=name, enabled=enabled)

        self.component = component
        self.waveform = waveform

    def update_fields(
        self, fields: ElectromagneticFields, time: float, dt: float
    ) -> None:
        """
        Update electromagnetic fields with point source contribution.

        Parameters
        ----------
        fields : ElectromagneticFields
            Electromagnetic fields to update.
        time : float
            Current simulation time in seconds.
        dt : float
            Time step in seconds.
        """
        if not self.enabled:
            return

        # Calculate waveform value at current time
        amplitude = self.waveform(time)

        # Get source region indices
        indices = self._source_region[self.component]

        # Apply source to the specified component
        field_component = fields[self.component]
        field_component[indices] += amplitude


class ElectricDipole(PointSource):
    """
    Electric dipole source for exciting electromagnetic fields.

    Parameters
    ----------
    position : Tuple[float, float, float]
        Physical coordinates of the source (x, y, z) in meters.
    polarization : Literal["x", "y", "z"]
        Polarization direction of the dipole.
    frequency : float
        Source frequency in Hz.
    pulse : bool, optional
        Whether to use a Gaussian pulse (True) or continuous wave (False), default=True.
    pulse_width : float, optional
        Width of the Gaussian pulse in seconds, required if pulse=True.
    amplitude : float, optional
        Peak amplitude of the source, default=1.0.
    phase : float, optional
        Phase offset in radians, default=0.0.
    name : str, optional
        Name of the source for identification.
    enabled : bool, optional
        Flag to enable/disable the source, default=True.
    """

    def __init__(
        self,
        position: Tuple[float, float, float],
        polarization: Literal["x", "y", "z"],
        frequency: float,
        pulse: bool = True,
        pulse_width: Optional[float] = None,
        amplitude: float = 1.0,
        phase: float = 0.0,
        name: Optional[str] = None,
        enabled: bool = True,
    ):
        # Map polarization to field component
        component_map = {"x": "Ex", "y": "Ey", "z": "Ez"}
        component = component_map[polarization.lower()]

        # Create waveform based on parameters
        if pulse:
            if pulse_width is None:
                raise ValueError("pulse_width must be provided for pulsed sources")
            waveform = GaussianPulse(
                frequency=frequency,
                pulse_width=pulse_width,
                amplitude=amplitude,
                phase=phase,
            )
        else:
            waveform = ContinuousWave(
                frequency=frequency, amplitude=amplitude, phase=phase
            )

        super().__init__(
            position=position,
            component=component,
            waveform=waveform,
            name=name,
            enabled=enabled,
        )


class MagneticDipole(PointSource):
    """
    Magnetic dipole source for exciting electromagnetic fields.

    Parameters
    ----------
    position : Tuple[float, float, float]
        Physical coordinates of the source (x, y, z) in meters.
    polarization : Literal["x", "y", "z"]
        Polarization direction of the dipole.
    frequency : float
        Source frequency in Hz.
    pulse : bool, optional
        Whether to use a Gaussian pulse (True) or continuous wave (False), default=True.
    pulse_width : float, optional
        Width of the Gaussian pulse in seconds, required if pulse=True.
    amplitude : float, optional
        Peak amplitude of the source, default=1.0.
    phase : float, optional
        Phase offset in radians, default=0.0.
    name : str, optional
        Name of the source for identification.
    enabled : bool, optional
        Flag to enable/disable the source, default=True.
    """

    def __init__(
        self,
        position: Tuple[float, float, float],
        polarization: Literal["x", "y", "z"],
        frequency: float,
        pulse: bool = True,
        pulse_width: Optional[float] = None,
        amplitude: float = 1.0,
        phase: float = 0.0,
        name: Optional[str] = None,
        enabled: bool = True,
    ):
        # Map polarization to field component
        component_map = {"x": "Hx", "y": "Hy", "z": "Hz"}
        component = component_map[polarization.lower()]

        # Create waveform based on parameters
        if pulse:
            if pulse_width is None:
                raise ValueError("pulse_width must be provided for pulsed sources")
            waveform = GaussianPulse(
                frequency=frequency,
                pulse_width=pulse_width,
                amplitude=amplitude,
                phase=phase,
            )
        else:
            waveform = ContinuousWave(
                frequency=frequency, amplitude=amplitude, phase=phase
            )

        super().__init__(
            position=position,
            component=component,
            waveform=waveform,
            name=name,
            enabled=enabled,
        )
