"""
Plane wave sources for FDTD simulations.

This module implements plane wave sources for exciting uniform electromagnetic
fields propagating in a specified direction.
"""

from typing import Tuple, Dict, Optional, Union, Literal
import numpy as np

from prismo.core.grid import YeeGrid
from prismo.core.fields import ElectromagneticFields, FieldComponent
from prismo.sources.base import Source
from prismo.sources.waveform import Waveform, GaussianPulse, ContinuousWave


class PlaneWaveSource(Source):
    """
    Plane wave source for exciting uniform electromagnetic fields.

    Parameters
    ----------
    center : Tuple[float, float, float]
        Physical coordinates of the wave center (x, y, z) in meters.
    size : Tuple[float, float, float]
        Physical dimensions of the source region (Lx, Ly, Lz) in meters.
    direction : Literal["x", "y", "z", "+x", "-x", "+y", "-y", "+z", "-z"]
        Propagation direction of the wave, with optional sign.
    polarization : Literal["x", "y", "z"]
        Polarization direction of the electric field (must be perpendicular to direction).
    frequency : float
        Center frequency in Hz.
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
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        direction: str,
        polarization: Literal["x", "y", "z"],
        frequency: float,
        pulse: bool = True,
        pulse_width: Optional[float] = None,
        amplitude: float = 1.0,
        phase: float = 0.0,
        name: Optional[str] = None,
        enabled: bool = True,
    ):
        super().__init__(center=center, size=size, name=name, enabled=enabled)

        # Parse direction and sign
        self._parse_direction(direction)

        # Validate direction and polarization
        self._validate_direction_polarization(self.direction, polarization)

        self.polarization = polarization
        self.frequency = frequency
        self.wavelength = 299792458.0 / frequency  # c / f

        # Create waveform based on parameters
        if pulse:
            if pulse_width is None:
                raise ValueError("pulse_width must be provided for pulsed sources")
            self.waveform = GaussianPulse(
                frequency=frequency,
                pulse_width=pulse_width,
                amplitude=amplitude,
                phase=phase,
            )
        else:
            self.waveform = ContinuousWave(
                frequency=frequency, amplitude=amplitude, phase=phase
            )

        # Will be computed when initialized
        self._e_components: Dict[str, FieldComponent] = {}
        self._h_components: Dict[str, FieldComponent] = {}

    def _parse_direction(self, direction: str) -> None:
        """
        Parse the direction string to extract direction and sign.

        Parameters
        ----------
        direction : str
            Propagation direction with optional sign.
        """
        if direction.startswith("+"):
            self.direction = direction[1:]
            self.direction_sign = 1
        elif direction.startswith("-"):
            self.direction = direction[1:]
            self.direction_sign = -1
        else:
            self.direction = direction
            self.direction_sign = 1

        # Validate direction
        if self.direction.lower() not in ["x", "y", "z"]:
            raise ValueError(
                f"Invalid direction: {direction}. Must be one of: x, y, z, +x, -x, +y, -y, +z, -z"
            )

    def _validate_direction_polarization(
        self, direction: str, polarization: str
    ) -> None:
        """
        Validate that direction and polarization are perpendicular.

        Parameters
        ----------
        direction : str
            Propagation direction ("x", "y", or "z").
        polarization : str
            Electric field polarization ("x", "y", or "z").

        Raises
        ------
        ValueError
            If direction and polarization are the same.
        """
        if direction.lower() == polarization.lower():
            raise ValueError(
                f"Polarization ({polarization}) must be perpendicular to "
                f"propagation direction ({direction})"
            )

    def initialize(self, grid: YeeGrid) -> None:
        """
        Initialize the plane wave source on a specific grid.

        Parameters
        ----------
        grid : YeeGrid
            The grid on which to initialize the source.
        """
        super().initialize(grid)

        # Determine field components based on direction and polarization
        self._setup_field_components()

    def _setup_field_components(self) -> None:
        """
        Set up the field components for the plane wave.

        This determines which field components to update based on the
        propagation direction and polarization.
        """
        # Define the components to update based on direction and polarization
        direction_map = {
            # For x-propagation
            ("x", "y"): {"E": ["Ey"], "H": ["Hz"]},
            ("x", "z"): {"E": ["Ez"], "H": ["Hy"]},
            # For y-propagation
            ("y", "x"): {"E": ["Ex"], "H": ["Hz"]},
            ("y", "z"): {"E": ["Ez"], "H": ["Hx"]},
            # For z-propagation
            ("z", "x"): {"E": ["Ex"], "H": ["Hy"]},
            ("z", "y"): {"E": ["Ey"], "H": ["Hx"]},
        }

        # Get components based on direction and polarization
        key = (self.direction.lower(), self.polarization.lower())
        if key not in direction_map:
            raise ValueError(
                f"Invalid direction ({self.direction}) and polarization ({self.polarization}) combination"
            )

        self._e_components = direction_map[key]["E"]
        self._h_components = direction_map[key]["H"]

    def update_fields(
        self, fields: ElectromagneticFields, time: float, dt: float
    ) -> None:
        """
        Update electromagnetic fields with plane wave source contribution.

        Parameters
        ----------
        fields : ElectromagneticFields
            Electromagnetic fields to update.
        time : float
            Current simulation time in seconds.
        dt : float
            Time step in seconds.
        """
        if not self.enabled or self._grid is None:
            return

        # Get waveform value at current time
        amplitude = self.waveform(time)

        # Determine the appropriate field components to update
        for comp in self._e_components:
            # Get source region indices
            indices = self._source_region[comp]

            # Apply source with uniform amplitude
            field_component = fields[comp]
            field_component[indices] += amplitude * self.direction_sign

        # Calculate corresponding H field components
        for comp in self._h_components:
            # Get source region indices
            indices = self._source_region[comp]

            # Apply source with uniform amplitude (E/Z₀)
            field_component = fields[comp]
            field_component[indices] += (amplitude / 377.0) * self.direction_sign
