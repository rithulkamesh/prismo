"""
Gaussian beam sources for FDTD simulations.

This module implements Gaussian beam sources for exciting focused electromagnetic
fields within the simulation domain.
"""

from typing import Tuple, Dict, Optional, Union, Literal
import numpy as np

from prismo.core.grid import YeeGrid
from prismo.core.fields import ElectromagneticFields, FieldComponent
from prismo.sources.base import Source
from prismo.sources.waveform import Waveform, GaussianPulse, ContinuousWave


class GaussianBeamSource(Source):
    """
    Gaussian beam source for exciting focused electromagnetic fields.

    Parameters
    ----------
    center : Tuple[float, float, float]
        Physical coordinates of the beam center (x, y, z) in meters.
    size : Tuple[float, float, float]
        Physical dimensions of the source region (Lx, Ly, Lz) in meters.
    direction : Literal["x", "y", "z"]
        Propagation direction of the beam.
    polarization : Literal["x", "y", "z"]
        Polarization direction of the electric field (must be perpendicular to direction).
    frequency : float
        Center frequency in Hz.
    beam_waist : float
        Beam waist (minimum radius) in meters.
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
        direction: Literal["x", "y", "z"],
        polarization: Literal["x", "y", "z"],
        frequency: float,
        beam_waist: float,
        pulse: bool = True,
        pulse_width: Optional[float] = None,
        amplitude: float = 1.0,
        phase: float = 0.0,
        name: Optional[str] = None,
        enabled: bool = True,
    ):
        super().__init__(center=center, size=size, name=name, enabled=enabled)

        # Validate direction and polarization
        self._validate_direction_polarization(direction, polarization)

        self.direction = direction
        self.polarization = polarization
        self.frequency = frequency
        self.beam_waist = beam_waist
        self.wavelength = 299792458.0 / frequency  # c / f

        # Calculate beam parameters
        self.k = 2 * np.pi / self.wavelength
        self.rayleigh_range = np.pi * beam_waist**2 / self.wavelength

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
        Initialize the Gaussian beam source on a specific grid.

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
        Set up the field components for the Gaussian beam.

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

    def _calculate_beam_profile(self, time: float) -> Dict[str, np.ndarray]:
        """
        Calculate the Gaussian beam profile at the current time.

        Parameters
        ----------
        time : float
            Current simulation time in seconds.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary mapping field components to their values.
        """
        if self._grid is None:
            raise RuntimeError("Source must be initialized with a grid first")

        # Get waveform value at current time
        amplitude = self.waveform(time)

        # Get grid coordinates for the source region
        x_min, x_max, y_min, y_max, z_min, z_max = self._compute_source_region()

        # For 2D grids, ensure z range has at least one point
        if not self._grid.is_3d and z_min == z_max:
            z_max = z_min + 1

        # Create coordinate meshgrid based on propagation direction
        if self.direction.lower() == "x":
            # For x-propagation, we need y and z coordinates
            ys, zs = np.meshgrid(
                np.arange(y_min, y_max), np.arange(z_min, z_max), indexing="ij"
            )
            xs = np.full_like(ys, x_min)  # Source plane is at x_min

            # Convert to physical coordinates
            y_phys = self._grid.index_to_coord(1, ys)
            z_phys = (
                self._grid.index_to_coord(2, zs)
                if self._grid.is_3d
                else np.zeros_like(ys)
            )

            # Calculate beam profile
            if self._grid.is_3d:
                r_squared = (y_phys - self.center[1]) ** 2 + (
                    z_phys - self.center[2]
                ) ** 2
            else:
                r_squared = (y_phys - self.center[1]) ** 2

        elif self.direction.lower() == "y":
            # For y-propagation, we need x and z coordinates
            xs, zs = np.meshgrid(
                np.arange(x_min, x_max), np.arange(z_min, z_max), indexing="ij"
            )
            ys = np.full_like(xs, y_min)  # Source plane is at y_min

            # Convert to physical coordinates
            x_phys = self._grid.index_to_coord(0, xs)
            z_phys = (
                self._grid.index_to_coord(2, zs)
                if self._grid.is_3d
                else np.zeros_like(xs)
            )

            # Calculate beam profile
            if self._grid.is_3d:
                r_squared = (x_phys - self.center[0]) ** 2 + (
                    z_phys - self.center[2]
                ) ** 2
            else:
                r_squared = (x_phys - self.center[0]) ** 2

        else:  # z-propagation
            # For z-propagation, we need x and y coordinates
            xs, ys = np.meshgrid(
                np.arange(x_min, x_max), np.arange(y_min, y_max), indexing="ij"
            )
            zs = np.full_like(xs, z_min)  # Source plane is at z_min

            # Convert to physical coordinates
            x_phys = self._grid.index_to_coord(0, xs)
            y_phys = self._grid.index_to_coord(1, ys)

            # Calculate beam profile
            r_squared = (x_phys - self.center[0]) ** 2 + (y_phys - self.center[1]) ** 2

        # Gaussian beam profile: E(r) = E₀ * exp(-r²/w₀²)
        beam_profile = amplitude * np.exp(-r_squared / (self.beam_waist**2))

        # Determine the appropriate field components to update
        field_values = {}
        for comp in self._e_components:
            field_values[comp] = beam_profile

        # Calculate corresponding H field components (simplified)
        # In a more accurate implementation, these would use the full Gaussian beam equations
        for comp in self._h_components:
            field_values[comp] = beam_profile / 377.0  # Approximate Z₀ = 377 Ω

        return field_values

    def update_fields(
        self, fields: ElectromagneticFields, time: float, dt: float
    ) -> None:
        """
        Update electromagnetic fields with Gaussian beam source contribution.

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

        # Calculate beam profile
        field_values = self._calculate_beam_profile(time)

        # Apply source to each component
        for comp, values in field_values.items():
            # Get source region indices
            indices = self._source_region[comp]

            # Apply source
            field_component = fields[comp]
            field_component[indices] += values.flat
