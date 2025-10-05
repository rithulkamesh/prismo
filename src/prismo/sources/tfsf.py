"""
Total-Field/Scattered-Field (TFSF) source implementation for FDTD simulations.

The TFSF formulation separates the simulation domain into a total-field region
(where both incident and scattered fields exist) and a scattered-field region
(where only scattered fields exist). This allows for clean injection of plane
waves without numerical artifacts.

Reference:
Taflove, A., & Hagness, S. C. (2005). Computational Electrodynamics:
The Finite-Difference Time-Domain Method (3rd ed.). Artech House.
"""

from typing import Tuple, Dict, Optional, Union, Literal
import numpy as np

from prismo.core.grid import YeeGrid
from prismo.core.fields import ElectromagneticFields, FieldComponent
from prismo.sources.base import Source
from prismo.sources.waveform import Waveform, GaussianPulse, ContinuousWave


class TFSFSource(Source):
    """
    Total-Field/Scattered-Field (TFSF) plane wave source.

    This source implements the TFSF formulation for injecting plane waves
    into the FDTD grid. The TFSF boundary separates the computational domain
    into two regions:
    - Total-field region (interior): Contains incident + scattered fields
    - Scattered-field region (exterior): Contains only scattered fields

    Parameters
    ----------
    center : Tuple[float, float, float]
        Physical coordinates of the TFSF region center (x, y, z) in meters.
    size : Tuple[float, float, float]
        Physical dimensions of the TFSF region (Lx, Ly, Lz) in meters.
        The TFSF surface will be placed on the boundaries of this region.
    direction : Literal["x", "y", "z", "+x", "-x", "+y", "-y", "+z", "-z"]
        Propagation direction of the plane wave.
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
    angle : float, optional
        Incidence angle in radians (for oblique incidence), default=0.0.
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
        angle: float = 0.0,
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
        self.angle = angle

        # Physical constants
        self.c = 299792458.0  # Speed of light (m/s)
        self.wavelength = self.c / frequency
        self.k = 2 * np.pi / self.wavelength  # Wave number
        self.omega = 2 * np.pi * frequency  # Angular frequency

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
        self._tfsf_boundaries: Dict[str, Dict] = {}
        self._e_component: str = ""
        self._h_component: str = ""

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
        Initialize the TFSF source on a specific grid.

        Parameters
        ----------
        grid : YeeGrid
            The grid on which to initialize the source.
        """
        super().initialize(grid)

        # Determine field components based on direction and polarization
        self._setup_field_components()

        # Set up TFSF boundary surfaces
        self._setup_tfsf_boundaries()

    def _setup_field_components(self) -> None:
        """
        Set up the field components for the plane wave.

        This determines which E and H components to use based on the
        propagation direction and polarization.
        """
        # Define the components based on direction and polarization
        # For x-propagation with y-polarization: Ey and Hz
        # For x-propagation with z-polarization: Ez and Hy
        direction_map = {
            ("x", "y"): {"E": "Ey", "H": "Hz"},
            ("x", "z"): {"E": "Ez", "H": "Hy"},
            ("y", "x"): {"E": "Ex", "H": "Hz"},
            ("y", "z"): {"E": "Ez", "H": "Hx"},
            ("z", "x"): {"E": "Ex", "H": "Hy"},
            ("z", "y"): {"E": "Ey", "H": "Hx"},
        }

        key = (self.direction, self.polarization)
        if key not in direction_map:
            raise ValueError(f"Invalid direction-polarization combination: {key}")

        self._e_component = direction_map[key]["E"]
        self._h_component = direction_map[key]["H"]

    def _setup_tfsf_boundaries(self) -> None:
        """
        Set up the TFSF boundary surfaces.

        The TFSF boundaries are the interfaces between the total-field and
        scattered-field regions. We need to correct the fields on these
        boundaries to properly inject the incident plane wave.
        """
        # Get the bounding box of the TFSF region in grid indices
        bbox_min, bbox_max = self._get_bbox_indices()

        # Store boundary information
        # We need to track which grid points are on the TFSF surface
        self._tfsf_boundaries = {
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "direction": self.direction,
            "direction_sign": self.direction_sign,
        }

    def _get_bbox_indices(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Get the bounding box indices for the TFSF region.

        Returns
        -------
        bbox_min : Tuple[int, int, int]
            Minimum indices (i_min, j_min, k_min) for the TFSF region.
        bbox_max : Tuple[int, int, int]
            Maximum indices (i_max, j_max, k_max) for the TFSF region.
        """
        # Get the monitor region from the base class (already computed)
        # Use the E-field component to determine the region
        indices = self._source_region[self._e_component]

        # Extract min and max indices
        i_min = indices[0].min() if len(indices[0]) > 0 else 0
        i_max = indices[0].max() if len(indices[0]) > 0 else 0
        j_min = indices[1].min() if len(indices[1]) > 0 else 0
        j_max = indices[1].max() if len(indices[1]) > 0 else 0

        if len(indices) > 2:
            k_min = indices[2].min() if len(indices[2]) > 0 else 0
            k_max = indices[2].max() if len(indices[2]) > 0 else 0
        else:
            k_min = 0
            k_max = 0

        return (i_min, j_min, k_min), (i_max, j_max, k_max)

    def update_fields(
        self, fields: ElectromagneticFields, time: float, dt: float
    ) -> None:
        """
        Update electromagnetic fields with TFSF source contribution.

        This applies corrections to the fields on the TFSF boundary to
        inject the incident plane wave.

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

        # Get waveform amplitude at current time
        e_amplitude = self.waveform(time)

        # H-field is offset by half a time step in the FDTD algorithm
        h_amplitude = self.waveform(time - 0.5 * dt)

        # Impedance of free space
        # Physical constants from the grid
        eps0 = 8.854187817e-12  # Vacuum permittivity (F/m)
        mu0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)
        eta0 = np.sqrt(mu0 / eps0)  # ~377 ohms
        h_amplitude = h_amplitude / eta0

        # Get TFSF boundary information
        bbox_min, bbox_max = (
            self._tfsf_boundaries["bbox_min"],
            self._tfsf_boundaries["bbox_max"],
        )
        direction = self._tfsf_boundaries["direction"]
        direction_sign = self._tfsf_boundaries["direction_sign"]

        # Apply TFSF corrections based on propagation direction
        if direction == "x":
            self._apply_tfsf_x(
                fields, bbox_min, bbox_max, e_amplitude, h_amplitude, direction_sign
            )
        elif direction == "y":
            self._apply_tfsf_y(
                fields, bbox_min, bbox_max, e_amplitude, h_amplitude, direction_sign
            )
        elif direction == "z":
            self._apply_tfsf_z(
                fields, bbox_min, bbox_max, e_amplitude, h_amplitude, direction_sign
            )

    def _apply_tfsf_x(
        self,
        fields: ElectromagneticFields,
        bbox_min: Tuple[int, int, int],
        bbox_max: Tuple[int, int, int],
        e_amp: float,
        h_amp: float,
        sign: int,
    ) -> None:
        """Apply TFSF corrections for x-propagating wave."""
        i_min, j_min, k_min = bbox_min
        i_max, j_max, k_max = bbox_max

        # Get field components
        e_field = fields[self._e_component]
        h_field = fields[self._h_component]

        # Apply corrections at the entry surface (x_min if +x, x_max if -x)
        if sign > 0:
            # Plane wave entering from x_min surface
            # Correct E-field on the surface
            if self._e_component in ["Ey", "Ez"]:
                e_field[i_min, :] -= e_amp
            # Correct H-field on the surface
            if self._h_component in ["Hy", "Hz"]:
                h_field[i_min, :] -= h_amp * sign
        else:
            # Plane wave entering from x_max surface
            if self._e_component in ["Ey", "Ez"]:
                e_field[i_max, :] += e_amp
            if self._h_component in ["Hy", "Hz"]:
                h_field[i_max, :] += h_amp * sign

    def _apply_tfsf_y(
        self,
        fields: ElectromagneticFields,
        bbox_min: Tuple[int, int, int],
        bbox_max: Tuple[int, int, int],
        e_amp: float,
        h_amp: float,
        sign: int,
    ) -> None:
        """Apply TFSF corrections for y-propagating wave."""
        i_min, j_min, k_min = bbox_min
        i_max, j_max, k_max = bbox_max

        # Get field components
        e_field = fields[self._e_component]
        h_field = fields[self._h_component]

        # Apply corrections at the entry surface
        if sign > 0:
            # Plane wave entering from y_min surface
            if self._e_component in ["Ex", "Ez"]:
                e_field[:, j_min] -= e_amp
            if self._h_component in ["Hx", "Hz"]:
                h_field[:, j_min] -= h_amp * sign
        else:
            # Plane wave entering from y_max surface
            if self._e_component in ["Ex", "Ez"]:
                e_field[:, j_max] += e_amp
            if self._h_component in ["Hx", "Hz"]:
                h_field[:, j_max] += h_amp * sign

    def _apply_tfsf_z(
        self,
        fields: ElectromagneticFields,
        bbox_min: Tuple[int, int, int],
        bbox_max: Tuple[int, int, int],
        e_amp: float,
        h_amp: float,
        sign: int,
    ) -> None:
        """Apply TFSF corrections for z-propagating wave."""
        i_min, j_min, k_min = bbox_min
        i_max, j_max, k_max = bbox_max

        # Get field components
        e_field = fields[self._e_component]
        h_field = fields[self._h_component]

        # Apply corrections at the entry surface
        if sign > 0:
            # Plane wave entering from z_min surface
            if self._e_component in ["Ex", "Ey"]:
                if len(e_field.shape) > 2:
                    e_field[:, :, k_min] -= e_amp
            if self._h_component in ["Hx", "Hy"]:
                if len(h_field.shape) > 2:
                    h_field[:, :, k_min] -= h_amp * sign
        else:
            # Plane wave entering from z_max surface
            if self._e_component in ["Ex", "Ey"]:
                if len(e_field.shape) > 2:
                    e_field[:, :, k_max] += e_amp
            if self._h_component in ["Hx", "Hy"]:
                if len(h_field.shape) > 2:
                    h_field[:, :, k_max] += h_amp * sign

    def __repr__(self) -> str:
        """Return string representation of the TFSF source."""
        return (
            f"TFSFSource(center={self.center}, size={self.size}, "
            f"direction={self.direction_sign:+d}{self.direction}, "
            f"polarization={self.polarization}, frequency={self.frequency:.2e} Hz, "
            f"wavelength={self.wavelength*1e6:.3f} µm)"
        )
