"""
Mode port boundary conditions for waveguide simulations.

This module implements mode ports that can inject and extract waveguide modes
at simulation boundaries, enabling accurate S-parameter calculations.
"""

from typing import List, Tuple, Optional, Dict, Literal
import numpy as np
from dataclasses import dataclass

from prismo.core.grid import YeeGrid
from prismo.core.fields import ElectromagneticFields
from prismo.modes.solver import WaveguideMode
from prismo.utils.mode_matching import (
    compute_mode_overlap,
    normalize_mode_to_power,
    interpolate_mode_to_grid,
)


@dataclass
class ModePortConfig:
    """
    Configuration for a mode port.

    Attributes
    ----------
    center : Tuple[float, float, float]
        Port center position.
    size : Tuple[float, float, float]
        Port size (transverse dimensions).
    direction : str
        Port normal direction and orientation ('+x', '-x', '+y', '-y', '+z', '-z').
    modes : List[WaveguideMode]
        Modes supported by this port.
    inject : bool
        Whether to inject modes (source) or only extract (monitor).
    """

    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    direction: Literal["+x", "-x", "+y", "-y", "+z", "-z"]
    modes: List[WaveguideMode]
    inject: bool = False


class ModePort:
    """
    Mode port for injecting and extracting waveguide modes.

    A mode port acts as both a source (injecting modes) and a monitor
    (extracting mode amplitudes) at a plane in the simulation domain.

    Parameters
    ----------
    config : ModePortConfig
        Port configuration.
    name : str, optional
        Port name for identification.
    enabled : bool, optional
        Enable/disable port, default=True.

    Examples
    --------
    >>> from prismo.modes.solver import ModeSolver
    >>> # Solve for waveguide modes
    >>> mode_solver = ModeSolver(wavelength=1.55e-6, x=x, y=y, epsilon=eps)
    >>> modes = mode_solver.solve(num_modes=2, mode_type='TE')
    >>>
    >>> # Create mode port
    >>> config = ModePortConfig(
    ...     center=(0.0, 0.0, 0.0),
    ...     size=(2e-6, 2e-6, 0.0),
    ...     direction='+z',
    ...     modes=modes,
    ...     inject=True,
    ... )
    >>> port = ModePort(config, name='input_port')
    """

    def __init__(
        self,
        config: ModePortConfig,
        name: Optional[str] = None,
        enabled: bool = True,
    ):
        self.config = config
        self.name = name or f"ModePort_{id(self)}"
        self.enabled = enabled

        # Grid information (set during initialization)
        self._grid: Optional[YeeGrid] = None
        self._port_region: Dict[str, np.ndarray] = {}

        # Interpolated mode profiles on simulation grid
        self._interpolated_modes: List[WaveguideMode] = []

        # Mode coefficients storage
        self._mode_coefficients: Dict[int, List[complex]] = {
            i: [] for i in range(len(config.modes))
        }
        self._time_points: List[float] = []

        # Parse direction
        self.sign = +1 if config.direction[0] == "+" else -1
        self.axis = config.direction[-1].lower()

    def initialize(self, grid: YeeGrid) -> None:
        """
        Initialize the mode port on the simulation grid.

        Parameters
        ----------
        grid : YeeGrid
            Simulation grid.
        """
        self._grid = grid
        self._setup_port_region()
        self._interpolate_modes()

    def _setup_port_region(self) -> None:
        """Set up port region in grid coordinates."""
        if self._grid is None:
            raise RuntimeError("Port must be initialized with a grid first")

        # Determine port plane indices based on direction
        center = self.config.center
        size = self.config.size

        # Get grid index for port position
        if self.axis == "x":
            port_idx = int((center[0] - self._grid.origin[0]) / self._grid.dx)
            y_min = int(
                (center[1] - size[1] / 2 - self._grid.origin[1]) / self._grid.dy
            )
            y_max = int(
                (center[1] + size[1] / 2 - self._grid.origin[1]) / self._grid.dy
            )
            z_min = (
                int((center[2] - size[2] / 2 - self._grid.origin[2]) / self._grid.dz)
                if self._grid.is_3d
                else 0
            )
            z_max = (
                int((center[2] + size[2] / 2 - self._grid.origin[2]) / self._grid.dz)
                if self._grid.is_3d
                else 1
            )

            self._port_region = {
                "x_idx": port_idx,
                "y_slice": slice(y_min, y_max),
                "z_slice": slice(z_min, z_max),
                "normal_axis": 0,
            }
        elif self.axis == "y":
            x_min = int(
                (center[0] - size[0] / 2 - self._grid.origin[0]) / self._grid.dx
            )
            x_max = int(
                (center[0] + size[0] / 2 - self._grid.origin[0]) / self._grid.dx
            )
            port_idx = int((center[1] - self._grid.origin[1]) / self._grid.dy)
            z_min = (
                int((center[2] - size[2] / 2 - self._grid.origin[2]) / self._grid.dz)
                if self._grid.is_3d
                else 0
            )
            z_max = (
                int((center[2] + size[2] / 2 - self._grid.origin[2]) / self._grid.dz)
                if self._grid.is_3d
                else 1
            )

            self._port_region = {
                "x_slice": slice(x_min, x_max),
                "y_idx": port_idx,
                "z_slice": slice(z_min, z_max),
                "normal_axis": 1,
            }
        else:  # z
            x_min = int(
                (center[0] - size[0] / 2 - self._grid.origin[0]) / self._grid.dx
            )
            x_max = int(
                (center[0] + size[0] / 2 - self._grid.origin[0]) / self._grid.dx
            )
            y_min = int(
                (center[1] - size[1] / 2 - self._grid.origin[1]) / self._grid.dy
            )
            y_max = int(
                (center[1] + size[1] / 2 - self._grid.origin[1]) / self._grid.dy
            )
            port_idx = (
                int((center[2] - self._grid.origin[2]) / self._grid.dz)
                if self._grid.is_3d
                else 0
            )

            self._port_region = {
                "x_slice": slice(x_min, x_max),
                "y_slice": slice(y_min, y_max),
                "z_idx": port_idx,
                "normal_axis": 2,
            }

    def _interpolate_modes(self) -> None:
        """Interpolate mode profiles to simulation grid."""
        if self._grid is None:
            raise RuntimeError("Port must be initialized first")

        # Get transverse grid coordinates from port region
        if self.axis == "x":
            y_slice = self._port_region["y_slice"]
            z_slice = self._port_region["z_slice"]
            y_coords = (
                np.arange(y_slice.start, y_slice.stop) * self._grid.dy
                + self._grid.origin[1]
            )
            z_coords = (
                np.arange(z_slice.start, z_slice.stop) * self._grid.dz
                + self._grid.origin[2]
                if self._grid.is_3d
                else np.array([0.0])
            )
            grid_coords = (y_coords, z_coords)
        elif self.axis == "y":
            x_slice = self._port_region["x_slice"]
            z_slice = self._port_region["z_slice"]
            x_coords = (
                np.arange(x_slice.start, x_slice.stop) * self._grid.dx
                + self._grid.origin[0]
            )
            z_coords = (
                np.arange(z_slice.start, z_slice.stop) * self._grid.dz
                + self._grid.origin[2]
                if self._grid.is_3d
                else np.array([0.0])
            )
            grid_coords = (x_coords, z_coords)
        else:  # z
            x_slice = self._port_region["x_slice"]
            y_slice = self._port_region["y_slice"]
            x_coords = (
                np.arange(x_slice.start, x_slice.stop) * self._grid.dx
                + self._grid.origin[0]
            )
            y_coords = (
                np.arange(y_slice.start, y_slice.stop) * self._grid.dy
                + self._grid.origin[1]
            )
            grid_coords = (x_coords, y_coords)

        # Interpolate each mode
        for mode in self.config.modes:
            # Interpolate mode to grid
            interp_mode = interpolate_mode_to_grid(mode, grid_coords[0], grid_coords[1])
            self._interpolated_modes.append(interp_mode)

    def inject_fields(
        self,
        fields: ElectromagneticFields,
        time: float,
        dt: float,
        mode_amplitudes: Optional[List[complex]] = None,
    ) -> None:
        """
        Inject mode fields into the simulation.

        This method adds mode field patterns to the simulation fields at the
        port location, with proper Yee grid staggering.

        Parameters
        ----------
        fields : ElectromagneticFields
            Field arrays to update.
        time : float
            Current simulation time.
        dt : float
            Time step.
        mode_amplitudes : List[complex], optional
            Complex amplitudes for each mode. If None, uses unit amplitude.
        """
        if not self.enabled or not self.config.inject:
            return

        if mode_amplitudes is None:
            mode_amplitudes = [1.0] * len(self._interpolated_modes)

        # For each mode, add its field pattern
        for mode_idx, (mode, amplitude) in enumerate(
            zip(self._interpolated_modes, mode_amplitudes)
        ):
            # Get mode phase (propagating wave)
            beta = 2 * np.pi * mode.neff.real / mode.wavelength
            omega = 2 * np.pi * mode.frequency

            # Time-dependent amplitude
            phase = omega * time
            amp_t = amplitude * np.exp(1j * phase)

            # Extract real part for time-domain injection
            amp_real = amp_t.real

            # Add mode fields to simulation fields at port region
            self._add_mode_to_fields(fields, mode, amp_real)

    def _add_mode_to_fields(
        self,
        fields: ElectromagneticFields,
        mode: WaveguideMode,
        amplitude: float,
    ) -> None:
        """
        Add mode field pattern to simulation fields.

        Properly handles Yee grid staggering for different field components.

        Parameters
        ----------
        fields : ElectromagneticFields
            Field arrays to update.
        mode : WaveguideMode
            Mode with field patterns.
        amplitude : float
            Real-valued amplitude factor.
        """
        # Get port region slicing
        region = self._port_region

        # Add fields based on axis orientation
        if self.axis == "z":
            # Port in xy-plane
            x_slice = region["x_slice"]
            y_slice = region["y_slice"]
            z_idx = region.get("z_idx", 0)

            # Add tangential E fields (Ex, Ey) and normal H field (Hz)
            if hasattr(fields, "Ex"):
                fields.Ex[x_slice, y_slice, z_idx] += amplitude * mode.Ex.real
            if hasattr(fields, "Ey"):
                fields.Ey[x_slice, y_slice, z_idx] += amplitude * mode.Ey.real
            if hasattr(fields, "Hz"):
                fields.Hz[x_slice, y_slice, z_idx] += amplitude * mode.Hz.real

        elif self.axis == "x":
            # Port in yz-plane
            x_idx = region["x_idx"]
            y_slice = region["y_slice"]
            z_slice = region["z_slice"]

            if hasattr(fields, "Ey"):
                fields.Ey[x_idx, y_slice, z_slice] += amplitude * mode.Ey.real
            if hasattr(fields, "Ez"):
                fields.Ez[x_idx, y_slice, z_slice] += amplitude * mode.Ez.real
            if hasattr(fields, "Hx"):
                fields.Hx[x_idx, y_slice, z_slice] += amplitude * mode.Hx.real

        else:  # y axis
            # Port in xz-plane
            x_slice = region["x_slice"]
            y_idx = region["y_idx"]
            z_slice = region["z_slice"]

            if hasattr(fields, "Ex"):
                fields.Ex[x_slice, y_idx, z_slice] += amplitude * mode.Ex.real
            if hasattr(fields, "Ez"):
                fields.Ez[x_slice, y_idx, z_slice] += amplitude * mode.Ez.real
            if hasattr(fields, "Hy"):
                fields.Hy[x_slice, y_idx, z_slice] += amplitude * mode.Hy.real

    def extract_mode_coefficients(
        self,
        fields: ElectromagneticFields,
        time: float,
    ) -> List[complex]:
        """
        Extract mode coefficients from simulation fields.

        Uses overlap integrals to decompose fields into mode amplitudes.

        Parameters
        ----------
        fields : ElectromagneticFields
            Current simulation fields.
        time : float
            Current time.

        Returns
        -------
        List[complex]
            Mode coefficients for each port mode.
        """
        if not self.enabled:
            return [0.0] * len(self._interpolated_modes)

        # Extract field slices at port
        Ex, Ey, Ez, Hx, Hy, Hz = self._extract_field_slice(fields)

        # Compute overlap with each mode
        coefficients = []

        for mode in self._interpolated_modes:
            coeff = compute_mode_overlap(
                Ex,
                Ey,
                Ez,
                Hx,
                Hy,
                Hz,
                mode,
                direction=self.axis,
                dx=self._grid.dx if self.axis != "x" else self._grid.dy,
                dy=(
                    self._grid.dy
                    if self.axis != "y"
                    else (self._grid.dz if self._grid.is_3d else self._grid.dx)
                ),
            )
            coefficients.append(coeff)

        # Store coefficients
        for mode_idx, coeff in enumerate(coefficients):
            self._mode_coefficients[mode_idx].append(coeff)
        self._time_points.append(time)

        return coefficients

    def _extract_field_slice(
        self,
        fields: ElectromagneticFields,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract field components at port plane.

        Returns
        -------
        Tuple of arrays
            (Ex, Ey, Ez, Hx, Hy, Hz) at port location.
        """
        region = self._port_region

        # Extract based on port orientation
        if self.axis == "z":
            x_slice = region["x_slice"]
            y_slice = region["y_slice"]
            z_idx = region.get("z_idx", 0)

            Ex = (
                fields.Ex[x_slice, y_slice, z_idx]
                if hasattr(fields, "Ex")
                else np.zeros((1, 1))
            )
            Ey = (
                fields.Ey[x_slice, y_slice, z_idx]
                if hasattr(fields, "Ey")
                else np.zeros((1, 1))
            )
            Ez = (
                fields.Ez[x_slice, y_slice, z_idx]
                if hasattr(fields, "Ez")
                else np.zeros((1, 1))
            )
            Hx = (
                fields.Hx[x_slice, y_slice, z_idx]
                if hasattr(fields, "Hx")
                else np.zeros((1, 1))
            )
            Hy = (
                fields.Hy[x_slice, y_slice, z_idx]
                if hasattr(fields, "Hy")
                else np.zeros((1, 1))
            )
            Hz = (
                fields.Hz[x_slice, y_slice, z_idx]
                if hasattr(fields, "Hz")
                else np.zeros((1, 1))
            )

        elif self.axis == "x":
            x_idx = region["x_idx"]
            y_slice = region["y_slice"]
            z_slice = region["z_slice"]

            Ex = (
                fields.Ex[x_idx, y_slice, z_slice]
                if hasattr(fields, "Ex")
                else np.zeros((1, 1))
            )
            Ey = (
                fields.Ey[x_idx, y_slice, z_slice]
                if hasattr(fields, "Ey")
                else np.zeros((1, 1))
            )
            Ez = (
                fields.Ez[x_idx, y_slice, z_slice]
                if hasattr(fields, "Ez")
                else np.zeros((1, 1))
            )
            Hx = (
                fields.Hx[x_idx, y_slice, z_slice]
                if hasattr(fields, "Hx")
                else np.zeros((1, 1))
            )
            Hy = (
                fields.Hy[x_idx, y_slice, z_slice]
                if hasattr(fields, "Hy")
                else np.zeros((1, 1))
            )
            Hz = (
                fields.Hz[x_idx, y_slice, z_slice]
                if hasattr(fields, "Hz")
                else np.zeros((1, 1))
            )

        else:  # y
            x_slice = region["x_slice"]
            y_idx = region["y_idx"]
            z_slice = region["z_slice"]

            Ex = (
                fields.Ex[x_slice, y_idx, z_slice]
                if hasattr(fields, "Ex")
                else np.zeros((1, 1))
            )
            Ey = (
                fields.Ey[x_slice, y_idx, z_slice]
                if hasattr(fields, "Ey")
                else np.zeros((1, 1))
            )
            Ez = (
                fields.Ez[x_slice, y_idx, z_slice]
                if hasattr(fields, "Ez")
                else np.zeros((1, 1))
            )
            Hx = (
                fields.Hx[x_slice, y_idx, z_slice]
                if hasattr(fields, "Hx")
                else np.zeros((1, 1))
            )
            Hy = (
                fields.Hy[x_slice, y_idx, z_slice]
                if hasattr(fields, "Hy")
                else np.zeros((1, 1))
            )
            Hz = (
                fields.Hz[x_slice, y_idx, z_slice]
                if hasattr(fields, "Hz")
                else np.zeros((1, 1))
            )

        return Ex, Ey, Ez, Hx, Hy, Hz

    def get_mode_coefficient(self, mode_index: int) -> List[complex]:
        """
        Get time series of mode coefficient.

        Parameters
        ----------
        mode_index : int
            Mode index.

        Returns
        -------
        List[complex]
            Mode coefficient vs time.
        """
        return self._mode_coefficients[mode_index]

    def get_time_points(self) -> List[float]:
        """
        Get recorded time points.

        Returns
        -------
        List[float]
            Time points where coefficients were recorded.
        """
        return self._time_points
