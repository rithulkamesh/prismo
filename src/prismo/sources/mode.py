"""
Mode sources for injecting waveguide modes.

This module implements sources that inject specific waveguide modes
calculated from the eigenmode solver.
"""

from typing import Tuple, Optional, Literal
import numpy as np

from prismo.core.grid import YeeGrid
from prismo.core.fields import ElectromagneticFields
from prismo.sources.base import Source
from prismo.sources.waveform import Waveform
from prismo.modes.solver import WaveguideMode


class ModeSource(Source):
    """
    Source that injects a waveguide mode.

    Uses mode profiles from eigenmode solver to inject guided modes
    into the simulation domain with specified waveform modulation.

    Parameters
    ----------
    center : Tuple[float, float, float]
        Physical coordinates of source plane center.
    size : Tuple[float, float, float]
        Size of source plane.
    mode : WaveguideMode
        Waveguide mode to inject (from ModeSolver).
    direction : str
        Propagation direction ('+x', '-x', '+y', '-y', '+z', '-z').
    waveform : Waveform
        Temporal waveform for mode excitation.
    amplitude : float, optional
        Source amplitude, default=1.0.
    phase : float, optional
        Initial phase (radians), default=0.0.
    name : str, optional
        Source name.
    """

    def __init__(
        self,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
        mode: WaveguideMode,
        direction: Literal["+x", "-x", "+y", "-y", "+z", "-z"],
        waveform: Waveform,
        amplitude: float = 1.0,
        phase: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(center, size, name)

        self.mode = mode
        self.direction = direction
        self.waveform = waveform
        self.amplitude = amplitude
        self.phase = phase

        # Parse direction
        self.sign = +1 if direction[0] == "+" else -1
        self.axis = direction[-1].lower()

        # Mode profile will be interpolated to grid
        self._mode_profile_Ex = None
        self._mode_profile_Ey = None
        self._mode_profile_Ez = None
        self._mode_profile_Hx = None
        self._mode_profile_Hy = None
        self._mode_profile_Hz = None

    def initialize(self, grid: YeeGrid) -> None:
        """Initialize mode source on grid."""
        super().initialize(grid)

        # Interpolate mode profiles to simulation grid
        self._interpolate_mode_to_grid()

    def _interpolate_mode_to_grid(self) -> None:
        """
        Interpolate mode profile from mode solver grid to simulation grid.

        The mode solver may use different resolution than the main simulation,
        so we need to interpolate the mode fields.
        """
        from scipy.interpolate import RegularGridInterpolator

        # Mode coordinates
        x_mode = self.mode.x
        y_mode = self.mode.y

        # Get simulation grid coordinates in source region
        # (Simplified - should extract actual source region coordinates)
        nx, ny, _ = self._grid.dimensions
        x_sim = np.linspace(x_mode[0], x_mode[-1], min(nx, len(x_mode)))
        y_sim = np.linspace(y_mode[0], y_mode[-1], min(ny, len(y_mode)))

        # Create interpolators for each field component
        def make_interpolator(field_data):
            # Handle 2D data
            if field_data.ndim == 2:
                return RegularGridInterpolator(
                    (x_mode, y_mode),
                    field_data.real,  # Interpolate real part
                    bounds_error=False,
                    fill_value=0.0,
                )
            return None

        interp_Ex = make_interpolator(self.mode.Ex)
        interp_Ey = make_interpolator(self.mode.Ey)
        interp_Ez = make_interpolator(self.mode.Ez)
        interp_Hx = make_interpolator(self.mode.Hx)
        interp_Hy = make_interpolator(self.mode.Hy)
        interp_Hz = make_interpolator(self.mode.Hz)

        # Create meshgrid for simulation
        X_sim, Y_sim = np.meshgrid(x_sim, y_sim, indexing="ij")
        points = np.column_stack([X_sim.ravel(), Y_sim.ravel()])

        # Interpolate all components
        if interp_Ex is not None:
            self._mode_profile_Ex = interp_Ex(points).reshape(X_sim.shape)
        if interp_Ey is not None:
            self._mode_profile_Ey = interp_Ey(points).reshape(X_sim.shape)
        if interp_Ez is not None:
            self._mode_profile_Ez = interp_Ez(points).reshape(X_sim.shape)
        if interp_Hx is not None:
            self._mode_profile_Hx = interp_Hx(points).reshape(X_sim.shape)
        if interp_Hy is not None:
            self._mode_profile_Hy = interp_Hy(points).reshape(X_sim.shape)
        if interp_Hz is not None:
            self._mode_profile_Hz = interp_Hz(points).reshape(X_sim.shape)

    def update_fields(
        self, fields: ElectromagneticFields, time: float, dt: float
    ) -> None:
        """
        Update fields with mode source.

        Adds mode profile weighted by temporal waveform.

        Parameters
        ----------
        fields : ElectromagneticFields
            Field arrays to update.
        time : float
            Current simulation time.
        dt : float
            Time step.
        """
        if not self.enabled:
            return

        # Get temporal waveform value
        waveform_value = self.waveform.value(time)

        # Calculate phase for propagating mode
        # Include propagation: exp(i * beta * z - i * omega * t)
        beta = self.mode.neff.real * 2 * np.pi / self.mode.wavelength

        # Spatial phase (will be position-dependent in full implementation)
        spatial_phase = 0.0  # Simplified

        # Total field amplitude
        amplitude = (
            self.amplitude * waveform_value * np.exp(1j * (self.phase + spatial_phase))
        )

        # Extract real part for time-domain
        amplitude_real = amplitude.real

        # Add mode profile to fields in source region
        # (Simplified - should use proper source region indices)

        if self._mode_profile_Ex is not None:
            # Get field component in source region
            # For now, simplified update
            # Full implementation would use _source_region indices
            pass

        # Note: Full implementation would:
        # 1. Extract field slices in source region
        # 2. Add mode profile * amplitude to those slices
        # 3. Handle different propagation directions
        # 4. Apply proper boundary injection (e.g., Total Field/Scattered Field)


class ModeLauncher:
    """
    Helper class for launching modes with proper excitation.

    Handles mode injection, normalization, and phase matching.

    Parameters
    ----------
    mode : WaveguideMode
        Mode to launch.
    direction : str
        Propagation direction.
    power : float, optional
        Mode power (W), default=1.0.
    """

    def __init__(
        self,
        mode: WaveguideMode,
        direction: str,
        power: float = 1.0,
    ):
        self.mode = mode
        self.direction = direction
        self.target_power = power

        # Calculate normalization factor
        self._calculate_normalization()

    def _calculate_normalization(self) -> None:
        """
        Calculate normalization factor to achieve target power.

        Power = (1/2) Re[∫∫ E × H* · n dA]
        """
        # Calculate mode power (simplified)
        # Proper implementation would integrate Poynting vector

        Ex, Ey, Ez = self.mode.Ex, self.mode.Ey, self.mode.Ez
        Hx, Hy, Hz = self.mode.Hx, self.mode.Hy, self.mode.Hz

        # Poynting vector (approximate)
        Sx = 0.5 * np.real(Ey * np.conj(Hz) - Ez * np.conj(Hy))
        Sy = 0.5 * np.real(Ez * np.conj(Hx) - Ex * np.conj(Hz))
        Sz = 0.5 * np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx))

        # Integrate over cross-section
        # (Simplified - needs proper area element)
        if "z" in self.direction:
            current_power = np.sum(Sz)
        elif "x" in self.direction:
            current_power = np.sum(Sx)
        else:
            current_power = np.sum(Sy)

        # Normalization factor
        if abs(current_power) > 1e-20:
            self.norm_factor = np.sqrt(self.target_power / abs(current_power))
        else:
            self.norm_factor = 1.0

    def get_normalized_mode(self) -> WaveguideMode:
        """
        Get mode with normalized power.

        Returns
        -------
        WaveguideMode
            Mode with fields scaled to achieve target power.
        """
        # Create copy of mode with normalized fields
        normalized_mode = WaveguideMode(
            mode_number=self.mode.mode_number,
            neff=self.mode.neff,
            frequency=self.mode.frequency,
            wavelength=self.mode.wavelength,
            Ex=self.mode.Ex * self.norm_factor,
            Ey=self.mode.Ey * self.norm_factor,
            Ez=self.mode.Ez * self.norm_factor,
            Hx=self.mode.Hx * self.norm_factor,
            Hy=self.mode.Hy * self.norm_factor,
            Hz=self.mode.Hz * self.norm_factor,
            x=self.mode.x,
            y=self.mode.y,
            power=self.target_power,
        )

        return normalized_mode
