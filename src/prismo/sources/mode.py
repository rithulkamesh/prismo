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

        # Determine source region coordinates based on propagation direction
        if self.axis == "z":
            # Mode profile in xy-plane
            # Get actual source region coordinates
            x_start = self.center[0] - self.size[0] / 2
            x_end = self.center[0] + self.size[0] / 2
            y_start = self.center[1] - self.size[1] / 2
            y_end = self.center[1] + self.size[1] / 2

            # Create grid in source region
            nx_src = max(int(self.size[0] / self._grid.dx), len(x_mode))
            ny_src = max(int(self.size[1] / self._grid.dy), len(y_mode))

            x_sim = np.linspace(x_start, x_end, nx_src)
            y_sim = np.linspace(y_start, y_end, ny_src)

        elif self.axis == "x":
            # Mode profile in yz-plane
            y_start = self.center[1] - self.size[1] / 2
            y_end = self.center[1] + self.size[1] / 2
            z_start = self.center[2] - self.size[2] / 2 if self._grid.is_3d else 0
            z_end = self.center[2] + self.size[2] / 2 if self._grid.is_3d else 0

            ny_src = max(int(self.size[1] / self._grid.dy), len(x_mode))
            nz_src = (
                max(int(self.size[2] / self._grid.dz), len(y_mode))
                if self._grid.is_3d
                else len(y_mode)
            )

            x_sim = np.linspace(y_start, y_end, ny_src)
            y_sim = np.linspace(z_start, z_end, nz_src)

        else:  # y axis
            # Mode profile in xz-plane
            x_start = self.center[0] - self.size[0] / 2
            x_end = self.center[0] + self.size[0] / 2
            z_start = self.center[2] - self.size[2] / 2 if self._grid.is_3d else 0
            z_end = self.center[2] + self.size[2] / 2 if self._grid.is_3d else 0

            nx_src = max(int(self.size[0] / self._grid.dx), len(x_mode))
            nz_src = (
                max(int(self.size[2] / self._grid.dz), len(y_mode))
                if self._grid.is_3d
                else len(y_mode)
            )

            x_sim = np.linspace(x_start, x_end, nx_src)
            y_sim = np.linspace(z_start, z_end, nz_src)

        # Create interpolators for each field component (handles complex fields)
        def make_interpolator(field_data):
            # Handle 2D data
            if field_data.ndim == 2:
                # Interpolate real and imaginary parts separately
                interp_real = RegularGridInterpolator(
                    (x_mode, y_mode),
                    field_data.real,
                    bounds_error=False,
                    fill_value=0.0,
                )
                interp_imag = RegularGridInterpolator(
                    (x_mode, y_mode),
                    field_data.imag,
                    bounds_error=False,
                    fill_value=0.0,
                )
                return interp_real, interp_imag
            return None, None

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
        def interpolate_field(interp_tuple):
            if interp_tuple[0] is not None:
                real_part = interp_tuple[0](points).reshape(X_sim.shape)
                imag_part = interp_tuple[1](points).reshape(X_sim.shape)
                return real_part + 1j * imag_part
            return None

        self._mode_profile_Ex = interpolate_field(interp_Ex)
        self._mode_profile_Ey = interpolate_field(interp_Ey)
        self._mode_profile_Ez = interpolate_field(interp_Ez)
        self._mode_profile_Hx = interpolate_field(interp_Hx)
        self._mode_profile_Hy = interpolate_field(interp_Hy)
        self._mode_profile_Hz = interpolate_field(interp_Hz)

        # Store interpolated grid info
        self._interp_shape = X_sim.shape
        self._interp_x = x_sim
        self._interp_y = y_sim

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
        omega = 2 * np.pi * self.mode.frequency

        # Time-harmonic variation
        time_phase = -omega * time + self.phase

        # Total complex amplitude
        amplitude_complex = self.amplitude * waveform_value * np.exp(1j * time_phase)

        # Extract real part for time-domain injection
        amplitude_real = amplitude_complex.real

        # Get source region indices
        x_min, x_max, y_min, y_max, z_min, z_max = self._compute_source_region()

        # Add mode fields to simulation based on propagation direction
        if self.axis == "z":
            # Mode propagating in z, inject in xy-plane
            self._inject_z_mode(
                fields, amplitude_real, x_min, x_max, y_min, y_max, z_min
            )
        elif self.axis == "x":
            # Mode propagating in x, inject in yz-plane
            self._inject_x_mode(
                fields, amplitude_real, x_min, y_min, y_max, z_min, z_max
            )
        else:  # y
            # Mode propagating in y, inject in xz-plane
            self._inject_y_mode(
                fields, amplitude_real, x_min, x_max, y_min, z_min, z_max
            )

    def _inject_z_mode(
        self,
        fields: ElectromagneticFields,
        amplitude: float,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        z_idx: int,
    ) -> None:
        """Inject mode propagating in +z or -z direction."""
        if self._mode_profile_Ex is not None:
            # Add tangential E-fields and normal H-field
            Ex_contribution = amplitude * self._mode_profile_Ex.real
            Ey_contribution = amplitude * self._mode_profile_Ey.real
            Hz_contribution = amplitude * self._mode_profile_Hz.real * self.sign

            # Resize to match field region if needed
            nx_field = x_max - x_min
            ny_field = y_max - y_min

            if Ex_contribution.shape != (nx_field, ny_field):
                from scipy.ndimage import zoom

                zoom_x = nx_field / Ex_contribution.shape[0]
                zoom_y = ny_field / Ex_contribution.shape[1]
                Ex_contribution = zoom(Ex_contribution, (zoom_x, zoom_y), order=1)
                Ey_contribution = zoom(Ey_contribution, (zoom_x, zoom_y), order=1)
                Hz_contribution = zoom(Hz_contribution, (zoom_x, zoom_y), order=1)

            # Add to fields
            if hasattr(fields, "Ex"):
                fields.Ex[x_min:x_max, y_min:y_max, z_idx] += Ex_contribution
            if hasattr(fields, "Ey"):
                fields.Ey[x_min:x_max, y_min:y_max, z_idx] += Ey_contribution
            if hasattr(fields, "Hz"):
                fields.Hz[x_min:x_max, y_min:y_max, z_idx] += Hz_contribution

    def _inject_x_mode(
        self,
        fields: ElectromagneticFields,
        amplitude: float,
        x_idx: int,
        y_min: int,
        y_max: int,
        z_min: int,
        z_max: int,
    ) -> None:
        """Inject mode propagating in +x or -x direction."""
        if self._mode_profile_Ey is not None:
            Ey_contribution = amplitude * self._mode_profile_Ey.real
            Ez_contribution = amplitude * self._mode_profile_Ez.real
            Hx_contribution = amplitude * self._mode_profile_Hx.real * self.sign

            ny_field = y_max - y_min
            nz_field = z_max - z_min

            if Ey_contribution.shape != (ny_field, nz_field):
                from scipy.ndimage import zoom

                zoom_y = ny_field / Ey_contribution.shape[0]
                zoom_z = nz_field / Ey_contribution.shape[1]
                Ey_contribution = zoom(Ey_contribution, (zoom_y, zoom_z), order=1)
                Ez_contribution = zoom(Ez_contribution, (zoom_y, zoom_z), order=1)
                Hx_contribution = zoom(Hx_contribution, (zoom_y, zoom_z), order=1)

            if hasattr(fields, "Ey"):
                fields.Ey[x_idx, y_min:y_max, z_min:z_max] += Ey_contribution
            if hasattr(fields, "Ez"):
                fields.Ez[x_idx, y_min:y_max, z_min:z_max] += Ez_contribution
            if hasattr(fields, "Hx"):
                fields.Hx[x_idx, y_min:y_max, z_min:z_max] += Hx_contribution

    def _inject_y_mode(
        self,
        fields: ElectromagneticFields,
        amplitude: float,
        x_min: int,
        x_max: int,
        y_idx: int,
        z_min: int,
        z_max: int,
    ) -> None:
        """Inject mode propagating in +y or -y direction."""
        if self._mode_profile_Ex is not None:
            Ex_contribution = amplitude * self._mode_profile_Ex.real
            Ez_contribution = amplitude * self._mode_profile_Ez.real
            Hy_contribution = amplitude * self._mode_profile_Hy.real * self.sign

            nx_field = x_max - x_min
            nz_field = z_max - z_min

            if Ex_contribution.shape != (nx_field, nz_field):
                from scipy.ndimage import zoom

                zoom_x = nx_field / Ex_contribution.shape[0]
                zoom_z = nz_field / Ex_contribution.shape[1]
                Ex_contribution = zoom(Ex_contribution, (zoom_x, zoom_z), order=1)
                Ez_contribution = zoom(Ez_contribution, (zoom_x, zoom_z), order=1)
                Hy_contribution = zoom(Hy_contribution, (zoom_x, zoom_z), order=1)

            if hasattr(fields, "Ex"):
                fields.Ex[x_min:x_max, y_idx, z_min:z_max] += Ex_contribution
            if hasattr(fields, "Ez"):
                fields.Ez[x_min:x_max, y_idx, z_min:z_max] += Ez_contribution
            if hasattr(fields, "Hy"):
                fields.Hy[x_min:x_max, y_idx, z_min:z_max] += Hy_contribution


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
