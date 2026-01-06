"""
Embedded 3D viewport widget for Dear PyGui.

This module provides an embedded viewport widget that wraps Viewport3D
and integrates it into the Dear PyGui interface with slice plane controls.
"""

from typing import Optional

try:
    import dearpygui.dearpygui as dpg

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    dpg = None

try:
    from .viewport_3d import Viewport3D

    VIEWPORT_AVAILABLE = True
except ImportError:
    VIEWPORT_AVAILABLE = False
    Viewport3D = None


class EmbeddedViewport3D:
    """
    Embedded 3D viewport widget for Dear PyGui.

    This class wraps Viewport3D and provides Dear PyGui controls for
    slice planes and viewport interaction. The actual 3D rendering
    happens in a separate window that stays synchronized.
    """

    def __init__(self, width: int = 800, height: int = 600):
        """
        Initialize the embedded viewport widget.

        Parameters
        ----------
        width : int
            Width of the viewport in pixels.
        height : int
            Height of the viewport in pixels.
        """
        if not GUI_AVAILABLE:
            raise ImportError("Dear PyGui is required for GUI")
        if not VIEWPORT_AVAILABLE:
            raise ImportError("Viewport3D is required for 3D visualization")

        self.width = width
        self.height = height

        # Create the actual viewport (will be shown when needed)
        self.viewport: Optional[Viewport3D] = None
        self.viewport_open = False
        
        # Store current simulation for syncing when window opens
        self.current_simulation = None

        # Slice plane states
        self.slice_xy_enabled = False
        self.slice_xz_enabled = False
        self.slice_yz_enabled = False
        self.slice_xy_position = 0.0
        self.slice_xz_position = 0.0
        self.slice_yz_position = 0.0

        # Simulation bounds for slice plane range
        self.simulation_size = (10.0, 10.0, 10.0)
        self.simulation_center = (0.0, 0.0, 0.0)

    def create_controls(self, parent: Optional[int] = None) -> None:
        """
        Create Dear PyGui controls for the viewport.

        Parameters
        ----------
        parent : int, optional
            Parent window/item ID. If None, uses default.
        """
        # Helpful text
        dpg.add_text("3D Visualization Controls", color=(255, 255, 255), parent=parent)
        dpg.add_text("Note: The 3D view opens in a separate window. Click 'Open 3D View' below.", color=(180, 180, 180), parent=parent)
        dpg.add_separator(parent=parent)
        
        # Viewport controls group
        with dpg.group(parent=parent, horizontal=True):
            dpg.add_button(
                label="Open 3D View",
                callback=self._open_viewport_window,
                tag="viewport_open_button",
            )
            dpg.add_button(
                label="Reset View",
                callback=self._reset_camera,
                tag="viewport_reset_button",
            )

        dpg.add_separator(parent=parent)

        # Slice plane controls
        with dpg.collapsing_header(
            label="Slice Planes", parent=parent, default_open=True
        ):
            # XY slice (horizontal)
            with dpg.group(parent=parent, horizontal=True):
                dpg.add_checkbox(
                    label="XY Plane",
                    default_value=False,
                    callback=lambda s, a: self._toggle_slice("xy", a),
                    tag="slice_xy_checkbox",
                )
                dpg.add_slider_float(
                    label="Z Position",
                    default_value=0.0,
                    min_value=-5.0,
                    max_value=5.0,
                    callback=lambda s, a: self._update_slice_position("xy", a),
                    tag="slice_xy_slider",
                    width=200,
                )

            # XZ slice (vertical along y)
            with dpg.group(parent=parent, horizontal=True):
                dpg.add_checkbox(
                    label="XZ Plane",
                    default_value=False,
                    callback=lambda s, a: self._toggle_slice("xz", a),
                    tag="slice_xz_checkbox",
                )
                dpg.add_slider_float(
                    label="Y Position",
                    default_value=0.0,
                    min_value=-5.0,
                    max_value=5.0,
                    callback=lambda s, a: self._update_slice_position("xz", a),
                    tag="slice_xz_slider",
                    width=200,
                )

            # YZ slice (vertical along x)
            with dpg.group(parent=parent, horizontal=True):
                dpg.add_checkbox(
                    label="YZ Plane",
                    default_value=False,
                    callback=lambda s, a: self._toggle_slice("yz", a),
                    tag="slice_yz_checkbox",
                )
                dpg.add_slider_float(
                    label="X Position",
                    default_value=0.0,
                    min_value=-5.0,
                    max_value=5.0,
                    callback=lambda s, a: self._update_slice_position("yz", a),
                    tag="slice_yz_slider",
                    width=200,
                )

    def initialize_viewport(self) -> None:
        """Initialize the PyVista viewport."""
        if self.viewport is None:
            self.viewport = Viewport3D(
                window_size=(self.width, self.height),
                title="Prismo 3D Viewport",
                show_axes=True,
                background="white",
            )

    def _open_viewport_window(self) -> None:
        """Open the 3D viewport window."""
        if self.viewport is None:
            self.initialize_viewport()

        if self.viewport is not None:
            # Show in non-blocking mode
            self.viewport.show(interactive=False)
            self.viewport_open = True
            
            # Sync with current simulation if one exists
            if self.current_simulation is not None:
                self.viewport.sync_with_simulation(self.current_simulation)
            
            # Update the viewport after opening
            try:
                self.viewport.update()
            except RuntimeError:
                # Window not fully initialized yet, that's okay
                pass
            
            if dpg.does_item_exist("status_text"):
                dpg.set_value("status_text", "3D viewport opened")

    def _reset_camera(self) -> None:
        """Reset the viewport camera."""
        if self.viewport is not None and self.viewport_open:
            self.viewport.reset_camera()
            self.viewport.update()

    def _update_viewport(self) -> None:
        """Update the viewport display."""
        if self.viewport is not None and self.viewport_open:
            self.viewport.update()

    def _toggle_slice(self, plane: str, enabled: bool) -> None:
        """
        Toggle a slice plane on/off.

        Parameters
        ----------
        plane : str
            Plane name ('xy', 'xz', 'yz').
        enabled : bool
            Whether the slice plane should be enabled.
        """
        if self.viewport is None:
            return

        if enabled:
            # Get current position
            if plane == "xy":
                position = self.slice_xy_position
            elif plane == "xz":
                position = self.slice_xz_position
            elif plane == "yz":
                position = self.slice_yz_position
            else:
                return

            self.viewport.add_slice_plane(plane, position=position, enabled=True)
        else:
            self.viewport.remove_slice_plane(plane)

        if self.viewport_open:
            self.viewport.update()

    def _update_slice_position(self, plane: str, position: float) -> None:
        """
        Update the position of a slice plane.

        Parameters
        ----------
        plane : str
            Plane name ('xy', 'xz', 'yz').
        position : float
            New position along the normal axis.
        """
        if self.viewport is None:
            return

        # Store position
        if plane == "xy":
            self.slice_xy_position = position
        elif plane == "xz":
            self.slice_xz_position = position
        elif plane == "yz":
            self.slice_yz_position = position

        # Update viewport if plane is enabled
        enabled = self.viewport.slice_plane_enabled.get(plane, False)
        if enabled:
            self.viewport.update_slice_plane_position(plane, position)
            if self.viewport_open:
                self.viewport.update()

    def sync_with_simulation(self, simulation) -> None:
        """
        Synchronize the viewport with a Simulation object.

        Parameters
        ----------
        simulation : Simulation
            The simulation object to sync with.
        """
        # Store simulation reference
        self.current_simulation = simulation
        
        if self.viewport is None:
            self.initialize_viewport()

        if self.viewport is not None:
            # Update simulation bounds for slice plane ranges
            self.simulation_size = simulation.size
            self.simulation_center = (
                simulation.size[0] / 2,
                simulation.size[1] / 2,
                simulation.size[2] / 2,
            )

            # Update slice plane slider ranges
            max_range = max(self.simulation_size) / 2
            if dpg.does_item_exist("slice_xy_slider"):
                dpg.configure_item(
                    "slice_xy_slider",
                    min_value=-max_range,
                    max_value=max_range,
                )
            if dpg.does_item_exist("slice_xz_slider"):
                dpg.configure_item(
                    "slice_xz_slider",
                    min_value=-max_range,
                    max_value=max_range,
                )
            if dpg.does_item_exist("slice_yz_slider"):
                dpg.configure_item(
                    "slice_yz_slider",
                    min_value=-max_range,
                    max_value=max_range,
                )

            # Sync viewport with simulation
            self.viewport.sync_with_simulation(simulation)
            # Only update if viewport window is open
            if self.viewport_open:
                self.viewport.update()

    def clear(self) -> None:
        """Clear all geometry from the viewport."""
        if self.viewport is not None:
            self.viewport.clear()

    def close(self) -> None:
        """Close the viewport."""
        if self.viewport is not None:
            self.viewport.close()
            self.viewport = None
            self.viewport_open = False

