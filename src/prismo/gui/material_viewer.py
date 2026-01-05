"""
3D material visualization widget.

This module provides a widget for visualizing material distributions and
geometries in 3D with interactive controls using Dear PyGui.
"""

from typing import Optional

try:
    import dearpygui.dearpygui as dpg

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    dpg = None

# Try to import 3D visualization libraries
try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None

if PYVISTA_AVAILABLE:
    try:
        from .viewport_3d import Viewport3D
    except ImportError:
        Viewport3D = None  # type: ignore
else:
    Viewport3D = None  # type: ignore


class MaterialViewer:
    """
    3D material visualization widget.

    Provides interactive 3D visualization of material distributions,
    geometry, and field overlays using Dear PyGui.
    """

    def __init__(self):
        """Initialize the material viewer."""
        if not GUI_AVAILABLE:
            raise ImportError(
                "Dear PyGui is required for GUI. Install with: pip install dearpygui"
            )

        # Create header
        dpg.add_text("3D Material Visualization", color=(255, 255, 255))

        # Initialize viewport
        self.viewport: Optional[Viewport3D] = None
        self.current_slice: Optional[str] = None
        self.slice_position: float = 0.0

        # Create 3D viewport placeholder
        # Dear PyGui doesn't have native 3D, but we can use PyVista for rendering
        # or create a 2D slice viewer
        if PYVISTA_AVAILABLE:
            self._setup_pyvista_viewport()
        else:
            # Fallback to 2D slice viewer using Dear PyGui's plot widget
            self._setup_2d_viewport()

        # Add controls
        with dpg.group(horizontal=True):
            dpg.add_button(label="Open 3D View", callback=self._open_viewport)
            dpg.add_button(label="Reset View", callback=self._reset_view)
            dpg.add_button(label="Slice XY", callback=lambda: self._set_slice("xy"))
            dpg.add_button(label="Slice XZ", callback=lambda: self._set_slice("xz"))
            dpg.add_button(label="Slice YZ", callback=lambda: self._set_slice("yz"))
            dpg.add_button(label="Clear Slice", callback=self._clear_slice)

    def _setup_pyvista_viewport(self) -> None:
        """Set up PyVista-based 3D viewport."""
        # Create viewport instance (will be shown when "Open 3D View" is clicked)
        if Viewport3D is None:
            dpg.add_text("PyVista not available. Install with: pip install pyvista", color=(255, 200, 200))
            dpg.add_separator()
            return

        try:
            self.viewport = Viewport3D(
                window_size=(800, 600),
                title="Prismo 3D Material Visualization",
                show_axes=True,
                background="white",
            )
            dpg.add_text("PyVista 3D viewport ready. Click 'Open 3D View' to display.", color=(200, 255, 200))
        except Exception as e:
            dpg.add_text(f"Error initializing PyVista: {e}", color=(255, 200, 200))
            self.viewport = None
        dpg.add_separator()

    def _setup_2d_viewport(self) -> None:
        """Set up 2D slice viewer using Dear PyGui plot widget."""
        # Add informational text before the plot
        dpg.add_text("2D slice visualization (install pyvista for 3D)", color=(200, 200, 200))
        dpg.add_separator()
        
        # Create a plot for 2D slice visualization
        with dpg.plot(label="Material Slice", height=400, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="X", tag="material_slice_x_axis")
            dpg.add_plot_axis(dpg.mvYAxis, label="Y", tag="material_slice_y_axis")

            # Placeholder: would add actual material data here
            # Note: Plot children must be plot-specific items (series, annotations, etc.)

    def _open_viewport(self) -> None:
        """Open the 3D viewport window."""
        if self.viewport is not None:
            # Show viewport in non-blocking mode
            self.viewport.show(interactive=True)

    def _reset_view(self) -> None:
        """Reset the view to default."""
        if self.viewport is not None:
            self.viewport.reset_camera()
            self.viewport.update()

    def _set_slice(self, plane: str) -> None:
        """
        Set slice view for a specific plane.

        Parameters
        ----------
        plane : str
            Plane to slice ('xy', 'xz', 'yz').
        """
        if self.viewport is not None:
            # Remove existing slice if different plane
            if self.current_slice is not None and self.current_slice != plane:
                self.viewport.remove_slice_plane(self.current_slice)

            # Add new slice plane
            self.viewport.add_slice_plane(plane, position=self.slice_position)
            self.current_slice = plane
            self.viewport.update()

    def _clear_slice(self) -> None:
        """Clear the current slice plane."""
        if self.viewport is not None and self.current_slice is not None:
            self.viewport.remove_slice_plane(self.current_slice)
            self.current_slice = None
            self.viewport.update()

    def show(self) -> None:
        """Show the material viewer (already visible in main window)."""
        pass

    def update_materials(self, materials: dict) -> None:
        """
        Update the displayed materials.

        Parameters
        ----------
        materials : dict
            Material dictionary with geometry and properties.
            Expected format: {name: {'material': Material, 'shapes': [Shape, ...]}}
        """
        if self.viewport is None:
            return

        # Clear existing geometry
        self.viewport.clear()

        # Add shapes from materials
        for name, data in materials.items():
            if "shapes" in data:
                for shape in data["shapes"]:
                    shape_name = f"{name}_{len(self.viewport.meshes)}"
                    self.viewport.add_shape(shape, name=shape_name)
            elif "shape" in data:
                # Single shape
                self.viewport.add_shape(data["shape"], name=name)

        # Update viewport
        self.viewport.update()

    def update_geometry(self, shapes: list) -> None:
        """
        Update the displayed geometry with a list of shapes.

        Parameters
        ----------
        shapes : list
            List of Shape objects to display.
        """
        if self.viewport is None:
            return

        # Clear existing geometry
        self.viewport.clear()

        # Add all shapes
        for i, shape in enumerate(shapes):
            self.viewport.add_shape(shape, name=f"shape_{i}")

        # Update viewport
        self.viewport.update()

    def set_slice_view(self, plane: str, position: float) -> None:
        """
        Set slice view for a specific plane.

        Parameters
        ----------
        plane : str
            Plane to slice ('xy', 'xz', 'yz').
        position : float
            Position along the normal axis.
        """
        if self.viewport is None:
            return

        self.slice_position = position

        # Remove existing slice if different plane
        if self.current_slice is not None and self.current_slice != plane:
            self.viewport.remove_slice_plane(self.current_slice)

        # Add/update slice plane
        self.viewport.add_slice_plane(plane, position=position)
        self.current_slice = plane
        self.viewport.update()
