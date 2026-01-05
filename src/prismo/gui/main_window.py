"""
Main application window for Prismo GUI.

This module provides the main window with menu structure, toolbars, and
layout for material visualization and simulation control using Dear PyGui.
"""

from typing import Optional

try:
    import dearpygui.dearpygui as dpg

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    dpg = None

if GUI_AVAILABLE:
    from .embedded_viewport import EmbeddedViewport3D
    from .property_plotter import PropertyPlotter, load_unicode_font
    from .results_viewer import ResultsViewer
    from .shape_dialog import ShapeDialog


class MainWindow:
    """
    Main application window for Prismo GUI.

    Provides the main interface with menus, toolbars, and layout for
    material visualization and simulation setup using Dear PyGui.
    """

    def __init__(self):
        """Initialize the main window."""
        if not GUI_AVAILABLE:
            raise ImportError(
                "Dear PyGui is required for GUI. Install with: pip install dearpygui"
            )

        # Initialize simulation state
        self.simulation: Optional["Simulation"] = None
        
        # Initialize shape dialog (will be created on first use)
        self.shape_dialog: Optional[ShapeDialog] = None

        # Initialize Dear PyGui context
        dpg.create_context()
        dpg.create_viewport(title="Prismo - Electromagnetic Simulation Tool", width=1200, height=800)
        
        # Load font with Greek symbol support (must be done before creating UI)
        # Create font registry and load Unicode font
        with dpg.font_registry() as font_registry:
            # Try to load Unicode font for Greek symbols
            unicode_font = load_unicode_font(font_registry=font_registry)
            # If no Unicode font found, unicode_font will be None
            # and we'll use Dear PyGui's default font (which may not support Greek symbols)
        
        self.unicode_font = unicode_font

        # Create primary window
        with dpg.window(label="Prismo", tag="primary_window"):
            # Apply Unicode font to the primary window and all children if available
            if self.unicode_font is not None:
                dpg.bind_item_font("primary_window", self.unicode_font)
            # Create menu bar (simulated with menu bar items)
            with dpg.menu_bar():
                with dpg.menu(label="File"):
                    dpg.add_menu_item(label="New Simulation", callback=self._new_simulation)
                    dpg.add_menu_item(label="Open...", callback=self._open_simulation)
                    dpg.add_menu_item(label="Save", callback=self._save_simulation)
                    dpg.add_menu_item(label="Save As...", callback=self._save_as_simulation)
                    dpg.add_separator()
                    dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())

                with dpg.menu(label="Edit"):
                    dpg.add_menu_item(label="Add Shape...", callback=self._show_add_shape_dialog)
                    dpg.add_separator()
                    dpg.add_menu_item(label="Preferences...", callback=self._show_preferences)

                with dpg.menu(label="View"):
                    dpg.add_menu_item(label="Material Viewer", callback=self._show_material_viewer)
                    dpg.add_menu_item(label="Property Plotter", callback=self._show_property_plotter)
                    dpg.add_menu_item(label="Results Viewer", callback=self._show_results_viewer)

                with dpg.menu(label="Simulation"):
                    dpg.add_menu_item(label="Run", callback=self._run_simulation)
                    dpg.add_menu_item(label="Stop", callback=self._stop_simulation)
                    dpg.add_menu_item(label="Reset", callback=self._reset_simulation)

                with dpg.menu(label="Help"):
                    dpg.add_menu_item(label="About", callback=self._show_about)
                    dpg.add_menu_item(label="Documentation", callback=self._show_documentation)

            # Create toolbar (simulated with buttons)
            with dpg.group(horizontal=True):
                dpg.add_button(label="New", callback=self._new_simulation)
                dpg.add_button(label="Open", callback=self._open_simulation)
                dpg.add_button(label="Save", callback=self._save_simulation)
                dpg.add_separator()
                dpg.add_button(label="Add Shape", callback=self._show_add_shape_dialog)
                dpg.add_separator()
                dpg.add_button(label="Run", callback=self._run_simulation)
                dpg.add_button(label="Stop", callback=self._stop_simulation)

            dpg.add_separator()

            # Create main content area with split layout
            with dpg.group(horizontal=True):
                # 3D Viewport (left side, 2/3 width)
                with dpg.child_window(width=800, height=-1, tag="viewport_container"):
                    # Create embedded viewport widget
                    self.viewport = EmbeddedViewport3D(width=800, height=600)
                    self.viewport.create_controls(parent="viewport_container")

                # Right panel (1/3 width) - split into shapes list and property plotter
                with dpg.child_window(width=-1, height=-1):
                    # Shapes list
                    with dpg.collapsing_header(label="Geometry", default_open=True):
                        dpg.add_text("Add shapes (boxes, spheres, cylinders) to your simulation:", color=(200, 200, 200))
                        self.shapes_list = dpg.add_text("No shapes added yet", tag="shapes_list_text")
                        dpg.add_button(
                            label="Add Shape",
                            callback=self._show_add_shape_dialog,
                            tag="add_shape_button",
                        )
                    
                    dpg.add_separator()
                    
                    # Property plotter
                    self.property_plotter = PropertyPlotter(font_id=self.unicode_font)
                    
                    dpg.add_separator()
                    
                    # Results viewer
                    with dpg.collapsing_header(label="Results Viewer", default_open=False, tag="results_viewer_header"):
                        self.results_viewer = ResultsViewer(font_id=self.unicode_font)
                        dpg.add_separator()
                        dpg.add_button(label="Load Results...", callback=self._load_results_file)
                        dpg.add_button(label="Load from Monitor", callback=self._load_results_from_monitor)

            # Status bar (simulated with text at bottom)
            dpg.add_separator()
            self.status_text = dpg.add_text("Ready", tag="status_text")

        # Set primary window
        dpg.setup_dearpygui()
        dpg.set_primary_window("primary_window", True)

        # Initialize with default simulation
        self._new_simulation()

    def show(self) -> None:
        """Show the main window."""
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

    def _new_simulation(self) -> None:
        """Create a new simulation."""
        # Import Simulation here to avoid circular imports
        from prismo.core.simulation import Simulation

        # Create a new simulation with default parameters
        self.simulation = Simulation(
            size=(10.0e-6, 10.0e-6, 1.0e-6),  # 10µm x 10µm x 1µm
            resolution=20.0e6,  # 20 points per µm
            boundary_conditions="pml",
            pml_layers=10,
        )

        # Clear and reset GUI state
        if hasattr(self, "viewport"):
            self.viewport.clear()

        # Sync viewport with new simulation
        if hasattr(self, "viewport"):
            self.viewport.sync_with_simulation(self.simulation)

        # Update shapes list display
        self._update_shapes_list()

        dpg.set_value("status_text", "New simulation created")

    def _open_simulation(self) -> None:
        """Open an existing simulation."""
        with dpg.file_dialog(
            directory_selector=False,
            show=True,
            callback=self._on_file_selected,
            tag="file_dialog_open",
            width=700,
            height=400,
        ):
            dpg.add_file_extension(".*", color=(255, 255, 255, 255))
            dpg.add_file_extension(".json", color=(0, 255, 0, 255))
            dpg.add_file_extension(".yaml", color=(0, 255, 0, 255))
            dpg.add_file_extension(".yml", color=(0, 255, 0, 255))

    def _on_file_selected(self, sender, app_data) -> None:
        """Handle file selection from dialog."""
        if app_data["file_name"]:
            dpg.set_value("status_text", f"Opening: {app_data['file_name']}")

    def _save_simulation(self) -> None:
        """Save the current simulation."""
        dpg.set_value("status_text", "Save simulation...")

    def _save_as_simulation(self) -> None:
        """Save the current simulation with a new name."""
        with dpg.file_dialog(
            directory_selector=False,
            show=True,
            callback=self._on_save_selected,
            tag="file_dialog_save",
            width=700,
            height=400,
            default_filename="simulation.json",
        ):
            dpg.add_file_extension(".*", color=(255, 255, 255, 255))
            dpg.add_file_extension(".json", color=(0, 255, 0, 255))
            dpg.add_file_extension(".yaml", color=(0, 255, 0, 255))
            dpg.add_file_extension(".yml", color=(0, 255, 0, 255))

    def _on_save_selected(self, sender, app_data) -> None:
        """Handle save file selection."""
        if app_data["file_name"]:
            dpg.set_value("status_text", f"Saving: {app_data['file_name']}")

    def _show_preferences(self) -> None:
        """Show preferences dialog."""
        dpg.set_value("status_text", "Preferences...")

    def _run_simulation(self) -> None:
        """Run the simulation."""
        dpg.set_value("status_text", "Running simulation...")

    def _stop_simulation(self) -> None:
        """Stop the running simulation."""
        dpg.set_value("status_text", "Stopping simulation...")

    def _reset_simulation(self) -> None:
        """Reset the simulation."""
        dpg.set_value("status_text", "Resetting simulation...")

    def _show_material_viewer(self) -> None:
        """Show material viewer window (now integrated in main viewport)."""
        if hasattr(self, "viewport"):
            self.viewport._open_viewport_window()

    def _show_property_plotter(self) -> None:
        """Show property plotter window."""
        if hasattr(self, "property_plotter"):
            self.property_plotter.show()

    def _show_results_viewer(self) -> None:
        """Show results viewer (opens collapsing header)."""
        if dpg.does_item_exist("results_viewer_header"):
            dpg.configure_item("results_viewer_header", default_open=True)
            dpg.set_value("status_text", "Results Viewer opened")

    def _load_results_file(self) -> None:
        """Load results from a file."""
        with dpg.file_dialog(
            directory_selector=False,
            show=True,
            callback=self._on_results_file_selected,
            tag="file_dialog_results",
            width=700,
            height=400,
        ):
            dpg.add_file_extension(".*", color=(255, 255, 255, 255))
            dpg.add_file_extension(".csv", color=(0, 255, 0, 255))
            dpg.add_file_extension(".parquet", color=(0, 255, 0, 255))

    def _on_results_file_selected(self, sender, app_data) -> None:
        """Handle results file selection."""
        if app_data.get("file_name"):
            filepath = app_data["file_name"]
            try:
                from .results_loader import load_from_file

                data = load_from_file(filepath)
                if hasattr(self, "results_viewer"):
                    self.results_viewer.load_data(data)
                    dpg.set_value("status_text", f"Loaded results from: {filepath}")
                else:
                    dpg.set_value("status_text", "Results viewer not initialized")
            except Exception as e:
                dpg.set_value("status_text", f"Error loading results: {str(e)}")

    def _load_results_from_monitor(self) -> None:
        """Load results from simulation monitors."""
        if self.simulation is None:
            dpg.set_value("status_text", "No simulation available. Create a simulation first.")
            return

        if not hasattr(self.simulation, "monitors") or len(self.simulation.monitors) == 0:
            dpg.set_value("status_text", "No monitors found in simulation")
            return

        # Get first monitor (could be extended to show a selection dialog)
        monitor = self.simulation.monitors[0]
        try:
            from .results_loader import load_from_monitor

            data = load_from_monitor(monitor)
            if hasattr(self, "results_viewer"):
                self.results_viewer.load_data(data)
                dpg.set_value("status_text", f"Loaded results from monitor: {monitor.name}")
            else:
                dpg.set_value("status_text", "Results viewer not initialized")
        except Exception as e:
            dpg.set_value("status_text", f"Error loading monitor data: {str(e)}")

    def _show_about(self) -> None:
        """Show about dialog."""
        with dpg.window(label="About Prismo", modal=True, show=True, tag="about_window"):
            dpg.add_text("Prismo - Electromagnetic Simulation Tool")
            dpg.add_separator()
            dpg.add_text("A high-performance Python-based FDTD/FEM solver for photonics.")
            dpg.add_separator()
            dpg.add_button(label="OK", callback=lambda: dpg.delete_item("about_window"))

    def _show_documentation(self) -> None:
        """Open documentation in browser."""
        import webbrowser

        webbrowser.open("https://prismo.readthedocs.io")
        dpg.set_value("status_text", "Opening documentation...")

    def _show_add_shape_dialog(self) -> None:
        """Show the shape creation dialog."""
        if self.simulation is None:
            dpg.set_value("status_text", "Error: No simulation created. Create a new simulation first.")
            return

        # Create shape dialog if it doesn't exist
        if self.shape_dialog is None:
            self.shape_dialog = ShapeDialog(
                on_shape_created=self._on_shape_created,
                font_id=self.unicode_font
            )

        # Show the dialog
        self.shape_dialog.show()

    def _on_shape_created(self, shape) -> None:
        """
        Handle shape creation from the dialog.

        Parameters
        ----------
        shape : Shape
            The created shape object.
        """
        if self.simulation is None:
            return

        # Add shape to simulation
        self.simulation.add_shape(shape)

        # Update viewport to show new shape
        if hasattr(self, "viewport"):
            # Sync entire simulation (including new shape)
            self.viewport.sync_with_simulation(self.simulation)

        # Update shapes list display
        self._update_shapes_list()

        # Update status
        shape_type = type(shape).__name__
        status_msg = f"Added {shape_type} ({shape.material.name}). Open the 3D View to see it!"
        dpg.set_value("status_text", status_msg)

    def _update_shapes_list(self) -> None:
        """Update the shapes list display in the GUI."""
        if not dpg.does_item_exist("shapes_list_text"):
            return

        if self.simulation is None or len(self.simulation.shapes) == 0:
            dpg.set_value("shapes_list_text", "No shapes added yet")
            return

        # Create a formatted list of shapes
        shapes_info = []
        for i, shape in enumerate(self.simulation.shapes):
            shape_type = type(shape).__name__
            material_name = shape.material.name
            center = shape.center
            if hasattr(shape, "size"):
                size = shape.size
                info = f"{i+1}. {shape_type} ({material_name}) - Center: ({center[0]:.2e}, {center[1]:.2e}, {center[2]:.2e}), Size: {size}"
            elif hasattr(shape, "radius"):
                radius = shape.radius
                if hasattr(shape, "height"):
                    height = shape.height
                    info = f"{i+1}. {shape_type} ({material_name}) - Center: ({center[0]:.2e}, {center[1]:.2e}, {center[2]:.2e}), Radius: {radius:.2e}, Height: {height:.2e}"
                else:
                    info = f"{i+1}. {shape_type} ({material_name}) - Center: ({center[0]:.2e}, {center[1]:.2e}, {center[2]:.2e}), Radius: {radius:.2e}"
            else:
                info = f"{i+1}. {shape_type} ({material_name}) - Center: ({center[0]:.2e}, {center[1]:.2e}, {center[2]:.2e})"
            shapes_info.append(info)

        shapes_text = "\n".join(shapes_info)
        dpg.set_value("shapes_list_text", shapes_text)
