"""
Shape creation dialog for Prismo GUI.

This module provides a dialog window for creating geometric shapes
with material assignment.
"""

from typing import Optional, Callable

try:
    import dearpygui.dearpygui as dpg

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    dpg = None

if GUI_AVAILABLE:
    from prismo.geometry.shapes import Box, Sphere, Cylinder, Material
    from prismo.materials.library import list_materials, get_material


class ShapeDialog:
    """
    Dialog for creating geometric shapes with material assignment.
    """

    def __init__(self, on_shape_created: Optional[Callable] = None, font_id: Optional[int] = None):
        """
        Initialize the shape dialog.

        Parameters
        ----------
        on_shape_created : callable, optional
            Callback function called when a shape is created.
            Signature: callback(shape) -> None
        font_id : int, optional
            Font ID for Unicode/Greek symbol support.
        """
        if not GUI_AVAILABLE:
            raise ImportError("Dear PyGui is required for GUI")

        self.on_shape_created = on_shape_created
        self.current_shape_type = "Box"
        self.dialog_tag = "shape_dialog"
        self.font_id = font_id
        self.shape_params_container_tag = "shape_params_container"
        
        # Default values
        self.shape_params = {
            "center_x": 5.0e-6,
            "center_y": 5.0e-6,
            "center_z": 0.5e-6,
            "size_x": 1.0e-6,
            "size_y": 1.0e-6,
            "size_z": 0.22e-6,
            "radius": 1.0e-6,
            "height": 1.0e-6,
            "axis": "z",
            "material_name": "Si",
            "epsilon_r": 12.0,
        }

    def show(self) -> None:
        """Show the shape creation dialog."""
        if dpg.does_item_exist(self.dialog_tag):
            dpg.delete_item(self.dialog_tag)

        with dpg.window(
            label="Add Shape to Simulation",
            modal=True,
            show=True,
            tag=self.dialog_tag,
            width=500,
            height=600,
        ):
            # Apply font if available
            if self.font_id is not None:
                dpg.bind_item_font(self.dialog_tag, self.font_id)
            
            dpg.add_text("Choose the type of shape you want to add:", color=(255, 255, 255))
            dpg.add_separator()
            # Shape type selection
            dpg.add_text("Shape Type:")
            dpg.add_radio_button(
                ["Box (Rectangle)", "Sphere (Ball)", "Cylinder (Tube)"],
                default_value=0,
                callback=self._on_shape_type_changed,
                tag="shape_type_radio",
            )

            dpg.add_separator()

            # Common parameters: Position
            dpg.add_text("Position (meters):")
            with dpg.group(horizontal=True):
                dpg.add_input_float(
                    label="X",
                    default_value=self.shape_params["center_x"],
                    format="%.2e",
                    width=150,
                    tag="center_x_input",
                )
                dpg.add_input_float(
                    label="Y",
                    default_value=self.shape_params["center_y"],
                    format="%.2e",
                    width=150,
                    tag="center_y_input",
                )
            with dpg.group(horizontal=True):
                dpg.add_input_float(
                    label="Z",
                    default_value=self.shape_params["center_z"],
                    format="%.2e",
                    width=150,
                    tag="center_z_input",
                )

            dpg.add_separator()

            # Shape-specific parameters container
            # Create a container that we can populate dynamically
            with dpg.group(tag=self.shape_params_container_tag):
                pass  # Will be populated by _create_shape_specific_controls
            
            # Initial shape-specific controls
            self._create_shape_specific_controls()

            dpg.add_separator()

            # Material selection
            dpg.add_text("Material:")
            with dpg.group(horizontal=True):
                # Material from library
                materials = list_materials()
                if not materials:
                    materials = ["Custom"]
                
                dpg.add_combo(
                    items=["Custom"] + materials,
                    default_value="Si",
                    width=200,
                    callback=self._on_material_selected,
                    tag="material_combo",
                )

            # Custom material properties
            with dpg.group(horizontal=True):
                dpg.add_input_float(
                    label="εᵣ (Permittivity - Custom)",
                    default_value=self.shape_params["epsilon_r"],
                    format="%.3f",
                    width=200,
                    tag="epsilon_r_input",
                )

            dpg.add_separator()

            # Buttons
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Add Shape",
                    callback=self._on_add_shape,
                    width=150,
                )
                dpg.add_button(
                    label="Cancel",
                    callback=lambda: dpg.delete_item(self.dialog_tag),
                    width=150,
                )

    def _create_shape_specific_controls(self) -> None:
        """Create controls specific to the selected shape type."""
        # Check if dialog and container exist
        if not dpg.does_item_exist(self.dialog_tag):
            return
        
        if not dpg.does_item_exist(self.shape_params_container_tag):
            # Container doesn't exist, can't create controls
            return
        
        # Clear existing controls in the container
        # Delete all children of the container
        try:
            children = dpg.get_item_children(self.shape_params_container_tag, slot=1)
            if children:
                for child in children:
                    try:
                        dpg.delete_item(child)
                    except Exception:
                        pass  # Ignore errors deleting individual children
        except Exception:
            pass  # Ignore errors getting children

        # Create controls within the container
        parent = self.shape_params_container_tag

        if self.current_shape_type == "Box":
            dpg.add_text("Size (meters):", parent=parent)
            with dpg.group(horizontal=True, parent=parent):
                dpg.add_input_float(
                    label="X",
                    default_value=self.shape_params["size_x"],
                    format="%.2e",
                    width=150,
                    tag="size_x_input",
                )
                dpg.add_input_float(
                    label="Y",
                    default_value=self.shape_params["size_y"],
                    format="%.2e",
                    width=150,
                    tag="size_y_input",
                )
            dpg.add_input_float(
                label="Z",
                default_value=self.shape_params["size_z"],
                format="%.2e",
                width=150,
                tag="size_z_input",
                parent=parent,
            )

        elif self.current_shape_type == "Sphere":
            dpg.add_text("Radius (meters):", parent=parent)
            dpg.add_input_float(
                label="Radius",
                default_value=self.shape_params["radius"],
                format="%.2e",
                width=200,
                tag="radius_input",
                parent=parent,
            )

        elif self.current_shape_type == "Cylinder":
            dpg.add_text("Dimensions (meters):", parent=parent)
            dpg.add_input_float(
                label="Radius",
                default_value=self.shape_params["radius"],
                format="%.2e",
                width=200,
                tag="radius_input",
                parent=parent,
            )
            dpg.add_input_float(
                label="Height",
                default_value=self.shape_params["height"],
                format="%.2e",
                width=200,
                tag="height_input",
                parent=parent,
            )
            dpg.add_text("Axis:", parent=parent)
            dpg.add_radio_button(
                ["x", "y", "z"],
                default_value=2,
                callback=self._on_axis_changed,
                tag="axis_radio",
                parent=parent,
            )

    def _on_shape_type_changed(self, sender, app_data) -> None:
        """Handle shape type change."""
        shape_types_map = {
            0: "Box",
            1: "Sphere", 
            2: "Cylinder"
        }
        self.current_shape_type = shape_types_map.get(app_data, "Box")
        self._create_shape_specific_controls()

    def _on_axis_changed(self, sender, app_data) -> None:
        """Handle cylinder axis change."""
        axes = ["x", "y", "z"]
        self.shape_params["axis"] = axes[app_data]

    def _on_material_selected(self, sender, app_data) -> None:
        """Handle material selection."""
        if app_data != "Custom":
            try:
                material = get_material(app_data)
                # Update epsilon_r display if material has a simple epsilon
                # (for dispersive materials, we'd need frequency, but for now use approximate)
                if hasattr(material, "epsilon_inf"):
                    self.shape_params["epsilon_r"] = material.epsilon_inf
                if dpg.does_item_exist("epsilon_r_input"):
                    dpg.set_value("epsilon_r_input", self.shape_params["epsilon_r"])
            except Exception:
                pass

    def _on_add_shape(self) -> None:
        """Create and add the shape based on dialog inputs."""
        try:
            # Get position
            center_x = dpg.get_value("center_x_input")
            center_y = dpg.get_value("center_y_input")
            center_z = dpg.get_value("center_z_input")
            center = (center_x, center_y, center_z)

            # Get material
            material_name = dpg.get_value("material_combo")
            if material_name == "Custom":
                epsilon_r = dpg.get_value("epsilon_r_input")
                material = Material(name="Custom", epsilon_r=epsilon_r)
            else:
                try:
                    # Try to get from library
                    lib_material = get_material(material_name)
                    # Convert to simple Material for geometry
                    if hasattr(lib_material, "epsilon_inf"):
                        epsilon_r = lib_material.epsilon_inf
                    else:
                        epsilon_r = 1.0
                    material = Material(name=material_name, epsilon_r=epsilon_r)
                except Exception:
                    epsilon_r = dpg.get_value("epsilon_r_input")
                    material = Material(name=material_name, epsilon_r=epsilon_r)

            # Create shape based on type
            if self.current_shape_type == "Box":
                size_x = dpg.get_value("size_x_input")
                size_y = dpg.get_value("size_y_input")
                size_z = dpg.get_value("size_z_input")
                shape = Box(material=material, center=center, size=(size_x, size_y, size_z))

            elif self.current_shape_type == "Sphere":
                radius = dpg.get_value("radius_input")
                shape = Sphere(material=material, center=center, radius=radius)

            elif self.current_shape_type == "Cylinder":
                radius = dpg.get_value("radius_input")
                height = dpg.get_value("height_input")
                axis = self.shape_params["axis"]
                shape = Cylinder(
                    material=material, center=center, radius=radius, height=height, axis=axis
                )
            else:
                raise ValueError(f"Unknown shape type: {self.current_shape_type}")

            # Call callback if provided
            if self.on_shape_created is not None:
                self.on_shape_created(shape)

            # Close dialog
            dpg.delete_item(self.dialog_tag)

        except Exception as e:
            # Show error message
            error_msg = f"Error creating shape: {str(e)}"
            if dpg.does_item_exist("shape_error_text"):
                dpg.set_value("shape_error_text", error_msg)
            else:
                dpg.add_text(error_msg, color=(255, 0, 0), tag="shape_error_text", parent=self.dialog_tag)

