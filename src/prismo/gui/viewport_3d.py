"""
3D viewport using PyVista for geometry and material visualization.

This module provides a Viewport3D class that wraps PyVista functionality
for visualizing simulation geometry, materials, and field data.
"""

from typing import Optional, Union

import numpy as np

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None

from prismo.geometry.shapes import (
    Box,
    Cylinder,
    GeometryGroup,
    Material,
    Polygon,
    Shape,
    Sphere,
)


class Viewport3D:
    """
    3D viewport for visualizing simulation geometry and materials.

    This class wraps PyVista functionality to provide interactive 3D
    visualization of geometric shapes, material distributions, and field data.
    """

    def __init__(
        self,
        window_size: tuple[int, int] = (800, 600),
        title: str = "Prismo 3D Viewport",
        show_axes: bool = True,
        background: str = "white",
    ):
        """
        Initialize the 3D viewport.

        Parameters
        ----------
        window_size : tuple[int, int]
            Window size (width, height) in pixels.
        title : str
            Window title.
        show_axes : bool
            Whether to show coordinate axes.
        background : str
            Background color ('white', 'black', or color name).
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError(
                "PyVista is required for 3D visualization. Install with: pip install pyvista"
            )

        # Create plotter
        self.plotter = pv.Plotter(window_size=window_size, title=title)
        self.plotter.background_color = background

        # Show axes if requested
        if show_axes:
            self.plotter.show_axes()

        # Store meshes and actors for updates
        self.meshes: dict[str, pv.PolyData] = {}
        self.actors: dict[str, pv.Actor] = {}
        self.slice_planes: dict[str, dict] = {}  # Store slice plane info
        self.slice_plane_positions: dict[str, float] = {"xy": 0.0, "xz": 0.0, "yz": 0.0}
        self.slice_plane_enabled: dict[str, bool] = {"xy": False, "xz": False, "yz": False}

        # Material color mapping
        self.material_colors: dict[str, tuple[float, float, float]] = {}
        self._color_index = 0

        # Default color palette (distinct colors)
        self._color_palette = [
            (0.8, 0.2, 0.2),  # Red
            (0.2, 0.8, 0.2),  # Green
            (0.2, 0.2, 0.8),  # Blue
            (0.8, 0.8, 0.2),  # Yellow
            (0.8, 0.2, 0.8),  # Magenta
            (0.2, 0.8, 0.8),  # Cyan
            (0.9, 0.5, 0.1),  # Orange
            (0.5, 0.1, 0.9),  # Purple
        ]

    def _get_material_color(self, material: Material) -> tuple[float, float, float]:
        """
        Get color for a material.

        Parameters
        ----------
        material : Material
            Material to get color for.

        Returns
        -------
        tuple[float, float, float]
            RGB color tuple (0-1 range).
        """
        # Use cached color if available
        if material.name in self.material_colors:
            return self.material_colors[material.name]

        # Assign new color based on permittivity or palette
        if isinstance(material.epsilon_r, (int, float)):
            # Use a colormap based on epsilon_r (normalized to 0-1)
            # Assuming typical range 1-20 for epsilon_r
            normalized_eps = min(max((material.epsilon_r - 1) / 19, 0), 1)
            # Use a colormap (blue to red)
            color = (normalized_eps, 0.3, 1 - normalized_eps)
        else:
            # Use palette color
            color = self._color_palette[self._color_index % len(self._color_palette)]
            self._color_index += 1

        # Cache the color
        self.material_colors[material.name] = color
        return color

    def add_shape(self, shape: Shape, name: Optional[str] = None) -> None:
        """
        Add a geometric shape to the viewport.

        Parameters
        ----------
        shape : Shape
            Shape to add (Box, Sphere, Cylinder, etc.).
        name : str, optional
            Name for the shape (auto-generated if not provided).
        """
        if name is None:
            name = f"shape_{len(self.meshes)}"

        # Convert shape to PyVista mesh
        mesh = self._shape_to_mesh(shape)
        if mesh is None:
            return

        # Store mesh
        self.meshes[name] = mesh

        # Get material color
        color = self._get_material_color(shape.material)

        # Add to plotter
        actor = self.plotter.add_mesh(mesh, color=color, opacity=0.8, name=name)
        self.actors[name] = actor

    def _shape_to_mesh(self, shape: Shape) -> Optional[pv.PolyData]:
        """
        Convert a Shape to a PyVista mesh.

        Parameters
        ----------
        shape : Shape
            Shape to convert.

        Returns
        -------
        pv.PolyData or None
            PyVista mesh, or None if conversion fails.
        """
        if isinstance(shape, Box):
            return self._box_to_mesh(shape)
        elif isinstance(shape, Sphere):
            return self._sphere_to_mesh(shape)
        elif isinstance(shape, Cylinder):
            return self._cylinder_to_mesh(shape)
        elif isinstance(shape, Polygon):
            return self._polygon_to_mesh(shape)
        elif isinstance(shape, GeometryGroup):
            return self._geometry_group_to_mesh(shape)
        else:
            # For other shapes, use rasterization approach
            return self._rasterize_shape(shape)

    def _box_to_mesh(self, box: Box) -> pv.PolyData:
        """Convert Box to PyVista mesh."""
        center = box.center
        size = box.size

        # Create box mesh
        mesh = pv.Box(
            bounds=(
                center[0] - size[0] / 2,
                center[0] + size[0] / 2,
                center[1] - size[1] / 2,
                center[1] + size[1] / 2,
                center[2] - size[2] / 2,
                center[2] + size[2] / 2,
            )
        )
        return mesh

    def _sphere_to_mesh(self, sphere: Sphere) -> pv.PolyData:
        """Convert Sphere to PyVista mesh."""
        center = sphere.center
        radius = sphere.radius

        # Create sphere mesh
        mesh = pv.Sphere(radius=radius, center=center, resolution=20)
        return mesh

    def _cylinder_to_mesh(self, cylinder: Cylinder) -> pv.PolyData:
        """Convert Cylinder to PyVista mesh."""
        center = cylinder.center
        radius = cylinder.radius
        height = cylinder.height
        axis = cylinder.axis

        # Create cylinder aligned with axis
        if axis == "z":
            direction = (0, 0, 1)
        elif axis == "y":
            direction = (0, 1, 0)
        else:  # axis == "x"
            direction = (1, 0, 0)

        # Create cylinder
        mesh = pv.Cylinder(
            center=center,
            direction=direction,
            radius=radius,
            height=height,
            resolution=20,
        )
        return mesh

    def _polygon_to_mesh(self, polygon: Polygon) -> Optional[pv.PolyData]:
        """Convert Polygon to PyVista mesh by extruding the 2D polygon."""
        vertices = polygon.vertices
        z_min = polygon.z_min
        z_max = polygon.z_max

        # Check if z bounds are reasonable
        if z_min == -np.inf or z_max == np.inf:
            # Use a default height
            z_min = polygon.center[2] - 0.5
            z_max = polygon.center[2] + 0.5

        # Create 2D polygon face
        # Close the polygon if not already closed
        if len(vertices) > 0 and not np.allclose(vertices[0], vertices[-1]):
            vertices_closed = np.vstack([vertices, vertices[0:1]])
        else:
            vertices_closed = vertices

        # Create polygon face using PyVista's proper API
        try:
            # Create 2D polygon by creating a face with vertices
            # Add z=0 for 2D vertices
            vertices_2d = vertices_closed[:, :2]
            n_verts = len(vertices_2d)
            
            # Create face array: [n_verts, v0, v1, v2, ..., vn-1]
            face = np.concatenate([[n_verts], np.arange(n_verts)])
            
            # Create 3D points (z=0 for base)
            points_3d = np.column_stack([vertices_2d, np.zeros(n_verts)])
            
            # Create PolyData with face
            polygon_2d = pv.PolyData(points_3d, faces=face)
            
            # Extrude in z direction
            height = z_max - z_min
            center_z = (z_min + z_max) / 2
            
            # Extrude along z-axis
            mesh = polygon_2d.extrude((0, 0, height))
            
            # Translate to correct z position
            mesh.translate((0, 0, center_z - height / 2))
            
            return mesh
        except Exception:
            # Fallback to rasterization
            return self._rasterize_shape(polygon)

    def _geometry_group_to_mesh(self, group: GeometryGroup) -> Optional[pv.PolyData]:
        """
        Convert GeometryGroup to PyVista mesh.

        For complex boolean operations, we rasterize the geometry.
        """
        # For now, just add individual shapes
        # In the future, could use PyVista's boolean operations
        if len(group.shapes) == 0:
            return None

        # Add first shape
        mesh = self._shape_to_mesh(group.shapes[0])
        if mesh is None:
            return None

        # For union, we can combine meshes
        # For intersection/difference, rasterization is more reliable
        if group.operation == "union" and len(group.shapes) > 1:
            for shape in group.shapes[1:]:
                shape_mesh = self._shape_to_mesh(shape)
                if shape_mesh is not None:
                    # Combine meshes (simple approach)
                    mesh = mesh + shape_mesh

        return mesh

    def _rasterize_shape(self, shape: Shape) -> Optional[pv.PolyData]:
        """
        Rasterize a shape to create a mesh.

        This is a fallback for shapes that don't have direct PyVista equivalents.
        """
        # Estimate bounds from center (assume reasonable size)
        center = shape.center
        size_estimate = 1.0  # Default size estimate

        # Create a grid around the shape
        n_points = 20  # Resolution for rasterization
        x = np.linspace(center[0] - size_estimate, center[0] + size_estimate, n_points)
        y = np.linspace(center[1] - size_estimate, center[1] + size_estimate, n_points)
        z = np.linspace(center[2] - size_estimate, center[2] + size_estimate, n_points)

        # Rasterize
        mask = shape.rasterize(x, y, z)

        # Convert to mesh using marching cubes
        if np.any(mask):
            # Create structured grid
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            grid = pv.StructuredGrid(X, Y, Z)
            grid["mask"] = mask.ravel(order="F")

            # Extract surface using contour
            try:
                mesh = grid.contour([0.5], scalars="mask")
                return mesh
            except Exception:
                # If contour fails, return None
                return None

        return None

    def add_slice_plane(
        self,
        plane: str,
        position: Optional[float] = None,
        normal: Optional[tuple[float, float, float]] = None,
        origin: Optional[tuple[float, float, float]] = None,
        size: float = 10.0,
        enabled: bool = True,
    ) -> None:
        """
        Add or update a slice plane to the viewport.

        Parameters
        ----------
        plane : str
            Plane orientation ('xy', 'xz', 'yz').
        position : float, optional
            Position along the normal axis.
        normal : tuple[float, float, float], optional
            Normal vector for the plane.
        origin : tuple[float, float, float], optional
            Origin point for the plane.
        size : float, optional
            Size of the slice plane, default=10.0.
        enabled : bool, optional
            Whether the slice plane is enabled, default=True.
        """
        plane = plane.lower()
        if plane not in ["xy", "xz", "yz"]:
            raise ValueError(f"Unknown plane: {plane}. Use 'xy', 'xz', or 'yz'")

        # Update position
        if position is not None:
            self.slice_plane_positions[plane] = position

        # Get current position
        current_pos = self.slice_plane_positions.get(plane, 0.0)
        
        # Determine normal and origin from plane string
        if plane == "xy":
            if normal is None:
                normal = (0, 0, 1)
            if origin is None:
                origin = (0, 0, current_pos)
        elif plane == "xz":
            if normal is None:
                normal = (0, 1, 0)
            if origin is None:
                origin = (0, current_pos, 0)
        elif plane == "yz":
            if normal is None:
                normal = (1, 0, 0)
            if origin is None:
                origin = (current_pos, 0, 0)

        # Remove existing slice plane if present
        if plane in self.slice_planes:
            self.remove_slice_plane(plane)

        # Store enabled state
        self.slice_plane_enabled[plane] = enabled

        if not enabled:
            return

        # Create plane with larger size for visibility
        plane_obj = pv.Plane(center=origin, direction=normal, i_size=size, j_size=size)

        # Add to plotter
        actor = self.plotter.add_mesh(
            plane_obj, color="cyan", opacity=0.4, name=f"slice_{plane}", show_edges=True
        )
        self.slice_planes[plane] = {
            "plane": plane_obj,
            "normal": normal,
            "origin": origin,
            "size": size,
            "enabled": enabled,
        }
        self.actors[f"slice_{plane}"] = actor

    def update_slice_plane_position(self, plane: str, position: float) -> None:
        """
        Update the position of an existing slice plane.

        Parameters
        ----------
        plane : str
            Plane orientation ('xy', 'xz', 'yz').
        position : float
            New position along the normal axis.
        """
        plane = plane.lower()
        if plane not in self.slice_planes:
            # Create new slice plane if it doesn't exist
            self.add_slice_plane(plane, position=position)
            return

        # Update position
        self.slice_plane_positions[plane] = position
        enabled = self.slice_plane_enabled.get(plane, False)

        if not enabled:
            return

        # Remove old plane
        self.remove_slice_plane(plane)

        # Re-add with new position
        self.add_slice_plane(plane, position=position, enabled=enabled)

    def remove_slice_plane(self, plane: str) -> None:
        """
        Remove a slice plane from the viewport.

        Parameters
        ----------
        plane : str
            Plane to remove ('xy', 'xz', 'yz').
        """
        plane = plane.lower()
        actor_name = f"slice_{plane}"
        if actor_name in self.actors:
            self.plotter.remove_actor(self.actors[actor_name])
            del self.actors[actor_name]
        if plane in self.slice_planes:
            del self.slice_planes[plane]
        self.slice_plane_enabled[plane] = False

    def clear(self) -> None:
        """Clear all meshes and actors from the viewport."""
        self.plotter.clear()
        self.meshes.clear()
        self.actors.clear()
        self.slice_planes.clear()
        self.slice_plane_positions = {"xy": 0.0, "xz": 0.0, "yz": 0.0}
        self.slice_plane_enabled = {"xy": False, "xz": False, "yz": False}

    def reset_camera(self) -> None:
        """Reset the camera to default view."""
        self.plotter.reset_camera()

    def set_camera_position(
        self,
        position: tuple[float, float, float],
        focal_point: Optional[tuple[float, float, float]] = None,
        view_up: Optional[tuple[float, float, float]] = None,
    ) -> None:
        """
        Set camera position and orientation.

        Parameters
        ----------
        position : tuple[float, float, float]
            Camera position.
        focal_point : tuple[float, float, float], optional
            Focal point (what the camera looks at).
        view_up : tuple[float, float, float], optional
            Up vector for the camera.
        """
        self.plotter.camera_position = position
        if focal_point is not None:
            self.plotter.camera.focal_point = focal_point
        if view_up is not None:
            self.plotter.camera.up = view_up

    def show(self, interactive: bool = True) -> None:
        """
        Show the viewport.

        Parameters
        ----------
        interactive : bool
            Whether to show interactively (blocking) or return immediately.
        """
        if interactive:
            self.plotter.show()
        else:
            self.plotter.show(auto_close=False)

    def update(self) -> None:
        """Update the viewport (for non-interactive mode)."""
        # Only update if the render window has been initialized
        try:
            self.plotter.update()
        except (RuntimeError, AttributeError):
            # Render window not initialized yet, skip update
            pass

    def close(self) -> None:
        """Close the viewport."""
        self.plotter.close()

    def sync_with_simulation(self, simulation) -> None:
        """
        Synchronize viewport with a Simulation object.

        Adds all shapes, monitors, and sources from the simulation.

        Parameters
        ----------
        simulation : Simulation
            The simulation object to sync with.
        """
        # Clear existing geometry
        self.clear()

        # Add all shapes
        for shape in simulation.shapes:
            self.add_shape(shape)

        # Add monitors as wireframe boxes
        for monitor in simulation.monitors:
            self.add_monitor_visualization(monitor)

        # Add sources as markers/icons
        for source in simulation.sources:
            self.add_source_visualization(source)

        # Reset camera to show all geometry
        self.reset_camera()

    def add_monitor_visualization(self, monitor) -> None:
        """
        Add a monitor as a wireframe box visualization.

        Parameters
        ----------
        monitor : Monitor
            The monitor to visualize.
        """
        center = monitor.center
        size = monitor.size

        # Create box bounds
        bounds = (
            center[0] - size[0] / 2,
            center[0] + size[0] / 2,
            center[1] - size[1] / 2,
            center[1] + size[1] / 2,
            center[2] - size[2] / 2,
            center[2] + size[2] / 2,
        )

        # Create wireframe box
        box = pv.Box(bounds)
        outline = box.outline()

        # Add as wireframe
        monitor_name = f"monitor_{monitor.name}"
        actor = self.plotter.add_mesh(
            outline,
            color="yellow",
            line_width=2,
            opacity=0.8,
            name=monitor_name,
            style="wireframe",
        )
        self.actors[monitor_name] = actor

    def add_source_visualization(self, source) -> None:
        """
        Add a source as a marker/icon visualization.

        Parameters
        ----------
        source : Source
            The source to visualize.
        """
        # For point sources, show as a sphere
        # For extended sources, show as a box outline
        center = getattr(source, "center", (0, 0, 0))
        size = getattr(source, "size", (0, 0, 0))

        source_name = f"source_{getattr(source, 'name', id(source))}"

        # Check if it's a point source
        if all(s == 0 or s is None for s in size):
            # Point source - use sphere marker
            sphere = pv.Sphere(radius=0.1, center=center, resolution=10)
            actor = self.plotter.add_mesh(
                sphere, color="red", opacity=0.9, name=source_name
            )
        else:
            # Extended source - use box outline
            bounds = (
                center[0] - size[0] / 2,
                center[0] + size[0] / 2,
                center[1] - size[1] / 2,
                center[1] + size[1] / 2,
                center[2] - size[2] / 2,
                center[2] + size[2] / 2,
            )
            box = pv.Box(bounds)
            outline = box.outline()
            actor = self.plotter.add_mesh(
                outline,
                color="green",
                line_width=3,
                opacity=0.9,
                name=source_name,
                style="wireframe",
            )

        self.actors[source_name] = actor

    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, "plotter"):
            try:
                self.plotter.close()
            except Exception:
                pass

