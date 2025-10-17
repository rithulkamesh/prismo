"""
Geometric shapes for defining structures in FDTD simulations.

This module provides classes for common geometric shapes and their
rasterization onto the FDTD grid.
"""

from typing import Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class Material:
    """
    Simple material definition for geometry.

    Attributes
    ----------
    name : str
        Material name.
    epsilon_r : float or array
        Relative permittivity (can be tensor for anisotropic).
    mu_r : float or array, optional
        Relative permeability, default=1.0.
    """

    name: str
    epsilon_r: float = 1.0
    mu_r: float = 1.0


class Shape(ABC):
    """
    Abstract base class for geometric shapes.

    All shapes must implement the `contains()` method to determine
    if a point is inside the shape.

    Parameters
    ----------
    material : Material
        Material filling the shape.
    center : Tuple[float, float, float]
        Shape center coordinates.
    """

    def __init__(self, material: Material, center: Tuple[float, float, float]):
        self.material = material
        self.center = np.array(center)

    @abstractmethod
    def contains(self, points: np.ndarray) -> np.ndarray:
        """
        Check if points are inside the shape.

        Parameters
        ----------
        points : ndarray
            Array of points with shape (N, 3) for 3D or (N, 2) for 2D.

        Returns
        -------
        ndarray
            Boolean array of shape (N,) indicating if each point is inside.
        """
        pass

    def rasterize(
        self, x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Rasterize shape onto a grid.

        Parameters
        ----------
        x, y, z : ndarray
            1D coordinate arrays for grid points.

        Returns
        -------
        ndarray
            Boolean mask indicating where material is present.
        """
        if z is None:
            # 2D case
            X, Y = np.meshgrid(x, y, indexing="ij")
            points = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
            mask = self.contains(points).reshape(X.shape)
        else:
            # 3D case
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
            mask = self.contains(points).reshape(X.shape)

        return mask


class Box(Shape):
    """
    Rectangular box/cuboid shape.

    Parameters
    ----------
    material : Material
        Material filling the box.
    center : Tuple[float, float, float]
        Box center coordinates.
    size : Tuple[float, float, float]
        Box dimensions (Lx, Ly, Lz). For 2D, set Lz=0.
    """

    def __init__(
        self,
        material: Material,
        center: Tuple[float, float, float],
        size: Tuple[float, float, float],
    ):
        super().__init__(material, center)
        self.size = np.array(size)
        self.half_size = self.size / 2.0

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Check if points are inside the box."""
        # Distance from center
        dist = np.abs(points - self.center)

        # Inside if distance < half_size in all dimensions
        inside = np.all(dist <= self.half_size, axis=1)
        return inside


class Sphere(Shape):
    """
    Spherical shape.

    Parameters
    ----------
    material : Material
        Material filling the sphere.
    center : Tuple[float, float, float]
        Sphere center coordinates.
    radius : float
        Sphere radius.
    """

    def __init__(
        self, material: Material, center: Tuple[float, float, float], radius: float
    ):
        super().__init__(material, center)
        self.radius = radius

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Check if points are inside the sphere."""
        # Distance from center
        dist = np.linalg.norm(points - self.center, axis=1)
        return dist <= self.radius


class Cylinder(Shape):
    """
    Cylindrical shape.

    Parameters
    ----------
    material : Material
        Material filling the cylinder.
    center : Tuple[float, float, float]
        Cylinder center coordinates.
    radius : float
        Cylinder radius.
    height : float
        Cylinder height.
    axis : str
        Cylinder axis ('x', 'y', or 'z'), default='z'.
    """

    def __init__(
        self,
        material: Material,
        center: Tuple[float, float, float],
        radius: float,
        height: float,
        axis: str = "z",
    ):
        super().__init__(material, center)
        self.radius = radius
        self.height = height
        self.axis = axis.lower()

        if self.axis not in ["x", "y", "z"]:
            raise ValueError("axis must be 'x', 'y', or 'z'")

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Check if points are inside the cylinder."""
        # Get axis indices
        axis_map = {"x": 0, "y": 1, "z": 2}
        ax_idx = axis_map[self.axis]

        # Perpendicular indices
        perp_idx = [i for i in range(3) if i != ax_idx]

        # Distance along axis
        axial_dist = np.abs(points[:, ax_idx] - self.center[ax_idx])

        # Radial distance
        radial_dist = np.sqrt(
            (points[:, perp_idx[0]] - self.center[perp_idx[0]]) ** 2
            + (points[:, perp_idx[1]] - self.center[perp_idx[1]]) ** 2
        )

        # Inside if within radius and height
        return (radial_dist <= self.radius) & (axial_dist <= self.height / 2)


class Polygon(Shape):
    """
    Polygonal shape (2D, extruded in z).

    Parameters
    ----------
    material : Material
        Material filling the polygon.
    vertices : array-like
        Polygon vertices as (N, 2) array of (x, y) coordinates.
    z_min, z_max : float
        Z-extent for extrusion.
    """

    def __init__(
        self,
        material: Material,
        vertices: np.ndarray,
        z_min: float = -np.inf,
        z_max: float = np.inf,
    ):
        vertices = np.asarray(vertices)
        center = np.mean(vertices, axis=0)
        center_3d = np.array([center[0], center[1], (z_min + z_max) / 2])

        super().__init__(material, center_3d)
        self.vertices = vertices
        self.z_min = z_min
        self.z_max = z_max

    def contains(self, points: np.ndarray) -> np.ndarray:
        """
        Check if points are inside the polygon.

        Uses ray casting algorithm for 2D point-in-polygon test.
        """
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2] if points.shape[1] > 2 else np.zeros_like(x)

        # Check z bounds
        z_inside = (z >= self.z_min) & (z <= self.z_max)

        # Ray casting for 2D polygon
        n = len(self.vertices)
        inside = np.zeros(len(points), dtype=bool)

        for i in range(len(points)):
            if not z_inside[i]:
                continue

            xi, yi = x[i], y[i]
            count = 0

            for j in range(n):
                v1 = self.vertices[j]
                v2 = self.vertices[(j + 1) % n]

                if ((v1[1] > yi) != (v2[1] > yi)) and (
                    xi < (v2[0] - v1[0]) * (yi - v1[1]) / (v2[1] - v1[1]) + v1[0]
                ):
                    count += 1

            inside[i] = count % 2 == 1

        return inside & z_inside


class CustomShape(Shape):
    """
    Custom shape defined by a function.

    Parameters
    ----------
    material : Material
        Material filling the shape.
    center : Tuple[float, float, float]
        Approximate shape center.
    function : callable
        Function that takes points array and returns boolean mask.
        Signature: function(points: ndarray) -> ndarray
    """

    def __init__(
        self,
        material: Material,
        center: Tuple[float, float, float],
        function: Callable[[np.ndarray], np.ndarray],
    ):
        super().__init__(material, center)
        self.function = function

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Check if points are inside using custom function."""
        return self.function(points)


class GeometryGroup:
    """
    Group of shapes with boolean operations.

    Supports union, intersection, and difference operations.

    Parameters
    ----------
    shapes : list of Shape
        List of shapes to combine.
    operation : str
        Boolean operation: 'union', 'intersection', 'difference'.
    """

    def __init__(self, shapes: list, operation: str = "union"):
        self.shapes = shapes
        self.operation = operation.lower()

        if self.operation not in ["union", "intersection", "difference"]:
            raise ValueError(
                "operation must be 'union', 'intersection', or 'difference'"
            )

    def rasterize(
        self, x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, list]:
        """
        Rasterize combined geometry.

        Returns
        -------
        Tuple[ndarray, list]
            Combined mask and list of material indices for each shape.
        """
        if len(self.shapes) == 0:
            shape = (len(x), len(y), len(z) if z is not None else 1)
            return np.zeros(shape, dtype=bool), []

        # Rasterize first shape
        masks = [self.shapes[0].rasterize(x, y, z)]

        # Rasterize remaining shapes
        for shape in self.shapes[1:]:
            masks.append(shape.rasterize(x, y, z))

        # Combine with boolean operation
        if self.operation == "union":
            combined = masks[0]
            for mask in masks[1:]:
                combined = combined | mask

        elif self.operation == "intersection":
            combined = masks[0]
            for mask in masks[1:]:
                combined = combined & mask

        elif self.operation == "difference":
            combined = masks[0]
            for mask in masks[1:]:
                combined = combined & ~mask

        return combined, masks
