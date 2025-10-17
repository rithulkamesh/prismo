"""
Geometry module for defining structures in FDTD simulations.

This module provides geometric shapes and tools for defining material
distributions in the simulation domain.
"""

from .shapes import (
    Shape,
    Material,
    Box,
    Sphere,
    Cylinder,
    Polygon,
    CustomShape,
    GeometryGroup,
)

__all__ = [
    "Shape",
    "Material",
    "Box",
    "Sphere",
    "Cylinder",
    "Polygon",
    "CustomShape",
    "GeometryGroup",
]
