"""
Geometry module for defining structures in FDTD simulations.

This module provides geometric shapes and tools for defining material
distributions in the simulation domain.
"""

from .shapes import (
    Box,
    CustomShape,
    Cylinder,
    GeometryGroup,
    Material,
    Polygon,
    Shape,
    Sphere,
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
