"""
Mode solver and analysis for waveguide structures.

This module provides eigenmode calculation for waveguide structures,
mode sources, and mode expansion monitors.
"""

from .solver import ModeSolver, WaveguideMode

__all__ = ["ModeSolver", "WaveguideMode"]
