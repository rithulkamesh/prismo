"""
Field monitors and data collection.

This module provides tools for monitoring and extracting data
from FDTD simulations:
- Field probes and samplers
- Power flux monitors
- Mode monitors for waveguides
- Near-to-far field transforms
"""

from .base import Monitor
from .field import FieldMonitor

# Monitor implementations:
# - FieldMonitor: Spatial and temporal field sampling ✓
# - FluxMonitor: Power flux calculation through surfaces (to be implemented)
# - ModeMonitor: Waveguide mode amplitude extraction (to be implemented)
# - NearToFarTransform: Far-field radiation pattern calculation (to be implemented)
# - ReflectionTransmission: S-parameter calculation (to be implemented)

__all__ = [
    "Monitor",
    "FieldMonitor",
]
