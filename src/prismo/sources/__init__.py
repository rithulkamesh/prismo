"""
Source implementations for FDTD simulations.

This module provides various electromagnetic sources:
- Point sources (dipoles)
- Plane wave sources
- Gaussian beam sources
- Mode sources for waveguides
- Total-field/scattered-field (TFSF) sources
"""

from .base import Source
from .gaussian import GaussianBeamSource
from .plane_wave import PlaneWaveSource
from .point import ElectricDipole, MagneticDipole, PointSource
from .tfsf import TFSFSource
from .waveform import (
    ContinuousWave,
    CustomWaveform,
    GaussianPulse,
    RickerWavelet,
    Waveform,
)

# Source implementations:
# - PointSource: Point dipole sources ✓
# - ElectricDipole: Electric field point source ✓
# - MagneticDipole: Magnetic field point source ✓
# - PlaneWaveSource: Plane wave excitation ✓
# - GaussianBeamSource: Focused Gaussian beam sources ✓
# - TFSFSource: Total-field/scattered-field formulation ✓
# - ModeSource: Waveguide eigenmode sources (to be implemented)

__all__ = [
    "Source",
    "Waveform",
    "ContinuousWave",
    "GaussianPulse",
    "RickerWavelet",
    "CustomWaveform",
    "PointSource",
    "ElectricDipole",
    "MagneticDipole",
    "GaussianBeamSource",
    "PlaneWaveSource",
    "TFSFSource",
]
