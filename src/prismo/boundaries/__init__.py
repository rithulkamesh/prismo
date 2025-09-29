"""
Boundary conditions for FDTD simulations.

This module implements various boundary conditions:
- Absorbing boundaries (PML, CPML)
- Periodic boundaries (Bloch conditions)
- Perfect electric/magnetic conductor boundaries
- Waveguide mode ports
"""

# Boundary condition implementations:
# - PML: Perfectly Matched Layer absorbing boundary
# - CPML: Convolutional PML implementation
# - PeriodicBoundary: Periodic and Bloch boundary conditions
# - PECBoundary: Perfect Electric Conductor
# - PMCBoundary: Perfect Magnetic Conductor
# - ModePort: Waveguide mode injection/extraction

__all__ = []
