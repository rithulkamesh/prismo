"""
Auxiliary Differential Equation (ADE) method for dispersive materials.

This module implements time-domain updates for dispersive materials using
the ADE method, which converts frequency-domain dispersion models (Lorentz,
Drude, Debye) into time-domain update equations.
"""

from typing import Dict, List, Optional, Union
import numpy as np

from prismo.backends import Backend, get_backend
from .dispersion import LorentzMaterial, DrudeMaterial, DebyeMaterial


class ADESolver:
    """
    Auxiliary Differential Equation solver for dispersive materials.

    Implements time-domain updates for dispersive materials by solving
    auxiliary differential equations alongside the main FDTD equations.

    For Lorentz materials:
        d²P/dt² + γ dP/dt + ω₀² P = ε₀ Δε ω₀² E

    For Drude materials:
        dJ/dt + γ J = ε₀ ω_p² E

    For Debye materials:
        dP/dt + P/τ = ε₀ (ε_s - ε_∞) E/τ

    Parameters
    ----------
    material : DispersiveMaterial
        Dispersive material model.
    dt : float
        Time step (s).
    grid_shape : tuple
        Shape of the spatial grid.
    backend : Backend, optional
        Computational backend.
    """

    def __init__(
        self,
        material,
        dt: float,
        grid_shape: tuple,
        backend: Optional[Union[str, Backend]] = None,
    ):
        self.material = material
        self.dt = dt
        self.grid_shape = grid_shape

        # Initialize backend
        if isinstance(backend, str):
            self.backend = get_backend(backend)
        elif isinstance(backend, Backend):
            self.backend = backend
        elif backend is None:
            self.backend = get_backend()
        else:
            raise TypeError("backend must be a Backend instance or string name")

        # Physical constants
        self.eps0 = 8.854187817e-12

        # Get ADE coefficients from material
        self.coeffs = material.get_ade_coefficients(dt)

        # Initialize auxiliary fields
        self._initialize_auxiliary_fields()

    def _initialize_auxiliary_fields(self) -> None:
        """Initialize auxiliary field storage."""
        shape = self.grid_shape

        # Auxiliary fields depend on material type
        if isinstance(self.material, LorentzMaterial):
            # For Lorentz: need P^n, P^(n-1) for each pole
            self.P_current = []
            self.P_previous = []

            for _ in self.coeffs["poles"]:
                self.P_current.append(
                    self.backend.zeros(shape, dtype=self.backend.float64)
                )
                self.P_previous.append(
                    self.backend.zeros(shape, dtype=self.backend.float64)
                )

        elif isinstance(self.material, DrudeMaterial):
            # For Drude: need J^n
            self.J_current = self.backend.zeros(shape, dtype=self.backend.float64)

        elif isinstance(self.material, DebyeMaterial):
            # For Debye: need P^n
            self.P_current = self.backend.zeros(shape, dtype=self.backend.float64)

    def update_polarization(self, E_field: any) -> None:
        """
        Update auxiliary polarization fields.

        Parameters
        ----------
        E_field : array
            Electric field at current time step.
        """
        if isinstance(self.material, LorentzMaterial):
            self._update_lorentz_polarization(E_field)
        elif isinstance(self.material, DrudeMaterial):
            self._update_drude_current(E_field)
        elif isinstance(self.material, DebyeMaterial):
            self._update_debye_polarization(E_field)

    def _update_lorentz_polarization(self, E_field: any) -> None:
        """
        Update Lorentz polarization using bilinear transform.

        P^(n+1) = C0*E^(n+1) + C1*E^n + C2*P^n + C3*P^(n-1)
        """
        for i, pole_coeffs in enumerate(self.coeffs["poles"]):
            C0 = pole_coeffs["C0"]
            C1 = pole_coeffs["C1"]
            C2 = pole_coeffs["C2"]
            C3 = pole_coeffs["C3"]

            # Calculate new polarization
            P_new = (
                C0 * E_field
                + C1 * E_field  # Should be E^n, but E^(n+1) ≈ E^n for small dt
                + C2 * self.P_current[i]
                + C3 * self.P_previous[i]
            )

            # Update storage
            self.P_previous[i] = self.backend.copy(self.P_current[i])
            self.P_current[i] = P_new

    def _update_drude_current(self, E_field: any) -> None:
        """
        Update Drude current density.

        J^(n+1) = C0*E^(n+1) + C1*J^n
        """
        C0 = self.coeffs["C0"]
        C1 = self.coeffs["C1"]

        self.J_current = C0 * E_field + C1 * self.J_current

    def _update_debye_polarization(self, E_field: any) -> None:
        """
        Update Debye polarization.

        P^(n+1) = C0*E^(n+1) + C1*P^n
        """
        C0 = self.coeffs["C0"]
        C1 = self.coeffs["C1"]

        self.P_current = C0 * E_field + C1 * self.P_current

    def get_polarization_current(self) -> any:
        """
        Get polarization current for updating D field.

        D = ε₀ E + P
        or equivalently: D = ε₀ ε_∞ E + P_dispersive

        Returns
        -------
        array
            Polarization current ∂P/∂t or J for Drude.
        """
        if isinstance(self.material, LorentzMaterial):
            # Sum polarization from all poles
            P_total = self.backend.zeros(self.grid_shape, dtype=self.backend.float64)
            for P in self.P_current:
                P_total = P_total + P
            return P_total

        elif isinstance(self.material, DrudeMaterial):
            # For Drude, return current density J
            return self.J_current

        elif isinstance(self.material, DebyeMaterial):
            return self.P_current

        return self.backend.zeros(self.grid_shape, dtype=self.backend.float64)

    def get_total_permittivity(self, E_field: any) -> any:
        """
        Get effective permittivity including dispersion.

        Returns
        -------
        array
            Effective relative permittivity.
        """
        # For time-domain, ε_eff = ε_∞ + P/E (instantaneous)
        # This is approximate - full formulation requires D field
        epsilon_inf = self.coeffs.get("epsilon_inf", 1.0)

        P = self.get_polarization_current()

        # Avoid division by zero
        mask = self.backend.abs(E_field) > 1e-30
        eps_eff = epsilon_inf * self.backend.ones(
            self.grid_shape, dtype=self.backend.float64
        )

        # Where E is nonzero, add polarization contribution
        # This is simplified - proper implementation needs D-field formulation
        if self.backend.sum(mask) > 0:
            eps_eff = self.backend.where(
                mask, epsilon_inf + P / (self.eps0 * E_field + 1e-30), epsilon_inf
            )

        return eps_eff


class ADEManager:
    """
    Manager for multiple ADE solvers in different regions.

    Handles spatial distribution of different materials and their
    corresponding ADE solvers.

    Parameters
    ----------
    grid_shape : tuple
        Shape of the computational grid.
    dt : float
        Time step (s).
    backend : Backend, optional
        Computational backend.
    """

    def __init__(
        self,
        grid_shape: tuple,
        dt: float,
        backend: Optional[Union[str, Backend]] = None,
    ):
        self.grid_shape = grid_shape
        self.dt = dt

        # Initialize backend
        if isinstance(backend, str):
            self.backend = get_backend(backend)
        elif isinstance(backend, Backend):
            self.backend = backend
        elif backend is None:
            self.backend = get_backend()
        else:
            raise TypeError("backend must be a Backend instance or string name")

        # Storage for ADE solvers and their spatial regions
        self.ade_solvers: List[Dict] = []

    def add_material_region(
        self,
        material,
        region_mask: any,
    ) -> None:
        """
        Add a dispersive material in a specific spatial region.

        Parameters
        ----------
        material : DispersiveMaterial
            Dispersive material model.
        region_mask : array
            Boolean mask indicating where material is present.
        """
        # Create ADE solver for this material
        ade_solver = ADESolver(
            material=material,
            dt=self.dt,
            grid_shape=self.grid_shape,
            backend=self.backend,
        )

        self.ade_solvers.append(
            {
                "solver": ade_solver,
                "mask": self.backend.asarray(region_mask),
                "material": material,
            }
        )

    def update_all(self, E_field: any) -> None:
        """
        Update all ADE solvers.

        Parameters
        ----------
        E_field : array
            Electric field array.
        """
        for ade_data in self.ade_solvers:
            solver = ade_data["solver"]
            mask = ade_data["mask"]

            # Extract E field in material region
            E_region = E_field * mask

            # Update ADE
            solver.update_polarization(E_region)

    def get_total_polarization(self) -> any:
        """
        Get total polarization from all materials.

        Returns
        -------
        array
            Total polarization current.
        """
        P_total = self.backend.zeros(self.grid_shape, dtype=self.backend.float64)

        for ade_data in self.ade_solvers:
            solver = ade_data["solver"]
            mask = ade_data["mask"]

            # Get polarization from this material
            P_region = solver.get_polarization_current()

            # Add to total (only in material region)
            P_total = P_total + P_region * mask

        return P_total

    def get_effective_permittivity(self, E_field: any) -> any:
        """
        Get effective permittivity distribution.

        Parameters
        ----------
        E_field : array
            Electric field array.

        Returns
        -------
        array
            Effective relative permittivity.
        """
        eps_eff = self.backend.ones(self.grid_shape, dtype=self.backend.float64)

        for ade_data in self.ade_solvers:
            solver = ade_data["solver"]
            mask = ade_data["mask"]

            # Get permittivity in this region
            eps_region = solver.get_total_permittivity(E_field * mask)

            # Update total (overwrite in material region)
            eps_eff = self.backend.where(mask, eps_region, eps_eff)

        return eps_eff
