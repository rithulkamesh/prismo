"""
Dispersive material models for frequency-dependent materials.

This module implements various dispersion models (Lorentz, Drude, Debye, Sellmeier)
for materials with frequency-dependent permittivity and permeability.
"""

from typing import List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from prismo.backends import Backend, get_backend


@dataclass
class LorentzPole:
    """
    Parameters for a single Lorentz pole.

    The Lorentz model describes resonant behavior:
    ε(ω) = ε_∞ + Σ (ω_p² / (ω_0² - ω² - jωΓ))

    Parameters
    ----------
    omega_0 : float
        Resonance frequency (rad/s).
    delta_epsilon : float
        Oscillator strength (dimensionless).
    gamma : float
        Damping rate (rad/s).
    """

    omega_0: float
    delta_epsilon: float
    gamma: float


@dataclass
class DrudePole:
    """
    Parameters for Drude model (metals, plasmas).

    The Drude model describes free carriers:
    ε(ω) = ε_∞ - (ω_p² / (ω² + jωΓ))

    Parameters
    ----------
    omega_p : float
        Plasma frequency (rad/s).
    gamma : float
        Collision frequency (rad/s).
    """

    omega_p: float
    gamma: float


@dataclass
class DebyePole:
    """
    Parameters for Debye model (dielectric relaxation).

    The Debye model describes relaxation:
    ε(ω) = ε_∞ + (ε_s - ε_∞) / (1 + jωτ)

    Parameters
    ----------
    epsilon_s : float
        Static permittivity.
    tau : float
        Relaxation time (s).
    """

    epsilon_s: float
    tau: float


class DispersiveMaterial(ABC):
    """
    Abstract base class for dispersive materials.

    All dispersive material models must implement methods for:
    - Computing permittivity at a given frequency
    - Generating ADE update coefficients for FDTD
    """

    def __init__(self, epsilon_inf: float = 1.0, name: str = ""):
        """
        Initialize dispersive material.

        Parameters
        ----------
        epsilon_inf : float
            High-frequency permittivity (ε_∞).
        name : str
            Material name for identification.
        """
        self.epsilon_inf = epsilon_inf
        self.name = name or self.__class__.__name__

    @abstractmethod
    def permittivity(
        self, omega: Union[float, np.ndarray]
    ) -> Union[complex, np.ndarray]:
        """
        Calculate complex permittivity at given frequency.

        Parameters
        ----------
        omega : float or array
            Angular frequency (rad/s).

        Returns
        -------
        complex or array of complex
            Complex relative permittivity ε(ω).
        """
        pass

    def refractive_index(
        self, omega: Union[float, np.ndarray]
    ) -> Union[complex, np.ndarray]:
        """
        Calculate complex refractive index at given frequency.

        Parameters
        ----------
        omega : float or array
            Angular frequency (rad/s).

        Returns
        -------
        complex or array of complex
            Complex refractive index n(ω) = sqrt(ε(ω)).
        """
        return np.sqrt(self.permittivity(omega))

    @abstractmethod
    def get_ade_coefficients(self, dt: float) -> dict:
        """
        Get Auxiliary Differential Equation (ADE) coefficients for FDTD.

        Parameters
        ----------
        dt : float
            Time step (s).

        Returns
        -------
        dict
            Dictionary of ADE coefficients for time-domain updates.
        """
        pass


class LorentzMaterial(DispersiveMaterial):
    """
    Material with Lorentz dispersion model.

    The Lorentz model describes resonant dielectric behavior,
    suitable for dielectrics with resonances.

    Parameters
    ----------
    epsilon_inf : float
        High-frequency permittivity.
    poles : List[LorentzPole]
        List of Lorentz poles.
    name : str, optional
        Material name.
    """

    def __init__(self, epsilon_inf: float, poles: List[LorentzPole], name: str = ""):
        super().__init__(epsilon_inf, name)
        self.poles = poles

    def permittivity(
        self, omega: Union[float, np.ndarray]
    ) -> Union[complex, np.ndarray]:
        """Calculate complex permittivity using Lorentz model."""
        eps = self.epsilon_inf * np.ones_like(omega, dtype=complex)

        for pole in self.poles:
            denominator = pole.omega_0**2 - omega**2 - 1j * omega * pole.gamma
            eps += pole.delta_epsilon * pole.omega_0**2 / denominator

        return eps

    def get_ade_coefficients(self, dt: float) -> dict:
        """
        Get ADE coefficients for Lorentz model.

        For each Lorentz pole, we solve:
        d²P/dt² + γ dP/dt + ω₀² P = ε₀ Δε ω₀² E

        Using bilinear transform (Trapezoidal rule):
        """
        coeffs = {
            "poles": [],
            "epsilon_inf": self.epsilon_inf,
        }

        for pole in self.poles:
            # Bilinear transform coefficients
            w0 = pole.omega_0
            g = pole.gamma
            de = pole.delta_epsilon

            # Denominator coefficient
            denom = 4.0 + 2 * g * dt + w0**2 * dt**2

            # Update coefficients for recursive convolution
            # P^(n+1) = C0 * E^(n+1) + C1 * E^n + C2 * P^n + C3 * P^(n-1)
            C0 = 2 * de * w0**2 * dt**2 / denom
            C1 = C0  # Symmetric
            C2 = (8.0 - 2 * w0**2 * dt**2) / denom
            C3 = -(4.0 - 2 * g * dt + w0**2 * dt**2) / denom

            coeffs["poles"].append(
                {
                    "C0": C0,
                    "C1": C1,
                    "C2": C2,
                    "C3": C3,
                    "omega_0": w0,
                    "gamma": g,
                    "delta_epsilon": de,
                }
            )

        return coeffs


class DrudeMaterial(DispersiveMaterial):
    """
    Material with Drude dispersion model (metals, plasmas).

    The Drude model describes free carrier behavior,
    suitable for metals and doped semiconductors.

    Parameters
    ----------
    epsilon_inf : float
        High-frequency permittivity.
    omega_p : float
        Plasma frequency (rad/s).
    gamma : float
        Collision frequency (rad/s).
    name : str, optional
        Material name.
    """

    def __init__(
        self, epsilon_inf: float, omega_p: float, gamma: float, name: str = ""
    ):
        super().__init__(epsilon_inf, name)
        self.omega_p = omega_p
        self.gamma = gamma

    def permittivity(
        self, omega: Union[float, np.ndarray]
    ) -> Union[complex, np.ndarray]:
        """Calculate complex permittivity using Drude model."""
        eps = self.epsilon_inf - self.omega_p**2 / (omega**2 + 1j * omega * self.gamma)
        return eps

    def get_ade_coefficients(self, dt: float) -> dict:
        """
        Get ADE coefficients for Drude model.

        The Drude equation:
        dJ/dt + γ J = ε₀ ω_p² E
        """
        # Exponential time-stepping
        exp_term = np.exp(-self.gamma * dt)

        C0 = self.omega_p**2 / self.gamma * (1.0 - exp_term)
        C1 = exp_term

        return {
            "C0": C0,  # Coefficient for E field
            "C1": C1,  # Coefficient for previous J
            "omega_p": self.omega_p,
            "gamma": self.gamma,
            "epsilon_inf": self.epsilon_inf,
        }


class DebyeMaterial(DispersiveMaterial):
    """
    Material with Debye dispersion model.

    The Debye model describes dielectric relaxation,
    suitable for polar dielectrics.

    Parameters
    ----------
    epsilon_inf : float
        High-frequency permittivity.
    epsilon_s : float
        Static permittivity.
    tau : float
        Relaxation time (s).
    name : str, optional
        Material name.
    """

    def __init__(
        self, epsilon_inf: float, epsilon_s: float, tau: float, name: str = ""
    ):
        super().__init__(epsilon_inf, name)
        self.epsilon_s = epsilon_s
        self.tau = tau

    def permittivity(
        self, omega: Union[float, np.ndarray]
    ) -> Union[complex, np.ndarray]:
        """Calculate complex permittivity using Debye model."""
        delta_eps = self.epsilon_s - self.epsilon_inf
        eps = self.epsilon_inf + delta_eps / (1.0 + 1j * omega * self.tau)
        return eps

    def get_ade_coefficients(self, dt: float) -> dict:
        """Get ADE coefficients for Debye model."""
        exp_term = np.exp(-dt / self.tau)

        C0 = (self.epsilon_s - self.epsilon_inf) * (1.0 - exp_term)
        C1 = exp_term

        return {
            "C0": C0,
            "C1": C1,
            "tau": self.tau,
            "epsilon_s": self.epsilon_s,
            "epsilon_inf": self.epsilon_inf,
        }


class SellmeierMaterial(DispersiveMaterial):
    """
    Material with Sellmeier dispersion formula.

    The Sellmeier formula is commonly used for transparent optical materials.
    n²(λ) = 1 + Σ (B_i λ² / (λ² - C_i))

    Parameters
    ----------
    B_coeffs : List[float]
        B coefficients in Sellmeier formula.
    C_coeffs : List[float]
        C coefficients in Sellmeier formula (in μm²).
    name : str, optional
        Material name.
    """

    def __init__(self, B_coeffs: List[float], C_coeffs: List[float], name: str = ""):
        super().__init__(epsilon_inf=1.0, name=name)
        self.B = np.array(B_coeffs)
        self.C = np.array(C_coeffs)

    def permittivity(
        self, omega: Union[float, np.ndarray]
    ) -> Union[complex, np.ndarray]:
        """Calculate permittivity from Sellmeier formula."""
        # Convert angular frequency to wavelength (μm)
        c = 299792458.0  # m/s
        wavelength_m = 2 * np.pi * c / omega
        wavelength_um = wavelength_m * 1e6  # Convert to μm

        lambda_sq = wavelength_um**2

        n_sq = 1.0
        for B, C in zip(self.B, self.C):
            n_sq += B * lambda_sq / (lambda_sq - C)

        return n_sq  # Return n² (which equals ε for non-magnetic materials)

    def get_ade_coefficients(self, dt: float) -> dict:
        """
        Sellmeier is not directly suitable for time-domain.

        Should be fitted to Lorentz poles for FDTD use.
        """
        raise NotImplementedError(
            "Sellmeier model should be fitted to Lorentz poles for time-domain simulation"
        )
