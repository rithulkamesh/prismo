"""
S-parameter extraction and analysis.

This module provides tools for computing S-parameters from FDTD simulations
using mode expansion or power flux methods.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np


class SParameterAnalyzer:
    """
    S-parameter analyzer for multi-port devices.

    Computes scattering parameters from mode expansion or flux monitors
    and provides export capabilities.

    Parameters
    ----------
    num_ports : int
        Number of ports in the device.
    frequencies : ndarray
        Array of frequencies (Hz) for which to compute S-parameters.
    reference_impedance : float, optional
        Reference impedance (Ω), default=50.
    """

    def __init__(
        self, num_ports: int, frequencies: np.ndarray, reference_impedance: float = 50.0
    ):
        self.num_ports = num_ports
        self.frequencies = np.asarray(frequencies)
        self.z0 = reference_impedance

        # Storage for S-parameters
        # Shape: (n_frequencies, n_ports, n_ports)
        self.s_matrix = np.zeros(
            (len(self.frequencies), num_ports, num_ports), dtype=complex
        )

        # Port power normalization data
        self.port_powers = {}

    def add_port_data(
        self,
        port_index: int,
        excitation_port: int,
        power_forward: np.ndarray,
        power_backward: np.ndarray,
    ) -> None:
        """
        Add port measurement data from a simulation.

        Parameters
        ----------
        port_index : int
            Port at which measurement was made.
        excitation_port : int
            Port that was excited in this simulation.
        power_forward : ndarray
            Forward-propagating power at measurement port vs frequency.
        power_backward : ndarray
            Backward-propagating power at measurement port vs frequency.
        """
        if excitation_port not in self.port_powers:
            self.port_powers[excitation_port] = power_forward

        # S_ij = sqrt(P_i / P_j) where P_i is power at port i when j is excited
        # For reflection: S_ii = P_reflected / P_incident
        # For transmission: S_ij = sqrt(P_transmitted_i / P_incident_j)

        incident_power = self.port_powers[excitation_port]

        if port_index == excitation_port:
            # Reflection coefficient
            self.s_matrix[:, port_index, excitation_port] = (
                power_backward / incident_power
            ) ** 0.5
        else:
            # Transmission coefficient
            self.s_matrix[:, port_index, excitation_port] = (
                power_forward / incident_power
            ) ** 0.5

    def add_mode_data(
        self,
        port_index: int,
        excitation_port: int,
        mode_coefficients: Dict[str, np.ndarray],
    ) -> None:
        """
        Add S-parameter data from mode expansion.

        Parameters
        ----------
        port_index : int
            Port at which measurement was made.
        excitation_port : int
            Port that was excited.
        mode_coefficients : dict
            Dictionary with 'forward' and 'backward' mode coefficients vs frequency.
        """
        forward_coeff = mode_coefficients.get(
            "forward", np.zeros_like(self.frequencies)
        )
        backward_coeff = mode_coefficients.get(
            "backward", np.zeros_like(self.frequencies)
        )

        if port_index == excitation_port:
            # Reflection: S_ii = backward / forward
            self.s_matrix[:, port_index, excitation_port] = (
                backward_coeff / forward_coeff
            )
        else:
            # Transmission: S_ij = transmitted / incident
            self.s_matrix[:, port_index, excitation_port] = (
                forward_coeff / backward_coeff
            )

    def get_s_parameter(self, i: int, j: int) -> np.ndarray:
        """
        Get a specific S-parameter vs frequency.

        Parameters
        ----------
        i : int
            Output port index (0-based).
        j : int
            Input port index (0-based).

        Returns
        -------
        ndarray
            Complex S-parameter S_ij vs frequency.
        """
        return self.s_matrix[:, i, j]

    def get_s_matrix(self, frequency_index: int) -> np.ndarray:
        """
        Get the full S-matrix at a specific frequency.

        Parameters
        ----------
        frequency_index : int
            Frequency index.

        Returns
        -------
        ndarray
            S-matrix (num_ports × num_ports).
        """
        return self.s_matrix[frequency_index, :, :]

    def get_insertion_loss_db(self, i: int, j: int) -> np.ndarray:
        """
        Get insertion loss in dB.

        IL = -20 log10(|S_ij|)

        Parameters
        ----------
        i, j : int
            Port indices.

        Returns
        -------
        ndarray
            Insertion loss in dB vs frequency.
        """
        s_ij = self.get_s_parameter(i, j)
        return -20 * np.log10(np.abs(s_ij))

    def get_return_loss_db(self, port: int) -> np.ndarray:
        """
        Get return loss in dB.

        RL = -20 log10(|S_ii|)

        Parameters
        ----------
        port : int
            Port index.

        Returns
        -------
        ndarray
            Return loss in dB vs frequency.
        """
        s_ii = self.get_s_parameter(port, port)
        return -20 * np.log10(np.abs(s_ii))

    def check_reciprocity(self) -> float:
        """
        Check reciprocity: S_ij should equal S_ji.

        Returns
        -------
        float
            Maximum reciprocity error.
        """
        max_error = 0.0
        for i in range(self.num_ports):
            for j in range(i + 1, self.num_ports):
                error = np.max(np.abs(self.s_matrix[:, i, j] - self.s_matrix[:, j, i]))
                max_error = max(max_error, error)
        return max_error

    def check_unitarity(self, frequency_index: int) -> float:
        """
        Check unitarity: S†S should equal I for lossless system.

        Parameters
        ----------
        frequency_index : int
            Frequency index to check.

        Returns
        -------
        float
            Unitarity error (Frobenius norm of S†S - I).
        """
        S = self.get_s_matrix(frequency_index)
        S_dag_S = np.dot(S.conj().T, S)
        I = np.eye(self.num_ports)
        error = np.linalg.norm(S_dag_S - I, "fro")
        return error


def export_touchstone(
    filename: Path,
    frequencies: np.ndarray,
    s_matrix: np.ndarray,
    z0: float = 50.0,
    comments: Optional[List[str]] = None,
) -> None:
    """
    Export S-parameters to Touchstone format (.sNp file).

    Parameters
    ----------
    filename : Path
        Output filename (e.g., "device.s2p" for 2-port).
    frequencies : ndarray
        Frequency array (Hz).
    s_matrix : ndarray
        S-matrix data, shape (n_freq, n_ports, n_ports).
    z0 : float
        Reference impedance (Ω).
    comments : list of str, optional
        Comment lines to include in header.
    """
    num_ports = s_matrix.shape[1]

    with open(filename, "w") as f:
        # Write header comments
        f.write(f"! Touchstone file generated by Prismo\n")
        if comments:
            for comment in comments:
                f.write(f"! {comment}\n")

        # Write option line
        f.write(f"# Hz S RI R {z0}\n")

        # Write data
        for freq_idx, freq in enumerate(frequencies):
            f.write(f"{freq:.10e}")

            # Write S-parameters in row-major order
            for i in range(num_ports):
                for j in range(num_ports):
                    s_val = s_matrix[freq_idx, i, j]
                    # Write as real and imaginary parts
                    f.write(f" {s_val.real:.10e} {s_val.imag:.10e}")

            f.write("\n")


def compute_group_delay(frequencies: np.ndarray, s_parameter: np.ndarray) -> np.ndarray:
    """
    Compute group delay from S-parameter phase.

    τ_g = -dφ/dω

    Parameters
    ----------
    frequencies : ndarray
        Frequency array (Hz).
    s_parameter : ndarray
        Complex S-parameter vs frequency.

    Returns
    -------
    ndarray
        Group delay (s) vs frequency.
    """
    omega = 2 * np.pi * frequencies
    phase = np.unwrap(np.angle(s_parameter))

    # Numerical derivative
    tau_g = -np.gradient(phase, omega)

    return tau_g


def compute_group_index(
    frequencies: np.ndarray, s21: np.ndarray, length: float
) -> np.ndarray:
    """
    Compute group index from transmission phase.

    n_g = c * τ_g / L

    Parameters
    ----------
    frequencies : ndarray
        Frequency array (Hz).
    s21 : ndarray
        Transmission coefficient S21.
    length : float
        Device length (m).

    Returns
    -------
    ndarray
        Group index vs frequency.
    """
    c = 299792458.0  # Speed of light
    tau_g = compute_group_delay(frequencies, s21)
    n_g = c * tau_g / length
    return n_g
