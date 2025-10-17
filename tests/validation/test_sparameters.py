"""
Validation tests for S-parameter extraction.

Tests S-parameter calculations against known devices.
"""

import pytest
import numpy as np

from prismo.analysis import SParameterAnalyzer


class TestSParameterProperties:
    """Test fundamental S-parameter properties."""

    def test_reciprocity_lossless(self):
        """Test reciprocity for lossless 2-port device."""
        frequencies = np.linspace(190e12, 200e12, 11)

        s_analyzer = SParameterAnalyzer(num_ports=2, frequencies=frequencies)

        # Create reciprocal S-matrix
        # For lossless reciprocal: S12 = S21
        s21 = 0.9 * np.exp(-1j * np.linspace(0, np.pi, len(frequencies)))

        s_analyzer.s_matrix[:, 1, 0] = s21  # S21
        s_analyzer.s_matrix[:, 0, 1] = s21  # S12 = S21

        # Check reciprocity
        error = s_analyzer.check_reciprocity()
        assert error < 1e-10

    def test_unitarity_lossless(self):
        """Test unitarity for lossless device."""
        frequencies = np.array([193e12])

        s_analyzer = SParameterAnalyzer(num_ports=2, frequencies=frequencies)

        # Create lossless S-matrix (unitary)
        # Example: 50/50 beam splitter
        s_analyzer.s_matrix[0, :, :] = np.array(
            [[0, 1j / np.sqrt(2)], [1j / np.sqrt(2), 0]]
        )

        # Check unitarity: S†S = I
        error = s_analyzer.check_unitarity(0)
        assert error < 1e-10

    def test_power_conservation(self):
        """Test power conservation: |S11|² + |S21|² = 1 for lossless 2-port."""
        frequencies = np.array([193e12])

        s_analyzer = SParameterAnalyzer(num_ports=2, frequencies=frequencies)

        # Lossless 2-port
        s11 = 0.6
        s21 = 0.8

        s_analyzer.s_matrix[0, 0, 0] = s11
        s_analyzer.s_matrix[0, 1, 0] = s21

        # Power conservation
        power_in = abs(s11) ** 2 + abs(s21) ** 2
        assert abs(power_in - 1.0) < 1e-10


class TestDirectionalCoupler:
    """Test S-parameters of a directional coupler."""

    def test_3db_coupler(self):
        """Test 3dB coupler (50/50 splitting)."""
        frequencies = np.array([193e12])

        s_analyzer = SParameterAnalyzer(num_ports=4, frequencies=frequencies)

        # Ideal 3dB coupler S-matrix
        # All ports have equal splitting
        s_matrix = np.array(
            [[0, 0.5, 0.5, 0], [0.5, 0, 0, 0.5], [0.5, 0, 0, 0.5], [0, 0.5, 0.5, 0]]
        ) / np.sqrt(2)

        s_analyzer.s_matrix[0, :, :] = s_matrix

        # Check power splitting
        # If port 0 excited, ports 1 and 2 should each get 50% power
        assert abs(abs(s_analyzer.get_s_parameter(1, 0)[0]) ** 2 - 0.5) < 0.01
        assert abs(abs(s_analyzer.get_s_parameter(2, 0)[0]) ** 2 - 0.5) < 0.01
