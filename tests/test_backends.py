"""
Tests for backend abstraction layer.

This module tests the backend interface and compares CPU vs GPU results
for numerical equivalence.
"""

import pytest
import numpy as np

from prismo.backends import list_available_backends, get_backend, set_backend
from prismo.backends.numpy_backend import NumPyBackend


class TestBackendInterface:
    """Test backend interface implementation."""

    def test_list_backends(self):
        """Test listing available backends."""
        backends = list_available_backends()
        assert isinstance(backends, list)
        assert "numpy" in backends

    def test_numpy_backend_creation(self):
        """Test NumPy backend creation."""
        backend = NumPyBackend()
        assert backend.name == "numpy"
        assert not backend.is_gpu

    def test_set_backend(self):
        """Test setting backend."""
        backend = set_backend("numpy")
        assert backend.name == "numpy"

    def test_array_operations(self):
        """Test basic array operations."""
        backend = get_backend("numpy")

        # Create arrays
        arr = backend.zeros((10, 10))
        assert arr.shape == (10, 10)

        ones = backend.ones((5, 5))
        assert np.allclose(ones, 1.0)

    def test_math_operations(self):
        """Test mathematical operations."""
        backend = get_backend("numpy")

        arr = backend.array([1.0, 4.0, 9.0])
        sqrt_arr = backend.sqrt(arr)

        assert np.allclose(sqrt_arr, [1.0, 2.0, 3.0])

    @pytest.mark.skipif(
        "cupy" not in list_available_backends(), reason="CuPy not available"
    )
    def test_gpu_backend(self):
        """Test GPU backend if available."""
        backend = set_backend("cupy")
        assert backend.is_gpu

        # Create GPU array
        arr = backend.ones((100, 100))

        # Test conversion to NumPy
        arr_np = backend.to_numpy(arr)
        assert isinstance(arr_np, np.ndarray)
        assert np.allclose(arr_np, 1.0)


class TestBackendEquivalence:
    """Test CPU and GPU backends produce equivalent results."""

    @pytest.mark.skipif(
        "cupy" not in list_available_backends(), reason="CuPy not available"
    )
    def test_field_operations_equivalence(self):
        """Test that CPU and GPU backends give same results."""
        # Create test data
        shape = (50, 50, 50)
        data = np.random.random(shape)

        # CPU backend
        backend_cpu = get_backend("numpy")
        arr_cpu = backend_cpu.array(data)
        result_cpu = backend_cpu.sqrt(arr_cpu)

        # GPU backend
        backend_gpu = get_backend("cupy")
        arr_gpu = backend_gpu.array(data)
        result_gpu = backend_gpu.sqrt(arr_gpu)
        result_gpu_np = backend_gpu.to_numpy(result_gpu)

        # Compare
        assert np.allclose(result_cpu, result_gpu_np, rtol=1e-10)

    @pytest.mark.skipif(
        "cupy" not in list_available_backends(), reason="CuPy not available"
    )
    def test_fft_equivalence(self):
        """Test FFT equivalence between backends."""
        data = np.random.random(100) + 1j * np.random.random(100)

        backend_cpu = get_backend("numpy")
        backend_gpu = get_backend("cupy")

        fft_cpu = backend_cpu.fft(data)
        fft_gpu = backend_gpu.to_numpy(backend_gpu.fft(data))

        assert np.allclose(fft_cpu, fft_gpu, rtol=1e-10)
