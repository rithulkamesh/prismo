"""
NumPy backend for CPU computations.

This module implements the Backend interface using NumPy for
standard CPU-based array operations.
"""

from typing import Any, Tuple, Optional, Union
import numpy as np
from .base import Backend


class NumPyBackend(Backend):
    """
    NumPy-based backend for CPU computations.

    This backend uses NumPy for all array operations and is always available.
    """

    @property
    def name(self) -> str:
        return "numpy"

    @property
    def is_gpu(self) -> bool:
        return False

    def zeros(self, shape: Tuple[int, ...], dtype: Any = None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype or np.float64)

    def ones(self, shape: Tuple[int, ...], dtype: Any = None) -> np.ndarray:
        return np.ones(shape, dtype=dtype or np.float64)

    def empty(self, shape: Tuple[int, ...], dtype: Any = None) -> np.ndarray:
        return np.empty(shape, dtype=dtype or np.float64)

    def array(self, data: Any, dtype: Any = None) -> np.ndarray:
        return np.array(data, dtype=dtype)

    def asarray(self, data: Any, dtype: Any = None) -> np.ndarray:
        return np.asarray(data, dtype=dtype)

    def to_numpy(self, array: Any) -> np.ndarray:
        """Convert to NumPy array (no-op for NumPy backend)."""
        return np.asarray(array)

    def copy(self, array: Any) -> np.ndarray:
        return np.copy(array)

    # Mathematical operations
    def sqrt(self, array: Any) -> np.ndarray:
        return np.sqrt(array)

    def exp(self, array: Any) -> np.ndarray:
        return np.exp(array)

    def sin(self, array: Any) -> np.ndarray:
        return np.sin(array)

    def cos(self, array: Any) -> np.ndarray:
        return np.cos(array)

    def abs(self, array: Any) -> np.ndarray:
        return np.abs(array)

    def sum(
        self, array: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Any:
        return np.sum(array, axis=axis)

    def max(
        self, array: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Any:
        return np.max(array, axis=axis)

    def min(
        self, array: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Any:
        return np.min(array, axis=axis)

    def mean(
        self, array: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Any:
        return np.mean(array, axis=axis)

    # FFT operations
    def fft(self, array: Any, axis: int = -1) -> np.ndarray:
        return np.fft.fft(array, axis=axis)

    def ifft(self, array: Any, axis: int = -1) -> np.ndarray:
        return np.fft.ifft(array, axis=axis)

    def fft2(self, array: Any) -> np.ndarray:
        return np.fft.fft2(array)

    def ifft2(self, array: Any) -> np.ndarray:
        return np.fft.ifft2(array)

    # Linear algebra
    def dot(self, a: Any, b: Any) -> np.ndarray:
        return np.dot(a, b)

    def matmul(self, a: Any, b: Any) -> np.ndarray:
        return np.matmul(a, b)

    # Indexing and slicing helpers
    def where(self, condition: Any, x: Any, y: Any) -> np.ndarray:
        return np.where(condition, x, y)

    # Memory management
    def synchronize(self) -> None:
        """No-op for CPU backend."""
        pass

    def get_memory_info(self) -> dict:
        """Get memory usage information (limited for CPU)."""
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "backend": "numpy",
            "device": "CPU",
            "rss_bytes": memory_info.rss,
            "rss_mb": memory_info.rss / (1024**2),
            "vms_bytes": memory_info.vms,
            "vms_mb": memory_info.vms / (1024**2),
        }

    # Type information
    @property
    def float32(self) -> Any:
        return np.float32

    @property
    def float64(self) -> Any:
        return np.float64

    @property
    def complex64(self) -> Any:
        return np.complex64

    @property
    def complex128(self) -> Any:
        return np.complex128

    @property
    def int32(self) -> Any:
        return np.int32

    @property
    def int64(self) -> Any:
        return np.int64

    @property
    def pi(self) -> float:
        return np.pi
