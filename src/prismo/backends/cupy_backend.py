"""
CuPy backend for GPU computations.

This module implements the Backend interface using CuPy for
GPU-accelerated array operations with CUDA.
"""

from typing import Any, Tuple, Optional, Union
import numpy as np
from .base import Backend

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


class CuPyBackend(Backend):
    """
    CuPy-based backend for GPU computations.

    This backend uses CuPy for GPU-accelerated array operations.
    Requires CUDA and CuPy to be installed.

    Parameters
    ----------
    device_id : int, optional
        CUDA device ID to use. Default is 0.
    """

    def __init__(self, device_id: int = 0):
        if not CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy is not available. Install with: pip install cupy-cuda12x"
            )

        self.device_id = device_id
        self.device = cp.cuda.Device(device_id)
        self.device.use()

    @property
    def name(self) -> str:
        return "cupy"

    @property
    def is_gpu(self) -> bool:
        return True

    def zeros(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        with self.device:
            return cp.zeros(shape, dtype=dtype or cp.float64)

    def ones(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        with self.device:
            return cp.ones(shape, dtype=dtype or cp.float64)

    def empty(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        with self.device:
            return cp.empty(shape, dtype=dtype or cp.float64)

    def array(self, data: Any, dtype: Any = None) -> Any:
        with self.device:
            return cp.array(data, dtype=dtype)

    def asarray(self, data: Any, dtype: Any = None) -> Any:
        with self.device:
            return cp.asarray(data, dtype=dtype)

    def to_numpy(self, array: Any) -> np.ndarray:
        """Convert CuPy array to NumPy array (transfers from GPU to CPU)."""
        return cp.asnumpy(array)

    def copy(self, array: Any) -> Any:
        return cp.copy(array)

    # Mathematical operations
    def sqrt(self, array: Any) -> Any:
        return cp.sqrt(array)

    def exp(self, array: Any) -> Any:
        return cp.exp(array)

    def sin(self, array: Any) -> Any:
        return cp.sin(array)

    def cos(self, array: Any) -> Any:
        return cp.cos(array)

    def abs(self, array: Any) -> Any:
        return cp.abs(array)

    def sum(
        self, array: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Any:
        return cp.sum(array, axis=axis)

    def max(
        self, array: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Any:
        return cp.max(array, axis=axis)

    def min(
        self, array: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Any:
        return cp.min(array, axis=axis)

    def mean(
        self, array: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Any:
        return cp.mean(array, axis=axis)

    # FFT operations
    def fft(self, array: Any, axis: int = -1) -> Any:
        return cp.fft.fft(array, axis=axis)

    def ifft(self, array: Any, axis: int = -1) -> Any:
        return cp.fft.ifft(array, axis=axis)

    def fft2(self, array: Any) -> Any:
        return cp.fft.fft2(array)

    def ifft2(self, array: Any) -> Any:
        return cp.fft.ifft2(array)

    # Linear algebra
    def dot(self, a: Any, b: Any) -> Any:
        return cp.dot(a, b)

    def matmul(self, a: Any, b: Any) -> Any:
        return cp.matmul(a, b)

    # Indexing and slicing helpers
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        return cp.where(condition, x, y)

    # Memory management
    def synchronize(self) -> None:
        """Synchronize CUDA device."""
        cp.cuda.Stream.null.synchronize()

    def get_memory_info(self) -> dict:
        """Get GPU memory usage information."""
        mempool = cp.get_default_memory_pool()
        return {
            "backend": "cupy",
            "device": f"GPU:{self.device_id}",
            "device_name": self.device.name,
            "used_bytes": mempool.used_bytes(),
            "used_mb": mempool.used_bytes() / (1024**2),
            "total_bytes": mempool.total_bytes(),
            "total_mb": mempool.total_bytes() / (1024**2),
            "free_bytes": self.device.mem_info[0],
            "free_mb": self.device.mem_info[0] / (1024**2),
            "total_device_bytes": self.device.mem_info[1],
            "total_device_mb": self.device.mem_info[1] / (1024**2),
        }

    # Type information
    @property
    def float32(self) -> Any:
        return cp.float32

    @property
    def float64(self) -> Any:
        return cp.float64

    @property
    def complex64(self) -> Any:
        return cp.complex64

    @property
    def complex128(self) -> Any:
        return cp.complex128

    @property
    def int32(self) -> Any:
        return cp.int32

    @property
    def int64(self) -> Any:
        return cp.int64

    @property
    def pi(self) -> float:
        return float(cp.pi)

    def __repr__(self) -> str:
        """String representation."""
        return f"CuPyBackend(device={self.device_id}, name='{self.device.name}')"
