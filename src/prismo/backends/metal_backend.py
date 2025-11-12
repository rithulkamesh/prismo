"""
Metal backend for GPU computations on macOS.

This module implements the Backend interface using Metal for
GPU-accelerated array operations on Apple Silicon and Intel Macs.
"""

import platform
from typing import Any, Optional, Union

import numpy as np

from .base import Backend

try:
    import Metal
    from Metal import (
        MTLBuffer,
        MTLResourceStorageModePrivate,
        MTLResourceStorageModeShared,
    )

    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
    Metal = None
    MTLDevice = None


class MetalBackend(Backend):
    """
    Metal-based backend for GPU computations on macOS.

    This backend uses Metal for GPU-accelerated array operations.
    Requires macOS and Metal framework to be available.

    Parameters
    ----------
    device_id : int, optional
        Metal device ID to use. Default is 0.
    use_unified_memory : bool, optional
        Whether to use unified memory (shared storage mode). Default is True.
    """

    def __init__(self, device_id: int = 0, use_unified_memory: bool = True):
        if not METAL_AVAILABLE:
            raise RuntimeError(
                "Metal is not available. This backend requires macOS with Metal framework."
            )

        if platform.system() != "Darwin":
            raise RuntimeError("Metal backend requires macOS")

        self.device_id = device_id
        self.use_unified_memory = use_unified_memory

        # Get Metal device
        devices = Metal.MTLCopyAllDevices()
        if device_id >= len(devices):
            raise ValueError(
                f"Device ID {device_id} not available. Found {len(devices)} devices."
            )

        self.device = devices[device_id]
        self.device_name = self.device.name()

        # Create command queue
        self.command_queue = self.device.newCommandQueue()

        # Storage mode for buffers
        self.storage_mode = (
            MTLResourceStorageModeShared
            if use_unified_memory
            else MTLResourceStorageModePrivate
        )

        # Cache for compiled compute pipelines
        self._pipeline_cache = {}

        # Memory pool for buffer reuse
        self._buffer_pool = {}

    @property
    def name(self) -> str:
        return "metal"

    @property
    def is_gpu(self) -> bool:
        return True

    def _get_buffer(self, size: int, dtype: Any = None) -> MTLBuffer:
        """Get or create a Metal buffer."""
        # For now, create new buffers. In production, implement pooling
        if dtype is None:
            dtype = np.float64

        # Convert numpy dtype to Metal data type
        if dtype == np.float32:
            metal_dtype = Metal.MTLDataTypeFloat
        elif dtype == np.float64:
            metal_dtype = Metal.MTLDataTypeDouble
        elif dtype == np.int32:
            metal_dtype = Metal.MTLDataTypeInt
        elif dtype == np.int64:
            metal_dtype = Metal.MTLDataTypeLong
        else:
            metal_dtype = Metal.MTLDataTypeFloat  # Default to float

        buffer = self.device.newBufferWithLength_options_(size, self.storage_mode)
        return buffer

    def _numpy_to_metal_buffer(self, array: np.ndarray) -> MTLBuffer:
        """Convert NumPy array to Metal buffer."""
        buffer = self._get_buffer(array.nbytes, array.dtype)

        if self.use_unified_memory:
            # Copy data directly to shared memory
            buffer.contents().as_buffer(array.nbytes)[:] = array.tobytes()
        else:
            # For private memory, would need explicit copy
            # This is simplified for now
            buffer.contents().as_buffer(array.nbytes)[:] = array.tobytes()

        return buffer

    def _metal_buffer_to_numpy(
        self, buffer: MTLBuffer, shape: tuple, dtype: Any
    ) -> np.ndarray:
        """Convert Metal buffer to NumPy array."""
        if self.use_unified_memory:
            # Direct access to shared memory
            data = np.frombuffer(
                buffer.contents().as_buffer(buffer.length()), dtype=dtype
            ).reshape(shape)
        else:
            # Would need explicit copy for private memory
            data = np.frombuffer(
                buffer.contents().as_buffer(buffer.length()), dtype=dtype
            ).reshape(shape)

        return data.copy()  # Return a copy to avoid memory issues

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        """Create an array filled with zeros."""
        if dtype is None:
            dtype = np.float64

        array = np.zeros(shape, dtype=dtype)
        return self._numpy_to_metal_buffer(array)

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        """Create an array filled with ones."""
        if dtype is None:
            dtype = np.float64

        array = np.ones(shape, dtype=dtype)
        return self._numpy_to_metal_buffer(array)

    def empty(self, shape: tuple[int, ...], dtype: Any = None) -> Any:
        """Create an uninitialized array."""
        if dtype is None:
            dtype = np.float64

        array = np.empty(shape, dtype=dtype)
        return self._numpy_to_metal_buffer(array)

    def array(self, data: Any, dtype: Any = None) -> Any:
        """Convert data to backend array."""
        if isinstance(data, np.ndarray):
            if dtype is not None and data.dtype != dtype:
                data = data.astype(dtype)
            return self._numpy_to_metal_buffer(data)
        else:
            array = np.array(data, dtype=dtype)
            return self._numpy_to_metal_buffer(array)

    def asarray(self, data: Any, dtype: Any = None) -> Any:
        """Convert data to backend array (no-copy if possible)."""
        return self.array(data, dtype)

    def to_numpy(self, array: Any) -> np.ndarray:
        """Convert Metal buffer to NumPy array."""
        if isinstance(array, MTLBuffer):
            # This is a simplified implementation
            # In practice, we'd need to track buffer metadata
            raise NotImplementedError(
                "Buffer to NumPy conversion requires metadata tracking"
            )
        elif isinstance(array, np.ndarray):
            return array
        else:
            raise TypeError(f"Cannot convert {type(array)} to NumPy array")

    def copy(self, array: Any) -> Any:
        """Create a copy of an array."""
        if isinstance(array, MTLBuffer):
            # Create new buffer and copy data
            new_buffer = self._get_buffer(array.length())
            new_buffer.contents().as_buffer(array.length())[
                :
            ] = array.contents().as_buffer(array.length())
            return new_buffer
        elif isinstance(array, np.ndarray):
            return self._numpy_to_metal_buffer(array.copy())
        else:
            raise TypeError(f"Cannot copy {type(array)}")

    # Mathematical operations
    def sqrt(self, array: Any) -> Any:
        """Element-wise square root."""
        # For now, fall back to NumPy operations
        # In production, implement Metal compute shaders
        if isinstance(array, MTLBuffer):
            # Convert to NumPy, compute, convert back
            np_array = self._metal_buffer_to_numpy(
                array, (1,), np.float64
            )  # Simplified
            result = np.sqrt(np_array)
            return self._numpy_to_metal_buffer(result)
        else:
            return np.sqrt(array)

    def exp(self, array: Any) -> Any:
        """Element-wise exponential."""
        if isinstance(array, MTLBuffer):
            np_array = self._metal_buffer_to_numpy(
                array, (1,), np.float64
            )  # Simplified
            result = np.exp(np_array)
            return self._numpy_to_metal_buffer(result)
        else:
            return np.exp(array)

    def sin(self, array: Any) -> Any:
        """Element-wise sine."""
        if isinstance(array, MTLBuffer):
            np_array = self._metal_buffer_to_numpy(
                array, (1,), np.float64
            )  # Simplified
            result = np.sin(np_array)
            return self._numpy_to_metal_buffer(result)
        else:
            return np.sin(array)

    def cos(self, array: Any) -> Any:
        """Element-wise cosine."""
        if isinstance(array, MTLBuffer):
            np_array = self._metal_buffer_to_numpy(
                array, (1,), np.float64
            )  # Simplified
            result = np.cos(np_array)
            return self._numpy_to_metal_buffer(result)
        else:
            return np.cos(array)

    def abs(self, array: Any) -> Any:
        """Element-wise absolute value."""
        if isinstance(array, MTLBuffer):
            np_array = self._metal_buffer_to_numpy(
                array, (1,), np.float64
            )  # Simplified
            result = np.abs(np_array)
            return self._numpy_to_metal_buffer(result)
        else:
            return np.abs(array)

    def sum(
        self, array: Any, axis: Optional[Union[int, tuple[int, ...]]] = None
    ) -> Any:
        """Sum array elements."""
        if isinstance(array, MTLBuffer):
            np_array = self._metal_buffer_to_numpy(
                array, (1,), np.float64
            )  # Simplified
            result = np.sum(np_array, axis=axis)
            return self._numpy_to_metal_buffer(result)
        else:
            return np.sum(array, axis=axis)

    def max(
        self, array: Any, axis: Optional[Union[int, tuple[int, ...]]] = None
    ) -> Any:
        """Maximum of array elements."""
        if isinstance(array, MTLBuffer):
            np_array = self._metal_buffer_to_numpy(
                array, (1,), np.float64
            )  # Simplified
            result = np.max(np_array, axis=axis)
            return self._numpy_to_metal_buffer(result)
        else:
            return np.max(array, axis=axis)

    def min(
        self, array: Any, axis: Optional[Union[int, tuple[int, ...]]] = None
    ) -> Any:
        """Minimum of array elements."""
        if isinstance(array, MTLBuffer):
            np_array = self._metal_buffer_to_numpy(
                array, (1,), np.float64
            )  # Simplified
            result = np.min(np_array, axis=axis)
            return self._numpy_to_metal_buffer(result)
        else:
            return np.min(array, axis=axis)

    def mean(
        self, array: Any, axis: Optional[Union[int, tuple[int, ...]]] = None
    ) -> Any:
        """Mean of array elements."""
        if isinstance(array, MTLBuffer):
            np_array = self._metal_buffer_to_numpy(
                array, (1,), np.float64
            )  # Simplified
            result = np.mean(np_array, axis=axis)
            return self._numpy_to_metal_buffer(result)
        else:
            return np.mean(array, axis=axis)

    # FFT operations
    def fft(self, array: Any, axis: int = -1) -> Any:
        """1D Fast Fourier Transform."""
        if isinstance(array, MTLBuffer):
            np_array = self._metal_buffer_to_numpy(
                array, (1,), np.float64
            )  # Simplified
            result = np.fft.fft(np_array, axis=axis)
            return self._numpy_to_metal_buffer(result)
        else:
            return np.fft.fft(array, axis=axis)

    def ifft(self, array: Any, axis: int = -1) -> Any:
        """1D Inverse Fast Fourier Transform."""
        if isinstance(array, MTLBuffer):
            np_array = self._metal_buffer_to_numpy(
                array, (1,), np.float64
            )  # Simplified
            result = np.fft.ifft(np_array, axis=axis)
            return self._numpy_to_metal_buffer(result)
        else:
            return np.fft.ifft(array, axis=axis)

    def fft2(self, array: Any) -> Any:
        """2D Fast Fourier Transform."""
        if isinstance(array, MTLBuffer):
            np_array = self._metal_buffer_to_numpy(
                array, (1,), np.float64
            )  # Simplified
            result = np.fft.fft2(np_array)
            return self._numpy_to_metal_buffer(result)
        else:
            return np.fft.fft2(array)

    def ifft2(self, array: Any) -> Any:
        """2D Inverse Fast Fourier Transform."""
        if isinstance(array, MTLBuffer):
            np_array = self._metal_buffer_to_numpy(
                array, (1,), np.float64
            )  # Simplified
            result = np.fft.ifft2(np_array)
            return self._numpy_to_metal_buffer(result)
        else:
            return np.fft.ifft2(array)

    # Linear algebra
    def dot(self, a: Any, b: Any) -> Any:
        """Dot product of two arrays."""
        if isinstance(a, MTLBuffer) or isinstance(b, MTLBuffer):
            # Convert to NumPy for now
            if isinstance(a, MTLBuffer):
                a = self._metal_buffer_to_numpy(a, (1,), np.float64)  # Simplified
            if isinstance(b, MTLBuffer):
                b = self._metal_buffer_to_numpy(b, (1,), np.float64)  # Simplified
            result = np.dot(a, b)
            return self._numpy_to_metal_buffer(result)
        else:
            return np.dot(a, b)

    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication."""
        if isinstance(a, MTLBuffer) or isinstance(b, MTLBuffer):
            # Convert to NumPy for now
            if isinstance(a, MTLBuffer):
                a = self._metal_buffer_to_numpy(a, (1,), np.float64)  # Simplified
            if isinstance(b, MTLBuffer):
                b = self._metal_buffer_to_numpy(b, (1,), np.float64)  # Simplified
            result = np.matmul(a, b)
            return self._numpy_to_metal_buffer(result)
        else:
            return np.matmul(a, b)

    # Indexing and slicing helpers
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """Return elements chosen from x or y depending on condition."""
        # Convert all to NumPy for now
        if isinstance(condition, MTLBuffer):
            condition = self._metal_buffer_to_numpy(
                condition, (1,), np.bool_
            )  # Simplified
        if isinstance(x, MTLBuffer):
            x = self._metal_buffer_to_numpy(x, (1,), np.float64)  # Simplified
        if isinstance(y, MTLBuffer):
            y = self._metal_buffer_to_numpy(y, (1,), np.float64)  # Simplified

        result = np.where(condition, x, y)
        return self._numpy_to_metal_buffer(result)

    # Memory management
    def synchronize(self) -> None:
        """Synchronize Metal device."""
        # For now, this is a no-op since we're using shared memory
        # In production with private memory, would need to wait for command buffer
        pass

    def get_memory_info(self) -> dict:
        """Get Metal device memory usage information."""
        # Metal doesn't provide detailed memory info like CUDA
        # This is a simplified implementation
        return {
            "backend": "metal",
            "device": f"Metal:{self.device_id}",
            "device_name": self.device_name,
            "unified_memory": self.use_unified_memory,
            "storage_mode": "shared" if self.use_unified_memory else "private",
            "max_buffer_size": self.device.maxBufferLength(),
            "recommended_max_working_set_size": self.device.recommendedMaxWorkingSetSize(),
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

    def __repr__(self) -> str:
        """String representation."""
        return f"MetalBackend(device={self.device_id}, name='{self.device_name}', unified_memory={self.use_unified_memory})"
