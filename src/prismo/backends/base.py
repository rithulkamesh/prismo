"""
Abstract base class for computational backends.

This module defines the interface that all backends must implement,
providing array operations for FDTD computations.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, Union, Literal
import numpy as np


class Backend(ABC):
    """
    Abstract base class for computational backends.

    All backends must implement this interface to provide array operations
    for electromagnetic field computations.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the backend (e.g., 'numpy', 'cupy')."""
        pass

    @property
    @abstractmethod
    def is_gpu(self) -> bool:
        """Whether this backend uses GPU acceleration."""
        pass

    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create an array filled with zeros."""
        pass

    @abstractmethod
    def ones(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create an array filled with ones."""
        pass

    @abstractmethod
    def empty(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create an uninitialized array."""
        pass

    @abstractmethod
    def array(self, data: Any, dtype: Any = None) -> Any:
        """Convert data to backend array."""
        pass

    @abstractmethod
    def asarray(self, data: Any, dtype: Any = None) -> Any:
        """Convert data to backend array (no-copy if possible)."""
        pass

    @abstractmethod
    def to_numpy(self, array: Any) -> np.ndarray:
        """Convert backend array to NumPy array (for CPU)."""
        pass

    @abstractmethod
    def copy(self, array: Any) -> Any:
        """Create a copy of an array."""
        pass

    # Mathematical operations
    @abstractmethod
    def sqrt(self, array: Any) -> Any:
        """Element-wise square root."""
        pass

    @abstractmethod
    def exp(self, array: Any) -> Any:
        """Element-wise exponential."""
        pass

    @abstractmethod
    def sin(self, array: Any) -> Any:
        """Element-wise sine."""
        pass

    @abstractmethod
    def cos(self, array: Any) -> Any:
        """Element-wise cosine."""
        pass

    @abstractmethod
    def abs(self, array: Any) -> Any:
        """Element-wise absolute value."""
        pass

    @abstractmethod
    def sum(
        self, array: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Any:
        """Sum array elements."""
        pass

    @abstractmethod
    def max(
        self, array: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Any:
        """Maximum of array elements."""
        pass

    @abstractmethod
    def min(
        self, array: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Any:
        """Minimum of array elements."""
        pass

    @abstractmethod
    def mean(
        self, array: Any, axis: Optional[Union[int, Tuple[int, ...]]] = None
    ) -> Any:
        """Mean of array elements."""
        pass

    # FFT operations
    @abstractmethod
    def fft(self, array: Any, axis: int = -1) -> Any:
        """1D Fast Fourier Transform."""
        pass

    @abstractmethod
    def ifft(self, array: Any, axis: int = -1) -> Any:
        """1D Inverse Fast Fourier Transform."""
        pass

    @abstractmethod
    def fft2(self, array: Any) -> Any:
        """2D Fast Fourier Transform."""
        pass

    @abstractmethod
    def ifft2(self, array: Any) -> Any:
        """2D Inverse Fast Fourier Transform."""
        pass

    # Linear algebra
    @abstractmethod
    def dot(self, a: Any, b: Any) -> Any:
        """Dot product of two arrays."""
        pass

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication."""
        pass

    # Indexing and slicing helpers
    @abstractmethod
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """Return elements chosen from x or y depending on condition."""
        pass

    # Memory management
    @abstractmethod
    def synchronize(self) -> None:
        """Synchronize device (for GPU backends)."""
        pass

    @abstractmethod
    def get_memory_info(self) -> dict:
        """Get memory usage information."""
        pass

    # Type information
    @property
    @abstractmethod
    def float32(self) -> Any:
        """32-bit float dtype."""
        pass

    @property
    @abstractmethod
    def float64(self) -> Any:
        """64-bit float dtype."""
        pass

    @property
    @abstractmethod
    def complex64(self) -> Any:
        """64-bit complex dtype."""
        pass

    @property
    @abstractmethod
    def complex128(self) -> Any:
        """128-bit complex dtype."""
        pass

    @property
    @abstractmethod
    def int32(self) -> Any:
        """32-bit integer dtype."""
        pass

    @property
    @abstractmethod
    def int64(self) -> Any:
        """64-bit integer dtype."""
        pass

    # Constants
    @property
    @abstractmethod
    def pi(self) -> float:
        """Value of pi."""
        pass

    def __repr__(self) -> str:
        """String representation."""
        device = "GPU" if self.is_gpu else "CPU"
        return f"{self.__class__.__name__}(name='{self.name}', device={device})"
