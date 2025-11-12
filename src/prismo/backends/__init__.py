"""
Backend abstraction layer for array operations.

This module provides a unified interface for array operations that can be
executed on different backends (CPU with NumPy, GPU with CuPy, etc.).
"""

from .backend_manager import get_backend, list_available_backends, set_backend
from .base import Backend

# Try to import Metal backend if available
try:
    from .metal_backend import MetalBackend

    __all__ = [
        "Backend",
        "get_backend",
        "set_backend",
        "list_available_backends",
        "MetalBackend",
    ]
except ImportError:
    __all__ = [
        "Backend",
        "get_backend",
        "set_backend",
        "list_available_backends",
    ]
