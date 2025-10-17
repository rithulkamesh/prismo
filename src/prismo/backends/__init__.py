"""
Backend abstraction layer for array operations.

This module provides a unified interface for array operations that can be
executed on different backends (CPU with NumPy, GPU with CuPy, etc.).
"""

from .base import Backend
from .backend_manager import get_backend, set_backend, list_available_backends

__all__ = [
    "Backend",
    "get_backend",
    "set_backend",
    "list_available_backends",
]
