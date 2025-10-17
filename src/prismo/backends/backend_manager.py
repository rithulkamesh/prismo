"""
Backend manager for selecting and configuring computational backends.

This module provides utilities for automatically detecting available backends
and selecting the appropriate one based on hardware and user preferences.
"""

from typing import Optional, List, Dict
import warnings
from .base import Backend
from .numpy_backend import NumPyBackend

# Try to import CuPy backend
try:
    from .cupy_backend import CuPyBackend, CUPY_AVAILABLE
except ImportError:
    CUPY_AVAILABLE = False
    CuPyBackend = None


# Global backend instance
_CURRENT_BACKEND: Optional[Backend] = None


def list_available_backends() -> List[str]:
    """
    List all available backends on this system.

    Returns
    -------
    List[str]
        List of backend names (e.g., ['numpy', 'cupy']).
    """
    available = ["numpy"]  # NumPy is always available

    if CUPY_AVAILABLE:
        try:
            # Try to initialize CuPy to verify CUDA is working
            import cupy as cp

            cp.cuda.Device(0).use()
            available.append("cupy")
        except Exception:
            pass

    return available


def get_backend_info() -> Dict[str, any]:
    """
    Get information about available backends and their capabilities.

    Returns
    -------
    Dict[str, any]
        Dictionary with backend information.
    """
    info = {
        "available_backends": list_available_backends(),
        "current_backend": _CURRENT_BACKEND.name if _CURRENT_BACKEND else None,
        "numpy_available": True,
        "cupy_available": CUPY_AVAILABLE,
    }

    if CUPY_AVAILABLE:
        try:
            import cupy as cp

            info["cuda_version"] = cp.cuda.runtime.runtimeGetVersion()
            info["num_devices"] = cp.cuda.runtime.getDeviceCount()

            # Get info for each device
            devices = []
            for i in range(info["num_devices"]):
                device = cp.cuda.Device(i)
                devices.append(
                    {
                        "id": i,
                        "name": device.name,
                        "compute_capability": device.compute_capability,
                        "total_memory_mb": device.mem_info[1] / (1024**2),
                    }
                )
            info["cuda_devices"] = devices
        except Exception as e:
            info["cuda_error"] = str(e)

    return info


def set_backend(backend: str, device_id: int = 0) -> Backend:
    """
    Set the global backend for computations.

    Parameters
    ----------
    backend : str
        Backend name ('numpy' or 'cupy').
    device_id : int, optional
        CUDA device ID for GPU backends. Default is 0.

    Returns
    -------
    Backend
        The initialized backend instance.

    Raises
    ------
    ValueError
        If the requested backend is not available.
    """
    global _CURRENT_BACKEND

    backend = backend.lower()

    if backend == "numpy":
        _CURRENT_BACKEND = NumPyBackend()
    elif backend == "cupy":
        if not CUPY_AVAILABLE:
            raise ValueError(
                "CuPy backend requested but CuPy is not available. "
                "Install with: pip install cupy-cuda12x"
            )
        _CURRENT_BACKEND = CuPyBackend(device_id=device_id)
    else:
        available = list_available_backends()
        raise ValueError(
            f"Unknown backend '{backend}'. Available backends: {available}"
        )

    return _CURRENT_BACKEND


def get_backend(backend: Optional[str] = None, device_id: int = 0) -> Backend:
    """
    Get a backend instance.

    If no backend is specified, returns the current global backend or
    automatically selects the best available backend (GPU preferred).

    Parameters
    ----------
    backend : str, optional
        Backend name ('numpy' or 'cupy'). If None, uses current or auto-detects.
    device_id : int, optional
        CUDA device ID for GPU backends. Default is 0.

    Returns
    -------
    Backend
        The backend instance.
    """
    global _CURRENT_BACKEND

    # If specific backend requested, set and return it
    if backend is not None:
        return set_backend(backend, device_id)

    # If we have a current backend, return it
    if _CURRENT_BACKEND is not None:
        return _CURRENT_BACKEND

    # Auto-detect best backend
    available = list_available_backends()

    # Prefer GPU if available
    if "cupy" in available:
        warnings.warn(
            "No backend specified. Auto-selecting CuPy (GPU) backend. "
            "Set explicitly with set_backend() or get_backend(backend='numpy')",
            UserWarning,
        )
        return set_backend("cupy", device_id)
    else:
        # Fall back to NumPy
        return set_backend("numpy")


def auto_select_backend() -> Backend:
    """
    Automatically select the best available backend.

    Prefers GPU (CuPy) if available, otherwise uses CPU (NumPy).

    Returns
    -------
    Backend
        The selected backend instance.
    """
    return get_backend()


# Initialize with NumPy backend by default
_CURRENT_BACKEND = NumPyBackend()
