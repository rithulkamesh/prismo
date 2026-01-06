"""
Desktop GUI for Prismo electromagnetic simulation tool.

This module provides a Dear PyGui-based graphical user interface for:
- Material visualization (3D geometry and property plots)
- Simulation setup and control
- Low-code simulation configuration
"""

# Module-level variables for error tracking
_import_error: Exception | None = None
_missing_library_error: bool = False

try:
    import dearpygui.dearpygui as dpg

    GUI_AVAILABLE = True
except ImportError as e:
    GUI_AVAILABLE = False
    dpg = None
    # Store the error for better diagnostics
    _import_error = e
    # Check if it's a missing system library issue
    if "libX11" in str(e) or "cannot open shared object file" in str(e):
        _missing_library_error = True
    else:
        _missing_library_error = False
except Exception as e:
    # Catch other errors (like missing system libraries)
    GUI_AVAILABLE = False
    dpg = None
    _import_error = e
    if "libX11" in str(e) or "cannot open shared object file" in str(e):
        _missing_library_error = True
    else:
        _missing_library_error = False

__all__ = ["GUI_AVAILABLE", "get_gui_error_message"]

# Always try to import MainWindow for type checking, but handle ImportError gracefully
try:
    from .main_window import MainWindow

    if GUI_AVAILABLE:
        __all__.append("MainWindow")
except ImportError:
    MainWindow = None  # type: ignore


def get_gui_error_message() -> str:
    """Get a helpful error message explaining why GUI is not available."""
    if GUI_AVAILABLE:
        return ""
    
    if _import_error is None:
        return "GUI is not available. Install with: pip install dearpygui"
    
    if _missing_library_error:
        return (
            f"GUI dependencies are installed but system libraries are missing.\n"
            f"Error: {_import_error}\n\n"
            f"On NixOS: Use 'nix develop .#gui' to enter the GUI development shell.\n"
            f"  (Works on both X11 and Wayland - XWayland provides compatibility)\n"
            f"On other systems: Install X11 development libraries:\n"
            f"  - Ubuntu/Debian: sudo apt-get install libx11-dev libxext-dev libgl1-mesa-dev\n"
            f"  - Fedora: sudo dnf install libX11-devel libXext-devel mesa-libGL-devel\n"
            f"  - Wayland: X11 libraries work via XWayland (usually pre-installed)\n"
            f"  - macOS: X11 libraries are usually pre-installed\n"
        )
    
    return f"Error importing GUI: {_import_error}\nInstall GUI dependencies with: pip install dearpygui"
