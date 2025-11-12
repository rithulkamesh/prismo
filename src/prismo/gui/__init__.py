"""
Desktop GUI for Prismo electromagnetic simulation tool.

This module provides a PySide6-based graphical user interface for:
- Material visualization (3D geometry and property plots)
- Simulation setup and control
- Low-code simulation configuration
"""

try:
    from PySide6 import QtCore, QtGui, QtWidgets

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    QtCore = None
    QtGui = None
    QtWidgets = None

__all__ = ["GUI_AVAILABLE"]

if GUI_AVAILABLE:
    from .main_window import MainWindow

    __all__.append("MainWindow")
