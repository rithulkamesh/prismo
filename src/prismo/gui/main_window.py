"""
Main application window for Prismo GUI.

This module provides the main window with menu structure, toolbars, and
layout for material visualization and simulation control.
"""

from typing import Optional

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QMainWindow,
        QMenuBar,
        QStatusBar,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    QMainWindow = object
    QWidget = object

if GUI_AVAILABLE:
    from .material_viewer import MaterialViewer
    from .property_plotter import PropertyPlotter


class MainWindow(QMainWindow):
    """
    Main application window for Prismo GUI.

    Provides the main interface with menus, toolbars, and layout for
    material visualization and simulation setup.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the main window."""
        if not GUI_AVAILABLE:
            raise ImportError(
                "PySide6 is required for GUI. Install with: pip install PySide6"
            )

        super().__init__(parent)
        self.setWindowTitle("Prismo - Electromagnetic Simulation Tool")
        self.setMinimumSize(1200, 800)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create main components
        self.material_viewer = MaterialViewer(self)
        self.property_plotter = PropertyPlotter(self)

        # Add components to layout
        layout.addWidget(self.material_viewer, 2)  # 2/3 of space
        layout.addWidget(self.property_plotter, 1)  # 1/3 of space

        # Create menu bar
        self._create_menu_bar()

        # Create toolbar
        self._create_toolbar()

        # Create status bar
        self.statusBar().showMessage("Ready")

    def _create_menu_bar(self) -> None:
        """Create the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")
        file_menu.addAction("&New Simulation", self._new_simulation)
        file_menu.addAction("&Open...", self._open_simulation)
        file_menu.addAction("&Save", self._save_simulation)
        file_menu.addAction("Save &As...", self._save_as_simulation)
        file_menu.addSeparator()
        file_menu.addAction("E&xit", self.close)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction("&Preferences...", self._show_preferences)

        # View menu
        view_menu = menubar.addMenu("&View")
        view_menu.addAction("&Material Viewer", self.material_viewer.show)
        view_menu.addAction("&Property Plotter", self.property_plotter.show)

        # Simulation menu
        sim_menu = menubar.addMenu("&Simulation")
        sim_menu.addAction("&Run", self._run_simulation)
        sim_menu.addAction("&Stop", self._stop_simulation)
        sim_menu.addAction("&Reset", self._reset_simulation)

        # Help menu
        help_menu = menubar.addMenu("&Help")
        help_menu.addAction("&About", self._show_about)
        help_menu.addAction("&Documentation", self._show_documentation)

    def _create_toolbar(self) -> None:
        """Create the application toolbar."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Add toolbar actions
        toolbar.addAction("New", self._new_simulation)
        toolbar.addAction("Open", self._open_simulation)
        toolbar.addAction("Save", self._save_simulation)
        toolbar.addSeparator()
        toolbar.addAction("Run", self._run_simulation)
        toolbar.addAction("Stop", self._stop_simulation)

    def _new_simulation(self) -> None:
        """Create a new simulation."""
        self.statusBar().showMessage("New simulation...", 2000)

    def _open_simulation(self) -> None:
        """Open an existing simulation."""
        self.statusBar().showMessage("Open simulation...", 2000)

    def _save_simulation(self) -> None:
        """Save the current simulation."""
        self.statusBar().showMessage("Save simulation...", 2000)

    def _save_as_simulation(self) -> None:
        """Save the current simulation with a new name."""
        self.statusBar().showMessage("Save simulation as...", 2000)

    def _show_preferences(self) -> None:
        """Show preferences dialog."""
        self.statusBar().showMessage("Preferences...", 2000)

    def _run_simulation(self) -> None:
        """Run the simulation."""
        self.statusBar().showMessage("Running simulation...", 2000)

    def _stop_simulation(self) -> None:
        """Stop the running simulation."""
        self.statusBar().showMessage("Stopping simulation...", 2000)

    def _reset_simulation(self) -> None:
        """Reset the simulation."""
        self.statusBar().showMessage("Resetting simulation...", 2000)

    def _show_about(self) -> None:
        """Show about dialog."""
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "About Prismo",
            "Prismo - Electromagnetic Simulation Tool\n\n"
            "A high-performance Python-based FDTD/FEM solver for photonics.",
        )

    def _show_documentation(self) -> None:
        """Open documentation in browser."""
        import webbrowser

        webbrowser.open("https://prismo.readthedocs.io")
