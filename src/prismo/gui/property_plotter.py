"""
Material property plotting widget.

This module provides widgets for plotting material properties as a function
of frequency, including dispersion curves and loss tangents.
"""

from typing import Optional

try:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    Qt = None
    QWidget = object

# Try to import plotting libraries
try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    FigureCanvas = None
    Figure = None

try:
    import pyqtgraph as pg

    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    pg = None


class PropertyPlotter(QWidget):
    """
    Material property plotting widget.

    Provides plots for frequency-dependent material properties including
    permittivity, permeability, refractive index, and loss tangents.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        """Initialize the property plotter."""
        if not GUI_AVAILABLE:
            raise ImportError(
                "PySide6 is required for GUI. Install with: pip install PySide6"
            )

        super().__init__(parent)
        self.setWindowTitle("Material Property Plotter")

        # Create layout
        layout = QVBoxLayout(self)

        # Add header
        header = QLabel("Material Properties")
        header.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(header)

        # Add property selection
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Property:"))
        self.property_combo = QComboBox()
        self.property_combo.addItems(
            [
                "Permittivity (ε)",
                "Permeability (μ)",
                "Refractive Index (n)",
                "Loss Tangent",
            ]
        )
        self.property_combo.currentTextChanged.connect(self._update_plot)
        controls_layout.addWidget(self.property_combo)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Create plot widget
        if MATPLOTLIB_AVAILABLE:
            self._setup_matplotlib_plot(layout)
        elif PYQTGRAPH_AVAILABLE:
            self._setup_pyqtgraph_plot(layout)
        else:
            placeholder = QLabel(
                "Plotting requires Matplotlib or PyQtGraph.\n"
                "Install with: pip install matplotlib"
            )
            placeholder.setAlignment(Qt.AlignCenter)
            layout.addWidget(placeholder)

    def _setup_matplotlib_plot(self, layout: QVBoxLayout) -> None:
        """Set up Matplotlib-based plotting."""
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Property Value")
        self.ax.grid(True)
        layout.addWidget(self.canvas)

    def _setup_pyqtgraph_plot(self, layout: QVBoxLayout) -> None:
        """Set up PyQtGraph-based plotting."""
        # PyQtGraph integration would go here
        placeholder = QLabel("PyQtGraph plot (to be implemented)")
        placeholder.setMinimumHeight(200)
        layout.addWidget(placeholder)

    def _update_plot(self, property_name: str) -> None:
        """
        Update the plot based on selected property.

        Parameters
        ----------
        property_name : str
            Name of the property to plot.
        """
        if MATPLOTLIB_AVAILABLE and hasattr(self, "ax"):
            self.ax.clear()
            self.ax.set_xlabel("Frequency (Hz)")
            self.ax.set_ylabel(property_name)
            self.ax.grid(True)
            # TODO: Plot actual material data
            self.canvas.draw()

    def plot_material(self, material, frequency_range: tuple[float, float]) -> None:
        """
        Plot material properties over frequency range.

        Parameters
        ----------
        material
            Material object to plot.
        frequency_range : tuple
            (min_frequency, max_frequency) in Hz.
        """
        # This would calculate and plot material properties
        # For now, placeholder
        pass
