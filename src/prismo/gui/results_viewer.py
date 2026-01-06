"""
Results viewer widget for displaying simulation results and plots.

This module provides a GUI widget for visualizing simulation results including
field plots, S-parameters, spectra, and time series using Dear PyGui.
"""

from typing import Any, Optional

try:
    import dearpygui.dearpygui as dpg

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    dpg = None

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from .results_loader import ResultsData


class ResultsViewer:
    """
    Results viewer widget for displaying simulation results.

    Provides interactive visualization of field data, S-parameters, spectra,
    and time series using Dear PyGui's built-in plotting capabilities.
    """

    def __init__(self, font_id: Optional[int] = None):
        """
        Initialize the results viewer.

        Parameters
        ----------
        font_id : int, optional
            Font ID for Unicode/Greek symbol support.
        """
        if not GUI_AVAILABLE:
            raise ImportError(
                "Dear PyGui is required for GUI. Install with: pip install dearpygui"
            )

        self.font_id = font_id
        self.current_data: Optional[ResultsData] = None
        self.plot_tags: dict[str, str] = {}

        # Create UI elements
        self._create_ui()

    def _create_ui(self) -> None:
        """Create the UI elements for the results viewer."""
        # Header
        header_text = dpg.add_text("Results Viewer", color=(255, 255, 255))
        if self.font_id is not None:
            dpg.bind_item_font(header_text, self.font_id)

        dpg.add_separator()

        # Data source selection
        with dpg.group(horizontal=True):
            dpg.add_text("Data Source:")
            self.source_combo = dpg.add_combo(
                items=["File", "Monitor"],
                default_value="File",
                width=150,
                callback=self._on_source_changed,
                tag="results_source_combo",
            )

        dpg.add_separator()

        # Plot type selection
        with dpg.group(horizontal=True):
            dpg.add_text("Plot Type:")
            self.plot_type_combo = dpg.add_combo(
                items=["Auto", "Field Plot", "S-Parameters", "Spectrum", "Time Series"],
                default_value="Auto",
                width=150,
                callback=self._on_plot_type_changed,
                tag="results_plot_type_combo",
            )

        dpg.add_separator()

        # Component selection (for field plots)
        with dpg.group(horizontal=True):
            dpg.add_text("Component:")
            self.component_combo = dpg.add_combo(
                items=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
                default_value="Ex",
                width=100,
                callback=self._update_plot,
                tag="results_component_combo",
            )
            dpg.hide_item("results_component_combo")  # Hidden by default

        dpg.add_separator()

        # Main plot area
        with dpg.plot(
            label="Results Plot",
            height=400,
            width=-1,
            tag="results_plot",
            anti_aliased=True,
        ):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="X", tag="results_plot_x_axis")
            dpg.add_plot_axis(dpg.mvYAxis, label="Y", tag="results_plot_y_axis")
            if self.font_id is not None:
                dpg.bind_item_font("results_plot", self.font_id)

        # Status text
        self.status_text = dpg.add_text("No data loaded", tag="results_status_text", color=(200, 200, 200))

    def load_data(self, data: ResultsData) -> None:
        """
        Load results data for visualization.

        Parameters
        ----------
        data : ResultsData
            Results data to visualize
        """
        self.current_data = data
        self._update_plot()

    def _on_source_changed(self, sender: Any, app_data: Any) -> None:
        """Handle data source change."""
        if app_data == "File":
            dpg.set_value("results_status_text", "Select 'Load Results...' to load from file")
        else:
            dpg.set_value("results_status_text", "Select monitor from simulation")

    def _on_plot_type_changed(self, sender: Any, app_data: Any) -> None:
        """Handle plot type change."""
        # Show/hide component selector based on plot type
        if app_data in ["Field Plot", "Auto"]:
            dpg.show_item("results_component_combo")
        else:
            dpg.hide_item("results_component_combo")

        self._update_plot()

    def _update_plot(self, sender: Any = None, app_data: Any = None) -> None:
        """Update the plot with current data."""
        if self.current_data is None:
            return

        if not NUMPY_AVAILABLE:
            dpg.set_value("results_status_text", "NumPy required for plotting")
            return

        # Clear existing plot series
        for tag in self.plot_tags.values():
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self.plot_tags.clear()

        plot_type = dpg.get_value("results_plot_type_combo")
        data_type = self.current_data.data_type

        # Auto-detect plot type if needed
        if plot_type == "Auto":
            if data_type == "sparameters":
                plot_type = "S-Parameters"
            elif data_type == "spectrum":
                plot_type = "Spectrum"
            elif data_type == "time_series":
                plot_type = "Time Series"
            elif data_type == "field":
                plot_type = "Field Plot"

        try:
            if plot_type == "S-Parameters" and data_type == "sparameters":
                self._plot_sparameters()
            elif plot_type == "Spectrum" and data_type == "spectrum":
                self._plot_spectrum()
            elif plot_type == "Time Series" and data_type == "time_series":
                self._plot_time_series()
            elif plot_type == "Field Plot" and data_type == "field":
                self._plot_field()
            else:
                dpg.set_value("results_status_text", f"Plot type '{plot_type}' not compatible with data type '{data_type}'")
        except Exception as e:
            dpg.set_value("results_status_text", f"Error plotting: {str(e)}")

    def _plot_sparameters(self) -> None:
        """Plot S-parameters."""
        data = self.current_data.data
        frequencies = data["frequencies"]
        sparameters = data["sparameters"]

        # Convert frequency to wavelength in nm for display
        c = 299792458.0  # Speed of light
        wavelengths = (c / frequencies) * 1e9  # Convert to nm

        # Plot magnitude in dB
        dpg.set_axis_labels("results_plot_x_axis", "Wavelength (nm)")
        dpg.set_axis_labels("results_plot_y_axis", "Magnitude (dB)")

        for param_name, s_param in sparameters.items():
            magnitude_db = -20 * np.log10(np.abs(s_param))
            tag = f"results_series_{param_name}"
            dpg.add_line_series(
                wavelengths.tolist(),
                magnitude_db.tolist(),
                label=param_name,
                parent="results_plot_y_axis",
                tag=tag,
            )
            self.plot_tags[param_name] = tag

        dpg.set_value("results_status_text", f"Plotted {len(sparameters)} S-parameters")

    def _plot_spectrum(self) -> None:
        """Plot frequency spectrum."""
        data = self.current_data.data
        frequencies = data["frequencies"]
        spectrum = data["spectrum"]

        # Convert frequency to wavelength in nm for display
        c = 299792458.0
        wavelengths = (c / frequencies) * 1e9

        dpg.set_axis_labels("results_plot_x_axis", "Wavelength (nm)")
        dpg.set_axis_labels("results_plot_y_axis", "Power / Transmission")

        tag = "results_series_spectrum"
        dpg.add_line_series(
            wavelengths.tolist(),
            np.abs(spectrum).tolist(),
            label="Spectrum",
            parent="results_plot_y_axis",
            tag=tag,
        )
        self.plot_tags["spectrum"] = tag

        dpg.set_value("results_status_text", "Plotted spectrum")

    def _plot_time_series(self) -> None:
        """Plot time series data."""
        data = self.current_data.data
        time = data["time"]
        field = data["field"]
        component = data.get("component", "Field")

        # For 3D field data, extract a slice or average
        if len(field.shape) > 1:
            # Take middle slice or average
            if len(field.shape) == 3:
                # 3D: take middle z-slice
                mid_z = field.shape[2] // 2
                field_1d = field[:, :, mid_z].mean(axis=1)  # Average over y
            elif len(field.shape) == 2:
                # 2D: average over y
                field_1d = field.mean(axis=1)
            else:
                field_1d = field.flatten()
        else:
            field_1d = field

        # Convert time to femtoseconds for display
        time_fs = time * 1e15

        dpg.set_axis_labels("results_plot_x_axis", "Time (fs)")
        dpg.set_axis_labels("results_plot_y_axis", f"{component} (V/m or A/m)")

        tag = "results_series_time"
        dpg.add_line_series(
            time_fs.tolist(),
            field_1d.tolist(),
            label=component,
            parent="results_plot_y_axis",
            tag=tag,
        )
        self.plot_tags["time"] = tag

        dpg.set_value("results_status_text", f"Plotted time series for {component}")

    def _plot_field(self) -> None:
        """Plot 2D field data."""
        data = self.current_data.data
        component = dpg.get_value("results_component_combo")

        # Get field data
        if "fields" in data:
            fields = data["fields"]
            if component not in fields:
                dpg.set_value("results_status_text", f"Component {component} not available")
                return

            field_data = fields[component]
        elif "field" in data:
            field_data = data["field"]
        else:
            dpg.set_value("results_status_text", "No field data available")
            return

        # Handle 2D field data
        if len(field_data.shape) == 2:
            # 2D field - create heatmap using line series for each row
            # (Dear PyGui doesn't have native heatmap, so we'll use a workaround)
            ny, nx = field_data.shape
            x_coords = np.linspace(0, 1, nx)  # Normalized coordinates
            y_coords = np.linspace(0, 1, ny)

            # Plot as multiple line series (one per row) - not ideal but works
            # For better visualization, we'd need to use image widget
            dpg.set_axis_labels("results_plot_x_axis", "X (normalized)")
            dpg.set_axis_labels("results_plot_y_axis", f"{component} (V/m or A/m)")

            # Plot a few representative rows
            step = max(1, ny // 20)  # Plot every Nth row
            for i in range(0, ny, step):
                tag = f"results_series_field_row_{i}"
                dpg.add_line_series(
                    x_coords.tolist(),
                    field_data[i, :].tolist(),
                    label=f"Row {i}",
                    parent="results_plot_y_axis",
                    tag=tag,
                )
                self.plot_tags[f"row_{i}"] = tag

            dpg.set_value("results_status_text", f"Plotted 2D field {component} (showing {ny//step} rows)")
        elif len(field_data.shape) == 1:
            # 1D field
            x_coords = np.linspace(0, 1, len(field_data))
            dpg.set_axis_labels("results_plot_x_axis", "X (normalized)")
            dpg.set_axis_labels("results_plot_y_axis", f"{component} (V/m or A/m)")

            tag = "results_series_field_1d"
            dpg.add_line_series(
                x_coords.tolist(),
                field_data.tolist(),
                label=component,
                parent="results_plot_y_axis",
                tag=tag,
            )
            self.plot_tags["field"] = tag

            dpg.set_value("results_status_text", f"Plotted 1D field {component}")
        else:
            dpg.set_value("results_status_text", f"Unsupported field dimension: {field_data.shape}")

    def clear(self) -> None:
        """Clear the current plot and data."""
        for tag in self.plot_tags.values():
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        self.plot_tags.clear()
        self.current_data = None
        dpg.set_value("results_status_text", "No data loaded")

