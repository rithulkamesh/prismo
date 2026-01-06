"""
Material property plotting widget.

This module provides widgets for plotting material properties as a function
of frequency, including dispersion curves and loss tangents using Dear PyGui.
"""

import os
from typing import Optional

try:
    import dearpygui.dearpygui as dpg

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    dpg = None

# Try to import plotting libraries
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


# Module-level font ID for Greek symbol support
_UNICODE_FONT_ID: Optional[int] = None


def _find_font_with_fontconfig() -> Optional[str]:
    """
    Try to find a font that supports Greek characters using fontconfig.
    
    Returns
    -------
    str or None
        Path to a font file, or None if not found.
    """
    import subprocess
    
    try:
        # Try to find DejaVu Sans using fc-match
        result = subprocess.run(
            ["fc-match", "-f", "%{file}", "DejaVu Sans"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            font_path = result.stdout.strip()
            if os.path.exists(font_path):
                return font_path
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    try:
        # Try to find any sans-serif font that supports Greek
        result = subprocess.run(
            ["fc-match", "-f", "%{file}", "sans-serif"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            font_path = result.stdout.strip()
            if os.path.exists(font_path):
                return font_path
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    return None


def load_unicode_font(font_registry: Optional[int] = None) -> Optional[int]:
    """
    Load a font that supports Unicode symbols, especially Greek characters.
    
    Tries multiple methods to find a suitable font:
    1. Uses fontconfig (Linux) to find system fonts
    2. Tries common system font paths
    3. Falls back to default font
    
    Should be called once during GUI initialization.
    
    Parameters
    ----------
    font_registry : int, optional
        Font registry ID. If None, creates a new registry.
    
    Returns
    -------
    int or None
        Font ID if successful, None otherwise.
    """
    global _UNICODE_FONT_ID
    
    if not GUI_AVAILABLE:
        return None
    
    if _UNICODE_FONT_ID is not None:
        return _UNICODE_FONT_ID
    
    font_paths = []
    
    # Method 1: Try fontconfig (Linux)
    fontconfig_font = _find_font_with_fontconfig()
    if fontconfig_font:
        font_paths.append(fontconfig_font)
    
    # Method 2: Common system font paths (ordered by likelihood of supporting Greek)
    font_paths.extend([
        # Linux - DejaVu fonts (excellent Unicode support)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        # Linux - Liberation fonts
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        # Linux - Noto fonts (excellent Unicode support)
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        # Linux - Other common locations
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/arphic/ukai.ttc",
        # macOS - Unicode fonts
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/ARIALUNI.TTF",  # Arial Unicode MS
        # NixOS/other Linux distributions
        "/nix/store/*/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ])
    
    # Also try to find DejaVu in common locations using glob
    try:
        import glob
        dejavu_patterns = [
            "/usr/share/fonts/**/DejaVuSans*.ttf",
            "/usr/share/fonts/**/*DejaVu*.ttf",
            "/nix/store/*/share/fonts/**/DejaVuSans*.ttf",
        ]
        for pattern in dejavu_patterns:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                font_paths.extend(matches[:2])  # Add first 2 matches
                break
    except Exception:
        pass
    
    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in font_paths:
        if path and path not in seen:
            seen.add(path)
            unique_paths.append(path)
    
    # Try to load a font
    for font_path in unique_paths:
        if not font_path or not os.path.exists(font_path):
            continue
            
        try:
            # Load font with default size (13px is Dear PyGui default)
            # Dear PyGui automatically supports Unicode if the font supports it
            if font_registry is not None:
                font_id = dpg.add_font(font_path, 13, parent=font_registry)
            else:
                with dpg.font_registry() as reg_id:
                    font_id = dpg.add_font(font_path, 13, parent=reg_id)
            
            # Store and return the font ID
            _UNICODE_FONT_ID = font_id
            return font_id
        except Exception as e:
            # If loading fails, try next font
            continue
    
    # If no system font found, return None
    # Dear PyGui's default font might support basic Unicode on some systems
    return None


def get_unicode_font() -> Optional[int]:
    """Get the loaded Unicode font ID, or None if not loaded."""
    return _UNICODE_FONT_ID


class PropertyPlotter:
    """
    Material property plotting widget.

    Provides plots for frequency-dependent material properties including
    permittivity, permeability, refractive index, and loss tangents using Dear PyGui.
    """

    def __init__(self, font_id: Optional[int] = None):
        """
        Initialize the property plotter.
        
        Parameters
        ----------
        font_id : int, optional
            Font ID for Unicode/Greek symbol support. If None, tries to get
            the global Unicode font.
        """
        if not GUI_AVAILABLE:
            raise ImportError(
                "Dear PyGui is required for GUI. Install with: pip install dearpygui"
            )

        # Use provided font or get the global Unicode font
        self.font_id = font_id if font_id is not None else get_unicode_font()

        # Create header
        header_text = dpg.add_text("Material Properties", color=(255, 255, 255))
        if self.font_id is not None:
            dpg.bind_item_font(header_text, self.font_id)

        # Add property selection
        with dpg.group(horizontal=True):
            prop_label = dpg.add_text("Property:")
            if self.font_id is not None:
                dpg.bind_item_font(prop_label, self.font_id)
            
            self.property_combo = dpg.add_combo(
                items=[
                    "Permittivity (ε)",
                    "Permeability (μ)",
                    "Refractive Index (n)",
                    "Loss Tangent",
                ],
                default_value="Permittivity (ε)",
                width=200,
                callback=self._update_plot,
            )
            if self.font_id is not None:
                dpg.bind_item_font(self.property_combo, self.font_id)

        dpg.add_separator()

        # Create plot widget using Dear PyGui's built-in plotting
        if not NUMPY_AVAILABLE:
            dpg.add_text("NumPy required for plotting", color=(200, 200, 200))
            dpg.add_separator()
        
        with dpg.plot(label="Material Property", height=300, width=-1, tag="property_plot"):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="Frequency (Hz)", tag="property_plot_x_axis")
            dpg.add_plot_axis(dpg.mvYAxis, label="Property Value", tag="property_plot_y_axis")
            # Apply Unicode font to plot if available (affects all plot text including axes)
            if self.font_id is not None:
                dpg.bind_item_font("property_plot", self.font_id)

            # Placeholder plot data
            if NUMPY_AVAILABLE:
                # Create sample data for demonstration
                x_data = np.linspace(1e9, 1e12, 100)  # 1 GHz to 1 THz
                y_data = np.ones_like(x_data)  # Placeholder: constant value
                dpg.add_line_series(
                    x_data.tolist(),
                    y_data.tolist(),
                    label="Property",
                    parent="property_plot_y_axis",
                    tag="property_plot_series",
                )

    def _update_plot(self, sender, app_data) -> None:
        """
        Update the plot based on selected property.

        Parameters
        ----------
        sender : int
            Sender ID (Dear PyGui internal).
        app_data : str
            Selected property name.
        """
        # This would update the plot with actual material data
        # For now, just update the y-axis label
        dpg.set_axis_labels("property_plot_y_axis", app_data)

    def show(self) -> None:
        """Show the property plotter (already visible in main window)."""
        pass

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
