"""
CSV data exporter.

This module provides functionality to export simulation results to CSV format.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import csv

from .base import Exporter


class CSVExporter(Exporter):
    """
    CSV data exporter.

    Exports field data, spectra, and S-parameters to CSV format.
    Suitable for small to medium datasets that can be easily opened in
    spreadsheet applications.
    """

    def export_fields(
        self,
        filename: str,
        fields: Dict[str, Any],
        coordinates: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export field data to CSV.

        Creates a CSV file with columns: x, y, z, Ex, Ey, Ez, Hx, Hy, Hz, |E|, |H|

        Parameters
        ----------
        filename : str
            Output filename (without .csv extension).
        fields : dict
            Dictionary with keys 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz'.
        coordinates : dict
            Dictionary with keys 'x', 'y', 'z' (coordinate arrays).
        metadata : dict, optional
            Additional metadata to write as header comments.

        Returns
        -------
        Path
            Path to exported CSV file.
        """
        output_path = self.output_dir / f"{filename}.csv"

        # Convert all arrays to numpy
        fields_np = {k: np.asarray(v) for k, v in fields.items()}
        coords_np = {k: np.asarray(v) for k, v in coordinates.items()}

        # Create meshgrid of coordinates
        if len(coords_np["x"].shape) == 1:
            # 1D arrays - create meshgrid
            x, y, z = np.meshgrid(
                coords_np.get("x", [0]),
                coords_np.get("y", [0]),
                coords_np.get("z", [0]),
                indexing="ij",
            )
        else:
            # Already gridded
            x, y, z = coords_np["x"], coords_np.get("y", 0), coords_np.get("z", 0)

        # Flatten all arrays
        x_flat = x.ravel()
        y_flat = y.ravel() if hasattr(y, "ravel") else np.full_like(x_flat, y)
        z_flat = z.ravel() if hasattr(z, "ravel") else np.full_like(x_flat, z)

        # Flatten field arrays
        field_components = {}
        for comp in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            if comp in fields_np:
                field_components[comp] = fields_np[comp].ravel()
            else:
                field_components[comp] = np.zeros_like(x_flat)

        # Compute field magnitudes
        E_mag = np.sqrt(
            field_components["Ex"] ** 2
            + field_components["Ey"] ** 2
            + field_components["Ez"] ** 2
        )
        H_mag = np.sqrt(
            field_components["Hx"] ** 2
            + field_components["Hy"] ** 2
            + field_components["Hz"] ** 2
        )

        # Write CSV file
        with open(output_path, "w", newline="") as f:
            # Write metadata as comments
            if metadata:
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")

            # Write header
            writer = csv.writer(f)
            writer.writerow(
                ["x", "y", "z", "Ex", "Ey", "Ez", "Hx", "Hy", "Hz", "|E|", "|H|"]
            )

            # Write data rows
            for i in range(len(x_flat)):
                writer.writerow(
                    [
                        x_flat[i],
                        y_flat[i],
                        z_flat[i],
                        field_components["Ex"][i],
                        field_components["Ey"][i],
                        field_components["Ez"][i],
                        field_components["Hx"][i],
                        field_components["Hy"][i],
                        field_components["Hz"][i],
                        E_mag[i],
                        H_mag[i],
                    ]
                )

        return output_path

    def export_spectrum(
        self,
        filename: str,
        frequencies: Any,
        spectrum: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export spectrum data to CSV.

        Creates a CSV file with columns: frequency, spectrum, (optional phase).

        Parameters
        ----------
        filename : str
            Output filename.
        frequencies : array
            Frequency values (Hz).
        spectrum : array
            Spectrum data (can be real or complex).
        metadata : dict, optional
            Additional metadata.

        Returns
        -------
        Path
            Path to exported CSV file.
        """
        output_path = self.output_dir / f"{filename}.csv"

        freq = np.asarray(frequencies)
        spec = np.asarray(spectrum)

        with open(output_path, "w", newline="") as f:
            # Write metadata
            if metadata:
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")

            writer = csv.writer(f)

            # Determine if spectrum is complex
            if np.iscomplexobj(spec):
                writer.writerow(["frequency_Hz", "magnitude", "phase_rad"])
                for i in range(len(freq)):
                    writer.writerow([freq[i], np.abs(spec[i]), np.angle(spec[i])])
            else:
                writer.writerow(["frequency_Hz", "spectrum"])
                for i in range(len(freq)):
                    writer.writerow([freq[i], spec[i]])

        return output_path

    def export_sparameters(
        self,
        filename: str,
        frequencies: Any,
        sparameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export S-parameters to CSV.

        Creates a CSV file with columns for each S-parameter (magnitude and phase).

        Parameters
        ----------
        filename : str
            Output filename.
        frequencies : array
            Frequency values (Hz).
        sparameters : dict
            Dictionary of S-parameters (e.g., {'S11': array, 'S21': array}).
        metadata : dict, optional
            Additional metadata.

        Returns
        -------
        Path
            Path to exported CSV file.
        """
        output_path = self.output_dir / f"{filename}.csv"

        freq = np.asarray(frequencies)

        # Convert S-parameters to numpy arrays
        s_params = {k: np.asarray(v) for k, v in sparameters.items()}

        with open(output_path, "w", newline="") as f:
            # Write metadata
            if metadata:
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")

            writer = csv.writer(f)

            # Build header
            header = ["frequency_Hz"]
            for param_name in sorted(s_params.keys()):
                header.extend([f"{param_name}_mag", f"{param_name}_phase_deg"])
            writer.writerow(header)

            # Write data
            for i in range(len(freq)):
                row = [freq[i]]
                for param_name in sorted(s_params.keys()):
                    param_value = s_params[param_name][i]
                    row.append(np.abs(param_value))
                    row.append(np.angle(param_value, deg=True))
                writer.writerow(row)

        return output_path

    def export_field_slice(
        self,
        filename: str,
        field_slice: np.ndarray,
        extent: tuple,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export a 2D field slice to CSV.

        Parameters
        ----------
        filename : str
            Output filename.
        field_slice : ndarray
            2D array of field values.
        extent : tuple
            Physical extent (xmin, xmax, ymin, ymax).
        metadata : dict, optional
            Additional metadata.

        Returns
        -------
        Path
            Path to exported CSV file.
        """
        output_path = self.output_dir / f"{filename}.csv"

        xmin, xmax, ymin, ymax = extent
        ny, nx = field_slice.shape

        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(x, y)

        with open(output_path, "w", newline="") as f:
            if metadata:
                for key, value in metadata.items():
                    f.write(f"# {key}: {value}\n")

            writer = csv.writer(f)
            writer.writerow(["x", "y", "value"])

            for i in range(ny):
                for j in range(nx):
                    writer.writerow([X[i, j], Y[i, j], field_slice[i, j]])

        return output_path
