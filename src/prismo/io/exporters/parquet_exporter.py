"""
Parquet data exporter using Polars.

This module provides functionality to export simulation results to Apache Parquet
format using Polars for efficient columnar storage and fast I/O.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np

try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

from .base import Exporter


class ParquetExporter(Exporter):
    """
    Parquet data exporter using Polars.

    Exports field data, spectra, and S-parameters to Apache Parquet format.
    Parquet provides efficient columnar storage with compression, making it
    ideal for large datasets.

    Parameters
    ----------
    output_dir : Path, optional
        Output directory for exported files.
    compression : str, optional
        Compression algorithm ('snappy', 'gzip', 'lz4', 'zstd'). Default='snappy'.
    """

    def __init__(self, output_dir: Optional[Path] = None, compression: str = "snappy"):
        if not POLARS_AVAILABLE:
            raise ImportError(
                "Polars is required for Parquet export. Install with: pip install polars"
            )

        super().__init__(output_dir)
        self.compression = compression

    def export_fields(
        self,
        filename: str,
        fields: Dict[str, Any],
        coordinates: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export field data to Parquet format.

        Creates a Parquet file with efficient columnar storage.

        Parameters
        ----------
        filename : str
            Output filename (without .parquet extension).
        fields : dict
            Dictionary with field arrays.
        coordinates : dict
            Dictionary with coordinate arrays.
        metadata : dict, optional
            Additional metadata (stored in Parquet metadata).

        Returns
        -------
        Path
            Path to exported Parquet file.
        """
        output_path = self.output_dir / f"{filename}.parquet"

        # Convert to numpy arrays
        fields_np = {k: np.asarray(v) for k, v in fields.items()}
        coords_np = {k: np.asarray(v) for k, v in coordinates.items()}

        # Create coordinate meshgrid if needed
        if len(coords_np["x"].shape) == 1:
            x, y, z = np.meshgrid(
                coords_np.get("x", [0]),
                coords_np.get("y", [0]),
                coords_np.get("z", [0]),
                indexing="ij",
            )
        else:
            x, y, z = coords_np["x"], coords_np.get("y", 0), coords_np.get("z", 0)

        # Flatten arrays
        x_flat = x.ravel()
        y_flat = y.ravel() if hasattr(y, "ravel") else np.full_like(x_flat, y)
        z_flat = z.ravel() if hasattr(z, "ravel") else np.full_like(x_flat, z)

        # Prepare field data
        data = {
            "x": x_flat,
            "y": y_flat,
            "z": z_flat,
        }

        # Add field components
        for comp in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            if comp in fields_np:
                data[comp] = fields_np[comp].ravel()
            else:
                data[comp] = np.zeros_like(x_flat)

        # Add field magnitudes
        data["E_magnitude"] = np.sqrt(
            data["Ex"] ** 2 + data["Ey"] ** 2 + data["Ez"] ** 2
        )
        data["H_magnitude"] = np.sqrt(
            data["Hx"] ** 2 + data["Hy"] ** 2 + data["Hz"] ** 2
        )

        # Create Polars DataFrame
        df = pl.DataFrame(data)

        # Write to Parquet with compression
        df.write_parquet(
            output_path,
            compression=self.compression,
        )

        # Write metadata as separate JSON file if provided
        if metadata:
            import json

            meta_path = output_path.with_suffix(".meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return output_path

    def export_spectrum(
        self,
        filename: str,
        frequencies: Any,
        spectrum: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export spectrum data to Parquet.

        Parameters
        ----------
        filename : str
            Output filename.
        frequencies : array
            Frequency values.
        spectrum : array
            Spectrum data (real or complex).
        metadata : dict, optional
            Additional metadata.

        Returns
        -------
        Path
            Path to exported Parquet file.
        """
        output_path = self.output_dir / f"{filename}.parquet"

        freq = np.asarray(frequencies)
        spec = np.asarray(spectrum)

        # Prepare data
        data = {"frequency_Hz": freq}

        if np.iscomplexobj(spec):
            data["magnitude"] = np.abs(spec)
            data["phase_rad"] = np.angle(spec)
            data["real"] = np.real(spec)
            data["imag"] = np.imag(spec)
        else:
            data["spectrum"] = spec

        # Create DataFrame and write
        df = pl.DataFrame(data)
        df.write_parquet(output_path, compression=self.compression)

        if metadata:
            import json

            meta_path = output_path.with_suffix(".meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return output_path

    def export_sparameters(
        self,
        filename: str,
        frequencies: Any,
        sparameters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export S-parameters to Parquet.

        Parameters
        ----------
        filename : str
            Output filename.
        frequencies : array
            Frequency values.
        sparameters : dict
            Dictionary of S-parameters.
        metadata : dict, optional
            Additional metadata.

        Returns
        -------
        Path
            Path to exported Parquet file.
        """
        output_path = self.output_dir / f"{filename}.parquet"

        freq = np.asarray(frequencies)

        # Prepare data
        data = {"frequency_Hz": freq}

        # Add each S-parameter
        for param_name, param_values in sparameters.items():
            param_array = np.asarray(param_values)

            # Store magnitude and phase
            data[f"{param_name}_magnitude"] = np.abs(param_array)
            data[f"{param_name}_phase_deg"] = np.angle(param_array, deg=True)

            # Also store real and imaginary parts for exact reconstruction
            data[f"{param_name}_real"] = np.real(param_array)
            data[f"{param_name}_imag"] = np.imag(param_array)

        # Create DataFrame and write
        df = pl.DataFrame(data)
        df.write_parquet(output_path, compression=self.compression)

        if metadata:
            import json

            meta_path = output_path.with_suffix(".meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return output_path

    def export_timeseries(
        self,
        filename: str,
        time: Any,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Export time-series data to Parquet.

        Parameters
        ----------
        filename : str
            Output filename.
        time : array
            Time values.
        data : dict
            Dictionary of time-series data.
        metadata : dict, optional
            Additional metadata.

        Returns
        -------
        Path
            Path to exported Parquet file.
        """
        output_path = self.output_dir / f"{filename}.parquet"

        # Prepare data
        df_data = {"time": np.asarray(time)}

        for key, values in data.items():
            df_data[key] = np.asarray(values)

        # Create DataFrame and write
        df = pl.DataFrame(df_data)
        df.write_parquet(output_path, compression=self.compression)

        if metadata:
            import json

            meta_path = output_path.with_suffix(".meta.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return output_path

    @staticmethod
    def read_parquet(filepath: Path) -> pl.DataFrame:
        """
        Read a Parquet file.

        Parameters
        ----------
        filepath : Path
            Path to Parquet file.

        Returns
        -------
        pl.DataFrame
            Polars DataFrame with the data.
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars is required to read Parquet files")

        return pl.read_parquet(filepath)

    @staticmethod
    def read_with_filter(
        filepath: Path,
        filter_expr: Optional[Any] = None,
        columns: Optional[list] = None,
    ) -> pl.DataFrame:
        """
        Read Parquet file with lazy evaluation and filtering.

        Parameters
        ----------
        filepath : Path
            Path to Parquet file.
        filter_expr : polars expression, optional
            Filter expression to apply.
        columns : list, optional
            Specific columns to read.

        Returns
        -------
        pl.DataFrame
            Filtered DataFrame.
        """
        if not POLARS_AVAILABLE:
            raise ImportError("Polars is required")

        lf = pl.scan_parquet(filepath)

        if columns is not None:
            lf = lf.select(columns)

        if filter_expr is not None:
            lf = lf.filter(filter_expr)

        return lf.collect()
