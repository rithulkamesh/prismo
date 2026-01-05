"""
Results loader for GUI visualization.

This module provides functions to load simulation results from various sources:
- CSV files (exported by CSVExporter)
- Parquet files (exported by ParquetExporter)
- Live simulation monitors
"""

from pathlib import Path
from typing import Any, Optional, Union

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Try to import polars for Parquet support
try:
    import polars as pl

    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None


class ResultsData:
    """Container for loaded simulation results data."""

    def __init__(
        self,
        data_type: str,
        data: dict[str, Any],
        metadata: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize results data container.

        Parameters
        ----------
        data_type : str
            Type of data: 'field', 'spectrum', 'sparameters', 'time_series'
        data : dict
            Dictionary containing the actual data arrays
        metadata : dict, optional
            Additional metadata about the data
        """
        self.data_type = data_type
        self.data = data
        self.metadata = metadata or {}


def load_csv_spectrum(filepath: Union[str, Path]) -> ResultsData:
    """
    Load spectrum data from CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file

    Returns
    -------
    ResultsData
        Loaded spectrum data
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("NumPy is required to load CSV files")

    filepath = Path(filepath)
    metadata = {}
    frequencies = []
    spectrum = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                # Parse metadata
                if ":" in line:
                    key, value = line[1:].split(":", 1)
                    metadata[key.strip()] = value.strip()
            elif line and not line.startswith("frequency"):
                # Data line
                parts = line.split(",")
                if len(parts) >= 2:
                    frequencies.append(float(parts[0]))
                    spectrum.append(float(parts[1]))

    return ResultsData(
        data_type="spectrum",
        data={
            "frequencies": np.array(frequencies),
            "spectrum": np.array(spectrum),
        },
        metadata=metadata,
    )


def load_csv_sparameters(filepath: Union[str, Path]) -> ResultsData:
    """
    Load S-parameters from CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file

    Returns
    -------
    ResultsData
        Loaded S-parameter data
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("NumPy is required to load CSV files")

    filepath = Path(filepath)
    metadata = {}
    frequencies = []
    sparameters: dict[str, list[complex]] = {}

    with open(filepath, "r") as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                # Parse metadata
                if ":" in line:
                    key, value = line[1:].split(":", 1)
                    metadata[key.strip()] = value.strip()
            elif line.startswith("frequency"):
                # Header line
                header = line.split(",")
                # Initialize sparameter lists
                for i, col in enumerate(header[1:], 1):
                    if "_mag" in col:
                        param_name = col.replace("_mag", "")
                        if param_name not in sparameters:
                            sparameters[param_name] = []
            elif line and header:
                # Data line
                parts = line.split(",")
                if len(parts) >= 2:
                    frequencies.append(float(parts[0]))
                    # Parse S-parameters (magnitude and phase)
                    i = 1
                    while i < len(parts) - 1:
                        mag = float(parts[i])
                        phase_deg = float(parts[i + 1])
                        phase_rad = np.deg2rad(phase_deg)
                        # Extract parameter name from header
                        if i < len(header) - 1:
                            param_name = header[i].replace("_mag", "")
                            sparameters[param_name].append(
                                mag * np.exp(1j * phase_rad)
                            )
                        i += 2

    # Convert to numpy arrays
    sparams_dict = {
        name: np.array(values) for name, values in sparameters.items()
    }

    return ResultsData(
        data_type="sparameters",
        data={
            "frequencies": np.array(frequencies),
            "sparameters": sparams_dict,
        },
        metadata=metadata,
    )


def load_csv_fields(filepath: Union[str, Path]) -> ResultsData:
    """
    Load field data from CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to CSV file

    Returns
    -------
    ResultsData
        Loaded field data
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("NumPy is required to load CSV files")

    filepath = Path(filepath)
    metadata = {}
    x_coords = []
    y_coords = []
    z_coords = []
    fields: dict[str, list[float]] = {
        "Ex": [],
        "Ey": [],
        "Ez": [],
        "Hx": [],
        "Hy": [],
        "Hz": [],
    }

    with open(filepath, "r") as f:
        header = None
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                # Parse metadata
                if ":" in line:
                    key, value = line[1:].split(":", 1)
                    metadata[key.strip()] = value.strip()
            elif line.startswith("x,") or line.startswith("x "):
                # Header line
                header = [col.strip() for col in line.split(",")]
            elif line and header:
                # Data line
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    x_coords.append(float(parts[0]))
                    y_coords.append(float(parts[1]))
                    z_coords.append(float(parts[2]))
                    # Parse field components
                    for i, comp in enumerate(["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"], 3):
                        if i < len(parts):
                            fields[comp].append(float(parts[i]))

    return ResultsData(
        data_type="field",
        data={
            "x": np.array(x_coords),
            "y": np.array(y_coords),
            "z": np.array(z_coords),
            "fields": {k: np.array(v) for k, v in fields.items()},
        },
        metadata=metadata,
    )


def load_parquet(filepath: Union[str, Path]) -> ResultsData:
    """
    Load data from Parquet file.

    Parameters
    ----------
    filepath : str or Path
        Path to Parquet file

    Returns
    -------
    ResultsData
        Loaded data (type determined from file structure)
    """
    if not POLARS_AVAILABLE:
        raise ImportError("Polars is required to load Parquet files. Install with: pip install polars")

    filepath = Path(filepath)
    df = pl.read_parquet(filepath)

    # Determine data type from columns
    columns = df.columns

    if "frequency_Hz" in columns:
        if any("S" in col for col in columns):
            # S-parameters
            frequencies = df["frequency_Hz"].to_numpy()
            sparameters = {}
            for col in columns:
                if col.endswith("_mag") and col.startswith("S"):
                    param_name = col.replace("_mag", "")
                    phase_col = col.replace("_mag", "_phase_deg")
                    if phase_col in columns:
                        mag = df[col].to_numpy()
                        phase = np.deg2rad(df[phase_col].to_numpy())
                        sparameters[param_name] = mag * np.exp(1j * phase)
                    else:
                        sparameters[param_name] = df[col].to_numpy()

            return ResultsData(
                data_type="sparameters",
                data={"frequencies": frequencies, "sparameters": sparameters},
            )
        else:
            # Spectrum
            frequencies = df["frequency_Hz"].to_numpy()
            spectrum_col = [c for c in columns if c != "frequency_Hz"][0]
            spectrum = df[spectrum_col].to_numpy()

            return ResultsData(
                data_type="spectrum",
                data={"frequencies": frequencies, "spectrum": spectrum},
            )
    else:
        # Field data
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        z = df["z"].to_numpy() if "z" in columns else np.zeros_like(x)
        fields = {}
        for comp in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            if comp in columns:
                fields[comp] = df[comp].to_numpy()

        return ResultsData(
            data_type="field",
            data={"x": x, "y": y, "z": z, "fields": fields},
        )


def load_from_file(filepath: Union[str, Path]) -> ResultsData:
    """
    Load results from a file (auto-detect format).

    Parameters
    ----------
    filepath : str or Path
        Path to results file

    Returns
    -------
    ResultsData
        Loaded data
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = filepath.suffix.lower()

    if suffix == ".parquet":
        return load_parquet(filepath)
    elif suffix == ".csv":
        # Try to determine CSV type by reading first few lines
        with open(filepath, "r") as f:
            first_lines = [f.readline().strip() for _ in range(5)]
            header = first_lines[-1] if first_lines else ""

        if "frequency_Hz" in header:
            if "S" in header or "_mag" in header:
                return load_csv_sparameters(filepath)
            else:
                return load_csv_spectrum(filepath)
        elif "x," in header or "x " in header:
            return load_csv_fields(filepath)
        else:
            # Default to spectrum
            return load_csv_spectrum(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_from_monitor(monitor: Any) -> ResultsData:
    """
    Load data from a simulation monitor.

    Parameters
    ----------
    monitor : Monitor
        Simulation monitor object (FieldMonitor, FluxMonitor, etc.)

    Returns
    -------
    ResultsData
        Loaded monitor data
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("NumPy is required to load monitor data")

    from prismo.monitors.field import FieldMonitor
    from prismo.monitors.flux import FluxMonitor

    if isinstance(monitor, FieldMonitor):
        # Get time domain data
        if monitor.time_domain:
            # Get first available component
            components = monitor.components
            if not components:
                raise ValueError("Monitor has no components recorded")

            component = components[0]
            time_points, field_data = monitor.get_time_data(component)

            return ResultsData(
                data_type="time_series",
                data={
                    "time": time_points,
                    "component": component,
                    "field": field_data,
                },
                metadata={"monitor_name": monitor.name},
            )
        elif monitor.frequencies:
            # Frequency domain data
            component = monitor.components[0]
            frequency = monitor.frequencies[0]
            field_data = monitor.get_frequency_data(component, frequency)

            return ResultsData(
                data_type="field",
                data={
                    "frequency": frequency,
                    "component": component,
                    "field": field_data,
                },
                metadata={"monitor_name": monitor.name},
            )

    elif isinstance(monitor, FluxMonitor):
        # Flux monitor - get frequency domain power
        if monitor.frequencies:
            frequencies = np.array(monitor.frequencies)
            power = monitor.get_frequency_domain_power()
            
            # Ensure power is an array
            if not isinstance(power, np.ndarray):
                power = np.array([power])

            return ResultsData(
                data_type="spectrum",
                data={
                    "frequencies": frequencies,
                    "spectrum": power,
                },
                metadata={"monitor_name": monitor.name, "type": "power"},
            )
        else:
            # Time domain power
            time_points, power = monitor.get_time_domain_power()
            return ResultsData(
                data_type="time_series",
                data={
                    "time": time_points,
                    "field": power,
                    "component": "Power",
                },
                metadata={"monitor_name": monitor.name, "type": "power"},
            )

    raise ValueError(f"Unsupported monitor type: {type(monitor)}")

