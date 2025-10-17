# Data Export and Analysis

## Export Formats

Prismo supports multiple export formats for different use cases:

| Format     | Use Case                  | File Size       | Speed      |
| ---------- | ------------------------- | --------------- | ---------- |
| CSV        | Spreadsheets, universal   | Large           | Slow       |
| Parquet    | Big data, Python analysis | 10-100x smaller | 10x faster |
| Touchstone | Circuit simulators        | Compact         | Fast       |

## CSV Export

### Basic Usage

```python
from prismo import CSVExporter

exporter = CSVExporter(output_dir="./results")
```

### Export Field Data

```python
# Export electromagnetic fields
exporter.export_fields(
    filename="fields_t100",
    fields={
        'Ex': Ex_array,
        'Ey': Ey_array,
        'Ez': Ez_array,
        'Hx': Hx_array,
        'Hy': Hy_array,
        'Hz': Hz_array,
    },
    coordinates={'x': x_array, 'y': y_array, 'z': z_array},
    metadata={'time': 100e-15, 'wavelength': 1.55e-6}
)
```

Output columns: `x, y, z, Ex, Ey, Ez, Hx, Hy, Hz, |E|, |H|`

### Export Spectra

```python
exporter.export_spectrum(
    filename="transmission_spectrum",
    frequencies=freq_array,
    spectrum=transmission_array,
    metadata={'component': 'S21', 'units': 'linear'}
)
```

### Export S-Parameters

```python
exporter.export_sparameters(
    filename="device_sparameters",
    frequencies=frequencies,
    sparameters={
        'S11': s11_array,
        'S21': s21_array,
        'S12': s12_array,
        'S22': s22_array,
    },
    metadata={'device': 'directional_coupler'}
)
```

## Parquet Export

More efficient for large datasets:

```python
from prismo import ParquetExporter

exporter = ParquetExporter(
    output_dir="./results",
    compression='snappy'  # Options: 'snappy', 'gzip', 'lz4', 'zstd'
)

# Use same API as CSV
exporter.export_fields(...)
exporter.export_spectrum(...)
exporter.export_sparameters(...)
```

### Reading Parquet Files

```python
import polars as pl

# Simple read
df = pl.read_parquet("results/device_sparameters.parquet")

# Lazy read with filtering
df = exporter.read_with_filter(
    filepath="results/fields.parquet",
    filter_expr=pl.col('frequency_Hz') > 190e12,
    columns=['frequency_Hz', 'S21_magnitude']
)

# Process with Polars
transmission_db = -10 * np.log10(df['S21_magnitude'])
```

## Touchstone Export

Standard format for circuit simulators (ADS, HFSS, etc.):

```python
from prismo import export_touchstone

export_touchstone(
    filename="device.s2p",  # .s2p for 2-port
    frequencies=frequencies,
    s_matrix=s_matrix,  # Shape: (n_freq, n_ports, n_ports)
    z0=50.0,
    comments=[
        "Silicon photonic waveguide",
        "Operating wavelength: 1550 nm",
        "Simulated with Prismo FDTD"
    ]
)
```

The `.s2p` file can be directly imported into:

- Keysight ADS
- Ansys HFSS
- Cadence
- LTspice
- Other RF/photonic circuit simulators

## Metadata

All exporters support metadata for provenance tracking:

```python
metadata = {
    'simulation_id': 'wg_001',
    'date': '2025-10-17',
    'device': 'silicon_waveguide',
    'dimensions': '500nm x 220nm',
    'wavelength_range': '1500-1600 nm',
    'backend': 'cupy',
    'grid_resolution': '20 nm',
    'pml_layers': 10,
}

exporter.export_sparameters(..., metadata=metadata)
```

Metadata is saved as:

- Comments in CSV files (header lines starting with `#`)
- Separate JSON file for Parquet (`.meta.json`)
- Comments in Touchstone files (lines starting with `!`)

## Best Practices

1. **Use Parquet for large datasets**: 10-100x smaller files, much faster I/O
2. **Use CSV for small data**: Easy to inspect in spreadsheets
3. **Use Touchstone for S-parameters**: Standard format for circuit design
4. **Always include metadata**: Makes results self-documenting
5. **Compress Parquet files**: Use 'snappy' for speed, 'zstd' for size
