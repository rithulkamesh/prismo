# Advanced Features

## Frequency-Domain Monitors

### DFT Monitors

Compute frequency-domain fields on-the-fly without storing all time steps:

```python
from prismo import DFTMonitor

# Define frequencies of interest
wavelengths = np.linspace(1.5e-6, 1.6e-6, 11)
frequencies = 299792458.0 / wavelengths

# Create DFT monitor
dft = DFTMonitor(
    center=(4e-6, 0, 0),
    size=(0, 2e-6, 0),
    frequencies=frequencies.tolist(),
    components=['Ex', 'Ey', 'Ez'],
)

sim.add_monitor(dft)
sim.run(time=50e-15)

# Get frequency-domain data
field_freq = dft.get_frequency_data('Ex', frequency_index=5)
intensity = dft.get_intensity('Ex', frequency_index=5)
spectrum = dft.get_power_spectrum('Ex')
```

### Flux Monitors

Calculate electromagnetic power flow through surfaces:

```python
from prismo import FluxMonitor

flux = FluxMonitor(
    center=(4e-6, 0, 0),
    size=(0, 2e-6, 0),
    direction='x',  # Power flow direction
    frequencies=frequencies.tolist(),
)

sim.add_monitor(flux)
sim.run(time=50e-15)

# Get power flow
time_array, power_array = flux.get_time_domain_power()
freq_power = flux.get_frequency_domain_power()
transmission = flux.get_transmission()
```

## Mode Solver

### Computing Waveguide Modes

```python
from prismo import ModeSolver

# Define waveguide cross-section
x = np.linspace(-2e-6, 2e-6, 100)
y = np.linspace(-1e-6, 1e-6, 80)

# Create permittivity profile
epsilon = np.ones((len(x), len(y)))  # Background
# Add waveguide core (e.g., Silicon)
for i, xi in enumerate(x):
    for j, yj in enumerate(y):
        if abs(xi) < 0.25e-6 and abs(yj) < 0.11e-6:
            epsilon[i, j] = 11.68  # Si at 1550nm

# Solve for modes
mode_solver = ModeSolver(
    wavelength=1.55e-6,
    x=x,
    y=y,
    epsilon=epsilon
)

modes = mode_solver.solve(num_modes=3, mode_type='TE')

# Access mode properties
fundamental = mode_solver.get_mode(0)
print(f"Effective index: {fundamental.neff.real:.4f}")

# Mode fields are available
Ex, Ey, Ez = fundamental.Ex, fundamental.Ey, fundamental.Ez
Hx, Hy, Hz = fundamental.Hx, fundamental.Hy, fundamental.Hz
```

### Mode Expansion Monitors

Decompose fields into mode coefficients:

```python
from prismo import ModeExpansionMonitor

# First, solve for modes
modes = mode_solver.solve(num_modes=2)

# Create mode expansion monitor
mode_monitor = ModeExpansionMonitor(
    center=(4e-6, 0, 0),
    size=(0, 2e-6, 0),
    modes=modes,
    direction='x',
    frequencies=frequencies.tolist(),
)

sim.add_monitor(mode_monitor)
sim.run(time=50e-15)

# Get mode coefficients
coeff_0 = mode_monitor.get_mode_coefficient(mode_index=0, domain='frequency')
power_0 = mode_monitor.get_mode_power(mode_index=0)
```

## S-Parameter Extraction

### Multi-Port S-Parameters

```python
from prismo import SParameterAnalyzer, export_touchstone

# Create analyzer for 2-port device
s_analyzer = SParameterAnalyzer(
    num_ports=2,
    frequencies=frequencies,
    reference_impedance=50.0
)

# Add measurement data (from flux or mode monitors)
s_analyzer.add_port_data(
    port_index=1,
    excitation_port=0,
    power_forward=transmitted_power,
    power_backward=reflected_power
)

# Access S-parameters
s11 = s_analyzer.get_s_parameter(0, 0)  # Reflection
s21 = s_analyzer.get_s_parameter(1, 0)  # Transmission

# Calculate metrics
insertion_loss = s_analyzer.get_insertion_loss_db(1, 0)
return_loss = s_analyzer.get_return_loss_db(0)

# Check reciprocity
error = s_analyzer.check_reciprocity()
```

### Export to Touchstone

For integration with circuit simulators:

```python
export_touchstone(
    filename="device.s2p",
    frequencies=frequencies,
    s_matrix=s_analyzer.s_matrix,
    z0=50.0,
    comments=["Silicon waveguide", "1500-1600 nm"]
)
```

## Data Export

### CSV Export

```python
from prismo import CSVExporter

exporter = CSVExporter(output_dir="./results")

# Export S-parameters
exporter.export_sparameters(
    filename="device_sparams",
    frequencies=frequencies,
    sparameters={'S11': s11, 'S21': s21},
    metadata={'device': 'waveguide', 'date': '2025-10-17'}
)

# Export spectrum
exporter.export_spectrum(
    filename="transmission",
    frequencies=frequencies,
    spectrum=transmission_data
)
```

### Parquet Export

More efficient for large datasets:

```python
from prismo import ParquetExporter

exporter = ParquetExporter(
    output_dir="./results",
    compression='snappy'  # or 'gzip', 'lz4', 'zstd'
)

# Same API as CSV
exporter.export_sparameters(...)
exporter.export_spectrum(...)

# Read back with Polars
import polars as pl
df = pl.read_parquet("results/device_sparams.parquet")
```

## Parameter Sweeps

Automate design space exploration:

```python
from prismo import ParameterSweep, SweepParameter

def run_simulation(params):
    # Run simulation with params['width'], params['height']
    # Return results dict
    return {'transmission': 0.85, 'bandwidth': 50e12}

sweep = ParameterSweep(
    parameters=[
        SweepParameter('width', np.linspace(0.4e-6, 0.6e-6, 11), unit='m'),
        SweepParameter('height', np.linspace(0.2e-6, 0.3e-6, 6), unit='m'),
    ],
    simulation_func=run_simulation,
    output_dir="./sweep_results",
    parallel=True,  # Use multi-core
    num_workers=4
)

# Run sweep
results = sweep.run()

# Find optimal parameters
optimal_params, optimal_results = sweep.find_optimal('transmission', maximize=True)

# Save results
sweep.save_results("sweep_results.json")

# Visualize
sweep.plot_sweep_1d('width', ['transmission'], save_path="sweep_1d.png")
```

## Lumerical Import

### Import FSP Files

```python
from prismo import FSPParser

# Parse Lumerical project file
parser = FSPParser("my_project.fsp")
project = parser.parse()

# Inspect contents
print(f"Geometries: {len(project.geometries)}")
print(f"Sources: {len(project.sources)}")
print(f"Monitors: {len(project.monitors)}")

# Convert to Prismo simulation
sim = parser.to_prismo_simulation()

# Export summary
parser.export_summary("project_summary.json")
```

### Import Material Database

```python
from prismo import import_lumerical_material

# Import material from Lumerical file
material = import_lumerical_material("path/to/Silicon.txt", "Si_imported")

# Add to Prismo library
prismo.add_material('Silicon_imported', material)
```
