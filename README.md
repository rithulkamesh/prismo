# Prismo

[![PyPI version](https://badge.fury.io/py/pyprismo.svg)](https://pypi.org/project/pyprismo/)
[![CI](https://github.com/rithulkamesh/prismo/actions/workflows/tests.yml/badge.svg)](https://github.com/rithulkamesh/prismo/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/rithulkamesh/prismo/branch/main/graph/badge.svg)](https://codecov.io/gh/rithulkamesh/prismo)
[![Documentation Status](https://readthedocs.org/projects/prismo/badge/?version=latest)](https://prismo.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A high-performance Python-based FDTD solver for waveguide photonics.

Prismo implements the Finite-Difference Time-Domain method for solving Maxwell's equations, with features designed for photonic integrated circuits, waveguides, and optical devices.

## Features

- **3D vectorial FDTD** on Yee grids with automatic time-stepping
- **GPU acceleration** via CuPy with transparent CPU fallback
- **Dispersive materials** including Lorentz, Drude, Debye, and Sellmeier models
- **Material library** with Si, SiO2, Si3N4, Au, Ag, Al, and ITO
- **Anisotropic materials** supporting full permittivity and permeability tensors
- **PML absorbing boundaries** using the CPML formulation
- **Eigenmode solver** for 2D waveguide structures
- **Advanced monitors** for frequency-domain (DFT), power flux, and mode expansion
- **S-parameter extraction** with Touchstone export
- **Data export** to CSV and Parquet formats
- **Lumerical compatibility** for importing FSP files and material databases
- **Parameter sweeps** with parallel execution

## Installation

```bash
pip install pyprismo
```

For GPU acceleration:

```bash
pip install pyprismo[acceleration]
```

For development:

```bash
git clone https://github.com/rithulkamesh/prismo.git
cd prismo
pip install -e ".[all]"
```

**Note**: The package name on PyPI is `pyprismo`, but you still import it as `prismo`:

```python
import prismo  # Import name stays as 'prismo'
```

## Quick Start

```python
import numpy as np
import prismo

# Select backend
prismo.set_backend('cupy')  # Use GPU, or 'numpy' for CPU

# Create simulation
sim = prismo.Simulation(
    size=(10e-6, 5e-6, 0),  # 10×5 μm, 2D
    resolution=50e6,         # 20 nm grid spacing
    boundary_conditions="pml",
)

# Add source
source = prismo.GaussianBeamSource(
    center=(-4e-6, 0, 0),
    size=(0, 2e-6, 0),
    frequency=193e12,  # 1550 nm
    pulse_width=10e-15
)
sim.add_source(source)

# Add DFT monitor
wavelengths = np.linspace(1.5e-6, 1.6e-6, 11)
dft = prismo.DFTMonitor(
    center=(4e-6, 0, 0),
    size=(0, 2e-6, 0),
    frequencies=(299792458.0 / wavelengths).tolist()
)
sim.add_monitor(dft)

# Run
sim.run(time=50e-15)

# Analyze
spectrum = dft.get_power_spectrum('Ex')
```

## Documentation

- [User Guide](docs/source/user_guide/) - Tutorials and how-to guides
- [API Reference](docs/source/api/) - Complete API documentation
- [Examples](examples/) - Working example scripts

## Materials

```python
# Use pre-defined materials
si = prismo.get_material('Si')
sio2 = prismo.get_material('SiO2')
au = prismo.get_material('Au')

# List all materials
print(prismo.list_materials())

# Create custom dispersive material
material = prismo.LorentzMaterial(
    epsilon_inf=2.0,
    poles=[prismo.LorentzPole(omega_0=2e15, delta_epsilon=1.0, gamma=1e13)]
)
```

## Mode Solver

```python
# Solve for waveguide modes
mode_solver = prismo.ModeSolver(
    wavelength=1.55e-6,
    x=x_coords,
    y=y_coords,
    epsilon=epsilon_profile
)

modes = mode_solver.solve(num_modes=3, mode_type='TE')
fundamental = modes[0]
print(f"Effective index: {fundamental.neff.real:.4f}")

# Use mode as source
mode_source = prismo.ModeSource(mode=fundamental, direction='+x', ...)
```

## S-Parameters

```python
# Extract S-parameters
s_analyzer = prismo.SParameterAnalyzer(
    num_ports=2,
    frequencies=frequencies
)

# Calculate metrics
s21 = s_analyzer.get_s_parameter(1, 0)
insertion_loss = s_analyzer.get_insertion_loss_db(1, 0)

# Export to Touchstone
prismo.export_touchstone("device.s2p", frequencies, s_analyzer.s_matrix)
```

## Data Export

```python
# Parquet (efficient, compressed)
exporter = prismo.ParquetExporter(output_dir="./results")
exporter.export_sparameters(
    filename="device",
    frequencies=frequencies,
    sparameters={'S21': s21}
)

# CSV (universal)
csv_exporter = prismo.CSVExporter(output_dir="./results")
csv_exporter.export_spectrum(...)
```

## Performance

Typical performance on NVIDIA A100:

| Dimension | Grid Size   | Throughput  |
| --------- | ----------- | ----------- |
| 2D        | 1000×1000   | 80M cells/s |
| 3D        | 100×100×100 | 8M cells/s  |

CPU performance (NumPy): ~1-2% of GPU performance.

## Requirements

- Python ≥3.9
- NumPy ≥1.21
- SciPy ≥1.7
- Polars ≥0.20 (for Parquet export)

Optional:

- CuPy ≥12.0 (for GPU acceleration)
- Matplotlib (for visualization)

## Testing

```bash
pytest tests/
```

Run specific test categories:

```bash
pytest tests/test_backends.py
pytest tests/validation/
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Documentation

📚 **[Full Documentation on ReadTheDocs](https://prismo.readthedocs.io/)**

### Quick Links

- **[Installation Guide](https://prismo.readthedocs.io/en/latest/user_guide/installation.html)** - Get set up quickly
- **[Quick Start Tutorial](https://prismo.readthedocs.io/en/latest/user_guide/quickstart.html)** - Your first simulation in 5 minutes
- **[Tutorials](https://prismo.readthedocs.io/en/latest/tutorials/index.html)** - Step-by-step guides
  - [Basic Simulation](https://prismo.readthedocs.io/en/latest/tutorials/basic_simulation.html)
  - [Waveguide Coupling](https://prismo.readthedocs.io/en/latest/tutorials/waveguide_coupling.html)
  - [S-Parameter Extraction](https://prismo.readthedocs.io/en/latest/tutorials/sparameters.html)
  - [Parameter Optimization](https://prismo.readthedocs.io/en/latest/tutorials/optimization.html)
- **[User Guide](https://prismo.readthedocs.io/en/latest/user_guide/index.html)** - Comprehensive documentation
  - [Mode Ports](https://prismo.readthedocs.io/en/latest/user_guide/mode_ports.html)
  - [Boundary Conditions](https://prismo.readthedocs.io/en/latest/user_guide/boundaries.html)
  - [Validation](https://prismo.readthedocs.io/en/latest/user_guide/validation.html)
- **[API Reference](https://prismo.readthedocs.io/en/latest/api/index.html)** - Complete API documentation
- **[Examples](https://prismo.readthedocs.io/en/latest/examples/index.html)** - Sample code and demos

### For Developers

- **[Architecture](https://prismo.readthedocs.io/en/latest/developer/architecture.html)** - Code structure and design
- **[Contributing](https://prismo.readthedocs.io/en/latest/developer/contributing.html)** - How to contribute
- **[Testing Guide](https://prismo.readthedocs.io/en/latest/developer/testing.html)** - Testing practices
- **[Benchmarks](https://prismo.readthedocs.io/en/latest/developer/benchmarks.html)** - Performance metrics

## Citation

If you use Prismo in academic work, please cite:

```bibtex
@software{prismo2025,
  author = {Kamesh, Rithul},
  title = {Prismo: Python FDTD Solver for Photonics},
  year = {2025},
  url = {https://github.com/rithulkamesh/prismo}
}
```

## References

- Taflove, A., & Hagness, S. C. (2005). _Computational Electrodynamics: The Finite-Difference Time-Domain Method_. Artech House.
- Yee, K. S. (1966). "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media." _IEEE Trans. Antennas Propagation_, 14(3), 302-307.
