# Prismo 🔬

[![CI](https://github.com/rithulkamesh/prismo/actions/workflows/ci.yml/badge.svg)](https://github.com/rithulkamesh/prismo/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/rithulkamesh/prismo/branch/main/graph/badge.svg)](https://codecov.io/gh/rithulkamesh/prismo)
[![PyPI version](https://badge.fury.io/py/prismo.svg)](https://badge.fury.io/py/prismo)
[![Python versions](https://img.shields.io/pypi/pyversions/prismo.svg)](https://pypi.org/project/prismo/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance Python-based Finite-Difference Time-Domain (FDTD) solver specifically designed for waveguide photonics. Prismo provides professional-grade electromagnetic simulation capabilities with modern Python tooling, making it accessible, extensible, and suitable for both academic research and engineering applications.

## 🎯 Features

### Core Capabilities

- **2D and 3D FDTD simulations** with Yee-grid discretization
- **Advanced boundary conditions**: CPML absorbing boundaries, periodic/Bloch conditions
- **Comprehensive source types**: Gaussian pulses, CW sources, waveguide modes, TFSF
- **Material models**: Dispersive media (Drude, Lorentz), metals, anisotropic materials
- **Waveguide mode analysis**: Eigenmode calculation and S-parameter extraction

### Performance & Scalability

- **Multi-threading** support for CPU parallelization
- **GPU acceleration** with CUDA/CuPy (optional)
- **Memory-efficient** algorithms with sparse matrix operations
- **HDF5 data format** for large-scale simulation storage

### Developer Experience

- **Modern Python packaging** with UV and pyproject.toml
- **Type hints** throughout for better IDE support
- **Comprehensive testing** with pytest and benchmarking
- **Rich documentation** with Sphinx and Jupyter examples
- **Extensible architecture** for custom materials and sources

## 🚀 Quick Start

### Installation

#### Using UV (Recommended)

```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Prismo
uv add prismo

# Or with all optional dependencies
uv add "prismo[all]"
```

#### Using pip

```bash
pip install prismo

# With GPU acceleration support
pip install "prismo[acceleration]"

# With visualization tools
pip install "prismo[visualization]"
```

### Development Setup with Nix

For a reproducible development environment:

```bash
# Clone the repository
git clone https://github.com/rithulkamesh/prismo.git
cd prismo

# Enter Nix development shell (installs all dependencies)
nix develop

# Or use direnv for automatic environment activation
echo "use flake" > .envrc
direnv allow
```

### Basic Example

```python
import numpy as np
from prismo import Simulation, Rectangle, Gaussian, FieldMonitor

# Create a simple waveguide simulation
sim = Simulation(
    size=(10, 5, 0),      # 2D simulation: 10×5 μm
    resolution=20,         # 20 points per μm
    boundary_conditions="pml"
)

# Add a silicon waveguide
waveguide = Rectangle(
    center=(0, 0, 0),
    size=(10, 0.22, 0),
    material="Si"  # n=3.45
)
sim.add_structure(waveguide)

# Add a Gaussian source
source = Gaussian(
    center=(-4, 0, 0),
    size=(0, 0.4, 0),
    frequency=200e12,     # 200 THz (1.55 μm)
    pulse_width=1e-15     # 1 fs pulse
)
sim.add_source(source)

# Add field monitor
monitor = FieldMonitor(
    center=(4, 0, 0),
    size=(0, 2, 0),
    frequency=200e12
)
sim.add_monitor(monitor)

# Run simulation
sim.run(time=20e-15)    # 20 fs

# Extract results
fields = sim.get_field_data(monitor)
transmission = sim.get_transmission(monitor)
```

## 📖 Documentation

- **[User Guide](docs/user_guide/)**: Comprehensive tutorials and examples
- **[API Reference](docs/api/)**: Detailed API documentation
- **[Examples](examples/)**: Jupyter notebooks with complete simulation examples
- **[Developer Guide](docs/developer/)**: Contributing and extending Prismo

## 🛠️ Development

### Setting up the Development Environment

```bash
# Clone and enter development environment
git clone https://github.com/rithulkamesh/prismo.git
cd prismo
nix develop  # or use your preferred Python environment

# Install in development mode
make install

# Run tests
make test

# Check code quality
make lint

# Build documentation
make docs
```

### Available Make Targets

```bash
make help           # Show all available commands
make install        # Install in development mode
make test           # Run test suite
make test-cov       # Run tests with coverage
make lint           # Run linting and type checking
make format         # Format code with black/isort
make docs           # Build documentation
make build          # Build distribution packages
make clean          # Clean build artifacts
```

### Project Structure

```
prismo/
├── src/prismo/           # Main package
│   ├── core/            # Core FDTD solver
│   ├── solvers/         # Different solver implementations
│   ├── materials/       # Material models
│   ├── boundaries/      # Boundary conditions
│   ├── sources/         # Source implementations
│   ├── monitors/        # Field monitors and analysis
│   ├── geometry/        # Geometric primitives
│   ├── utils/           # Utilities and helpers
│   └── visualization/   # Plotting and visualization
├── tests/               # Test suite
├── examples/            # Example simulations
├── benchmarks/          # Performance benchmarks
├── docs/                # Documentation source
├── flake.nix           # Nix development environment
├── pyproject.toml      # Project configuration
└── Makefile           # Development automation
```

## 🎯 Roadmap

### Phase 1: Core Solver (Months 2-5)

- [x] Project scaffolding and tooling
- [ ] Basic 2D/3D FDTD implementation
- [ ] Gaussian sources and field monitors
- [ ] Plane wave validation cases

### Phase 2: Waveguide Features (Months 6-9)

- [ ] CPML absorbing boundaries
- [ ] Waveguide mode solver
- [ ] Mode injection/extraction ports
- [ ] S-parameter calculation

### Phase 3: Advanced Materials (Months 10-12)

- [ ] Dispersive material models (ADE)
- [ ] Metallic materials (Drude model)
- [ ] Mesh refinement algorithms

### Phase 4: Performance & Acceleration (Months 13-15)

- [ ] Multi-threading optimization
- [ ] GPU acceleration with CuPy
- [ ] Memory usage optimization

### Phase 5: Extensibility (Months 16-18)

- [ ] Plugin architecture
- [ ] Parameter sweep utilities
- [ ] Optimization workflow integration
- [ ] Full documentation and tutorials

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run quality checks (`make lint test`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yee, K. S.** (1966). "Numerical solution of initial boundary value problems involving Maxwell's equations in isotropic media." _IEEE Transactions on Antennas and Propagation_, 14(3), 302-307.
- **Taflove, A., & Hagness, S. C.** (2005). _Computational Electrodynamics: The Finite-Difference Time-Domain Method_. Artech House.
- The open-source scientific Python ecosystem (NumPy, SciPy, Matplotlib)

## 📞 Support

- **Documentation**: [https://prismo.readthedocs.io](https://prismo.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/rithulkamesh/prismo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/rithulkamesh/prismo/discussions)

---

_Prismo: Illuminating the path to advanced photonic simulations_ ✨
