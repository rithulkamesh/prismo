# Installation

## Requirements

Prismo requires Python 3.9 or later and the following dependencies:

- **NumPy** (≥1.21): Numerical computing
- **Matplotlib** (≥3.4): Visualization
- **SciPy** (≥1.7): Scientific computing utilities
- **h5py** (≥3.0): HDF5 file support

## Installation from PyPI (Recommended)

Install the latest stable release from PyPI:

```bash
pip install pyprismo
```

**Note**: The package name on PyPI is `pyprismo`, but you import it as `prismo`:

```python
import prismo  # Import name is 'prismo'
```

### With Optional Dependencies

```bash
# GPU acceleration (requires CUDA)
pip install pyprismo[acceleration]

# Visualization tools
pip install pyprismo[visualization]

# Everything
pip install pyprismo[all]
```

## Installation from Source

For development or the latest features, install from source:

```bash
git clone https://github.com/rithulkamesh/prismo.git
cd prismo
pip install -e .
```

This will install Prismo in editable mode along with all required dependencies.

## Verifying Installation

Verify the installation by running:

```python
import prismo
print(prismo.__version__)
```

You should see the version number (e.g., `0.1.0-dev`).

## Running Tests

To ensure everything is working correctly, run the test suite:

```bash
pytest tests/
```

You should see over 100 tests passing.

## Optional Dependencies

For enhanced functionality, you can install optional dependencies:

### Development Tools

```bash
pip install -e ".[dev]"
```

This includes:

- pytest: Testing framework
- pytest-cov: Code coverage
- black: Code formatting
- flake8: Linting

### Documentation Building

```bash
pip install -e ".[docs]"
```

This includes:

- Sphinx: Documentation generator
- MyST-Parser: Markdown support
- sphinx-rtd-theme: ReadTheDocs theme

## Troubleshooting

### Import Errors

If you get import errors, ensure the `src` directory is in your Python path:

```python
import sys
sys.path.insert(0, 'path/to/prismo/src')
```

### NumPy Issues

If you encounter NumPy-related issues, try updating to the latest version:

```bash
pip install --upgrade numpy
```

### Platform-Specific Notes

**Linux**: No special requirements

**macOS**: Ensure you have Xcode command line tools installed

**Windows**: Some features may require Microsoft Visual C++ 14.0 or greater

## Next Steps

- Continue to [Quick Start](quickstart.md) to run your first simulation
- Read the [User Guide](../user_guide/index.md) for detailed usage
- Check out [Examples](../examples/index.md) for practical demonstrations
