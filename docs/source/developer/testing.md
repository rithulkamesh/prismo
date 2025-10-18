# Testing Guide

Comprehensive testing ensures Prismo remains reliable and correct.

## Test Organization

```
tests/
├── test_core.py              # Core FDTD engine tests
├── test_sources.py           # Source tests
├── test_monitors.py          # Monitor tests
├── test_materials.py         # Material model tests
├── test_backends.py          # Backend tests
└── validation/               # Validation against known solutions
    ├── test_plane_wave_validation.py
    ├── test_mode_solver.py
    ├── test_sparameters.py
    └── test_mode_ports.py
```

## Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_modes.py

# Specific test
pytest tests/test_modes.py::TestModeSolver::test_fundamental_mode

# With coverage
pytest --cov=prismo --cov-report=html --cov-report=term

# Parallel execution
pytest -n auto

# Only fast tests (skip slow/GPU tests)
pytest -m "not slow and not gpu"
```

## Writing Tests

### Unit Tests

Test individual functions/classes:

```python
def test_gaussian_waveform():
    """Test Gaussian pulse waveform."""
    pulse = GaussianPulse(frequency=1e14, width=1e-15)

    # Peak at t=0
    assert pulse.value(0.0) == pytest.approx(1.0)

    # Decay at ±3σ
    assert pulse.value(3e-15) < 0.05
    assert pulse.value(-3e-15) < 0.05

    # Symmetry
    assert pulse.value(1e-15) == pytest.approx(pulse.value(-1e-15))
```

### Integration Tests

Test component interactions:

```python
@pytest.mark.integration
def test_source_monitor_workflow():
    """Test source and monitor integration."""
    sim = Simulation(size=(5e-6, 5e-6, 0.0), resolution=20e6)

    source = ElectricDipole(position=(2.5e-6, 2.5e-6, 0.0), ...)
    monitor = FieldMonitor(center=(2.5e-6, 2.5e-6, 0.0), ...)

    sim.add_source(source)
    sim.add_monitor(monitor)
    sim.run(50e-15)

    times, Ex_data = monitor.get_time_data('Ex')

    assert len(times) > 0
    assert Ex_data.shape[0] == len(times)
    assert np.max(np.abs(Ex_data)) > 0
```

### Validation Tests

Compare to analytical/reference solutions:

```python
@pytest.mark.validation
def test_waveguide_mode_neff():
    """Validate mode effective index against known values."""
    # Silicon strip waveguide (well-studied geometry)
    wavelength = 1.55e-6
    width = 0.5e-6
    height = 0.22e-6

    # Create structure and solve
    solver = create_si_waveguide_solver(wavelength, width, height)
    modes = solver.solve(num_modes=1)

    neff_calculated = modes[0].neff.real
    neff_reference = 2.45  # From literature/commercial tool

    error = abs(neff_calculated - neff_reference) / neff_reference

    assert error < 0.01, f"neff error {error*100:.2f}% too large"
```

## Test Fixtures

Reuse common setups:

```python
@pytest.fixture
def simple_simulation():
    """Standard test simulation."""
    return Simulation(
        size=(10e-6, 10e-6, 0.0),
        resolution=20e6,
        boundary_conditions='pml',
    )

@pytest.fixture
def si_waveguide():
    """Silicon waveguide cross-section."""
    x = np.linspace(-2e-6, 2e-6, 100)
    y = np.linspace(-1e-6, 1e-6, 100)
    epsilon = create_si_waveguide_permittivity(x, y)
    return x, y, epsilon

def test_with_fixtures(simple_simulation, si_waveguide):
    """Use fixtures in tests."""
    sim = simple_simulation
    x, y, eps = si_waveguide
    # Test code...
```

## Parametrized Tests

Test multiple inputs:

```python
@pytest.mark.parametrize("wavelength,expected_neff", [
    (1.3e-6, 2.52),
    (1.55e-6, 2.45),
    (1.8e-6, 2.38),
])
def test_neff_vs_wavelength(wavelength, expected_neff):
    """Test neff variation with wavelength."""
    solver = ModeSolver(wavelength, x, y, epsilon)
    modes = solver.solve(num_modes=1)

    assert modes[0].neff.real == pytest.approx(expected_neff, rel=0.02)
```

## Testing Checklist

For each new feature:

- [ ] Unit tests for individual functions
- [ ] Integration test with other components
- [ ] Edge case tests (empty input, extreme values, etc.)
- [ ] Error handling tests (invalid input)
- [ ] Performance test (if critical path)
- [ ] Documentation test (doctest examples work)

## Continuous Integration

Tests run automatically on GitHub Actions:

```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -e ".[dev]"
      - run: pytest --cov=prismo
```

## See Also

- {doc}`contributing` - Contribution guidelines
- {doc}`architecture` - Code architecture
- {doc}`benchmarks` - Performance benchmarks
