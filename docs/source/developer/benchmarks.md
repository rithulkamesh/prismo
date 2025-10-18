# Performance Benchmarks

Performance benchmarks help track Prismo's computational efficiency.

## Running Benchmarks

```bash
# Run all benchmarks
python benchmarks/test_performance.py

# Specific benchmark
pytest benchmarks/ -k test_fdtd_step --benchmark-only

# With pytest-benchmark
pytest benchmarks/ --benchmark-only --benchmark-autosave
```

## Typical Performance

### 2D Simulation

**Configuration:**

- Grid: 500 × 500 points
- Time steps: 1000
- Backend: NumPy (CPU)

**Performance:**

- Time per step: ~50 ms
- Total time: ~50 seconds
- Memory: ~200 MB

### 3D Simulation

**Configuration:**

- Grid: 200 × 200 × 200 points
- Time steps: 1000
- Backend: NumPy (CPU)

**Performance:**

- Time per step: ~2 seconds
- Total time: ~30 minutes
- Memory: ~2 GB

### GPU Acceleration

**Speedup (GPU vs CPU):**

- 2D: 5-10×
- 3D: 20-50×

## Benchmark Suite

```python
import pytest
from prismo import Simulation

@pytest.mark.benchmark
def test_fdtd_step_2d(benchmark):
    """Benchmark single FDTD time step (2D)."""
    sim = Simulation(
        size=(10e-6, 10e-6, 0.0),
        resolution=50e6,  # 500x500 grid
    )
    sim.initialize()

    result = benchmark(sim.step)

    assert result is not None

@pytest.mark.benchmark
def test_full_simulation(benchmark):
    """Benchmark full simulation run."""
    def run_sim():
        sim = Simulation(...)
        sim.add_source(...)
        sim.run(100e-15)
        return sim

    result = benchmark(run_sim)
```

## Profiling

### CPU Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run simulation
sim.run(100e-15)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def run_simulation():
    sim = Simulation(...)
    sim.run(100e-15)

run_simulation()
```

## Optimization Guidelines

1. **Use appropriate resolution**: More isn't always better
2. **Leverage vectorization**: NumPy/CuPy operations
3. **Minimize data transfer**: Keep data on GPU
4. **Profile before optimizing**: Measure, don't guess

## See Also

- {doc}`architecture` - Code organization
- {doc}`../user_guide/gpu_acceleration` - GPU usage
