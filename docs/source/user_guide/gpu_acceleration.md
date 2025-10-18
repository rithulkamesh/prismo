# GPU Acceleration

Prismo supports transparent GPU acceleration using CUDA for significant performance improvements.

## Backend Selection

### Automatic Selection

```python
import prismo

# Automatically select best available backend (GPU if available)
backend = prismo.get_backend()
```

### Manual Selection

```python
# Use GPU
prismo.set_backend('cupy')

# Use CPU
prismo.set_backend('numpy')

# Check what's available
backends = prismo.list_available_backends()
print(backends)  # ['numpy', 'cupy']
```

## Performance

Typical speedups with GPU acceleration:

| Grid Size        | CPU (cells/s) | GPU (cells/s) | Speedup |
| ---------------- | ------------- | ------------- | ------- |
| 2D (1000×1000)   | 1-5M          | 50-100M       | 50-100x |
| 3D (100×100×100) | 100K-1M       | 5-10M         | 50-100x |

## Installation

### CUDA Requirements

For GPU acceleration, you need:

- NVIDIA GPU with CUDA support
- CUDA Toolkit 12.x or later
- CuPy library

### Install with GPU Support

```bash
# Install CUDA-enabled Prismo
pip install pyprismo[acceleration]

# Or manually
pip install cupy-cuda12x
```

### Verify GPU

```python
import prismo

if 'cupy' in prismo.list_available_backends():
    backend = prismo.set_backend('cupy')
    print(f"GPU: {backend}")
    print(backend.get_memory_info())
else:
    print("GPU not available, using CPU")
```

## Usage in Simulations

Once you've selected a backend, all computations automatically use it:

```python
# Set backend once
prismo.set_backend('cupy')

# All subsequent operations use GPU
sim = prismo.Simulation(...)
sim.run(time=20e-15)  # Runs on GPU

# Monitors also use GPU
dft = prismo.DFTMonitor(..., backend='cupy')
```

## Memory Management

Check GPU memory usage:

```python
backend = prismo.get_backend('cupy')
mem_info = backend.get_memory_info()

print(f"Used: {mem_info['used_mb']:.0f} MB")
print(f"Free: {mem_info['free_mb']:.0f} MB")
print(f"Total: {mem_info['total_device_mb']:.0f} MB")
```

## Best Practices

1. **Set backend early**: Call `set_backend()` before creating simulations
2. **Monitor memory**: Large 3D grids can use significant GPU memory
3. **Batch operations**: Process multiple frequencies together for efficiency
4. **Fallback to CPU**: Code works identically on CPU if GPU unavailable
