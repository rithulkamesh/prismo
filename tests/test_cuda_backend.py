"""
Tests for CUDA/CuPy backend functionality.

This module tests CUDA kernel compilation, execution, and integration
with the FDTD solver.
"""

import os
import sys

import numpy as np
import pytest

# Set up LD_LIBRARY_PATH for Nix systems
nix_cuda_lib = "/nix/store/c3dgq22cnp210a9xwdnj37sq7297l3cw-nvidia-x11-580.119.02-6.12.63/lib"
nix_cuda_toolkit = "/nix/store/94m6rn64haflwrxb5wqrv4jqdccbhbbi-cuda-merged-12.8/lib"
if os.path.exists(nix_cuda_lib):
    os.environ["LD_LIBRARY_PATH"] = f"{nix_cuda_lib}:{nix_cuda_toolkit}:{os.environ.get('LD_LIBRARY_PATH', '')}"

try:
    import cupy as cp

    CUPY_AVAILABLE = cp.cuda.is_available()
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from prismo.backends import get_backend, list_available_backends, set_backend
from prismo.backends.cupy_backend import CuPyBackend


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available or CUDA not working")
class TestCuPyBackend:
    """Test CuPy backend basic functionality."""

    def test_cupy_import(self):
        """Test that CuPy can be imported and CUDA is available."""
        assert cp is not None
        assert cp.cuda.is_available()

    def test_cupy_backend_creation(self):
        """Test creating a CuPy backend."""
        backend = CuPyBackend(device_id=0)
        assert backend.name == "cupy"
        assert backend.is_gpu

    def test_cupy_backend_array_operations(self):
        """Test basic array operations on GPU."""
        backend = get_backend("cupy")

        # Create arrays
        arr = backend.zeros((100, 100))
        assert arr.shape == (100, 100)

        ones = backend.ones((50, 50))
        result = backend.to_numpy(ones)
        assert np.allclose(result, 1.0)

    def test_cupy_backend_math_operations(self):
        """Test mathematical operations on GPU."""
        backend = get_backend("cupy")

        data = np.array([1.0, 4.0, 9.0, 16.0])
        arr = backend.array(data)
        sqrt_arr = backend.sqrt(arr)
        result = backend.to_numpy(sqrt_arr)

        assert np.allclose(result, [1.0, 2.0, 3.0, 4.0])

    def test_cupy_backend_fft(self):
        """Test FFT operations on GPU."""
        backend = get_backend("cupy")

        data = np.random.random(128) + 1j * np.random.random(128)
        arr = backend.array(data)
        fft_arr = backend.fft(arr)
        ifft_arr = backend.ifft(fft_arr)
        result = backend.to_numpy(ifft_arr)

        assert np.allclose(data, result, rtol=1e-10)

    def test_cupy_backend_memory_info(self):
        """Test GPU memory information."""
        backend = get_backend("cupy")
        mem_info = backend.get_memory_info()

        assert "backend" in mem_info
        assert mem_info["backend"] == "cupy"
        assert "device_name" in mem_info
        assert "used_mb" in mem_info
        assert "total_mb" in mem_info


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available or CUDA not working")
class TestCUDAKernels:
    """Test CUDA kernel compilation and execution."""

    def test_cuda_kernels_import(self):
        """Test that CUDA kernels can be imported."""
        from prismo.backends.cuda_kernels import CUDAKernels, CUPY_AVAILABLE

        assert CUPY_AVAILABLE
        kernels = CUDAKernels()
        assert kernels is not None

    def test_cuda_kernel_compilation(self):
        """Test that CUDA kernels compile successfully."""
        from prismo.backends.cuda_kernels import CUDAKernels

        kernels = CUDAKernels()
        assert hasattr(kernels, "update_e_kernel_3d")
        assert hasattr(kernels, "update_h_kernel_3d")

    def test_cuda_kernel_launch(self):
        """Test launching a CUDA kernel with test data."""
        from prismo.backends.cuda_kernels import CUDAKernels

        kernels = CUDAKernels()

        # Create small test arrays
        nx, ny, nz = 10, 10, 10

        # Field arrays with correct Yee grid shapes
        Ex = cp.zeros((nx, ny - 1, nz - 1), dtype=cp.float64)
        Ey = cp.zeros((nx - 1, ny, nz - 1), dtype=cp.float64)
        Ez = cp.zeros((nx - 1, ny - 1, nz), dtype=cp.float64)
        Hx = cp.zeros((nx - 1, ny, nz), dtype=cp.float64)
        Hy = cp.zeros((nx, ny - 1, nz), dtype=cp.float64)
        Hz = cp.zeros((nx, ny, nz - 1), dtype=cp.float64)

        # Coefficient arrays
        Ca_ex = cp.ones((nx, ny - 1, nz - 1), dtype=cp.float64)
        Ca_ey = cp.ones((nx - 1, ny, nz - 1), dtype=cp.float64)
        Ca_ez = cp.ones((nx - 1, ny - 1, nz), dtype=cp.float64)
        Cb_ex = cp.ones((nx, ny - 1, nz - 1), dtype=cp.float64)
        Cb_ey = cp.ones((nx - 1, ny, nz - 1), dtype=cp.float64)
        Cb_ez = cp.ones((nx - 1, ny - 1, nz), dtype=cp.float64)

        dy, dz = 1.0, 1.0

        # Launch kernel - should not crash
        try:
            kernels.launch_e_update(
                Hx, Hy, Hz, Ex, Ey, Ez, Ca_ex, Ca_ey, Ca_ez, Cb_ex, Cb_ey, Cb_ez, dy, dz
            )
            cp.cuda.Stream.null.synchronize()
            success = True
        except Exception as e:
            success = False
            print(f"Kernel launch failed: {e}")

        assert success, "Kernel launch should succeed"


@pytest.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available or CUDA not working")
class TestCUDASolverIntegration:
    """Test CUDA backend integration with FDTD solver."""

    def test_solver_with_cupy_backend(self):
        """Test creating a solver with CuPy backend."""
        from prismo.core.grid import GridSpec, YeeGrid
        from prismo.core.solver import FDTDSolver

        backend = get_backend("cupy")

        # Create a small grid
        grid_spec = GridSpec(
            size=(1e-6, 1e-6, 1e-6),  # 1 micron cube
            resolution=1e8,  # 100 points per micron = 10 nm spacing
        )
        grid = YeeGrid(grid_spec)

        # Create solver with CuPy backend
        solver = FDTDSolver(grid, backend=backend)

        assert solver.backend.name == "cupy"
        assert solver.backend.is_gpu

    def test_solver_step_with_cupy(self):
        """Test running a solver step with CuPy backend."""
        from prismo.core.grid import GridSpec, YeeGrid
        from prismo.core.solver import FDTDSolver

        backend = get_backend("cupy")

        # Create a small grid
        grid_spec = GridSpec(
            size=(1e-6, 1e-6, 1e-6),
            resolution=1e8,
        )
        grid = YeeGrid(grid_spec)

        solver = FDTDSolver(grid, backend=backend)

        # Run a few steps
        initial_energy = solver.fields.get_field_energy()
        solver.run_steps(10)

        # Energy should change (or stay zero if no source)
        final_energy = solver.fields.get_field_energy()
        # Convert to float if it's a CuPy array
        if hasattr(final_energy, 'get'):
            final_energy = float(final_energy.get())
        assert isinstance(final_energy, (float, np.floating))

    def test_backend_equivalence_cpu_gpu(self):
        """Test that CPU and GPU backends produce equivalent results."""
        from prismo.core.grid import GridSpec, YeeGrid
        from prismo.core.solver import FDTDSolver

        # Create same grid for both
        grid_spec = GridSpec(
            size=(1e-6, 1e-6, 1e-6),
            resolution=1e8,
        )
        grid_cpu = YeeGrid(grid_spec)
        grid_gpu = YeeGrid(grid_spec)

        # Create solvers
        solver_cpu = FDTDSolver(grid_cpu, backend="numpy")
        solver_gpu = FDTDSolver(grid_gpu, backend="cupy")

        # Initialize with same source (if we had one)
        # For now, just run a few steps
        solver_cpu.run_steps(5)
        solver_gpu.run_steps(5)

        # Compare field values
        Ex_cpu = solver_cpu.fields.Ex
        Ex_gpu = solver_gpu.backend.to_numpy(solver_gpu.fields.Ex)

        # They should be close (allowing for floating point differences)
        assert np.allclose(Ex_cpu, Ex_gpu, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

