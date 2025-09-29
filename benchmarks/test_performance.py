"""
Performance benchmarks for Prismo FDTD solver.
"""

import pytest
import numpy as np
import time


class BenchmarkArrayOperations:
    """Benchmark basic array operations that form the foundation of FDTD."""

    def setup_method(self):
        """Set up test arrays."""
        self.size = 1000
        self.array_2d = np.random.rand(self.size, self.size)
        self.array_3d = np.random.rand(100, 100, 100)

    def test_array_multiplication(self, benchmark):
        """Benchmark element-wise array multiplication."""

        def multiply_arrays():
            return self.array_2d * self.array_2d

        result = benchmark(multiply_arrays)
        assert result.shape == self.array_2d.shape

    def test_array_addition(self, benchmark):
        """Benchmark array addition operations."""

        def add_arrays():
            return self.array_2d + self.array_2d

        result = benchmark(add_arrays)
        assert result.shape == self.array_2d.shape

    def test_fft_operations(self, benchmark):
        """Benchmark FFT operations (important for FDTD analysis)."""

        def fft_operation():
            return np.fft.fft2(self.array_2d)

        result = benchmark(fft_operation)
        assert result.shape == self.array_2d.shape


class BenchmarkFDTDOperations:
    """Benchmark operations specific to FDTD computations."""

    def setup_method(self):
        """Set up FDTD-like field arrays."""
        # Simulate typical FDTD grid sizes
        self.nx, self.ny, self.nz = 200, 200, 200

        # Electric field components
        self.Ex = np.random.rand(self.nx, self.ny - 1, self.nz - 1)
        self.Ey = np.random.rand(self.nx - 1, self.ny, self.nz - 1)
        self.Ez = np.random.rand(self.nx - 1, self.ny - 1, self.nz)

        # Magnetic field components
        self.Hx = np.random.rand(self.nx - 1, self.ny, self.nz)
        self.Hy = np.random.rand(self.nx, self.ny - 1, self.nz)
        self.Hz = np.random.rand(self.nx, self.ny, self.nz - 1)

    def test_curl_e_operation(self, benchmark):
        """Benchmark curl of E-field calculation (core FDTD operation)."""

        def curl_e():
            # Simplified curl E calculation (∇ × E)
            curl_ex = (self.Ez[:-1, 1:, :] - self.Ez[:-1, :-1, :]) - (
                self.Ey[:-1, :, 1:] - self.Ey[:-1, :, :-1]
            )
            curl_ey = (self.Ex[:, :-1, 1:] - self.Ex[:, :-1, :-1]) - (
                self.Ez[1:, :-1, :] - self.Ez[:-1, :-1, :]
            )
            curl_ez = (self.Ey[1:, :, :-1] - self.Ey[:-1, :, :-1]) - (
                self.Ex[:, 1:, :-1] - self.Ex[:, :-1, :-1]
            )
            return curl_ex, curl_ey, curl_ez

        result = benchmark(curl_e)
        assert len(result) == 3  # Three curl components

    def test_update_magnetic_field(self, benchmark):
        """Benchmark magnetic field update step."""

        def update_h():
            # Simplified H-field update (dH/dt = -∇ × E)
            dt = 1e-15  # 1 femtosecond
            mu0 = 4 * np.pi * 1e-7  # Permeability of free space

            curl_ex = self.Ez[:-1, 1:, :] - self.Ez[:-1, :-1, :]
            self.Hx -= (dt / mu0) * curl_ex
            return self.Hx

        result = benchmark(update_h)
        assert result.shape == self.Hx.shape


@pytest.mark.benchmark
class BenchmarkMemoryOperations:
    """Benchmark memory-intensive operations."""

    def test_large_array_allocation(self, benchmark):
        """Benchmark allocation of large arrays."""

        def allocate_large_array():
            return np.zeros((500, 500, 500))

        result = benchmark(allocate_large_array)
        assert result.size == 500**3

    def test_array_copying(self, benchmark):
        """Benchmark array copying operations."""
        source = np.random.rand(300, 300, 300)

        def copy_array():
            return np.copy(source)

        result = benchmark(copy_array)
        assert np.array_equal(result, source)


if __name__ == "__main__":
    # Run benchmarks directly
    print("Running Prismo performance benchmarks...")

    # Example of running a single benchmark
    bench = BenchmarkArrayOperations()
    bench.setup_method()

    start_time = time.time()
    result = bench.array_2d * bench.array_2d
    end_time = time.time()

    print(
        f"Array multiplication ({bench.size}x{bench.size}): {end_time - start_time:.4f} seconds"
    )
    print("For full benchmark suite, run: pytest benchmarks/ --benchmark-only")
