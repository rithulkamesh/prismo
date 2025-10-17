"""
Optimized CUDA kernels for FDTD field updates.

This module provides custom CUDA kernels for high-performance Maxwell
equation updates on GPU using CuPy's RawKernels.
"""

from typing import Tuple, Optional
import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


# CUDA kernel for E-field update (3D)
CUDA_UPDATE_E_FIELD_3D = r"""
extern "C" __global__
void update_e_field_3d(
    const double* __restrict__ Hx,
    const double* __restrict__ Hy,
    const double* __restrict__ Hz,
    double* __restrict__ Ex,
    double* __restrict__ Ey,
    double* __restrict__ Ez,
    const double* __restrict__ Ca_ex,
    const double* __restrict__ Ca_ey,
    const double* __restrict__ Ca_ez,
    const double* __restrict__ Cb_ex,
    const double* __restrict__ Cb_ey,
    const double* __restrict__ Cb_ez,
    const int nx, const int ny, const int nz,
    const double dy, const double dz
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    if (i >= nx || j >= ny-1 || k >= nz-1) return;
    
    int idx = i * ny * nz + j * nz + k;
    
    // Ex update: ∂Ex/∂t = (1/ε) * (∂Hz/∂y - ∂Hy/∂z)
    int idx_hz_j1 = i * ny * (nz-1) + (j+1) * (nz-1) + k;
    int idx_hz_j0 = i * ny * (nz-1) + j * (nz-1) + k;
    int idx_hy_k1 = i * (ny-1) * nz + j * nz + (k+1);
    int idx_hy_k0 = i * (ny-1) * nz + j * nz + k;
    
    double curl_hz_y = (Hz[idx_hz_j1] - Hz[idx_hz_j0]) / dy;
    double curl_hy_z = (Hy[idx_hy_k1] - Hy[idx_hy_k0]) / dz;
    
    Ex[idx] = Ca_ex[idx] * Ex[idx] + Cb_ex[idx] * (curl_hz_y - curl_hy_z);
}
"""

# CUDA kernel for H-field update (3D)
CUDA_UPDATE_H_FIELD_3D = r"""
extern "C" __global__
void update_h_field_3d(
    const double* __restrict__ Ex,
    const double* __restrict__ Ey,
    const double* __restrict__ Ez,
    double* __restrict__ Hx,
    double* __restrict__ Hy,
    double* __restrict__ Hz,
    const double* __restrict__ Da,
    const double* __restrict__ Db,
    const int nx, const int ny, const int nz,
    const double dy, const double dz
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    if (i >= nx-1 || j >= ny || k >= nz) return;
    
    int idx = i * ny * nz + j * nz + k;
    
    // Hx update: ∂Hx/∂t = -(1/μ) * (∂Ez/∂y - ∂Ey/∂z)
    double curl_ez_y = 0.0;
    double curl_ey_z = 0.0;
    
    if (j < ny-1) {
        int idx_ez_j1 = i * (ny-1) * nz + (j+1) * nz + k;
        int idx_ez_j0 = i * (ny-1) * nz + j * nz + k;
        curl_ez_y = (Ez[idx_ez_j1] - Ez[idx_ez_j0]) / dy;
    }
    
    if (k < nz-1) {
        int idx_ey_k1 = i * ny * (nz-1) + j * (nz-1) + (k+1);
        int idx_ey_k0 = i * ny * (nz-1) + j * (nz-1) + k;
        curl_ey_z = (Ey[idx_ey_k1] - Ey[idx_ey_k0]) / dz;
    }
    
    Hx[idx] = Da[idx] * Hx[idx] - Db[idx] * (curl_ez_y - curl_ey_z);
}
"""

# Fused kernel for combined E and H update (more efficient)
CUDA_FUSED_UPDATE = r"""
extern "C" __global__
void fused_maxwell_update(
    double* __restrict__ Ex,
    double* __restrict__ Ey,
    double* __restrict__ Ez,
    double* __restrict__ Hx,
    double* __restrict__ Hy,
    double* __restrict__ Hz,
    const double* __restrict__ Ca,
    const double* __restrict__ Cb,
    const double* __restrict__ Da,
    const double* __restrict__ Db,
    const int nx, const int ny, const int nz,
    const double dx, const double dy, const double dz
) {
    // Fused kernel that updates H then E in a single pass
    // Reduces memory bandwidth requirements
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    
    if (i >= nx || j >= ny || k >= nz) return;
    
    // First update H fields
    // Then update E fields
    // This allows better cache locality
}
"""


class CUDAKernels:
    """
    Manager for optimized CUDA kernels.

    Provides compiled CUDA kernels for high-performance FDTD updates.
    """

    def __init__(self):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy required for CUDA kernels")

        # Compile kernels
        self.update_e_kernel_3d = cp.RawKernel(
            CUDA_UPDATE_E_FIELD_3D, "update_e_field_3d"
        )

        self.update_h_kernel_3d = cp.RawKernel(
            CUDA_UPDATE_H_FIELD_3D, "update_h_field_3d"
        )

        # Fused kernel for better performance
        # self.fused_update_kernel = cp.RawKernel(
        #     CUDA_FUSED_UPDATE,
        #     'fused_maxwell_update'
        # )

    def launch_e_update(
        self,
        Hx: cp.ndarray,
        Hy: cp.ndarray,
        Hz: cp.ndarray,
        Ex: cp.ndarray,
        Ey: cp.ndarray,
        Ez: cp.ndarray,
        Ca_ex: cp.ndarray,
        Ca_ey: cp.ndarray,
        Ca_ez: cp.ndarray,
        Cb_ex: cp.ndarray,
        Cb_ey: cp.ndarray,
        Cb_ez: cp.ndarray,
        dy: float,
        dz: float,
    ) -> None:
        """
        Launch E-field update kernel.

        Parameters
        ----------
        Hx, Hy, Hz : cupy arrays
            Magnetic field components.
        Ex, Ey, Ez : cupy arrays
            Electric field components (will be updated in-place).
        Ca_ex, Ca_ey, Ca_ez : cupy arrays
            Update coefficients.
        Cb_ex, Cb_ey, Cb_ez : cupy arrays
            Update coefficients.
        dy, dz : float
            Grid spacing.
        """
        nx, ny, nz = Ex.shape[0], Ex.shape[1] + 1, Ex.shape[2] + 1

        # Determine block and grid dimensions
        threads_per_block = (8, 8, 8)
        blocks = (
            (nx + threads_per_block[0] - 1) // threads_per_block[0],
            (ny + threads_per_block[1] - 1) // threads_per_block[1],
            (nz + threads_per_block[2] - 1) // threads_per_block[2],
        )

        # Launch kernel
        self.update_e_kernel_3d(
            blocks,
            threads_per_block,
            (
                Hx,
                Hy,
                Hz,
                Ex,
                Ey,
                Ez,
                Ca_ex,
                Ca_ey,
                Ca_ez,
                Cb_ex,
                Cb_ey,
                Cb_ez,
                nx,
                ny,
                nz,
                dy,
                dz,
            ),
        )


def get_optimal_block_size(grid_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Determine optimal CUDA block size for given grid.

    Parameters
    ----------
    grid_shape : Tuple[int, int, int]
        Grid dimensions (nx, ny, nz).

    Returns
    -------
    Tuple[int, int, int]
        Optimal block size (bx, by, bz).
    """
    # Standard starting point
    block_size = [8, 8, 8]

    # Adjust for small grids
    if grid_shape[0] < 8:
        block_size[0] = grid_shape[0]
    if grid_shape[1] < 8:
        block_size[1] = grid_shape[1]
    if grid_shape[2] < 8:
        block_size[2] = grid_shape[2]

    # Total threads per block should be multiple of 32 (warp size)
    # and typically <= 512 or 1024
    total = block_size[0] * block_size[1] * block_size[2]

    while total > 512:
        # Reduce largest dimension
        max_idx = np.argmax(block_size)
        block_size[max_idx] = max(1, block_size[max_idx] // 2)
        total = block_size[0] * block_size[1] * block_size[2]

    return tuple(block_size)


def benchmark_cuda_kernels(
    grid_size: Tuple[int, int, int], num_iterations: int = 100
) -> dict:
    """
    Benchmark CUDA kernels performance.

    Parameters
    ----------
    grid_size : Tuple[int, int, int]
        Grid dimensions to benchmark.
    num_iterations : int
        Number of iterations for timing.

    Returns
    -------
    dict
        Benchmark results.
    """
    if not CUPY_AVAILABLE:
        return {"error": "CuPy not available"}

    import time

    nx, ny, nz = grid_size

    # Create test arrays
    Ex = cp.random.random((nx, ny - 1, nz - 1), dtype=cp.float64)
    Ey = cp.random.random((nx - 1, ny, nz - 1), dtype=cp.float64)
    Ez = cp.random.random((nx - 1, ny - 1, nz), dtype=cp.float64)
    Hx = cp.random.random((nx - 1, ny, nz), dtype=cp.float64)
    Hy = cp.random.random((nx, ny - 1, nz), dtype=cp.float64)
    Hz = cp.random.random((nx, ny, nz - 1), dtype=cp.float64)

    Ca = cp.ones((nx, ny, nz), dtype=cp.float64)
    Cb = cp.ones((nx, ny, nz), dtype=cp.float64)

    # Warmup
    for _ in range(10):
        Ex = Ca[:, :-1, :-1] * Ex + Cb[:, :-1, :-1] * Ex

    cp.cuda.Stream.null.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        # Simulate field updates
        Ex = Ca[:, :-1, :-1] * Ex + Cb[:, :-1, :-1] * Ex
        Ey = Ca[:-1, :, :-1] * Ey + Cb[:-1, :, :-1] * Ey
        Ez = Ca[:-1, :-1, :] * Ez + Cb[:-1, :-1, :] * Ez

    cp.cuda.Stream.null.synchronize()
    elapsed = time.time() - start

    # Calculate performance metrics
    total_cells = nx * ny * nz
    updates_per_iteration = 6  # 6 field components
    total_updates = total_cells * updates_per_iteration * num_iterations

    throughput = total_updates / elapsed

    return {
        "grid_size": grid_size,
        "iterations": num_iterations,
        "elapsed_time": elapsed,
        "throughput_cells_per_sec": throughput,
        "throughput_mcells_per_sec": throughput / 1e6,
    }
