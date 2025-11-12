"""
Optimized Metal kernels for FDTD field updates.

This module provides custom Metal Shading Language (MSL) kernels for
high-performance Maxwell equation updates on GPU using Metal compute shaders.
"""

from typing import Any, Optional, Tuple

import numpy as np

try:
    import Metal
    from Metal import (
        MTLCommandBuffer,
        MTLCommandQueue,
        MTLComputeCommandEncoder,
        MTLComputePipelineState,
        MTLDevice,
        MTLOrigin,
        MTLResourceStorageModeShared,
        MTLSize,
    )

    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
    Metal = None


# Metal Shading Language kernel for E-field update (3D)
METAL_UPDATE_E_FIELD_3D = """
#include <metal_stdlib>
using namespace metal;

kernel void update_e_field_3d(
    device const float* Hx [[buffer(0)]],
    device const float* Hy [[buffer(1)]],
    device const float* Hz [[buffer(2)]],
    device float* Ex [[buffer(3)]],
    device float* Ey [[buffer(4)]],
    device float* Ez [[buffer(5)]],
    device const float* Ca_ex [[buffer(6)]],
    device const float* Ca_ey [[buffer(7)]],
    device const float* Ca_ez [[buffer(8)]],
    device const float* Cb_ex [[buffer(9)]],
    device const float* Cb_ey [[buffer(10)]],
    device const float* Cb_ez [[buffer(11)]],
    constant uint3& grid_size [[buffer(12)]],
    constant float& dy [[buffer(13)]],
    constant float& dz [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;
    
    uint nx = grid_size.x;
    uint ny = grid_size.y;
    uint nz = grid_size.z;
    
    // Bounds checking
    if (i >= nx || j >= ny-1 || k >= nz-1) return;
    
    uint idx = i * ny * nz + j * nz + k;
    
    // Ex update: ∂Ex/∂t = (1/ε) * (∂Hz/∂y - ∂Hy/∂z)
    uint idx_hz_j1 = i * ny * (nz-1) + (j+1) * (nz-1) + k;
    uint idx_hz_j0 = i * ny * (nz-1) + j * (nz-1) + k;
    uint idx_hy_k1 = i * (ny-1) * nz + j * nz + (k+1);
    uint idx_hy_k0 = i * (ny-1) * nz + j * nz + k;
    
    float curl_hz_y = (Hz[idx_hz_j1] - Hz[idx_hz_j0]) / dy;
    float curl_hy_z = (Hy[idx_hy_k1] - Hy[idx_hy_k0]) / dz;
    
    Ex[idx] = Ca_ex[idx] * Ex[idx] + Cb_ex[idx] * (curl_hz_y - curl_hy_z);
}
"""

# Metal Shading Language kernel for H-field update (3D)
METAL_UPDATE_H_FIELD_3D = """
#include <metal_stdlib>
using namespace metal;

kernel void update_h_field_3d(
    device const float* Ex [[buffer(0)]],
    device const float* Ey [[buffer(1)]],
    device const float* Ez [[buffer(2)]],
    device float* Hx [[buffer(3)]],
    device float* Hy [[buffer(4)]],
    device float* Hz [[buffer(5)]],
    device const float* Da [[buffer(6)]],
    device const float* Db [[buffer(7)]],
    constant uint3& grid_size [[buffer(8)]],
    constant float& dy [[buffer(9)]],
    constant float& dz [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    uint k = gid.z;
    
    uint nx = grid_size.x;
    uint ny = grid_size.y;
    uint nz = grid_size.z;
    
    // Bounds checking
    if (i >= nx-1 || j >= ny || k >= nz) return;
    
    uint idx = i * ny * nz + j * nz + k;
    
    // Hx update: ∂Hx/∂t = -(1/μ) * (∂Ez/∂y - ∂Ey/∂z)
    float curl_ez_y = 0.0;
    float curl_ey_z = 0.0;
    
    if (j < ny-1) {
        uint idx_ez_j1 = i * (ny-1) * nz + (j+1) * nz + k;
        uint idx_ez_j0 = i * (ny-1) * nz + j * nz + k;
        curl_ez_y = (Ez[idx_ez_j1] - Ez[idx_ez_j0]) / dy;
    }
    
    if (k < nz-1) {
        uint idx_ey_k1 = i * ny * (nz-1) + j * (nz-1) + (k+1);
        uint idx_ey_k0 = i * ny * (nz-1) + j * (nz-1) + k;
        curl_ey_z = (Ey[idx_ey_k1] - Ey[idx_ey_k0]) / dz;
    }
    
    Hx[idx] = Da[idx] * Hx[idx] - Db[idx] * (curl_ez_y - curl_ey_z);
}
"""

# 2D variants for optimized 2D simulations
METAL_UPDATE_E_FIELD_2D = """
#include <metal_stdlib>
using namespace metal;

kernel void update_e_field_2d(
    device const float* Hz [[buffer(0)]],
    device float* Ex [[buffer(1)]],
    device float* Ey [[buffer(2)]],
    device const float* Ca [[buffer(3)]],
    device const float* Cb [[buffer(4)]],
    constant uint2& grid_size [[buffer(5)]],
    constant float& dx [[buffer(6)]],
    constant float& dy [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    
    uint nx = grid_size.x;
    uint ny = grid_size.y;
    
    // Bounds checking
    if (i >= nx || j >= ny) return;
    
    uint idx = i * ny + j;
    
    // Ex update: ∂Ex/∂t = (1/ε) * ∂Hz/∂y
    if (j < ny-1) {
        uint idx_hz_j1 = i * ny + (j+1);
        uint idx_hz_j0 = i * ny + j;
        float curl_hz_y = (Hz[idx_hz_j1] - Hz[idx_hz_j0]) / dy;
        Ex[idx] = Ca[idx] * Ex[idx] + Cb[idx] * curl_hz_y;
    }
    
    // Ey update: ∂Ey/∂t = -(1/ε) * ∂Hz/∂x
    if (i < nx-1) {
        uint idx_hz_i1 = (i+1) * ny + j;
        uint idx_hz_i0 = i * ny + j;
        float curl_hz_x = (Hz[idx_hz_i1] - Hz[idx_hz_i0]) / dx;
        Ey[idx] = Ca[idx] * Ey[idx] - Cb[idx] * curl_hz_x;
    }
}
"""

METAL_UPDATE_H_FIELD_2D = """
#include <metal_stdlib>
using namespace metal;

kernel void update_h_field_2d(
    device const float* Ex [[buffer(0)]],
    device const float* Ey [[buffer(1)]],
    device float* Hz [[buffer(2)]],
    device const float* Da [[buffer(3)]],
    device const float* Db [[buffer(4)]],
    constant uint2& grid_size [[buffer(5)]],
    constant float& dx [[buffer(6)]],
    constant float& dy [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint i = gid.x;
    uint j = gid.y;
    
    uint nx = grid_size.x;
    uint ny = grid_size.y;
    
    // Bounds checking
    if (i >= nx || j >= ny) return;
    
    uint idx = i * ny + j;
    
    // Hz update: ∂Hz/∂t = -(1/μ) * (∂Ey/∂x - ∂Ex/∂y)
    float curl_ey_x = 0.0;
    float curl_ex_y = 0.0;
    
    if (i < nx-1) {
        uint idx_ey_i1 = (i+1) * ny + j;
        uint idx_ey_i0 = i * ny + j;
        curl_ey_x = (Ey[idx_ey_i1] - Ey[idx_ey_i0]) / dx;
    }
    
    if (j < ny-1) {
        uint idx_ex_j1 = i * ny + (j+1);
        uint idx_ex_j0 = i * ny + j;
        curl_ex_y = (Ex[idx_ex_j1] - Ex[idx_ex_j0]) / dy;
    }
    
    Hz[idx] = Da[idx] * Hz[idx] - Db[idx] * (curl_ey_x - curl_ex_y);
}
"""


class MetalKernels:
    """
    Manager for optimized Metal kernels.

    Provides compiled Metal kernels for high-performance FDTD updates.
    """

    def __init__(self, device: MTLDevice):
        if not METAL_AVAILABLE:
            raise RuntimeError("Metal required for Metal kernels")

        self.device = device
        self.command_queue = device.newCommandQueue()

        # Compile kernels
        self._compile_kernels()

    def _compile_kernels(self) -> None:
        """Compile MSL kernels to compute pipeline states."""
        self.pipelines = {}

        # Compile 3D E-field update kernel
        try:
            library = self.device.newLibraryWithSource_options_error_(
                METAL_UPDATE_E_FIELD_3D, None, None
            )
            if library is None:
                raise RuntimeError("Failed to create Metal library for E-field 3D")

            function = library.newFunctionWithName_("update_e_field_3d")
            if function is None:
                raise RuntimeError("Failed to get E-field 3D function")

            self.pipelines["update_e_field_3d"] = (
                self.device.newComputePipelineStateWithFunction_error_(function, None)[
                    0
                ]
            )
        except Exception as e:
            print(f"Warning: Failed to compile E-field 3D kernel: {e}")
            self.pipelines["update_e_field_3d"] = None

        # Compile 3D H-field update kernel
        try:
            library = self.device.newLibraryWithSource_options_error_(
                METAL_UPDATE_H_FIELD_3D, None, None
            )
            if library is None:
                raise RuntimeError("Failed to create Metal library for H-field 3D")

            function = library.newFunctionWithName_("update_h_field_3d")
            if function is None:
                raise RuntimeError("Failed to get H-field 3D function")

            self.pipelines["update_h_field_3d"] = (
                self.device.newComputePipelineStateWithFunction_error_(function, None)[
                    0
                ]
            )
        except Exception as e:
            print(f"Warning: Failed to compile H-field 3D kernel: {e}")
            self.pipelines["update_h_field_3d"] = None

        # Compile 2D kernels
        try:
            library = self.device.newLibraryWithSource_options_error_(
                METAL_UPDATE_E_FIELD_2D, None, None
            )
            if library is None:
                raise RuntimeError("Failed to create Metal library for E-field 2D")

            function = library.newFunctionWithName_("update_e_field_2d")
            if function is None:
                raise RuntimeError("Failed to get E-field 2D function")

            self.pipelines["update_e_field_2d"] = (
                self.device.newComputePipelineStateWithFunction_error_(function, None)[
                    0
                ]
            )
        except Exception as e:
            print(f"Warning: Failed to compile E-field 2D kernel: {e}")
            self.pipelines["update_e_field_2d"] = None

        try:
            library = self.device.newLibraryWithSource_options_error_(
                METAL_UPDATE_H_FIELD_2D, None, None
            )
            if library is None:
                raise RuntimeError("Failed to create Metal library for H-field 2D")

            function = library.newFunctionWithName_("update_h_field_2d")
            if function is None:
                raise RuntimeError("Failed to get H-field 2D function")

            self.pipelines["update_h_field_2d"] = (
                self.device.newComputePipelineStateWithFunction_error_(function, None)[
                    0
                ]
            )
        except Exception as e:
            print(f"Warning: Failed to compile H-field 2D kernel: {e}")
            self.pipelines["update_h_field_2d"] = None

    def launch_e_update_3d(
        self,
        Hx: Any,
        Hy: Any,
        Hz: Any,
        Ex: Any,
        Ey: Any,
        Ez: Any,
        Ca_ex: Any,
        Ca_ey: Any,
        Ca_ez: Any,
        Cb_ex: Any,
        Cb_ey: Any,
        Cb_ez: Any,
        grid_size: Tuple[int, int, int],
        dy: float,
        dz: float,
    ) -> None:
        """
        Launch E-field update kernel for 3D.

        Parameters
        ----------
        Hx, Hy, Hz : Metal buffers
            Magnetic field components.
        Ex, Ey, Ez : Metal buffers
            Electric field components (will be updated in-place).
        Ca_ex, Ca_ey, Ca_ez : Metal buffers
            Update coefficients.
        Cb_ex, Cb_ey, Cb_ez : Metal buffers
            Update coefficients.
        grid_size : Tuple[int, int, int]
            Grid dimensions (nx, ny, nz).
        dy, dz : float
            Grid spacing.
        """
        pipeline = self.pipelines.get("update_e_field_3d")
        if pipeline is None:
            raise RuntimeError("E-field 3D kernel not available")

        nx, ny, nz = grid_size

        # Create command buffer and encoder
        command_buffer = self.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        # Set compute pipeline
        encoder.setComputePipelineState_(pipeline)

        # Set buffers
        encoder.setBuffer_offset_atIndex_(Hx, 0, 0)
        encoder.setBuffer_offset_atIndex_(Hy, 0, 1)
        encoder.setBuffer_offset_atIndex_(Hz, 0, 2)
        encoder.setBuffer_offset_atIndex_(Ex, 0, 3)
        encoder.setBuffer_offset_atIndex_(Ey, 0, 4)
        encoder.setBuffer_offset_atIndex_(Ez, 0, 5)
        encoder.setBuffer_offset_atIndex_(Ca_ex, 0, 6)
        encoder.setBuffer_offset_atIndex_(Ca_ey, 0, 7)
        encoder.setBuffer_offset_atIndex_(Ca_ez, 0, 8)
        encoder.setBuffer_offset_atIndex_(Cb_ex, 0, 9)
        encoder.setBuffer_offset_atIndex_(Cb_ey, 0, 10)
        encoder.setBuffer_offset_atIndex_(Cb_ez, 0, 11)

        # Set grid size and spacing as constants
        grid_size_data = np.array([nx, ny, nz], dtype=np.uint32)
        grid_size_buffer = self.device.newBufferWithBytes_length_options_(
            grid_size_data.tobytes(),
            grid_size_data.nbytes,
            MTLResourceStorageModeShared,
        )
        encoder.setBuffer_offset_atIndex_(grid_size_buffer, 0, 12)

        dy_data = np.array([dy], dtype=np.float32)
        dy_buffer = self.device.newBufferWithBytes_length_options_(
            dy_data.tobytes(), dy_data.nbytes, MTLResourceStorageModeShared
        )
        encoder.setBuffer_offset_atIndex_(dy_buffer, 0, 13)

        dz_data = np.array([dz], dtype=np.float32)
        dz_buffer = self.device.newBufferWithBytes_length_options_(
            dz_data.tobytes(), dz_data.nbytes, MTLResourceStorageModeShared
        )
        encoder.setBuffer_offset_atIndex_(dz_buffer, 0, 14)

        # Calculate threadgroup size and grid size
        threadgroup_size = MTLSize(8, 8, 8)
        grid_size_metal = MTLSize(
            (nx + threadgroup_size.width - 1) // threadgroup_size.width,
            (ny + threadgroup_size.height - 1) // threadgroup_size.height,
            (nz + threadgroup_size.depth - 1) // threadgroup_size.depth,
        )

        # Dispatch compute
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            grid_size_metal, threadgroup_size
        )

        # End encoding and commit
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()

    def launch_h_update_3d(
        self,
        Ex: Any,
        Ey: Any,
        Ez: Any,
        Hx: Any,
        Hy: Any,
        Hz: Any,
        Da: Any,
        Db: Any,
        grid_size: Tuple[int, int, int],
        dy: float,
        dz: float,
    ) -> None:
        """
        Launch H-field update kernel for 3D.

        Parameters
        ----------
        Ex, Ey, Ez : Metal buffers
            Electric field components.
        Hx, Hy, Hz : Metal buffers
            Magnetic field components (will be updated in-place).
        Da, Db : Metal buffers
            Update coefficients.
        grid_size : Tuple[int, int, int]
            Grid dimensions (nx, ny, nz).
        dy, dz : float
            Grid spacing.
        """
        pipeline = self.pipelines.get("update_h_field_3d")
        if pipeline is None:
            raise RuntimeError("H-field 3D kernel not available")

        nx, ny, nz = grid_size

        # Create command buffer and encoder
        command_buffer = self.command_queue.commandBuffer()
        encoder = command_buffer.computeCommandEncoder()

        # Set compute pipeline
        encoder.setComputePipelineState_(pipeline)

        # Set buffers
        encoder.setBuffer_offset_atIndex_(Ex, 0, 0)
        encoder.setBuffer_offset_atIndex_(Ey, 0, 1)
        encoder.setBuffer_offset_atIndex_(Ez, 0, 2)
        encoder.setBuffer_offset_atIndex_(Hx, 0, 3)
        encoder.setBuffer_offset_atIndex_(Hy, 0, 4)
        encoder.setBuffer_offset_atIndex_(Hz, 0, 5)
        encoder.setBuffer_offset_atIndex_(Da, 0, 6)
        encoder.setBuffer_offset_atIndex_(Db, 0, 7)

        # Set grid size and spacing as constants
        grid_size_data = np.array([nx, ny, nz], dtype=np.uint32)
        grid_size_buffer = self.device.newBufferWithBytes_length_options_(
            grid_size_data.tobytes(),
            grid_size_data.nbytes,
            MTLResourceStorageModeShared,
        )
        encoder.setBuffer_offset_atIndex_(grid_size_buffer, 0, 8)

        dy_data = np.array([dy], dtype=np.float32)
        dy_buffer = self.device.newBufferWithBytes_length_options_(
            dy_data.tobytes(), dy_data.nbytes, MTLResourceStorageModeShared
        )
        encoder.setBuffer_offset_atIndex_(dy_buffer, 0, 9)

        dz_data = np.array([dz], dtype=np.float32)
        dz_buffer = self.device.newBufferWithBytes_length_options_(
            dz_data.tobytes(), dz_data.nbytes, MTLResourceStorageModeShared
        )
        encoder.setBuffer_offset_atIndex_(dz_buffer, 0, 10)

        # Calculate threadgroup size and grid size
        threadgroup_size = MTLSize(8, 8, 8)
        grid_size_metal = MTLSize(
            (nx + threadgroup_size.width - 1) // threadgroup_size.width,
            (ny + threadgroup_size.height - 1) // threadgroup_size.height,
            (nz + threadgroup_size.depth - 1) // threadgroup_size.depth,
        )

        # Dispatch compute
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            grid_size_metal, threadgroup_size
        )

        # End encoding and commit
        encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()


def get_optimal_threadgroup_size(
    grid_shape: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """
    Determine optimal Metal threadgroup size for given grid.

    Parameters
    ----------
    grid_shape : Tuple[int, int, int]
        Grid dimensions (nx, ny, nz).

    Returns
    -------
    Tuple[int, int, int]
        Optimal threadgroup size (tx, ty, tz).
    """
    # Standard starting point for Metal
    threadgroup_size = [8, 8, 8]

    # Adjust for small grids
    if grid_shape[0] < 8:
        threadgroup_size[0] = grid_shape[0]
    if grid_shape[1] < 8:
        threadgroup_size[1] = grid_shape[1]
    if grid_shape[2] < 8:
        threadgroup_size[2] = grid_shape[2]

    # Total threads per threadgroup should be <= 1024 for most Metal devices
    total = threadgroup_size[0] * threadgroup_size[1] * threadgroup_size[2]

    while total > 1024:
        # Reduce largest dimension
        max_idx = np.argmax(threadgroup_size)
        threadgroup_size[max_idx] = max(1, threadgroup_size[max_idx] // 2)
        total = threadgroup_size[0] * threadgroup_size[1] * threadgroup_size[2]

    return tuple(threadgroup_size)


def benchmark_metal_kernels(
    grid_size: Tuple[int, int, int], num_iterations: int = 100
) -> dict:
    """
    Benchmark Metal kernels performance.

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
    if not METAL_AVAILABLE:
        return {"error": "Metal not available"}

    import time

    # This would need to be implemented with actual Metal device
    # For now, return placeholder
    return {
        "grid_size": grid_size,
        "iterations": num_iterations,
        "backend": "metal",
        "note": "Benchmark not yet implemented - requires Metal device setup",
    }
