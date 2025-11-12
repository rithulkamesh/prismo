"""
Metal Backend Demonstration

This example demonstrates the Metal backend capabilities on macOS,
including unified memory usage and performance comparisons.
"""

import platform
import time

import numpy as np

import prismo


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_metal_availability():
    """Check if Metal backend is available."""
    print_section("Metal Backend Availability")

    if platform.system() != "Darwin":
        print("❌ Metal backend requires macOS")
        return False

    backends = prismo.list_available_backends()
    print(f"Available backends: {backends}")

    if "metal" not in backends:
        print("❌ Metal backend not available")
        print("   Make sure you're on macOS with Metal framework")
        return False

    print("✅ Metal backend is available")
    return True


def demonstrate_metal_backend():
    """Demonstrate Metal backend features."""
    print_section("Metal Backend Features")

    # Set Metal backend
    backend = prismo.set_backend("metal")
    print(f"Using backend: {backend}")

    # Get device information
    mem_info = backend.get_memory_info()
    print(f"Device: {mem_info['device_name']}")
    print(f"Unified Memory: {mem_info['unified_memory']}")
    print(f"Storage Mode: {mem_info['storage_mode']}")
    print(f"Max Buffer Size: {mem_info['max_buffer_size'] / (1024**3):.1f} GB")

    # Test array operations
    print("\nTesting array operations...")

    # Create arrays
    arr1 = backend.zeros((100, 100))
    arr2 = backend.ones((100, 100))
    print("✅ Array creation successful")

    # Test mathematical operations
    sqrt_arr = backend.sqrt(arr2)
    exp_arr = backend.exp(arr1)
    sin_arr = backend.sin(arr2)
    print("✅ Mathematical operations successful")

    # Test reductions
    sum_result = backend.sum(arr2)
    max_result = backend.max(arr2)
    mean_result = backend.mean(arr2)
    print("✅ Reduction operations successful")

    # Test FFT
    complex_data = np.random.random(100) + 1j * np.random.random(100)
    arr_complex = backend.array(complex_data)
    fft_result = backend.fft(arr_complex)
    ifft_result = backend.ifft(fft_result)
    print("✅ FFT operations successful")

    # Test linear algebra
    a = backend.array(np.random.random((10, 10)))
    b = backend.array(np.random.random((10, 10)))
    dot_result = backend.dot(a, b)
    matmul_result = backend.matmul(a, b)
    print("✅ Linear algebra operations successful")


def performance_comparison():
    """Compare performance between backends."""
    print_section("Performance Comparison")

    # Test data
    size = (1000, 1000)
    data = np.random.random(size)

    # Test NumPy backend
    print("Testing NumPy backend...")
    backend_np = prismo.set_backend("numpy")
    arr_np = backend_np.array(data)

    start_time = time.time()
    for _ in range(10):
        result_np = backend_np.sqrt(arr_np)
    np_time = time.time() - start_time
    print(f"NumPy time: {np_time:.3f}s")

    # Test Metal backend
    if "metal" in prismo.list_available_backends():
        print("Testing Metal backend...")
        backend_metal = prismo.set_backend("metal")
        arr_metal = backend_metal.array(data)

        start_time = time.time()
        for _ in range(10):
            result_metal = backend_metal.sqrt(arr_metal)
        metal_time = time.time() - start_time
        print(f"Metal time: {metal_time:.3f}s")

        if metal_time < np_time:
            speedup = np_time / metal_time
            print(f"✅ Metal is {speedup:.1f}x faster than NumPy")
        else:
            print("ℹ️  Metal performance may be limited by current implementation")

    # Test CuPy backend if available
    if "cupy" in prismo.list_available_backends():
        print("Testing CuPy backend...")
        backend_cupy = prismo.set_backend("cupy")
        arr_cupy = backend_cupy.array(data)

        start_time = time.time()
        for _ in range(10):
            result_cupy = backend_cupy.sqrt(arr_cupy)
        cupy_time = time.time() - start_time
        print(f"CuPy time: {cupy_time:.3f}s")

        if cupy_time < np_time:
            speedup = np_time / cupy_time
            print(f"✅ CuPy is {speedup:.1f}x faster than NumPy")


def demonstrate_unified_memory():
    """Demonstrate unified memory features."""
    print_section("Unified Memory Demonstration")

    if "metal" not in prismo.list_available_backends():
        print("❌ Metal backend not available")
        return

    backend = prismo.set_backend("metal")
    mem_info = backend.get_memory_info()

    if not mem_info["unified_memory"]:
        print("ℹ️  Unified memory not available on this device")
        return

    print("✅ Using unified memory architecture")
    print("   - Zero-copy operations between CPU and GPU")
    print("   - Shared memory pool")
    print("   - Automatic memory coherency")

    # Create large array to demonstrate unified memory
    print("\nCreating large array...")
    large_size = (500, 500, 500)  # ~1GB array
    large_array = backend.zeros(large_size)
    print(f"✅ Created array of size {large_size}")

    # Perform operations on large array
    print("Performing operations on large array...")
    sqrt_result = backend.sqrt(large_array)
    sum_result = backend.sum(large_array)
    print("✅ Operations completed successfully")

    print("   This demonstrates unified memory's ability to handle large datasets")


def demonstrate_backend_switching():
    """Demonstrate switching between backends."""
    print_section("Backend Switching")

    data = np.random.random((100, 100))

    # Test switching between available backends
    backends_to_test = ["numpy"]

    if "metal" in prismo.list_available_backends():
        backends_to_test.append("metal")

    if "cupy" in prismo.list_available_backends():
        backends_to_test.append("cupy")

    for backend_name in backends_to_test:
        print(f"\nTesting {backend_name} backend...")
        backend = prismo.set_backend(backend_name)

        arr = backend.array(data)
        result = backend.sqrt(arr)

        print(f"✅ {backend_name} backend working correctly")
        print(f"   Backend: {backend}")


def main():
    """Run Metal backend demonstration."""
    print("\n" + "=" * 60)
    print("  PRISMO METAL BACKEND DEMONSTRATION")
    print("=" * 60)

    # Check availability
    if not check_metal_availability():
        print("\n❌ Metal backend not available. Exiting.")
        return 1

    try:
        # Demonstrate features
        demonstrate_metal_backend()

        # Performance comparison
        performance_comparison()

        # Unified memory demonstration
        demonstrate_unified_memory()

        # Backend switching
        demonstrate_backend_switching()

        print_section("Demonstration Complete")
        print("✅ All Metal backend features demonstrated successfully!")
        print("\nKey benefits of Metal backend:")
        print("  - Native macOS GPU acceleration")
        print("  - Unified memory on Apple Silicon")
        print("  - Zero-copy operations")
        print("  - Automatic device selection")
        print("  - Optimized for Apple hardware")

        return 0

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
