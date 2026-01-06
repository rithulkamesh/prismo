#!/bin/bash
# Setup CUDA environment for Nix systems
# This script sets up LD_LIBRARY_PATH for CUDA libraries

# Find NVIDIA driver libraries (64-bit)
NVIDIA_LIB=$(find /nix/store -path "*/nvidia-x11*/lib/libcuda.so*" -type f 2>/dev/null | head -1 | xargs dirname 2>/dev/null)

# Find CUDA toolkit libraries
CUDA_LIB=$(find /nix/store -path "*/cuda-merged-*/lib" -type d 2>/dev/null | head -1)

if [ -z "$NVIDIA_LIB" ] || [ -z "$CUDA_LIB" ]; then
    echo "⚠️  Warning: Could not find CUDA libraries in Nix store"
    echo "   You may need to install CUDA packages in your Nix environment"
    exit 1
fi

export LD_LIBRARY_PATH="$NVIDIA_LIB:$CUDA_LIB:${LD_LIBRARY_PATH:-}"

echo "✅ CUDA environment set up"
echo "   NVIDIA lib: $NVIDIA_LIB"
echo "   CUDA lib: $CUDA_LIB"
echo ""
echo "To use CUDA, run:"
echo "  export LD_LIBRARY_PATH=\"$NVIDIA_LIB:$CUDA_LIB:\$LD_LIBRARY_PATH\""
echo "  source .venv/bin/activate"
echo "  python -c 'import cupy as cp; print(\"CuPy version:\", cp.__version__); print(\"CUDA available:\", cp.cuda.is_available())'"

