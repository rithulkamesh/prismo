"""
Test suite for Prismo core functionality.
"""

import numpy as np
import pytest


class TestPrismoCore:
    """Test cases for core Prismo functionality."""

    def test_import_prismo(self):
        """Test that prismo can be imported successfully."""
        import prismo

        assert prismo.__version__ is not None

    def test_prismo_version(self):
        """Test that version information is available."""
        import prismo

        assert isinstance(prismo.__version__, str)
        assert "dev" in prismo.__version__ or "." in prismo.__version__

    def test_module_structure(self):
        """Test that all expected modules are available."""
        import prismo

        expected_modules = [
            "core",
            "solvers",
            "materials",
            "boundaries",
            "sources",
            "monitors",
            "geometry",
            "utils",
            "visualization",
        ]

        for module_name in expected_modules:
            assert hasattr(prismo, module_name), f"Missing module: {module_name}"


class TestBasicFunctionality:
    """Test basic functionality that should work even in early development."""

    def test_numpy_integration(self):
        """Test that NumPy works correctly with our setup."""
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.sum() == 15

    def test_cli_import(self):
        """Test that CLI module can be imported."""
        from prismo.cli import main

        assert callable(main)

    @pytest.mark.slow
    def test_example_script(self):
        """Test that example scripts can be run."""
        # This is marked as slow since it's an integration test
        import os
        import sys

        # Add the examples directory to the path
        examples_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "examples"
        )
        sys.path.insert(0, examples_dir)

        from basic_waveguide import run_basic_simulation

        result = run_basic_simulation()
        assert result["status"] == "success"


# Performance benchmarks (will be expanded later)
class TestPerformance:
    """Performance and benchmark tests."""

    @pytest.mark.benchmark
    def test_array_operations_benchmark(self, benchmark):
        """Benchmark basic array operations."""

        def array_operation():
            arr = np.random.rand(1000, 1000)
            return np.sum(arr * arr)

        result = benchmark(array_operation)
        assert result > 0


# Integration tests for when modules are implemented
class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.integration
    @pytest.mark.skip(reason="Implementation pending")
    def test_complete_simulation_workflow(self):
        """Test a complete simulation from setup to results."""
        # This will be implemented when core modules are ready
        pass

    @pytest.mark.gpu
    @pytest.mark.skip(reason="GPU support pending")
    def test_gpu_acceleration(self):
        """Test GPU-accelerated computations."""
        # This will test CUDA/CuPy functionality when implemented
        pass
