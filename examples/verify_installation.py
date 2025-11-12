"""
Installation Verification Script

Run this script to verify that all Prismo features are working correctly.
"""

import sys


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_imports():
    """Check that all modules can be imported."""
    print_section("1. Checking Imports")

    try:
        import prismo

        print("✅ prismo package imported successfully")
        print(f"   Version: {prismo.__version__}")

        # Check all major components
        from prismo import (  # Backends; Export; Monitors; Lumerical; Materials; Mode solver; Optimization; Core; Analysis
            ADESolver,
            Backend,
            CSVExporter,
            DFTMonitor,
            DrudeMaterial,
            FDTDSolver,
            FluxMonitor,
            FSPParser,
            GridSpec,
            LorentzMaterial,
            ModeExpansionMonitor,
            ModeSolver,
            ParameterSweep,
            ParquetExporter,
            Simulation,
            SParameterAnalyzer,
            TensorMaterial,
            YeeGrid,
            export_touchstone,
            get_backend,
            get_material,
            import_lumerical_material,
            list_available_backends,
            list_materials,
            set_backend,
        )

        print("✅ All core components imported successfully")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def check_backends():
    """Check available backends."""
    print_section("2. Checking Backends")

    try:
        import prismo

        backends = prismo.list_available_backends()
        print(f"✅ Available backends: {backends}")

        # Test NumPy backend
        backend_np = prismo.set_backend("numpy")
        print(f"✅ NumPy backend: {backend_np}")

        # Test CuPy backend if available
        if "cupy" in backends:
            backend_gpu = prismo.set_backend("cupy")
            print(f"✅ CuPy backend: {backend_gpu}")
            print(f"   GPU info: {backend_gpu.get_memory_info()}")
        else:
            print(
                "ℹ️  CuPy backend not available (install with: pip install cupy-cuda12x)"
            )

        # Test Metal backend if available (macOS)
        if "metal" in backends:
            backend_metal = prismo.set_backend("metal")
            print(f"✅ Metal backend: {backend_metal}")
            mem_info = backend_metal.get_memory_info()
            print(f"   Device: {mem_info['device_name']}")
            print(f"   Unified Memory: {mem_info['unified_memory']}")
        else:
            print(
                "ℹ️  Metal backend not available (requires macOS with Metal framework)"
            )

        return True

    except Exception as e:
        print(f"❌ Backend error: {e}")
        return False


def check_materials():
    """Check material library."""
    print_section("3. Checking Material Library")

    try:
        import numpy as np

        import prismo

        materials = prismo.list_materials()
        print(f"✅ Material library loaded: {len(materials)} materials")
        print(f"   Available: {', '.join(materials)}")

        # Test getting a material
        si = prismo.get_material("Si")
        print(f"✅ Silicon material: {si.name}")

        # Calculate refractive index
        wavelength = 1.55e-6
        omega = 2 * np.pi * 299792458.0 / wavelength
        n = si.refractive_index(omega)
        print(f"   n @ 1550nm = {n.real:.3f} (expected ~3.48)")

        return True

    except Exception as e:
        print(f"❌ Material error: {e}")
        return False


def check_simulation():
    """Check simulation creation."""
    print_section("4. Checking Simulation Creation")

    try:
        import prismo

        sim = prismo.Simulation(
            size=(5e-6, 3e-6, 0),
            resolution=20e6,
            boundary_conditions="pml",
            pml_layers=5,
        )

        print("✅ Simulation created successfully")
        print(f"   Grid dimensions: {sim.grid.dimensions}")
        print(f"   Time step: {sim.dt:.3e} s")

        return True

    except Exception as e:
        print(f"❌ Simulation error: {e}")
        return False


def check_monitors():
    """Check monitor creation."""
    print_section("5. Checking Monitors")

    try:

        import prismo

        frequencies = [190e12, 193e12, 200e12]

        # DFT monitor
        prismo.DFTMonitor(
            center=(0, 0, 0),
            size=(0, 1e-6, 0),
            frequencies=frequencies,
        )
        print("✅ DFT Monitor created")

        # Flux monitor
        prismo.FluxMonitor(
            center=(0, 0, 0),
            size=(0, 1e-6, 0),
            direction="x",
            frequencies=frequencies,
        )
        print("✅ Flux Monitor created")

        return True

    except Exception as e:
        print(f"❌ Monitor error: {e}")
        return False


def check_export():
    """Check data export functionality."""
    print_section("6. Checking Data Export")

    try:
        from pathlib import Path

        import numpy as np

        import prismo

        # Create temporary output directory
        output_dir = Path("./test_output")
        output_dir.mkdir(exist_ok=True)

        # CSV exporter
        csv_exp = prismo.CSVExporter(output_dir=output_dir)
        print("✅ CSV Exporter created")

        # Test data
        frequencies = np.array([190e12, 193e12, 200e12])
        spectrum = np.array([0.8, 0.9, 0.7])

        csv_path = csv_exp.export_spectrum(
            filename="test_spectrum", frequencies=frequencies, spectrum=spectrum
        )
        print(f"✅ CSV export successful: {csv_path}")

        # Parquet exporter
        try:
            parquet_exp = prismo.ParquetExporter(output_dir=output_dir)
            parquet_path = parquet_exp.export_spectrum(
                filename="test_spectrum", frequencies=frequencies, spectrum=spectrum
            )
            print(f"✅ Parquet export successful: {parquet_path}")
        except ImportError:
            print("ℹ️  Parquet export requires polars (pip install polars)")

        # Cleanup
        import shutil

        shutil.rmtree(output_dir)

        return True

    except Exception as e:
        print(f"❌ Export error: {e}")
        return False


def check_analysis():
    """Check analysis tools."""
    print_section("7. Checking Analysis Tools")

    try:
        import numpy as np

        import prismo

        frequencies = np.array([190e12, 193e12, 200e12])

        # S-parameter analyzer
        s_analyzer = prismo.SParameterAnalyzer(num_ports=2, frequencies=frequencies)
        print("✅ S-Parameter Analyzer created")

        # Test S-parameter functions
        s_analyzer.s_matrix[0, 0, 0] = 0.1 + 0.05j
        s_analyzer.s_matrix[0, 1, 0] = 0.9 + 0.1j

        il = s_analyzer.get_insertion_loss_db(1, 0)
        print(f"✅ Insertion loss calculated: {il[0]:.2f} dB")

        return True

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return False


def print_summary(results):
    """Print summary of checks."""
    print_section("VERIFICATION SUMMARY")

    all_passed = all(results.values())

    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {check_name}")

    print("\n" + "-" * 70)

    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("\nPrismo is fully functional and ready to use.")
        print("\nNext steps:")
        print("  - Run examples/advanced_features_demo.py for feature demonstration")
        print("  - Read NEW_FEATURES.md for comprehensive documentation")
        print("  - Read QUICK_START.md for quick reference")
        print("  - Check FINAL_IMPLEMENTATION_REPORT.md for complete feature list")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("\nPlease install missing dependencies:")
        print('  pip install -e ".[all]"')
        print("\nOr for GPU support:")
        print('  pip install -e ".[acceleration]"')

    return all_passed


def main():
    """Run all verification checks."""
    print("\n" + "=" * 70)
    print("  PRISMO FDTD ENGINE - INSTALLATION VERIFICATION")
    print("=" * 70)

    results = {
        "Imports": check_imports(),
        "Backends": check_backends(),
        "Materials": check_materials(),
        "Simulation": check_simulation(),
        "Monitors": check_monitors(),
        "Export": check_export(),
        "Analysis": check_analysis(),
    }

    success = print_summary(results)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
