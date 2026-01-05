"""
Command-line interface for Prismo.

This module provides command-line tools for running simulations,
processing results, and managing configurations.
"""

import argparse
import sys
from typing import Optional


def main(args: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="prismo", description="Prismo FDTD Solver for Waveguide Photonics"
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0-dev")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Simulation command
    sim_parser = subparsers.add_parser("simulate", help="Run FDTD simulation")
    sim_parser.add_argument("config", help="Configuration file path")
    sim_parser.add_argument("--output", "-o", help="Output directory")
    sim_parser.add_argument("--verbose", "-v", action="store_true")

    # Analysis command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze results")
    analyze_parser.add_argument("results", help="Results directory or file")
    analyze_parser.add_argument("--plot", action="store_true", help="Generate plots (matplotlib)")
    analyze_parser.add_argument("--gui", action="store_true", help="Open GUI viewer with results")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--gpu", action="store_true")

    # GUI command
    gui_parser = subparsers.add_parser("gui", help="Launch graphical user interface")
    gui_parser.add_argument("--theme", default="default", help="GUI theme")

    parsed_args = parser.parse_args(args)

    if parsed_args.command == "simulate":
        print(f"Running simulation with config: {parsed_args.config}")
        # TODO: Implement simulation runner
        return 0
    elif parsed_args.command == "analyze":
        results_path = parsed_args.results
        
        if parsed_args.gui:
            # Open GUI with results loaded
            try:
                from prismo.gui import GUI_AVAILABLE, MainWindow, get_gui_error_message
                from prismo.gui.results_loader import load_from_file
                from pathlib import Path

                if not GUI_AVAILABLE:
                    error_msg = get_gui_error_message()
                    print(f"Error: {error_msg}")
                    return 1

                # Load results file
                results_file = Path(results_path)
                if results_file.is_dir():
                    # Look for CSV or Parquet files in directory
                    csv_files = list(results_file.glob("*.csv"))
                    parquet_files = list(results_file.glob("*.parquet"))
                    if parquet_files:
                        results_file = parquet_files[0]
                    elif csv_files:
                        results_file = csv_files[0]
                    else:
                        print(f"Error: No CSV or Parquet files found in {results_path}")
                        return 1

                if not results_file.exists():
                    print(f"Error: Results file not found: {results_file}")
                    return 1

                # Create GUI window
                window = MainWindow()
                
                # Load results into viewer
                try:
                    data = load_from_file(results_file)
                    window.results_viewer.load_data(data)
                    # Open the results viewer section
                    import dearpygui.dearpygui as dpg
                    if dpg.does_item_exist("results_viewer_header"):
                        dpg.configure_item("results_viewer_header", default_open=True)
                    print(f"Loaded results from: {results_file}")
                except Exception as e:
                    print(f"Warning: Could not load results: {e}")
                    print("GUI will open without pre-loaded results")

                window.show()
                return 0
            except ImportError as e:
                print(f"Error launching GUI: {e}")
                print("Install GUI dependencies with: pip install dearpygui")
                return 1
        else:
            print(f"Analyzing results from: {results_path}")
            if parsed_args.plot:
                print("Generating plots...")
                # TODO: Implement matplotlib plotting
            # TODO: Implement analysis tools
            return 0
    elif parsed_args.command == "benchmark":
        print("Running benchmarks...")
        # TODO: Implement benchmark suite
        return 0
    elif parsed_args.command == "gui":
        try:
            from prismo.gui import GUI_AVAILABLE, MainWindow, get_gui_error_message

            if not GUI_AVAILABLE:
                error_msg = get_gui_error_message()
                print(f"Error: {error_msg}")
                return 1

            window = MainWindow()
            window.show()
            return 0
        except ImportError as e:
            print(f"Error launching GUI: {e}")
            print("Install GUI dependencies with: pip install dearpygui")
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
