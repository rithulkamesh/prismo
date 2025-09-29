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
    analyze_parser.add_argument("results", help="Results directory")
    analyze_parser.add_argument("--plot", action="store_true")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--gpu", action="store_true")

    parsed_args = parser.parse_args(args)

    if parsed_args.command == "simulate":
        print(f"Running simulation with config: {parsed_args.config}")
        # TODO: Implement simulation runner
        return 0
    elif parsed_args.command == "analyze":
        print(f"Analyzing results from: {parsed_args.results}")
        # TODO: Implement analysis tools
        return 0
    elif parsed_args.command == "benchmark":
        print("Running benchmarks...")
        # TODO: Implement benchmark suite
        return 0
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
