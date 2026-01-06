# Changelog

All notable changes to Prismo will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### GUI Enhancements
- **Embedded 3D Viewport**: Integrated 3D visualization directly in the main GUI window with interactive controls
- **Shape Dialog**: Interactive dialog for creating geometric shapes (Box, Sphere, Cylinder) with material assignment through the GUI
- **Results Viewer**: Built-in viewer for visualizing simulation results including:
  - Field plots (2D distributions with colormaps)
  - Frequency-domain spectra
  - S-parameters (reflection and transmission)
  - Time series data
- **Results Loader**: Support for loading results from:
  - CSV files (exported by CSVExporter)
  - Parquet files (exported by ParquetExporter)
  - Live simulation monitors
- **Enhanced Material Viewer**: Improved material property visualization with Unicode/Greek symbol support
- **Property Plotter**: Enhanced plotting capabilities for material properties and simulation data

#### Solver Integration
- **FEM Solver**: Initial integration of FEniCS/dolfinx for frequency-domain and eigenvalue electromagnetic problems
  - Support for frequency-domain field solving
  - Eigenvalue problem solving for waveguide modes
  - Note: Currently a placeholder implementation; full integration planned for future releases
- **Solver Abstraction**: Unified interface for all solver types (FDTD, MEEP, FEM) allowing seamless switching
- **Solver Selection**: Added `solver_type` parameter to `Simulation` class:
  - `"fdtd"` - Native FDTD solver (default)
  - `"meep"` - MIT MEEP wrapper
  - `"fem"` - FEniCS-based FEM solver

#### CLI Improvements
- Enhanced command-line interface with better error handling
- Improved GUI launch command (`prismo gui`)

#### Documentation
- **Solvers API Documentation**: Complete API reference for all solver classes
  - SolverBase (abstract base class)
  - TimeDomainSolver and FrequencyDomainSolver (base classes)
  - FDTDSolver, MEEPSolver, and FEMSolver implementations
- **Updated GUI Tutorial**: Comprehensive guide covering:
  - Shape Dialog usage
  - Results Viewer functionality
  - Loading results from files and monitors
  - Embedded viewport features

#### Development Environment
- Updated `flake.nix` with enhanced development environment setup
- Improved optional dependency management
- Better handling of GUI dependencies (Dear PyGui)

### Changed

- **GUI Framework**: Switched from PySide6 to Dear PyGui for lighter-weight GUI implementation
- **Solver Architecture**: Refactored solver system to use unified base classes for better extensibility
- **Material Backend**: Enhanced Metal backend support for macOS GPU acceleration

### Fixed

- Array broadcasting issues in Maxwell field updates for 2D simulations
- Documentation configuration for MyST parser
- Read the Docs configuration for proper dependency handling

### Notes

- **FEM Solver**: The FEniCS/dolfinx integration is currently a placeholder. Full implementation with mesh generation, weak form assembly, and boundary condition application is planned for future releases.
- **MEEP Solver**: The MEEP wrapper is functional but requires full integration with MEEP's geometry and material systems for complete feature parity.
- **Optional Dependencies**: 
  - FEniCS: Install via `conda install -c conda-forge fenics-dolfinx` or from source
  - MEEP: Install via `conda install -c conda-forge pymeeus meep` or from source
  - GUI: Install via `pip install dearpygui` or `pip install pyprismo[gui]`

## [Previous Releases]

See git tags and GitHub releases for previous version history.

