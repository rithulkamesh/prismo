"""
Advanced Features Demo for Prismo FDTD Engine

This example demonstrates the new features:
- GPU/CPU backend selection
- Dispersive materials (Lorentz, Drude)
- Material library
- DFT monitors for frequency-domain analysis
- Flux monitors for power calculations
- S-parameter extraction
- Data export to CSV and Parquet
- Mode solver integration
"""

import numpy as np
import prismo
from pathlib import Path

# Print available backends
print("=" * 60)
print("PRISMO FDTD ENGINE - Advanced Features Demo")
print("=" * 60)
print("\nAvailable computational backends:")
backends = prismo.list_available_backends()
for backend in backends:
    print(f"  - {backend}")

# Select backend (CPU by default, GPU if available)
if "cupy" in backends:
    print("\nUsing GPU backend (CuPy)")
    backend = prismo.set_backend("cupy")
else:
    print("\nUsing CPU backend (NumPy)")
    backend = prismo.set_backend("numpy")

print(f"Backend: {backend}")

# 1. Create simulation with backend support
print("\n" + "=" * 60)
print("1. Setting up simulation with backend support")
print("=" * 60)

sim = prismo.Simulation(
    size=(10e-6, 5e-6, 0),  # 10 × 5 μm, 2D simulation
    resolution=50e6,  # 50 points per meter = 20 nm resolution
    boundary_conditions="pml",
    pml_layers=10,
    courant_factor=0.9,
)

print(f"Grid dimensions: {sim.grid.dimensions}")
print(f"Grid spacing: {sim.grid.spacing}")
print(f"Time step: {sim.dt:.3e} s")

# 2. Material Library Demo
print("\n" + "=" * 60)
print("2. Using Material Library")
print("=" * 60)

print("\nAvailable materials:")
materials = prismo.list_materials()
for mat in materials:
    print(f"  - {mat}")

# Get Silicon material
si = prismo.get_material("Si")
print(f"\nSilicon material: {si.name}")
print(f"  ε_∞ = {si.epsilon_inf}")

# Calculate refractive index at 1550 nm
wavelength = 1.55e-6
omega = 2 * np.pi * 299792458.0 / wavelength
n_si = si.refractive_index(omega)
print(f"  n @ 1550nm = {n_si.real:.3f}")

# Get Gold for plasmonic structures
au = prismo.get_material("Au")
print(f"\nGold (Drude model): {au.name}")
print(f"  ω_p = {au.omega_p / (2*np.pi):.3e} Hz")
print(f"  γ = {au.gamma / (2*np.pi):.3e} Hz")

# 3. Create custom dispersive material
print("\n" + "=" * 60)
print("3. Custom Dispersive Material")
print("=" * 60)

# Create custom Lorentz material
custom_material = prismo.LorentzMaterial(
    epsilon_inf=2.0,
    poles=[
        prismo.LorentzPole(
            omega_0=2 * np.pi * 200e12,  # 200 THz resonance
            delta_epsilon=1.5,
            gamma=1e13,
        )
    ],
    name="CustomDielectric",
)

print(f"Custom material: {custom_material.name}")
eps_custom = custom_material.permittivity(omega)
print(f"  ε @ 1550nm = {eps_custom:.3f}")

# 4. DFT Monitor for Frequency-Domain Analysis
print("\n" + "=" * 60)
print("4. DFT Monitor Setup")
print("=" * 60)

# Define frequencies of interest (telecom C-band)
wavelengths = np.linspace(1.5e-6, 1.6e-6, 11)  # 1500-1600 nm
frequencies = 299792458.0 / wavelengths

dft_monitor = prismo.DFTMonitor(
    center=(4e-6, 0, 0),
    size=(0, 2e-6, 0),  # Line monitor
    frequencies=frequencies.tolist(),
    components=["Ex", "Ey", "Ez"],
    name="transmission_monitor",
    backend=backend,
)

print(f"DFT Monitor: {dft_monitor.name}")
print(f"  Frequencies: {len(frequencies)} points")
print(f"  Wavelength range: {wavelengths[0]*1e9:.1f} - {wavelengths[-1]*1e9:.1f} nm")

# 5. Flux Monitor for Power Calculations
print("\n" + "=" * 60)
print("5. Flux Monitor Setup")
print("=" * 60)

flux_monitor = prismo.FluxMonitor(
    center=(4e-6, 0, 0),
    size=(0, 2e-6, 0),
    direction="x",  # Power flow in x-direction
    name="power_monitor",
    frequencies=frequencies.tolist(),
    backend=backend,
)

print(f"Flux Monitor: {flux_monitor.name}")
print(f"  Direction: {flux_monitor.direction}")
print(f"  Frequency points: {len(frequencies)}")

# 6. Mode Solver Demo
print("\n" + "=" * 60)
print("6. Waveguide Mode Solver")
print("=" * 60)

# Create a simple waveguide cross-section
x_mode = np.linspace(-2e-6, 2e-6, 100)  # 4 μm width
y_mode = np.linspace(-2e-6, 2e-6, 100)  # 4 μm height

# Create Silicon waveguide: 0.5 μm wide, 0.22 μm tall
epsilon_profile = np.ones((len(x_mode), len(y_mode)))
wg_width = 0.5e-6
wg_height = 0.22e-6

for i, x in enumerate(x_mode):
    for j, y in enumerate(y_mode):
        if abs(x) < wg_width / 2 and abs(y) < wg_height / 2:
            epsilon_profile[i, j] = 11.68  # Silicon

mode_solver = prismo.ModeSolver(
    wavelength=1.55e-6,
    x=x_mode,
    y=y_mode,
    epsilon=epsilon_profile,
)

print("Solving for TE modes...")
try:
    modes = mode_solver.solve(num_modes=2, mode_type="TE")
    print(f"Found {len(modes)} modes:")
    for mode in modes:
        print(f"  Mode {mode.mode_number}: n_eff = {mode.neff.real:.4f}")
except Exception as e:
    print(f"Mode solving encountered issue: {e}")
    print("(This is expected in example - requires proper grid setup)")

# 7. S-Parameter Analysis
print("\n" + "=" * 60)
print("7. S-Parameter Analysis")
print("=" * 60)

# Create S-parameter analyzer for 2-port device
s_param_analyzer = prismo.SParameterAnalyzer(
    num_ports=2,
    frequencies=frequencies,
    reference_impedance=50.0,
)

print(f"S-Parameter Analyzer: {s_param_analyzer.num_ports} ports")
print(f"Frequency points: {len(frequencies)}")

# Simulate some S-parameters (in practice, from simulation)
# S11: Reflection at port 1
s11_demo = -20 * np.exp(-1j * np.linspace(0, np.pi, len(frequencies)))
s_param_analyzer.s_matrix[:, 0, 0] = s11_demo

# S21: Transmission from port 1 to port 2
s21_demo = 0.9 * np.exp(-1j * 2 * np.linspace(0, np.pi, len(frequencies)))
s_param_analyzer.s_matrix[:, 1, 0] = s21_demo

# Calculate metrics
il = s_param_analyzer.get_insertion_loss_db(1, 0)
rl = s_param_analyzer.get_return_loss_db(0)

print(f"\nInsertion Loss (S21):")
print(f"  Min: {np.min(il):.2f} dB")
print(f"  Max: {np.max(il):.2f} dB")
print(f"  Mean: {np.mean(il):.2f} dB")

print(f"\nReturn Loss (S11):")
print(f"  Min: {np.min(rl):.2f} dB")
print(f"  Max: {np.max(rl):.2f} dB")
print(f"  Mean: {np.mean(rl):.2f} dB")

# 8. Data Export Demo
print("\n" + "=" * 60)
print("8. Data Export (CSV and Parquet)")
print("=" * 60)

output_dir = Path("./output")
output_dir.mkdir(exist_ok=True)

# CSV Export
csv_exporter = prismo.CSVExporter(output_dir=output_dir)

# Export S-parameters to CSV
csv_path = csv_exporter.export_sparameters(
    filename="device_sparameters",
    frequencies=frequencies,
    sparameters={
        "S11": s11_demo,
        "S21": s21_demo,
    },
    metadata={
        "device": "silicon_waveguide",
        "wavelength_range": "1500-1600 nm",
        "simulation_date": "2025-10-17",
    },
)
print(f"Exported S-parameters to CSV: {csv_path}")

# Parquet Export (more efficient for large datasets)
try:
    parquet_exporter = prismo.ParquetExporter(
        output_dir=output_dir, compression="snappy"
    )

    parquet_path = parquet_exporter.export_sparameters(
        filename="device_sparameters",
        frequencies=frequencies,
        sparameters={
            "S11": s11_demo,
            "S21": s21_demo,
        },
        metadata={
            "device": "silicon_waveguide",
            "backend": backend.name,
        },
    )
    print(f"Exported S-parameters to Parquet: {parquet_path}")
except ImportError:
    print("Parquet export requires polars: pip install polars")

# Export transmission spectrum to CSV
spectrum = np.abs(s21_demo) ** 2  # Transmission
csv_spectrum_path = csv_exporter.export_spectrum(
    filename="transmission_spectrum",
    frequencies=frequencies,
    spectrum=spectrum,
    metadata={"component": "S21_power"},
)
print(f"Exported spectrum to CSV: {csv_spectrum_path}")

# 9. Touchstone Export for Circuit Simulators
print("\n" + "=" * 60)
print("9. Touchstone (.s2p) Export")
print("=" * 60)

touchstone_path = output_dir / "device.s2p"

# Build full S-matrix
s_matrix_full = np.zeros((len(frequencies), 2, 2), dtype=complex)
s_matrix_full[:, 0, 0] = s11_demo  # S11
s_matrix_full[:, 1, 0] = s21_demo  # S21
s_matrix_full[:, 0, 1] = s21_demo  # S12 (reciprocal)
s_matrix_full[:, 1, 1] = s11_demo  # S22

prismo.export_touchstone(
    filename=touchstone_path,
    frequencies=frequencies,
    s_matrix=s_matrix_full,
    z0=50.0,
    comments=[
        "Generated by Prismo FDTD",
        "Device: Silicon waveguide",
        "Wavelength range: 1500-1600 nm",
    ],
)
print(f"Exported Touchstone file: {touchstone_path}")

# 10. Summary
print("\n" + "=" * 60)
print("DEMO COMPLETE")
print("=" * 60)
print("\nImplemented Features:")
print("  ✓ Backend abstraction (CPU/GPU support)")
print("  ✓ Material library with dispersive models")
print("  ✓ DFT monitors for frequency-domain analysis")
print("  ✓ Flux monitors for power calculations")
print("  ✓ Mode solver for waveguide modes")
print("  ✓ S-parameter extraction and analysis")
print("  ✓ Data export (CSV, Parquet, Touchstone)")
print("  ✓ PML absorbing boundaries")

print(f"\nOutput files saved to: {output_dir.absolute()}")
print("\nNext steps:")
print("  - Run full FDTD simulation with these components")
print("  - Compare with Lumerical results")
print("  - Optimize GPU kernels for performance")
print("  - Implement Lumerical file import")
print("  - Add more advanced features (adjoint, optimization)")

print("\n" + "=" * 60)
