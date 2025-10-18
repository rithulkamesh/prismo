# Simulation Validation

Validating your FDTD simulations is crucial to ensure accuracy and reliability. This guide covers validation methods, best practices, and common checks.

## Why Validate?

FDTD simulations can produce incorrect results due to:

- Insufficient resolution
- Inappropriate time step
- Boundary reflections
- Source implementation errors
- Material model inaccuracies

**Always validate** against known solutions before trusting new simulations.

## Validation Hierarchy

```{mermaid}
graph TD
    A[Start Simulation] --> B[Basic Checks]
    B --> C[Analytical Validation]
    C --> D[Convergence Tests]
    D --> E[Physical Consistency]
    E --> F[Production Simulation]
```

## 1. Basic Sanity Checks

### Courant Stability Condition

FDTD requires dt ≤ Courant limit for stability:

```python
import numpy as np

def check_courant_condition(sim):
    """Verify Courant stability."""
    # 3D: dt ≤ 1 / (c * sqrt(1/dx² + 1/dy² + 1/dz²))
    # 2D: dt ≤ 1 / (c * sqrt(1/dx² + 1/dy²))

    c = 299792458.0  # Speed of light

    if sim.is_3d:
        courant_limit = 1.0 / (c * np.sqrt(
            1/sim.dx**2 + 1/sim.dy**2 + 1/sim.dz**2
        ))
    else:
        courant_limit = 1.0 / (c * np.sqrt(
            1/sim.dx**2 + 1/sim.dy**2
        ))

    courant_number = sim.dt / courant_limit

    print(f"Courant number: {courant_number:.4f}")
    print(f"  dt = {sim.dt:.3e} s")
    print(f"  Courant limit = {courant_limit:.3e} s")

    if courant_number > 1.0:
        raise ValueError("Unstable! Courant number > 1.0")
    elif courant_number > 0.99:
        print("Warning: Courant number very close to limit")
    else:
        print("✓ Courant condition satisfied")

    return courant_number

# Check your simulation
check_courant_condition(sim)
```

### Energy Conservation

In lossless regions, energy should be conserved:

```python
def check_energy_conservation(sim, monitors):
    """Check energy conservation over time."""
    # Calculate total electromagnetic energy
    # U = (ε₀ε|E|² + μ₀|H|²) / 2

    energies = []

    for monitor in monitors:
        Ex, Ey, Ez = monitor.get_field_data('Ex', 'Ey', 'Ez')
        Hx, Hy, Hz = monitor.get_field_data('Hx', 'Hy', 'Hz')

        # Electromagnetic energy density
        eps0 = 8.854187817e-12
        mu0 = 4 * np.pi * 1e-7

        U_E = 0.5 * eps0 * (np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2)
        U_H = 0.5 * mu0 * (np.abs(Hx)**2 + np.abs(Hy)**2 + np.abs(Hz)**2)

        total_energy = np.sum(U_E + U_H)
        energies.append(total_energy)

    # Check energy variation
    energy_variation = (np.max(energies) - np.min(energies)) / np.mean(energies)

    print(f"Energy variation: {energy_variation*100:.2f}%")

    if energy_variation < 0.01:
        print("✓ Energy well conserved")
    elif energy_variation < 0.05:
        print("⚠ Some energy loss (acceptable for lossy materials)")
    else:
        print("✗ Significant energy loss - check for numerical issues")

    return energies
```

### Field Divergence Check

Maxwell's equations require ∇·D = 0 in source-free regions:

```python
def check_divergence(fields, dx, dy):
    """Check ∇·D = 0."""
    # ∇·D = ε₀(∂Ex/∂x + ∂Ey/∂y + ∂Ez/∂z)

    dEx_dx = np.gradient(fields.Ex, dx, axis=0)
    dEy_dy = np.gradient(fields.Ey, dy, axis=1)

    divergence = dEx_dx + dEy_dy

    max_div = np.max(np.abs(divergence))
    max_field = np.max(np.abs([fields.Ex, fields.Ey]))

    relative_div = max_div / max_field if max_field > 0 else 0

    print(f"Max divergence (relative): {relative_div:.2e}")

    if relative_div < 1e-6:
        print("✓ Divergence well satisfied")
    else:
        print("⚠ Check source implementation")
```

## 2. Analytical Validation

### Plane Wave Propagation

Validate against analytical plane wave:

```python
def validate_plane_wave():
    """Compare FDTD result with analytical plane wave."""

    # Run FDTD simulation
    from prismo import Simulation
    from prismo.sources import TFSFSource

    wavelength = 1.55e-6
    frequency = 299792458.0 / wavelength

    sim = Simulation(
        size=(10e-6, 10e-6, 0.0),
        resolution=40e6,
        boundary_conditions='pml',
        pml_layers=10,
    )

    source = TFSFSource(
        center=(5e-6, 5e-6, 0.0),
        size=(6e-6, 6e-6, 0.0),
        direction='+x',
        polarization='y',
        frequency=frequency,
        pulse=False,  # CW for comparison
    )
    sim.add_source(source)

    monitor = FieldMonitor(
        center=(5e-6, 5e-6, 0.0),
        size=(8e-6, 8e-6, 0.0),
        components=['Ey'],
        time_domain=True,
    )
    sim.add_monitor(monitor)

    # Run simulation
    sim.run(100e-15)

    # Get FDTD result
    times, Ey_fdtd = monitor.get_time_data('Ey')

    # Analytical result
    k = 2 * np.pi / wavelength
    omega = 2 * np.pi * frequency
    x = sim.grid.x
    Ey_analytical = np.sin(k * x - omega * times[-1])

    # Compare
    error = np.abs(Ey_fdtd[-1, :, 0] - Ey_analytical)
    rms_error = np.sqrt(np.mean(error**2))

    print(f"RMS error vs analytical: {rms_error:.2e}")

    if rms_error < 0.01:
        print("✓ Excellent agreement with analytical solution")
    elif rms_error < 0.05:
        print("✓ Good agreement")
    else:
        print("✗ Poor agreement - check implementation")

    return rms_error

# Run validation
validate_plane_wave()
```

### Resonator Modes

Compare resonant frequencies with analytical solutions:

```python
def validate_cavity_modes():
    """Validate cavity resonances."""

    # Rectangular cavity: f_mnp = c/(2π) * sqrt((mπ/Lx)² + (nπ/Ly)² + (pπ/Lz)²)

    Lx, Ly = 5e-6, 5e-6
    m, n = 1, 1  # Mode numbers

    c = 299792458.0
    f_analytical = (c / (2 * np.pi)) * np.sqrt(
        (m * np.pi / Lx)**2 + (n * np.pi / Ly)**2
    )

    print(f"Analytical resonance (TE₁₁): {f_analytical/1e12:.3f} THz")

    # Run FDTD and extract resonance (would use DFT monitor)
    # f_fdtd = ... (from simulation)

    # error = abs(f_fdtd - f_analytical) / f_analytical
    # print(f"Error: {error*100:.2f}%")
```

## 3. Convergence Tests

### Spatial Convergence

Resolution should not affect results (for well-resolved simulations):

```python
def test_spatial_convergence():
    """Test convergence with grid refinement."""

    resolutions = [20e6, 40e6, 60e6, 80e6]  # Points per meter
    results = []

    for resolution in resolutions:
        sim = Simulation(
            size=(5e-6, 5e-6, 0.0),
            resolution=resolution,
            boundary_conditions='pml',
        )

        # ... add source and monitors ...

        sim.run(50e-15)

        # Extract metric (e.g., transmission)
        transmission = monitor.get_transmission()
        results.append(transmission)

        print(f"Resolution {resolution/1e6:.0f} ppμm: T = {transmission:.6f}")

    # Check convergence
    relative_change = abs(results[-1] - results[-2]) / results[-1]

    if relative_change < 0.001:
        print("✓ Converged (< 0.1% change)")
    else:
        print(f"⚠ Not fully converged ({relative_change*100:.2f}% change)")

    return results
```

### Temporal Convergence

For broadband sources, simulate long enough:

```python
def check_temporal_convergence(monitor, threshold=0.01):
    """Check if fields have decayed sufficiently."""

    times, Ex_data = monitor.get_time_data('Ex')

    # Check last 10% of simulation
    late_time = Ex_data[-len(times)//10:]
    initial_max = np.max(np.abs(Ex_data[:len(times)//10]))
    late_max = np.max(np.abs(late_time))

    decay_ratio = late_max / initial_max if initial_max > 0 else 0

    print(f"Late-time field ratio: {decay_ratio:.2e}")

    if decay_ratio < threshold:
        print("✓ Fields decayed sufficiently")
    else:
        print("⚠ Run longer for full decay")

    return decay_ratio
```

## 4. Physical Consistency Checks

### Reciprocity

S-parameters should satisfy reciprocity: S₁₂ = S₂₁

```python
def check_reciprocity(s_parameters):
    """Verify reciprocity of S-parameters."""

    S12 = s_parameters['S12']
    S21 = s_parameters['S21']

    error = np.abs(S12 - S21) / np.abs(S21)
    max_error = np.max(error)

    print(f"Reciprocity error: {max_error*100:.2f}%")

    if max_error < 0.01:
        print("✓ Reciprocity satisfied")
    else:
        print("✗ Reciprocity violated - check implementation")
```

### Energy Balance

Power in = Power out + Power absorbed:

```python
def check_energy_balance(flux_monitors):
    """Verify energy conservation via flux monitors."""

    power_in = flux_monitors['input'].get_power()
    power_out = flux_monitors['output'].get_power()
    power_absorbed = flux_monitors['absorption'].get_power()

    balance = power_in - (power_out + power_absorbed)
    relative_balance = balance / power_in if power_in > 0 else 0

    print(f"Energy balance error: {relative_balance*100:.2f}%")

    if abs(relative_balance) < 0.05:
        print("✓ Energy balanced")
    else:
        print("⚠ Energy imbalance - check monitors or PML")
```

### Causality

Effects cannot precede causes:

```python
def check_causality(monitor, source_start_time):
    """Ensure no fields before source turns on."""

    times, Ex_data = monitor.get_time_data('Ex')

    pre_source_idx = times < source_start_time
    pre_source_field = Ex_data[pre_source_idx]

    max_pre_source = np.max(np.abs(pre_source_field))

    if max_pre_source < 1e-10:
        print("✓ Causality satisfied")
    else:
        print("✗ Non-causal behavior detected")
```

## 5. Common Validation Examples

### Example: Waveguide Mode Validation

```python
def validate_waveguide_modes():
    """Validate mode solver against known results."""

    from prismo.modes.solver import ModeSolver

    # Silicon strip waveguide (well-studied)
    wavelength = 1.55e-6
    width = 0.5e-6
    height = 0.22e-6

    # Create structure
    nx, ny = 100, 100
    x = np.linspace(-2*width, 2*width, nx)
    y = np.linspace(-2*height, 2*height, ny)

    X, Y = np.meshgrid(x, y, indexing='ij')
    epsilon = np.ones((nx, ny)) * 1.45**2  # SiO2
    core_mask = (np.abs(X) < width/2) & (np.abs(Y) < height/2)
    epsilon[core_mask] = 3.48**2  # Si

    # Solve
    solver = ModeSolver(wavelength, x, y, epsilon)
    modes = solver.solve(num_modes=1, mode_type='TE')

    neff_fdtd = modes[0].neff.real

    # Compare with known value (from literature/commercial tools)
    neff_reference = 2.45  # Example reference value

    error = abs(neff_fdtd - neff_reference) / neff_reference

    print(f"FDTD neff: {neff_fdtd:.4f}")
    print(f"Reference neff: {neff_reference:.4f}")
    print(f"Error: {error*100:.2f}%")

    if error < 0.01:
        print("✓ Mode solver validated")
    else:
        print("⚠ Check mode solver implementation or reference")
```

## Validation Checklist

Before trusting your simulation:

- [ ] Courant condition satisfied (dt < Courant limit)
- [ ] Resolution adequate (≥20 points per wavelength)
- [ ] PML reflections minimal (< -30 dB)
- [ ] Fields decayed at end of simulation
- [ ] Energy conserved (for lossless regions)
- [ ] Results converged with resolution
- [ ] Validated against analytical solution (if available)
- [ ] Physical constraints satisfied (reciprocity, causality, etc.)

## Automated Validation Script

```python
class SimulationValidator:
    """Automated validation checks."""

    def __init__(self, sim):
        self.sim = sim
        self.checks = []

    def run_all_checks(self):
        """Run all validation checks."""

        print("="*60)
        print("SIMULATION VALIDATION")
        print("="*60)

        # Courant
        self.check_courant()

        # Resolution
        self.check_resolution()

        # PML
        self.check_pml_reflections()

        # Energy
        self.check_energy_conservation()

        # Summary
        self.print_summary()

    def check_courant(self):
        courant = check_courant_condition(self.sim)
        self.checks.append(('Courant', courant < 1.0))

    def check_resolution(self):
        wavelength = 1.55e-6  # Typical
        ppw = wavelength / self.sim.dx  # Points per wavelength
        self.checks.append(('Resolution', ppw >= 20))
        print(f"Points per wavelength: {ppw:.1f}")

    def check_pml_reflections(self):
        # Would need to measure actual reflections
        pass

    def check_energy_conservation(self):
        # Would need energy monitors
        pass

    def print_summary(self):
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)

        passed = sum(1 for _, result in self.checks if result)
        total = len(self.checks)

        for name, result in self.checks:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{name:20s}: {status}")

        print(f"\nPassed {passed}/{total} checks")

        if passed == total:
            print("✓ All validations passed!")
        else:
            print("⚠ Some validations failed - review results carefully")

# Usage
validator = SimulationValidator(sim)
validator.run_all_checks()
```

## When to Suspect Problems

🚨 **Red flags**:

- Fields grow exponentially (unstable)
- Energy increases over time (non-physical)
- Results change significantly with small parameter changes
- Symmetry not preserved (for symmetric structures)
- Reflections from "absorbing" boundaries

## See Also

- {doc}`quickstart` - Getting started
- {doc}`simulations` - Simulation setup
- {doc}`boundaries` - Boundary conditions
- {doc}`../examples/index` - Validation examples
- Python validation examples in `tests/validation/`
