# Tutorial 3: S-Parameter Extraction

**Time**: 30 minutes  
**Difficulty**: Intermediate  
**Prerequisites**: Tutorials 1-2

## Learning Objectives

- Set up two-port measurements
- Extract S11 (reflection) and S21 (transmission)
- Calculate insertion loss and return loss
- Perform frequency sweeps

## The Device: Directional Coupler

We'll analyze a 2×2 directional coupler and extract its full S-matrix.

## Quick Implementation

```python
import numpy as np
from prismo import Simulation
from prismo.modes.solver import ModeSolver
from prismo.monitors.mode_monitor import ModeExpansionMonitor
from prismo.sources.mode import ModeSource

# Setup (simplified for clarity)
wavelength = 1.55e-6
frequencies = np.linspace(185e12, 200e12, 16)  # Frequency sweep

# 1. Solve for waveguide modes
mode_solver = ModeSolver(wavelength, x, y, epsilon)
modes = mode_solver.solve(num_modes=1, mode_type='TE')

# 2. Create mode source at port 1
source = ModeSource(
    center=(0.0, 0.0, port1_z),
    size=(4e-6, 4e-6, 0.0),
    mode=modes[0],
    direction='+z',
    waveform=pulse,
)

# 3. Add mode monitors at all ports
port1_monitor = ModeExpansionMonitor(
    center=(0.0, 0.0, port1_z),
    size=(4e-6, 4e-6, 0.0),
    modes=modes,
    direction='z',
    frequencies=frequencies.tolist(),
    name='port1'
)

port2_monitor = ModeExpansionMonitor(
    center=(0.0, 0.0, port2_z),
    size=(4e-6, 4e-6, 0.0),
    modes=modes,
    direction='z',
    frequencies=frequencies.tolist(),
    name='port2'
)

# Add monitors for ports 3 and 4...

sim.add_source(source)
sim.add_monitor(port1_monitor)
sim.add_monitor(port2_monitor)

# 4. Run simulation
sim.run(200e-15)

# 5. Extract S-parameters
s11 = port1_monitor.compute_s_parameters(source_mode_index=0)['S_11']
s21 = port2_monitor.compute_s_parameters(source_mode_index=0)['S_11']

# 6. Plot results
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Magnitude
axes[0].plot(frequencies/1e12, 20*np.log10(np.abs(s11)), label='S11 (Reflection)')
axes[0].plot(frequencies/1e12, 20*np.log10(np.abs(s21)), label='S21 (Transmission)')
axes[0].set_ylabel('Magnitude (dB)')
axes[0].set_title('S-Parameters vs Frequency')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Phase
axes[1].plot(frequencies/1e12, np.angle(s11, deg=True), label='S11 Phase')
axes[1].plot(frequencies/1e12, np.angle(s21, deg=True), label='S21 Phase')
axes[1].set_xlabel('Frequency (THz)')
axes[1].set_ylabel('Phase (degrees)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tutorial3_sparameters.png', dpi=150)
plt.show()

# 7. Calculate metrics
insertion_loss = -20 * np.log10(np.abs(s21))  # dB
return_loss = -20 * np.log10(np.abs(s11))  # dB

print(f"At center frequency:")
print(f"  Insertion Loss: {insertion_loss[len(frequencies)//2]:.2f} dB")
print(f"  Return Loss: {return_loss[len(frequencies)//2]:.2f} dB")
```

## Understanding S-Parameters

For a 2-port device:

- **S11**: Reflection at port 1 (how much reflects back)
- **S21**: Transmission from port 1 to port 2
- **S12**: Transmission from port 2 to port 1 (= S21 for reciprocal devices)
- **S22**: Reflection at port 2

```
     Port 1              Device              Port 2
       →  ────────────────────────────────  →
     S11 ←                                S21 →

```

## Key Metrics

```python
def analyze_sparameters(s11, s21):
    """Extract key metrics from S-parameters."""

    # Insertion Loss (IL)
    IL = -20 * np.log10(np.abs(s21))

    # Return Loss (RL)
    RL = -20 * np.log10(np.abs(s11))

    # Power transmission
    T = np.abs(s21)**2

    # Power reflection
    R = np.abs(s11)**2

    # Check energy conservation (for lossless device)
    conservation = T + R

    print(f"Insertion Loss: {np.mean(IL):.2f} ± {np.std(IL):.2f} dB")
    print(f"Return Loss: {np.mean(RL):.2f} ± {np.std(RL):.2f} dB")
    print(f"Power Conservation: {np.mean(conservation):.4f} (should be ~1.0)")

    return {'IL': IL, 'RL': RL, 'T': T, 'R': R}

metrics = analyze_sparameters(s11, s21)
```

## Validation Checks

```python
# 1. Reciprocity (S12 should equal S21)
s12 = port1_monitor_reversed.compute_s_parameters(source_mode_index=0)['S_11']
reciprocity_error = np.abs(s12 - s21) / np.abs(s21)
print(f"Reciprocity error: {np.max(reciprocity_error)*100:.2f}%")

# 2. Energy conservation
energy_balance = np.abs(s11)**2 + np.abs(s21)**2
print(f"Energy balance: {np.mean(energy_balance):.4f} (should be ~1.0 for lossless)")

# 3. Passivity (|S| ≤ 1 for passive devices)
max_s11 = np.max(np.abs(s11))
max_s21 = np.max(np.abs(s21))
print(f"Max |S11|: {max_s11:.4f} (should be ≤ 1.0)")
print(f"Max |S21|: {max_s21:.4f} (should be ≤ 1.0)")
```

## Touchstone File Export

Export S-parameters in standard format:

```python
def export_touchstone(frequencies, s_params, filename='device.s2p'):
    """Export 2-port S-parameters to Touchstone format."""

    with open(filename, 'w') as f:
        # Header
        f.write('! 2-port S-parameters from Prismo\n')
        f.write('# HZ S MA R 50\n')  # Hz, S-params, Magnitude-Angle, 50Ω

        # Data
        for i, freq in enumerate(frequencies):
            s11 = s_params['S11'][i]
            s12 = s_params['S12'][i]
            s21 = s_params['S21'][i]
            s22 = s_params['S22'][i]

            # Format: freq S11_mag S11_ang S21_mag S21_ang S12_mag S12_ang S22_mag S22_ang
            f.write(f"{freq:.6e} ")
            f.write(f"{np.abs(s11):.6e} {np.angle(s11, deg=True):.6e} ")
            f.write(f"{np.abs(s21):.6e} {np.angle(s21, deg=True):.6e} ")
            f.write(f"{np.abs(s12):.6e} {np.angle(s12, deg=True):.6e} ")
            f.write(f"{np.abs(s22):.6e} {np.angle(s22, deg=True):.6e}\n")

    print(f"✓ Exported to {filename}")

# Export
s_params_dict = {
    'S11': s11, 'S12': s12,
    'S21': s21, 'S22': s22,
}
export_touchstone(frequencies, s_params_dict, 'coupler.s2p')
```

## Exercises

1. Simulate a simple straight waveguide and verify S21 ≈ 1, S11 ≈ 0
2. Add losses (lossy material) and observe increased insertion loss
3. Create a Bragg grating and observe reflection peaks in S11
4. Measure 3 dB bandwidth of a resonator

## Summary

**Key Points:**

- Mode monitors enable S-parameter extraction
- Always check reciprocity and energy conservation
- Frequency sweeps reveal bandwidth and resonances
- Touchstone format enables integration with circuit simulators

## Next Steps

- {doc}`optimization` - Use S-parameters to optimize designs
- {doc}`../user_guide/mode_ports` - Advanced mode port techniques
- {doc}`../examples/index` - More S-parameter examples
