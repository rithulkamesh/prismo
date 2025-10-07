# Sources and Monitors

This guide covers all available electromagnetic sources and field monitoring capabilities in Prismo.

## Overview

Prismo provides several types of sources for exciting electromagnetic fields:
- **Point sources**: Dipoles and point excitations
- **Gaussian beams**: Focused beams with Gaussian profiles
- **Plane waves**: Uniform waves (basic and TFSF)
- **Custom sources**: User-defined field patterns

And comprehensive monitoring capabilities:
- **Field monitors**: Record E and H fields
- **Time-domain**: Full time evolution
- **Frequency-domain**: DFT at specific frequencies

---

## Sources

### Point Sources

#### Electric Dipole

An electric dipole radiates electromagnetic waves from a point source:

```python
from prismo import ElectricDipole

dipole = ElectricDipole(
    position=(1.0e-6, 1.0e-6, 0.0),    # Position in meters
    polarization="y",                   # "x", "y", or "z"
    frequency=200e12,                   # 200 THz
    pulse=True,                         # Pulsed excitation
    pulse_width=10e-15,                 # 10 fs pulse
    amplitude=1.0,
    phase=0.0,                          # Phase offset in radians
)
sim.add_source(dipole)
```

**Parameters:**
- `position`: (x, y, z) coordinates in meters
- `polarization`: Direction of electric field oscillation
- `frequency`: Center frequency in Hz
- `pulse`: True for Gaussian pulse, False for continuous wave
- `pulse_width`: Temporal width for Gaussian pulse (seconds)
- `amplitude`: Peak amplitude (V/m)
- `phase`: Phase offset in radians

**Use cases:**
- Single-molecule emission
- Antenna radiation patterns
- Near-field studies

#### Magnetic Dipole

Similar to electric dipole but excites magnetic field:

```python
from prismo import MagneticDipole

mag_dipole = MagneticDipole(
    position=(1.0e-6, 1.0e-6, 0.0),
    polarization="z",
    frequency=200e12,
    pulse=False,  # Continuous wave
    amplitude=1.0,
)
sim.add_source(mag_dipole)
```

**Use cases:**
- Magnetic materials
- Coil/loop antennas
- Complementary to electric dipoles

### Gaussian Beam Sources

Gaussian beams provide focused electromagnetic fields:

```python
from prismo import GaussianBeamSource

beam = GaussianBeamSource(
    center=(1.0e-6, 1.5e-6, 0.0),      # Beam center
    size=(0.0, 1.0e-6, 0.0),           # Source region (line source)
    direction="x",                      # Propagation direction
    polarization="y",                   # E-field polarization
    frequency=193.4e12,                 # 193.4 THz (1550 nm)
    beam_waist=0.5e-6,                 # 500 nm waist
    pulse=True,
    pulse_width=10e-15,
    amplitude=1.0,
)
sim.add_source(beam)
```

**Parameters:**
- `center`: Beam center position (m)
- `size`: Source region dimensions (m)
- `direction`: Propagation direction ("x", "y", "z")
- `polarization`: E-field polarization (perpendicular to direction)
- `frequency`: Center frequency (Hz)
- `beam_waist`: Minimum beam radius (m)
- `pulse`: Pulsed or continuous
- `pulse_width`: Pulse duration (s)

**Calculated properties:**
- Wavelength: λ = c/f
- Rayleigh range: zᵣ = πw₀²/λ
- Wave number: k = 2π/λ

**Use cases:**
- Focused illumination
- Coupling to waveguides
- Beam propagation studies

### Plane Wave Sources

#### Basic Plane Wave

Uniform plane wave excitation:

```python
from prismo import PlaneWaveSource

plane_wave = PlaneWaveSource(
    center=(0.5e-6, 1.0e-6, 0.0),
    size=(1.5e-6, 0.0, 0.0),          # Line source along x
    direction="+y",                    # Propagate in +y
    polarization="z",                  # Ez polarization
    frequency=193.4e12,
    pulse=False,                       # Continuous wave
    amplitude=1.0,
)
sim.add_source(plane_wave)
```

**Direction options:**
- `"+x"`, `"-x"`: Positive/negative x direction
- `"+y"`, `"-y"`: Positive/negative y direction
- `"+z"`, `"-z"`: Positive/negative z direction

#### TFSF Plane Wave Source (Recommended)

The **Total-Field/Scattered-Field (TFSF)** formulation provides artifact-free plane wave injection:

```python
from prismo import TFSFSource

tfsf = TFSFSource(
    center=(1.0e-6, 1.0e-6, 0.0),     # TFSF region center
    size=(1.0e-6, 1.0e-6, 0.0),       # TFSF region size
    direction="+x",                    # Propagation direction
    polarization="y",                  # E-field polarization
    frequency=150e12,                  # 150 THz
    pulse=False,                       # Continuous or pulsed
    amplitude=1.0,
)
sim.add_source(tfsf)
```

**How TFSF works:**
- Creates a boundary separating total-field and scattered-field regions
- **Inside boundary**: Total field (incident + scattered)
- **Outside boundary**: Scattered field only
- No reflections or artifacts at the TFSF interface

**Advantages over basic plane wave:**
- ✅ Clean plane wave injection
- ✅ No spurious reflections
- ✅ Ideal for validation and scattering studies
- ✅ Proper impedance matching (η₀ = 377Ω)

**Use cases:**
- Plane wave scattering
- Validation against analytical solutions
- Interface studies (Fresnel coefficients)
- Large-scale illumination

---

## Waveforms

All sources support different temporal waveforms:

### Gaussian Pulse

```python
from prismo.sources import GaussianPulse

waveform = GaussianPulse(
    frequency=200e12,      # Center frequency
    pulse_width=10e-15,    # Temporal width (FWHM)
    amplitude=1.0,
    phase=0.0,
)
```

**Equation:**
```
E(t) = A exp(-((t-t₀)/τ)²) cos(2πf₀t + φ)
```

### Continuous Wave

```python
from prismo.sources import ContinuousWave

waveform = ContinuousWave(
    frequency=200e12,
    amplitude=1.0,
    phase=0.0,
)
```

**Equation:**
```
E(t) = A cos(2πf₀t + φ)
```

### Ricker Wavelet

Second derivative of Gaussian (Mexican hat):

```python
from prismo.sources import RickerWavelet

waveform = RickerWavelet(
    frequency=200e12,
    pulse_width=10e-15,
    amplitude=1.0,
)
```

### Custom Waveform

Define your own waveform:

```python
from prismo.sources import CustomWaveform
import numpy as np

def my_waveform(t):
    """Custom temporal profile."""
    return np.sin(2*np.pi*200e12*t) * np.exp(-t/50e-15)

waveform = CustomWaveform(callable=my_waveform)
```

---

## Monitors

### Field Monitor

Record electromagnetic field data during simulation:

```python
from prismo import FieldMonitor

monitor = FieldMonitor(
    center=(1.0e-6, 1.0e-6, 0.0),        # Monitor center
    size=(1.8e-6, 1.8e-6, 0.0),          # Monitor region
    components=["Ey", "Hz"],              # Components to record
    time_domain=True,                     # Enable time-domain recording
    frequencies=[150e12, 200e12],         # Frequency-domain DFT
    name="main_monitor",
)
sim.add_monitor(monitor)
```

**Parameters:**
- `center`: Monitor center position (m)
- `size`: Monitor region size (m)
- `components`: Field components to record
  - Individual: `["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]`
  - Groups: `"E"` (all electric), `"H"` (all magnetic), `"all"`
- `time_domain`: Record every time step (bool)
- `frequencies`: List of frequencies for DFT (Hz)
- `name`: Optional identifier

**Monitor Types by Size:**

```python
# Point monitor (single location)
point_monitor = FieldMonitor(
    center=(1.0e-6, 1.0e-6, 0.0),
    size=(0.0, 0.0, 0.0),
    components=["Ey"],
)

# Line monitor (1D profile)
line_monitor = FieldMonitor(
    center=(1.0e-6, 1.0e-6, 0.0),
    size=(1.8e-6, 0.0, 0.0),  # Along x
    components=["Ey"],
)

# Plane monitor (2D slice)
plane_monitor = FieldMonitor(
    center=(1.0e-6, 1.0e-6, 0.0),
    size=(1.8e-6, 1.8e-6, 0.0),  # xy plane
    components=["Ey"],
)

# Volume monitor (full 3D)
volume_monitor = FieldMonitor(
    center=(1.0e-6, 1.0e-6, 0.5e-6),
    size=(1.8e-6, 1.8e-6, 0.8e-6),  # 3D region
    components="all",
)
```

### Retrieving Monitor Data

#### Time-Domain Data

```python
# Run simulation
sim.run(100e-15)

# Get time-domain data
time_points, field_data = monitor.get_time_data("Ey")

print(f"Time points shape: {time_points.shape}")
print(f"Field data shape: {field_data.shape}")

# field_data shape: (time_steps, ny, nx) for 2D
#                   (time_steps, nz, ny, nx) for 3D
```

#### Frequency-Domain Data

```python
# Get frequency-domain data (from DFT)
freq_field = monitor.get_frequency_data("Ey", 150e12)

# freq_field is complex: amplitude and phase
amplitude = np.abs(freq_field)
phase = np.angle(freq_field)
```

#### All Frequencies

```python
# Get all recorded frequencies
freq_data = monitor.get_all_frequency_data("Ey")

for freq, field in freq_data.items():
    print(f"Frequency: {freq/1e12:.1f} THz")
    print(f"Field shape: {field.shape}")
```

---

## Best Practices

### Source Placement

1. **Avoid PML regions**: Place sources away from absorbing boundaries
2. **TFSF regions**: Should fit comfortably within computational domain
3. **Dipoles**: At least 1 wavelength from boundaries

### Component Selection

For 2D simulations:
- **TM modes**: Use Ez, Hx, Hy
- **TE modes**: Use Hz, Ex, Ey

For plane waves:
- Polarization must be perpendicular to propagation direction

### Monitor Optimization

1. **Selective recording**: Only record needed components
2. **Region size**: Match to area of interest
3. **Frequency-domain**: Use DFT instead of storing all time steps
4. **Sampling**: Consider using every Nth time step for large simulations

### Time Duration

Choose simulation time based on:
- **Pulse sources**: 5-10× pulse width
- **Continuous wave**: Multiple periods for steady state
- **Resonances**: Wait for transients to decay

---

## Advanced Topics

### Multiple Sources

Add multiple sources with different properties:

```python
# Two interfering dipoles
dipole1 = ElectricDipole(position=(0.5e-6, 1.0e-6, 0.0), 
                          polarization="y", frequency=200e12)
dipole2 = ElectricDipole(position=(1.5e-6, 1.0e-6, 0.0), 
                          polarization="y", frequency=200e12, phase=np.pi)

sim.add_source(dipole1)
sim.add_source(dipole2)
```

### Enabling/Disabling Sources

```python
# Disable a source temporarily
source.enabled = False

# Re-enable
source.enabled = True
```

### Monitor Data Analysis

```python
# Calculate total energy
time_points, ey_data = monitor.get_time_data("Ey")
energy = np.sum(ey_data**2) * sim.dx * sim.dy * sim.dt

# Find peak field
max_field = np.max(np.abs(ey_data))
peak_time = time_points[np.argmax(np.abs(ey_data[:, ny//2, nx//2]))]

# Spectral analysis
from scipy.fft import fft, fftfreq
signal = ey_data[:, ny//2, nx//2]
spectrum = np.abs(fft(signal))
freqs = fftfreq(len(signal), d=sim.dt)
```

---

## Examples

See the `examples/` directory for complete demonstrations:
- `tfsf_plane_wave.py`: TFSF plane wave
- `basic_waveguide.py`: Gaussian beam in waveguide
- `plane_wave_validation.py`: Multiple source types

## Next Steps

- Learn about [Materials](materials.md) for dielectric structures
- Explore [Visualization](visualization.md) for plotting results
- Check [API Reference](../api/index.md) for complete documentation
