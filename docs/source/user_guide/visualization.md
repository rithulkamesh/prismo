# Visualization

Visualize simulation results and analysis data.

## Field Visualization

### 2D Field Plots

```python
import matplotlib.pyplot as plt
import numpy as np

# Get field data from monitor
field_data = sim.get_field_data(monitor, 'Ez')

# Plot
plt.figure(figsize=(10, 6))
plt.imshow(field_data.T, cmap='RdBu', origin='lower')
plt.colorbar(label='Ez (V/m)')
plt.xlabel('x (grid points)')
plt.ylabel('y (grid points)')
plt.title('Electric Field (Ez component)')
plt.show()
```

### Mode Profiles

```python
# Visualize calculated modes
mode = mode_solver.get_mode(0)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Electric field components
axes[0, 0].imshow(mode.Ex.real.T, cmap='RdBu')
axes[0, 0].set_title('Ex')

axes[0, 1].imshow(mode.Ey.real.T, cmap='RdBu')
axes[0, 1].set_title('Ey')

axes[0, 2].imshow(mode.Ez.real.T, cmap='RdBu')
axes[0, 2].set_title('Ez')

# Magnetic field components
axes[1, 0].imshow(mode.Hx.real.T, cmap='RdBu')
axes[1, 0].set_title('Hx')

axes[1, 1].imshow(mode.Hy.real.T, cmap='RdBu')
axes[1, 1].set_title('Hy')

axes[1, 2].imshow(mode.Hz.real.T, cmap='RdBu')
axes[1, 2].set_title('Hz')

plt.tight_layout()
plt.show()
```

## Spectrum Visualization

### Transmission Spectrum

```python
# Get transmission data
transmission = flux_monitor.get_frequency_domain_power()
wavelengths = 299792458.0 / frequencies * 1e9  # Convert to nm

plt.figure(figsize=(10, 6))
plt.plot(wavelengths, transmission, 'b-', linewidth=2)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmission')
plt.title('Device Transmission Spectrum')
plt.grid(True, alpha=0.3)
plt.show()
```

### S-Parameter Plots

```python
# Plot S21 (transmission) in dB
s21_db = -10 * np.log10(np.abs(s21)**2)

plt.figure(figsize=(10, 6))
plt.plot(wavelengths, s21_db, 'r-', label='S21', linewidth=2)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Insertion Loss (dB)')
plt.title('S21 Transmission')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Parameter Sweep Results

```python
# 1D sweep
sweep.plot_sweep_1d(
    x_param='width',
    y_metrics=['transmission', 'bandwidth'],
    save_path='sweep_width.png'
)

# 2D heatmap
sweep.plot_sweep_2d(
    x_param='width',
    y_param='height',
    metric='transmission',
    save_path='sweep_2d.png'
)
```

## Using Polars for Analysis

```python
import polars as pl

# Load Parquet data
df = pl.read_parquet("results/device_sparams.parquet")

# Filter and analyze
high_transmission = df.filter(
    pl.col('S21_magnitude') > 0.9
)

# Calculate derived quantities
df = df.with_columns([
    (-10 * pl.col('S21_magnitude').log10()).alias('IL_dB'),
    (-10 * pl.col('S11_magnitude').log10()).alias('RL_dB'),
])

# Plot with matplotlib
plt.plot(df['frequency_Hz'], df['IL_dB'])
```

## Animation

### Time-Domain Animation

```python
import matplotlib.animation as animation

# Collect field snapshots
snapshots = []  # List of field arrays at different times

fig, ax = plt.subplots()
im = ax.imshow(snapshots[0], cmap='RdBu', animated=True)

def update(frame):
    im.set_array(snapshots[frame])
    return [im]

ani = animation.FuncAnimation(
    fig, update, frames=len(snapshots),
    interval=50, blit=True
)

ani.save('field_evolution.gif', writer='pillow')
```
