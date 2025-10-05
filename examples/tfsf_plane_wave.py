"""
Simple TFSF plane wave example to validate implementation.

This demonstrates the TFSF (Total-Field/Scattered-Field) formulation
for clean plane wave injection into FDTD simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from prismo import Simulation, TFSFSource, FieldMonitor


def main():
    """Run a simple TFSF plane wave simulation."""
    print("🔬 Prismo TFSF Plane Wave Example")

    # Create simulation
    sim = Simulation(
        size=(2.0, 2.0, 0.0),  # 2µm x 2µm 2D grid
        resolution=40.0,  # 40 points per µm
        boundary_conditions="pml",
        pml_layers=10,
    )

    # Wave parameters
    freq = 150e12  # 150 THz (2 µm wavelength)
    wavelength = 299792458.0 / freq
    print(f"Frequency: {freq/1e12:.1f} THz, Wavelength: {wavelength*1e6:.2f} µm")

    # Create TFSF source for clean plane wave injection
    source = TFSFSource(
        center=(1.0, 1.0, 0.0),
        size=(1.0, 1.0, 0.0),  # TFSF region in center
        direction="+x",  # Propagating in +x direction
        polarization="y",  # Ey polarization (TE mode)
        frequency=freq,
        pulse=False,  # Continuous wave
        amplitude=1.0,
    )
    sim.add_source(source)
    print(f"Added TFSF source: {source}")

    # Add field monitor
    monitor = FieldMonitor(
        center=(1.0, 1.0, 0.0),
        size=(1.8, 1.8, 0.0),
        components=["Ey"],
        time_domain=True,
    )
    sim.add_monitor(monitor)

    # Run for a few periods
    periods = 5
    sim_time = periods / freq
    print(f"Running for {periods} periods ({sim_time*1e15:.1f} fs)...")

    # Progress callback
    def progress_callback(step, total_steps, sim_time, elapsed_time):
        if step % 100 == 0 or step == total_steps:
            print(f"Step {step}/{total_steps} ({step/total_steps*100:.1f}%)")

    sim.run(sim_time, progress_callback=progress_callback)
    print("Simulation completed!")

    # Get field data
    time_points, ey_data = monitor.get_time_data("Ey")
    print(f"Captured {len(time_points)} time steps")
    print(f"Field data shape: {ey_data.shape}")

    # Plot final field distribution
    plt.figure(figsize=(10, 8))
    vmax = np.max(np.abs(ey_data)) * 0.8
    plt.imshow(
        ey_data[-1],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        origin="lower",
        extent=[0, 2.0, 0, 2.0],
    )
    plt.colorbar(label="Ey (V/m)")
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title(f"TFSF Plane Wave at t = {periods} periods")

    # Mark TFSF boundary
    plt.plot(
        [0.5, 1.5, 1.5, 0.5, 0.5],
        [0.5, 0.5, 1.5, 1.5, 0.5],
        "k--",
        linewidth=2,
        label="TFSF boundary",
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig("tfsf_plane_wave.png", dpi=150)
    print("Saved visualization to tfsf_plane_wave.png")

    # Plot field along center line
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Spatial profile at final time
    center_y_idx = ey_data.shape[1] // 2
    x_coords = np.linspace(0, 2.0, ey_data.shape[2])
    ax1.plot(x_coords, ey_data[-1, center_y_idx, :])
    ax1.axvline(0.5, color="k", linestyle="--", alpha=0.5, label="TFSF boundary")
    ax1.axvline(1.5, color="k", linestyle="--", alpha=0.5)
    ax1.set_xlabel("x (µm)")
    ax1.set_ylabel("Ey (V/m)")
    ax1.set_title("Spatial Profile (center line)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Temporal evolution at center point
    center_x_idx = ey_data.shape[2] // 2
    ax2.plot(time_points * 1e15, ey_data[:, center_y_idx, center_x_idx])
    ax2.set_xlabel("Time (fs)")
    ax2.set_ylabel("Ey (V/m)")
    ax2.set_title("Temporal Evolution (center point)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("tfsf_analysis.png", dpi=150)
    print("Saved analysis to tfsf_analysis.png")


if __name__ == "__main__":
    main()
