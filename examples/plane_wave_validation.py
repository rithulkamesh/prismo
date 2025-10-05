"""
Basic plane wave propagation example.

This example demonstrates the simulation of a plane wave propagating
through a vacuum, validating the basic FDTD implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import prismo
from prismo import Simulation, PlaneWaveSource, FieldMonitor


def run_plane_wave_simulation():
    """Run a plane wave propagation simulation."""
    print("Setting up plane wave simulation...")

    # Create a simulation
    sim = Simulation(
        size=(2.0, 2.0, 0.0),  # 2µm x 2µm 2D grid
        resolution=20.0,  # 20 points per µm
        boundary_conditions="pml",
        pml_layers=10,
    )

    # Calculate wavelength for 193.4 THz (1550 nm)
    freq = 193.4e12  # Hz
    wavelength = 299792458.0 / freq  # c / f ≈ 1.55 µm
    print(f"Frequency: {freq/1e12:.1f} THz, Wavelength: {wavelength*1e6:.2f} µm")

    # Create a plane wave source
    source = PlaneWaveSource(
        center=(0.5, 1.0, 0.0),
        size=(1.5, 0.0, 0.0),  # Line source along x
        direction="+y",  # Propagating in +y direction
        polarization="z",  # Ez polarization (TM mode)
        frequency=freq,
        pulse=False,  # Continuous wave
        amplitude=1.0,
    )
    sim.add_source(source)

    # Add field monitor to record the field at all points
    monitor = FieldMonitor(
        center=(1.0, 1.0, 0.0),
        size=(1.8, 1.8, 0.0),
        components=["Ez"],
        time_domain=True,
    )
    sim.add_monitor(monitor)

    # Calculate number of time steps for 5 wave periods
    period = 1 / freq
    time_steps = int(5 * period / sim.dt)
    sim_time = time_steps * sim.dt
    print(f"Running for {time_steps} steps ({sim_time*1e15:.1f} fs)...")

    # Define progress callback
    def progress_callback(step, total_steps, sim_time, elapsed_time):
        if step % 100 == 0 or step == total_steps:
            print(
                f"Step {step}/{total_steps} ({step/total_steps*100:.1f}%), "
                f"Sim time: {sim_time*1e15:.2f} fs, "
                f"Elapsed: {elapsed_time:.2f} s"
            )

    # Run simulation
    sim.run(sim_time, progress_callback=progress_callback)
    print("Simulation completed!")

    # Get field data
    time_points, ez_data = monitor.get_time_data("Ez")

    return sim, monitor, ez_data, wavelength


def create_animation(ez_data, wavelength):
    """Create animation of the plane wave propagation."""
    print("Creating animation...")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Set up plot
    vmax = np.max(np.abs(ez_data))
    im = ax.imshow(
        ez_data[0],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        origin="lower",
        extent=[0, 2.0, 0, 2.0],  # Physical dimensions in µm
    )

    # Add colorbar and labels
    cbar = fig.colorbar(im)
    cbar.set_label("Ez Field Amplitude")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title("Plane Wave Propagation")

    # Add wavelength annotation
    ax.annotate(
        f"λ = {wavelength*1e6:.2f} µm",
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8),
    )

    # Create animation
    def update(frame):
        im.set_array(ez_data[frame])
        ax.set_title(f"Plane Wave Propagation (Frame {frame})")
        return [im]

    ani = FuncAnimation(fig, update, frames=len(ez_data), interval=100, blit=True)

    return fig, ani


if __name__ == "__main__":
    # Run the simulation
    sim, monitor, ez_data, wavelength = run_plane_wave_simulation()

    # Create animation
    fig, ani = create_animation(ez_data, wavelength)

    # Show the animation
    plt.show()

    # Save the last frame as an image
    plt.figure(figsize=(8, 8))
    plt.imshow(
        ez_data[-1],
        cmap="RdBu_r",
        vmin=-np.max(np.abs(ez_data)),
        vmax=np.max(np.abs(ez_data)),
        origin="lower",
        extent=[0, 2.0, 0, 2.0],
    )
    plt.colorbar(label="Ez Field Amplitude")
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title("Plane Wave Propagation (Final Frame)")
    plt.savefig("plane_wave_propagation.png", dpi=150, bbox_inches="tight")
    print("Saved final frame as plane_wave_propagation.png")
