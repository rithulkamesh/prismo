"""
Basic waveguide simulation example.

This example demonstrates a simple 2D waveguide simulation with a Gaussian pulse
exciting a waveguide mode.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from prismo import Simulation, GaussianBeamSource, FieldMonitor


def run_basic_simulation():
    """
    Run a basic 2D waveguide simulation.

    This example demonstrates:
    - Creating a simple waveguide simulation domain
    - Adding a Gaussian beam source
    - Setting up field monitors
    - Running the simulation
    - Visualizing the fields
    """
    print("🔬 Prismo Basic Waveguide Example")

    # Create a simulation
    sim = Simulation(
        size=(5.0, 3.0, 0.0),  # 5µm x 3µm 2D grid
        resolution=20.0,  # 20 points per µm
        boundary_conditions="pml",
        pml_layers=10,
    )

    # Calculate wavelength for 193.4 THz (1550 nm)
    freq = 193.4e12  # Hz
    wavelength = 299792458.0 / freq  # c / f ≈ 1.55 µm
    print(f"Frequency: {freq/1e12:.1f} THz, Wavelength: {wavelength*1e6:.2f} µm")

    # Create a Gaussian beam source
    source = GaussianBeamSource(
        center=(1.0, 1.5, 0.0),
        size=(0.0, 1.0, 0.0),  # Line source along y
        direction="x",  # Propagating in x direction
        polarization="y",  # Ey polarization (TE mode)
        frequency=freq,
        beam_waist=0.5e-6,  # 500 nm beam waist
        pulse=True,  # Gaussian pulse
        pulse_width=10e-15,  # 10 fs pulse
        amplitude=1.0,
    )
    sim.add_source(source)

    # Add field monitor to record the field at all points
    monitor = FieldMonitor(
        center=(2.5, 1.5, 0.0),
        size=(4.5, 2.5, 0.0),
        components=["Ey"],
        time_domain=True,
    )
    sim.add_monitor(monitor)

    # Calculate number of time steps for 100 fs simulation
    sim_time = 100e-15  # 100 fs
    time_steps = int(sim_time / sim.dt)
    print(f"Running for {time_steps} steps ({sim_time*1e15:.1f} fs)...")

    # Define progress callback
    def progress_callback(step, total_steps, sim_time, elapsed_time):
        if step % 500 == 0 or step == total_steps:
            print(
                f"Step {step}/{total_steps} ({step/total_steps*100:.1f}%), "
                f"Sim time: {sim_time*1e15:.2f} fs, "
                f"Elapsed: {elapsed_time:.2f} s"
            )

    # Run simulation
    sim.run(sim_time, progress_callback=progress_callback)
    print("Simulation completed!")

    # Get field data
    time_points, ey_data = monitor.get_time_data("Ey")

    # Create a simple visualization of the final field
    plt.figure(figsize=(10, 6))
    vmax = np.max(np.abs(ey_data)) * 0.5  # Scale for better visibility
    plt.imshow(
        ey_data[-1],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        origin="lower",
        extent=[0, 5.0, 0, 3.0],  # Physical dimensions in µm
    )

    # Draw waveguide outline
    waveguide_y = [1.4, 1.4, 1.6, 1.6]
    waveguide_x = [1.0, 4.5, 4.5, 1.0]
    plt.plot(waveguide_x, waveguide_y, "k--", lw=1)

    plt.colorbar(label="Ey Field Amplitude")
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title(f"Waveguide Mode at t = {sim.current_time*1e15:.1f} fs")
    plt.savefig("waveguide_mode.png", dpi=150, bbox_inches="tight")
    print("Saved visualization as waveguide_mode.png")

    return {
        "status": "success",
        "time_steps": time_steps,
        "max_field": np.max(np.abs(ey_data)),
        "wavelength": wavelength,
        "dt": sim.dt,
    }


if __name__ == "__main__":
    result = run_basic_simulation()
    print(f"Result: {result}")
    plt.show()
