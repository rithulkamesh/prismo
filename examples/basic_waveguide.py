"""
Basic waveguide simulation example.

This module provides a simple example of how to set up and run
a basic waveguide simulation using Prismo.
"""


def run_basic_simulation():
    """
    Run a basic 2D waveguide simulation.

    This example demonstrates:
    - Creating a simple silicon waveguide geometry
    - Adding a Gaussian pulse source
    - Setting up field monitors
    - Running the simulation
    - Extracting transmission data
    """
    print("🔬 Prismo Basic Waveguide Example")
    print("This is a placeholder for the actual simulation code.")
    print("The full implementation will be added during development.")

    # TODO: Implement actual simulation when core modules are ready
    # from prismo import Simulation, Rectangle, Gaussian, FieldMonitor

    # Example structure (not yet functional):
    # sim = Simulation(size=(10, 5, 0), resolution=20)
    # waveguide = Rectangle(center=(0, 0, 0), size=(10, 0.22, 0), material="Si")
    # source = Gaussian(center=(-4, 0, 0), frequency=200e12)
    # monitor = FieldMonitor(center=(4, 0, 0))

    # sim.add_structure(waveguide)
    # sim.add_source(source)
    # sim.add_monitor(monitor)
    # sim.run(time=20e-15)

    return {"status": "placeholder", "message": "Implementation pending"}


if __name__ == "__main__":
    result = run_basic_simulation()
    print(f"Result: {result}")
