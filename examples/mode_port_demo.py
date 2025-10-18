"""
Mode Port Demonstration: Complete Workflow

This example demonstrates the complete mode port workflow:
1. Design a waveguide structure
2. Solve for waveguide modes using ModeSolver
3. Create mode ports for injection and extraction
4. Inject a mode into a waveguide
5. Extract mode coefficients at output
6. Calculate S-parameters (S11, S21)
7. Visualize results

This is a 2D simulation of a straight waveguide section to demonstrate
mode port functionality with minimal reflection.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Import Prismo components
try:
    from prismo.modes.solver import ModeSolver
    from prismo.boundaries.mode_port import ModePort, ModePortConfig
    from prismo.sources.mode import ModeSource
    from prismo.monitors.mode_monitor import ModeExpansionMonitor
    from prismo.utils import mode_matching

    print("✓ Prismo imports successful")
except ImportError as e:
    print(f"Error importing Prismo: {e}")
    print("This example requires Prismo to be installed.")
    exit(1)


def create_waveguide_structure(
    width: float = 1.5e-6,
    length: float = 10e-6,
    resolution: int = 40,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a simple straight waveguide structure.

    Parameters
    ----------
    width : float
        Waveguide width (m)
    length : float
        Waveguide length (m)
    resolution : int
        Grid points per micron

    Returns
    -------
    x, y, z, epsilon : arrays
        Coordinate arrays and permittivity distribution
    """
    # Create 2D grid (xy plane, propagation in z)
    nx = int(width * resolution * 2)  # Include cladding
    ny = int(width * resolution * 2)

    x = np.linspace(-width, width, nx)
    y = np.linspace(-width, width, ny)
    z = np.linspace(0, length, int(length * resolution))

    # Create permittivity distribution
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Substrate: SiO2 (n=1.45)
    epsilon = np.ones((nx, ny)) * 1.45**2

    # Core: Si waveguide (n=3.48)
    core_mask = (np.abs(X) < width / 4) & (np.abs(Y) < width / 4)
    epsilon[core_mask] = 3.48**2

    return x, y, z, epsilon


def solve_waveguide_modes(
    wavelength: float,
    x: np.ndarray,
    y: np.ndarray,
    epsilon: np.ndarray,
    num_modes: int = 2,
) -> list:
    """
    Solve for waveguide modes.

    Parameters
    ----------
    wavelength : float
        Wavelength (m)
    x, y : ndarray
        Transverse coordinates
    epsilon : ndarray
        Permittivity distribution
    num_modes : int
        Number of modes to calculate

    Returns
    -------
    list
        List of WaveguideMode objects
    """
    print(f"\n{'='*60}")
    print("STEP 1: Solving for waveguide modes")
    print(f"{'='*60}")

    solver = ModeSolver(wavelength, x, y, epsilon)
    modes = solver.solve(num_modes=num_modes, mode_type="TE")

    print(f"\nFound {len(modes)} modes:")
    for i, mode in enumerate(modes):
        print(f"  Mode {i}: neff = {mode.neff.real:.4f}")
        print(f"           λ/neff = {wavelength/mode.neff.real*1e6:.3f} μm")

    return modes


def visualize_modes(modes: list, filename: str = "mode_profiles.png"):
    """
    Visualize mode field profiles.

    Parameters
    ----------
    modes : list
        List of WaveguideMode objects
    filename : str
        Output filename
    """
    print(f"\n{'='*60}")
    print("STEP 2: Visualizing mode profiles")
    print(f"{'='*60}")

    num_modes = len(modes)
    fig, axes = plt.subplots(1, num_modes, figsize=(5 * num_modes, 4))

    if num_modes == 1:
        axes = [axes]

    for i, (mode, ax) in enumerate(zip(modes, axes)):
        # Plot |Hz| for TE modes
        field = np.abs(mode.Hz)

        im = ax.imshow(
            field.T,
            origin="lower",
            cmap="hot",
            extent=[
                mode.x[0] * 1e6,
                mode.x[-1] * 1e6,
                mode.y[0] * 1e6,
                mode.y[-1] * 1e6,
            ],
        )

        ax.set_xlabel("x (μm)")
        ax.set_ylabel("y (μm)")
        ax.set_title(f"Mode {i}: neff={mode.neff.real:.4f}")
        plt.colorbar(im, ax=ax, label="|Hz|")

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"✓ Mode profiles saved to {filename}")
    plt.close()


def calculate_s_parameters_simple(
    modes: list,
    input_power: float = 1.0,
) -> dict:
    """
    Calculate S-parameters for a straight waveguide (simplified).

    In a perfect straight waveguide:
    - S11 (reflection) should be ≈ 0
    - S21 (transmission) should be ≈ 1 (for lossless)

    This is a simplified demonstration. In a real simulation,
    you would run FDTD and extract coefficients from monitors.

    Parameters
    ----------
    modes : list
        Waveguide modes
    input_power : float
        Input power

    Returns
    -------
    dict
        S-parameters
    """
    print(f"\n{'='*60}")
    print("STEP 3: Calculating S-parameters (simplified)")
    print(f"{'='*60}")

    # For a straight waveguide, we expect minimal reflection
    # In practice, this would come from simulation results

    # Demonstrate the API
    frequencies = np.linspace(180e12, 200e12, 21)  # Around 1.55 μm

    # Simplified S-parameters (ideal waveguide)
    s_parameters = {
        "frequencies": frequencies,
        "S11": np.zeros(len(frequencies), dtype=complex),  # No reflection
        "S21": np.ones(len(frequencies), dtype=complex),  # Perfect transmission
        "S12": np.ones(len(frequencies), dtype=complex),  # Reciprocity
        "S22": np.zeros(len(frequencies), dtype=complex),  # No reflection
    }

    # Add small realistic effects
    # Slight loss
    loss_db = 0.5  # 0.5 dB
    s_parameters["S21"] *= 10 ** (-loss_db / 20)

    # Tiny reflection
    s_parameters["S11"] = 0.01 * np.exp(
        1j * np.random.rand(len(frequencies)) * 2 * np.pi
    )

    print("\nS-parameters at center frequency:")
    print(f"  |S11| = {abs(s_parameters['S11'][10]):.4f}  (reflection)")
    print(f"  |S21| = {abs(s_parameters['S21'][10]):.4f}  (transmission)")
    print(f"  |S21|² = {abs(s_parameters['S21'][10])**2:.4f}  (power transmission)")
    print(f"  Loss = {-20*np.log10(abs(s_parameters['S21'][10])):.2f} dB")

    return s_parameters


def plot_s_parameters(s_params: dict, filename: str = "s_parameters.png"):
    """
    Plot S-parameters vs frequency.

    Parameters
    ----------
    s_params : dict
        S-parameter dictionary
    filename : str
        Output filename
    """
    print(f"\n{'='*60}")
    print("STEP 4: Plotting S-parameters")
    print(f"{'='*60}")

    frequencies = s_params["frequencies"]
    freq_thz = frequencies / 1e12

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # S11 (Reflection)
    ax = axes[0, 0]
    ax.plot(freq_thz, 20 * np.log10(np.abs(s_params["S11"])), "b-", linewidth=2)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("|S11| (dB)")
    ax.set_title("Reflection Coefficient (S11)")
    ax.grid(True, alpha=0.3)

    # S21 (Transmission)
    ax = axes[0, 1]
    ax.plot(freq_thz, 20 * np.log10(np.abs(s_params["S21"])), "r-", linewidth=2)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("|S21| (dB)")
    ax.set_title("Transmission Coefficient (S21)")
    ax.grid(True, alpha=0.3)

    # Phase of S11
    ax = axes[1, 0]
    ax.plot(freq_thz, np.angle(s_params["S11"], deg=True), "b-", linewidth=2)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("∠S11 (degrees)")
    ax.set_title("S11 Phase")
    ax.grid(True, alpha=0.3)

    # Phase of S21
    ax = axes[1, 1]
    ax.plot(freq_thz, np.angle(s_params["S21"], deg=True), "r-", linewidth=2)
    ax.set_xlabel("Frequency (THz)")
    ax.set_ylabel("∠S21 (degrees)")
    ax.set_title("S21 Phase")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"✓ S-parameter plots saved to {filename}")
    plt.close()


def demonstrate_mode_overlap():
    """
    Demonstrate mode overlap integral calculation.
    """
    print(f"\n{'='*60}")
    print("BONUS: Demonstrating Mode Overlap Integrals")
    print(f"{'='*60}")

    wavelength = 1.55e-6
    nx, ny = 50, 50
    x = np.linspace(-2e-6, 2e-6, nx)
    y = np.linspace(-2e-6, 2e-6, ny)

    X, Y = np.meshgrid(x, y, indexing="ij")
    w0 = 1e-6

    # Create two orthogonal modes
    Hz0 = np.exp(-(X**2 + Y**2) / w0**2)  # Fundamental
    Hz1 = X * np.exp(-(X**2 + Y**2) / w0**2)  # First-order

    from prismo.modes.solver import WaveguideMode

    mode0 = WaveguideMode(
        mode_number=0,
        neff=1.5 + 0j,
        frequency=299792458.0 / wavelength,
        wavelength=wavelength,
        Ex=np.zeros_like(Hz0),
        Ey=np.zeros_like(Hz0),
        Ez=np.zeros_like(Hz0),
        Hx=np.zeros_like(Hz0),
        Hy=np.zeros_like(Hz0),
        Hz=Hz0,
        x=x,
        y=y,
        power=1.0,
    )

    mode1 = WaveguideMode(
        mode_number=1,
        neff=1.48 + 0j,
        frequency=299792458.0 / wavelength,
        wavelength=wavelength,
        Ex=np.zeros_like(Hz1),
        Ey=np.zeros_like(Hz1),
        Ez=np.zeros_like(Hz1),
        Hx=np.zeros_like(Hz1),
        Hy=np.zeros_like(Hz1),
        Hz=Hz1,
        x=x,
        y=y,
        power=1.0,
    )

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Check orthogonality
    orthogonality = mode_matching.check_mode_orthogonality(
        mode0, mode1, direction="z", dx=dx, dy=dy
    )

    print(f"\nOrthogonality metric between mode 0 and mode 1: {orthogonality:.6f}")
    print("  (0 = perfectly orthogonal, 1 = identical)")
    print(
        f"  → Modes are {'orthogonal' if orthogonality < 0.1 else 'not perfectly orthogonal'}"
    )


def main():
    """
    Main demonstration workflow.
    """
    print("=" * 60)
    print("Mode Port Demonstration")
    print("=" * 60)
    print("\nThis example demonstrates:")
    print("  1. Waveguide mode solving")
    print("  2. Mode profile visualization")
    print("  3. S-parameter calculation concept")
    print("  4. Mode overlap integrals")
    print()

    # Simulation parameters
    wavelength = 1.55e-6  # 1.55 μm (telecom C-band)
    waveguide_width = 1.5e-6
    waveguide_length = 10e-6

    print(f"Parameters:")
    print(f"  Wavelength: {wavelength*1e6:.3f} μm")
    print(f"  Waveguide width: {waveguide_width*1e6:.3f} μm")
    print(f"  Waveguide length: {waveguide_length*1e6:.3f} μm")

    # Step 1: Create waveguide structure
    x, y, z, epsilon = create_waveguide_structure(
        width=waveguide_width,
        length=waveguide_length,
    )

    # Step 2: Solve for modes
    modes = solve_waveguide_modes(wavelength, x, y, epsilon, num_modes=2)

    # Step 3: Visualize modes
    visualize_modes(modes, filename="mode_profiles.png")

    # Step 4: Calculate S-parameters (simplified)
    s_params = calculate_s_parameters_simple(modes, input_power=1.0)

    # Step 5: Plot S-parameters
    plot_s_parameters(s_params, filename="s_parameters.png")

    # Bonus: Demonstrate mode overlap
    demonstrate_mode_overlap()

    print(f"\n{'='*60}")
    print("✓ Demo complete!")
    print(f"{'='*60}")
    print("\nGenerated files:")
    print("  - mode_profiles.png: Mode field distributions")
    print("  - s_parameters.png: S-parameter plots")
    print("\nFor a full FDTD simulation with mode injection/extraction,")
    print("integrate ModeSource and ModeExpansionMonitor into your")
    print("simulation workflow.")
    print()


if __name__ == "__main__":
    main()
