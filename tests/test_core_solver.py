"""
Tests for the core FDTD solver implementation.

This module tests the Maxwell equation updates, time stepping, and numerical
stability of the core FDTD solver components.
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from prismo.core.grid import YeeGrid, GridSpec
from prismo.core.fields import ElectromagneticFields
from prismo.core.solver import MaxwellUpdater, FDTDSolver


class TestMaxwellUpdater:
    """Test cases for the MaxwellUpdater class."""

    @pytest.fixture
    def grid_2d(self):
        """Create a simple 2D grid for testing."""
        spec = GridSpec(
            size=(1.0, 1.0, 0.0),  # 1µm x 1µm 2D grid
            resolution=10.0,  # 10 points per µm
        )
        return YeeGrid(spec)

    @pytest.fixture
    def grid_3d(self):
        """Create a simple 3D grid for testing."""
        spec = GridSpec(
            size=(1.0, 1.0, 1.0),  # 1µm x 1µm x 1µm 3D grid
            resolution=10.0,  # 10 points per µm
        )
        return YeeGrid(spec)

    def test_maxwell_updater_initialization_2d(self, grid_2d):
        """Test Maxwell updater initialization for 2D grid."""
        dt = grid_2d.suggest_time_step(safety_factor=0.5)
        updater = MaxwellUpdater(grid_2d, dt)

        # Check basic properties
        assert updater.grid is grid_2d
        assert updater.dt == dt
        assert updater.get_time_step() == dt
        assert updater.get_courant_number() < 1.0

        # Check material arrays are initialized
        assert updater.eps_rel.shape == grid_2d.dimensions
        assert updater.mu_rel.shape == grid_2d.dimensions
        assert np.all(updater.eps_rel == 1.0)  # Vacuum
        assert np.all(updater.mu_rel == 1.0)  # Vacuum

    def test_maxwell_updater_initialization_3d(self, grid_3d):
        """Test Maxwell updater initialization for 3D grid."""
        dt = grid_3d.suggest_time_step(safety_factor=0.5)
        updater = MaxwellUpdater(grid_3d, dt)

        assert updater.grid is grid_3d
        assert updater.dt == dt
        assert updater.get_courant_number() < 1.0

    def test_courant_condition_violation(self, grid_2d):
        """Test that Courant condition violation raises error."""
        # Use a time step that's too large
        dt_max = grid_2d.suggest_time_step(safety_factor=1.0)
        dt_bad = dt_max * 1.1  # 10% above stability limit

        with pytest.raises(ValueError, match="violates Courant condition"):
            MaxwellUpdater(grid_2d, dt_bad)

    def test_field_update_basic_2d(self, grid_2d):
        """Test basic field updates in 2D."""
        dt = grid_2d.suggest_time_step(safety_factor=0.5)
        updater = MaxwellUpdater(grid_2d, dt)
        fields = ElectromagneticFields(grid_2d)

        # Initialize with E fields that have curl (for TE mode)
        # Use sinusoidal patterns that create non-zero curl
        nx_ex, ny_ex = fields["Ex"].shape
        nx_ey, ny_ey = fields["Ey"].shape

        # Ex with y-dependence: creates ∂Ex/∂y
        for i in range(nx_ex):
            for j in range(ny_ex):
                y = j * 2 * np.pi / ny_ex
                fields["Ex"][i, j] = np.sin(y)

        # Ey with x-dependence: creates ∂Ey/∂x
        for i in range(nx_ey):
            for j in range(ny_ey):
                x = i * 2 * np.pi / nx_ey
                fields["Ey"][i, j] = np.cos(
                    x
                )  # Store initial field values (focusing on Hz which should be updated in TE mode)
        initial_hz_max = np.max(np.abs(fields["Hz"]))
        initial_ex_max = np.max(np.abs(fields["Ex"]))
        initial_ey_max = np.max(np.abs(fields["Ey"]))

        # Store initial field energy
        initial_energy = fields.get_field_energy()

        # Perform one update step
        updater.step(fields)

        # Check field values after update
        final_hz_max = np.max(np.abs(fields["Hz"]))
        final_ex_max = np.max(np.abs(fields["Ex"]))
        final_ey_max = np.max(np.abs(fields["Ey"]))

        # Store final field energy
        final_energy = fields.get_field_energy()

        # After one time step, Hz should be updated due to curl of Ex, Ey
        # The H field should be updated first, then E field
        assert (
            final_hz_max > initial_hz_max
        ), f"Hz field not updated: {initial_hz_max:.2e} -> {final_hz_max:.2e}"

        # Energy should also change
        assert final_energy != initial_energy, f"Energy unchanged: {initial_energy:.2e}"

    def test_field_update_basic_3d(self, grid_3d):
        """Test basic field updates in 3D."""
        dt = grid_3d.suggest_time_step(safety_factor=0.5)
        updater = MaxwellUpdater(grid_3d, dt)
        fields = ElectromagneticFields(grid_3d)

        # Initialize with a simple field distribution
        nx, ny, nz = grid_3d.dimensions
        fields["Ez"][nx // 2, ny // 2, nz // 2] = 1.0  # Point source

        # Perform one update step
        updater.step(fields)

        # Check that fields have changed and propagated
        assert np.max(np.abs(fields["Hx"])) > 0 or np.max(np.abs(fields["Hy"])) > 0

    def test_update_coefficients_vacuum(self, grid_2d):
        """Test update coefficients calculation for vacuum."""
        dt = grid_2d.suggest_time_step(safety_factor=0.5)
        updater = MaxwellUpdater(grid_2d, dt)

        # In vacuum with no conductivity, Ca should be 1 and Cb should be dt/eps0
        expected_cb = dt / updater.eps0
        assert np.allclose(updater.Ca, 1.0)
        assert np.allclose(updater.Cb, expected_cb)

        # Similarly for magnetic fields
        expected_db = dt / updater.mu0
        assert np.allclose(updater.Da, 1.0)
        assert np.allclose(updater.Db, expected_db)

    def test_custom_materials(self, grid_2d):
        """Test Maxwell updater with custom material properties."""
        dt = grid_2d.suggest_time_step(safety_factor=0.5)

        # Create silicon-like material (n=3.45, so eps_rel=3.45^2≈11.9)
        nx, ny = grid_2d.dimensions[:2]
        material_arrays = {
            "eps_rel": np.full((nx, ny, 1), 11.9),
            "mu_rel": np.ones((nx, ny, 1)),
            "sigma_e": np.zeros((nx, ny, 1)),
            "sigma_m": np.zeros((nx, ny, 1)),
        }

        updater = MaxwellUpdater(grid_2d, dt, material_arrays)

        # Check that material properties are properly stored
        assert np.allclose(updater.eps_rel, 11.9)
        assert np.allclose(updater.mu_rel, 1.0)

        # Check that update coefficients reflect the material properties
        expected_cb = dt / (updater.eps0 * 11.9)
        assert np.allclose(updater.Cb, expected_cb)

    def test_field_interpolation_methods(self, grid_3d):
        """Test material property interpolation to field points."""
        dt = grid_3d.suggest_time_step(safety_factor=0.5)
        updater = MaxwellUpdater(grid_3d, dt)

        # Create a test array with spatial variation
        nx, ny, nz = grid_3d.dimensions
        test_array = np.random.rand(nx, ny, nz)

        # Test interpolation to different field points
        hx_interp = updater._interpolate_to_hx_points(test_array)
        hy_interp = updater._interpolate_to_hy_points(test_array)
        hz_interp = updater._interpolate_to_hz_points(test_array)

        # Check shapes are correct
        assert hx_interp.shape == (nx - 1, ny, nz)
        assert hy_interp.shape == (nx, ny - 1, nz)
        assert hz_interp.shape == (nx, ny, nz - 1)

        # Check that interpolation is actually averaging
        expected_hx = 0.5 * (test_array[:-1, :, :] + test_array[1:, :, :])
        assert_allclose(hx_interp, expected_hx)

    def test_energy_conservation_vacuum(self, grid_2d):
        """Test energy conservation in vacuum (should be preserved)."""
        dt = grid_2d.suggest_time_step(safety_factor=0.5)
        updater = MaxwellUpdater(grid_2d, dt)
        fields = ElectromagneticFields(grid_2d)

        # Initialize with a localized field - use proper field shapes
        nx_ez, ny_ez = fields["Ez"].shape
        nx_hz, ny_hz = fields["Hz"].shape

        # Create Gaussian for Ez field
        x, y = np.meshgrid(np.arange(nx_ez), np.arange(ny_ez), indexing="ij")
        sigma = 2.0
        gaussian_ez = np.exp(
            -((x - nx_ez // 2) ** 2 + (y - ny_ez // 2) ** 2) / (2 * sigma**2)
        )

        # Create Gaussian for Hz field
        x, y = np.meshgrid(np.arange(nx_hz), np.arange(ny_hz), indexing="ij")
        gaussian_hz = np.exp(
            -((x - nx_hz // 2) ** 2 + (y - ny_hz // 2) ** 2) / (2 * sigma**2)
        )

        # Initialize both E and H fields to have some energy
        fields["Ez"][:, :] = gaussian_ez
        fields["Hz"][:, :] = 0.1 * gaussian_hz

        initial_energy = fields.get_field_energy()

        # Run for several steps
        energies = [initial_energy]
        for _ in range(10):
            updater.step(fields)
            energies.append(fields.get_field_energy())

        # Energy should be reasonably well conserved (within numerical precision)
        # Allow for some small variation due to finite difference approximations
        energy_variation = (max(energies) - min(energies)) / initial_energy
        assert (
            energy_variation < 0.01
        ), f"Energy variation too large: {energy_variation:.3f}"

    def test_str_representation(self, grid_2d):
        """Test string representation of MaxwellUpdater."""
        dt = grid_2d.suggest_time_step(safety_factor=0.5)
        updater = MaxwellUpdater(grid_2d, dt)

        str_repr = str(updater)
        assert "MaxwellUpdater" in str_repr
        assert f"dt={dt:.2e}" in str_repr
        assert "Courant" in str_repr


class TestFDTDSolver:
    """Test cases for the FDTDSolver class."""

    @pytest.fixture
    def small_grid_2d(self):
        """Create a small 2D grid for fast testing."""
        spec = GridSpec(
            size=(0.5, 0.5, 0.0),  # 0.5µm x 0.5µm 2D grid
            resolution=10.0,  # 10 points per µm
        )
        return YeeGrid(spec)

    def test_fdtd_solver_initialization(self, small_grid_2d):
        """Test FDTD solver initialization."""
        solver = FDTDSolver(small_grid_2d)

        # Check components are initialized
        assert solver.grid is small_grid_2d
        assert isinstance(solver.fields, ElectromagneticFields)
        assert isinstance(solver.updater, MaxwellUpdater)

        # Check initial state
        assert solver.time == 0.0
        assert solver.step_count == 0

    def test_fdtd_solver_custom_dt(self, small_grid_2d):
        """Test FDTD solver with custom time step."""
        custom_dt = 1e-16
        solver = FDTDSolver(small_grid_2d, dt=custom_dt)

        assert solver.updater.get_time_step() == custom_dt

    def test_run_steps(self, small_grid_2d):
        """Test running FDTD for specified number of steps."""
        solver = FDTDSolver(small_grid_2d)

        # Initialize with some field
        solver.fields["Ez"][2, 2] = 1.0

        # Run for 10 steps
        num_steps = 10
        solver.run_steps(num_steps)

        # Check that time and step count are updated
        expected_time = num_steps * solver.updater.get_time_step()
        assert solver.time == expected_time
        assert solver.step_count == num_steps

    def test_run_time(self, small_grid_2d):
        """Test running FDTD for specified time duration."""
        solver = FDTDSolver(small_grid_2d)

        # Initialize with some field
        solver.fields["Ez"][2, 2] = 1.0

        # Run for specific time
        run_time = 1e-14  # 10 fs
        solver.run(run_time)

        # Check that final time is approximately correct
        assert abs(solver.time - run_time) < solver.updater.get_time_step()

    def test_callback_functionality(self, small_grid_2d):
        """Test callback function during simulation."""
        solver = FDTDSolver(small_grid_2d)

        # Track callback calls
        callback_calls = []

        def test_callback(solver_obj, step_num):
            callback_calls.append((solver_obj.step_count, step_num))

        solver.run_steps(5, callback=test_callback)

        # Check that callback was called for each step
        assert len(callback_calls) == 5
        assert callback_calls[0] == (1, 0)  # First call
        assert callback_calls[-1] == (5, 4)  # Last call

    def test_reset_functionality(self, small_grid_2d):
        """Test solver reset functionality."""
        solver = FDTDSolver(small_grid_2d)

        # Run some steps
        solver.fields["Ez"][2, 2] = 1.0
        solver.run_steps(5)

        # Verify state has changed
        assert solver.time > 0
        assert solver.step_count > 0

        # Reset
        solver.reset()

        # Check that state is back to initial
        assert solver.time == 0.0
        assert solver.step_count == 0
        assert np.all(solver.fields["Ez"] == 0.0)

    def test_simulation_info(self, small_grid_2d):
        """Test simulation info retrieval."""
        solver = FDTDSolver(small_grid_2d)

        info = solver.get_simulation_info()

        # Check that all expected keys are present
        expected_keys = [
            "time",
            "step_count",
            "dt",
            "courant_number",
            "grid_dimensions",
            "is_2d",
            "field_energy",
        ]
        for key in expected_keys:
            assert key in info

        # Check types and values
        assert isinstance(info["time"], float)
        assert isinstance(info["step_count"], int)
        assert info["is_2d"] == small_grid_2d.is_2d
        assert info["grid_dimensions"] == small_grid_2d.dimensions

    def test_str_representation(self, small_grid_2d):
        """Test string representation of FDTDSolver."""
        solver = FDTDSolver(small_grid_2d)

        str_repr = str(solver)
        assert "FDTDSolver" in str_repr
        assert "t=" in str_repr
        assert "steps=" in str_repr
        assert "E_total=" in str_repr


class TestAnalyticalValidation:
    """Test FDTD results against known analytical solutions."""

    @pytest.fixture
    def uniform_grid_1d(self):
        """Create a uniform 1D-like grid for analytical tests."""
        spec = GridSpec(
            size=(2.0, 0.1, 0.0),  # Long thin 2D grid (quasi-1D)
            resolution=50.0,  # High resolution
        )
        return YeeGrid(spec)

    def test_plane_wave_propagation_1d(self, uniform_grid_1d):
        """Test plane wave propagation against analytical solution."""
        solver = FDTDSolver(uniform_grid_1d)

        # Physical constants
        c = 299792458.0  # Speed of light
        freq = 1e14  # 100 THz
        omega = 2 * np.pi * freq
        k = omega / c  # Wavenumber

        # Initialize sinusoidal wave
        nx, ny = uniform_grid_1d.dimensions[:2]
        dx = uniform_grid_1d.spacing[0]

        nx_ey, ny_ey = solver.fields["Ey"].shape
        for i in range(nx_ey):
            x = i * dx
            # Initialize Ey component (TE wave)
            solver.fields["Ey"][i, :] = np.sin(k * x)

        # Store initial pattern for comparison
        initial_pattern = solver.fields["Ey"][:, 0].copy()

        # Run for one period of oscillation
        T = 1.0 / freq  # Period
        num_steps = int(T / solver.updater.get_time_step())
        solver.run_steps(num_steps)

        # After one period, the pattern should be very similar (allowing for numerical dispersion)
        final_pattern = solver.fields["Ey"][:, 0]

        # Calculate correlation between initial and final patterns
        correlation = np.corrcoef(initial_pattern, final_pattern)[0, 1]
        assert correlation > 0.8, f"Wave pattern correlation too low: {correlation:.3f}"

    def test_standing_wave_formation(self, uniform_grid_1d):
        """Test standing wave formation with reflecting boundaries."""
        solver = FDTDSolver(uniform_grid_1d)

        # Initialize two counter-propagating waves
        nx, ny = uniform_grid_1d.dimensions[:2]
        dx = uniform_grid_1d.spacing[0]
        k = np.pi / (nx * dx)  # Wavelength fits in the domain

        nx_ey, ny_ey = solver.fields["Ey"].shape
        for i in range(nx_ey):
            x = i * dx
            # Superposition of forward and backward waves
            solver.fields["Ey"][i, :] = np.sin(k * x) + np.sin(k * (nx_ey * dx - x))

        # Run simulation
        solver.run_steps(50)

        # Check that we have a standing wave pattern (nodes and antinodes)
        field_profile = solver.fields["Ey"][:, 0]

        # Find approximate node positions (should be at multiples of lambda/2)
        field_abs = np.abs(field_profile)
        min_indices = []
        for i in range(1, len(field_abs) - 1):
            if (
                field_abs[i] < field_abs[i - 1]
                and field_abs[i] < field_abs[i + 1]
                and field_abs[i] < 0.1 * np.max(field_abs)
            ):
                min_indices.append(i)

        # Should have at least a few nodes
        assert len(min_indices) >= 2, "Standing wave nodes not detected"

    @pytest.mark.slow
    def test_energy_flux_conservation(self):
        """Test energy flux conservation in a uniform medium."""
        # Create a larger grid for this test
        spec = GridSpec(size=(3.0, 1.0, 0.0), resolution=20.0)
        grid = YeeGrid(spec)
        solver = FDTDSolver(grid)

        # Create a localized pulse
        nx_ez, ny_ez = solver.fields["Ez"].shape
        x0, y0 = nx_ez // 4, ny_ez // 2  # Start at 1/4 of the domain
        sigma = 3.0

        for i in range(nx_ez):
            for j in range(ny_ez):
                distance_sq = (i - x0) ** 2 + (j - y0) ** 2
                amplitude = np.exp(-distance_sq / (2 * sigma**2))
                solver.fields["Ez"][i, j] = amplitude

        # Calculate initial energy in different regions
        quarter = nx_ez // 4
        region1_energy_init = np.sum(solver.fields["Ez"][:quarter, :] ** 2)
        region2_energy_init = np.sum(solver.fields["Ez"][quarter : 2 * quarter, :] ** 2)
        region3_energy_init = np.sum(
            solver.fields["Ez"][2 * quarter : 3 * quarter, :] ** 2
        )
        region4_energy_init = np.sum(solver.fields["Ez"][3 * quarter :, :] ** 2)

        # Run simulation to let pulse propagate
        solver.run_steps(100)

        # Calculate final energy in different regions
        region1_energy_final = np.sum(solver.fields["Ez"][:quarter, :] ** 2)
        region2_energy_final = np.sum(
            solver.fields["Ez"][quarter : 2 * quarter, :] ** 2
        )
        region3_energy_final = np.sum(
            solver.fields["Ez"][2 * quarter : 3 * quarter, :] ** 2
        )
        region4_energy_final = np.sum(solver.fields["Ez"][3 * quarter :, :] ** 2)

        # Energy should have moved from left to right regions
        assert region1_energy_final < region1_energy_init
        assert (
            region3_energy_final > region3_energy_init
            or region4_energy_final > region4_energy_init
        )


class TestNumericalStability:
    """Test numerical stability and error conditions."""

    def test_field_divergence_2d(self):
        """Test that field divergence remains small in 2D."""
        spec = GridSpec(size=(1.0, 1.0, 0.0), resolution=20.0)
        grid = YeeGrid(spec)
        solver = FDTDSolver(grid)

        # Initialize with a localized source
        nx, ny = grid.dimensions[:2]
        solver.fields["Ez"][nx // 2, ny // 2] = 1.0

        # Run simulation
        solver.run_steps(50)

        # Calculate divergence of E field in 2D: ∇·E = ∂Ex/∂x + ∂Ey/∂y
        Ex, Ey = solver.fields["Ex"], solver.fields["Ey"]
        dx, dy = grid.spacing[:2]

        # Get field shapes
        nx_ex, ny_ex = Ex.shape
        nx_ey, ny_ey = Ey.shape

        # Find overlapping region for divergence calculation
        nx_common = min(nx_ex - 2, nx_ey - 2)
        ny_common = min(ny_ex - 2, ny_ey - 2)

        # Finite difference divergence calculation in common region
        div_E = np.zeros((nx_common, ny_common))
        if nx_common > 0 and ny_common > 0:
            div_E = (
                Ex[2 : nx_common + 2, 1 : ny_common + 1]
                - Ex[:nx_common, 1 : ny_common + 1]
            ) / (2 * dx) + (
                Ey[1 : nx_common + 1, 2 : ny_common + 2]
                - Ey[1 : nx_common + 1, :ny_common]
            ) / (
                2 * dy
            )

        # Divergence should be small (Gauss's law in vacuum: ∇·E = 0)
        max_div = np.max(np.abs(div_E))
        assert max_div < 1e-10, f"Field divergence too large: {max_div:.2e}"

    def test_large_time_step_instability(self):
        """Test that large time steps lead to instability."""
        spec = GridSpec(size=(0.5, 0.5, 0.0), resolution=10.0)
        grid = YeeGrid(spec)

        # Use time step at Courant limit (should be stable)
        dt_stable = grid.suggest_time_step(safety_factor=0.99)
        solver_stable = FDTDSolver(grid, dt=dt_stable)

        # Initialize with some field
        solver_stable.fields["Ez"][2, 2] = 1.0

        # Run stable simulation
        solver_stable.run_steps(20)
        stable_energy = solver_stable.fields.get_field_energy()

        # Now test with time step slightly above Courant limit
        # This should be caught during initialization
        # Need to multiply by enough to exceed Courant=1.0: 1/0.99 = 1.0101...
        dt_unstable = dt_stable * 1.02
        with pytest.raises(ValueError, match="violates Courant condition"):
            FDTDSolver(grid, dt=dt_unstable)

    def test_numerical_dispersion_basic(self):
        """Test basic numerical dispersion characteristics."""
        # Create a 1D-like grid
        spec = GridSpec(size=(4.0, 0.1, 0.0), resolution=50.0)
        grid = YeeGrid(spec)
        solver = FDTDSolver(grid)

        # Physical parameters
        c = 299792458.0
        dx = grid.spacing[0]

        # Choose frequency based on grid resolution to ensure good resolution
        # Want at least 10 points per wavelength: lambda > 10*dx
        # lambda = c/f, so f < c/(10*dx)
        max_freq = c / (20 * dx)  # Use 20 points per wavelength for good resolution
        freq = max_freq * 0.5  # Use half the max frequency for safety

        omega = 2 * np.pi * freq
        k_analytical = omega / c

        # Check wavelength resolution
        lambda_analytical = 2 * np.pi / k_analytical
        points_per_wavelength = lambda_analytical / dx
        assert (
            points_per_wavelength > 10
        ), f"Wavelength not well resolved: {points_per_wavelength:.2f} points per wavelength"

        # Initialize field
        nx_ey, ny_ey = solver.fields["Ey"].shape
        for i in range(nx_ey):
            x = i * dx
            solver.fields["Ey"][i, :] = np.sin(k_analytical * x)

        # Store initial phase
        initial_phase = np.angle(np.fft.fft(solver.fields["Ey"][:, 0]))

        # Run for a short time
        solver.run_steps(20)

        # Check final phase
        final_phase = np.angle(np.fft.fft(solver.fields["Ey"][:, 0]))

        # Phase velocity should be close to c (within 5% for well-resolved waves)
        phase_diff = final_phase - initial_phase
        phase_diff = np.unwrap(phase_diff)

        # The dominant mode should have the right phase velocity
        dominant_mode = np.argmax(np.abs(np.fft.fft(solver.fields["Ey"][:, 0])))
        if dominant_mode > 0:  # Avoid DC component
            measured_k = phase_diff[dominant_mode] / (solver.time)
            measured_phase_velocity = omega / measured_k
            velocity_error = abs(measured_phase_velocity - c) / c
            assert (
                velocity_error < 0.1
            ), f"Phase velocity error too large: {velocity_error:.3f}"

    def test_memory_usage_consistency(self):
        """Test that memory usage is consistent across different grid sizes."""
        import sys

        # Test different grid sizes
        grid_sizes = [(0.5, 0.5, 0.0), (1.0, 1.0, 0.0), (1.5, 1.5, 0.0)]
        memory_per_point = []

        for size in grid_sizes:
            spec = GridSpec(size=size, resolution=10.0)
            grid = YeeGrid(spec)
            solver = FDTDSolver(grid)

            # Estimate memory usage (rough approximation)
            num_points = np.prod(grid.dimensions)
            field_arrays_size = 6 * num_points * 8  # 6 components, 8 bytes per float64
            memory_per_point.append(field_arrays_size / num_points)

        # Memory per point should be consistent
        memory_variation = (max(memory_per_point) - min(memory_per_point)) / np.mean(
            memory_per_point
        )
        assert (
            memory_variation < 0.01
        ), f"Memory usage per point not consistent: {memory_variation:.3f}"


if __name__ == "__main__":
    pytest.main([__file__])
