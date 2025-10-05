"""
Test suite for core Grid functionality.

This module tests the YeeGrid and GridSpec classes, including 2D/3D compatibility,
boundary conditions, coordinate systems, and edge cases.
"""

import pytest
import numpy as np
from prismo.core.grid import YeeGrid, GridSpec


class TestGridSpec:
    """Test cases for GridSpec class."""

    def test_basic_creation(self):
        """Test basic GridSpec creation."""
        spec = GridSpec(size=(10.0, 5.0, 3.0), resolution=20.0)
        assert spec.size == (10.0, 5.0, 3.0)
        assert spec.resolution == (20.0, 20.0, 20.0)
        assert spec.boundary_layers == 10

    def test_resolution_tuple(self):
        """Test GridSpec with tuple resolution."""
        spec = GridSpec(size=(10.0, 5.0, 3.0), resolution=(20.0, 30.0, 40.0))
        assert spec.resolution == (20.0, 30.0, 40.0)

    def test_2d_specification(self):
        """Test 2D grid specification."""
        spec = GridSpec(size=(10.0, 5.0, 0.0), resolution=20.0)
        assert spec.size == (10.0, 5.0, 0.0)

    def test_validation_negative_size(self):
        """Test validation of negative size."""
        with pytest.raises(
            ValueError, match="Grid size components must be non-negative"
        ):
            GridSpec(size=(-1.0, 5.0, 3.0), resolution=20.0)

    def test_validation_zero_resolution(self):
        """Test validation of zero resolution."""
        with pytest.raises(ValueError, match="Resolution must be positive"):
            GridSpec(size=(10.0, 5.0, 3.0), resolution=0.0)

    def test_validation_wrong_resolution_length(self):
        """Test validation of resolution tuple length."""
        with pytest.raises(ValueError, match="Resolution must be scalar or 3-tuple"):
            GridSpec(size=(10.0, 5.0, 3.0), resolution=(20.0, 30.0))


class TestYeeGrid3D:
    """Test cases for 3D YeeGrid."""

    @pytest.fixture
    def grid_spec_3d(self):
        """Create a 3D grid specification."""
        return GridSpec(size=(10.0, 5.0, 3.0), resolution=20.0, boundary_layers=5)

    @pytest.fixture
    def grid_3d(self, grid_spec_3d):
        """Create a 3D YeeGrid."""
        return YeeGrid(grid_spec_3d)

    def test_3d_grid_creation(self, grid_3d):
        """Test 3D grid creation."""
        assert not grid_3d.is_2d
        assert grid_3d.Lx == 10.0
        assert grid_3d.Ly == 5.0
        assert grid_3d.Lz == 3.0

    def test_3d_grid_spacing(self, grid_3d):
        """Test 3D grid spacing calculation."""
        expected_dx = 1.0 / 20.0  # dx = 1/resolution
        assert abs(grid_3d.dx - expected_dx) < 1e-10
        assert abs(grid_3d.dy - expected_dx) < 1e-10
        assert abs(grid_3d.dz - expected_dx) < 1e-10

    def test_3d_grid_dimensions(self, grid_3d):
        """Test 3D grid dimension calculation."""
        # Physical points: ceil(L * res)
        expected_nx = int(np.ceil(10.0 * 20.0))  # 200
        expected_ny = int(np.ceil(5.0 * 20.0))  # 100
        expected_nz = int(np.ceil(3.0 * 20.0))  # 60

        assert grid_3d.Nx == expected_nx
        assert grid_3d.Ny == expected_ny
        assert grid_3d.Nz == expected_nz

        # Total dimensions include PML
        assert grid_3d.Nx_total == expected_nx + 2 * 5  # +10 for PML
        assert grid_3d.Ny_total == expected_ny + 2 * 5
        assert grid_3d.Nz_total == expected_nz + 2 * 5

    def test_3d_field_shapes(self, grid_3d):
        """Test field component shapes for 3D grid."""
        nx, ny, nz = grid_3d.dimensions

        # Electric field shapes
        assert grid_3d.get_field_shape("Ex") == (nx, ny - 1, nz - 1)
        assert grid_3d.get_field_shape("Ey") == (nx - 1, ny, nz - 1)
        assert grid_3d.get_field_shape("Ez") == (nx - 1, ny - 1, nz)

        # Magnetic field shapes
        assert grid_3d.get_field_shape("Hx") == (nx - 1, ny, nz)
        assert grid_3d.get_field_shape("Hy") == (nx, ny - 1, nz)
        assert grid_3d.get_field_shape("Hz") == (nx, ny, nz - 1)

    def test_3d_coordinates(self, grid_3d):
        """Test coordinate generation for 3D grid."""
        x, y, z = grid_3d.get_coordinates("Ex")

        # Check coordinate array lengths
        nx, ny, nz = grid_3d.dimensions
        assert len(x) == nx
        assert len(y) == ny - 1  # Ex is offset in y
        assert len(z) == nz - 1  # Ex is offset in z

        # Check coordinate spacing
        assert abs(x[1] - x[0] - grid_3d.dx) < 1e-10
        assert abs(y[1] - y[0] - grid_3d.dy) < 1e-10
        assert abs(z[1] - z[0] - grid_3d.dz) < 1e-10

    def test_3d_pml_detection(self, grid_3d):
        """Test PML region detection for 3D grid."""
        pml = grid_3d.pml_layers

        # Points inside PML
        assert grid_3d.is_inside_pml(0, 10, 10)  # Left boundary
        assert grid_3d.is_inside_pml(pml - 1, 10, 10)
        assert grid_3d.is_inside_pml(grid_3d.Nx_total - 1, 10, 10)  # Right boundary
        assert grid_3d.is_inside_pml(10, 0, 10)  # Bottom boundary
        assert grid_3d.is_inside_pml(10, 10, 0)  # Back boundary

        # Point inside physical domain
        center_i = grid_3d.Nx_total // 2
        center_j = grid_3d.Ny_total // 2
        center_k = grid_3d.Nz_total // 2
        assert not grid_3d.is_inside_pml(center_i, center_j, center_k)

    def test_3d_physical_indices(self, grid_3d):
        """Test physical domain index slices."""
        x_slice, y_slice, z_slice = grid_3d.get_physical_indices()
        pml = grid_3d.pml_layers

        assert x_slice == slice(pml, grid_3d.Nx_total - pml)
        assert y_slice == slice(pml, grid_3d.Ny_total - pml)
        assert z_slice == slice(pml, grid_3d.Nz_total - pml)


class TestYeeGrid2D:
    """Test cases for 2D YeeGrid."""

    @pytest.fixture
    def grid_spec_2d(self):
        """Create a 2D grid specification."""
        return GridSpec(size=(8.0, 6.0, 0.0), resolution=25.0, boundary_layers=8)

    @pytest.fixture
    def grid_2d(self, grid_spec_2d):
        """Create a 2D YeeGrid."""
        return YeeGrid(grid_spec_2d)

    def test_2d_grid_creation(self, grid_2d):
        """Test 2D grid creation."""
        assert grid_2d.is_2d
        assert grid_2d.Lx == 8.0
        assert grid_2d.Ly == 6.0
        assert grid_2d.Lz == 0.0
        assert grid_2d.Nz == 1
        assert grid_2d.dz == 0.0

    def test_2d_field_shapes(self, grid_2d):
        """Test field component shapes for 2D grid."""
        nx, ny, _ = grid_2d.dimensions  # nz = 1 for 2D

        # Electric field shapes (2D)
        assert grid_2d.get_field_shape("Ex") == (nx, ny - 1)
        assert grid_2d.get_field_shape("Ey") == (nx - 1, ny)
        assert grid_2d.get_field_shape("Ez") == (nx - 1, ny - 1)

        # Magnetic field shapes (2D)
        assert grid_2d.get_field_shape("Hx") == (nx - 1, ny)
        assert grid_2d.get_field_shape("Hy") == (nx, ny - 1)
        assert grid_2d.get_field_shape("Hz") == (nx, ny)

    def test_2d_coordinates(self, grid_2d):
        """Test coordinate generation for 2D grid."""
        x, y = grid_2d.get_coordinates("Ez")

        # Check coordinate array lengths
        nx, ny, _ = grid_2d.dimensions
        assert len(x) == nx - 1  # Ez is offset in x
        assert len(y) == ny - 1  # Ez is offset in y

        # Should return only x, y for 2D
        coords = grid_2d.get_coordinates("Hz")
        assert len(coords) == 2

    def test_2d_pml_detection(self, grid_2d):
        """Test PML region detection for 2D grid."""
        pml = grid_2d.pml_layers

        # Points inside PML (k=0 for 2D)
        assert grid_2d.is_inside_pml(0, 10, 0)  # Left boundary
        assert grid_2d.is_inside_pml(grid_2d.Nx_total - 1, 10, 0)  # Right boundary
        assert grid_2d.is_inside_pml(10, 0, 0)  # Bottom boundary

        # Point inside physical domain
        center_i = grid_2d.Nx_total // 2
        center_j = grid_2d.Ny_total // 2
        assert not grid_2d.is_inside_pml(center_i, center_j, 0)

    def test_2d_physical_indices(self, grid_2d):
        """Test physical domain index slices for 2D."""
        x_slice, y_slice, z_slice = grid_2d.get_physical_indices()
        pml = grid_2d.pml_layers

        assert x_slice == slice(pml, grid_2d.Nx_total - pml)
        assert y_slice == slice(pml, grid_2d.Ny_total - pml)
        assert z_slice == slice(None)  # No z restriction for 2D


class TestGridStability:
    """Test cases for grid stability and Courant condition."""

    @pytest.fixture
    def test_grid(self):
        """Create test grid for stability tests."""
        spec = GridSpec(size=(5.0, 5.0, 5.0), resolution=50.0)
        return YeeGrid(spec)

    def test_courant_number_calculation(self, test_grid):
        """Test Courant number calculation."""
        dt = 1e-17  # Very small time step
        courant = test_grid.get_courant_number(dt)

        # Should be much less than 1 for stability
        assert courant < 1.0
        assert courant > 0.0

    def test_stable_time_step_suggestion(self, test_grid):
        """Test stable time step suggestion."""
        dt_suggested = test_grid.suggest_time_step()

        # Check that suggested time step gives stable Courant number
        courant = test_grid.get_courant_number(dt_suggested)
        assert courant < 1.0
        assert courant > 0.0

        # Should be reasonable time step (not too large or too small)
        assert dt_suggested > 1e-20
        assert dt_suggested < 1e-10

    def test_courant_2d_vs_3d(self):
        """Test Courant number difference between 2D and 3D."""
        # Same x-y resolution and spacing
        spec_2d = GridSpec(size=(4.0, 4.0, 0.0), resolution=40.0)
        spec_3d = GridSpec(size=(4.0, 4.0, 4.0), resolution=40.0)

        grid_2d = YeeGrid(spec_2d)
        grid_3d = YeeGrid(spec_3d)

        dt = 1e-16
        courant_2d = grid_2d.get_courant_number(dt)
        courant_3d = grid_3d.get_courant_number(dt)

        # 3D should have larger Courant number (more restrictive)
        assert courant_3d > courant_2d


class TestGridEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_field_component(self):
        """Test error on invalid field component."""
        spec = GridSpec(size=(2.0, 2.0, 2.0), resolution=10.0)
        grid = YeeGrid(spec)

        with pytest.raises(ValueError, match="Unknown field component"):
            grid.get_field_shape("Invalid")

    def test_very_small_grid(self):
        """Test very small grid dimensions."""
        spec = GridSpec(size=(0.1, 0.1, 0.1), resolution=5.0)
        grid = YeeGrid(spec)

        # Should still work, just very few points
        assert grid.Nx >= 1
        assert grid.Ny >= 1
        assert grid.Nz >= 1

    def test_high_resolution_grid(self):
        """Test high resolution grid."""
        spec = GridSpec(size=(1.0, 1.0, 1.0), resolution=200.0)
        grid = YeeGrid(spec)

        # Should handle high resolution properly
        assert grid.Nx == 200
        assert grid.dx == 0.005  # 1/200

    def test_asymmetric_resolution(self):
        """Test grid with different resolution in each dimension."""
        spec = GridSpec(size=(4.0, 2.0, 1.0), resolution=(50.0, 100.0, 200.0))
        grid = YeeGrid(spec)

        assert abs(grid.dx - 0.02) < 1e-10  # 1/50
        assert abs(grid.dy - 0.01) < 1e-10  # 1/100
        assert abs(grid.dz - 0.005) < 1e-10  # 1/200

        assert grid.Nx == 200  # ceil(4.0 * 50)
        assert grid.Ny == 200  # ceil(2.0 * 100)
        assert grid.Nz == 200  # ceil(1.0 * 200)

    def test_zero_boundary_layers(self):
        """Test grid with no boundary layers."""
        spec = GridSpec(size=(2.0, 2.0, 2.0), resolution=10.0, boundary_layers=0)
        grid = YeeGrid(spec)

        assert grid.pml_layers == 0
        assert grid.Nx_total == grid.Nx
        assert grid.Ny_total == grid.Ny
        assert grid.Nz_total == grid.Nz

    def test_grid_string_representation(self):
        """Test string representation of grid."""
        spec = GridSpec(size=(3.0, 2.0, 1.0), resolution=20.0)
        grid = YeeGrid(spec)

        repr_str = repr(grid)
        assert "YeeGrid" in repr_str
        assert "3D" in repr_str
        assert "PML" in repr_str

        # Test 2D representation
        spec_2d = GridSpec(size=(3.0, 2.0, 0.0), resolution=20.0)
        grid_2d = YeeGrid(spec_2d)
        repr_str_2d = repr(grid_2d)
        assert "2D" in repr_str_2d


class TestGridProperties:
    """Test grid properties and derived quantities."""

    @pytest.fixture
    def standard_grid(self):
        """Create a standard test grid."""
        spec = GridSpec(size=(6.0, 4.0, 2.0), resolution=30.0, boundary_layers=6)
        return YeeGrid(spec)

    def test_dimensions_property(self, standard_grid):
        """Test dimensions property."""
        dims = standard_grid.dimensions
        assert dims == (
            standard_grid.Nx_total,
            standard_grid.Ny_total,
            standard_grid.Nz_total,
        )

    def test_physical_dimensions_property(self, standard_grid):
        """Test physical_dimensions property."""
        phys_dims = standard_grid.physical_dimensions
        assert phys_dims == (standard_grid.Nx, standard_grid.Ny, standard_grid.Nz)

    def test_spacing_property(self, standard_grid):
        """Test spacing property."""
        spacing = standard_grid.spacing
        assert spacing == (standard_grid.dx, standard_grid.dy, standard_grid.dz)

    def test_grid_origin(self, standard_grid):
        """Test grid origin calculation."""
        expected_origin = np.array(
            [
                -standard_grid.pml_layers * standard_grid.dx,
                -standard_grid.pml_layers * standard_grid.dy,
                -standard_grid.pml_layers * standard_grid.dz,
            ]
        )
        np.testing.assert_allclose(standard_grid.origin, expected_origin)

    def test_coordinate_offset_consistency(self, standard_grid):
        """Test that coordinate offsets are consistent across components."""
        # Get coordinates for different components
        x_ex, y_ex, z_ex = standard_grid.get_coordinates("Ex")
        x_hy, y_hy, z_hy = standard_grid.get_coordinates("Hy")

        # Check that offsets are half grid spacing
        # Ex has no offset in x, Hy has offset in x
        expected_offset = standard_grid.dx / 2
        actual_offset = x_hy[0] - x_ex[0]
        assert abs(actual_offset - expected_offset) < 1e-10
