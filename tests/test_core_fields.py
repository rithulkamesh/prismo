"""
Test suite for core Fields functionality.

This module tests the ElectromagneticFields class, including field storage,
manipulation, energy calculations, and memory management.
"""

import pytest
import numpy as np
from prismo.core.grid import YeeGrid, GridSpec
from prismo.core.fields import ElectromagneticFields


class TestElectromagneticFieldsCreation:
    """Test cases for ElectromagneticFields creation and basic operations."""

    @pytest.fixture
    def grid_2d(self):
        """Create a 2D grid for testing."""
        spec = GridSpec(size=(4.0, 3.0, 0.0), resolution=20.0, boundary_layers=4)
        return YeeGrid(spec)

    @pytest.fixture
    def grid_3d(self):
        """Create a 3D grid for testing."""
        spec = GridSpec(size=(3.0, 2.0, 1.5), resolution=30.0, boundary_layers=3)
        return YeeGrid(spec)

    @pytest.fixture
    def fields_2d(self, grid_2d):
        """Create 2D electromagnetic fields."""
        return ElectromagneticFields(grid_2d)

    @pytest.fixture
    def fields_3d(self, grid_3d):
        """Create 3D electromagnetic fields."""
        return ElectromagneticFields(grid_3d)

    def test_fields_initialization_2d(self, fields_2d, grid_2d):
        """Test 2D fields initialization."""
        # Check that all field components exist and have correct shapes
        components = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

        for component in components:
            field = fields_2d[component]
            expected_shape = grid_2d.get_field_shape(component)
            assert field.shape == expected_shape
            assert field.dtype == np.float64  # Default dtype
            assert np.all(field == 0.0)  # Should be initialized to zero

    def test_fields_initialization_3d(self, fields_3d, grid_3d):
        """Test 3D fields initialization."""
        components = ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

        for component in components:
            field = fields_3d[component]
            expected_shape = grid_3d.get_field_shape(component)
            assert field.shape == expected_shape
            assert field.dtype == np.float64
            assert np.all(field == 0.0)

    def test_fields_custom_dtype(self, grid_2d):
        """Test fields with custom dtype."""
        fields = ElectromagneticFields(grid_2d, dtype=np.float32)

        for component in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            assert fields[component].dtype == np.float32

    def test_field_access_by_name(self, fields_2d):
        """Test field access by component name."""
        # Test getter
        ex_field = fields_2d["Ex"]
        assert isinstance(ex_field, np.ndarray)

        # Test invalid component
        with pytest.raises(KeyError, match="Unknown field component"):
            _ = fields_2d["Invalid"]

    def test_field_access_by_property(self, fields_2d):
        """Test field access by property."""
        # Test that properties return the same arrays as indexing
        assert np.array_equal(fields_2d.Ex, fields_2d["Ex"])
        assert np.array_equal(fields_2d.Ey, fields_2d["Ey"])
        assert np.array_equal(fields_2d.Ez, fields_2d["Ez"])
        assert np.array_equal(fields_2d.Hx, fields_2d["Hx"])
        assert np.array_equal(fields_2d.Hy, fields_2d["Hy"])
        assert np.array_equal(fields_2d.Hz, fields_2d["Hz"])


class TestFieldManipulation:
    """Test cases for field value manipulation."""

    @pytest.fixture
    def fields(self):
        """Create test fields."""
        spec = GridSpec(size=(2.0, 2.0, 0.0), resolution=10.0, boundary_layers=2)
        grid = YeeGrid(spec)
        return ElectromagneticFields(grid)

    def test_set_field_scalar(self, fields):
        """Test setting field to scalar value."""
        fields["Ex"] = 1.5
        assert np.all(fields["Ex"] == 1.5)

        # Test that other fields are unaffected
        assert np.all(fields["Ey"] == 0.0)

    def test_set_field_array(self, fields):
        """Test setting field to array values."""
        ex_shape = fields["Ex"].shape
        test_values = np.random.rand(*ex_shape)

        fields["Ex"] = test_values
        np.testing.assert_allclose(fields["Ex"], test_values)

    def test_set_field_wrong_shape(self, fields):
        """Test error when setting field with wrong shape."""
        wrong_shape_array = np.ones((5, 5))  # Wrong shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            fields["Ex"] = wrong_shape_array

    def test_set_invalid_component(self, fields):
        """Test error when setting invalid field component."""
        with pytest.raises(KeyError, match="Unknown field component"):
            fields["Invalid"] = 1.0

    def test_get_field_components(self, fields):
        """Test getting field component groups."""
        # Set some test values
        fields["Ex"] = 1.0
        fields["Hy"] = 2.0

        # Test electric field components
        ex, ey, ez = fields.get_electric_field_components()
        assert np.all(ex == 1.0)
        assert np.all(ey == 0.0)
        assert np.all(ez == 0.0)

        # Test magnetic field components
        hx, hy, hz = fields.get_magnetic_field_components()
        assert np.all(hx == 0.0)
        assert np.all(hy == 2.0)
        assert np.all(hz == 0.0)

    def test_get_field_type_components(self, fields):
        """Test getting components by field type."""
        fields["Ey"] = 3.0
        fields["Hz"] = 4.0

        # Test E field type
        ex, ey, ez = fields.get_field_type_components("E")
        assert np.all(ey == 3.0)

        # Test H field type
        hx, hy, hz = fields.get_field_type_components("H")
        assert np.all(hz == 4.0)

        # Test invalid field type
        with pytest.raises(ValueError, match="Unknown field type"):
            fields.get_field_type_components("Invalid")


class TestFieldOperations:
    """Test cases for field operations and calculations."""

    @pytest.fixture
    def test_fields(self):
        """Create test fields with known values."""
        spec = GridSpec(size=(3.0, 3.0, 3.0), resolution=10.0, boundary_layers=2)
        grid = YeeGrid(spec)
        fields = ElectromagneticFields(grid)

        # Set test values
        fields["Ex"] = 1.0  # V/m
        fields["Ey"] = 0.0
        fields["Ez"] = 0.0
        fields["Hx"] = 0.0
        fields["Hy"] = 1.0 / (4 * np.pi * 1e-7)  # A/m (normalized to give equal energy)
        fields["Hz"] = 0.0

        return fields

    def test_zero_fields_all(self, test_fields):
        """Test zeroing all fields."""
        test_fields.zero_fields()

        for component in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            assert np.all(test_fields[component] == 0.0)

    def test_zero_fields_electric(self, test_fields):
        """Test zeroing only electric fields."""
        original_hy = test_fields["Hy"].copy()

        test_fields.zero_fields("E")

        # Electric fields should be zero
        assert np.all(test_fields["Ex"] == 0.0)
        assert np.all(test_fields["Ey"] == 0.0)
        assert np.all(test_fields["Ez"] == 0.0)

        # Magnetic fields should be unchanged
        np.testing.assert_allclose(test_fields["Hy"], original_hy)

    def test_zero_fields_magnetic(self, test_fields):
        """Test zeroing only magnetic fields."""
        original_ex = test_fields["Ex"].copy()

        test_fields.zero_fields("H")

        # Magnetic fields should be zero
        assert np.all(test_fields["Hx"] == 0.0)
        assert np.all(test_fields["Hy"] == 0.0)
        assert np.all(test_fields["Hz"] == 0.0)

        # Electric fields should be unchanged
        np.testing.assert_allclose(test_fields["Ex"], original_ex)

    def test_zero_fields_invalid_type(self, test_fields):
        """Test error for invalid field type."""
        with pytest.raises(ValueError, match="Unknown field type"):
            test_fields.zero_fields("Invalid")

    def test_copy_fields_from(self):
        """Test copying fields from another Fields object."""
        # Create two identical grids
        spec = GridSpec(size=(2.0, 2.0, 0.0), resolution=15.0)
        grid1 = YeeGrid(spec)
        grid2 = YeeGrid(spec)

        fields1 = ElectromagneticFields(grid1)
        fields2 = ElectromagneticFields(grid2)

        # Set values in fields1
        fields1["Ex"] = 2.5
        fields1["Hz"] = 1.7

        # Copy to fields2
        fields2.copy_fields_from(fields1)

        # Check that values were copied
        np.testing.assert_allclose(fields2["Ex"], fields1["Ex"])
        np.testing.assert_allclose(fields2["Hz"], fields1["Hz"])

    def test_copy_fields_incompatible_grid(self):
        """Test error when copying from incompatible grid."""
        spec1 = GridSpec(size=(2.0, 2.0, 0.0), resolution=10.0)
        spec2 = GridSpec(size=(3.0, 3.0, 0.0), resolution=10.0)  # Different size

        grid1 = YeeGrid(spec1)
        grid2 = YeeGrid(spec2)

        fields1 = ElectromagneticFields(grid1)
        fields2 = ElectromagneticFields(grid2)

        with pytest.raises(ValueError, match="Grid dimensions must match"):
            fields2.copy_fields_from(fields1)

    def test_copy_fields_wrong_type(self):
        """Test error when copying from wrong type."""
        spec = GridSpec(size=(2.0, 2.0, 0.0), resolution=10.0)
        grid = YeeGrid(spec)
        fields = ElectromagneticFields(grid)

        with pytest.raises(
            TypeError, match="Can only copy from another ElectromagneticFields"
        ):
            fields.copy_fields_from("not_a_fields_object")


class TestFieldEnergy:
    """Test cases for field energy calculations."""

    @pytest.fixture
    def energy_test_fields(self):
        """Create fields with known energy."""
        spec = GridSpec(size=(1.0, 1.0, 1.0), resolution=10.0, boundary_layers=1)
        grid = YeeGrid(spec)
        fields = ElectromagneticFields(grid)

        # Set uniform fields: E = 1 V/m, H = 1/(μ₀c) A/m for equal energies
        mu0 = 4 * np.pi * 1e-7
        c = 299792458.0
        h_value = 1.0 / (mu0 * c)

        fields["Ex"] = 1.0
        fields["Hx"] = h_value

        return fields, grid

    def test_electric_field_energy(self, energy_test_fields):
        """Test electric field energy calculation."""
        fields, grid = energy_test_fields

        e_energy = fields.get_field_energy("E")

        # Should be positive and reasonable
        assert e_energy > 0

        # For uniform field E=1 V/m in volume V, energy = (1/2)ε₀E²V
        eps0 = 8.854187817e-12
        dx, dy, dz = grid.spacing
        volume = dx * dy * dz
        n_cells = fields["Ex"].size  # Number of field points
        expected_energy = 0.5 * eps0 * 1.0**2 * volume * n_cells

        # Should be close (within order of magnitude due to grid discretization)
        assert abs(e_energy - expected_energy) / expected_energy < 1.0

    def test_magnetic_field_energy(self, energy_test_fields):
        """Test magnetic field energy calculation."""
        fields, grid = energy_test_fields

        h_energy = fields.get_field_energy("H")

        # Should be positive
        assert h_energy > 0

    def test_total_field_energy(self, energy_test_fields):
        """Test total field energy calculation."""
        fields, grid = energy_test_fields

        e_energy = fields.get_field_energy("E")
        h_energy = fields.get_field_energy("H")
        total_energy = fields.get_field_energy()  # No field_type = total

        # Total should equal sum of E and H energies
        assert abs(total_energy - (e_energy + h_energy)) < 1e-15

    def test_energy_with_region(self):
        """Test energy calculation in specific region."""
        spec = GridSpec(size=(2.0, 2.0, 0.0), resolution=20.0, boundary_layers=3)
        grid = YeeGrid(spec)
        fields = ElectromagneticFields(grid)

        # Set field only in a subset of the domain
        fields["Ex"][10:20, 10:20] = 2.0  # Small region with field

        # Calculate energy in the entire domain
        total_energy = fields.get_field_energy("E")

        # Calculate energy in the region with field
        region = (slice(10, 20), slice(10, 20))
        region_energy = fields.get_field_energy("E", region)

        # Region energy should be less than total (since field is only in subset)
        assert region_energy <= total_energy
        assert region_energy > 0


class TestFieldMagnitude:
    """Test cases for field magnitude calculations."""

    @pytest.fixture
    def magnitude_test_fields(self):
        """Create fields for magnitude testing."""
        spec = GridSpec(size=(2.0, 2.0, 2.0), resolution=8.0, boundary_layers=2)
        grid = YeeGrid(spec)
        fields = ElectromagneticFields(grid)

        # Set orthogonal field components with known magnitude
        fields["Ex"] = 3.0  # 3-4-5 triangle
        fields["Ey"] = 4.0
        fields["Ez"] = 0.0
        # Expected |E| = sqrt(3² + 4² + 0²) = 5.0

        return fields

    def test_electric_field_magnitude(self, magnitude_test_fields):
        """Test electric field magnitude calculation."""
        fields = magnitude_test_fields

        e_magnitude = fields.get_field_magnitude("E")

        # Should be close to 5.0 everywhere (within numerical precision)
        # Note: This is approximate due to field interpolation to common grid
        assert e_magnitude.shape[0] > 0  # Should have some points
        assert np.all(e_magnitude >= 0)  # Magnitude should be non-negative

        # Check that it's reasonably close to expected value
        expected_magnitude = 5.0
        mean_magnitude = np.mean(e_magnitude)
        assert abs(mean_magnitude - expected_magnitude) / expected_magnitude < 0.5

    def test_magnetic_field_magnitude(self, magnitude_test_fields):
        """Test magnetic field magnitude calculation."""
        fields = magnitude_test_fields

        # Set magnetic field components
        fields["Hx"] = 1.0
        fields["Hy"] = 0.0
        fields["Hz"] = 1.0
        # Expected |H| = sqrt(1² + 0² + 1²) = sqrt(2)

        h_magnitude = fields.get_field_magnitude("H")

        assert h_magnitude.shape[0] > 0
        assert np.all(h_magnitude >= 0)

        expected_magnitude = np.sqrt(2.0)
        mean_magnitude = np.mean(h_magnitude)
        assert abs(mean_magnitude - expected_magnitude) / expected_magnitude < 0.5

    def test_magnitude_invalid_field_type(self, magnitude_test_fields):
        """Test error for invalid field type in magnitude calculation."""
        fields = magnitude_test_fields

        with pytest.raises(ValueError, match="Unknown field type"):
            fields.get_field_magnitude("Invalid")


class TestBoundaryConditions:
    """Test cases for boundary condition application."""

    @pytest.fixture
    def boundary_test_fields(self):
        """Create fields for boundary condition testing."""
        spec = GridSpec(size=(1.0, 1.0, 0.0), resolution=12.0, boundary_layers=2)
        grid = YeeGrid(spec)
        fields = ElectromagneticFields(grid)

        # Set uniform field values
        fields["Ex"] = 1.0
        fields["Ey"] = 1.0
        fields["Ez"] = 1.0

        return fields

    def test_pec_boundary_conditions(self, boundary_test_fields):
        """Test perfect electric conductor boundary conditions."""
        fields = boundary_test_fields

        # Apply PEC boundary conditions
        fields.apply_boundary_conditions("pec")

        # Tangential E fields should be zero at boundaries
        # Check a few boundary points (exact implementation may vary)
        assert np.all(fields["Ey"][0, :] == 0)  # Left boundary
        assert np.all(fields["Ey"][-1, :] == 0)  # Right boundary
        assert np.all(fields["Ex"][:, 0] == 0)  # Bottom boundary
        assert np.all(fields["Ex"][:, -1] == 0)  # Top boundary

    def test_invalid_boundary_condition(self, boundary_test_fields):
        """Test error for invalid boundary condition."""
        fields = boundary_test_fields

        with pytest.raises(
            NotImplementedError, match="Boundary condition 'invalid' not implemented"
        ):
            fields.apply_boundary_conditions("invalid")


class TestMemoryUsage:
    """Test cases for memory usage tracking."""

    def test_memory_usage_calculation(self):
        """Test memory usage calculation."""
        spec = GridSpec(size=(1.0, 1.0, 0.0), resolution=10.0, boundary_layers=1)
        grid = YeeGrid(spec)
        fields = ElectromagneticFields(grid)

        mem_info = fields.get_memory_usage()

        # Check structure
        assert "total_bytes" in mem_info
        assert "total_megabytes" in mem_info
        assert "total_gigabytes" in mem_info
        assert "components" in mem_info

        # Check that total bytes is sum of components
        component_bytes = sum(info["bytes"] for info in mem_info["components"].values())
        assert mem_info["total_bytes"] == component_bytes

        # Check conversions
        assert (
            abs(mem_info["total_megabytes"] - mem_info["total_bytes"] / (1024**2))
            < 1e-10
        )
        assert (
            abs(mem_info["total_gigabytes"] - mem_info["total_bytes"] / (1024**3))
            < 1e-10
        )

        # Check component info
        for component in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]:
            comp_info = mem_info["components"][component]
            assert "shape" in comp_info
            assert "dtype" in comp_info
            assert "bytes" in comp_info
            assert "megabytes" in comp_info

    def test_memory_usage_different_dtypes(self):
        """Test memory usage with different data types."""
        spec = GridSpec(size=(1.0, 1.0, 1.0), resolution=5.0)
        grid = YeeGrid(spec)

        fields_float64 = ElectromagneticFields(grid, dtype=np.float64)
        fields_float32 = ElectromagneticFields(grid, dtype=np.float32)

        mem_64 = fields_float64.get_memory_usage()["total_bytes"]
        mem_32 = fields_float32.get_memory_usage()["total_bytes"]

        # float64 should use twice the memory of float32
        assert abs(mem_64 - 2 * mem_32) < 100  # Allow some small difference


class TestFieldsStringRepresentation:
    """Test cases for string representation."""

    def test_repr_and_str(self):
        """Test __repr__ and __str__ methods."""
        spec = GridSpec(size=(2.0, 1.0, 0.0), resolution=15.0)
        grid = YeeGrid(spec)
        fields = ElectromagneticFields(grid)

        repr_str = repr(fields)
        str_str = str(fields)

        # Both should contain key information
        for s in [repr_str, str_str]:
            assert "ElectromagneticFields" in s
            assert "grid=" in s
            assert "dtype=" in s
            assert "memory=" in s
            assert "MB" in s

    def test_repr_consistency(self):
        """Test that __repr__ and __str__ are consistent."""
        spec = GridSpec(size=(1.0, 1.0, 1.0), resolution=10.0)
        grid = YeeGrid(spec)
        fields = ElectromagneticFields(grid)

        assert repr(fields) == str(fields)
