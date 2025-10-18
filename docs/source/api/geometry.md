# Geometry API

Complete API reference for geometric primitives and shapes in Prismo.

## Geometric Shapes

### Box

```{eval-rst}
.. autoclass:: prismo.geometry.shapes.Box
   :members:
   :undoc-members:
   :show-inheritance:
```

### Sphere

```{eval-rst}
.. autoclass:: prismo.geometry.shapes.Sphere
   :members:
   :undoc-members:
   :show-inheritance:
```

### Cylinder

```{eval-rst}
.. autoclass:: prismo.geometry.shapes.Cylinder
   :members:
   :undoc-members:
   :show-inheritance:
```

### Polygon

```{eval-rst}
.. autoclass:: prismo.geometry.shapes.Polygon
   :members:
   :undoc-members:
   :show-inheritance:
```

## Composite Structures

### GeometryGroup

```{eval-rst}
.. autoclass:: prismo.geometry.shapes.GeometryGroup
   :members:
   :undoc-members:
   :show-inheritance:
```

## Usage Examples

### Creating Basic Shapes

```python
from prismo.geometry import Box, Cylinder
from prismo.materials import Material

# Silicon waveguide core
si = Material(epsilon=3.48**2)

core = Box(
    center=(0, 0, 0),
    size=(10e-6, 0.5e-6, 0.22e-6),
    material=si,
)

# Cylindrical post
post = Cylinder(
    center=(5e-6, 0, 0.5e-6),
    radius=0.3e-6,
    height=1e-6,
    material=si,
    axis='z',
)
```

### Combining Shapes

```python
from prismo.geometry import GeometryGroup

# Create a group of structures
waveguide_array = GeometryGroup()
for i in range(5):
    wg = Box(
        center=(i * 5e-6, 0, 0),
        size=(3e-6, 0.5e-6, 0.22e-6),
        material=si,
    )
    waveguide_array.add(wg)

# Add to simulation
sim.add_structure(waveguide_array)
```

## See Also

- {doc}`../user_guide/simulations` - Simulation setup
- {doc}`materials` - Material definitions
- {doc}`../examples/index` - Geometry examples

