# Materials API

## Material Library

```{eval-rst}
.. autofunction:: prismo.materials.get_material
.. autofunction:: prismo.materials.list_materials
.. autofunction:: prismo.materials.add_material
```

## Dispersion Models

### DispersiveMaterial (Base Class)

```{eval-rst}
.. autoclass:: prismo.materials.dispersion.DispersiveMaterial
   :members:
   :undoc-members:
```

### LorentzMaterial

```{eval-rst}
.. autoclass:: prismo.materials.dispersion.LorentzMaterial
   :members:
   :undoc-members:
```

### DrudeMaterial

```{eval-rst}
.. autoclass:: prismo.materials.dispersion.DrudeMaterial
   :members:
   :undoc-members:
```

### DebyeMaterial

```{eval-rst}
.. autoclass:: prismo.materials.dispersion.DebyeMaterial
   :members:
   :undoc-members:
```

### SellmeierMaterial

```{eval-rst}
.. autoclass:: prismo.materials.dispersion.SellmeierMaterial
   :members:
   :undoc-members:
```

## Data Classes

### LorentzPole

```{eval-rst}
.. autoclass:: prismo.materials.dispersion.LorentzPole
   :members:
   :undoc-members:
```

### DrudePole

```{eval-rst}
.. autoclass:: prismo.materials.dispersion.DrudePole
   :members:
   :undoc-members:
```

### DebyePole

```{eval-rst}
.. autoclass:: prismo.materials.dispersion.DebyePole
   :members:
   :undoc-members:
```

## Anisotropic Materials

### TensorMaterial

```{eval-rst}
.. autoclass:: prismo.materials.tensor.TensorMaterial
   :members:
   :undoc-members:
```

### TensorComponents

```{eval-rst}
.. autoclass:: prismo.materials.tensor.TensorComponents
   :members:
   :undoc-members:
```

### Helper Functions

```{eval-rst}
.. autofunction:: prismo.materials.tensor.create_uniaxial_material
.. autofunction:: prismo.materials.tensor.create_biaxial_material
```

## ADE Solver

```{eval-rst}
.. autoclass:: prismo.materials.ade.ADESolver
   :members:
   :undoc-members:
```

```{eval-rst}
.. autoclass:: prismo.materials.ade.ADEManager
   :members:
   :undoc-members:
```
