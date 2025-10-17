# Materials and Dispersion

Prismo provides comprehensive support for modeling realistic optical materials with frequency-dependent properties.

## Material Library

Access pre-defined materials with validated optical properties:

```python
import prismo

# List available materials
materials = prismo.list_materials()
# ['Si', 'SiO2', 'Si3N4', 'Au', 'Ag', 'Al', 'ITO']

# Get a material
silicon = prismo.get_material('Si')
silica = prismo.get_material('SiO2')
gold = prismo.get_material('Au')

# Calculate refractive index
wavelength = 1.55e-6  # 1550 nm
omega = 2 * np.pi * 299792458.0 / wavelength
n = silicon.refractive_index(omega)
print(f"n_Si @ 1550nm = {n.real:.3f}")  # ~3.48
```

### Available Materials

| Material                | Type      | n @ 1550nm | Applications           |
| ----------------------- | --------- | ---------- | ---------------------- |
| Si (Silicon)            | Lorentz   | 3.48       | Waveguides, modulators |
| SiO2 (Silica)           | Sellmeier | 1.44       | Cladding, substrates   |
| Si3N4 (Silicon Nitride) | Sellmeier | 2.0        | Low-loss waveguides    |
| Au (Gold)               | Drude     | -          | Plasmonic devices      |
| Ag (Silver)             | Drude     | -          | Plasmonic devices      |
| Al (Aluminum)           | Drude     | -          | Mirrors, interconnects |
| ITO                     | Drude     | -          | Transparent conductors |

## Dispersion Models

### Lorentz Model

For dielectrics with resonances:

```python
from prismo import LorentzMaterial, LorentzPole

material = LorentzMaterial(
    epsilon_inf=2.0,
    poles=[
        LorentzPole(
            omega_0=2 * np.pi * 200e12,  # Resonance at 200 THz
            delta_epsilon=1.5,            # Oscillator strength
            gamma=1e13,                   # Damping rate
        )
    ],
    name="CustomDielectric"
)

# Add to library
prismo.add_material('MyMaterial', material)
```

### Drude Model

For metals and plasmas:

```python
from prismo import DrudeMaterial

metal = DrudeMaterial(
    epsilon_inf=1.0,
    omega_p=2 * np.pi * 2e15,  # Plasma frequency
    gamma=1e13,                 # Collision frequency
    name="CustomMetal"
)
```

### Debye Model

For dielectric relaxation:

```python
from prismo import DebyeMaterial

material = DebyeMaterial(
    epsilon_inf=2.0,
    epsilon_s=10.0,  # Static permittivity
    tau=1e-12,        # Relaxation time
    name="PolarDielectric"
)
```

## Anisotropic Materials

### Uniaxial Materials

Materials with different indices parallel and perpendicular to an axis (e.g., liquid crystals):

```python
# Create uniaxial material
liquid_crystal = prismo.create_uniaxial_material(
    n_ordinary=1.5,
    n_extraordinary=1.7,
    optic_axis='z',
    name="LiquidCrystal"
)
```

### Biaxial Materials

Materials with three different principal indices:

```python
biaxial = prismo.create_biaxial_material(
    nx=2.0,
    ny=2.2,
    nz=2.5,
    name="BiaxialCrystal"
)
```

### Custom Tensor Materials

Full control over permittivity and permeability tensors:

```python
from prismo import TensorMaterial, TensorComponents

# Define permittivity tensor
epsilon = TensorComponents(
    xx=4.0, yy=4.5, zz=5.0,  # Diagonal
    xy=0.1, xz=0.0, yz=0.0    # Off-diagonal
)

material = TensorMaterial(epsilon=epsilon, name="CustomAnisotropic")
```

## Time-Domain Integration

Dispersive materials are automatically integrated in time-domain simulations using the Auxiliary Differential Equation (ADE) method. This happens transparently when you use dispersive materials in your simulation.
