# Backends API

## Backend Interface

```{eval-rst}
.. automodule:: prismo.backends
   :members:
   :undoc-members:
   :show-inheritance:
```

## Backend Functions

### get_backend

```{eval-rst}
.. autofunction:: prismo.backends.get_backend
```

### set_backend

```{eval-rst}
.. autofunction:: prismo.backends.set_backend
```

### list_available_backends

```{eval-rst}
.. autofunction:: prismo.backends.list_available_backends
```

## Backend Classes

### Backend (Abstract)

```{eval-rst}
.. autoclass:: prismo.backends.base.Backend
   :members:
   :undoc-members:
```

### NumPyBackend

```{eval-rst}
.. autoclass:: prismo.backends.numpy_backend.NumPyBackend
   :members:
   :undoc-members:
```

### CuPyBackend

```{eval-rst}
.. autoclass:: prismo.backends.cupy_backend.CuPyBackend
   :members:
   :undoc-members:
```
