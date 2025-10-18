# Contributing to Prismo

Thank you for your interest in contributing to Prismo! This guide will help you get started.

## Quick Start

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/prismo.git
cd prismo

# 2. Install development dependencies
pip install -e ".[dev]"

# 3. Create a branch for your changes
git checkout -b feature/my-new-feature

# 4. Make your changes and test
pytest tests/
black src/prismo/
ruff check src/prismo/

# 5. Commit and push
git add .
git commit -m "Add my new feature"
git push origin feature/my-new-feature

# 6. Open a pull request on GitHub
```

## Development Setup

### Prerequisites

- Python 3.9+
- Git
- (Optional) CUDA for GPU support

### Installation

```bash
# Clone repository
git clone https://github.com/rithulkamesh/prismo.git
cd prismo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev,docs]"

# Verify installation
pytest tests/ -v
```

## Code Style

We follow strict code style guidelines:

### Python Style

- **PEP 8** compliance
- **Type hints** for all public functions
- **NumPy docstrings** format
- **Maximum line length**: 88 characters (Black default)

### Formatting Tools

```bash
# Auto-format with Black
black src/prismo/

# Check with ruff
ruff check src/prismo/

# Type checking with mypy
mypy src/prismo/

# All checks
make lint
```

### Example

```python
def compute_overlap(
    Ex: np.ndarray,
    Ey: np.ndarray,
    mode: WaveguideMode,
    dx: float,
    dy: float,
) -> complex:
    """
    Compute mode overlap integral.

    Parameters
    ----------
    Ex, Ey : ndarray
        Electric field components.
    mode : WaveguideMode
        Waveguide mode for overlap.
    dx, dy : float
        Grid spacing.

    Returns
    -------
    complex
        Overlap coefficient.

    Examples
    --------
    >>> overlap = compute_overlap(Ex, Ey, mode, 1e-8, 1e-8)
    >>> assert abs(overlap) <= 1.0
    """
    # Implementation with type hints and clear logic
    overlap = 0.5 * np.sum((Ex * np.conj(mode.Ey) - Ey * np.conj(mode.Ex))) * dx * dy
    return complex(overlap)
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_modes.py

# Run with coverage
pytest --cov=prismo --cov-report=html

# Run only fast tests
pytest -m "not slow"
```

### Writing Tests

Every new feature must include tests:

```python
import pytest
import numpy as np
from prismo.modes.solver import ModeSolver

class TestModeSolver:
    """Test mode solver functionality."""

    @pytest.fixture
    def simple_waveguide(self):
        """Create a simple test waveguide."""
        x = np.linspace(-2e-6, 2e-6, 50)
        y = np.linspace(-2e-6, 2e-6, 50)
        epsilon = np.ones((50, 50)) * 1.5**2
        return x, y, epsilon

    def test_mode_solver_creation(self, simple_waveguide):
        """Test that ModeSolver can be created."""
        x, y, eps = simple_waveguide
        solver = ModeSolver(1.55e-6, x, y, eps)
        assert solver is not None

    def test_fundamental_mode(self, simple_waveguide):
        """Test solving for fundamental mode."""
        x, y, eps = simple_waveguide
        solver = ModeSolver(1.55e-6, x, y, eps)
        modes = solver.solve(num_modes=1)

        assert len(modes) > 0
        assert modes[0].neff.real > 1.0
        assert modes[0].neff.real < 3.5
```

### Test Categories

Use markers for different test types:

```python
@pytest.mark.slow
def test_large_simulation():
    """This test takes a while."""
    pass

@pytest.mark.gpu
def test_gpu_backend():
    """Requires GPU."""
    pass

@pytest.mark.integration
def test_full_workflow():
    """Tests multiple components together."""
    pass
```

## Documentation

### Docstring Format

Use NumPy style docstrings:

```python
def function_name(param1: type, param2: type) -> return_type:
    """
    Short one-line summary.

    Longer description explaining what the function does,
    how it works, and any important details.

    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type
        Description of param2.

    Returns
    -------
    return_type
        Description of return value.

    Raises
    ------
    ValueError
        When invalid input is provided.

    See Also
    --------
    related_function : Related functionality.

    Examples
    --------
    >>> result = function_name(1.0, 2.0)
    >>> print(result)
    3.0

    Notes
    -----
    Additional mathematical or implementation details.

    References
    ----------
    .. [1] Author, "Title", Journal, Year.
    """
    pass
```

### Building Documentation

```bash
# Build HTML documentation
cd docs
make html

# View documentation
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
start build/html/index.html  # Windows

# Clean and rebuild
make clean html
```

## Pull Request Process

### 1. Before Submitting

- [ ] Code follows style guidelines (Black, Ruff)
- [ ] All tests pass (`pytest`)
- [ ] New tests added for new features
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] Type hints added
- [ ] No linter warnings

### 2. PR Description

Include:

- **What**: Brief description of changes
- **Why**: Motivation for changes
- **How**: Technical approach
- **Testing**: How you tested
- **Screenshots**: If UI/visualization changes

Example:

```markdown
## Add Mode Port Boundary Condition

### What

Implements mode port boundaries for waveguide simulations with mode injection and extraction.

### Why

Enables accurate S-parameter calculations for waveguide devices.

### How

- Created `ModePort` class in `boundaries/mode_port.py`
- Implemented mode overlap integrals
- Added S-parameter extraction methods

### Testing

- Unit tests in `tests/validation/test_mode_ports.py`
- Validated against analytical waveguide solution
- Example in `examples/mode_port_demo.py`

### Related Issues

Closes #6
```

### 3. Review Process

1. Automated checks run (CI/CD)
2. Maintainer review
3. Address feedback
4. Merge when approved

## Commit Messages

Follow conventional commits:

```
type(scope): subject

body

footer
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(modes): add mode port boundary condition

Implement mode ports for waveguide S-parameter extraction.
Includes mode injection, extraction, and overlap integrals.

Closes #6

---

fix(solver): correct Courant condition for 3D

The 3D Courant limit calculation was incorrect.
Now properly accounts for all three dimensions.

---

docs(tutorials): add S-parameter extraction tutorial

New tutorial showing how to extract S-parameters using
mode expansion monitors.
```

## Code Review Guidelines

When reviewing PRs:

### What to Look For

- **Correctness**: Does it work as intended?
- **Tests**: Are there adequate tests?
- **Documentation**: Is it well documented?
- **Performance**: Any performance concerns?
- **Style**: Follows code style?
- **Breaking changes**: Any API changes?

### Providing Feedback

Be constructive and specific:

**Good:**

```
The mode overlap calculation looks correct, but could be optimized
by pre-computing the mode normalization. Consider moving line 45
to the __init__ method.
```

**Not as helpful:**

```
This is slow.
```

## Feature Requests

### Proposing New Features

1. Check existing issues/PRs
2. Open a feature request issue
3. Describe use case and proposed API
4. Discuss with maintainers
5. Implement after approval

### Template

````markdown
## Feature Request: [Feature Name]

### Use Case

Describe what you want to accomplish.

### Proposed API

```python
# Example of how it would be used
sim.add_feature(...)
```
````

### Alternatives Considered

Other approaches you've thought about.

### Additional Context

Any other relevant information.

```

## Getting Help

- **Documentation**: Check docs first
- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Report bugs via GitHub Issues
- **Chat**: Join our developer chat (if available)

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help newcomers feel welcome

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors are recognized in:
- README.md contributors section
- Release notes
- Documentation credits

Thank you for contributing to Prismo! 🎉

```
