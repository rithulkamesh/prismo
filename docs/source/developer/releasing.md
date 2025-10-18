# Releasing to PyPI

Complete guide for publishing Prismo releases to PyPI.

## Quick Release

For maintainers who have already set up PyPI:

```bash
# 1. Ensure everything is ready
pytest tests/ -v
ruff check src/
black --check src/

# 2. Create and push tag
git tag -a v0.2.0 -m "Release v0.2.0: Add mode ports"
git push origin v0.2.0

# 3. Create GitHub Release
# Go to: https://github.com/rithulkamesh/prismo/releases/new
# Select the tag, add release notes, publish

# 4. Automatic PyPI upload via GitHub Actions! ✨
# Check: https://pypi.org/project/prismo/
```

## First-Time Setup

### 1. PyPI Account

```bash
# Create accounts
# - https://pypi.org/account/register/
# - https://test.pypi.org/account/register/

# Enable 2FA (required)
```

### 2. Configure Trusted Publishing

**On PyPI:**

1. Go to: https://pypi.org/manage/account/publishing/
2. Click "Add a new publisher"
3. Fill in:
   - PyPI Project Name: `pyprismo`
   - Owner: `rithulkamesh`
   - Repository: `prismo`
   - Workflow: `publish-pypi.yml`
   - Environment: `pypi`

**Important**: The PyPI package name is `pyprismo` (because `prismo` was taken), but the import name remains `prismo`.

**On GitHub:**

1. Go to: Repository → Settings → Environments
2. Create environment: `pypi`
3. (Optional) Add protection rules requiring approval

Repeat for TestPyPI with environment `testpypi`.

### 3. Verify Configuration

The repository already includes:

- ✅ `.github/workflows/publish-pypi.yml` - Automated publishing
- ✅ `pyproject.toml` - Package metadata
- ✅ `MANIFEST.in` - File inclusion rules
- ✅ `LICENSE` - MIT license
- ✅ `README.md` - Package description

## Local Testing

### Build Package Locally

```bash
# Use the provided script
./scripts/build_package.sh

# Or manually
python -m build

# Check the built files
twine check dist/*
```

### Test Installation Locally

```bash
# Install from local wheel
pip install dist/pyprismo-*.whl

# Verify
python -c "import prismo; print(prismo.__version__)"
python -c "from prismo import Simulation; print('✓ Import successful')"
```

### Test on TestPyPI

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    pyprismo

# Test
python -c "import prismo; print(prismo.__version__)"
```

## Automated Release (Recommended)

### Step 1: Prepare

```bash
# Update version in commit messages or prepare changelog
# Version is automatically determined from git tags

# Run full test suite
pytest tests/ -v

# Ensure code quality
ruff check src/
black src/
```

### Step 2: Tag

```bash
# Create annotated tag
git tag -a v0.2.0 -m "Release v0.2.0

Added features:
- Mode port boundary conditions
- S-parameter extraction
- Complete documentation

Bug fixes:
- Fixed ElementTree import

Breaking changes:
- None
"

# Push tag
git push origin v0.2.0
```

### Step 3: Create GitHub Release

1. Go to https://github.com/rithulkamesh/prismo/releases/new
2. Select tag: `v0.2.0`
3. Release title: `v0.2.0 - Mode Ports and Complete Documentation`
4. Description:

   ````markdown
   ## 🎉 What's New

   ### Features

   - ✨ Mode port boundary conditions for waveguide simulations
   - ✨ S-parameter extraction and analysis
   - ✨ Complete documentation on ReadTheDocs
   - ✨ Four comprehensive tutorials

   ### Improvements

   - Enhanced mode source implementation
   - Better mode expansion monitors
   - Mode matching utilities

   ### Bug Fixes

   - Fixed missing ElementTree import

   ### Documentation

   - 📚 Complete user guides
   - 📚 API reference
   - 📚 Tutorials (basic → advanced)
   - 📚 Developer documentation

   ## 📦 Installation

   ```bash
   pip install pyprismo==0.2.0

   # With GPU support
   pip install pyprismo[acceleration]==0.2.0
   ```
   ````

   ## 📖 Documentation

   https://prismo.readthedocs.io/

   ## 🙏 Contributors

   Thanks to all contributors!

   ```

   ```

5. Click "Publish release"

**GitHub Actions will automatically:**

- Build the package
- Run tests
- Publish to PyPI
- Update documentation on ReadTheDocs

### Step 4: Verify

```bash
# Wait 2-5 minutes for PyPI to process

# Check PyPI page
open https://pypi.org/project/pyprismo/

# Test installation
pip install --upgrade pyprismo
python -c "import prismo; print(f'Version: {prismo.__version__}')"

# Run quick verification
python examples/verify_installation.py
```

## Version Management

### Current Version

Version is automatically determined by `hatch-vcs` from git tags:

```bash
# Check current version
python -c "from prismo import __version__; print(__version__)"

# Or
git describe --tags
```

### Dev Versions

Commits between tags get dev versions:

- Tag: `v0.1.0`
- Next commit: `0.1.1.dev1+g1234567`

### Version Scheme

```
v0.1.0          → 0.1.0           (release)
v0.1.0-rc1      → 0.1.0rc1        (release candidate)
v0.1.0-alpha    → 0.1.0a0         (alpha)
v0.1.0-beta.1   → 0.1.0b1         (beta)
```

## Release Types

### Patch Release (0.1.0 → 0.1.1)

Bug fixes only:

```bash
git tag -a v0.1.1 -m "Patch release: Bug fixes"
```

### Minor Release (0.1.0 → 0.2.0)

New features (backward compatible):

```bash
git tag -a v0.2.0 -m "Minor release: New features"
```

### Major Release (0.2.0 → 1.0.0)

Breaking changes or major milestone:

```bash
git tag -a v1.0.0 -m "Major release: Production ready"
```

### Pre-releases

Alpha, beta, or release candidates:

```bash
# Alpha
git tag -a v0.2.0a1 -m "Alpha release for testing"

# Beta
git tag -a v0.2.0b1 -m "Beta release"

# Release Candidate
git tag -a v0.2.0rc1 -m "Release candidate 1"
```

## PyPI Package Page

Enhance your PyPI presence:

### Project Description

The README.md is automatically used as the long description on PyPI.

### Project URLs

Already configured in `pyproject.toml`:

```toml
[project.urls]
Homepage = "https://github.com/rithulkamesh/prismo"
Documentation = "https://prismo.readthedocs.io"
Repository = "https://github.com/rithulkamesh/prismo.git"
Issues = "https://github.com/rithulkamesh/prismo/issues"
```

### Classifiers

Update as project matures:

```toml
classifiers = [
    "Development Status :: 3 - Alpha",  # Update to "4 - Beta" or "5 - Production/Stable"
    # ... other classifiers
]
```

## Rollback

If a release has critical issues:

```bash
# 1. Delete the GitHub release (not the tag)
# 2. Contact PyPI to yank the release
#    (https://pypi.org/help/#yanked)
# 3. Release a patch version with fixes
git tag -a v0.2.1 -m "Fix critical bug from v0.2.0"
```

Note: You cannot delete PyPI releases, only "yank" them.

## Automation Summary

**Current Setup:**

- ✅ Automated PyPI publishing via GitHub Actions
- ✅ Trusted Publishing (no tokens needed)
- ✅ Automatic version from git tags
- ✅ Build verification
- ✅ TestPyPI support

**When you create a GitHub release:**

1. GitHub Actions automatically triggers
2. Package is built
3. Tests run
4. Upload to PyPI happens automatically
5. Package is available at https://pypi.org/project/prismo/

**No manual steps required after GitHub release!** 🎉

## See Also

- {doc}`contributing` - Development workflow
- {doc}`testing` - Test guidelines
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions](https://docs.github.com/en/actions)
