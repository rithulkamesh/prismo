#!/bin/bash
# Build script for Prismo package

set -e  # Exit on error

echo "════════════════════════════════════════════════════════"
echo "Prismo Package Build Script"
echo "════════════════════════════════════════════════════════"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Run this script from the project root."
    exit 1
fi

echo ""
echo "Step 1: Clean previous builds..."
rm -rf dist/ build/ *.egg-info
echo "✓ Cleaned"

echo ""
echo "Step 2: Install build tools..."
pip install --upgrade build twine hatch hatchling hatch-vcs
echo "✓ Build tools ready"

echo ""
echo "Step 3: Build package..."
python -m build
echo "✓ Package built"

echo ""
echo "Step 4: Check distribution..."
twine check dist/*
echo "✓ Distribution checked"

echo ""
echo "Step 5: List built files..."
ls -lh dist/
echo ""

echo "════════════════════════════════════════════════════════"
echo "✓ Build complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Built files in dist/:"
for file in dist/*; do
    echo "  - $(basename $file)"
done

echo ""
echo "Next steps:"
echo "  Test locally:  pip install dist/*.whl"
echo "  Upload to TestPyPI:  twine upload --repository testpypi dist/*"
echo "  Upload to PyPI:  twine upload dist/*"
echo ""

