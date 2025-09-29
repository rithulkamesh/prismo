# Prismo FDTD Solver - Makefile
# A comprehensive build system for the Prismo photonics simulation library

# Project configuration
PROJECT_NAME := prismo
PYTHON_VERSION := 3.11
VENV_DIR := .venv
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs
EXAMPLES_DIR := examples
BENCHMARKS_DIR := benchmarks

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Default target
.DEFAULT_GOAL := help

# Check if we're in a Nix shell
ifdef PRISMO_DEV
	PYTHON := python
	UV := uv
else
	PYTHON := $(VENV_DIR)/bin/python
	UV := $(VENV_DIR)/bin/uv
endif

##@ Help
help: ## Display this help message
	@echo "$(BLUE)Prismo FDTD Solver - Development Commands$(RESET)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(YELLOW)<target>$(RESET)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(YELLOW)%-20s$(RESET) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(RESET)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Environment Setup
install: ## Install the project in development mode
	@echo "$(BLUE)Installing Prismo in development mode...$(RESET)"
	$(UV) sync --all-extras
	$(UV) pip install -e .
	@echo "$(GREEN)✓ Installation complete!$(RESET)"

install-prod: ## Install production dependencies only
	@echo "$(BLUE)Installing production dependencies...$(RESET)"
	$(UV) sync --no-dev
	$(UV) pip install .
	@echo "$(GREEN)✓ Production installation complete!$(RESET)"

dev-install: install ## Alias for install (development mode)

clean-install: ## Clean environment and reinstall
	@echo "$(BLUE)Cleaning and reinstalling...$(RESET)"
	rm -rf $(VENV_DIR) .uv-cache uv.lock
	$(MAKE) install
	@echo "$(GREEN)✓ Clean installation complete!$(RESET)"

##@ Code Quality
format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(RESET)"
	$(PYTHON) -m black $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	$(PYTHON) -m isort $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	@echo "$(GREEN)✓ Code formatted!$(RESET)"

lint: ## Lint code with ruff and mypy
	@echo "$(BLUE)Linting code...$(RESET)"
	$(PYTHON) -m ruff check $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	$(PYTHON) -m mypy $(SRC_DIR)
	@echo "$(GREEN)✓ Linting complete!$(RESET)"

lint-fix: ## Fix linting issues automatically
	@echo "$(BLUE)Fixing linting issues...$(RESET)"
	$(PYTHON) -m ruff check --fix $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	@echo "$(GREEN)✓ Linting fixes applied!$(RESET)"

check: lint ## Run all code quality checks
	@echo "$(GREEN)✓ All quality checks passed!$(RESET)"

pre-commit-install: ## Install pre-commit hooks
	@echo "$(BLUE)Installing pre-commit hooks...$(RESET)"
	$(PYTHON) -m pre_commit install
	@echo "$(GREEN)✓ Pre-commit hooks installed!$(RESET)"

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(RESET)"
	$(PYTHON) -m pre_commit run --all-files
	@echo "$(GREEN)✓ Pre-commit checks complete!$(RESET)"

##@ Testing
test: ## Run tests with pytest
	@echo "$(BLUE)Running tests...$(RESET)"
	$(PYTHON) -m pytest $(TEST_DIR) -v
	@echo "$(GREEN)✓ Tests complete!$(RESET)"

test-cov: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(RESET)"
	$(PYTHON) -m pytest $(TEST_DIR) --cov=$(SRC_DIR)/$(PROJECT_NAME) --cov-report=html --cov-report=term-missing -v
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(RESET)"

test-fast: ## Run tests in parallel (fast)
	@echo "$(BLUE)Running tests in parallel...$(RESET)"
	$(PYTHON) -m pytest $(TEST_DIR) -n auto -v
	@echo "$(GREEN)✓ Fast tests complete!$(RESET)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(RESET)"
	$(PYTHON) -m pytest $(TEST_DIR) -m "integration" -v
	@echo "$(GREEN)✓ Integration tests complete!$(RESET)"

test-benchmark: ## Run benchmark tests
	@echo "$(BLUE)Running benchmark tests...$(RESET)"
	$(PYTHON) -m pytest $(BENCHMARKS_DIR) --benchmark-only -v
	@echo "$(GREEN)✓ Benchmark tests complete!$(RESET)"

test-gpu: ## Run GPU-specific tests (requires CUDA)
	@echo "$(BLUE)Running GPU tests...$(RESET)"
	$(PYTHON) -m pytest $(TEST_DIR) -m "gpu" -v
	@echo "$(GREEN)✓ GPU tests complete!$(RESET)"

##@ Documentation
docs: ## Build documentation with Sphinx
	@echo "$(BLUE)Building documentation...$(RESET)"
	cd $(DOCS_DIR) && $(PYTHON) -m sphinx -b html . _build/html
	@echo "$(GREEN)✓ Documentation built in docs/_build/html/$(RESET)"

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation at http://localhost:8000$(RESET)"
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

docs-clean: ## Clean documentation build
	@echo "$(BLUE)Cleaning documentation...$(RESET)"
	rm -rf $(DOCS_DIR)/_build
	@echo "$(GREEN)✓ Documentation cleaned!$(RESET)"

docs-auto: ## Auto-rebuild documentation on changes
	@echo "$(BLUE)Auto-rebuilding documentation...$(RESET)"
	cd $(DOCS_DIR) && $(PYTHON) -m sphinx_autobuild . _build/html --host 0.0.0.0 --port 8000

##@ Building and Distribution
build: ## Build source and wheel distributions
	@echo "$(BLUE)Building distributions...$(RESET)"
	$(UV) build
	@echo "$(GREEN)✓ Build complete! Check dist/ directory$(RESET)"

build-check: ## Check built distributions
	@echo "$(BLUE)Checking distributions...$(RESET)"
	$(PYTHON) -m twine check dist/*
	@echo "$(GREEN)✓ Distribution check complete!$(RESET)"

publish-test: ## Upload to TestPyPI
	@echo "$(BLUE)Publishing to TestPyPI...$(RESET)"
	$(PYTHON) -m twine upload --repository testpypi dist/*
	@echo "$(GREEN)✓ Published to TestPyPI!$(RESET)"

publish: ## Upload to PyPI (production)
	@echo "$(YELLOW)Warning: This will publish to production PyPI!$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo; \
		echo "$(BLUE)Publishing to PyPI...$(RESET)"; \
		$(PYTHON) -m twine upload dist/*; \
		echo "$(GREEN)✓ Published to PyPI!$(RESET)"; \
	else \
		echo; \
		echo "$(YELLOW)Publication cancelled.$(RESET)"; \
	fi

##@ Development Utilities
profile: ## Profile the code for performance analysis
	@echo "$(BLUE)Running performance profiling...$(RESET)"
	$(PYTHON) -m cProfile -o profile.stats -m prismo.examples.basic_waveguide
	@echo "$(GREEN)✓ Profile saved to profile.stats$(RESET)"

memory-profile: ## Profile memory usage
	@echo "$(BLUE)Running memory profiling...$(RESET)"
	$(PYTHON) -m memory_profiler $(EXAMPLES_DIR)/basic_simulation.py
	@echo "$(GREEN)✓ Memory profiling complete!$(RESET)"

run-example: ## Run a basic example simulation
	@echo "$(BLUE)Running basic example...$(RESET)"
	$(PYTHON) -m prismo.examples.basic_waveguide
	@echo "$(GREEN)✓ Example simulation complete!$(RESET)"

jupyter: ## Start Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter notebook server...$(RESET)"
	cd $(EXAMPLES_DIR) && $(PYTHON) -m jupyter notebook

##@ Cleanup
clean: ## Clean build artifacts and cache
	@echo "$(BLUE)Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "$(GREEN)✓ Cleanup complete!$(RESET)"

clean-all: clean docs-clean ## Clean everything including docs
	@echo "$(BLUE)Deep cleaning...$(RESET)"
	rm -rf $(VENV_DIR)
	rm -rf .uv-cache
	rm -rf uv.lock
	@echo "$(GREEN)✓ Deep cleanup complete!$(RESET)"

##@ CI/CD
ci: format lint test-cov build build-check ## Run full CI pipeline
	@echo "$(GREEN)✓ CI pipeline complete!$(RESET)"

ci-fast: lint test build ## Run fast CI pipeline (no coverage)
	@echo "$(GREEN)✓ Fast CI pipeline complete!$(RESET)"

##@ Project Info
info: ## Show project information
	@echo "$(BLUE)Project Information$(RESET)"
	@echo "Name: $(PROJECT_NAME)"
	@echo "Python Version: $(PYTHON_VERSION)"
	@echo "Virtual Environment: $(VENV_DIR)"
	@echo "Source Directory: $(SRC_DIR)"
	@echo "Test Directory: $(TEST_DIR)"
	@echo ""
	@echo "$(BLUE)Environment Status$(RESET)"
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "✓ Virtual environment exists"; \
		$(PYTHON) --version 2>/dev/null && echo "✓ Python is available" || echo "✗ Python not found"; \
		$(UV) --version 2>/dev/null && echo "✓ UV is available" || echo "✗ UV not found"; \
	else \
		echo "✗ Virtual environment not found"; \
	fi

deps-update: ## Update all dependencies to latest versions
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	$(UV) lock --upgrade
	@echo "$(GREEN)✓ Dependencies updated!$(RESET)"

security-check: ## Run security vulnerability checks
	@echo "$(BLUE)Running security checks...$(RESET)"
	$(PYTHON) -m safety check
	$(PYTHON) -m bandit -r $(SRC_DIR)
	@echo "$(GREEN)✓ Security checks complete!$(RESET)"

##@ Benchmarking
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running performance benchmarks...$(RESET)"
	cd $(BENCHMARKS_DIR) && $(PYTHON) -m pytest --benchmark-only --benchmark-sort=mean
	@echo "$(GREEN)✓ Benchmarks complete!$(RESET)"

benchmark-compare: ## Compare benchmark results
	@echo "$(BLUE)Comparing benchmark results...$(RESET)"
	cd $(BENCHMARKS_DIR) && $(PYTHON) -m pytest --benchmark-compare --benchmark-sort=mean
	@echo "$(GREEN)✓ Benchmark comparison complete!$(RESET)"

# Phony targets
.PHONY: help install install-prod dev-install clean-install format lint lint-fix check pre-commit-install pre-commit test test-cov test-fast test-integration test-benchmark test-gpu docs docs-serve docs-clean docs-auto build build-check publish-test publish profile memory-profile run-example jupyter clean clean-all ci ci-fast info deps-update security-check benchmark benchmark-compare