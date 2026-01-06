{
  description = "Prismo - A high-performance Python-based FDTD solver for waveguide photonics";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # Custom Python with required packages
        python = pkgs.python311;

        # Base development dependencies (without GUI - fast to build)
        baseDevDeps =
          with pkgs;
          [
            # Core Python and package management
            python
            uv

            # Build tools and compilers for C extensions
            gcc
            gfortran
            cmake
            pkg-config

            # Linear algebra libraries
            blas
            lapack
            openblas
            fftw

            # HDF5 for data storage
            hdf5

            # Git and version control
            git
            git-lfs

            # Documentation tools
            pandoc
            texlive.combined.scheme-medium

            # Development utilities
            gnumake
            pre-commit

            # Performance monitoring
            htop

            # Jupyter and notebook dependencies
            nodejs_20

            # Note: CUDA support can be added by setting NIXPKGS_ALLOW_UNFREE=1
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
            # Linux-specific packages
            glibc
          ];

        # GUI dependencies (only included in gui shell)
        # Using lightweight Dear PyGui instead of heavy Qt/PySide6
        # Note: Dear PyGui uses X11 (works on Wayland via XWayland)
        # PyVista requires additional X11 libraries for VTK rendering
        guiDeps = with pkgs; [
          # Minimal GUI dependencies for Dear PyGui
          # X11 libraries (required even on Wayland - XWayland provides compatibility)
          xorg.libX11
          xorg.libXext
          xorg.libXrender # Required for PyVista/VTK rendering
          xorg.libXrandr # For display management
          xorg.libXi # For input handling
          xorg.libXfixes # X fixes extension
          libGL
          mesa # OpenGL implementation
          # Dear PyGui is installed via pip/uv, not nix (much lighter!)
        ];

        # Base shell hook (without GUI)
        baseShellHook = ''
          echo "🔬 Welcome to Prismo FDTD Solver Development Environment"
          echo "📦 Python version: $(python --version)"
          echo "🚀 UV version: $(uv --version)"
          echo ""

          # Set up environment variables
          export PYTHONPATH="$PWD/src:$PYTHONPATH"
          export PRISMO_DEV=1

          # Set up UV cache directory
          export UV_CACHE_DIR="$PWD/.uv-cache"
          mkdir -p "$UV_CACHE_DIR"

          # Set up library paths for compiled extensions
          export LD_LIBRARY_PATH="${
            pkgs.lib.makeLibraryPath [
              pkgs.blas
              pkgs.lapack
              pkgs.openblas
              pkgs.fftw
              pkgs.hdf5
            ]
          }:$LD_LIBRARY_PATH"

          # Set up PKG_CONFIG_PATH
          export PKG_CONFIG_PATH="${
            pkgs.lib.makeSearchPathOutput "dev" "lib/pkgconfig" [
              pkgs.blas
              pkgs.lapack
              pkgs.openblas
              pkgs.fftw
              pkgs.hdf5
            ]
          }:$PKG_CONFIG_PATH"

          # CUDA setup can be added manually if needed
          # To enable CUDA: export NIXPKGS_ALLOW_UNFREE=1 before nix develop

          # Initialize UV project if not already done
          if [ ! -f ".python-version" ]; then
            echo "🔧 Initializing UV project..."
            uv python install 3.11
            uv python pin 3.11
          fi

          # Create virtual environment if it doesn't exist
          if [ ! -d ".venv" ]; then
            echo "🐍 Creating virtual environment..."
            uv venv --python 3.11
          fi

          # Activate virtual environment
          source .venv/bin/activate

          # Install development dependencies if lock file exists
          if [ -f "uv.lock" ]; then
            echo "📦 Installing core dependencies (excluding GPU/visualization)..."
            uv sync --extra dev --extra docs
          else
            echo "📦 Initializing project dependencies..."
            uv add --dev pytest pytest-cov black isort ruff mypy pre-commit
            uv add numpy scipy matplotlib h5py pydantic tqdm rich
          fi

          echo ""
          echo "🎉 Development environment ready!"
          echo "💡 Run 'make help' to see available commands"
          echo "🧪 Run 'make test' to run tests"
          echo "🚀 Run 'make dev-install' to install in development mode"
          echo ""
        '';

        # GUI shell hook (includes minimal GUI library paths)
        guiShellHook = baseShellHook + ''
          # Add GUI library paths for Dear PyGui and PyVista
          export LD_LIBRARY_PATH="${
            pkgs.lib.makeLibraryPath [
              pkgs.libGL
              pkgs.mesa
              pkgs.xorg.libX11
              pkgs.xorg.libXext
              pkgs.xorg.libXrender
              pkgs.xorg.libXrandr
              pkgs.xorg.libXi
              pkgs.xorg.libXfixes
            ]
          }:$LD_LIBRARY_PATH"
        '';

      in
      {
        # Default dev shell (fast - no GUI/QT compilation)
        devShells.default = pkgs.mkShell {
          name = "prismo-dev";
          packages = baseDevDeps;
          shellHook = baseShellHook + ''
            echo "💡 Tip: Use 'nix develop .#gui' for GUI development (lightweight Dear PyGui)"
            echo ""
          '';

          # Environment variables for the shell
          PRISMO_DEV = "1";
          PYTHONPATH = "./src";

          # Ensure proper locale for Python
          LOCALE_ARCHIVE = "${pkgs.glibcLocales}/lib/locale/locale-archive";
          LANG = "en_US.UTF-8";
          LC_ALL = "en_US.UTF-8";
        };

        # GUI dev shell (includes PySide6 - use binary caches for faster builds)
        devShells.gui = pkgs.mkShell {
          name = "prismo-dev-gui";
          packages = baseDevDeps ++ guiDeps;
          shellHook = guiShellHook + ''
            echo "🎨 GUI development environment (Dear PyGui - lightweight!)"
            echo "💡 Install GUI: uv pip install dearpygui"
            echo "🌐 Works on both X11 and Wayland (via XWayland)"
            echo ""
          '';

          # Environment variables for the shell
          PRISMO_DEV = "1";
          PYTHONPATH = "./src";

          # Ensure proper locale for Python
          LOCALE_ARCHIVE = "${pkgs.glibcLocales}/lib/locale/locale-archive";
          LANG = "en_US.UTF-8";
          LC_ALL = "en_US.UTF-8";
        };

        # Package definition for Prismo
        packages.default = python.pkgs.buildPythonPackage rec {
          pname = "prismo";
          version = "0.1.0-dev";

          src = ./.;

          format = "pyproject";

          nativeBuildInputs =
            with python.pkgs;
            [
              setuptools
              wheel
              hatchling
              hatch-vcs
            ]
            ++ (with pkgs; [
              gcc
              gfortran
              cmake
              pkg-config
            ]);

          buildInputs = with pkgs; [
            blas
            lapack
            openblas
            fftw
            hdf5
          ];

          propagatedBuildInputs = with python.pkgs; [
            numpy
            scipy
            matplotlib
            h5py
            pydantic
            tqdm
            rich
          ];

          checkInputs = with python.pkgs; [
            pytest
            pytest-cov
          ];

          # Skip tests during build (they can be run separately)
          doCheck = false;

          meta = with pkgs.lib; {
            description = "A high-performance Python-based FDTD solver for waveguide photonics";
            homepage = "https://github.com/rithulkamesh/prismo";
            license = licenses.mit;
            maintainers = [ maintainers.rkamesh or "rkamesh" ];
            platforms = platforms.unix;
          };
        };

        # Development apps
        apps = {
          # Run tests
          test = flake-utils.lib.mkApp {
            drv = pkgs.writeShellScriptBin "test" ''
              cd "$(git rev-parse --show-toplevel)"
              source .venv/bin/activate
              pytest "$@"
            '';
          };

          # Format code
          format = flake-utils.lib.mkApp {
            drv = pkgs.writeShellScriptBin "format" ''
              cd "$(git rev-parse --show-toplevel)"
              source .venv/bin/activate
              black src/ tests/ examples/
              isort src/ tests/ examples/
            '';
          };

          # Lint code
          lint = flake-utils.lib.mkApp {
            drv = pkgs.writeShellScriptBin "lint" ''
              cd "$(git rev-parse --show-toplevel)"
              source .venv/bin/activate
              ruff check src/ tests/ examples/
              mypy src/
            '';
          };
        };
      }
    );
}
