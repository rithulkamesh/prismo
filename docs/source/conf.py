# Configuration file for the Sphinx documentation builder.

import os
import sys
from pathlib import Path

# Add the source directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# -- Project information -----------------------------------------------------
project = "Prismo"
copyright = "2024, Rithul Kamesh"
author = "Rithul Kamesh"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
try:
    from prismo import __version__

    version = __version__
    release = __version__
except ImportError:
    version = "0.1.0-dev"
    release = "0.1.0-dev"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "nbsphinx",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Theme options are theme-specific and customize the look and feel of a theme
html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "",
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# -- Extension configuration -------------------------------------------------

# autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# autosummary configuration
autosummary_generate = True
autosummary_generate_overwrite = False

# napoleon configuration
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
}

# nbsphinx configuration
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True
nbsphinx_kernel_name = "python3"

# MyST configuration
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
    "preamble": "",
    "figure_align": "htbp",
}

# Grouping the document tree into LaTeX files. List of tuples
latex_documents = [
    (master_doc, "prismo.tex", "Prismo Documentation", "Rithul Kamesh", "manual"),
]

# -- Options for manual page output ------------------------------------------
man_pages = [(master_doc, "prismo", "Prismo Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (
        master_doc,
        "Prismo",
        "Prismo Documentation",
        author,
        "Prismo",
        "A high-performance Python-based FDTD solver for waveguide photonics.",
        "Miscellaneous",
    ),
]
