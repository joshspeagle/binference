"""Sphinx configuration for the pastry documentation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

from pastry import __version__  # noqa: E402

project = "pastry"
author = "Joshua S. Speagle"
copyright = "2026, Joshua S. Speagle"
release = __version__
version = ".".join(__version__.split(".")[:2])

extensions = [
    "myst_nb",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = ["_build", "_scripts", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb"}

# -- MyST / notebooks ---------------------------------------------------------
myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3
nb_execution_mode = os.environ.get("PASTRY_DOCS_EXECUTE", "cache")
nb_execution_timeout = 900
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_merge_streams = True

# -- API reference ------------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {"members": True, "inherited-members": False}
autodoc_typehints = "none"
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_validation_checks: set[str] = set()

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# -- HTML ---------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = f"pastry {version}"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/joshspeagle/pastry",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "header_links_before_dropdown": 6,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
}
html_context = {"default_mode": "auto"}
