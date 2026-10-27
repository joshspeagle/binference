"""Sphinx configuration for the binference documentation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath("_ext"))
# the tutorials' plot style, written by _scripts/brand.py; the notebook kernels inherit it
os.environ["MATPLOTLIBRC"] = os.path.abspath("_ext/matplotlibrc")

from binference import __version__

project = "binference"
author = "Joshua S. Speagle"
copyright = "2026, Joshua S. Speagle"
release = __version__
version = ".".join(__version__.split(".")[:2])

extensions = [
    "bnf_theme",
    "myst_nb",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = [
    "_build",
    "_scripts",
    "_ext",
    "_themes",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb"}

# -- MyST / notebooks ---------------------------------------------------------
myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3
nb_execution_mode = os.environ.get("BINFERENCE_DOCS_EXECUTE", "cache")
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

# -- HTML ---------------------------------------------------------------------
html_theme = "binference"
html_theme_path = ["_themes"]
html_title = "binference"
html_static_path = ["_static"]
html_favicon = "_themes/binference/static/brand/favicon.svg"
html_theme_options = {"source_repository": "https://github.com/joshspeagle/binference"}
html_copy_source = False
# MathJax 3, pinned (Sphinx's default loads an unpinned version)
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-mml-chtml.js"
