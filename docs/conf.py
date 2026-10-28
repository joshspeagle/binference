"""Sphinx configuration for the binference documentation."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../src"))
sys.path.insert(0, os.path.abspath("_ext"))
# the executed pages' plot style, written by _scripts/brand.py; the notebook kernels inherit it
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
    "sphinxcontrib.bibtex",
    "bnf_cite",
    "bnf_explainer",
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
# keyed on the plot style too: a new style re-executes every page
_style = hashlib.sha256(Path(os.environ["MATPLOTLIBRC"]).read_bytes()).hexdigest()[:12]
nb_execution_cache_path = os.path.abspath(f"_build/.jupyter_cache/{_style}")
nb_execution_timeout = 900
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_merge_streams = True

# -- Bibliography -------------------------------------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year_round"

# -- API reference ------------------------------------------------------------
autosummary_generate = True
autodoc_default_options = {"members": True, "inherited-members": False}
autodoc_typehints = "none"
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
# every parameter of the public API is documented; the first excluded classes are built
# by the library, and their fields are listed under Attributes
numpydoc_validation_checks = {"PR01"}
numpydoc_validation_exclude = {
    r"\.(Density|Measure|PointEstimate|PopulationResult|Mock|Truth|CoverageResult"
    r"|ExactPosterior|SBCResult)$",
    r"\.(PopulationPrior|Hyperprior)\.__init__$",  # base classes: object's *args, **kwargs
}

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

# -- Link check (weekly in CI) --------------------------------------------------
# DOIs are checked against the DOI resolver instead (_scripts/check_dois.py), since many
# publishers answer automated requests with 403; git.ligo.org does the same.
linkcheck_ignore = [r"https://doi\.org/", r"https://git\.ligo\.org/"]
