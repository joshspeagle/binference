"""Engines against stored reference posteriors of benchmark-matrix cells (DECISIONS #42).

The references (`tests/data/matrix_reference.json`) are 20,000-draw NUTS runs written by
``python benchmarks/matrix.py run --only n-100,width-0.3`` (cells whose reference had no
divergence); each catalogue is rebuilt
from the matrix's own builder and seed. Every mean must agree within four Monte Carlo
standard errors, from the bulk ESS of both runs (importance ESS for Laplace), and every
s.d. within four standard errors of an s.d. estimate, sqrt((kurtosis - 1) / (4 ESS))
relative, with the ESS of the squared deviations (about half the bulk ESS under NUTS).
"""

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import binference
from binference.diagnostics import ess

pytestmark = pytest.mark.slow

ROOT = Path(__file__).parents[1]
REFERENCES = {
    (r["scenario"], r["prior"]): r
    for r in json.loads((ROOT / "tests" / "data" / "matrix_reference.json").read_text())
}
# the engines checked per cell, and the one method="auto" picks there
ENGINES = {
    ("n-100", "dirichlet"): (("nuts", "gibbs"), "gibbs"),
    ("width-0.3", "randomwalk"): (("nuts",), "nuts"),
    ("width-0.3", "dirichlet"): (("nuts", "gibbs"), "gibbs"),
    ("width-0.3", "polyatree"): (("laplace",), "gibbs"),
}


@pytest.fixture(scope="module")
def matrix():
    spec = importlib.util.spec_from_file_location("matrix", ROOT / "benchmarks" / "matrix.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["matrix"] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def _ess(res):
    """ESS of the draws and of their squared deviations (importance ESS for Laplace)."""
    if res.method == "laplace":
        return res.info["ess_importance"], res.info["ess_importance"]
    sq = (res.chains - res.mean) ** 2
    return res.diagnostics["ess_bulk"], np.array([ess(sq[..., k]) for k in range(res.K)])


@pytest.mark.parametrize(
    ("cell", "engine"),
    [(cell, e) for cell, (engines, _) in ENGINES.items() for e in ("auto", *engines)],
)
def test_engines_reproduce_the_reference_posterior_of_a_matrix_cell(matrix, cell, engine):
    name, prior_name = cell
    sc = matrix.SCENARIOS[name]
    lk, _ = matrix.catalogue(name)
    ref = REFERENCES[cell]
    assert ref["n_divergent"] == 0 and ref["rhat_max"] < 1.01
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", binference.BinferenceWarning)
        res = binference.infer(lk, sc.priors[prior_name](), method=engine, rng=3)
    if engine == "auto":  # the matrix's verdict for this cell
        assert res.method == ENGINES[cell][1]
    mean, sd = np.asarray(ref["mean"]), np.asarray(ref["sd"])
    ess_mean, ess_sq = _ess(res)
    z_mean = (res.mean - mean) / (sd * np.sqrt(1 / ess_mean + 1 / ref["ess"]))
    kurt = stats.kurtosis(res.samples, axis=0, fisher=False)
    z_sd = (res.std / sd - 1) / np.sqrt((kurt - 1) / 4 * (1 / ess_sq + 2 / ref["ess"]))
    assert np.max(np.abs(z_mean)) < 4, z_mean.round(1)
    assert np.max(np.abs(z_sd)) < 4, z_sd.round(1)
