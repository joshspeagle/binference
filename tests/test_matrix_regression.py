"""Engines against stored reference posteriors of benchmark-matrix cells (DECISIONS #42).

The references (`tests/data/matrix_reference.json`) are 20,000-draw NUTS runs written by
``python benchmarks/matrix.py run --only n-100,width-0.3`` (cells whose reference had no
divergence), with the Monte Carlo errors of their means and s.d.s; each catalogue is
rebuilt from the matrix's own builder and seed. Every mean and s.d. must agree within
four combined Monte Carlo errors: :func:`binference.diagnostics.mcse` and
:func:`sd_mcse` for the samplers, the importance ESS for Laplace.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import binference
from binference import diagnostics as diag

from .conftest import sd_mcse

pytestmark = pytest.mark.slow

REFERENCES = {
    (r["scenario"], r["prior"]): r
    for r in json.loads((Path(__file__).parent / "data" / "matrix_reference.json").read_text())
}
# the engines checked per cell, and the one method="auto" picks there
ENGINES = {
    ("n-100", "dirichlet"): (("nuts", "gibbs"), "gibbs"),
    ("width-0.3", "randomwalk"): (("nuts",), "nuts"),
    ("width-0.3", "dirichlet"): (("nuts", "gibbs"), "gibbs"),
    ("width-0.3", "polyatree"): (("laplace",), "gibbs"),
}


def _mcse(res):
    """Monte Carlo errors of each bin's mean and s.d."""
    if res.method == "laplace":  # resampled draws: errors from the importance ESS
        n = res.info["ess_importance"]
        kurt = stats.kurtosis(res.samples, axis=0, fisher=False)
        return res.std / np.sqrt(n), res.std * np.sqrt((kurt - 1) / (4 * n))
    C = res.chains
    return (
        np.array([diag.mcse(C[..., k]) for k in range(res.K)]),
        np.array([sd_mcse(C[..., k]) for k in range(res.K)]),
    )


@pytest.mark.parametrize(
    ("cell", "engine"),
    # "auto" covers the engine it picks
    [
        (cell, e)
        for cell, (engines, pick) in ENGINES.items()
        for e in ("auto", *(e for e in engines if e != pick))
    ],
)
def test_engines_reproduce_the_reference_posterior_of_a_matrix_cell(matrix, cell, engine):
    name, prior_name = cell
    ref = REFERENCES[cell]
    assert ref["n_divergent"] == 0 and ref["rhat_max"] < 1.01
    lk, _ = matrix.catalogue(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", binference.BinferenceWarning)
        res = binference.infer(
            lk, matrix.SCENARIOS[name].priors[prior_name](), method=engine, rng=3
        )
    if engine == "auto":  # the matrix's verdict for this cell
        assert res.method == ENGINES[cell][1]
    err_mean, err_sd = _mcse(res)
    z_mean = (res.mean - ref["mean"]) / np.hypot(err_mean, ref["mcse_mean"])
    z_sd = (res.std - ref["sd"]) / np.hypot(err_sd, ref["mcse_sd"])
    assert np.max(np.abs(z_mean)) < 4, z_mean.round(1)
    assert np.max(np.abs(z_sd)) < 4, z_sd.round(1)
