"""MAP under the flat Dirichlet against other NPMLE solvers (issue #77).

The references are written by the scripts in `tests/data/golden/`, which describe each
problem and the software. The NPMLE's log-likelihood is unique, and so are its weights
where they are sparse, as mixsqp's are.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import binference

DATA = Path(__file__).parent / "data" / "golden"
REFERENCES = {
    **json.loads((DATA / "mixsqp.json").read_text()),
    **json.loads((DATA / "empirikos.json").read_text()),
}
# each problem's likelihood matrix from its stored inputs, as the scripts build it
LIKELIHOODS = {
    "scale": lambda p: stats.norm.pdf(
        np.array(p["x"])[:, None], scale=np.sqrt(1 + np.square(p["s"]))
    ),
    "glmix": lambda p: stats.norm.pdf(
        np.array(p["y"])[:, None], p["grid"], np.array(p["se"])[:, None]
    ),
    "bichsel": lambda p: stats.poisson.pmf(np.array(p["z"])[:, None], p["grid"]),
    "tacks": lambda p: stats.binom.pmf(
        np.array(p["x"])[:, None], np.array(p["k"])[:, None], p["grid"]
    ),
}


def _npmle(name):
    ref = REFERENCES[name]
    L = LIKELIHOODS[name](ref)
    result = binference.infer(binference.Likelihoods(L, weights=ref.get("count")), method="map")
    return L, ref, result


@pytest.mark.parametrize("name", ["scale", "glmix"])
def test_map_matches_the_mixsqp_npmle(name):
    L, ref, result = _npmle(name)
    w0 = np.array(ref["weights"])
    # both solvers stop at a KKT gap near 1e-8; the log-likelihood agrees far closer
    assert result.log_likelihood == pytest.approx(ref["loglik"], rel=1e-10)
    np.testing.assert_allclose(L @ result.rho, L @ w0, rtol=1e-5)
    np.testing.assert_allclose(result.rho, w0, atol=1e-5)


@pytest.mark.parametrize("name", ["bichsel", "tacks"])
def test_map_reaches_the_empirikos_npmle_log_likelihood(name):
    _, ref, result = _npmle(name)
    # Hypatia stops short of the optimum, by about 1e-7 relative
    gap = (result.log_likelihood - ref["loglik"]) / abs(ref["loglik"])
    assert -1e-12 < gap < 1e-6
