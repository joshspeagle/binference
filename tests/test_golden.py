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
MIXSQP, EMPIRIKOS = (
    {k: v for k, v in json.loads((DATA / f).read_text()).items() if k != "software"}
    for f in ("mixsqp.json", "empirikos.json")
)
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


def _npmle(name, ref):
    L = LIKELIHOODS[name](ref)
    result = binference.infer(binference.Likelihoods(L, weights=ref.get("count")), method="map")
    return L, result


@pytest.mark.parametrize("name", MIXSQP)
def test_map_matches_the_mixsqp_npmle(name):
    ref = MIXSQP[name]
    L, result = _npmle(name, ref)
    w0 = np.array(ref["weights"])
    # mixsqp ran to convtol.sqp = 1e-10 and MAP stops at a KKT gap of 2e-9 or less here;
    # the log-likelihoods agreed to 2e-15 relative
    assert result.log_likelihood == pytest.approx(ref["loglik"], rel=1e-10)
    np.testing.assert_allclose(L @ result.rho, L @ w0, rtol=1e-5)
    np.testing.assert_allclose(result.rho, w0, atol=1e-5)


@pytest.mark.parametrize("name", EMPIRIKOS)
def test_map_reaches_the_empirikos_npmle_log_likelihood(name):
    ref = EMPIRIKOS[name]
    _, result = _npmle(name, ref)
    # Hypatia stops short of the optimum, here by 1e-8 to 1e-7 relative. MAP stops at a
    # KKT gap below 1e-8 per object, so it falls short of the optimum by less than 1e-8
    # times the number of objects, under 1e-8 relative here (|loglik| exceeds 2 per object).
    gap = (result.log_likelihood - ref["loglik"]) / abs(ref["loglik"])
    assert -1e-8 < gap < 1e-6
