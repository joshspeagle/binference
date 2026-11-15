"""MAP under the flat Dirichlet against other NPMLE solvers (issue #77).

The references in `tests/data/golden/` hold each problem's inputs and another solver's
solution, written by the scripts beside them (software versions inside):
`make_mixsqp.R` (mixsqp, active-set SQP) and `make_empirikos.jl` (Empirikos with
Hypatia, an interior-point solver). The NPMLE's log-likelihood and fitted values
L w are unique; its weights are unique where they are sparse, as mixsqp's are.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import binference
from binference import Bins

DATA = Path(__file__).parent / "data" / "golden"
MIXSQP = json.loads((DATA / "mixsqp.json").read_text())
EMPIRIKOS = json.loads((DATA / "empirikos.json").read_text())


def _npmle(L, counts=None):
    """binference's NPMLE on a grid: MAP under the default flat Dirichlet."""
    lk = binference.Likelihoods(L, Bins.categories(L.shape[1]), weights=counts)
    w = binference.infer(lk, method="map").rho
    return w, (np.ones(len(L)) if counts is None else counts) @ np.log(L @ w)


def _mixsqp_problem(name):
    p = MIXSQP[name]
    if name == "scale":  # mixsqp's simulatemixdata: N(x_i; 0, 1 + s_j^2)
        L = stats.norm.pdf(np.array(p["x"])[:, None], scale=np.sqrt(1 + np.square(p["s"])))
    else:  # REBayes's GLmix problem: N(y_i; v_j, se_i^2)
        L = stats.norm.pdf(np.array(p["y"])[:, None], p["grid"], np.array(p["se"])[:, None])
    return L, p


@pytest.mark.parametrize("name", ["scale", "glmix"])
def test_map_matches_the_mixsqp_npmle(name):
    L, ref = _mixsqp_problem(name)
    w0 = np.array(ref["weights"])
    w, loglik = _npmle(L)
    # both solvers stop at a KKT gap near 1e-8; the log-likelihood agrees far closer
    assert loglik == pytest.approx(ref["loglik"], rel=1e-10)
    np.testing.assert_allclose(L @ w, L @ w0, rtol=1e-5)
    np.testing.assert_allclose(w, w0, atol=1e-5)


def _empirikos_problem(name):
    p = EMPIRIKOS[name]
    grid = np.array(p["grid"])
    if name == "bichsel":
        L = stats.poisson.pmf(np.array(p["z"])[:, None], grid)
    else:
        L = stats.binom.pmf(np.array(p["x"])[:, None], np.array(p["k"])[:, None], grid)
    return L, np.array(p["count"]), p


@pytest.mark.parametrize("name", ["bichsel", "tacks"])
def test_map_reaches_the_empirikos_npmle_log_likelihood(name):
    L, counts, ref = _empirikos_problem(name)
    _, loglik = _npmle(L, counts)
    # Hypatia stops within about 1e-7 of the optimum (NearOptimal on Bichsel), so MAP may
    # exceed it by that much but never fall short of it beyond round-off
    gap = (loglik - ref["loglik"]) / abs(ref["loglik"])
    assert -1e-12 < gap < 1e-6
