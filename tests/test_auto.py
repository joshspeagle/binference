"""method="auto", the default of infer: one test per rule (DECISIONS #88)."""

import functools
import warnings

import numpy as np
import pytest

import binference
from binference import Bins
from binference import inference as inf
from binference import priors as P
from binference.samplers import NUTS


def _lk(width, n=400, K=10, seed=1):
    # values drawn uniformly within bins, Gaussian errors `width` bins wide
    bins = Bins.linear(0.0, 1.0, K)
    rho = binference.simulate.population(bins, "bimodal")
    return binference.simulate.mock_1d(n, bins, density=rho, noise=width / K, rng=seed).likelihoods


def _plan(lk, prior=None, **kw):
    return [
        key for key, _, _ in inf.Auto(binference.PopulationModel(lk, prior), **kw).candidates()
    ]


@pytest.fixture
def short_nuts(monkeypatch):
    monkeypatch.setitem(inf.METHODS, "nuts", functools.partial(NUTS, n_chains=1, n_warmup=50))


def test_auto_is_the_default_and_records_its_choice():
    res = binference.infer(_lk(0.3), rng=0)
    assert res.method == res.info["auto_method"] == "gibbs"
    assert "missing information" in res.info["auto_reason"]


def test_gibbs_runs_long_enough_for_the_target_ess():
    lk = _lk(0.6)
    model = binference.PopulationModel(lk)
    f = model.missing_information(binference.infer(model, method="map").rho)
    assert 0.9 < f <= 0.98  # where the default 1000 draws fall short
    res = binference.infer(lk, rng=0, n_chains=2, target_ess=300)
    expected = int(np.ceil(300 * (1 + f) / (1 - f) / 2))
    assert res.chains.shape[:2] == (2, expected)
    assert res.diagnostics["ess_bulk"].min() > 300 / 2


def test_broad_likelihoods_leave_gibbs_and_a_failed_laplace_check_goes_to_nuts(short_nuts):
    lk = _lk(3.0)
    assert _plan(lk) == ["laplace", "nuts"]  # f > 0.98: Gibbs would mix too slowly
    assert _plan(lk, gibbs_max_missing=1.0)[0] == "gibbs"
    res = binference.infer(lk, P.Dirichlet(0.1), rng=0, n_samples=50, check=False)
    assert (
        res.method == "nuts" and res.info["auto_reason"] == "the Laplace importance check failed"
    )


def test_laplace_is_kept_when_its_importance_check_passes():
    res = binference.infer(_lk(1.0, n=1000, K=20), P.PolyaTree(), rng=0)
    assert res.method == "laplace" and not res.approximate
    assert res.info["ess_importance"] >= 400


def test_very_large_problems_keep_laplace_even_when_its_check_fails():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", binference.ApproximationWarning)
        res = binference.infer(_lk(3.0), P.Dirichlet(0.1), rng=0, nnz_limit=0)
    assert res.method == "laplace" and res.approximate
    assert "likelihood entries" in res.info["auto_reason"]


def test_engine_options_rule_out_the_engines_that_do_not_take_them(short_nuts):
    lk = _lk(0.3)
    assert _plan(lk, n_warmup=100) == ["gibbs", "nuts"]
    assert _plan(lk, target_accept=0.9) == ["nuts"]
    assert _plan(lk, df=4) == ["laplace"]
    res = binference.infer(
        lk, P.LogisticRandomWalk(), rng=0, n_samples=50, target_accept=0.9, check=False
    )
    assert res.info["auto_reason"] == "the only engine that takes these options"
    with pytest.raises(ValueError, match="No engine"):
        binference.infer(lk, df=4, target_accept=0.9)
