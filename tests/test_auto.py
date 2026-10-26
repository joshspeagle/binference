"""method="auto", the default of infer: one test per rule (DECISIONS #88)."""

import functools

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
    assert 0.9 < f <= inf.AUTO_GIBBS_MAX_MISSING  # where the default 1000 draws fall short
    res = binference.infer(lk, rng=0, n_chains=2)
    expected = int(np.ceil(inf.AUTO_TARGET_ESS * (1 + f) / (1 - f) / 2))
    assert res.chains.shape[:2] == (2, expected)
    assert res.diagnostics["ess_bulk"].min() > inf.AUTO_TARGET_ESS / 2


def test_broad_likelihoods_leave_gibbs_and_a_failed_laplace_check_goes_to_nuts(short_nuts):
    lk = _lk(3.0)
    key, _, _ = inf._auto_method(binference.PopulationModel(lk))
    assert key == "laplace?"  # f > 0.98: Gibbs would mix too slowly
    res = binference.infer(lk, rng=0, n_samples=50, check=False)
    assert (
        res.method == "nuts" and res.info["auto_reason"] == "the Laplace importance check failed"
    )


def test_laplace_is_kept_when_its_importance_check_passes():
    res = binference.infer(_lk(1.0, n=1000, K=20), P.PolyaTree(), rng=0)
    assert res.method == "laplace" and not res.approximate
    assert res.info["ess_importance"] >= inf.AUTO_TARGET_ESS


def test_very_large_problems_keep_laplace_flagged_approximate(monkeypatch):
    monkeypatch.setattr(inf, "AUTO_NNZ_LIMIT", 0)
    with pytest.warns(binference.ApproximationWarning, match="k-hat"):
        res = binference.infer(_lk(3.0), P.LogisticRandomWalk(), rng=0)
    assert res.method == "laplace" and res.approximate
    assert "likelihood entries" in res.info["auto_reason"]


def test_engine_options_rule_out_the_engines_that_do_not_take_them():
    model = binference.PopulationModel(_lk(0.3))
    assert inf._auto_method(model, {"n_warmup": 100})[0] == "gibbs"
    assert inf._auto_method(model, {"target_accept": 0.9})[0] == "nuts"
    assert inf._auto_method(model, {"df": 4})[0] == "laplace?"
