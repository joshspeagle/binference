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
    assert _plan(lk, gibbs_max_missing=0.99999, max_gibbs_draws=10**7)[0] == "gibbs"
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
        res = binference.infer(_lk(3.0), P.Dirichlet(0.1), rng=0, nnz_limit=1)
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
    with pytest.raises(ValueError, match="None of the engines"):
        binference.infer(lk, df=4, target_accept=0.9)


def test_a_hopeless_laplace_trial_stops_after_its_first_thousand_proposals(monkeypatch):
    from binference.samplers import Laplace

    calls = []
    run = Laplace.run
    monkeypatch.setattr(
        Laplace, "run", lambda self, n: calls.append(self.n_proposals) or run(self, n)
    )
    model = binference.PopulationModel(_lk(3.0), P.Dirichlet(0.1))
    res, reason = inf.Auto(model)._try(Laplace(model, rng=0), 4000)
    assert res is None and reason == "the Laplace importance check failed"
    assert calls == [1000]


@pytest.mark.filterwarnings("ignore::binference.ApproximationWarning")
def test_laplace_finds_its_mode_once_per_engine(monkeypatch):
    from binference.samplers import Laplace

    engine = Laplace(binference.PopulationModel(_lk(1.0)), rng=0)
    first = engine.run(100)
    monkeypatch.setattr(Laplace, "_find_mode_and_eigen", lambda self: pytest.fail("refound"))
    assert np.array_equal(engine.run(100).info["mode"], first.info["mode"])


def test_auto_works_on_basis_likelihoods(short_nuts):
    # the missing information is taken on the component weights, not the bin fractions
    bins = Bins.linear(0.0, 1.0, 6)
    rng = np.random.default_rng(0)
    z = rng.uniform(0.0, 1.0, 300)
    basis = binference.Basis(bins, order=1, resolution=2)
    lk = binference.Likelihoods.from_gaussian(z + rng.normal(0, 0.1, 300), 0.1, basis)
    res = binference.infer(lk, rng=0, n_samples=50, check=False)
    assert res.info["auto_method"] in ("gibbs", "laplace", "nuts")


def test_auto_options_are_checked_by_name():
    model = binference.PopulationModel(_lk(0.3))
    for kw, name in (
        ({"target_ess": 0}, "target_ess"),
        ({"gibbs_max_missing": 1.0}, "gibbs_max_missing"),
        ({"nnz_limit": -1}, "nnz_limit"),
    ):
        with pytest.raises(ValueError, match=name):
            inf.Auto(model, **kw)


def test_laplace_draws_as_many_as_four_chains_would():
    res = binference.infer(_lk(1.0, n=1000, K=20), P.PolyaTree(), rng=0, n_samples=300)
    assert res.method == "laplace" and res.n_draws == 1200


def test_laplace_mode_and_covariance_are_copies_and_follow_the_step():
    from binference.samplers import Laplace

    engine = Laplace(binference.PopulationModel(_lk(1.0), P.PolyaTree()), rng=0)
    _, cov, _ = engine.mode_and_covariance()
    cov *= 2
    np.testing.assert_array_equal(engine.mode_and_covariance()[1] * 2, cov)
    engine.step = 1e-4
    engine.mode_and_covariance()
    assert engine._eigen[0] == 1e-4  # found again for the new step


def test_a_gibbs_run_is_extended_until_it_meets_the_target_ess_and_r_hat():
    # a selection inferred jointly mixes more slowly than the default 1000 draws allow
    from .test_injections import _problem

    lk, sel = _problem()
    res = binference.infer(lk, selection=sel, rng=0, target_ess=600)
    d = res.diagnostics
    assert res.method == "gibbs" and res.chains.shape[1] > 1000
    assert min(np.min(d["ess_bulk"]), np.min(d["ess_bulk_selection"])) >= 600
    assert max(np.max(d["rhat"]), np.max(d["rhat_selection"])) <= 1.01


def test_gibbs_gives_way_when_it_cannot_mix_within_its_draw_limit(short_nuts):
    from .test_injections import _problem

    lk, sel = _problem()
    res = binference.infer(
        lk, selection=sel, rng=0, target_ess=600, max_gibbs_draws=1100, check=False
    )
    assert res.method != "gibbs" and res.info["auto_reason"].startswith("Gibbs mixed too slowly")
    # and a run the formula already puts beyond the limit is not tried
    assert _plan(_lk(3.0), gibbs_max_missing=0.9999, max_gibbs_draws=1500)[0] == "laplace"
