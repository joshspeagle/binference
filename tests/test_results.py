"""Tests for PopulationResult and PointEstimate (spec 7; R-OUT-*)."""

import pickle
import warnings

import numpy as np
import pytest

import binference
from binference import Bins, Likelihoods, PopulationModel, PopulationResult
from binference.warnings import ConvergenceWarning


@pytest.fixture(scope="module")
def fitted(small_1d):
    return binference.infer(
        small_1d.likelihoods, method="gibbs", rng=1, n_samples=400, n_warmup=100, check=False
    )


def test_basic_summaries(fitted):
    X = fitted.samples
    assert fitted.n_chains == 4 and fitted.n_draws == 1600 and fitted.K == 12
    np.testing.assert_allclose(X.sum(1), 1.0)
    np.testing.assert_allclose(fitted.mean, X.mean(0))
    np.testing.assert_allclose(fitted.median, np.median(X, 0))
    np.testing.assert_allclose(fitted.std, X.std(0, ddof=1))
    np.testing.assert_allclose(fitted.cov(), np.cov(X.T))
    assert fitted.corr().shape == (12, 12)
    np.testing.assert_allclose(fitted.quantile([0.1, 0.9]), np.quantile(X, [0.1, 0.9], axis=0))
    lo, hi = fitted.interval(0.9)
    np.testing.assert_allclose(lo, np.quantile(X, 0.05, axis=0))
    hlo, hhi = fitted.interval(0.9, kind="hdi")
    assert np.all(hhi - hlo <= hi - lo + 1e-12)
    with pytest.raises(ValueError):
        fitted.interval(1.5)
    with pytest.raises(ValueError):
        fitted.interval(0.5, kind="mode")
    np.testing.assert_allclose(fitted.density(), X / fitted.bins.volumes)
    # derived rates (DECISIONS #78): Lambda | rho ~ Gamma(W, 1) without a selection
    W = fitted.model.W
    np.testing.assert_allclose(fitted.rates, fitted.total[:, None] * X)
    assert abs(fitted.total.mean() - W) < 5 * np.sqrt(W / fitted.n_draws)
    np.testing.assert_allclose(
        fitted.quantile(0.5, which="rates"), np.median(fitted.rates, axis=0)
    )


def test_summary_table(fitted):
    s = fitted.summary()
    assert len(s) == 12
    for col in ("bin", "mean", "sd", "median", "q16", "q84", "ess_bulk", "rhat"):
        assert col in s.columns
    text = repr(s)
    assert "binference fractions posterior" in text and "[0, 0.25)" in text
    np.testing.assert_allclose(s.mean, fitted.mean)
    assert set(s.to_dict()) == set(s.columns)


def test_functional_and_rebin(fitted):
    zbar = fitted.functional(lambda r: r @ fitted.bins.centers)
    assert zbar.shape == (fitted.n_draws,)
    np.testing.assert_allclose(
        fitted.functional(lambda X: X @ fitted.bins.centers, vectorized=True), zbar
    )
    _, mapping = fitted.bins.coarsen(3)
    coarse = fitted.rebin(mapping)
    assert coarse.shape == (fitted.n_draws, 4)
    np.testing.assert_allclose(coarse.sum(1), 1.0)
    np.testing.assert_allclose(
        fitted.rebin([[0, 1, 2], list(range(3, 12))])[:, 0], fitted.samples[:, :3].sum(1)
    )


def test_individual_outputs(fitted, small_1d):
    lk = small_1d.likelihoods
    p = fitted.assignment_probabilities(n_draws=100)
    assert p.shape == (lk.N, lk.K)
    np.testing.assert_allclose(p.sum(1), 1.0)
    # Monte Carlo check against explicit averaging
    m = fitted.model
    R = fitted._select_draws(100)
    ref = np.mean([m.responsibilities(r, idx=[5, 7]) for r in R], axis=0)
    np.testing.assert_allclose(fitted.assignment_probabilities(idx=[5, 7], n_draws=100), ref)
    labels = fitted.sample_assignments(idx=[5], n_draws=None, rng=1)
    assert labels.shape == (fitted.n_draws, 1)
    freq = np.bincount(labels[:, 0], minlength=lk.K) / labels.shape[0]
    np.testing.assert_allclose(
        freq, fitted.assignment_probabilities(idx=[5], n_draws=None)[0], atol=0.04
    )
    counts = fitted.realized_counts(n_draws=50, rng=2)
    assert counts.shape == (50, lk.K)
    np.testing.assert_allclose(counts.sum(1), lk.W)
    vals = fitted.sample_values(idx=[0, 1, 2], n_draws=20, rng=3)
    assert vals.shape == (20, 3)
    # sample_values draws the labels first, as sample_assignments does, so the same seed
    # reproduces them: each value must lie in its own drawn bin
    labels = fitted.sample_assignments(idx=[0, 1, 2], n_draws=20, rng=3)
    e = fitted.bins.edges[0]  # closed on both sides: a draw clipped onto an edge is in
    assert np.all((vals >= e[labels]) & (vals <= e[labels + 1]))


def test_realized_counts_narrower_than_population_for_precise_data():
    rng = np.random.default_rng(4)
    bins = Bins.linear(0, 1, 5)
    labels = rng.integers(0, 5, size=300)
    lk = Likelihoods(np.eye(5)[labels] + 1e-9, bins)
    res = binference.infer(lk, method="gibbs", rng=1, n_samples=500, check=False)
    rc = res.realized_counts(n_draws=200, rng=1)
    # with (essentially) perfect measurements the realised histogram is known exactly
    np.testing.assert_allclose(rc.std(0), 0.0, atol=1e-6)
    np.testing.assert_allclose(rc.mean(0), np.bincount(labels, minlength=5))
    assert np.all((res.samples * lk.W).std(0) > 3.0)  # population fractions still uncertain


def test_realized_counts_add_fractional_weights():
    # objects with non-integer weights take the branch that draws labels and adds their
    # weights, which no other test reaches
    rng = np.random.default_rng(5)
    labels = rng.integers(0, 5, size=40)
    w = rng.choice([0.5, 1.5, 2.25], size=40)
    with pytest.warns(binference.InputWarning, match="pseudo-likelihood"):
        lk = Likelihoods(
            np.eye(5)[labels] + 1e-9, Bins.linear(0, 1, 5), weights=w, fractional_weights=True
        )
    res = PopulationResult.from_samples(PopulationModel(lk), np.full((20, 5), 0.2))
    # near one-hot likelihoods: every draw is the weighted histogram of the true bins
    counts = res.realized_counts(n_draws=None, rng=1)
    expect = np.bincount(labels, weights=w, minlength=5)
    np.testing.assert_allclose(counts, np.broadcast_to(expect, (20, 5)))


def test_a_result_keeps_read_only_copies_of_its_draws_and_diagnostics(small_1d):
    # the draws were the caller's array, and the cached diagnostics writeable: a change on
    # either side changed the result, and what check() reports, under its cached summaries
    model = PopulationModel(small_1d.likelihoods)
    draws = np.random.default_rng(0).dirichlet(np.ones(12), size=(2, 30))
    info = {"hyperparameters": {"tau": np.ones(3)}}
    res = PopulationResult(model, draws, method="test", info=info)
    mean = res.mean
    draws[:] = 1.0 / 12
    info["hyperparameters"]["tau"][0] = 99.0
    np.testing.assert_array_equal(res.mean, mean)
    assert res.info["hyperparameters"]["tau"][0] == 1.0
    res.diagnostics  # cached before pickling: a cache that cannot be pickled fails here
    for r in (res, pickle.loads(pickle.dumps(res))):  # unpickled, still read-only
        for arr in (r.samples, r.total, r.summary().rhat, r.info["hyperparameters"]["tau"]):
            with pytest.raises(ValueError, match="read-only"):
                arr[0] = 0.0
        with pytest.raises(TypeError):
            r.diagnostics["rhat"] = None
    value = np.full(12, 1.0 / 12)
    est = binference.PointEstimate(model, value)
    value[0] = 1.0
    assert est.value[0] == 1.0 / 12 and not est.value.flags.writeable


def test_predict_and_log_predictive(fitted, small_1d):
    new = small_1d.likelihoods.subset(np.arange(20))
    p = fitted.predict_proba(new, n_draws=50)
    assert p.shape == (20, 12)
    np.testing.assert_allclose(p.sum(1), 1.0)
    lp = fitted.log_predictive(new, n_draws=50)
    assert lp.shape == (20,) and np.all(np.isfinite(lp))
    # objects are scored a chunk at a time, with the same values
    np.testing.assert_allclose(fitted.log_predictive(new, n_draws=50, chunk=7), lp, rtol=1e-12)
    # a deliberately wrong population scores worse on held-out data
    draws = np.tile(np.r_[np.ones(6), np.full(6, 1e-3)], (50, 1))
    wrong = PopulationResult(fitted.model, draws / draws.sum(-1, keepdims=True), method="test")
    assert lp.sum() > wrong.log_predictive(new).sum()


def test_new_objects_under_a_selection_on_the_truth_match_the_direct_formula():
    # the selection enters through the column weights, a chunk of rows at a time
    mock = binference.simulate.mock_1d(200, Bins.linear(0, 1, 5), noise=0.15, rng=3)
    S = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    res = binference.infer(
        mock.likelihoods.with_selection(S, "on_truth"),
        method="gibbs",
        n_samples=50,
        rng=0,
        check=False,
    )
    new = mock.likelihoods.subset(np.arange(30)).to_sparse(1e-3)
    R = res._select_draws(50, components=True)
    Lt = new.toarray() * S  # the selected likelihoods
    T = Lt @ R.T
    p_direct = np.mean(Lt[:, None, :] * R[None] / T[:, :, None], axis=1)
    lp_direct = np.log(np.mean(T / (R @ S)[None, :], axis=1)) + new.log_scale
    np.testing.assert_allclose(res.predict_proba(new, n_draws=50, chunk=7), p_direct, rtol=1e-10)
    np.testing.assert_allclose(res.log_predictive(new, n_draws=50, chunk=7), lp_direct, rtol=1e-10)


def test_check_warns_on_stuck_chains(fitted):
    bad = fitted.chains.copy()
    bad[0] = bad[0] * 0.5 + 0.5 * np.eye(12)[0]
    res = PopulationResult(fitted.model, bad, method="gibbs")
    with pytest.warns(ConvergenceWarning, match="R-hat"):
        msgs = res.check()
    assert msgs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError):
            res.check(strict=True)
    assert repr(res).startswith("PopulationResult(method='gibbs'")


def test_point_estimate(small_1d):
    est = binference.infer(small_1d.likelihoods, method="map")
    assert isinstance(est, binference.PointEstimate)
    np.testing.assert_allclose(est.rho.sum(), 1.0)
    np.testing.assert_allclose(est.rates, small_1d.likelihoods.W * est.rho)  # W / (S . rho)
    p = est.assignment_probabilities(idx=[0, 1])
    np.testing.assert_allclose(p.sum(1), 1.0)
    np.testing.assert_allclose(est.density(), est.rho / small_1d.bins.volumes)
    with pytest.warns(binference.ApproximationWarning):
        lap = est.laplace(n_samples=500, rng=1, importance=False)
    assert isinstance(lap, PopulationResult) and lap.approximate
    assert "PointEstimate(method='map'" in repr(est)


def test_from_samples_and_attach(fitted, small_1d):
    ext = PopulationResult.from_samples(fitted.model, fitted.chains, method="emcee")
    np.testing.assert_allclose(ext.mean, fitted.mean)
    detached = PopulationResult(None, fitted.chains, method="nuts", bins=small_1d.bins)
    with pytest.raises(RuntimeError, match="attach"):
        detached.assignment_probabilities()
    detached.attach(small_1d.likelihoods)
    assert detached.assignment_probabilities(idx=[0], n_draws=10).shape == (1, 12)
    with pytest.raises(ValueError):
        detached.attach(Likelihoods(np.ones((3, 4))))


def test_rates_result():
    mock = binference.simulate.mock_1d(300, Bins.linear(0, 3, 4), rng=8)
    res = binference.infer(
        mock.likelihoods,
        binference.priors.GammaRates(1.0, 1e-3),
        method="gibbs",
        rng=1,
        n_samples=300,
        check=False,
    )
    assert res.kind == "rates"
    np.testing.assert_allclose(res.samples.sum(1), 1.0)
    np.testing.assert_allclose(res.rates.sum(1).mean(), 300, rtol=0.15)
    np.testing.assert_allclose(res.density(), res.rates / mock.bins.volumes)
    assert res.summary(which="rates").title.startswith("binference rates")


def test_categorical_labels_in_summary():
    mock = binference.simulate.mock_categorical(300, 3, rng=1)
    res = binference.infer(mock.likelihoods, method="gibbs", rng=1, n_samples=200, check=False)
    assert "class_0" in repr(res.summary())


def test_model_none_has_bins():
    res = PopulationResult(None, np.full((10, 3), 1 / 3), method="x")
    assert res.bins.K == 3
    assert isinstance(PopulationModel(np.ones((2, 3))).prior_spec, binference.priors.Dirichlet)
