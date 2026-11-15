"""Tests for PopulationResult and PointEstimate (spec 7; R-OUT-*)."""

import pickle
import warnings

import numpy as np
import pytest

import binference
from binference import (
    Bins,
    ConvergenceWarning,
    Likelihoods,
    PopulationModel,
    PopulationResult,
    Selection,
)
from binference import priors as P
from binference.engines import Gibbs

from .conftest import fit_with_selection, ignore_few_recoveries, selection_mock, small_mock


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
    assert "rhat" in res.diagnostics  # cached first: a cache that cannot be pickled fails
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


def test_the_ess_message_never_rounds_the_per_chain_value_up_to_the_limit():
    with warnings.catch_warnings():  # a short run; only the message's format is tested
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = binference.infer(
            small_mock(n=100, K=3).likelihoods, method="gibbs", n_samples=20, rng=0
        )
    for ess, shown in ((399.0, "399 (99.7"), (399.9, "399 (99.9")):  # 99.75, 99.975 per chain
        res._cache["diagnostics"] = dict(res.diagnostics, ess_tail=np.full(3, ess))
        msgs = [m for m in res._mixing_problems() if "tail ESS" in m]
        assert msgs and f"= {shown} per chain < 100)" in msgs[0]


def test_a_low_e_bfmi_for_a_gp_on_a_grid_suggests_a_smaller_basis(monkeypatch):
    from binference import results

    bins = Bins.grid(np.linspace(0, 1, 4), np.linspace(0, 1, 4))
    lk = binference.simulate.mock_2d(200, bins, rng=0).likelihoods
    res = binference.infer(
        lk,
        P.LogisticGP(),
        method="nuts",
        n_chains=2,
        n_warmup=30,
        n_samples=30,
        rng=0,
        check=False,
    )
    monkeypatch.setattr(results, "EBFMI_MIN", 100.0)  # any E-BFMI is low
    with pytest.warns(binference.ConvergenceWarning, match="n_basis=10"):
        res.check()


def test_attaching_other_likelihoods_redraws_the_total_and_missing_information():
    # both were kept from the previous likelihoods (W objects, their information)
    bins = Bins.linear(0, 1, 4)
    mock = binference.simulate.mock_1d(200, bins, noise=0.1, rng=0)
    res = Gibbs(binference.PopulationModel(mock.likelihoods), n_warmup=20, rng=0).run(20)
    total, info = res.total.copy(), res._missing_information()
    half = binference.PopulationModel(mock.likelihoods[:100])
    res.attach(half)
    fresh = PopulationResult(half, res.component_draws, method="gibbs")
    np.testing.assert_array_equal(res.total, fresh.total)
    assert res.total.mean() < 0.7 * total.mean()  # about W = 100 objects, not 200
    assert res._missing_information() == fresh._missing_information() != info


def test_likelihoods_on_other_bins_are_refused(tmp_path):
    lk, res = fit_with_selection()
    log_bins = Bins.log(1, 100, 4)
    elsewhere = Likelihoods(lk.toarray(), log_bins)
    res.save(str(tmp_path / "fit.npz"))
    with pytest.raises(ValueError, match="other bins"):
        PopulationResult.load(str(tmp_path / "fit.npz")).attach(elsewhere)
    for method in (res.predict_proba, res.log_predictive):
        with pytest.raises(ValueError, match="same bins"):
            method(elsewhere)


def _external(draws, model, **info):
    return PopulationResult.from_samples(model, draws, info=info)


def test_check_flags_energy_tail_mixing_one_chain_and_unconverged_fits():
    _, res = fit_with_selection()
    m = res.model
    rng = np.random.default_rng(0)
    good = rng.dirichlet(np.ones(4) * 50, size=(4, 400))
    assert _external(good, m).check() == []
    with pytest.warns(binference.ConvergenceWarning, match="E-BFMI"):
        _external(good, m, ebfmi=np.array([0.1, 0.9, 0.9, 0.9])).check()
    with pytest.warns(binference.ConvergenceWarning, match="did not converge"):
        _external(good, m, converged=False).check()
    # one chain with a trend: split R-hat catches it (audit O-7)
    trend = good[:1].copy()
    trend[0, :, 0] += np.linspace(0, 0.2, 400)
    trend /= trend.sum(axis=-1, keepdims=True)
    with pytest.warns(binference.ConvergenceWarning, match="R-hat"):
        _external(trend, m).check()
    # external draws get the ESS check, tail ESS included
    stuck = np.repeat(good[:, :40], 10, axis=1)
    with pytest.warns(binference.ConvergenceWarning, match="tail ESS"):
        _external(stuck, m).check()


def test_check_flags_chains_too_short_or_constant_to_diagnose():
    # chains below MIXING_MIN_DRAWS get no R-hat or ESS, and constant ones NaN: check()
    # passed both silently
    model = binference.PopulationModel(Likelihoods(np.ones((5, 4)), Bins.linear(0, 1, 4)))
    short = np.random.default_rng(0).dirichlet(np.ones(4), size=(4, 7))
    constant = np.full((4, 500, 4), 0.25)
    for draws in (short, constant):
        with pytest.warns(binference.ConvergenceWarning, match="R-hat and ESS are undefined"):
            assert len(_external(draws, model).check()) == 1
    assert PopulationResult(model, short, method="bootstrap").check() == []  # independent


@ignore_few_recoveries
def test_check_reads_the_tail_ess_of_an_inferred_selection_and_keeps_its_own_draws():
    # the selection's tail ESS was never computed, so its check could not fire
    bins = Bins.linear(0, 1, 4)
    z = np.repeat(bins.centers, 20)
    sel = Selection.from_injections(z, np.arange(z.size) % 2 == 0, bins, "on_truth")
    model = binference.PopulationModel(Likelihoods(np.ones((5, 4)), bins), selection=sel)
    rng = np.random.default_rng(0)
    rho = rng.dirichlet(np.ones(4) * 50, size=(4, 400))
    S = np.repeat(rng.uniform(0.3, 0.7, size=(4, 40, 4)), 10, axis=1)  # stuck
    res = _external(rho, model, selection_draws=S)
    with pytest.warns(binference.ConvergenceWarning, match="tail ESS of the selection"):
        res.check()
    constant = _external(rho, model, selection_draws=np.full((4, 400, 4), 0.5))
    with pytest.warns(binference.ConvergenceWarning, match="ESS of the selection are undefined"):
        constant.check()
    # the selection's draws are copied with the population's
    before = res.assignment_probabilities(n_draws=None)
    S[:] = 0.5
    np.testing.assert_array_equal(res.assignment_probabilities(n_draws=None), before)


def test_a_strict_check_raises_a_diagnostics_error():
    _, res = fit_with_selection()
    stuck = np.repeat(np.random.default_rng(0).dirichlet(np.ones(4) * 50, (4, 40)), 10, axis=1)
    with pytest.raises(binference.DiagnosticsError):
        _external(stuck, res.model).check(strict=True)


def test_summary_mixing_columns_describe_the_quantity_summarised():
    # fractions mix well while the rates' total does not (audit O-8)
    rng = np.random.default_rng(1)
    rho = rng.dirichlet(np.ones(3) * 50, size=(2, 300))
    total = np.repeat(rng.gamma(50.0, 2.0, size=(2, 30)), 10, axis=1)[..., None]
    model = binference.PopulationModel(
        Likelihoods(np.ones((5, 3)), Bins.linear(0, 1, 3)), binference.priors.GammaRates(1.0, 0.01)
    )
    res = PopulationResult.from_samples(model, rho * total)
    assert res.summary(which="fraction")["ess_bulk"].min() > 300
    assert res.summary(which="rates")["ess_bulk"].max() < 150
    with pytest.raises(ValueError, match="which"):
        res.summary(which="bogus")


def test_summary_and_interval_refuse_a_level_outside_zero_and_one():
    # summary passed level to np.quantile: 1.5 raised numpy's error, 0 gave one column
    res = PopulationResult(None, np.random.default_rng(0).dirichlet(np.ones(3), 40), method="x")
    for level in (0.0, 1.0, 1.5):
        for method in (res.summary, res.interval):
            with pytest.raises(ValueError, match="level must be in"):
                method(level=level)


def test_external_draws_are_checked_and_normalised():
    _, res = fit_with_selection()
    m = res.model
    with pytest.raises(ValueError, match="shape"):
        PopulationResult.from_samples(m, np.full((10, 3), 1 / 3))
    with pytest.raises(ValueError, match="sum to one"):
        PopulationResult.from_samples(m, np.full((10, 4), 2.0))
    with pytest.raises(ValueError, match="non-negative"):
        PopulationResult.from_samples(m, np.full((10, 4), -0.25))
    near = np.full((10, 4), 0.25) * (1 + 1e-9)
    np.testing.assert_allclose(PopulationResult.from_samples(m, near).samples.sum(1), 1.0)


def test_which_takes_both_spellings_everywhere():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    _, res = fit_with_selection()
    np.testing.assert_allclose(
        res.rebin([0, 0, 1, 1], which="rate"), res.rebin([[0, 1], [2, 3]], "rates")
    )
    assert res.cov("rates").shape == (4, 4)
    np.testing.assert_allclose(res.functional(np.sum, which="rates"), res.total)
    res.plot(which="fractions")
    res.plot_corner(which="rates")
    res.plot_trace(which="rate")
    res.plot_individual(0)
    est = binference.infer(selection_mock(), method="map")
    ax = est.plot(which="rates")
    np.testing.assert_allclose(ax.lines[0].get_ydata()[:-1], est.rates)
    with pytest.raises(ValueError, match="which"):
        est.plot(which="bogus")
    mpl.pyplot.close("all")


def test_rebin_takes_integral_floats_and_refuses_others():
    _, res = fit_with_selection()
    np.testing.assert_allclose(res.rebin(np.array([0.0, 0.0, 1.0, 1.0])), res.rebin([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="integer group"):
        res.rebin(np.array([0.0, 0.5, 1.0, 1.0]))


def test_independent_draws_get_no_chain_diagnostics():
    _, res = fit_with_selection()
    trend = np.random.default_rng(0).dirichlet(np.ones(4) * 50, size=(1, 200))
    trend[0, :, 0] += np.linspace(0, 0.2, 200)
    trend /= trend.sum(axis=-1, keepdims=True)
    assert PopulationResult(res.model, trend, method="bootstrap").check() == []


def test_attach_refuses_a_selection_the_fit_did_not_have():
    lk = selection_mock()
    res = binference.infer(lk, method="gibbs", n_samples=20, rng=0, check=False)
    with pytest.raises(ValueError, match="another selection"):
        res.attach(lk.with_selection(np.full(4, 0.5), "on_truth"))


def test_responsibilities_survive_a_subnormal_population_draw():
    from binference.results import _mean_responsibilities

    L = np.array([[1.0, 0.0], [0.5, 0.5]])
    R = np.array([[1e-310, 1.0], [0.5, 0.5]])
    out = _mean_responsibilities(L, R, chunk=10)
    np.testing.assert_allclose(out, [[1.0, 0.0], [0.5 * 1e-310 / (1 + 1e-310) + 0.25, 0.75]])


def test_densities_do_not_rebin():
    _, res = fit_with_selection()
    with pytest.raises(ValueError, match="Densities"):
        res.rebin([0, 0, 1, 1], which="density")


def _nuts_result(prior, weights=None, **info):
    # well-mixed synthetic NUTS draws with injected engine statistics
    L = np.random.default_rng(1).random((30, 4)) + 0.1
    lk = Likelihoods(L, Bins.linear(0, 1, 4), weights=weights, fractional_weights=True)
    model = binference.PopulationModel(lk, prior)
    draws = np.random.default_rng(0).dirichlet(np.ones(4) * 50, size=(4, 400))
    return PopulationResult(model, draws, method="nuts", info=info)


def test_check_reports_divergences_with_their_share_and_a_target_accept_hint():
    res = _nuts_result(None, n_divergent=16)
    with pytest.warns(binference.ConvergenceWarning) as rec:
        msgs = res.check()
    assert msgs == ["16 divergent transitions (1% of draws)"]  # 16 of 4 x 400
    assert "Hint: try a higher target_accept" in str(rec[0].message)
    assert _nuts_result(None, n_divergent=0).check() == []


def test_check_compares_pareto_k_hat_with_the_threshold_the_engine_reported():
    with pytest.warns(binference.ConvergenceWarning, match=r"k-hat = 0\.60 >= 0\.55"):
        _nuts_result(None, k_hat=0.6, k_hat_threshold=0.55).check()
    assert _nuts_result(None, k_hat=0.6).check() == []  # the default threshold is 0.7
    with pytest.warns(binference.ConvergenceWarning, match=r"k-hat = 0\.70 >= 0\.70"):
        _nuts_result(None, k_hat=0.7).check()


@pytest.mark.parametrize(
    ("prior", "hint"),
    [
        (binference.priors.Dirichlet(0.5), True),
        (binference.priors.PolyaTree(c=0.1), True),
        (binference.priors.Dirichlet(1.0), False),
        # unbounded on the simplex, but not in the coordinates NUTS samples
        (binference.priors.PolyaTree(), False),
        # the same, with branches at exactly one that a sum of coefficients rounds below it
        (binference.priors.GeneralizedDirichlet([2, 2, 1], [1, 2.9, 1.2]), False),
    ],
    ids=repr,
)
def test_saturated_trees_suggest_gibbs_only_for_priors_with_long_tails_where_nuts_samples(
    prior, hint
):
    with pytest.warns(binference.ConvergenceWarning) as rec:
        msgs = _nuts_result(prior, max_depth_fraction=0.2).check()
    assert msgs == ["20% of NUTS iterations hit the maximum tree depth"]
    assert ("method='gibbs'" in str(rec[0].message)) is hint


def test_saturated_trees_suggest_gibbs_only_where_gibbs_can_run():
    with pytest.warns(binference.InputWarning, match="pseudo-likelihood"):
        res = _nuts_result(
            binference.priors.Dirichlet(0.5), np.full(30, 1.5), max_depth_fraction=0.2
        )
    with pytest.warns(binference.ConvergenceWarning) as rec:
        res.check()
    assert "method='gibbs'" not in str(rec[0].message)  # Gibbs needs integer weights


def test_a_few_saturated_trees_are_tolerated():
    assert _nuts_result(None, max_depth_fraction=0.05).check() == []  # 5% is tolerated


def test_the_log_predictive_of_a_rate_model_under_a_selection_matches_the_direct_formula():
    mock = binference.simulate.mock_1d(400, Bins.linear(0, 3, 4), rng=5)
    S = np.array([1.0, 0.6, 0.3, 0.1])
    model = PopulationModel(mock.likelihoods, P.GammaRates(1.0, 1e-3),
                            selection=Selection(S, "on_truth"))  # fmt: skip
    lam = np.random.default_rng(0).gamma(5.0, 20.0, size=(50, 4))
    res = PopulationResult(model, lam, method="test", kind="rates")
    new = mock.likelihoods[:5]
    Lt = new.toarray() * S
    expected = np.log(np.mean((Lt @ lam.T) / (lam @ S), axis=1)) + new.log_scale
    np.testing.assert_allclose(res.log_predictive(new, n_draws=None), expected, rtol=1e-10)


def test_rate_diagnostics_use_rates():
    mock = binference.simulate.mock_1d(300, Bins.linear(0, 3, 4), rng=6)
    res = binference.infer(mock.likelihoods, P.GammaRates(1.0, 1e-3), method="gibbs", rng=1,
                       n_samples=200, check=False)  # fmt: skip
    r = res.rates.reshape(res.n_chains, -1, 4)
    expected = binference.diagnostics.ess(r[:, :, 0], "bulk")
    assert res.diagnostics["ess_bulk"][0] == pytest.approx(expected)


def test_check_reports_missing_information_when_gibbs_mixes_slowly():
    mock = binference.simulate.mock_1d(500, Bins.linear(0, 3, 15), noise=(0.4, 0.6), rng=1)
    res = binference.infer(
        mock.likelihoods, method="gibbs", n_chains=2, n_warmup=20, n_samples=60, rng=0, check=False
    )
    with pytest.warns(binference.ConvergenceWarning, match=r"missing information [01]\.\d\d"):
        res.check()
    assert 0.5 < res.diagnostics["missing_information"] <= 1.0
