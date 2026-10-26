"""Results, io and plotting fixes from the outputs audit (O-1–O-21, C12a)."""

import numpy as np
import pytest

import binference
from binference import Bins, Likelihoods, PopulationResult
from binference.samplers import Gibbs


def _fit_with_selection():
    bins = Bins.linear(0, 1, 4)
    mock = binference.simulate.mock_1d(200, bins, noise=0.1, rng=0)
    lk = mock.likelihoods.with_selection(None)
    S = np.array([1.0, 0.7, 0.4, 0.2])
    res = binference.infer(
        lk,
        method="gibbs",
        selection=S,
        selection_mode="on_truth",
        n_samples=50,
        rng=0,
        check=False,
    )
    return lk, res


def test_a_reloaded_fit_keeps_its_selection_and_total(tmp_path):
    # the selection was lost on save and attach redrew the total without it (O-1)
    lk, res = _fit_with_selection()
    before = res.assignment_probabilities(n_draws=20), res.log_predictive(lk[:5], n_draws=20)
    total = res.total.copy()
    res.save(str(tmp_path / "fit"))
    back = PopulationResult.load(str(tmp_path / "fit"))  # without the .npz suffix (O-10)
    np.testing.assert_allclose(back.total, total)
    back.attach(lk)
    assert back.selection == res.selection
    np.testing.assert_allclose(back.assignment_probabilities(n_draws=20), before[0])
    np.testing.assert_allclose(back.log_predictive(lk[:5], n_draws=20), before[1])
    np.testing.assert_allclose(back.total, total)
    other = lk.with_selection(np.full(4, 0.5), "on_truth")
    with pytest.raises(ValueError, match="another selection"):
        PopulationResult.load(str(tmp_path / "fit")).attach(other)


def test_likelihoods_on_other_bins_are_refused(tmp_path):
    lk, res = _fit_with_selection()
    log_bins = Bins.log(1, 100, 4)
    elsewhere = Likelihoods(lk.toarray(), log_bins)
    res.save(str(tmp_path / "fit.npz"))
    with pytest.raises(ValueError, match="other bins"):
        PopulationResult.load(str(tmp_path / "fit.npz")).attach(elsewhere)
    for method in (res.predict_proba, res.log_predictive):
        with pytest.raises(ValueError, match="same bins"):
            method(elsewhere)


def test_bin_labels_and_label_draws_are_kept_apart_and_round_trip(tmp_path):
    for labels in (["x", "y", "z"], [0.5, 1.5, 2.5], [1, "b", 3]):
        bins = Bins.categories(labels)
        L = np.random.default_rng(0).random((10, 3))
        res = Gibbs(binference.PopulationModel(Likelihoods(L, bins)), rng=0, store_labels=True)
        res = res.run(10)
        d = res.to_dict()
        assert d["meta"]["labels"] == labels and d["label_draws"].shape == (4, 10, 10)
        res.save(str(tmp_path / "c.npz"))
        back = PopulationResult.load(str(tmp_path / "c.npz"))
        assert back.bins == bins and np.array_equal(back.labels, res.labels)


def test_engine_statistics_that_cannot_be_saved_are_named():
    _, res = _fit_with_selection()
    res.info["metrics"] = [np.eye(2), np.eye(2)]
    assert res.to_dict()["meta"]["info_dropped"] == ["metrics"]


def _external(draws, model, **info):
    return PopulationResult.from_samples(model, draws, info=info)


def test_check_flags_energy_tail_mixing_one_chain_and_unconverged_fits():
    _, res = _fit_with_selection()
    m = res.model
    rng = np.random.default_rng(0)
    good = rng.dirichlet(np.ones(4) * 50, size=(4, 400))
    assert _external(good, m).check() == []
    with pytest.warns(binference.ConvergenceWarning, match="E-BFMI"):
        _external(good, m, ebfmi=np.array([0.1, 0.9, 0.9, 0.9])).check()
    with pytest.warns(binference.ConvergenceWarning, match="did not converge"):
        _external(good, m, converged=False).check()
    # one chain with a trend: split R-hat catches it (O-7)
    trend = good[:1].copy()
    trend[0, :, 0] += np.linspace(0, 0.2, 400)
    trend /= trend.sum(axis=-1, keepdims=True)
    with pytest.warns(binference.ConvergenceWarning, match="R-hat"):
        _external(trend, m).check()
    # external draws get the ESS check, tail ESS included
    stuck = np.repeat(good[:, :40], 10, axis=1)
    with pytest.warns(binference.ConvergenceWarning, match="tail ESS"):
        _external(stuck, m).check()


def test_a_strict_check_raises_a_diagnostics_error():
    _, res = _fit_with_selection()
    stuck = np.repeat(np.random.default_rng(0).dirichlet(np.ones(4) * 50, (4, 40)), 10, axis=1)
    with pytest.raises(binference.DiagnosticsError):
        _external(stuck, res.model).check(strict=True)


def test_summary_mixing_columns_describe_the_quantity_summarised():
    # fractions mix well while the rates' total does not (O-8)
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


def test_psis_rounds_the_tail_length_up():
    # 3 sqrt(1000) = 94.9: the tail is the 95 largest weights, as in ArviZ and loo (O-20)
    from binference.diagnostics import _gpdfit, psis

    lw = np.random.default_rng(2).standard_t(3, 1000)
    lw -= lw.max()
    s = np.sort(lw)
    cut = np.exp(s[-96])
    k_ref, _ = _gpdfit(np.exp(s[-95:]) - cut)
    assert psis(lw)[1] == pytest.approx(k_ref)


def test_external_draws_are_checked_and_normalised():
    _, res = _fit_with_selection()
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
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    _, res = _fit_with_selection()
    np.testing.assert_allclose(
        res.rebin([0, 0, 1, 1], which="rate"), res.rebin([[0, 1], [2, 3]], "rates")
    )
    assert res.cov("rates").shape == (4, 4)
    np.testing.assert_allclose(res.functional(np.sum, which="rates"), res.total)
    res.plot(which="fractions")
    res.plot_corner(which="rates")
    res.plot_trace(which="rate")
    res.plot_individual(0)
    est = binference.infer(_fit_with_selection()[0], method="map")
    ax = est.plot(which="rates")
    np.testing.assert_allclose(ax.lines[0].get_ydata()[:-1], est.rates)
    with pytest.raises(ValueError, match="which"):
        est.plot(which="bogus")


def test_rebin_takes_integral_floats_and_refuses_others():
    _, res = _fit_with_selection()
    np.testing.assert_allclose(res.rebin(np.array([0.0, 0.0, 1.0, 1.0])), res.rebin([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="integer group"):
        res.rebin(np.array([0.0, 0.5, 1.0, 1.0]))


def test_plots_refuse_three_axes_and_warn_when_two_drop_overlays():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    grid3 = Bins.grid([0, 1, 2], [0, 1, 2], [0, 1, 2])
    cube = PopulationResult(None, np.full((5, 8), 1 / 8), method="external", bins=grid3)
    with pytest.raises(ValueError, match="1-D and 2-D"):
        cube.plot()
    grid2 = Bins.grid([0, 1, 2], [0, 1, 2])
    square = PopulationResult(None, np.full((5, 4), 1 / 4), method="external", bins=grid2)
    with pytest.warns(binference.InputWarning, match="2-D plots"):
        square.plot(truth=np.full(4, 0.25))


def test_plotting_and_io_are_reachable_from_the_package():
    import subprocess
    import sys

    code = "import binference; print(binference.plotting.__name__, binference.io.__name__)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.split() == ["binference.plotting", "binference.io"], out.stderr


def test_method_em_runs_em_and_auto_is_recorded_in_any_case():
    lk, _ = _fit_with_selection()
    assert binference.infer(lk, method="em").info["algorithm"] in ("em", "squarem")
    res = binference.infer(lk, method="AUTO", n_samples=20, rng=0, check=False)
    assert res.info["auto_method"] == res.method


def test_estimator_works_with_scikit_learn_model_selection():
    model_selection = pytest.importorskip("sklearn.model_selection")
    bins = Bins.linear(0, 1, 4)
    lk = binference.simulate.mock_1d(120, bins, noise=0.1, rng=0).likelihoods
    est = binference.PopulationEstimator(method="map", bins=bins)
    assert np.isfinite(model_selection.cross_val_score(est, lk, cv=3)).all()
    grid = {"prior": [binference.priors.Dirichlet(0.5), binference.priors.Dirichlet(1.0)]}
    assert model_selection.GridSearchCV(est, grid, cv=3).fit(lk.toarray()).best_params_


def test_loglike_values_need_support_in_the_bin_and_call_loglike_once_per_object():
    bins = Bins.linear(0, 1, 4)
    mock = binference.simulate.mock_1d(30, bins, noise=0.3, rng=0)
    lk = Likelihoods(mock.likelihoods.toarray(), bins)
    res = binference.infer(lk, method="gibbs", n_samples=20, rng=0, check=False)
    calls = []

    def loglike(z, i):
        calls.append(i)
        return -0.5 * ((z - mock.measurements["mean"][i]) / 0.3) ** 2

    res.sample_values(idx=[0, 1, 2], n_draws=20, rng=0, loglike=loglike)
    assert sorted(calls) == [0, 1, 2]

    def narrow(z, i):
        return np.where(np.abs(z - 0.08) < 1e-6, 0.0, -np.inf)

    with pytest.raises(ValueError, match="sub-points"):
        res.sample_values(idx=[0], n_draws=20, rng=0, loglike=narrow)


def test_the_datatree_uses_arviz_names_numeric_centres_and_hyperparameters():
    pytest.importorskip("xarray")
    bins = Bins.linear(0, 1, 6)
    lk = binference.simulate.mock_1d(100, bins, noise=0.1, rng=0).likelihoods
    res = binference.infer(
        lk, binference.priors.LogisticRandomWalk(), n_samples=20, n_warmup=30, rng=0, check=False
    )
    tree = res.to_datatree()
    post, stats = tree["posterior"], tree["sample_stats"]
    np.testing.assert_allclose(post["bin_center"].values, bins.centers)
    assert post.attrs["kind"] == "fractions" and "binference_version" in post.attrs
    assert set(res.info["hyperparameters"]) <= set(post.data_vars)
    assert stats["diverging"].dtype == bool
    assert "acceptance_rate" in stats.data_vars and "n_steps" in stats.data_vars


def test_a_reloaded_fit_on_a_transform_axis_attaches_without_a_basis(tmp_path):
    # the axis loads as a table, which compares unequal to the transform
    tb = Bins(np.linspace(0, 3, 7), measure=binference.Measure.transform(np.arcsinh, np.sinh))
    lk = binference.simulate.mock_1d(80, tb, rng=7).likelihoods
    res = binference.infer(lk, method="gibbs", n_samples=20, n_warmup=10, rng=0, check=False)
    res.save(str(tmp_path / "t"))
    PopulationResult.load(str(tmp_path / "t")).attach(lk).assignment_probabilities(idx=[0])


def test_independent_draws_get_no_chain_diagnostics():
    _, res = _fit_with_selection()
    trend = np.random.default_rng(0).dirichlet(np.ones(4) * 50, size=(1, 200))
    trend[0, :, 0] += np.linspace(0, 0.2, 200)
    trend /= trend.sum(axis=-1, keepdims=True)
    assert PopulationResult(res.model, trend, method="bootstrap").check() == []


def test_load_takes_paths_and_file_objects(tmp_path):
    import io
    import pathlib

    _, res = _fit_with_selection()
    res.save(pathlib.Path(tmp_path / "p"))
    assert PopulationResult.load(pathlib.Path(tmp_path / "p")).n_draws == res.n_draws
    buf = io.BytesIO((tmp_path / "p.npz").read_bytes())
    assert PopulationResult.load(buf).n_draws == res.n_draws


def test_method_em_on_a_joint_model_runs_its_optimiser():
    lk, _ = _fit_with_selection()
    z = np.linspace(0.05, 0.95, 40)
    sel = binference.Selection.from_injections(z, np.tile([True, False], 20), lk.bins, "on_truth")
    assert binference.infer(lk, selection=sel, method="em").info["space"] == "unconstrained"


def test_attach_refuses_a_selection_the_fit_did_not_have(tmp_path):
    lk, _ = _fit_with_selection()
    res = binference.infer(lk, method="gibbs", n_samples=20, rng=0, check=False)
    with pytest.raises(ValueError, match="another selection"):
        res.attach(lk.with_selection(np.full(4, 0.5), "on_truth"))


def test_a_star_import_leaves_the_standard_io_module_alone():
    ns: dict = {}
    exec("import io\nfrom binference import *", ns)
    assert hasattr(ns["io"], "BytesIO")


def test_responsibilities_survive_a_subnormal_population_draw():
    from binference.results import _mean_responsibilities

    L = np.array([[1.0, 0.0], [0.5, 0.5]])
    R = np.array([[1e-310, 1.0], [0.5, 0.5]])
    out = _mean_responsibilities(L, R, chunk=10)
    np.testing.assert_allclose(out, [[1.0, 0.0], [0.5 * 1e-310 / (1 + 1e-310) + 0.25, 0.75]])


def test_densities_do_not_rebin():
    _, res = _fit_with_selection()
    with pytest.raises(ValueError, match="Densities"):
        res.rebin([0, 0, 1, 1], which="density")


def _nuts_result(prior, **info):
    # well-mixed synthetic NUTS draws with injected engine statistics
    L = np.random.default_rng(1).random((30, 4)) + 0.1
    model = binference.PopulationModel(Likelihoods(L, Bins.linear(0, 1, 4)), prior)
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


def test_saturated_trees_suggest_gibbs_only_for_priors_with_unbounded_density():
    with pytest.warns(binference.ConvergenceWarning) as rec:
        msgs = _nuts_result(binference.priors.Dirichlet(0.5), max_depth_fraction=0.2).check()
    assert msgs == ["20% of NUTS iterations hit the maximum tree depth"]
    assert "method='gibbs'" in str(rec[0].message)
    with pytest.warns(binference.ConvergenceWarning) as rec:
        _nuts_result(binference.priors.Dirichlet(1.0), max_depth_fraction=0.2).check()
    assert "Hint" not in str(rec[0].message)
    assert _nuts_result(None, max_depth_fraction=0.05).check() == []  # 5% is tolerated
