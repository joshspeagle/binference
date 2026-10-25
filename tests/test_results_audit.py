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
    lk, res = _fit_with_selection()
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
    with pytest.raises(binference.DiagnosticsError):
        _external(stuck, m).check(strict=True)


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
