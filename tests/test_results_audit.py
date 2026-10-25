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
