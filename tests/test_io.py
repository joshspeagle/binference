"""Saving, loading and exporting results: `PopulationResult.save` and `load`, `to_datatree`."""

import numpy as np
import pytest

import binference
from binference import (
    Bins,
    Likelihoods,
    PopulationModel,
    PopulationResult,
    Selection,
)
from binference import priors as P
from binference.engines import Gibbs
from binference.priors import Dirichlet, PolyaTree

from .conftest import fit_with_selection, selection_mock


def test_save_load_round_trip(res1d, small_1d, tmp_path):
    path = tmp_path / "res.npz"
    res1d.save(path)
    back = PopulationResult.load(path)
    np.testing.assert_array_equal(back.chains, res1d.chains)
    assert back.method == res1d.method and back.kind == res1d.kind
    assert back.bins == res1d.bins
    np.testing.assert_array_equal(back.labels, res1d.labels)
    assert back.prior_repr == res1d.prior_repr
    assert back.model is None
    back.attach(small_1d.likelihoods)
    assert back.assignment_probabilities(idx=[0], n_draws=5).shape == (1, 12)
    d = res1d.to_dict()
    assert d["meta"]["method"] == "gibbs" and d["meta"]["bins"] == "grid"


def test_save_load_categories_and_rates(tmp_path):
    mc = binference.simulate.mock_categorical(100, 3, rng=1)
    r = binference.infer(
        mc.likelihoods,
        binference.priors.GammaRates(1.0, 1e-2),
        method="gibbs",
        rng=1,
        n_samples=30,
        check=False,
    )
    path = tmp_path / "cat.npz"
    r.save(path)
    back = PopulationResult.load(path)
    assert back.bins.is_categorical and back.bins.labels == ("class_0", "class_1", "class_2")
    np.testing.assert_allclose(back.rates, r.rates)


def test_boolean_category_labels_reload_as_booleans():
    from binference.io import _bins_from_parts, _bins_to_parts

    bins = Bins.categories([False, True])  # as Response.from_labelled builds for boolean labels
    labels = _bins_from_parts(*_bins_to_parts(bins)).labels
    assert labels == (False, True) and all(type(lab) is bool for lab in labels)


def test_to_datatree(res1d):
    pytest.importorskip("xarray")
    dt = res1d.to_datatree()
    assert "posterior" in dt.children
    assert dt["posterior"]["rho"].shape == (4, 100, 12)
    # rates of a fractions fit are the derived ones, not the fractions (DECISIONS #78)
    np.testing.assert_allclose(dt["posterior"]["rate"].values.reshape(-1, 12), res1d.rates)


def test_a_reloaded_fit_keeps_its_selection_and_total(tmp_path):
    # the selection was lost on save and attach redrew the total without it (audit O-1)
    lk, res = fit_with_selection()
    before = res.assignment_probabilities(n_draws=20), res.log_predictive(lk[:5], n_draws=20)
    total = res.total.copy()
    res.save(str(tmp_path / "fit"))
    back = PopulationResult.load(str(tmp_path / "fit"))  # without the .npz suffix (audit O-10)
    np.testing.assert_allclose(back.total, total)
    back.attach(lk)
    assert back.selection == res.selection
    np.testing.assert_allclose(back.assignment_probabilities(n_draws=20), before[0])
    np.testing.assert_allclose(back.log_predictive(lk[:5], n_draws=20), before[1])
    np.testing.assert_allclose(back.total, total)
    other = lk.with_selection(np.full(4, 0.5), "on_truth")
    with pytest.raises(ValueError, match="another selection"):
        PopulationResult.load(str(tmp_path / "fit")).attach(other)


@pytest.mark.parametrize("kind", ["combined", "injections", "smooth injections"])
@pytest.mark.filterwarnings("ignore::binference.ApproximationWarning")  # Laplace's k-hat
def test_a_reloaded_fit_keeps_a_combined_or_injection_selection(tmp_path, kind):
    # the row of a combined selection and the injections are saved beside the values
    lk = selection_mock()
    bins = lk.bins
    rng = np.random.default_rng(1)
    z = rng.uniform(0, 1, 400)
    detected = rng.random(400) < 1 - 0.5 * z
    if kind == "combined":
        sel = Selection.combined([1.0, 0.8, 0.6, 0.4], [0.9, 0.9, 0.8, 0.7])
    else:
        smooth = 0.5 if kind == "smooth injections" else None
        sel = Selection.from_injections(z, detected, bins, "on_truth", smooth=smooth)
    method = "laplace" if kind == "smooth injections" else "gibbs"  # Gibbs: independent Betas
    res = binference.infer(lk, selection=sel, method=method, n_samples=50, rng=0, check=False)
    res.save(str(tmp_path / "fit"))
    back = PopulationResult.load(str(tmp_path / "fit"))
    assert back.selection == res.selection  # values, row and injections


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


def test_the_datatree_uses_arviz_names_numeric_centres_and_hyperparameters():
    pytest.importorskip("xarray")
    bins = Bins.linear(0, 1, 6)
    lk = binference.simulate.mock_1d(100, bins, noise=0.1, rng=0).likelihoods
    res = binference.infer(
        lk,
        binference.priors.LogisticRandomWalk(),
        method="nuts",
        n_samples=20,
        n_warmup=30,
        rng=0,
        check=False,
    )
    tree = res.to_datatree()
    post, stats = tree["posterior"], tree["sample_stats"]
    np.testing.assert_allclose(post["bin_center"].values, bins.centers)
    assert post.attrs["kind"] == "fractions" and "binference_version" in post.attrs
    assert set(res.info["hyperparameters"]) <= set(post.data_vars)
    assert stats["diverging"].dtype == bool
    assert "acceptance_rate" in stats.data_vars and "n_steps" in stats.data_vars


def test_load_takes_paths_and_file_objects(tmp_path):
    import io
    import pathlib

    _, res = fit_with_selection()
    res.save(pathlib.Path(tmp_path / "p"))
    assert PopulationResult.load(pathlib.Path(tmp_path / "p")).n_draws == res.n_draws
    buf = io.BytesIO((tmp_path / "p.npz").read_bytes())
    assert PopulationResult.load(buf).n_draws == res.n_draws


def test_a_reloaded_fit_keeps_its_draws_and_needs_its_own_prior_for_the_best_fit(tmp_path):
    lk = binference.simulate.mock_1d(200, rng=3).likelihoods
    prior = binference.priors.LogisticRandomWalk(scale=1.0)
    res = binference.infer(lk, prior, method="laplace", rng=0)
    res.save(str(tmp_path / "r.npz"))
    back = binference.PopulationResult.load(str(tmp_path / "r.npz"))
    with pytest.raises(ValueError, match="needs the fit's prior"):
        _ = back.attach(binference.PopulationModel(lk)).best  # the default prior
    back.attach(binference.PopulationModel(lk, prior))
    np.testing.assert_allclose(back.best.rho, res.best.rho)
    # prior_information_ratio evaluates at the mean unconstrained draw, before and after
    np.testing.assert_array_equal(back.x, res.x)
    assert back.prior_information_ratio() == res.prior_information_ratio()


@pytest.mark.parametrize(
    ("fit", "other"),
    [
        (Dirichlet(np.full(10, 0.5)), Dirichlet(np.full(10, 3.0))),
        # base measures of the same length (a Density's repr gives only their number)
        (PolyaTree(base=np.linspace(1, 2, 10)), PolyaTree(base=np.linspace(2, 1, 10))),
        (
            Dirichlet.from_base_measure(2.0, np.linspace(1, 2, 10)),
            Dirichlet.from_base_measure(2.0, np.linspace(2, 1, 10)),
        ),
    ],
    ids=["alpha", "polya-tree-base", "dirichlet-base"],
)
def test_a_prior_that_differs_only_in_the_values_of_an_array_is_not_the_fits(tmp_path, fit, other):
    # both reprs shorten the array, so they are equal; the full arguments differ
    lk = binference.simulate.mock_1d(200, Bins.linear(0, 1, 10), rng=3).likelihoods
    res = Gibbs(binference.PopulationModel(lk, fit), n_warmup=10, rng=0).run(10)
    res.save(str(tmp_path / "r.npz"))
    back = PopulationResult.load(str(tmp_path / "r.npz"))
    other = binference.PopulationModel(lk, other)
    assert repr(other.prior_spec) == back.prior_repr
    with pytest.raises(ValueError, match="needs the fit's prior"):
        _ = back.attach(other).best
    assert back.attach(binference.PopulationModel(lk, fit)).best is not None


def test_a_saved_result_keeps_integer_labels_and_hyperparameter_draws(tmp_path):
    lk = Likelihoods(np.random.default_rng(0).random((50, 3)) + 0.1)
    res = PopulationResult(PopulationModel(lk), np.full((10, 3), 1 / 3), method="x",
                           labels=np.zeros((10, 50), int))  # fmt: skip
    res.save(tmp_path / "a.npz")
    back = PopulationResult.load(tmp_path / "a.npz")
    assert back.bins == lk.bins  # integer category labels survive
    mock = binference.simulate.mock_1d(200, Bins.linear(0, 3, 6), rng=7)
    gp = binference.infer(mock.likelihoods, P.LogisticGP(), method="nuts", rng=1, n_samples=30,
                      n_warmup=60, n_chains=2, check=False)  # fmt: skip
    gp.save(tmp_path / "gp.npz")
    back = PopulationResult.load(tmp_path / "gp.npz")
    for name, draws in gp.info["hyperparameters"].items():
        np.testing.assert_array_equal(back.info["hyperparameters"][name], draws)


def test_a_reloaded_fit_on_a_transform_axis_attaches_without_a_basis(tmp_path):
    # the axis loads as a table, which compares unequal to the transform
    tb = Bins(np.linspace(0, 3, 7), measure=binference.Measure.transform(np.arcsinh, np.sinh))
    lk = binference.simulate.mock_1d(80, tb, rng=7).likelihoods
    res = binference.infer(lk, method="gibbs", n_samples=20, n_warmup=10, rng=0, check=False)
    res.save(str(tmp_path / "t"))
    PopulationResult.load(str(tmp_path / "t")).attach(lk).assignment_probabilities(idx=[0])


def test_engine_statistics_that_cannot_be_saved_are_named():
    _, res = fit_with_selection()
    res.info["metrics"] = [np.eye(2), np.eye(2)]
    assert res.to_dict()["meta"]["info_dropped"] == ["metrics"]
