"""Smoke tests for plotting and round-trip tests for saving/loading."""

import numpy as np
import pytest

import binference
from binference import Bins, Likelihoods, PopulationModel, PopulationResult, Selection
from binference import priors as P
from binference.engines import Gibbs
from binference.priors import Dirichlet, PolyaTree

from .conftest import fit_with_selection

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(scope="module")
def res1d(small_1d):
    return binference.infer(
        small_1d.likelihoods,
        method="gibbs",
        rng=1,
        n_samples=100,
        n_warmup=20,
        check=False,
        store_labels=[0, 1],
    )


def test_plot_1d_2d_categorical(res1d, small_1d):
    ax = res1d.plot(truth=small_1d.truth.rho, stack=small_1d.likelihoods.stack())
    assert ax.get_ylabel() == "density"
    res1d.plot(which="fraction", step=False)
    binference.infer(small_1d.likelihoods, method="map").plot()
    m2 = binference.simulate.mock_2d(
        200, Bins.grid(np.linspace(0, 1, 6), np.linspace(0, 1, 5)), rng=1
    )
    r2 = binference.infer(m2.likelihoods, method="gibbs", rng=1, n_samples=50, check=False)
    axes = r2.plot(color="#123456")
    assert len(axes) == 2
    from matplotlib.colors import same_color

    assert same_color(axes[0].collections[0].get_cmap()(1.0), "#123456")  # ramp to color
    mc = binference.simulate.mock_categorical(200, 3, rng=1)
    rc = binference.infer(mc.likelihoods, method="gibbs", rng=1, n_samples=50, check=False)
    rc.plot(truth=mc.truth.rho)
    assert res1d.plot(which="rate").get_ylabel() == "rate"  # derived rates (DECISIONS #78)
    plt.close("all")


def test_corner_trace_individual_sbc(res1d):
    fig = res1d.plot_corner()
    assert fig is not None
    res1d.plot_trace(idx=[0, 1])
    from binference.plotting import plot_individual, plot_sbc

    plot_individual(res1d, 0)
    sbc = binference.validation.SBCResult(
        np.random.default_rng(1).integers(0, 20, (30, 3)), 19, ["a", "b", "c"]
    )
    fig = plot_sbc(sbc)
    # the uniformity band spans whole bins, the outer halves of the end bins included
    band = fig.axes[0].patches[0].get_data()
    assert band.edges[0] == -0.5 and band.edges[-1] == len(band.values) - 0.5
    sbc.plot()
    plt.close("all")


def test_plot_individual_shows_the_own_likelihood_per_bin_on_a_basis():
    # flat components at resolution 2 sum back to the bins: the flat-prior posterior per bin
    bins = Bins.linear(0, 3, 6)
    mock = binference.simulate.mock_1d(50, bins, rng=0)
    d, s = mock.measurements["mean"], mock.measurements["sigma"]
    lk = binference.Likelihoods.from_gaussian(d, s, binference.Basis(bins, resolution=2))
    est = binference.PointEstimate(binference.PopulationModel(lk), np.full(12, 1 / 12))
    own = est.plot_individual(1).get_lines()[0].get_ydata()[:-1] * bins.volumes
    plain = mock.likelihoods.toarray()[1] * bins.volumes
    np.testing.assert_allclose(own, plain / plain.sum(), rtol=1e-12)
    plt.close("all")


@pytest.mark.parametrize("shape", [{"order": 1}, {"resolution": 2}])
def test_a_stack_over_a_basis_is_drawn_per_bin(shape):
    # Likelihoods.stack weighs the components: as many as the bins for hats, twice as many
    bins = Bins.linear(0, 3, 6)
    mock = binference.simulate.mock_1d(50, bins, rng=0)
    basis = binference.Basis(bins, **shape)
    d, s = mock.measurements["mean"], mock.measurements["sigma"]
    lk = binference.Likelihoods.from_gaussian(d, s, basis)
    est = binference.PointEstimate(binference.PopulationModel(lk), np.full(basis.C, 1 / basis.C))
    stack = lk.stack()
    drawn = est.plot(which="fractions", stack=stack).get_lines()[-1].get_ydata()[:-1]
    np.testing.assert_allclose(drawn, basis.to_bins(stack), rtol=1e-12)
    with pytest.raises(ValueError, match=rf"stack must have shape \({basis.C},\)"):
        est.plot(stack=stack[:-1])
    with pytest.raises(ValueError, match=r"truth must have shape \(6,\)"):
        est.plot(truth=np.ones(5))
    plt.close("all")


def test_plot_refuses_a_credible_level_outside_zero_and_one(res1d):
    # 1.5 failed in np.quantile naming no argument, and 0 drew an empty band
    for levels in ((1.5,), (0.0, 0.68)):
        with pytest.raises(ValueError, match="levels must be in"):
            res1d.plot(levels=levels)


def test_the_sbc_plot_always_shows_the_joint_log_likelihood():
    names = ["a", "b", "c", "loglike"]
    ranks = np.random.default_rng(1).integers(0, 20, (30, 4))
    sbc = binference.validation.SBCResult(ranks, 19, names)
    shown = [a for a in sbc.plot(max_panels=2).axes if a.get_visible()]
    assert [a.get_title() for a in shown] == ["a", "loglike"]
    with pytest.raises(ValueError, match="max_panels"):
        sbc.plot(max_panels=0)
    np.testing.assert_array_equal(shown[-1].containers[0].datavalues, sbc.histogram()[-1])
    plt.close("all")


def test_plots_take_population_and_sample_colours_from_the_style_cycle(res1d, small_1d):
    from matplotlib.colors import same_color

    from binference.plotting import plot_individual

    cycle = mpl.cycler(color=["#123456", "#abcdef"])
    with plt.rc_context({"axes.prop_cycle": cycle, "text.color": "#654321"}):
        ax = res1d.plot(truth=small_1d.truth.rho, stack=small_1d.likelihoods.stack())
        median, truth, stack = ax.get_lines()
        assert same_color(median.get_color(), "#123456")
        assert same_color(stack.get_color(), "#abcdef")
        assert same_color(truth.get_color(), "#654321") and truth.get_linestyle() == "--"
        own, shrunk = plot_individual(res1d, 0).get_lines()
        assert same_color(own.get_color(), "#abcdef")
        assert same_color(shrunk.get_color(), "#123456")
    plt.close("all")


def test_sample_color_and_color_override_the_style(res1d, small_1d):
    from matplotlib.colors import same_color

    ax = res1d.plot(stack=small_1d.likelihoods.stack(), color="red", sample_color="green")
    median, stack = ax.get_lines()
    assert same_color(median.get_color(), "red") and same_color(stack.get_color(), "green")
    with plt.rc_context({"axes.prop_cycle": mpl.cycler(color=["#123456", "#abcdef"])}):
        ax = res1d.plot(color="C1")  # resolved inside the style, not when drawn later
    assert same_color(ax.get_lines()[0].get_color(), "#abcdef")
    with pytest.raises(ValueError, match="sample_color"):
        res1d.plot(sample_color="not a colour")
    plt.close("all")


def test_a_list_of_colours_is_for_the_bars_of_categories_only(res1d):
    from matplotlib.colors import same_color

    with pytest.raises(ValueError, match="color must be a matplotlib colour or None"):
        res1d.plot(color=["red", "blue"])
    cats = PopulationResult(
        None, np.full((5, 3), 1 / 3), method="external", bins=Bins.categories(3)
    )
    with pytest.raises(ValueError, match="a list of 3 of them"):
        cats.plot(color=["red", "green"])
    rgb = ["red", "lime", "blue"]
    for colors in (rgb, np.eye(3)):  # names, or rows as from a colormap
        ax = cats.plot(color=colors)
        assert same_color([bar.get_facecolor()[:3] for bar in ax.patches], rgb)
    plt.close("all")


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
    # the selection was lost on save and attach redrew the total without it (O-1)
    lk, res = fit_with_selection()
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


@pytest.mark.parametrize("kind", ["combined", "injections", "smooth injections"])
@pytest.mark.filterwarnings("ignore::binference.ApproximationWarning")  # Laplace's k-hat
def test_a_reloaded_fit_keeps_a_combined_or_injection_selection(tmp_path, kind):
    # the row of a combined selection and the injections are saved beside the values
    bins = Bins.linear(0, 1, 4)
    lk = binference.simulate.mock_1d(200, bins, noise=0.1, rng=0).likelihoods.with_selection(None)
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
