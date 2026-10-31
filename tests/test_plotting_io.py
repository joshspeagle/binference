"""Smoke tests for plotting and round-trip tests for saving/loading."""

import numpy as np
import pytest

import binference
from binference import Bins, PopulationResult

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
    plain = binference.Likelihoods.from_gaussian(d, s, bins).toarray()[1] * bins.volumes
    np.testing.assert_allclose(own, plain / plain.sum(), rtol=1e-12)
    plt.close("all")


def test_the_sbc_plot_always_shows_the_joint_log_likelihood():
    from binference.plotting import plot_sbc

    names = [f"rho[{k}]" for k in range(15)] + ["loglike"]
    ranks = np.random.default_rng(1).integers(0, 20, (30, 16))
    sbc = binference.validation.SBCResult(ranks, 19, names)
    shown = [a for a in plot_sbc(sbc).axes if a.get_visible()]
    assert [a.get_title() for a in shown] == [*names[:11], "loglike"]
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
    mc = binference.simulate.mock_categorical(100, 3, rng=1)
    ax = binference.infer(mc.likelihoods, method="map").plot(color=["red", "green", "blue"])
    assert same_color([bar.get_facecolor()[:3] for bar in ax.patches], ["red", "green", "blue"])
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


def test_to_datatree(res1d):
    pytest.importorskip("xarray")
    dt = res1d.to_datatree()
    assert "posterior" in dt.children
    assert dt["posterior"]["rho"].shape == (4, 100, 12)
    # rates of a fractions fit are the derived ones, not the fractions (DECISIONS #78)
    np.testing.assert_allclose(dt["posterior"]["rate"].values.reshape(-1, 12), res1d.rates)
