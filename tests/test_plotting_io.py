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
    axes = r2.plot()
    assert len(axes) == 2
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
    assert d["method"] == "gibbs" and d["bins"] == "grid"


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
