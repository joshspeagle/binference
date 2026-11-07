"""Totals and rates derived from fraction draws (DECISIONS #9, #78)."""

import numpy as np
import pytest
from scipy import stats

import binference
from binference import Bins, Likelihoods, PopulationModel, Selection
from binference.samplers import Gibbs


def _model(selection=None, n=300):
    mock = binference.simulate.mock_1d(n, Bins.linear(0, 3, 6), rng=1)
    return PopulationModel(mock.likelihoods, selection=selection)


def test_each_total_is_the_gamma_posterior_given_its_fractions():
    S = np.linspace(0.9, 0.3, 6)
    m = _model(Selection(S, "on_truth"))
    res = Gibbs(m, n_warmup=50, rng=0).run(300)
    # Lambda (S . rho) ~ Gamma(W, 1) draw by draw
    g = res.total * (res.samples @ S)
    assert stats.kstest(g, stats.gamma(m.W).cdf).pvalue > 1e-3
    np.testing.assert_allclose(res.rates, res.total[:, None] * res.samples)


def test_a_relative_selection_gives_no_total():
    m = _model(Selection(np.linspace(1.0, 0.4, 6), "on_measurement", scale="relative"))
    res = Gibbs(m, n_warmup=20, rng=0).run(20)
    assert res.total is None and res.rates is None
    with pytest.raises(ValueError, match="relative"):
        res.summary(which="rates")
    assert binference.infer(m.likelihoods, selection=m.selection_spec, method="map").total is None


def test_totals_are_reproducible_and_independent_across_draws():
    m = _model()
    eng = Gibbs(m, n_warmup=20, rng=3)
    res = eng.run(30)
    np.testing.assert_array_equal(res.total, eng.result.total)
    np.testing.assert_array_equal(Gibbs(m, n_warmup=20, rng=3).run(30).total, res.total)
    # wrapped external draws too, and two runs of one engine get independent variates
    wrapped = binference.PopulationResult.from_samples(m, res.samples)
    np.testing.assert_array_equal(
        wrapped.total, binference.PopulationResult.from_samples(m, res.samples).total
    )
    vb = binference.samplers.MeanFieldVB(m, rng=0)
    with pytest.warns(binference.ApproximationWarning):
        a, b = vb.run(200), vb.run(200)
    g = lambda r: r.total * r.samples.sum(axis=1)  # noqa: E731  Gamma(W, 1) variates
    assert not np.allclose(g(a), g(b))


def test_deriving_totals_leaves_the_callers_generator_alone():
    m = _model()
    g1, g2 = np.random.default_rng(5), np.random.default_rng(5)
    Gibbs(m, n_warmup=10, rng=g1).run(10).total  # noqa: B018  draws the totals
    Gibbs(m, n_warmup=10, rng=g2).run(10)
    np.testing.assert_array_equal(g1.random(3), g2.random(3))


def test_saved_results_keep_their_totals(tmp_path):
    res = Gibbs(_model(), n_warmup=20, rng=0).run(20)
    path = str(tmp_path / "r.npz")
    res.save(path)
    back = binference.PopulationResult.load(path)
    np.testing.assert_array_equal(back.total, res.total)
    np.testing.assert_array_equal(back.attach(res.model).total, res.total)  # drawn again, alike


def test_rates_with_a_basis_are_per_bin():
    mock = binference.simulate.mock_1d(200, Bins.linear(0, 3, 6), rng=2)
    basis = binference.Basis(mock.likelihoods.bins, order=1, resolution=2)
    lk = Likelihoods.from_gaussian(mock.measurements["mean"], mock.measurements["sigma"], basis)
    res = Gibbs(PopulationModel(lk), n_warmup=20, rng=0).run(20)
    assert res.rates.shape == (20 * 4, 6)
    np.testing.assert_allclose(res.rates.sum(axis=1), res.total)


def test_point_estimates_plot_their_rates():
    import matplotlib

    matplotlib.use("Agg")
    est = binference.infer(_model().likelihoods, method="map")
    ax = est.plot(which="rate")
    heights = np.concatenate([line.get_ydata() for line in ax.get_lines()])
    assert heights.max() > 1.0  # rates (W rho), not fractions
    rel = Selection(np.linspace(1.0, 0.4, 6), "on_measurement", scale="relative")
    est = binference.infer(_model().likelihoods, selection=rel, method="map")
    with pytest.raises(ValueError, match="relative"):
        est.plot(which="rate")
