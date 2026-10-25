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


def test_totals_are_reproducible_and_leave_the_draws_unchanged():
    m = _model()
    eng = Gibbs(m, n_warmup=20, rng=3)
    res = eng.run(30)
    np.testing.assert_array_equal(res.total, eng.result.total)  # one stream per engine
    again = Gibbs(m, n_warmup=20, rng=3).run(30)
    np.testing.assert_array_equal(again.total, res.total)


def test_saved_results_keep_their_totals(tmp_path):
    res = Gibbs(_model(), n_warmup=20, rng=0).run(20)
    path = str(tmp_path / "r.npz")
    res.save(path)
    np.testing.assert_array_equal(binference.PopulationResult.load(path).total, res.total)


def test_rates_with_a_basis_are_per_bin():
    mock = binference.simulate.mock_1d(200, Bins.linear(0, 3, 6), rng=2)
    basis = binference.Basis(mock.likelihoods.bins, order=1, resolution=2)
    lk = Likelihoods.from_gaussian(mock.measurements["mean"], mock.measurements["sigma"], basis)
    res = Gibbs(PopulationModel(lk), n_warmup=20, rng=0).run(20)
    assert res.rates.shape == (20 * 4, 6)
    np.testing.assert_allclose(res.rates.sum(axis=1), res.total)
