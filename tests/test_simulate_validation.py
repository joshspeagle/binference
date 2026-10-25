"""Tests for binference.simulate and binference.validation (spec 9)."""

import numpy as np
import pytest

import binference
from binference import Bins, Likelihoods
from binference import priors as P
from binference.simulate import mock_1d, mock_2d, mock_categorical, mock_unfolding, population


def test_population_from_density_and_array():
    b = Bins.linear(0, 3, 30)
    rho = population(b, "bimodal")
    np.testing.assert_allclose(rho.sum(), 1.0)
    assert np.argmax(rho) < 15
    np.testing.assert_allclose(population(b, np.ones(30)), 1 / 30)
    with pytest.raises(ValueError):
        population(b, np.ones(3))
    with pytest.raises(ValueError):
        population(b, "lumpy")


def test_mock_1d_truth_consistency():
    mock = mock_1d(500, rng=1)
    lk, truth = mock
    assert lk.N == 500 and lk.K == 30
    np.testing.assert_array_equal(truth.counts, np.bincount(truth.labels, minlength=30))
    np.testing.assert_array_equal(mock.bins.index(truth.values), truth.labels)
    assert truth.n_drawn == 500
    assert set(mock.measurements) == {"mean", "sigma"}


def test_mock_1d_posteriors_round_trip():
    mock = mock_1d(200, interim="gaussian", rng=2)
    assert mock.posteriors.shape == (200, 30)
    np.testing.assert_allclose(mock.posteriors.sum(1), 1.0)
    rec = Likelihoods.from_table(mock.posteriors, mock.bins, interim=mock.interim)
    np.testing.assert_allclose(rec.toarray(), mock.likelihoods.toarray(), rtol=1e-8, atol=1e-12)


def test_mock_1d_samples():
    mock = mock_1d(50, interim="gaussian", n_samples=300, rng=3)
    assert mock.samples.shape == (50, 300)
    lk = Likelihoods.from_samples(mock.samples, mock.bins, mock.interim)
    # histogrammed likelihoods correlate with the exact ones
    a, b = lk.toarray().ravel(), mock.likelihoods.toarray().ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.85


def test_mock_selection_and_outliers():
    S = np.linspace(1.0, 0.2, 30)
    mock = mock_1d(400, selection=S, outlier_fraction=0.1, rng=4)
    assert mock.likelihoods.selection.mode == "true"
    assert mock.truth.n_drawn > 400


def test_mock_2d_categorical_unfolding():
    m2 = mock_2d(300, rng=5)
    assert m2.likelihoods.K == 400 and m2.truth.values.shape == (300, 2)
    mc = mock_categorical(300, 3, rng=6)
    assert mc.likelihoods.bins.is_categorical
    np.testing.assert_allclose(mc.posteriors.sum(1), 1.0)
    mu = mock_unfolding(2000, rng=7)
    assert mu.likelihoods.selection.mode == "observed"
    assert mu.response.matrix.shape == (25, 20)
    np.testing.assert_allclose(mu.likelihoods.W, mu.measurements["reco_counts"].sum())


def test_exact_posterior_conjugate():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 3, size=40)
    lk = Likelihoods(np.eye(3)[labels])
    ex = binference.validation.exact_posterior(lk, P.Dirichlet(2.0), n_grid=800)
    a = 2.0 + np.bincount(labels, minlength=3)
    np.testing.assert_allclose(ex.mean, a / a.sum(), atol=1e-3)
    np.testing.assert_allclose(ex.weights.sum(), 1.0)
    lk2 = Likelihoods(np.eye(2)[labels % 2])
    ex2 = binference.validation.exact_posterior(lk2, n_grid=4000)
    a2 = 1.0 + np.bincount(labels % 2, minlength=2)
    np.testing.assert_allclose(ex2.mean, a2 / a2.sum(), atol=1e-4)
    with pytest.raises(ValueError):
        binference.validation.exact_posterior(Likelihoods(np.ones((3, 4))))
    with pytest.raises(ValueError):
        binference.validation.exact_posterior(lk, P.LogisticNormal())


def test_sbc_and_coverage_smoke():
    bins = Bins.linear(0, 1, 3)
    sim = binference.simulate.prior_predictive(P.Dirichlet(1.0), bins, n_objects=30, noise=0.2)

    def fit(lk, rng):
        return binference.infer(
            lk, method="gibbs", rng=rng, n_samples=200, n_warmup=50, n_chains=1, check=False
        )

    res = binference.validation.sbc(sim, fit, n_simulations=15, n_draws=49, rng=1)
    assert res.ranks.shape == (15, 4)
    assert res.ranks.min() >= 0 and res.ranks.max() <= 49
    assert res.names[-1] == "loglike"
    assert res.pvalues().shape == (4,)
    assert res.histogram(5).sum(1).tolist() == [15] * 4
    assert isinstance(res.passed(), bool)
    cov = binference.validation.coverage(
        sim,
        fit,
        n_simulations=10,
        levels=(0.5, 0.9),
        rng=2,
        functionals={"mean": lambda r: r @ bins.centers},
    )
    assert cov.rates().shape == (2, 4)
    assert cov.average().shape == (2,)
    assert cov.stderr().shape == (2, 1)
    with pytest.raises(ValueError):
        binference.simulate.prior_predictive(P.GammaRates(), bins)
