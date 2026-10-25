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
    assert mock.likelihoods.selection.mode == "on_truth"
    assert mock.truth.n_drawn > 400


def test_mock_2d_categorical_unfolding():
    m2 = mock_2d(300, rng=5)
    assert m2.likelihoods.K == 400 and m2.truth.values.shape == (300, 2)
    mc = mock_categorical(300, 3, rng=6)
    assert mc.likelihoods.bins.is_categorical
    np.testing.assert_allclose(mc.posteriors.sum(1), 1.0)
    mu = mock_unfolding(2000, rng=7)
    assert mu.likelihoods.selection.mode == "on_measurement"
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

    with pytest.warns(binference.ConvergenceWarning, match="ESS is below n_draws"):
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


def test_sbc_p_values_of_uniform_ranks_reject_at_the_nominal_rate():
    # 50 possible ranks in 20 bins hold 2 or 3 ranks each; the chi-square test assumed equal
    # bins and rejected uniform ranks 9% of the time at 1% (audit G-1)
    rng = np.random.default_rng(0)
    p = [
        binference.validation.SBCResult(rng.integers(0, 50, size=(150, 1)), 49, ["x"]).pvalues()[0]
        for _ in range(1000)
    ]
    # 1000 replicates: the rejection rate has a Monte Carlo SD of about 0.003 at 1%
    assert np.mean(np.array(p) <= 0.01) < 0.025


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.2])
def test_the_exact_posterior_is_accurate_at_the_simplex_boundary(alpha):
    # one flat object and one-hot counts (20, 15, 0): the posterior is Dir(alpha + counts);
    # the midpoint grid missed the rho_3 mean by up to 0.6 SD and its lower tail entirely
    from scipy import stats

    rows = [np.ones(3)] + [np.eye(3)[0]] * 20 + [np.eye(3)[1]] * 15
    a = alpha + np.array([20.0, 15.0, 0.0])
    ex = binference.validation.exact_posterior(Likelihoods(np.array(rows)), P.Dirichlet(alpha))
    sd = np.sqrt(a * (a.sum() - a) / (a.sum() ** 2 * (a.sum() + 1)))
    np.testing.assert_allclose((ex.mean - a / a.sum()) / sd, 0.0, atol=0.01)
    if alpha >= 0.5:  # at 0.2 the 5% quantile is 6e-9 and its tail mass is 17% off
        q05 = stats.beta.ppf(0.05, a[2], a.sum() - a[2])
        np.testing.assert_allclose(ex.weights[ex.points[:, 2] < q05].sum(), 0.05, atol=0.005)
