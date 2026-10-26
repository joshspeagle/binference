"""Tests for binference.simulate and binference.validation (spec 9)."""

import numpy as np
import pytest
from scipy import stats

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


@pytest.mark.parametrize("kind", ["callable", "masses"])
def test_mock_1d_posteriors_and_samples_follow_the_interim_prior_given(kind):
    bins = Bins.linear(0, 3, 6)
    e = bins.edges[0]
    if kind == "callable":
        interim, Pi = (lambda z: np.exp(-2 * z)), -np.diff(np.exp(-2 * e))
    else:
        interim = Pi = np.arange(1.0, 7.0)  # masses: flat within each bin
    Pi = Pi / Pi.sum()
    mock = mock_1d(200, bins, noise=(0.1, 0.4), interim=interim, n_samples=200, rng=8)
    np.testing.assert_allclose(mock.interim, Pi, rtol=1e-6)
    post = mock.likelihoods.toarray() * Pi
    np.testing.assert_allclose(mock.posteriors, post / post.sum(1, keepdims=True), rtol=1e-6)
    # each object's samples fall in the bins with its exact posterior masses, from the
    # likelihood times the interim density on a fine grid in each bin
    z = e[:-1, None] + np.diff(e)[:, None] * (np.arange(2000) + 0.5) / 2000
    obs, sig = mock.measurements["mean"], mock.measurements["sigma"]
    dens = np.exp(-2 * z) if kind == "callable" else np.broadcast_to(Pi[:, None], z.shape)
    like = np.exp(-0.5 * ((z[None] - obs[:, None, None]) / sig[:, None, None]) ** 2)
    p = (like * dens).sum(-1)
    p /= p.sum(1, keepdims=True)
    counts = np.stack([np.bincount(bins.index(s), minlength=6) for s in mock.samples])
    S = mock.samples.shape[1]
    # sum over objects of multinomial counts: mean S sum_i p_ik, variance S sum_i p(1 - p)
    zscore = (counts.sum(0) - S * p.sum(0)) / np.sqrt(S * (p * (1 - p)).sum(0) + 1e-12)
    assert np.all(np.abs(zscore) < 4)


def test_mock_selection_thins_each_bin_and_adds_the_outlier_fraction():
    S = np.linspace(1.0, 0.2, 30)
    f_out, N = 0.1, 20000
    mock = mock_1d(N, selection=S, outlier_fraction=f_out, rng=4)
    assert mock.likelihoods.selection.mode == "on_truth"
    t = mock.truth
    # observed bin fractions are rho_k S_k / (rho . S): binomial errors per bin
    p = t.rho * S / (t.rho @ S)
    z = (t.counts - N * p) / np.sqrt(N * p * (1 - p))
    assert np.all(np.abs(z) < 4), z
    # n_drawn is negative binomial: draws needed for N successes at rate rho . S
    q = t.rho @ S
    assert abs(t.n_drawn - N / q) < 4 * np.sqrt(N * (1 - q)) / q
    # outliers are replaced by U(0, 3) draws regardless of selection; an outlier lands
    # more than 5 sigma from its true value with probability p_far, an inlier never does
    obs, sig, v = mock.measurements["mean"], mock.measurements["sigma"], t.values
    near = np.minimum(v + 5 * sig, 3.0) - np.maximum(v - 5 * sig, 0.0)
    p_far = f_out * (1 - near / 3.0)
    n_far = np.sum(np.abs(obs - v) > 5 * sig)
    assert abs(n_far - p_far.sum()) < 4 * np.sqrt(np.sum(p_far * (1 - p_far)))


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
    ex = binference.validation.exact_posterior(lk, P.Dirichlet(2.0))
    a = 2.0 + np.bincount(labels, minlength=3)
    np.testing.assert_allclose(ex.mean, a / a.sum(), atol=1e-3)
    np.testing.assert_allclose(ex.weights.sum(), 1.0)
    lk2 = Likelihoods(np.eye(2)[labels % 2])
    ex2 = binference.validation.exact_posterior(lk2)
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

    with pytest.warns(
        binference.ConvergenceWarning,
        match=r"In \d+ of 15 fits the smallest bulk ESS",
    ):
        res = binference.validation.sbc(sim, fit, n_simulations=15, n_draws=49, rng=1)
    assert res.ranks.shape == (15, 4)
    assert res.ranks.min() >= 0 and res.ranks.max() <= 49
    assert res.names[-1] == "loglike"
    assert res.pvalues().shape == (4,)
    assert res.histogram(5).sum(1).tolist() == [15] * 4
    assert isinstance(res.passed(), bool)
    with pytest.raises(ValueError, match="n_bins"):
        res.pvalues(n_bins=51)  # empty bins would make the statistic 0/0
    cov = binference.validation.coverage(
        sim,
        fit,
        n_simulations=10,
        levels=(0.5, 0.9),
        rng=2,
        functionals={"mean": lambda r: r @ bins.centers},
    )
    assert cov.coverage_rates().shape == (2, 4)
    assert cov.average().shape == (2,)
    assert cov.stderr().shape == (2, 1)
    with pytest.raises(ValueError, match="informative"):
        binference.simulate.prior_predictive(P.GammaRates(), bins)(np.random.default_rng(0))
    # the limit is a tunable; a prior expecting almost no objects fails instead of spinning
    with pytest.raises(ValueError, match="above max_objects"):  # about 4000
        binference.simulate.prior_predictive(P.GammaRates(1e4, 10.0), bins, max_objects=3e3)(
            np.random.default_rng(0)
        )
    with pytest.raises(ValueError, match="1000 populations"):
        binference.simulate.prior_predictive(P.GammaRates(1e4, 1e12), bins)(
            np.random.default_rng(0)
        )


DETECT = np.array([1.0, 0.6, 0.35, 0.2])


@pytest.mark.parametrize("mode", ["on_truth", "on_measurement"])
def test_prior_predictive_keeps_objects_in_proportion_to_the_selection(mode):
    # a near-uniform population measured almost exactly: the measured bin is the true
    # bin, so the observed counts are multinomial with p_k proportional to S_k
    bins, n = Bins.linear(0, 1, 4), 4000
    sim = binference.simulate.prior_predictive(
        P.Dirichlet(1e6), bins, n, noise=1e-4, selection=DETECT, selection_mode=mode
    )
    lk, rho = sim(np.random.default_rng(3))
    np.testing.assert_allclose(rho, 0.25, atol=0.002)
    assert lk.selection.mode == mode
    np.testing.assert_allclose(lk.selection.values, DETECT, atol=1e-3)
    counts = np.bincount(lk.toarray().argmax(axis=1), minlength=4)
    p = DETECT / DETECT.sum()
    assert np.all(np.abs(counts - n * p) < 4 * np.sqrt(n * p * (1 - p))), counts


def test_prior_predictive_selection_on_measurement_is_the_bin_average_keep_probability():
    # S_k = P(kept | true bin k): the measured value's bin (end bins extended) decides,
    # averaged over z uniform in the bin and sigma ~ U(0.05, 0.3); checked by Monte Carlo
    bins, noise = Bins.linear(0, 1, 4), (0.05, 0.3)
    sim = binference.simulate.prior_predictive(
        P.Dirichlet(1.0), bins, 10, noise=noise, selection=DETECT, selection_mode="on_measurement"
    )
    S = sim(np.random.default_rng(0))[0].selection.values
    rng, m = np.random.default_rng(1), 200_000
    for k in range(4):
        z = 0.25 * (k + rng.random(m))
        d = z + rng.uniform(*noise, m) * rng.standard_normal(m)
        kept = DETECT[np.clip(np.floor(d / 0.25).astype(int), 0, 3)]
        assert abs(kept.mean() - S[k]) < 4 * kept.std() / np.sqrt(m), (k, kept.mean(), S[k])


def test_prior_predictive_draws_a_poisson_number_of_objects_for_rates():
    # rates fixed near 100 per bin by a tight Gamma prior; N ~ Poisson(S . lambda)
    bins, S = Bins.linear(0, 1, 4), np.array([1.5, 1.0, 0.6, 0.3])
    sim = binference.simulate.prior_predictive(
        P.GammaRates(1e6, 1e4), bins, noise=0.1, selection=S, selection_mode="on_truth"
    )
    rng = np.random.default_rng(4)
    runs = [sim(rng) for _ in range(300)]
    n = np.array([lk.N for lk, _ in runs])
    lam = np.array([truth for _, truth in runs])
    np.testing.assert_allclose(lam, 100.0, rtol=0.005)  # the true rates, not fractions
    mean = float(S @ lam.mean(axis=0))
    assert abs(n.mean() - mean) < 4 * np.sqrt(mean / n.size)
    assert abs(n.var(ddof=1) / mean - 1) < 4 * np.sqrt(2 / (n.size - 1))  # Poisson dispersion
    lk = runs[0][0]
    assert lk.selection.scale == "exposure" and lk.selection.mode == "on_truth"
    np.testing.assert_array_equal(lk.selection.values, S)


def test_prior_predictive_validates_its_inputs():
    bins, pp = Bins.linear(0, 1, 4), binference.simulate.prior_predictive
    with pytest.raises(ValueError, match="selection_mode"):
        pp(P.Dirichlet(1.0), bins, selection=DETECT)
    with pytest.raises(ValueError, match="selection_mode"):
        pp(P.Dirichlet(1.0), bins, selection=DETECT, selection_mode="true")
    with pytest.raises(ValueError, match="without a selection"):
        pp(P.Dirichlet(1.0), bins, selection_mode="on_truth")
    with pytest.raises(ValueError, match="one value per bin"):
        pp(P.Dirichlet(1.0), bins, selection=[0.5, 0.5], selection_mode="on_truth")
    with pytest.raises(ValueError, match="must not exceed one"):
        pp(P.Dirichlet(1.0), bins, selection=[2.0, 1, 1, 1], selection_mode="on_truth")
    with pytest.raises(ValueError, match="n_objects must not be given"):
        pp(P.GammaRates(1.0, 1.0), bins, 50)
    with pytest.raises(ValueError, match="n_objects must be a positive integer"):
        pp(P.Dirichlet(1.0), bins, 0)
    with pytest.raises(ValueError, match="one-dimensional"):
        pp(P.Dirichlet(1.0), Bins.categories(3))


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


def test_sbc_ecdf_p_value_is_the_null_probability_of_a_smaller_band_statistic():
    # the exact recursion against Monte Carlo of the statistic for uniform ranks
    rng = np.random.default_rng(3)
    S, M, R = 60, 9, 20000
    F = np.arange(1, M + 1) / (M + 1)
    c = np.arange(S + 1)
    tail = 2 * np.minimum(stats.binom.cdf(c, S, F[:, None]), stats.binom.sf(c - 1, S, F[:, None]))
    C_null = rng.multinomial(S, np.full(M + 1, 1 / (M + 1)), size=R).cumsum(1)[:, :-1]
    g_null = tail[np.arange(M), C_null].min(1)
    for a in (0.7, 1.0, 1.5):  # U-shaped, uniform and humped ranks
        ranks = rng.binomial(M, rng.beta(a, a, size=(S, 1)))
        p = binference.validation.SBCResult(ranks, M, ["x"]).pvalues(method="ecdf")[0]
        g = tail[np.arange(M), np.bincount(ranks[:, 0], minlength=M + 1).cumsum()[:-1]].min()
        p_mc = np.mean(g_null <= g)
        assert abs(p - p_mc) < 4 * np.sqrt(max(p * (1 - p), 1e-3) / R), (a, p, p_mc)
    # badly skewed ranks return the Bonferroni bound at once (the full recursion took 26 s)
    skewed = np.minimum(rng.geometric(0.3, size=(4000, 1)) - 1, 99)
    p = binference.validation.SBCResult(skewed, 99, ["x"]).pvalues(method="ecdf")[0]
    assert 0 <= p < 1e-12
    res = binference.validation.SBCResult(ranks, M, ["x"])
    with pytest.raises(ValueError, match="n_bins"):
        res.pvalues(5, method="ecdf")
    with pytest.raises(ValueError, match="method"):
        res.passed(method="ks")


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
    if alpha >= 0.5:  # at 0.2 the 5% quantile is 6e-9 and its tail mass is 22% off
        q05 = stats.beta.ppf(0.05, a[2], a.sum() - a[2])
        np.testing.assert_allclose(ex.weights[ex.points[:, 2] < q05].sum(), 0.05, atol=0.005)
