"""Slow statistical validation (run with ``pytest -m slow``; test plan 2.11).

* Simulation-based calibration of NUTS and Gibbs (and the expected failure of
  mean-field VB).
* Coverage of central credible intervals.
* Geweke "getting it right" tests of the Gibbs conditionals, with and without
  selection effects.
* Recovery of true populations from mock catalogues, and the interim-prior
  regression test (posteriors fed as likelihoods are biased toward the prior).
"""

import warnings

import numpy as np
import pytest

import binference
from binference import Bins, Likelihoods, PopulationModel
from binference import _kernels as kern
from binference import priors as P

from .conftest import sd_mcse

pytestmark = pytest.mark.slow

BINS4 = Bins.linear(0.0, 1.0, 4)


def _fit(method, **kw):
    def fit(lk, rng):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", binference.BinferenceWarning)
            return binference.infer(lk, method=method, rng=rng, check=False, **kw)

    return fit


def _sbc(prior, fit, n_simulations, n_draws, rng, *, n_objects=40, noise=0.2, **sim):
    simulate = binference.simulate.prior_predictive(prior, BINS4, n_objects, noise=noise, **sim)
    return binference.validation.sbc(
        simulate, fit, n_simulations=n_simulations, n_draws=n_draws, rng=rng
    )


def _pvalues(res, n_bins=10):
    # the chi-square is the more sensitive to width errors, the ECDF test to bias
    return {"chisquare": res.pvalues(n_bins), "ecdf": res.pvalues(method="ecdf")}


def _assert_uniform_ranks(res, alpha=1e-3, n_bins=10):
    """Both tests pass at the Bonferroni-corrected level of SBCResult.passed (DECISIONS #87)."""
    for method, p in _pvalues(res, n_bins).items():
        nb = n_bins if method == "chisquare" else None
        assert res.passed(alpha, nb, method=method), (method, dict(zip(res.names, p, strict=True)))


@pytest.mark.parametrize(
    "prior",
    [P.Dirichlet(1.0), P.LogisticRandomWalk(order=1, scale=0.8)],
    ids=["dirichlet", "randomwalk"],
)
# low ESS over-disperses the ranks, which only makes the uniformity test stricter
@pytest.mark.filterwarnings(
    "ignore:In .* fits the smallest bulk ESS:binference.ConvergenceWarning"
)
def test_sbc_nuts(prior):
    fit = _fit("nuts", prior=prior, n_chains=2, n_warmup=300, n_samples=300)
    _assert_uniform_ranks(_sbc(prior, fit, n_simulations=200, n_draws=99, rng=2026))


# Gibbs at K = 4, N = 40 takes 0.14 s per fit with the Dirichlet prior, which affords 1000
# simulations, and 0.24 s with the generalised Dirichlet, which gets 500 to bound the run
# time. Ranking 19 draws per fit keeps them nearly independent: with these settings the
# smallest bulk ESS of a fit was 34 in 3000 simulations for the Dirichlet prior and 45 in
# 2000 for the generalised Dirichlet.
GIBBS_DRAWS = 19
GIBBS_DIRICHLET = {"n_chains": 4, "n_warmup": 100, "n_samples": 600}


@pytest.mark.parametrize(
    ("prior", "kw", "n_simulations"),
    [
        (P.Dirichlet(1.0), GIBBS_DIRICHLET, 1000),
        (P.GeneralizedDirichlet(2.0), {"n_chains": 4, "n_warmup": 100, "n_samples": 400}, 500),
    ],
    ids=["dirichlet", "gdirichlet"],
)
def test_sbc_gibbs(prior, kw, n_simulations):
    # unfiltered, sbc's ESS warning is an error: every fit has min_ess >= GIBBS_DRAWS
    res = _sbc(prior, _fit("gibbs", prior=prior, **kw), n_simulations, GIBBS_DRAWS, 2026)
    assert np.all(res.min_ess >= GIBBS_DRAWS)
    _assert_uniform_ranks(res)


@pytest.mark.filterwarnings(
    "ignore:In .* fits the smallest bulk ESS:binference.ConvergenceWarning"
)
def test_sbc_gibbs_rejects_posteriors_about_18_percent_too_narrow():
    """Negative control: counting half the objects twice narrows the posteriors by about
    18% (sqrt(40/60) = 0.82). The Dirichlet Gibbs configuration rejects it with both tests
    already at 300 simulations: in 10 of 10 seeds, every smallest p-value below 4e-10
    against a threshold of 2e-4."""
    prior = P.Dirichlet(1.0)
    fit = _fit("gibbs", prior=prior, **GIBBS_DIRICHLET)

    def fit_doubled(lk, rng):
        return fit(lk.with_weights(np.repeat([1, 2], lk.N // 2)), rng)

    res = _sbc(prior, fit_doubled, 300, GIBBS_DRAWS, rng=2027)
    for method, p in _pvalues(res).items():
        assert np.any(p <= 1e-3 / p.size), (method, dict(zip(res.names, p, strict=True)))


@pytest.mark.xfail(strict=True, reason="mean-field VB is over-confident (documented)")
def test_sbc_fails_for_vb():
    sim = binference.simulate.prior_predictive(P.Dirichlet(1.0), BINS4, n_objects=80, noise=0.35)
    res = binference.validation.sbc(
        sim, _fit("vb", n_samples=400), n_simulations=150, n_draws=99, rng=7
    )
    p = res.pvalues(n_bins=10)
    assert np.all(p > 1e-3 / p.size)


def test_coverage_gibbs():
    # 1000 simulations: the 95% tolerance is +-0.028, so posteriors 15% too narrow (95%
    # coverage 0.904) fail in each quantity with probability 0.98 (0.79 with 300)
    sim = binference.simulate.prior_predictive(P.Dirichlet(1.0), BINS4, n_objects=60, noise=0.25)
    cov = binference.validation.coverage(
        sim,
        _fit("gibbs", n_samples=500, n_warmup=100),
        n_simulations=1000,
        levels=(0.68, 0.95),
        functionals={"mean": lambda r: r @ BINS4.centers},
        rng=99,
    )
    rates = cov.rates()
    se = cov.stderr()
    assert np.all(np.abs(rates - cov.levels[:, None]) < 4 * se), rates


def _geweke_z_scores(selection, augment_missed=True, prior=None):
    """Marginal-conditional vs successive-conditional simulation of (population, data).

    Returns z-scores comparing the means of the population, its square and the data
    histogram. ``prior`` defaults to a Dirichlet; for a rate prior the number of
    objects is part of the data, ``N ~ Poisson(S . lambda)`` with ``selection`` the
    exposure. ``augment_missed=False`` deliberately breaks the Gibbs kernel under
    selection (the missed objects are not imputed) as a negative control.
    """
    rng = np.random.default_rng(31)
    K, J, N = 3, 4, 12
    E = rng.dirichlet(np.full(J, 0.7), size=K)  # emission probabilities p(d | bin)
    prior = P.Dirichlet([1.5, 1.0, 2.0]) if prior is None else prior
    bound = prior.bind(Bins.categories(K))
    rates = bound.kind == "rates"
    S = np.ones(K) if selection is None else np.asarray(selection)

    def draw_objects(value, g):  # true bins of the observed objects
        q = value * S
        return g.choice(K, size=g.poisson(q.sum()) if rates else N, p=q / q.sum())

    def draw_data(z, g):
        u = g.random(z.size)[:, None]
        return (np.cumsum(E[z], axis=1) < u).sum(axis=1)

    def stats_of(value, d):
        return np.r_[value, value**2, np.bincount(d, minlength=J) / N]

    # marginal-conditional
    M = 20000
    mc = np.empty((M, 2 * K + J))
    for m in range(M):
        value = bound.sample(1, rng)[0]
        mc[m] = stats_of(value, draw_data(draw_objects(value, rng), rng))
    # successive-conditional: Gibbs on (z, value) | d, then d | z (and, for rates,
    # (z, d) | value, since the number of objects is data too)
    T = 60000
    sc = np.empty((T, 2 * K + J))
    value = bound.sample(1, rng)[0]
    z = draw_objects(value, rng)
    d = draw_data(z, rng)
    for t in range(T):
        L = E[:, d].T * S  # effective likelihoods (selection mode "on_truth")
        counts, z = kern.sample_counts(L, np.ones(d.size), value, rng, unit_weights=True)
        if selection is not None and augment_missed and not rates:
            n_missed = rng.negative_binomial(N, S @ value)
            if n_missed:
                qm = value * (1 - S)
                counts = counts + rng.multinomial(n_missed, qm / qm.sum())
        value = bound.posterior_sample(counts, rng, exposure=S if rates else None)
        if rates:
            z = draw_objects(value, rng)
        d = draw_data(z, rng)
        sc[t] = stats_of(value, d)
    # compare means with batch-means standard errors for the autocorrelated chain
    nb = 60
    batches = sc[: (T // nb) * nb].reshape(nb, -1, sc.shape[1]).mean(axis=1)
    se_sc = batches.std(axis=0, ddof=1) / np.sqrt(nb)
    se_mc = mc.std(axis=0, ddof=1) / np.sqrt(M)
    return (sc.mean(0) - mc.mean(0)) / np.sqrt(se_sc**2 + se_mc**2)


@pytest.mark.parametrize("selection", [None, (0.9, 0.5, 0.25)])
def test_geweke_getting_it_right(selection):
    z_scores = _geweke_z_scores(selection)
    assert np.all(np.abs(z_scores) < 4.5), z_scores


def test_geweke_has_power_against_a_broken_kernel():
    """Negative control: forgetting the missed objects under selection must be detected."""
    z_scores = _geweke_z_scores((0.9, 0.5, 0.25), augment_missed=False)
    assert np.max(np.abs(z_scores)) > 8.0, z_scores


def test_gibbs_and_nuts_agree_on_larger_problem():
    mock = binference.simulate.mock_1d(5000, Bins.linear(0, 3, 15), noise=(0.05, 0.2), rng=11)
    g = binference.infer(mock.likelihoods, method="gibbs", rng=1, n_samples=4000, check=False)
    n = binference.infer(mock.likelihoods, rng=2, n_samples=1000, check=False)
    mcse = binference.diagnostics.mcse
    for k in range(15):
        a, b = g.chains[:, :, k], n.chains[:, :, k]
        assert abs(a.mean() - b.mean()) < 5 * np.hypot(mcse(a), mcse(b)), k
        assert abs(a.std() - b.std()) < 5 * np.hypot(sd_mcse(a), sd_mcse(b)), k


# The recovery tests draw true values uniformly within bins (density given as bin
# fractions), so the histogram model is exactly right. With a continuous density
# and coarse bins the piecewise-constant model is misspecified and, for large N,
# the resulting binning bias exceeds the statistical errors (see the docs).
def _whitened_residuals(res, truth):
    """Error of the posterior mean of the first K - 1 fractions, whitened by their posterior
    covariance: independent N(0, 1) components if the posterior is calibrated and Gaussian."""
    chol = np.linalg.cholesky(res.cov()[:-1, :-1])
    return np.linalg.solve(chol, (res.mean - truth)[:-1])


def _assert_unit_variance(y):
    """Pooled mean square of whitened residuals within 4 chi-square standard errors of 1.

    The mean square is 1/c**2 + b**2 for posteriors c times the true width with bias b
    sd, so it catches width errors and biases alike. (The posterior mean's own
    shrinkage toward the prior adds about 0.03 with selection.) With n = 300-400 the
    tolerance is 0.28-0.33: posteriors 15% too narrow fail with probability 0.83-0.87
    (chi-square power; the +-25% sd window this replaced had 0.08).
    """
    v = np.mean(y**2)
    assert abs(v - 1) < 4 * np.sqrt(2 / y.size), v


def test_recovery_with_interim_prior_posteriors():
    """Posterior means vs true population over 80 mocks: whitened residuals have unit variance."""
    bins = Bins.linear(0, 3, 6)
    rho_true = binference.simulate.population(bins, "gaussian")
    ys = []
    for s in range(80):
        mock = binference.simulate.mock_1d(
            800, bins, density=rho_true, interim="gaussian", rng=1000 + s
        )
        lk = Likelihoods.from_table(mock.posteriors, bins, interim=mock.interim)
        res = binference.infer(lk, method="gibbs", rng=s, n_samples=500, check=False)
        ys.append(_whitened_residuals(res, mock.truth.rho))
    _assert_unit_variance(np.concatenate(ys))


def test_interim_prior_regression():
    """Feeding posteriors as likelihoods biases the population toward the interim prior."""
    bins = Bins.linear(0, 3, 6)
    rho_true = binference.simulate.population(bins, "bimodal")
    mock = binference.simulate.mock_1d(
        20000, bins, density=rho_true, interim="gaussian", noise=(0.3, 0.6), rng=5
    )
    lk_right = Likelihoods.from_table(mock.posteriors, bins, interim=mock.interim)
    lk_wrong = Likelihoods(mock.posteriors, bins)
    right = binference.infer(lk_right, method="gibbs", rng=1, n_samples=500, check=False)
    wrong = binference.infer(lk_wrong, method="gibbs", rng=2, n_samples=500, check=False)
    truth = mock.truth.rho
    assert np.all(np.abs(right.mean - truth) < 4 * right.std)
    assert np.max(np.abs(wrong.mean - truth) / wrong.std) > 8
    # Using posteriors adds W log(pi . rho) to the log-likelihood, which drags the
    # population toward where the interim prior is large.
    pi = mock.interim
    assert pi @ wrong.mean > pi @ right.mean + 0.01


def test_selection_recovery():
    bins = Bins.linear(0, 3, 4)
    S = np.array([0.95, 0.6, 0.35, 0.15])
    rho_true = binference.simulate.population(bins, "gaussian")
    ys = []
    for s in range(100):
        mock = binference.simulate.mock_1d(
            600, bins, density=rho_true, noise=(0.2, 0.5), selection=S, rng=2000 + s
        )
        res = binference.infer(mock.likelihoods, method="gibbs", rng=s, n_samples=500, check=False)
        ys.append(_whitened_residuals(res, mock.truth.rho))
    _assert_unit_variance(np.concatenate(ys))


def test_binning_bias_with_coarse_bins_is_real():
    """A continuous population on coarse bins: the bias grows with N (documented pitfall)."""
    bins = Bins.linear(0, 3, 6)
    mock = binference.simulate.mock_1d(20000, bins, density="bimodal", noise=(0.3, 0.6), rng=5)
    res = binference.infer(mock.likelihoods, method="gibbs", rng=1, n_samples=500, check=False)
    assert np.max(np.abs(res.mean - mock.truth.rho) / res.std) > 4


def test_vb_underestimates_widths_vs_nuts():
    mock = binference.simulate.mock_1d(3000, Bins.linear(0, 3, 15), noise=(0.2, 0.4), rng=3)
    n = binference.infer(mock.likelihoods, rng=1, n_samples=600, check=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", binference.ApproximationWarning)
        v = binference.infer(mock.likelihoods, method="vb", rng=1)
    assert np.median(v.std / n.std) < 0.5
    # (the per-object PDFs are broad compared with the bins: most information is missing)
    model = PopulationModel(mock.likelihoods)
    assert model.missing_information(n.mean) > 0.9
    # but its mean is in the right place: within one NUTS sd in every bin, while NUTS's
    # own MC error is about 0.03 sd. A VB mean shrunk 20% toward uniform misses by up
    # to 18 sd in the sparse upper bins.
    assert np.all(np.abs(v.mean - n.mean) < n.std)
