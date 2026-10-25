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

pytestmark = pytest.mark.slow

BINS4 = Bins.linear(0.0, 1.0, 4)


def _fit(method, **kw):
    def fit(lk, rng):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", binference.BinferenceWarning)
            return binference.infer(lk, method=method, rng=rng, check=False, **kw)

    return fit


@pytest.mark.parametrize(
    ("method", "kw", "prior"),
    [
        ("nuts", {"n_chains": 2, "n_warmup": 300, "n_samples": 300}, P.Dirichlet(1.0)),
        ("gibbs", {"n_chains": 4, "n_warmup": 100, "n_samples": 400}, P.Dirichlet(1.0)),
        ("gibbs", {"n_chains": 4, "n_warmup": 100, "n_samples": 400}, P.GeneralizedDirichlet(2.0)),
        (
            "nuts",
            {"n_chains": 2, "n_warmup": 300, "n_samples": 300},
            P.LogisticRandomWalk(order=1, scale=0.8),
        ),
    ],
    ids=["nuts-dirichlet", "gibbs-dirichlet", "gibbs-gdirichlet", "nuts-randomwalk"],
)
def test_sbc(method, kw, prior):
    sim = binference.simulate.prior_predictive(prior, BINS4, n_objects=40, noise=0.2)
    res = binference.validation.sbc(
        sim, _fit(method, prior=prior, **kw), n_simulations=200, n_draws=99, rng=2026
    )
    p = res.pvalues(n_bins=10)
    assert np.all(p > 1e-3 / p.size), dict(zip(res.names, p, strict=True))


@pytest.mark.xfail(strict=True, reason="mean-field VB is over-confident (documented)")
def test_sbc_fails_for_vb():
    sim = binference.simulate.prior_predictive(P.Dirichlet(1.0), BINS4, n_objects=80, noise=0.35)
    res = binference.validation.sbc(
        sim, _fit("vb", n_samples=400), n_simulations=150, n_draws=99, rng=7
    )
    p = res.pvalues(n_bins=10)
    assert np.all(p > 1e-3 / p.size)


def test_coverage_gibbs():
    sim = binference.simulate.prior_predictive(P.Dirichlet(1.0), BINS4, n_objects=60, noise=0.25)
    cov = binference.validation.coverage(
        sim,
        _fit("gibbs", n_samples=500, n_warmup=100),
        n_simulations=300,
        levels=(0.68, 0.95),
        functionals={"mean": lambda r: r @ BINS4.centers},
        rng=99,
    )
    rates = cov.rates()
    se = cov.stderr()
    assert np.all(np.abs(rates - cov.levels[:, None]) < 4 * se), rates


def _geweke_z_scores(selection, augment_missed=True):
    """Marginal-conditional vs successive-conditional simulation of (rho, data).

    Returns z-scores comparing the means of rho, rho**2 and the data histogram.
    ``augment_missed=False`` deliberately breaks the Gibbs kernel under
    selection (the missed objects are not imputed) as a negative control.
    """
    rng = np.random.default_rng(31)
    K, J, N = 3, 4, 12
    E = rng.dirichlet(np.full(J, 0.7), size=K)  # emission probabilities p(d | bin)
    alpha = np.array([1.5, 1.0, 2.0])
    prior = P.Dirichlet(alpha).bind(Bins.categories(K))
    S = np.ones(K) if selection is None else np.asarray(selection)

    def draw_data(z, g):
        u = g.random(z.size)[:, None]
        return (np.cumsum(E[z], axis=1) < u).sum(axis=1)

    def stats_of(rho, d):
        return np.r_[rho, rho**2, np.bincount(d, minlength=J) / N]

    # marginal-conditional
    M = 20000
    mc = np.empty((M, 2 * K + J))
    for m in range(M):
        rho = rng.dirichlet(alpha)
        q = rho * S / (S @ rho)
        z = rng.choice(K, size=N, p=q)
        mc[m] = stats_of(rho, draw_data(z, rng))
    # successive-conditional: Gibbs on (z, rho) | d, then d | z
    T = 60000
    sc = np.empty((T, 2 * K + J))
    rho = rng.dirichlet(alpha)
    z = rng.choice(K, size=N, p=rho * S / (S @ rho))
    d = draw_data(z, rng)
    for t in range(T):
        L = E[:, d].T * S  # effective likelihoods (selection mode "true")
        counts, z = kern.sample_counts(L, np.ones(N), rho, rng, unit_weights=True)
        if selection is not None and augment_missed:
            n_missed = rng.negative_binomial(N, S @ rho)
            if n_missed:
                qm = rho * (1 - S)
                counts = counts + rng.multinomial(n_missed, qm / qm.sum())
        rho = prior.posterior_sample(counts, rng)
        d = draw_data(z, rng)
        sc[t] = stats_of(rho, d)
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
    tau = np.array([binference.diagnostics.autocorr_time(g.chains[:, :, k]) for k in range(15)])
    se = np.sqrt(g.std**2 * tau / g.n_draws + n.std**2 / (0.5 * n.n_draws))
    assert np.all(np.abs(g.mean - n.mean) < 5 * se)
    np.testing.assert_allclose(g.std, n.std, rtol=0.25)


# The recovery tests draw true values uniformly within bins (density given as bin
# fractions), so the histogram model is exactly right. With a continuous density
# and coarse bins the piecewise-constant model is misspecified and, for large N,
# the resulting binning bias exceeds the statistical errors (see the docs).
def test_recovery_with_interim_prior_posteriors():
    """Posterior means vs true population over many mocks: standardised residuals ~ N(0, 1)."""
    bins = Bins.linear(0, 3, 6)
    rho_true = binference.simulate.population(bins, "gaussian")
    zs = []
    for s in range(40):
        mock = binference.simulate.mock_1d(
            800, bins, density=rho_true, interim_prior="gaussian", rng=1000 + s
        )
        lk = Likelihoods.from_posteriors(mock.posteriors, bins, mock.interim_prior)
        res = binference.infer(lk, method="gibbs", rng=s, n_samples=500, check=False)
        zs.append((res.mean - mock.truth.rho) / res.std)
    zs = np.concatenate(zs)
    assert abs(zs.mean()) < 0.2
    assert 0.75 < zs.std() < 1.25


def test_interim_prior_regression():
    """Feeding posteriors as likelihoods biases the population toward the interim prior."""
    bins = Bins.linear(0, 3, 6)
    rho_true = binference.simulate.population(bins, "bimodal")
    mock = binference.simulate.mock_1d(
        20000, bins, density=rho_true, interim_prior="gaussian", noise=(0.3, 0.6), rng=5
    )
    lk_right = Likelihoods.from_posteriors(mock.posteriors, bins, mock.interim_prior)
    lk_wrong = Likelihoods(mock.posteriors, bins)
    right = binference.infer(lk_right, method="gibbs", rng=1, n_samples=500, check=False)
    wrong = binference.infer(lk_wrong, method="gibbs", rng=2, n_samples=500, check=False)
    truth = mock.truth.rho
    assert np.all(np.abs(right.mean - truth) < 4 * right.std)
    assert np.max(np.abs(wrong.mean - truth) / wrong.std) > 8
    # Using posteriors adds W log(pi . rho) to the log-likelihood, which drags the
    # population toward where the interim prior is large.
    pi = mock.interim_prior
    assert pi @ wrong.mean > pi @ right.mean + 0.01


def test_selection_recovery():
    bins = Bins.linear(0, 3, 4)
    S = np.array([0.95, 0.6, 0.35, 0.15])
    rho_true = binference.simulate.population(bins, "gaussian")
    zs = []
    for s in range(30):
        mock = binference.simulate.mock_1d(
            600, bins, density=rho_true, noise=(0.2, 0.5), selection=S, rng=2000 + s
        )
        res = binference.infer(mock.likelihoods, method="gibbs", rng=s, n_samples=500, check=False)
        zs.append((res.mean - mock.truth.rho) / res.std)
    zs = np.concatenate(zs)
    assert abs(zs.mean()) < 0.25
    assert 0.7 < zs.std() < 1.3


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
    # but its mean is in the right place on well-constrained bins
    model = PopulationModel(mock.likelihoods)
    assert model.missing_information(n.mean) > 0.9
