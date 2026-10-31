"""Tests for population priors (spec 5)."""

import numpy as np
import pytest
from scipy import integrate, stats

from binference import Bins
from binference import priors as P
from binference._simplex import helmert_basis, softmax

B3 = Bins.linear(0.0, 3.0, 3)
B4 = Bins.linear(0.0, 4.0, 4)


def simplex_grid(n=600):
    u = (np.arange(n) + 0.5) / n
    a, b = np.meshgrid(u, u, indexing="ij")
    keep = a + b < 1
    return np.stack([a[keep], b[keep], 1 - a[keep] - b[keep]], axis=1), 1.0 / n**2


SIMPLEX_PRIORS = [
    P.Dirichlet(1.0),
    P.Dirichlet([2.0, 3.0, 1.5]),
    P.GeneralizedDirichlet(2.0),
    P.GeneralizedDirichlet([2.0, 1.5], [3.0, 1.2]),
    P.DirichletTree([[0, 1], 2], alpha=[[2.0, 1.5], [1.2, 2.5]]),
    P.PolyaTree(c=1.5),
]


@pytest.mark.parametrize("prior", SIMPLEX_PRIORS, ids=repr)
def test_density_normalised_and_moments(prior, rng):
    b = prior.bind(B3)
    pts, area = simplex_grid()
    dens = np.exp(b.log_density(pts))
    assert dens.sum() * area == pytest.approx(1.0, abs=5e-3)
    mean_q = (dens[:, None] * pts).sum(0) / dens.sum()
    draws = b.sample(40000, rng)
    assert draws.shape == (40000, 3)
    np.testing.assert_allclose(draws.sum(1), 1.0)
    np.testing.assert_allclose(draws.mean(0), mean_q, atol=6e-3)


@pytest.mark.parametrize("prior", SIMPLEX_PRIORS, ids=repr)
def test_ilr_density_and_gradient(prior, rng):
    K = 3
    b = prior.bind(B3)
    V = helmert_basis(K)
    zero = np.zeros(K)
    for _ in range(3):
        x = rng.normal(size=K - 1)
        eta, cache = b.forward(x)
        rho = softmax(eta)
        lp, g = b.log_prior_and_grad(x, cache, rho, zero)
        # density in ILR space = density on simplex x Jacobian
        expected = b.log_density(rho) + np.log(rho).sum() + 0.5 * np.log(K)
        assert lp == pytest.approx(expected, rel=1e-10, abs=1e-10)
        h = 1e-6
        fd = np.empty(K - 1)
        for j in range(K - 1):
            e = np.zeros(K - 1)
            e[j] = h
            fp = b.log_prior_and_grad(x + e, b.forward(x + e)[1], softmax(V @ (x + e)), zero)
            fm = b.log_prior_and_grad(x - e, b.forward(x - e)[1], softmax(V @ (x - e)), zero)
            fd[j] = (fp[0] - fm[0]) / (2 * h)
        np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)
        # the likelihood gradient is chained through V
        d_eta = rng.normal(size=K)
        _, g2 = b.log_prior_and_grad(x, cache, rho, d_eta)
        np.testing.assert_allclose(g2 - g, V.T @ d_eta, atol=1e-10)


@pytest.mark.parametrize("prior", SIMPLEX_PRIORS, ids=repr)
def test_conjugate_posterior_and_map(prior, rng):
    b = prior.bind(B3)
    counts = np.array([7.0, 2.0, 11.0])
    pts, _ = simplex_grid()
    logpost = b.log_density(pts) + (counts * np.log(pts)).sum(1)
    w = np.exp(logpost - logpost.max())
    w /= w.sum()
    mean_q = w @ pts
    draws = np.array([b.posterior_sample(counts, rng) for _ in range(20000)])
    np.testing.assert_allclose(draws.mean(0), mean_q, atol=4e-3)
    # MAP from counts maximises log prior + multinomial log-likelihood
    mode = b.map_from_counts(counts)
    assert np.isclose(mode.sum(), 1.0)
    best = pts[np.argmax(logpost)]
    np.testing.assert_allclose(mode, best, atol=3e-3)


@pytest.mark.parametrize("prior", SIMPLEX_PRIORS, ids=repr)
def test_vb_expected_log(prior, rng):
    b = prior.bind(B3)
    state = b.vb_update(np.array([3.0, 5.0, 1.0]))
    draws = b.vb_sample(state, 100000, rng)
    np.testing.assert_allclose(b.vb_expected_log(state), np.log(draws).mean(0), atol=0.02)
    np.testing.assert_allclose(b.vb_mean(state), draws.mean(0), atol=4e-3)


def test_dirichlet_constructors():
    assert np.allclose(P.Dirichlet.reference().resolve_alpha(B4), 0.25)
    assert np.allclose(P.Dirichlet.reference(4).resolve_alpha(B4), 0.25)
    np.testing.assert_allclose(
        P.Dirichlet.from_counts([1, 0, 3], strength=0.5).resolve_alpha(B3), [1.5, 1.0, 2.5]
    )
    b = Bins([0.0, 1.0, 3.0])
    np.testing.assert_allclose(P.Dirichlet.from_base_measure(3.0).resolve_alpha(b), [1.0, 2.0])
    with pytest.raises(ValueError):
        P.Dirichlet(0.0)
    with pytest.raises(ValueError):
        P.Dirichlet([1.0, 2.0]).bind(B3)
    with pytest.raises(ValueError):
        P.Dirichlet("flat")


def test_small_concentration_draws_have_no_exact_zeros(rng):
    """Gamma(0.003) underflows to exactly 0 ~12% of the time in naive sampling."""
    a = np.array([0.003, 0.003, 0.5, 2.0])
    b = P.Dirichlet(a).bind(Bins.linear(0, 1, 4))
    d = b.sample(20000, rng)
    assert np.all(d > 0)
    assert np.all(np.isfinite(b.log_density(d)))
    A = a.sum()
    np.testing.assert_allclose(d.mean(0), a / A, atol=0.01)
    post = b.posterior_sample(np.zeros(4), rng)
    assert np.all(post > 0)
    tree = P.GeneralizedDirichlet(0.003).bind(Bins.linear(0, 1, 4))
    assert np.all(tree.sample(2000, rng) > 0)
    g = P.GammaRates(0.003, 1.0).bind(Bins.linear(0, 1, 4))
    assert np.all(g.sample(2000, rng) > 0)
    assert np.all(g.posterior_sample(np.zeros(4), rng) > 0)


def test_dirichlet_moments_analytic(rng):
    a = np.array([0.5, 2.0, 4.0])
    d = P.Dirichlet(a).bind(B3).sample(100000, rng)
    A = a.sum()
    np.testing.assert_allclose(d.mean(0), a / A, atol=4e-3)
    np.testing.assert_allclose(d.var(0), a * (A - a) / (A**2 * (A + 1)), rtol=0.03)


def test_generalized_dirichlet_reduces_to_dirichlet():
    alpha = np.array([1.5, 2.0, 0.7, 3.0])
    gd = P.GeneralizedDirichlet(alpha[:-1], np.cumsum(alpha[::-1])[::-1][1:]).bind(B4)
    d = P.Dirichlet(alpha).bind(B4)
    rng = np.random.default_rng(3)
    pts = rng.dirichlet(np.ones(4), size=50)
    np.testing.assert_allclose(gd.log_density(pts), d.log_density(pts), rtol=1e-10)
    with pytest.raises(ValueError):
        P.GeneralizedDirichlet([1.0, 2.0, 3.0]).bind(B4)


def test_generalized_dirichlet_positive_correlation(rng):
    gd = P.GeneralizedDirichlet([1.0, 50.0], [1.0, 50.0]).bind(B3)
    d = gd.sample(20000, rng)
    assert np.corrcoef(d[:, 1], d[:, 2])[0, 1] > 0.9
    # a Dirichlet can only produce negative correlations
    dd = P.Dirichlet(1.0).bind(B3).sample(20000, rng)
    assert np.corrcoef(dd[:, 1], dd[:, 2])[0, 1] < 0


def test_dirichlet_tree_consistent_collapses_to_dirichlet():
    # inner node concentration equals the sum of its children's -> plain Dirichlet
    tree = P.DirichletTree([[0, 1], 2], alpha=[[5.0, 2.0], [2.0, 3.0]]).bind(B3)
    d = P.Dirichlet([2.0, 3.0, 2.0]).bind(B3)
    rng = np.random.default_rng(1)
    pts = rng.dirichlet(np.ones(3), size=30)
    np.testing.assert_allclose(tree.log_density(pts), d.log_density(pts), rtol=1e-10)


def test_dirichlet_tree_validation():
    with pytest.raises(ValueError, match=r"more than once: bin 1 \[1, 2\)"):
        P.DirichletTree([[0, 1], 1]).bind(B3)
    with pytest.raises(ValueError, match="a group needs at least two members"):
        P.DirichletTree([[0], 1, 2]).bind(B3)
    with pytest.raises(ValueError, match="one alpha array per internal node"):
        P.DirichletTree([[0, 1], 2], alpha=[[1.0, 1.0]]).bind(B3)


def test_polya_tree_mean_is_base_and_concentrates(rng):
    b = Bins([0.0, 1.0, 2.0, 4.0, 8.0, 9.0])
    p1 = P.PolyaTree(c=1.0).bind(b)
    p2 = P.PolyaTree(c=20.0).bind(b)
    d1, d2 = p1.sample(40000, rng), p2.sample(40000, rng)
    base = b.volumes / b.volumes.sum()
    np.testing.assert_allclose(d1.mean(0), base, atol=6e-3)
    np.testing.assert_allclose(d2.mean(0), base, atol=3e-3)
    assert np.all(d2.std(0) < d1.std(0))
    with pytest.raises(ValueError):
        P.PolyaTree().bind(Bins.categories(3))


def test_gamma_rates(rng):
    a, bb = np.array([2.0, 3.0, 0.5]), np.array([1.0, 0.5, 2.0])
    b = P.GammaRates(a, bb).bind(B3)
    d = b.sample(100000, rng)
    np.testing.assert_allclose(d.mean(0), a / bb, rtol=0.02)
    counts, S = np.array([4.0, 0.0, 10.0]), np.array([1.0, 2.0, 0.5])
    post = np.array([b.posterior_sample(counts, rng, exposure=S) for _ in range(40000)])
    np.testing.assert_allclose(post.mean(0), (a + counts) / (bb + S), rtol=0.02)
    np.testing.assert_allclose(
        b.map_from_counts(counts, S), np.maximum(a + counts - 1, 0) / (bb + S)
    )
    lam = np.array([1.0, 2.0, 0.3])
    lp = b.log_density(lam)
    assert lp == pytest.approx(stats.gamma.logpdf(lam, a, scale=1 / bb).sum())
    h = 1e-6
    fd = [(b.log_density(lam + h * e) - b.log_density(lam - h * e)) / (2 * h) for e in np.eye(3)]
    np.testing.assert_allclose(b.grad_log_density(lam), fd, rtol=1e-6)
    with pytest.raises(ValueError):
        P.GammaRates(-1.0)


@pytest.mark.parametrize(
    "h",
    [P.HalfNormal(1.5), P.LogNormal(0.3, 0.7), P.InverseGamma(3.0, 2.0), P.Gamma(2.0, 3.0)],
    ids=repr,
)
def test_hyperpriors(h, rng):
    total = integrate.quad(lambda u: np.exp(h.logpdf_u(u)), -30, 30, limit=400)[0]
    assert total == pytest.approx(1.0, rel=1e-6)
    for u in (-1.0, 0.2, 1.3):
        fd = (h.logpdf_u(u + 1e-6) - h.logpdf_u(u - 1e-6)) / 2e-6
        assert h.dlogpdf_u(u) == pytest.approx(fd, rel=1e-5, abs=1e-8)
    s = h.sample(200000, rng)
    assert np.median(s) == pytest.approx(h.median(), rel=0.02)


def test_fixed_and_as_hyperprior():
    f = P.Fixed(2.0)
    assert f.fixed and f.median() == 2.0
    with pytest.raises(ValueError):
        P.Fixed(-1.0)
    from binference.priors.hyper import as_hyperprior

    assert isinstance(as_hyperprior(3), P.Fixed)
    with pytest.raises(TypeError):
        as_hyperprior("x")


LOGISTIC_PRIORS = [
    P.LogisticNormal(),
    P.LogisticNormal(cov=0.5 * np.eye(6) + 0.3),
    P.LogisticRandomWalk(order=1),
    P.LogisticRandomWalk(order=2, scale=0.7),
    P.LogisticGP(),
    P.LogisticGP(kernel="rbf", amplitude=1.0, length_scale=1.0),
]


# 5 x 4 cells of unequal widths: per-axis differences, Kronecker bases, the 2-D spectral
# density and one inferred length scale per axis (ARD)
GRID_2D = Bins.grid(np.linspace(0.0, 1.0, 6), np.linspace(0.0, 2.0, 5))
LOGISTIC_PRIORS_2D = [
    P.LogisticRandomWalk(order=1),
    P.LogisticRandomWalk(order=2, scale=0.7),
    P.LogisticGP(),
    P.LogisticGP(kernel="rbf", amplitude=1.0, length_scale=[0.5, 1.2]),
]
LOGISTIC_CASES = [pytest.param(p, Bins.linear(0.0, 3.0, 6), id=repr(p)) for p in LOGISTIC_PRIORS]
LOGISTIC_CASES += [pytest.param(p, GRID_2D, id=f"2d-{p!r}") for p in LOGISTIC_PRIORS_2D]


@pytest.mark.parametrize("centred", [False, True], ids=["noncentred", "partial"])
@pytest.mark.parametrize(("prior", "bins"), LOGISTIC_CASES)
def test_logistic_gradients(prior, bins, centred, rng):
    b = prior.bind(bins)
    if centred:
        b = b.with_centering(rng.random(b.n_active))
    for _ in range(3):
        x = rng.normal(scale=0.5, size=b.dim)
        d_eta = rng.normal(size=bins.K)
        d_eta -= d_eta.mean()  # likelihood gradients w.r.t. softmax logits sum to zero

        def f(xx, d_eta=d_eta):
            eta, cache = b.forward(xx)
            lp, _ = b.log_prior_and_grad(xx, cache, softmax(eta), d_eta)
            return lp + float(d_eta @ eta)

        eta, cache = b.forward(x)
        _, g = b.log_prior_and_grad(x, cache, softmax(eta), d_eta)
        fd = np.array([(f(x + 1e-6 * e) - f(x - 1e-6 * e)) / 2e-6 for e in np.eye(b.dim)])
        np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("prior", LOGISTIC_PRIORS, ids=repr)
def test_logistic_sampling_and_init(prior, rng):
    bins = Bins.linear(0.0, 3.0, 6)
    b = prior.bind(bins)
    d = b.sample(200, rng)
    assert d.shape == (200, 6)
    np.testing.assert_allclose(d.sum(1), 1.0)
    target = np.array([0.05, 0.1, 0.3, 0.3, 0.15, 0.1])
    x0 = b.init_from_value(target)
    eta, _ = b.forward(x0)
    assert np.corrcoef(np.log(softmax(eta)), np.log(target))[0, 1] > 0.8


@pytest.mark.parametrize("prior", LOGISTIC_PRIORS[2:], ids=repr)
def test_partial_centring_is_a_reparametrisation(prior, rng):
    """Same field for mapped coordinates; the prior on the field is unchanged."""
    bins = Bins.linear(0.0, 3.0, 6)
    b = prior.bind(bins)
    lam = rng.random(b.n_active)
    c = b.with_centering(lam)
    assert np.all(b.lam == 1.0)  # the original is untouched
    x = rng.normal(scale=0.5, size=b.dim)
    y = b.recentre(x, lam)
    np.testing.assert_allclose(c.forward(y)[0], b.forward(x)[0], atol=1e-12)
    np.testing.assert_allclose(c.recentre(y, np.ones(b.n_active)), x, atol=1e-12)

    # log-density difference between parametrisations is the log-Jacobian
    # sum_j (1 - lam_j) log s_j, whatever the field values
    def lp(bound, xx):
        eta, cache = bound.forward(xx)
        return bound.log_prior_and_grad(xx, cache, softmax(eta), np.zeros(6))[0]

    log_s, _ = b.log_scales(b.theta_from_x(x))
    np.testing.assert_allclose(lp(b, x) - lp(c, y), float((1 - lam) @ log_s), atol=1e-10)
    x2 = x.copy()
    x2[: b.n_active] = rng.normal(size=b.n_active)
    log_s2, _ = b.log_scales(b.theta_from_x(x2))
    y2 = b.recentre(x2, lam)
    np.testing.assert_allclose(lp(b, x2) - lp(c, y2), float((1 - lam) @ log_s2), atol=1e-10)
    with pytest.raises(ValueError):
        b.with_centering(np.ones(b.n_active + 1))


def test_logistic_prior_survives_scale_underflow():
    """Extreme hyperparameters underflow some scales to zero without NaNs or warnings."""
    b = P.LogisticGP(kernel="rbf").bind(Bins.linear(0.0, 3.0, 12))
    lam = np.r_[np.full(6, 0.5), np.ones(b.n_active - 6)]
    for prior in (b, b.with_centering(lam)):
        x = np.zeros(prior.dim)
        x[prior.n_active :] = [-800.0, 800.0]  # absurd amplitude and length scale
        eta, cache = prior.forward(x)
        lp, g = prior.log_prior_and_grad(x, cache, softmax(eta), np.zeros(12))
        assert np.isfinite(lp) or lp == -np.inf
        assert not np.any(np.isnan(g))


def test_inverse_gamma_from_quantiles():
    from scipy import stats

    h = P.InverseGamma.from_quantiles(0.2, 3.0)
    d = stats.invgamma(h.a, scale=h.b)
    assert d.cdf(0.2) == pytest.approx(0.01, abs=1e-8)
    assert d.sf(3.0) == pytest.approx(0.01, abs=1e-8)
    h2 = P.InverseGamma.from_quantiles(1.0, 2.0, tail=0.05)
    assert stats.invgamma(h2.a, scale=h2.b).cdf(1.0) == pytest.approx(0.05, abs=1e-8)
    with pytest.raises(ValueError):
        P.InverseGamma.from_quantiles(2.0, 1.0)
    with pytest.raises(ValueError):
        P.InverseGamma.from_quantiles(1.0, 2.0, tail=0.6)


def test_gp_default_length_scale_prior_tracks_the_grid_and_the_basis():
    from scipy import stats

    # 30 bins of width 0.1: the default 30 sines on a box of half-width 3 * 1.5 resolve
    # 2.2 * 3 * 1.5 / 30 = 0.33 (Matérn-3/2); 120 sines resolve two bin widths
    for n_basis, lower in ((None, 0.33), (120, 0.2)):
        b = P.LogisticGP(n_basis=n_basis).bind(Bins.linear(0.0, 3.0, 30))
        h = dict(b.hypers)["length_scale"]
        d = stats.invgamma(h.a, scale=h.b)
        assert d.cdf(lower) == pytest.approx(0.01, abs=1e-6)
        assert d.sf(3.0) == pytest.approx(0.01, abs=1e-6)  # the range


def test_logistic_priors_need_grids():
    with pytest.raises(ValueError):
        P.LogisticRandomWalk().bind(Bins.categories(4))
    with pytest.raises(ValueError):
        P.LogisticGP().bind(Bins.categories(4))
    P.LogisticNormal().bind(Bins.categories(4))  # fine on categories


def test_random_walk_roughness(rng):
    bins = Bins.linear(0.0, 1.0, 40)
    for tau in (0.3, 1.2):
        b = P.LogisticRandomWalk(order=2, scale=tau).bind(bins)
        d = np.log(b.sample(2000, rng))
        second = np.diff(d, n=2, axis=1)
        assert second.std() == pytest.approx(tau, rel=0.1)


def test_random_walk_2d_structure():
    bins = Bins.grid(np.linspace(0, 1, 6), np.linspace(0, 1, 5))
    b = P.LogisticRandomWalk(order=1, scale=1.0).bind(bins)
    assert b.dim == bins.K - 1
    b2 = P.LogisticRandomWalk(order=2, scale=1.0).bind(bins)
    assert int(b2.null.sum()) == 3  # x, y and xy trends (constant dropped)


def test_gp_matches_exact_kernel():
    bins = Bins.linear(0.0, 1.0, 30)
    ell, amp = 0.15, 1.3
    b = P.LogisticGP(kernel="matern32", amplitude=amp, length_scale=ell, n_basis=30).bind(bins)
    s = np.exp(b.log_scales(np.array([amp, ell]))[0])
    Bm = b.basis.factors[0]
    cov = (Bm * s**2) @ Bm.T
    x = bins.centers
    r = np.abs(x[:, None] - x[None, :])
    exact = amp**2 * (1 + np.sqrt(3) * r / ell) * np.exp(-np.sqrt(3) * r / ell)
    Pm = np.eye(30) - 1.0 / 30
    np.testing.assert_allclose(Pm @ cov @ Pm, Pm @ exact @ Pm, atol=0.06 * amp**2)


@pytest.mark.parametrize("kernel", ["rbf", "matern32"])
def test_2d_gp_matches_exact_ard_kernel(kernel):
    """The Kronecker basis and the 2-D spectral density reproduce k(r), r^2 = sum (dx_d/l_d)^2.

    For the RBF that is the product of the 1-D kernels; the Matern is not separable.
    """
    bins = Bins.grid(np.linspace(0.0, 1.0, 13), np.linspace(0.0, 2.0, 11))
    amp, ell = 1.3, np.array([0.25, 0.6])  # a different length scale per axis
    prior = P.LogisticGP(kernel=kernel, amplitude=amp, length_scale=list(ell), n_basis=30)
    b = prior.bind(bins)
    s = np.exp(b.log_scales(np.r_[amp, ell])[0])
    cols = b.basis.matvec(np.diag(s))  # (J, K): the scaled basis functions
    cov = cols.T @ cols
    x = bins.centers  # (K, 2)
    r = np.sqrt((((x[:, None] - x[None, :]) / ell) ** 2).sum(-1))
    if kernel == "rbf":
        exact = amp**2 * np.exp(-0.5 * r**2)
    else:
        exact = amp**2 * (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    Pm = np.eye(bins.K) - 1.0 / bins.K
    # the basis truncation leaves 6e-3 (Matern) and 2e-11 (RBF); the product of 1-D
    # Materns would be off by 5e-2, swapped length scales by 0.6
    np.testing.assert_allclose(Pm @ cov @ Pm, Pm @ exact @ Pm, atol=0.02 * amp**2)


def test_logistic_normal_samples_have_the_given_moments(rng):
    """Zero-sum log-fractions of prior draws have the centred mean and covariance."""
    K, n = 5, 20000
    mean = np.array([0.5, -1.0, 0.0, 1.0, -0.5])
    i = np.arange(K)
    cov = 0.8 * np.exp(-np.abs(i[:, None] - i[None, :]) / 2.0) + 0.2 * np.eye(K)
    b = P.LogisticNormal(mean=mean, cov=cov).bind(Bins.linear(0.0, 1.0, K))
    eta = np.log(b.sample(n, rng))
    eta -= eta.mean(axis=1, keepdims=True)
    Pm = np.eye(K) - 1.0 / K
    mu, C = Pm @ mean, Pm @ cov @ Pm
    # 5 standard errors per entry: of a mean, sqrt(C_ii / n); of a Gaussian covariance,
    # sqrt((C_ii C_jj + C_ij^2) / n)
    np.testing.assert_array_less(np.abs(eta.mean(0) - mu), 5 * np.sqrt(np.diag(C) / n))
    se = np.sqrt((np.outer(np.diag(C), np.diag(C)) + C**2) / n)
    np.testing.assert_array_less(np.abs(np.cov(eta.T) - C), 5 * se)


def test_gp_spectral_density_rbf_1d():
    bins = Bins.linear(0.0, 2.0, 10)
    b = P.LogisticGP(kernel="rbf", amplitude=1.0, length_scale=0.5).bind(bins)
    s = np.exp(b.log_scales(np.array([1.0, 0.5]))[0])
    w = np.sqrt(b.omega2[:, 0])
    expected = np.sqrt(np.sqrt(2 * np.pi) * 0.5 * np.exp(-0.5 * (0.5 * w) ** 2))
    np.testing.assert_allclose(s, expected, rtol=1e-12)


def test_reprs():
    for p in SIMPLEX_PRIORS + LOGISTIC_PRIORS + [P.GammaRates()]:
        assert type(p).__name__ in repr(p)


GALAXIES = Bins.categories(["Sa", "Sb", "E0", "E3", "S0"])
GROUPS = {"spiral": ["Sa", "Sb"], "early": {"elliptical": ["E0", "E3"], "lenticular": ["S0"]}}


def test_describe_draws_each_branch_with_its_concentration_share_and_mass():
    assert P.PolyaTree().describe(Bins.linear(0.0, 1.0, 4)) == "\n".join(
        [
            "PolyaTree: prior, on 4 bins",
            "root: bins 0–3",
            "├── α=1  share 0.50  mass 0.500  bins 0–1",
            "│   ├── α=4  share 0.50  mass 0.250  bin 0 [0, 0.25)",
            "│   └── α=4  share 0.50  mass 0.250  bin 1 [0.25, 0.5)",
            "└── α=1  share 0.50  mass 0.500  bins 2–3",
            "    ├── α=4  share 0.50  mass 0.250  bin 2 [0.5, 0.75)",
            "    └── α=4  share 0.50  mass 0.250  bin 3 [0.75, 1)",
        ]
    )


def test_a_dirichlet_is_described_as_a_tree_of_one_node():
    text = P.Dirichlet([1.0, 3.0]).describe(Bins.categories(["a", "b"]))
    assert text.splitlines()[2:] == [
        "├── α=1  share 0.25  mass 0.250  a",
        "└── α=3  share 0.75  mass 0.750  b",
    ]


def test_a_generalized_dirichlet_is_described_as_its_chain_of_splits():
    gd = P.GeneralizedDirichlet([1.0, 2.0], [1.0, 1.0])
    text = gd.describe(Bins.linear(0.0, 1.0, 3), counts=[2, 0, 1])
    # Beta(1 + 2, 1 + 0 + 1) for bin 0 against the rest, then Beta(2 + 0, 1 + 1)
    assert text.splitlines() == [
        "GeneralizedDirichlet: posterior given the counts, on 3 bins",
        "root: bins 0–2",
        "├── α=3  share 0.60  mass 0.600  bin 0 [0, 0.3333)",
        "└── α=2  share 0.40  mass 0.400  bins 1–2",
        "    ├── α=2  share 0.50  mass 0.200  bin 1 [0.3333, 0.6667)",
        "    └── α=2  share 0.50  mass 0.200  bin 2 [0.6667, 1)",
    ]
    with pytest.raises(ValueError, match="one value per bin"):
        gd.describe(Bins.linear(0.0, 1.0, 3), counts=[1, 2])


def test_named_groups_bind_to_the_tree_their_nested_lists_give():
    named = P.DirichletTree(GROUPS, alpha=2.0).bind(GALAXIES).tree
    nested = P.DirichletTree([[0, 1], [[2, 3], 4]], alpha=2.0).bind(GALAXIES).tree
    for attr in ("node_start", "slot_alpha", "slot_mem", "node_mem"):
        np.testing.assert_array_equal(getattr(named, attr), getattr(nested, attr))


def test_an_alpha_mapping_sets_exactly_the_branches_it_names():
    prior = P.DirichletTree(GROUPS, alpha={"spiral": 3.0, "E0": 5.0, "lenticular": 2.0})
    # slots in pre-order: root (spiral, early), spiral (Sa, Sb), early (elliptical, S0),
    # elliptical (E0, E3)
    np.testing.assert_array_equal(prior.bind(GALAXIES).tree.slot_alpha, [3, 1, 1, 1, 1, 2, 5, 1])


def test_an_alpha_key_names_a_bin_as_the_tree_writes_it():
    # the tree writes bins by index, so the key 1 is index 1, never the label 1 (index 0)
    tree = P.DirichletTree([[0, 1], 2], alpha={1: 5.0}).bind(Bins.categories([1, 0, 2])).tree
    np.testing.assert_array_equal(tree.slot_alpha, [1, 1, 1, 5])


def test_named_groups_are_drawn_with_their_names():
    # the drawing docs/guides/priors.md shows
    assert P.DirichletTree(GROUPS, alpha={"spiral": 3.0}).describe(GALAXIES).splitlines() == [
        "DirichletTree: prior, on 5 bins",
        "root: Sa, Sb, E0, E3, S0",
        "├── α=3  share 0.75  mass 0.750  spiral: Sa, Sb",
        "│   ├── α=1  share 0.50  mass 0.375  Sa",
        "│   └── α=1  share 0.50  mass 0.375  Sb",
        "└── α=1  share 0.25  mass 0.250  early: E0, E3, S0",
        "    ├── α=1  share 0.50  mass 0.125  elliptical: E0, E3",
        "    │   ├── α=1  share 0.50  mass 0.062  E0",
        "    │   └── α=1  share 0.50  mass 0.062  E3",
        "    └── α=1  share 0.50  mass 0.125  lenticular: S0",
    ]


@pytest.mark.parametrize(
    ("tree", "alpha", "bins", "message"),
    [
        (
            {"spiral": ["Sa", "Sx"], "early": ["E0", "E3", "S0"]},
            1.0,
            GALAXIES,
            "'Sx' is not one of the bins' labels",
        ),
        ({"spiral": ["Sa", "Sb"], "early": ["E0", "E3"]}, 1.0, GALAXIES, "in no group: S0"),
        (GROUPS, {"nope": 2.0}, GALAXIES, "'nope' names no group"),
        ([["a", "b"], 2], 1.0, B3, "labels need categories"),
        ([b"ab", 2], 1.0, B3, "b'ab' is not a bin index"),
        ([{0, 1}, 2], 1.0, B3, "neither a bin nor a group"),
        ({"Sa": ["Sa", "Sb"], "x": ["E0", "E3", "S0"]}, {"Sa": 3.0}, GALAXIES, "more than one"),
        (GROUPS, {"lenticular": 2.0, "S0": 7.0}, GALAXIES, "'lenticular' and 'S0' name one"),
        (GROUPS, {"spiral": [1.0, 2.0]}, GALAXIES, "must be one number"),
    ],
)
def test_a_wrong_tree_or_alpha_key_is_refused_with_an_error_naming_it(tree, alpha, bins, message):
    with pytest.raises(ValueError, match=message):
        P.DirichletTree(tree, alpha=alpha).bind(bins)


def test_a_dict_of_one_group_inside_a_list_names_that_group():
    tree = P.DirichletTree([{"a": [0, 1]}, 2], alpha={"a": 4.0}).bind(B3).tree
    np.testing.assert_array_equal(tree.slot_alpha, [4, 1, 1, 1])


def test_a_group_name_used_twice_is_refused():
    with pytest.raises(ValueError, match="group name 'g' is used twice"):
        P.DirichletTree({"g": [0, 1], "h": {"g": [2, 3], "i": [4]}})
