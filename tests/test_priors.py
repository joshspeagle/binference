"""Tests for population priors (spec 5)."""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy import integrate, stats

import binference
from binference import Bins
from binference import priors as P
from binference._simplex import helmert_basis, softmax
from binference.priors.hyper import Fixed, as_hyperprior

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
    assert repr(P.Dirichlet.reference()) == "Dirichlet.reference()"  # a prior one can build
    # the counts can carry alpha alone (base 0), as long as every bin has some
    np.testing.assert_allclose(P.Dirichlet.from_counts([3, 5, 2], base=0.0).alpha, [3, 5, 2])
    np.testing.assert_allclose(
        P.Dirichlet.from_counts([1, 0, 3], strength=0.5).resolve_alpha(B3), [1.5, 1.0, 2.5]
    )
    b = Bins([0.0, 1.0, 3.0])
    np.testing.assert_allclose(P.Dirichlet.from_base_measure(3.0).resolve_alpha(b), [1.0, 2.0])
    with pytest.raises(ValueError):
        P.Dirichlet(0.0)
    with pytest.raises(ValueError):
        P.Dirichlet([1.0, 2.0]).bind(B3)
    with pytest.raises(ValueError, match="Dirichlet alpha"):  # not NumPy's cast error
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


def test_a_tree_branch_whose_exponent_is_exactly_zero_keeps_the_density_bounded():
    # the branch into bins 2-4 has alpha 3 over 3 bins; a sum of coefficients gives -7e-16
    bound = P.GeneralizedDirichlet([5, 2, 1], [3, 2.9, 1.2]).bind(Bins.linear(0, 1, 4))
    assert not bound.unbounded_density()


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
    assert isinstance(as_hyperprior(3, "scale"), P.Fixed)
    assert as_hyperprior(np.array(2.0), "scale").value == 2.0
    for invalid in ("x", True):
        with pytest.raises(TypeError):
            as_hyperprior(invalid, "scale")


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
    gamma = rng.random(b.n_active)
    c = b.with_centering(gamma)
    assert np.all(b.gamma == 1.0)  # the original is untouched
    x = rng.normal(scale=0.5, size=b.dim)
    y = b.recentre(x, gamma)
    np.testing.assert_allclose(c.forward(y)[0], b.forward(x)[0], atol=1e-12)
    np.testing.assert_allclose(c.recentre(y, np.ones(b.n_active)), x, atol=1e-12)

    # log-density difference between parametrisations is the log-Jacobian
    # sum_j (1 - gamma_j) log s_j, whatever the field values
    def lp(bound, xx):
        eta, cache = bound.forward(xx)
        return bound.log_prior_and_grad(xx, cache, softmax(eta), np.zeros(6))[0]

    log_s, _ = b.log_scales(b.theta_from_x(x))
    np.testing.assert_allclose(lp(b, x) - lp(c, y), float((1 - gamma) @ log_s), atol=1e-10)
    x2 = x.copy()
    x2[: b.n_active] = rng.normal(size=b.n_active)
    log_s2, _ = b.log_scales(b.theta_from_x(x2))
    y2 = b.recentre(x2, gamma)
    np.testing.assert_allclose(lp(b, x2) - lp(c, y2), float((1 - gamma) @ log_s2), atol=1e-10)
    with pytest.raises(ValueError):
        b.with_centering(np.ones(b.n_active + 1))


def test_logistic_prior_survives_scale_underflow():
    """Extreme hyperparameters underflow some scales to zero without NaNs or warnings."""
    b = P.LogisticGP(kernel="rbf").bind(Bins.linear(0.0, 3.0, 12))
    gamma = np.r_[np.full(6, 0.5), np.ones(b.n_active - 6)]
    for prior in (b, b.with_centering(gamma)):
        x = np.zeros(prior.dim)
        x[prior.n_active :] = [-800.0, 800.0]  # absurd amplitude and length scale
        eta, cache = prior.forward(x)
        lp, g = prior.log_prior_and_grad(x, cache, softmax(eta), np.zeros(12))
        assert np.isfinite(lp) or lp == -np.inf
        assert not np.any(np.isnan(g))


def test_inverse_gamma_from_quantiles():
    for lower, upper in ((0.2, 3.0), (1.0, 1e5)):
        h = P.InverseGamma.from_quantiles(lower, upper)
        d = stats.invgamma(h.a, scale=h.b)
        assert d.cdf(lower) == pytest.approx(0.01, abs=1e-8)
        assert d.sf(upper) == pytest.approx(0.01, abs=1e-8)
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


def test_a_sparse_tree_gives_the_values_of_its_dense_membership_on_batched_inputs():
    bound = P.PolyaTree().bind(Bins.linear(0, 1, 256))
    tree = bound.tree
    assert sp.issparse(tree.slot_mem) and sp.issparse(tree.nonroot_mem)
    rho = np.random.default_rng(0).dirichlet(np.ones(256), (2, 3))
    density = bound.log_density(rho)
    assert density.shape == (2, 3)
    flat = np.array([bound.log_density(r) for r in rho.reshape(-1, 256)])
    np.testing.assert_allclose(density.ravel(), flat, rtol=1e-12)
    dense = rho @ tree.nonroot_mem.toarray().T
    np.testing.assert_allclose(
        rho.reshape(-1, 256) @ tree.nonroot_mem.T, dense.reshape(6, -1), rtol=1e-12
    )


def test_a_small_tree_stays_dense():
    assert isinstance(P.PolyaTree().bind(Bins.linear(0, 1, 8)).tree.slot_mem, np.ndarray)


def test_a_generalized_dirichlet_on_a_grid_needs_an_order():
    grid = Bins.grid(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
    with pytest.raises(ValueError, match="order="):
        P.GeneralizedDirichlet().bind(grid)
    assert P.GeneralizedDirichlet(order=[3, 1, 0, 2]).bind(grid).dim == 3


def test_a_gp_prior_works_on_two_bins():
    mock = binference.simulate.mock_1d(200, Bins.linear(0, 3, 2), rng=4)
    est = binference.infer(mock.likelihoods, P.LogisticGP(), method="map")
    assert np.isfinite(est.value).all()


def test_a_tree_takes_one_alpha_array_per_node_of_any_arity():
    tree = [[0, 1, 2], [3, [4, 5]]]
    alpha = [[1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 2.0], [1.0, 1.0]]  # pre-order
    for form in (alpha, tuple(alpha), [np.array(a) for a in alpha]):
        b = P.DirichletTree(tree, alpha=form).bind(Bins.categories(6))
        assert b.sample(3, np.random.default_rng(0)).shape == (3, 6)
    with pytest.raises(ValueError, match="node 1"):
        P.DirichletTree(tree, alpha=[[1, 1], [1, 1], [1, 1], [1, 1]])
    with pytest.raises(ValueError, match="positive"):
        P.DirichletTree(tree, alpha=-1.0)


def test_the_trend_scale_means_the_same_whatever_the_number_of_bins():
    # SD of the end-to-end change of log rho from the linear mode: 3.46 * t * sqrt((K-1)/(K+1))
    # (about 3.1 t at K = 10 and 3.4 t at K = 200); it fell as 1/sqrt(K) before
    rng = np.random.default_rng(0)
    sd = {}
    for K in (10, 200):
        prior = P.LogisticRandomWalk(2, scale=Fixed(1e-6), trend_scale=1.0)
        r = prior.bind(Bins.linear(0, 1, K)).sample(4000, rng)
        sd[K] = np.std(np.log(r[:, -1] / r[:, 0]))
    # each SD has a relative Monte Carlo error of about 1.1%
    np.testing.assert_allclose(sd[10], 3.46 * np.sqrt(9 / 11), rtol=0.05)
    np.testing.assert_allclose(sd[200], 3.46 * np.sqrt(199 / 201), rtol=0.05)


@pytest.mark.parametrize(
    "cov",
    [
        -1.0,
        np.diag([1, 1, -5, 1, 1.0]),
        *(2.0 * np.ones((k, k)) for k in (3, 4, 5)),  # constant: no field at all
        np.array([[1.0, 2.0], [0.0, 1.0]]),
    ],
)
def test_a_logistic_normal_refuses_an_invalid_covariance(cov):
    with pytest.raises(ValueError, match="cov"):
        P.LogisticNormal(cov=cov)


def test_a_logistic_normal_covariance_must_match_the_bins():
    with pytest.raises(ValueError, match="K=5"):
        P.LogisticNormal(cov=np.eye(4)).bind(Bins.linear(0, 1, 5))


def test_a_polya_tree_needs_two_bins_and_a_sensible_power():
    with pytest.raises(ValueError, match="two bins"):
        P.PolyaTree().bind(Bins.linear(0, 1, 1))
    with pytest.raises(ValueError, match="power"):
        P.PolyaTree(power=-1.0)


@pytest.mark.parametrize("kwargs", [{"n_basis": 0}, {"n_basis": 2.7}, {"boundary_factor": np.nan}])
def test_gp_arguments_are_checked_when_the_prior_is_made(kwargs):
    with pytest.raises(ValueError):
        P.LogisticGP(**kwargs)


def test_gp_per_axis_arguments_must_match_the_axes():
    bins = Bins.grid([np.linspace(0, 1, 6), np.linspace(0, 1, 5)])
    with pytest.raises(ValueError, match="n_basis"):
        P.LogisticGP(n_basis=[4, 4, 4]).bind(bins)
    b = P.LogisticGP(length_scale=np.array([1.0, 1.5])).bind(bins)  # an array, like a list
    assert [h.value for _, h in b.hypers[1:]] == [1.0, 1.5]


@pytest.mark.parametrize(
    ("make", "name"),
    [
        (lambda: P.LogisticGP(amplitude=-1.0), "amplitude"),
        (lambda: P.LogisticRandomWalk(scale=0.0), "scale"),
        (lambda: P.LogisticNormal(scale="x"), "scale"),
        (lambda: P.LogisticGP(length_scale=np.inf), "length_scale"),
        (lambda: P.LogisticGP(length_scale=[1.0, -2.0]), r"length_scale\[1\]"),
    ],
    ids=["amplitude", "scale", "scale-type", "length-scale", "length-scale-per-axis"],
)
def test_a_bad_hyperparameter_is_named_in_the_error(make, name):
    with pytest.raises((TypeError, ValueError), match=rf"^{name} must be"):
        make()


def test_a_dirichlet_from_counts_names_a_bad_argument():
    # a negative strength subtracted the reference sample; a bad base blamed alpha (a zero
    # base is valid where every bin has counts)
    for kw, name in (({"strength": -0.1}, "strength"), ({"base": -0.5}, "base")):
        with pytest.raises(ValueError, match=rf"^{name} must be"):
            P.Dirichlet.from_counts([1.0, 2.0], **kw)
    with pytest.raises(ValueError, match=r"^counts must be"):
        P.Dirichlet.from_counts([1.0, np.nan])


def test_generalized_dirichlet_arguments_are_checked():
    with pytest.raises(ValueError, match="positive"):
        P.GeneralizedDirichlet(a=[1.0, -1.0], b=1.0)
    with pytest.raises(ValueError, match="b must be a scalar or have length 3"):
        P.GeneralizedDirichlet(a=1.0, b=[1.0, 2.0]).bind(Bins.categories(4))


def test_tree_vb_draws_have_no_exact_zeros():
    b = P.PolyaTree(c=0.001).bind(Bins.linear(0, 1, 64))  # products of tiny branches underflow
    draws = b.vb_sample(b.tree.slot_alpha, 200, np.random.default_rng(0))
    assert draws.min() > 0


def test_every_argument_that_changes_a_prior_shows_in_its_repr():
    pairs = [
        (P.PolyaTree(), P.PolyaTree(base=[1.0, 2.0, 3.0])),
        (P.LogisticNormal(), P.LogisticNormal(mean=[0.0, 1.0, 2.0])),
        (P.LogisticRandomWalk(), P.LogisticRandomWalk(trend_scale=1.0)),
        (P.LogisticGP(), P.LogisticGP(n_basis=10)),
        (P.LogisticGP(), P.LogisticGP(boundary_factor=2.0)),
    ]
    for a, b in pairs:
        assert repr(a) != repr(b)
    assert P.Dirichlet.from_base_measure().alpha is None


def test_a_custom_prior_has_a_repr_without_matching_attributes():
    class Mine(P.LogisticNormal):
        def __init__(self, width=1.0):
            super().__init__(cov=width)

    class Bare(P.PopulationPrior):
        pass

    assert repr(Mine()).startswith("Mine(") and repr(Bare()) == "Bare()"


def test_priors_draw_zero_samples():
    for prior in (P.Dirichlet(), P.PolyaTree(), P.LogisticNormal(), P.GammaRates()):
        assert prior.bind(Bins.linear(0, 1, 5)).sample(0, np.random.default_rng(0)).shape == (
            0,
            5,
        )


def test_more_arguments_are_checked_when_the_prior_is_made():
    with pytest.raises(ValueError, match="order"):
        P.GeneralizedDirichlet(order=[0, 2, 2])
    with pytest.raises(ValueError, match="mean"):
        P.LogisticNormal(mean=[0.0, np.nan, 1.0])
    with pytest.raises(ValueError, match="concentrations"):
        P.PolyaTree(power=400.0).bind(Bins.linear(0, 1, 64))


def test_a_gp_on_log_bins_is_stationary_in_log_z():
    # equal steps in log z have equal prior spread; in z the low bins were frozen together
    r = (
        P.LogisticGP(length_scale=1.0)
        .bind(Bins.log(1.0, 1e4, 40))
        .sample(4000, np.random.default_rng(0))
    )
    steps = np.std(np.diff(np.log(r), axis=1), axis=0)
    # each SD has a Monte Carlo error of about 1.1%; the edges differ by the approximation
    np.testing.assert_allclose(steps[5:-5], np.median(steps), rtol=0.1)


@pytest.mark.parametrize("K", [2, 3, 5])
def test_a_gp_on_a_coarse_grid_gets_a_broad_length_scale_prior(K):
    h = dict(P.LogisticGP().bind(Bins.linear(0, 1, K)).hypers)["length_scale"]
    # centred on half the range, or on the shortest resolved length if that is longer
    assert isinstance(h, P.LogNormal) and h.sigma == 1.0 and h.median() >= 0.5


def test_centred_gp_gradients_agree_with_the_density_where_scales_underflow():
    # RBF scales of high-frequency modes underflow; the value used to be floored but not the
    # gradient (audit P-5)
    b = P.LogisticGP(kernel="rbf").bind(Bins.linear(0, 1, 48))
    b = b.with_centering(np.zeros(b.n_active))
    rng = np.random.default_rng(1)
    u = np.log([1.0, 2.0])
    log_s = b.log_scales(np.exp(u))[0]
    assert log_s.min() < np.log(1e-150)  # below the old floor
    # centred coefficients drawn at their own scales, so the density is finite
    x = np.concatenate([np.exp(log_s) * rng.normal(size=b.n_active), u])

    def f(xx):
        eta, cache = b.forward(xx)
        return b.log_prior_and_grad(xx, cache, np.exp(eta), np.zeros(48))

    g = f(x)[1]
    for i in (b.n_active, b.n_active + 1):  # amplitude and length scale
        e = np.zeros_like(x)
        e[i] = 1e-6
        fd = (f(x + e)[0] - f(x - e)[0]) / 2e-6
        np.testing.assert_allclose(g[i], fd, rtol=1e-5)


def test_the_default_gp_basis_has_one_sine_per_bin_up_to_1024_modes():
    assert P.LogisticGP().bind(Bins.linear(0, 1, 12)).n_active == 12
    grid = Bins.grid(np.linspace(0, 1, 41), np.linspace(0, 1, 11))
    assert P.LogisticGP().bind(grid).n_active == 32 * 10  # 32 per axis in 2-D


def test_the_wider_gp_box_keeps_the_edges_free_at_long_length_scales():
    # at ell = half the range the box of 1.5 half-widths pulled the edge bins to the mean
    # (edge SD 0.68-0.80 of exact, audit P-3); the default box keeps them within 5%
    K, ell = 40, 0.5
    b = P.LogisticGP(kernel="matern32", amplitude=1.0, length_scale=ell).bind(Bins.linear(0, 1, K))
    s = np.exp(b.log_scales(np.array([1.0, ell]))[0])
    B = b.basis.factors[0]
    Pm = np.eye(K) - 1.0 / K
    hs = Pm @ (B * s**2) @ B.T @ Pm
    r = np.abs(np.subtract.outer(b.bins.centers, b.bins.centers)) / ell
    exact = Pm @ ((1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)) @ Pm
    np.testing.assert_allclose(np.diag(hs)[[0, -1]], np.diag(exact)[[0, -1]], rtol=0.05)


def test_a_fixed_gp_length_scale_below_the_resolution_warns():
    with pytest.warns(binference.InputWarning, match="n_basis"):
        P.LogisticGP(length_scale=0.05).bind(Bins.linear(0, 1, 40))


def test_recentring_a_gp_from_centred_coordinates_stays_finite():
    b = P.LogisticGP(kernel="rbf").bind(Bins.linear(0, 1, 48))
    c = b.with_centering(np.zeros(b.n_active))
    u = np.log([1.0, 2.0])
    log_s = c.log_scales(np.exp(u))[0]
    x = np.concatenate([np.exp(log_s) * np.random.default_rng(2).normal(size=b.n_active), u])
    y = c.recentre(x, np.ones(b.n_active))  # warnings are errors here
    assert np.all(np.isfinite(y))


def test_gp_binds_on_log_and_large_grids():
    for bins in (Bins.log(0.01, 10, 20), Bins.linear(0, 1, 3000)):
        P.LogisticGP().bind(bins)


def test_logistic_prior_mean_is_a_flat_density():
    bins = Bins([0.0, 0.1, 0.5, 2.0])
    for prior in (P.LogisticRandomWalk(), P.LogisticGP(), P.LogisticNormal()):
        mu = prior.bind(bins).mu
        lv = np.log(bins.volumes)
        np.testing.assert_allclose(mu, lv - lv.mean())


def test_a_generalized_dirichlet_from_a_base_measure_has_its_masses_as_mean(seed):
    base = np.array([1.0, 3.0, 2.0, 5.0, 1.0, 4.0])
    prior = P.GeneralizedDirichlet.from_base_measure(3.0, base, order=[5, 0, 4, 1, 3, 2])
    draws = prior.bind(Bins.linear(0, 1, 6)).sample(40_000, np.random.default_rng(seed))
    se = draws.std(0) / np.sqrt(len(draws))
    np.testing.assert_array_less(np.abs(draws.mean(0) - base / base.sum()), 4 * se)


@pytest.mark.parametrize("conc", [2.0, [1.0, 2.0, 3.0, 4.0]])
def test_every_stick_of_a_generalized_dirichlet_from_a_base_measure_has_its_concentration(conc):
    prior = P.GeneralizedDirichlet.from_base_measure(conc, [1.0, 1e-17, 2.0, 3.0, 1e-20])
    tree = prior.bind(Bins.linear(0, 1, 5)).tree
    sticks = np.add.reduceat(tree.slot_alpha, tree.node_start[:-1])
    np.testing.assert_allclose(sticks, np.broadcast_to(conc, 4), rtol=1e-12)
    assert np.all(tree.slot_alpha > 0)  # no stick loses its b to round-off


def test_a_generalized_dirichlet_from_a_base_measure_checks_its_arguments():
    with pytest.raises(ValueError, match="concentration"):
        P.GeneralizedDirichlet.from_base_measure(0.0)
    with pytest.raises(ValueError, match="positive mass"):
        P.GeneralizedDirichlet.from_base_measure(2.0, [1.0, 0.0, 1.0]).bind(Bins.linear(0, 1, 3))
    with pytest.raises(ValueError, match="one value per stick"):
        P.GeneralizedDirichlet.from_base_measure(np.ones((2, 2)))
    assert repr(P.GeneralizedDirichlet.from_base_measure(2)) == repr(
        P.GeneralizedDirichlet.from_base_measure(2.0)
    )


def test_a_first_order_random_walk_steps_in_the_measure_coordinate(seed):
    # on bins unequal in u, a neighbour difference has variance in proportion to its spacing
    edges = np.array([0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
    bins = Bins(edges)
    prior = P.LogisticRandomWalk(order=1, scale=Fixed(1.0))
    r = prior.bind(bins).sample(20_000, np.random.default_rng(seed))
    log_density = np.log(r / np.diff(edges))
    centres = 0.5 * (edges[1:] + edges[:-1])
    ratio = np.var(np.diff(log_density, axis=1), axis=0) / np.diff(centres)
    # each variance has a Monte Carlo error of about 1% (sqrt(2 / 20000))
    np.testing.assert_allclose(ratio, np.mean(ratio), rtol=0.06)


def test_a_second_order_random_walk_leaves_lines_in_u_unpenalised():
    from binference.priors.logistic import _rw_differences, _rw_steps

    x = _rw_steps(np.array([0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]))
    D = _rw_differences(x, 2)
    np.testing.assert_allclose(D @ np.ones(6), 0.0, atol=1e-12)
    np.testing.assert_allclose(D @ x, 0.0, atol=1e-12)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_a_random_walk_on_equal_bins_is_the_one_of_unit_steps(order):
    from binference.priors.logistic import _rw_differences, _rw_steps

    x = _rw_steps(Bins.log(1.0, 100.0, 9).u_edges[0])
    np.testing.assert_array_equal(x, np.arange(9.0))
    np.testing.assert_allclose(
        _rw_differences(x, order), np.diff(np.eye(9), n=order, axis=0), atol=1e-12
    )


def test_a_random_walk_on_very_unequal_bins_has_finite_scales():
    with pytest.warns(binference.InputWarning, match="geometrically spaced"):
        bins = Bins(np.geomspace(1e-2, 1e4, 49))  # widths spanning six decades, linear axis
    for order in (1, 2, 3):
        bound = P.LogisticRandomWalk(order=order).bind(bins)
        log_s, _ = bound.log_scales(np.ones(len(bound.hypers)))
        assert np.all(np.isfinite(log_s))


def test_a_dirichlet_with_one_concentration_warns_on_bins_of_unequal_width():
    unequal = Bins([0.0, 0.1, 0.3, 1.0])
    for prior in (P.Dirichlet(1.0), P.Dirichlet.reference()):
        with pytest.warns(binference.InputWarning, match="from_base_measure"):
            bound = prior.bind(unequal)
        assert np.ptp(bound.alpha) == 0.0  # still one concentration for every bin
    # equal widths, an array alpha or a base measure stay silent
    P.Dirichlet(1.0).bind(Bins.linear(1000.0, 1001.0, 50))
    P.Dirichlet([1.0, 2.0, 3.0]).bind(unequal)
    P.Dirichlet.from_base_measure(3.0).bind(unequal)
    with pytest.warns(binference.InputWarning, match="GammaRates"):
        P.GammaRates(1.0).bind(unequal)
    P.GammaRates([0.1, 0.2, 0.7]).bind(unequal)


def test_the_default_gp_basis_resolves_the_narrowest_bin_up_to_four_per_bin():
    # 0.25-wide bins on a range of 3 need 12 sines for 6 bins
    assert P.LogisticGP().bind(Bins([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])).basis.J == 12
    # 0.1-wide bins on a range of 3.2 would need 32: capped at 4 per bin, 24
    assert P.LogisticGP().bind(Bins([0.0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])).basis.J == 24
    # equal bins keep one per bin
    assert P.LogisticGP().bind(Bins.log(1.0, 100.0, 16)).basis.J == 16
