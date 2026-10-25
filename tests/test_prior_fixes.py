"""Tests for the prior fixes of audit P-1 to P-22 (DECISIONS #72, #73)."""

import numpy as np
import pytest
from scipy import stats

from binference import Bins
from binference import priors as P
from binference.priors.hyper import Fixed, as_hyperprior


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
    b = P.LogisticGP(length_scale=np.array([0.3, 0.5])).bind(bins)  # an array, like a list
    assert [h.value for _, h in b.hypers[1:]] == [0.3, 0.5]


def test_hyperpriors_accept_numpy_scalars_but_not_booleans():
    assert as_hyperprior(np.array(2.0)).value == 2.0
    with pytest.raises(TypeError):
        as_hyperprior(True)


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


def test_gamma_rates_have_a_prior_transform():
    b = P.GammaRates(shape=[0.5, 2.0], rate=[1.0, 3.0]).bind(Bins.categories(2))
    u = np.array([[0.1, 0.9], [0.5, 0.5]])
    expected = stats.gamma.ppf(u, a=[0.5, 2.0], scale=1 / np.array([1.0, 3.0]))
    np.testing.assert_allclose(b.prior_transform(u), expected, rtol=1e-10)


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
        P.LogisticGP(length_scale=0.5)
        .bind(Bins.log(1.0, 1e4, 40))
        .sample(4000, np.random.default_rng(0))
    )
    steps = np.std(np.diff(np.log(r), axis=1), axis=0)
    # each SD has a Monte Carlo error of about 1.1%; the edges differ by the approximation
    np.testing.assert_allclose(steps[5:-5], np.median(steps), rtol=0.1)


@pytest.mark.parametrize("K", [2, 3, 5])
def test_a_gp_on_a_coarse_grid_gets_a_broad_length_scale_prior(K):
    h = dict(P.LogisticGP().bind(Bins.linear(0, 1, K)).hypers)["length_scale"]
    assert isinstance(h, P.LogNormal) and h.sigma == 1.0 and h.median() == pytest.approx(0.5)


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
