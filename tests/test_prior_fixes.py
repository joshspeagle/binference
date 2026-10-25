"""Tests for the prior fixes of audit P-4 to P-22 (DECISIONS #72)."""

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
    [-1.0, np.diag([1, 1, -5, 1, 1.0]), np.ones((4, 4)), np.array([[1.0, 2.0], [0.0, 1.0]])],
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
        (P.LogisticGP(), P.LogisticGP(boundary_factor=3.0)),
    ]
    for a, b in pairs:
        assert repr(a) != repr(b)
    assert P.Dirichlet.from_base_measure().alpha is None


def test_gamma_rates_have_a_prior_transform():
    b = P.GammaRates(shape=[0.5, 2.0], rate=[1.0, 3.0]).bind(Bins.categories(2))
    u = np.array([[0.1, 0.9], [0.5, 0.5]])
    expected = stats.gamma.ppf(u, a=[0.5, 2.0], scale=1 / np.array([1.0, 3.0]))
    np.testing.assert_allclose(b.prior_transform(u), expected, rtol=1e-10)
