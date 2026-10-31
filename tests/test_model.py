"""Tests for PopulationModel (spec 3, 6.1)."""

import numpy as np
import pytest
import scipy.sparse as sp

import binference
from binference import Bins, Likelihoods, PopulationModel
from binference import priors as P


@pytest.fixture
def lk6(rng):
    bins = Bins.linear(0.0, 3.0, 6)
    return binference.simulate.mock_1d(80, bins, noise=(0.2, 0.6), rng=rng).likelihoods


def test_log_likelihood_direct(lk6, rng):
    m = PopulationModel(lk6)
    rho = rng.dirichlet(np.ones(6))
    L = lk6.toarray() * np.exp(lk6.log_scale)[:, None]
    assert m.log_likelihood(rho) == pytest.approx(np.log(L @ rho).sum(), rel=1e-12)
    assert m.log_likelihood(rho, absolute=False) == pytest.approx(
        np.log(lk6.toarray() @ rho).sum(), rel=1e-12
    )


def test_dense_sparse_weights_equivalence(lk6, rng):
    rho = rng.dirichlet(np.ones(6))
    m_dense = PopulationModel(lk6)
    m_sparse = PopulationModel(lk6.to_sparse())
    assert m_sparse.log_likelihood(rho) == pytest.approx(m_dense.log_likelihood(rho), rel=1e-12)
    # duplicating a row == weight 2
    doubled = Likelihoods(np.vstack([lk6.toarray(), lk6.toarray()[:1]]), lk6.bins)
    weighted = Likelihoods(lk6.toarray(), lk6.bins, weights=np.r_[2.0, np.ones(lk6.N - 1)])
    assert PopulationModel(doubled).log_likelihood(rho, absolute=False) == pytest.approx(
        PopulationModel(weighted).log_likelihood(rho, absolute=False), rel=1e-12
    )


def test_invariances(lk6, rng):
    rho1, rho2 = rng.dirichlet(np.ones(6), size=2)
    m = PopulationModel(lk6)
    d = m.log_likelihood(rho1) - m.log_likelihood(rho2)
    scaled = Likelihoods(lk6.toarray() * rng.uniform(0.1, 10, size=(lk6.N, 1)), lk6.bins)
    perm = Likelihoods(lk6.toarray()[rng.permutation(lk6.N)], lk6.bins)
    for other in (scaled, perm):
        mo = PopulationModel(other)
        assert mo.log_likelihood(rho1) - mo.log_likelihood(rho2) == pytest.approx(d, rel=1e-10)


GRAD_PRIORS = [
    P.Dirichlet(1.0),
    P.Dirichlet(0.7),
    P.GeneralizedDirichlet(1.5),
    P.PolyaTree(2.0),
    P.DirichletTree([[0, 1, 2], [3, [4, 5]]], alpha=2.0),
    P.LogisticNormal(),
    P.LogisticRandomWalk(order=2),
    P.LogisticGP(),
    P.GammaRates(1.5, 0.01),
]


@pytest.fixture
def lk_grid(rng):
    bins = Bins.grid(np.linspace(0.0, 1.0, 6), np.linspace(0.0, 1.0, 5))  # 5 x 4 cells
    return binference.simulate.mock_2d(80, bins, noise=(0.1, 0.3), rng=rng).likelihoods


# smoothness priors on a 2-D grid, with amplitude and per-axis length scales inferred
GRID_PRIORS = [P.LogisticRandomWalk(order=1), P.LogisticRandomWalk(order=2), P.LogisticGP()]
GRAD_CASES = [pytest.param(p, "lk6", id=repr(p)) for p in GRAD_PRIORS]
GRAD_CASES += [pytest.param(p, "lk_grid", id=f"2d-{p!r}") for p in GRID_PRIORS]


@pytest.mark.parametrize(("prior", "data"), GRAD_CASES)
@pytest.mark.parametrize("mode", [None, "on_measurement", "on_truth"])
def test_gradients_match_finite_differences(prior, data, mode, request, rng):
    lk = request.getfixturevalue(data)
    S = None if mode is None else np.linspace(0.9, 0.2, lk.bins.K)
    m = PopulationModel(lk, prior, selection=S, selection_mode=mode)
    for _ in range(2):
        x = m.from_value(m.prior.sample(1, rng)[0] * (lk.N if m.kind == "rates" else 1))
        x = x + rng.normal(scale=0.05, size=x.size)
        lp, g = m.log_density_unconstrained_and_grad(x)
        assert np.isfinite(lp)
        h = 1e-6
        fd = np.array(
            [
                (m.log_density_unconstrained(x + h * e) - m.log_density_unconstrained(x - h * e))
                / (2 * h)
                for e in np.eye(m.dim)
            ]
        )
        np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-4 * max(1.0, np.abs(fd).max()))


def test_round_trip_and_transform(lk6, rng):
    m = PopulationModel(lk6)
    rho = rng.dirichlet(np.ones(6))
    np.testing.assert_allclose(m.to_rho(m.from_value(rho)), rho, rtol=1e-10)
    X = rng.normal(size=(4, m.dim))
    vals = m.transform(X)
    assert vals.shape == (4, 6)
    np.testing.assert_allclose(vals.sum(1), 1.0)


def test_responsibilities_and_counts(lk6, rng):
    m = PopulationModel(lk6)
    rho = rng.dirichlet(np.ones(6))
    R = m.responsibilities(rho)
    np.testing.assert_allclose(R.sum(1), 1.0)
    np.testing.assert_allclose(m.expected_counts(rho), R.sum(0), rtol=1e-12)
    np.testing.assert_allclose(m.responsibilities(rho, idx=[3, 1]), R[[3, 1]])
    ms = PopulationModel(lk6.to_sparse())
    np.testing.assert_allclose(ms.responsibilities(rho), R, rtol=1e-12)


def test_selection_modes(lk6, rng):
    S = np.array([0.9, 0.7, 0.5, 0.4, 0.3, 0.2])
    rho = rng.dirichlet(np.ones(6))
    L = lk6.toarray()
    obs = PopulationModel(lk6, selection=S, selection_mode="on_measurement")
    tru = PopulationModel(lk6, selection=S, selection_mode="on_truth")
    W = lk6.N
    assert obs.log_likelihood(rho, absolute=False) == pytest.approx(
        np.log(L @ rho).sum() - W * np.log(S @ rho), rel=1e-12
    )
    assert tru.log_likelihood(rho, absolute=False) == pytest.approx(
        np.log((L * S) @ rho).sum() - W * np.log(S @ rho), rel=1e-12
    )
    with pytest.raises(ValueError, match="selection_mode"):
        PopulationModel(lk6, selection=S)
    # rates: Poisson normalisation e^{-S.lambda}
    rate = PopulationModel(lk6, P.GammaRates(), selection=S, selection_mode="on_measurement")
    lam = 50 * rho
    assert rate.log_likelihood(lam, absolute=False) == pytest.approx(
        np.log(L @ lam).sum() - S @ lam, rel=1e-12
    )


def test_selection_attached_to_likelihoods_is_used(lk6):
    S = np.linspace(0.3, 1.0, 6)
    m = PopulationModel(lk6.with_selection(S, "on_truth"))
    assert m.selection_mode == "on_truth"
    np.testing.assert_allclose(m.selection, S)
    # explicit selection overrides
    m2 = PopulationModel(
        lk6.with_selection(S, "on_truth"), selection=np.ones(6), selection_mode="on_measurement"
    )
    assert m2.selection_mode == "on_measurement"


def test_kkt_gap_and_missing_information(rng):
    bins = Bins.linear(0.0, 1.0, 4)
    onehot = Likelihoods(np.eye(4)[rng.integers(0, 4, size=200)], bins)
    m1 = PopulationModel(onehot)
    mle = binference.infer(onehot, method="map").rho
    assert m1.kkt_gap(mle) < 1e-8
    assert m1.missing_information(mle) < 1e-8
    assert m1.kkt_gap(np.array([0.7, 0.1, 0.1, 0.1])) > 1e-3
    broad = binference.simulate.mock_1d(500, bins, noise=2.0, rng=rng).likelihoods
    m2 = PopulationModel(broad)
    assert m2.missing_information(broad.stack()) > 0.9


def test_stack_is_one_em_step(lk6):
    m = PopulationModel(lk6)
    flat = lk6.bins.volumes / lk6.bins.volumes.sum()
    np.testing.assert_allclose(lk6.stack(), m.expected_counts(flat) / lk6.W, rtol=1e-12)


def test_raw_array_and_bins_checks():
    m = PopulationModel(np.array([[1.0, 2.0], [3.0, 1.0]]))
    assert m.bins.is_categorical and m.K == 2
    with pytest.raises(ValueError):
        PopulationModel(Likelihoods(np.ones((2, 2))), bins=Bins.linear(0, 1, 2))
    with pytest.raises(TypeError):
        PopulationModel(np.ones((2, 2)), prior="dirichlet")
    assert "PopulationModel(N=2" in repr(m)
    assert sp.issparse(PopulationModel(Likelihoods(np.ones((2, 2))).to_sparse())._L)


@pytest.mark.parametrize(
    "case", ["fractions", "rates", "selection", "logistic", "sparse", "joint"]
)
def test_many_points_take_the_log_density_of_one_point_at_a_time(lk6, case):
    from .test_injections import _problem

    lk, prior, kw = lk6, None, {}
    if case == "rates":
        prior = P.GammaRates(1.5, 1e-2)
    elif case == "selection":
        kw = {"selection": np.linspace(0.3, 1.0, 6), "selection_mode": "on_truth"}
    elif case == "logistic":
        prior = P.LogisticRandomWalk()
    elif case == "sparse":
        lk = lk6.to_sparse()
    elif case == "joint":
        lk, sel = _problem()
        kw = {"selection": sel}
    m = PopulationModel(lk, prior, **kw)
    X = np.random.default_rng(3).normal(0.0, 1.0, (40, m.dim))
    one = [m.log_density_unconstrained(x) for x in X]
    np.testing.assert_allclose(m._log_density_many(X), one, rtol=1e-10)


def test_the_batched_log_likelihood_is_minus_infinity_where_an_object_is_impossible():
    from binference import _kernels as kern

    L = np.random.default_rng(0).random((30, 4))
    L[5, 1:] = 0.0  # object 5 is only possible in bin 0
    V = np.random.default_rng(1).dirichlet(np.ones(4), 10)
    V[[2, 7], 0] = 0.0
    w = np.arange(1.0, 31.0)
    want = [kern.loglike(L, w, v) for v in V]
    for M in (L, sp.csr_array(L)):
        got = kern.loglike_many(M, w, V, block=3, chunk=7)
        np.testing.assert_allclose(got, want, rtol=1e-12)
    assert np.isneginf(got[[2, 7]]).all() and np.isfinite(np.delete(got, [2, 7])).all()


@pytest.mark.parametrize("case", ["selection", "response"])
def test_the_batched_log_likelihood_of_a_joint_model_is_that_of_one_point_at_a_time(case):
    from .test_injections import _problem
    from .test_joint_response import _unfolding

    if case == "selection":
        lk, sel = _problem()
        m = PopulationModel(lk, selection=sel)
    else:
        m = PopulationModel(_unfolding()[0])
    V = np.random.default_rng(4).dirichlet(np.ones(m.K), 20)
    one = [m.log_likelihood(v, absolute=False) for v in V]
    np.testing.assert_allclose(m._log_likelihood_many(V), one, rtol=1e-12)


def test_every_likelihood_kernel_is_minus_infinity_where_an_evidence_is_infinite():
    from binference import _kernels as kern

    L, w, rho = np.array([[1.0, 0.5], [0.3, 1.0]]), np.ones(2), np.array([np.inf, 0.5])
    assert kern.loglike(L, w, rho) == -np.inf
    assert kern.loglike_many(L, w, rho[None])[0] == -np.inf


def test_expected_counts_refuse_a_population_under_which_an_object_is_impossible():
    m = PopulationModel(np.array([[1.0, 0.0, 0.0], [0.2, 0.5, 0.3], [0.1, 0.9, 0.4]]))
    with pytest.raises(ValueError, match=r"1 object\(s\) zero likelihood \(first: \[0\]\)"):
        m.expected_counts(np.array([0.0, 0.5, 0.5]))


def test_missing_information_is_the_same_in_chunks(lk6):
    m = PopulationModel(lk6)
    rho = m.expected_counts(np.full(6, 1 / 6)) / m.W
    whole = m.missing_information(rho, chunk=None)
    assert m.missing_information(rho, chunk=7) == pytest.approx(whole, rel=1e-10)
