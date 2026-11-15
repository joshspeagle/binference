"""Engine tests on small problems with exact references (test plan 2.6)."""

import pickle
import time
import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from scipy import optimize

import binference
from binference import (
    Bins,
    ConvergenceWarning,
    Likelihoods,
    PopulationModel,
    Selection,
)
from binference import _kernels as kern
from binference import priors as P
from binference.engines import MAP, NUTS, Bootstrap, Gibbs, Laplace, MeanFieldVB
from binference.engines._nuts_core import _MAX_ENERGY_ERROR, Metric, _leaf, adaptation_windows
from binference.engines.nuts import _Chain
from binference.warnings import ApproximationWarning, InputWarning

from .conftest import assert_draws_agree, assert_matches_reference, small_mock

GAP = 1e3 * 1e-8  # MAP's guarantee: tol = 1e-8, or 1e3 * tol once round-off stops it


@pytest.fixture(scope="module")
def onehot():
    rng = np.random.default_rng(5)
    bins = Bins.linear(0.0, 1.0, 4)
    labels = rng.choice(4, size=60, p=[0.1, 0.2, 0.3, 0.4])
    lk = Likelihoods(np.eye(4)[labels], bins)
    counts = np.bincount(labels, minlength=4).astype(float)
    return lk, counts


def dirichlet_moments(a):
    A = a.sum()
    return a / A, np.sqrt(a * (A - a) / (A**2 * (A + 1)))


# --------------------------------------------------------------------- NUTS
@pytest.mark.parametrize("method", ["nuts", "gibbs"])
def test_bin_without_likelihood_support_is_not_frozen(method, seed):
    """A bin no object can be in keeps its prior-driven mass alpha_k / (A + N) (frankenz F11)."""
    rng = np.random.default_rng(3)
    N, K = 300, 5
    L = rng.random((N, K)) + 0.1
    L[:, -1] = 0.0
    lk = Likelihoods(L, Bins.linear(0, 1, K))
    res = binference.infer(lk, method=method, rng=seed + 1, n_samples=1000, check=False)
    exact_mean = 1.0 / (K + N)  # rho_K | D ~ Beta(1, K - 1 + N) under Dir(1)
    assert np.all(res.samples[:, -1] > 0)
    assert res.mean[-1] == pytest.approx(exact_mean, rel=0.15)


def test_nuts_matches_exact(k3_problem, seed):
    mock, exact = k3_problem
    res = binference.infer(
        mock.likelihoods,
        method="nuts",
        rng=seed + 1,
        n_chains=2,
        n_warmup=400,
        n_samples=1500,
        check=False,
    )
    assert res.method == "nuts" and not res.approximate
    assert_matches_reference(res, exact.mean, exact.std)
    for k in range(3):
        q = exact.quantile([0.05, 0.95], k)
        np.testing.assert_allclose(np.quantile(res.samples[:, k], [0.05, 0.95]), q, atol=0.02)


@pytest.mark.parametrize("metric", ["diag", "dense"])
def test_nuts_onehot_is_conjugate(onehot, metric, seed):
    lk, counts = onehot
    alpha = np.array([1.0, 2.0, 0.5, 1.5])
    res = binference.infer(
        lk,
        P.Dirichlet(alpha),
        method="nuts",
        rng=seed + 2,
        n_chains=2,
        n_warmup=300,
        n_samples=1500,
        metric=metric,
        check=False,
    )
    mean, sd = dirichlet_moments(alpha + counts)
    assert_matches_reference(res, mean, sd)


def test_nuts_reproducible_resumable_and_generator(onehot, seed):
    lk, _ = onehot
    m = PopulationModel(lk)
    a = NUTS(m, n_chains=2, n_warmup=50, rng=seed + 11).run(30)
    b = NUTS(m, n_chains=2, n_warmup=50, rng=seed + 11).run(30)
    np.testing.assert_array_equal(a.samples, b.samples)
    c = NUTS(m, n_chains=2, n_warmup=50, rng=seed + 12).run(30)
    assert not np.array_equal(a.samples, c.samples)
    eng = NUTS(m, n_chains=2, n_warmup=50, rng=seed + 3)
    eng.run(20)
    res = eng.run(15)
    assert res.chains.shape == (2, 35, 4)
    shapes = [x.shape for x in eng.sample(3)]
    assert shapes == [(2, 3)] * 3
    assert eng.result.chains.shape == (2, 38, 4)
    info = res.info
    for key in ("accept_stat", "tree_depth", "diverging", "n_leapfrog", "energy", "step_size"):
        assert key in info


@pytest.mark.parametrize("engine", [NUTS, Gibbs])
def test_resuming_is_bitwise_identical_to_one_long_run(engine, onehot, seed):
    m = PopulationModel(onehot[0])
    e1 = engine(m, n_chains=2, n_warmup=30, rng=seed + 4)
    e1.run(20)
    split = e1.run(15)
    whole = engine(m, n_chains=2, n_warmup=30, rng=seed + 4).run(35)
    np.testing.assert_array_equal(split.chains, whole.chains)


def test_parallel_chains_match_serial(onehot, seed):
    m = PopulationModel(onehot[0], P.LogisticRandomWalk(order=1))
    serial = NUTS(m, n_chains=2, n_warmup=40, rng=seed + 5).run(15)
    eng = NUTS(m, n_chains=2, n_warmup=40, rng=seed + 5, n_jobs=2)
    parallel = eng.run(15)
    np.testing.assert_array_equal(serial.chains, parallel.chains)
    np.testing.assert_array_equal(serial.info["n_leapfrog"], parallel.info["n_leapfrog"])
    assert eng.run(5).chains.shape == (2, 20, 4)  # resuming works across processes
    with pytest.raises(ValueError):
        NUTS(m, n_jobs=0)


def test_hessian_helpers():
    from binference.engines._curvature import covariance_from_hessian, fd_hessian

    A = np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.2], [0.0, 0.2, 3.0]])

    def fn(x):
        return -0.5 * float(x @ A @ x), -A @ x

    H = fd_hessian(fn, np.array([0.3, -1.0, 2.0]))
    np.testing.assert_allclose(H, -A, atol=1e-6)
    np.testing.assert_allclose(covariance_from_hessian(H), np.linalg.inv(A), rtol=1e-6)
    cols = fd_hessian(fn, np.zeros(3), cols=[1])
    np.testing.assert_allclose(cols[:, 0], -A[:, 1], atol=1e-6)
    # indefinite curvature is clipped, not inverted
    cov = covariance_from_hessian(np.diag([-1.0, 0.5]))
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    assert covariance_from_hessian(np.diag([1.0, 2.0])) is None
    # a flat or negatively curved direction gets the largest variance seen, not 1e6 of them
    for flat in (0.5, 0.0, -1e-9):
        cov = covariance_from_hessian(np.diag([-4.0, -1.0, flat]))
        np.testing.assert_allclose(np.diag(cov), [0.25, 1.0, 1.0])
    assert covariance_from_hessian(np.full((2, 2), np.nan)) is None


def test_nuts_reparametrises_logistic_priors_without_touching_the_model(seed):
    mock = binference.simulate.mock_1d(5000, Bins.linear(0, 3, 8), noise=(0.05, 0.1), rng=seed + 2)
    m = PopulationModel(mock.likelihoods, P.LogisticRandomWalk(order=1))
    eng = NUTS(m, n_chains=2, n_warmup=150, rng=seed + 1)
    assert eng.metric == "dense" and eng.target_accept == 0.9
    res = eng.run(40)
    assert np.all(m.prior.gamma == 1.0)  # the user's model keeps its parametrisation
    assert eng.model is not m
    assert not np.all(eng.model.prior.gamma == 1.0)  # strong data: some modes centred
    # results are in the user's (non-centred) coordinates (audit N-1)
    assert res.model is m and np.array_equal(res.info["centring"], eng.model.prior.gamma)
    np.testing.assert_allclose(m.to_value(res.x[0, -1]), res.chains[0, -1])
    np.testing.assert_allclose(
        res.log_density_unconstrained[0, -5:],
        [m.log_density_unconstrained(x) for x in res.x[0, -5:]],
        rtol=1e-10,
    )
    plain = NUTS(m, n_chains=2, n_warmup=60, rng=seed + 1, reparametrize=False, metric="diag")
    assert plain.metric == "diag" and plain.run(5).model.prior is m.prior
    dirichlet = NUTS(PopulationModel(mock.likelihoods), n_chains=1)
    assert dirichlet.metric == "diag" and dirichlet.target_accept == 0.8


def test_nuts_validation():
    m = PopulationModel(np.ones((3, 2)))
    with pytest.raises(TypeError):
        NUTS("not a model")
    with pytest.raises(ValueError):
        NUTS(m, n_chains=0)
    with pytest.raises(ValueError):
        NUTS(m, metric="full")
    with pytest.raises(ValueError):
        NUTS(m, target_accept=1.5)
    with pytest.raises(ValueError, match="init_x"):
        NUTS(m, init_x=np.zeros(5))  # its shape when the engine is made
    with pytest.raises(ValueError, match="zero posterior density"):  # its density when run
        NUTS(PopulationModel(Likelihoods(np.eye(2))), init_x=[1e3]).run(1)
    with pytest.raises(ValueError, match="fractions or rates of shape"):
        NUTS(m, init=np.full(m.dim, 0.2))  # a point in unconstrained coordinates
    with pytest.raises(ValueError, match="not both"):
        NUTS(m, init=[0.5, 0.5], init_x=[0.0])
    with pytest.raises(RuntimeError):
        _ = NUTS(m).result


def test_nuts_backs_off_a_jitter_that_leaves_the_support_from_init_as_from_the_mode(seed):
    # this model's density is zero beyond |x| of about 700, so jitter 1e4 leaves it
    nuts = NUTS(
        PopulationModel(Likelihoods(np.eye(2))), init=[0.5, 0.5], jitter=1e4, n_chains=4, rng=seed
    )
    starts = nuts._initial_points()
    assert np.all(np.abs(starts) < 700)


def test_nuts_chains_start_at_a_given_init_jittered_only_when_it_is_one_point(k3_problem, seed):
    m = PopulationModel(k3_problem[0].likelihoods)  # its mode is near (1.3, -0.4)
    starts = np.array([[2.0, -2.0], [-2.0, 2.0]])
    eng = NUTS(m, n_chains=2, n_warmup=0, init_x=starts, rng=seed)
    eng.warmup()
    np.testing.assert_array_equal([c.x for c in eng._chains], starts)
    # init is fractions, as for the other engines: the same point, converted
    v = m.to_value(starts[0])
    eng = NUTS(m, n_chains=2, n_warmup=0, init=v, jitter=0.0, rng=seed)
    eng.warmup()
    np.testing.assert_allclose([c.x for c in eng._chains], [starts[0]] * 2, atol=1e-12)
    rates = PopulationModel(k3_problem[0].likelihoods, P.GammaRates(1.0, 1e-3))
    lam = np.array([30.0, 70.0, 5.0])  # rates above 1, whose logs are positive too
    x = NUTS(rates, n_chains=1, init=lam, jitter=0.0)._initial_points()[0]
    np.testing.assert_allclose(rates.to_value(x), lam)
    eng = NUTS(m, n_chains=2, n_warmup=0, init_x=starts[0], jitter=0.1, rng=seed)
    eng.warmup()
    x = np.array([c.x for c in eng._chains])
    assert np.all(np.abs(x - starts[0]) < 0.6) and not np.allclose(x[0], x[1])


def test_nuts_with_a_logistic_prior_matches_a_long_run(seed):
    mock = binference.simulate.mock_1d(300, Bins.linear(0, 3, 8), rng=4)  # the reference's data
    res = binference.infer(
        mock.likelihoods,
        P.LogisticRandomWalk(order=2),
        method="nuts",
        rng=seed + 1,
        n_chains=2,
        n_warmup=200,
        n_samples=200,
        check=False,
    )
    assert res.samples.shape == (400, 8)
    assert "scale" in res.info["hyperparameters"]
    # Reference: the same call with rng=seed + 99, 4 chains x 10000 draws after 1000 warm-up
    # (ESS > 46000 per bin, R-hat < 1.0002), so its own MC error is negligible here;
    # infer(mock.likelihoods, P.LogisticRandomWalk(order=2), rng=seed + 99, n_chains=4,
    #       n_warmup=1000, n_samples=10000) -> .mean, .std, rounded to 6 digits
    mean = [0.028023, 0.261029, 0.163405, 0.175175, 0.270258, 0.091127, 0.009948, 0.001036]
    sd = [0.013765, 0.031731, 0.030073, 0.028615, 0.031246, 0.020168, 0.005868, 0.00146]
    # sd_mcse underestimates the error of an sd by up to 1.3x on the skewed edge bins of
    # runs this short (40 seeds), so z = 5 is about 3.8 true sigma there
    assert_matches_reference(res, mean, sd)


# -------------------------------------------------------------------- Gibbs
def test_gibbs_matches_exact(k3_problem, seed):
    mock, exact = k3_problem
    res = binference.infer(
        mock.likelihoods, method="gibbs", rng=seed + 3, n_samples=3000, n_warmup=200, check=False
    )
    assert_matches_reference(res, exact.mean, exact.std)


def test_gibbs_keeps_the_last_draw_of_every_thinning_block_of_each_chain(k3_problem, seed):
    m = PopulationModel(k3_problem[0].likelihoods)
    thinned = Gibbs(m, n_chains=2, n_warmup=20, thin=3, rng=seed + 4).run(10)
    every = Gibbs(m, n_chains=2, n_warmup=20, rng=seed + 4).run(30)
    np.testing.assert_array_equal(thinned.chains, every.chains[:, 2::3])


def test_gibbs_onehot_is_conjugate(onehot, seed):
    lk, counts = onehot
    res = binference.infer(
        lk, method="gibbs", rng=seed + 4, n_samples=3000, n_warmup=10, check=False
    )
    mean, sd = dirichlet_moments(1.0 + counts)
    assert_matches_reference(res, mean, sd)


def test_gibbs_selection_matches_exact(k3_selection_problem, seed):
    mock, exact = k3_selection_problem
    res = binference.infer(
        mock.likelihoods, method="gibbs", rng=seed + 5, n_samples=3000, n_warmup=200, check=False
    )
    assert_matches_reference(res, exact.mean, exact.std)
    nuts = binference.infer(
        mock.likelihoods,
        method="nuts",
        rng=seed + 5,
        n_chains=2,
        n_warmup=300,
        n_samples=1500,
        check=False,
    )
    assert_matches_reference(nuts, exact.mean, exact.std)


@pytest.mark.parametrize("mode", ["on_truth", "on_measurement"])
def test_fraction_selection_only_depends_on_ratios(k3_selection_problem, mode, seed):
    """S and 3 S (values above one) give the same posterior for fractions."""
    mock, _ = k3_selection_problem
    lk = mock.likelihoods.with_selection(None)
    S = np.array([0.9, 0.5, 0.25])
    fits = {}
    for scale in (1.0, 3.0):
        m = PopulationModel(lk, selection=Selection(scale * S, mode, scale="relative"))
        est = MAP(m).run()
        gibbs = Gibbs(m, n_chains=2, n_warmup=200, rng=seed + 3).run(1500)
        fits[scale] = (est.rho, gibbs)
        assert np.all(gibbs.samples > 0)
    np.testing.assert_allclose(fits[3.0][0], fits[1.0][0], rtol=1e-6)
    np.testing.assert_allclose(fits[3.0][1].samples, fits[1.0][1].samples, rtol=1e-12)
    exact_m = PopulationModel(lk, selection=S, selection_mode=mode)
    nuts = NUTS(exact_m, n_chains=2, n_warmup=300, rng=seed + 4).run(800)
    assert_draws_agree(fits[3.0][1].chains, nuts.chains)


def test_gibbs_rates_agree_with_nuts(k3_selection_problem, seed):
    mock, _ = k3_selection_problem
    prior = P.GammaRates(1.0, 1e-3)
    g = binference.infer(
        mock.likelihoods, prior, method="gibbs", rng=seed + 6, n_samples=3000, check=False
    )
    n = binference.infer(
        mock.likelihoods,
        prior,
        method="nuts",
        rng=seed + 6,
        n_chains=2,
        n_warmup=300,
        n_samples=1500,
        check=False,
    )
    assert g.rates is not None and n.rates is not None
    assert_draws_agree(*(r.rates.reshape(r.chains.shape) for r in (g, n)))
    np.testing.assert_allclose(g.samples.sum(1), 1.0)


@pytest.mark.parametrize("prior", [P.GeneralizedDirichlet(2.0), P.PolyaTree(1.0)], ids=repr)
def test_gibbs_tree_priors_agree_with_nuts(prior, seed):
    mock = binference.simulate.mock_1d(200, Bins.linear(0, 3, 4), noise=(0.2, 0.6), rng=seed + 9)
    g = binference.infer(
        mock.likelihoods, prior, method="gibbs", rng=seed + 7, n_samples=3000, check=False
    )
    n = binference.infer(
        mock.likelihoods,
        prior,
        method="nuts",
        rng=seed + 7,
        n_chains=2,
        n_warmup=300,
        n_samples=1500,
        check=False,
    )
    assert_draws_agree(g.chains, n.chains)


def test_gibbs_validation_and_labels(onehot, seed):
    lk, _ = onehot
    with pytest.raises(ValueError, match="conjugate"):
        Gibbs(PopulationModel(lk, P.LogisticNormal()))
    with pytest.warns(InputWarning, match="pseudo-likelihood"):
        half = lk.with_weights(np.full(lk.N, 0.5), fractional_weights=True)
    with pytest.raises(ValueError, match="integer"):
        Gibbs(PopulationModel(half))
    g = Gibbs(PopulationModel(lk), n_chains=2, n_warmup=5, store_labels=[0, 3], rng=seed + 1)
    res = g.run(50)
    assert res.labels.shape == (2, 50, 2)
    np.testing.assert_array_equal(res.label_index, [0, 3])
    a = Gibbs(PopulationModel(lk), n_warmup=5, rng=seed + 2).run(10)
    b = Gibbs(PopulationModel(lk), n_warmup=5, rng=seed + 2).run(10)
    np.testing.assert_array_equal(a.samples, b.samples)


def test_gibbs_integer_weights_compressed(rng, seed):
    L = np.array([[1.0, 0.5, 0.1], [0.2, 1.0, 0.3], [0.1, 0.4, 1.0]])
    rows = rng.integers(0, 3, size=300)
    lk_full = Likelihoods(L[rows])
    lk_comp = lk_full.compress()
    assert lk_comp.N == 3
    a = binference.infer(lk_full, method="gibbs", rng=seed + 1, n_samples=4000, check=False)
    b = binference.infer(lk_comp, method="gibbs", rng=seed + 1, n_samples=4000, check=False)
    assert_draws_agree(b.chains, a.chains)


def test_label_sampling_dense_sparse_agree(rng):
    L = rng.uniform(size=(5000, 6)) * (rng.uniform(size=(5000, 6)) > 0.4)
    L[:, 0] += 0.01
    rho = rng.dirichlet(np.ones(6))
    import scipy.sparse as sp

    a = kern.sample_labels(L, rho, np.random.default_rng(1))
    b = kern.sample_labels(sp.csr_array(L), rho, np.random.default_rng(1))
    assert np.mean(a == b) > 0.999
    # labels never land on zero-probability bins
    assert np.all(L[np.arange(5000), b] > 0)


# ---------------------------------------------------------------------- MAP
def test_map_mle_matches_scipy(k3_problem):
    mock, _ = k3_problem
    lk = mock.likelihoods
    est = binference.infer(lk, method="map")
    assert est.converged and est.kkt_gap <= GAP
    L = lk.toarray()

    def nll(eta):
        r = np.exp(eta - eta.max())
        r /= r.sum()
        return -np.log(L @ r).sum()

    ref = optimize.minimize(
        nll,
        np.zeros(3),
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000},
    )
    r_ref = np.exp(ref.x - ref.x.max())
    np.testing.assert_allclose(est.rho, r_ref / r_ref.sum(), atol=1e-5)


def test_map_squarem_faster_than_em(seed):
    # 300 objects keep plain EM to ~11000 iterations (~0.3 s) and SQUAREM still needs
    # ~40 times fewer
    mock = binference.simulate.mock_1d(300, Bins.linear(0, 3, 15), noise=(0.3, 0.6), rng=seed + 1)
    m = PopulationModel(mock.likelihoods)
    fast = MAP(m, algorithm="em", accelerate=True).run()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", binference.ConvergenceWarning)
        slow = MAP(m, algorithm="em", accelerate=False, max_iter=100000).run()
    # plain EM stops early along flat directions, so the mix-SQP optimum is the reference
    np.testing.assert_allclose(fast.rho, MAP(m).run().rho, atol=2e-4)
    assert fast.n_iter * 5 < slow.n_iter


@pytest.mark.parametrize("alpha", [1.0, 2.5])
@pytest.mark.parametrize("sparse", [False, True])
def test_mixsqp_matches_em_and_is_certified(alpha, sparse, seed):
    mock = binference.simulate.mock_1d(1500, Bins.linear(0, 3, 20), noise=(0.2, 0.5), rng=seed + 3)
    lk = mock.likelihoods.to_sparse(1e-8) if sparse else mock.likelihoods
    m = PopulationModel(lk, P.Dirichlet(alpha))
    sqp = MAP(m).run()
    assert sqp.info["algorithm"] == "mixsqp"
    # mix-SQP stops on round-off with the KKT gap below 1e3 times tol = 1e-8 (macOS
    # BLAS stops there at 2.5e-7 where Linux reaches 1e-13)
    assert sqp.converged and sqp.kkt_gap < 1e-5 and sqp.n_iter < 50
    em = MAP(m, algorithm="em", max_iter=200000).run()
    # EM crawls along flat directions of sparse maximum-likelihood solutions, so the
    # better-certified mix-SQP optimum is the reference
    np.testing.assert_allclose(sqp.rho, em.rho, atol=1e-3)
    assert sqp.log_posterior >= em.log_posterior - 1e-8
    # weighted fits (used by the bootstrap) agree too
    w = np.random.default_rng(0).poisson(1.0, lk.N).astype(float)
    np.testing.assert_allclose(
        MAP(m._with_weights(w)).run().rho,
        MAP(m._with_weights(w), algorithm="em", max_iter=200000).run().rho,
        atol=1e-3,
    )


def test_mixsqp_on_tall_dense_inputs_and_fallbacks(seed):
    mock = binference.simulate.mock_1d(6000, Bins.linear(0, 3, 30), noise=(0.1, 0.3), rng=seed + 4)
    m = PopulationModel(mock.likelihoods)  # tall and dense
    est = MAP(m).run()
    assert est.info["algorithm"] == "mixsqp" and est.kkt_gap <= GAP
    # informative priors with selection fall back to EM; asking for mix-SQP there is an error
    sel = PopulationModel(
        mock.likelihoods,
        P.Dirichlet(2.0),
        selection=np.linspace(1, 0.5, 30),
        selection_mode="on_truth",
    )
    assert MAP(sel).run().info["algorithm"] == "squarem"
    with pytest.raises(ValueError, match="concentrations >= 1"):
        MAP(sel, algorithm="mixsqp")
    with pytest.raises(ValueError):
        MAP(m, algorithm="newton")


def test_mixsqp_stops_at_the_round_off_floor(seed):
    # the KKT gap of this optimum cannot be pushed much below 1e-8 in floating point
    mock = binference.simulate.mock_1d(800, Bins.linear(0, 3, 15), rng=seed + 6)
    est = MAP(PopulationModel(mock.likelihoods)).run()  # warnings are errors here
    assert est.converged and est.n_iter < 50 and est.kkt_gap <= GAP


@pytest.mark.parametrize("mode", ["on_truth", "on_measurement"])
def test_mixsqp_with_selection_solves_for_observed_fractions(mode, seed):
    mock = binference.simulate.mock_1d(3000, Bins.linear(0, 3, 20), noise=(0.1, 0.3), rng=seed + 2)
    S = np.linspace(0.2, 1.0, 20) ** 2
    m = PopulationModel(mock.likelihoods, selection=Selection(3.0 * S, mode, scale="relative"))
    sqp = MAP(m).run()
    assert sqp.info["algorithm"] == "mixsqp"
    assert sqp.converged and sqp.n_iter < 50
    # mix-SQP bounds the gap of the observed fractions by tol (1e3 * tol once round-off
    # stops it); the gap in rho is larger by up to max(S) / (S . rho)
    assert sqp.kkt_gap < GAP * S.max() / (S @ sqp.rho)
    em = MAP(m, algorithm="em", max_iter=200000).run()
    np.testing.assert_allclose(sqp.rho, em.rho, atol=1e-4)
    assert sqp.log_posterior >= em.log_posterior - 1e-8


def test_map_is_em_fixed_point(k3_problem):
    mock, _ = k3_problem
    m = PopulationModel(mock.likelihoods)
    est = MAP(m).run()
    np.testing.assert_allclose(m.expected_counts(est.rho) / m.W, est.rho, atol=1e-7)


def test_map_with_prior_and_selection(k3_selection_problem):
    mock, exact = k3_selection_problem
    m = PopulationModel(mock.likelihoods, P.Dirichlet(2.0))
    est = MAP(m).run()
    # the reference used a flat prior; Dirichlet(2) adds sum(log rho) to its log density
    best = exact.points[np.argmax(exact.log_density + np.log(exact.points).sum(axis=1))]
    np.testing.assert_allclose(est.rho, best, atol=5e-3)
    assert est.kkt_gap <= GAP


def test_map_tree_prior_matches_grid(seed):
    mock = binference.simulate.mock_1d(150, Bins.linear(0, 3, 3), noise=(0.3, 0.8), rng=seed + 2)
    m = PopulationModel(mock.likelihoods, P.GeneralizedDirichlet([3.0, 2.0], [4.0, 2.5]))
    est = MAP(m).run()
    ex = binference.validation.exact_posterior(m, n_grid=600)  # mode from the nodes
    best = ex.points[np.argmax(ex.log_density)]
    np.testing.assert_allclose(est.rho, best, atol=3e-3)


def test_map_unconstrained_and_logistic(seed):
    mock = binference.simulate.mock_1d(300, Bins.linear(0, 3, 8), rng=seed + 3)
    est = binference.infer(mock.likelihoods, P.LogisticRandomWalk(order=2), method="map")
    assert est.info["space"] == "unconstrained"
    assert est.info["grad_norm"] < 1e-3 * mock.likelihoods.W
    with pytest.warns(InputWarning):
        MAP(PopulationModel(mock.likelihoods, P.Dirichlet(0.5)), space="simplex").run()
    auto = MAP(PopulationModel(mock.likelihoods, P.Dirichlet(0.5))).run()
    assert auto.info["space"] == "unconstrained"
    with pytest.raises(ValueError):
        MAP(PopulationModel(mock.likelihoods, P.LogisticNormal()), space="simplex").run()


# ------------------------------------------------------------------ Laplace
def test_laplace_with_importance(k3_problem, seed):
    mock, exact = k3_problem
    res = binference.infer(mock.likelihoods, method="laplace", rng=seed + 4, n_samples=4000)
    assert "k_hat" in res.info
    assert_matches_reference(res, exact.mean, exact.std)


def test_a_widened_gaussian_laplace_proposal_recovers_the_exact_posterior(onehot, seed):
    # one-hot likelihoods: the posterior is Dirichlet(1 + counts), close to Gaussian in
    # log-ratio coordinates
    lk, counts = onehot
    m = PopulationModel(lk)
    res = Laplace(m, df=None, inflate=1.5, rng=seed).run(4000)
    assert not res.approximate  # k-hat passed
    # a wrong sign of log q narrows the draws by about a quarter
    assert_matches_reference(res, *dirichlet_moments(1.0 + counts))
    # without importance the draws are the proposals, which inflate widens
    with pytest.warns(ApproximationWarning):
        raw = Laplace(m, inflate=1.5, importance=False, rng=seed).run(8000)
    ratio = raw.x[0].std(axis=0) / (1.5 * np.sqrt(np.diag(raw.info["cov"])))
    np.testing.assert_allclose(ratio, 1.0, atol=5 / np.sqrt(2 * 8000))


def test_laplace_uncorrected_is_flagged(k3_problem, seed):
    mock, _ = k3_problem
    with pytest.warns(ApproximationWarning):
        res = Laplace(PopulationModel(mock.likelihoods), importance=False, rng=seed + 1).run(500)
    assert res.approximate


def test_laplace_clips_negative_curvature_to_a_wide_positive_variance(k3_problem, monkeypatch):
    from binference.engines import laplace

    real = laplace.fd_hessian

    def indefinite(fn, x, h):  # flip the curvature along the first eigenvector of -H
        lam, U = np.linalg.eigh(-real(fn, x, h=h))
        lam[0] = -lam[0]
        return -(U * lam) @ U.T

    monkeypatch.setattr(laplace, "fd_hessian", indefinite)
    with pytest.warns(ApproximationWarning, match="not positive definite; clipping"):
        _, cov, H = Laplace(PopulationModel(k3_problem[0].likelihoods)).mode_and_covariance()
    lam, U = np.linalg.eigh(-H)
    assert lam[0] < 0 < lam[1]
    floor = 1e-8 * max(1.0, lam[1])
    np.testing.assert_allclose(cov @ U[:, 0], U[:, 0] / floor, rtol=1e-6)
    np.testing.assert_allclose(cov @ U[:, 1], U[:, 1] / lam[1], rtol=1e-6)


# ----------------------------------------------------------------------- VB
def test_vb_exact_for_onehot_and_overconfident_otherwise(onehot, k3_problem, seed):
    lk, counts = onehot
    with pytest.warns(ApproximationWarning):
        res = MeanFieldVB(PopulationModel(lk), rng=seed + 1).run(20000)
    assert_matches_reference(res, *dirichlet_moments(1.0 + counts))
    mock, exact = k3_problem
    with pytest.warns(ApproximationWarning):
        vb = MeanFieldVB(PopulationModel(mock.likelihoods), rng=seed + 1).run(4000)
    assert vb.approximate
    assert np.all(vb.std < 0.8 * exact.std)
    with pytest.raises(ValueError, match="conjugate"):
        MeanFieldVB(PopulationModel(lk, P.LogisticNormal()))


def test_vb_means_with_a_selection_are_close_to_the_exact_posterior(k3_selection_problem, seed):
    mock, exact = k3_selection_problem
    with pytest.warns(ApproximationWarning):
        vb = MeanFieldVB(PopulationModel(mock.likelihoods), rng=seed + 1).run(4000)
    # VB's means carry a bias of a few 1e-3 here (the plug-in of the missed objects and
    # the mean-field factorisation; measured 0.003); its Monte Carlo error is 0.025 /
    # sqrt(4000) = 4e-4. Ignoring the selection moves the means by 0.2.
    np.testing.assert_allclose(vb.mean, exact.mean, atol=0.01)


def test_vb_rates_with_a_selection_match_gibbs(seed):
    # likelihoods much narrower than the bins: the mean-field factorisation is nearly
    # exact (exact for one-hot likelihoods); the plug-in of the missed objects adds a bias
    # of at most 0.5% of the rate (three seeds), allowed for at 1% beside the MC error
    bins = Bins([0.0, 1.0, 2.0, 3.0])
    S = np.array([0.9, 0.5, 0.25])
    mock = binference.simulate.mock_1d(
        250,
        bins,
        density=np.array([0.4, 0.35, 0.25]),
        noise=(0.05, 0.2),
        selection=S,
        rng=seed + 777,
    )
    m = PopulationModel(mock.likelihoods, P.GammaRates(1.0, 1e-3))
    with pytest.warns(ApproximationWarning):
        vb = MeanFieldVB(m, rng=seed + 1).run(4000)
    g = Gibbs(m, n_warmup=200, rng=seed + 6).run(2000)
    R = g.rates.reshape(g.n_chains, -1, 3)
    se = np.hypot(
        [binference.diagnostics.mcse(R[:, :, k]) for k in range(3)],
        vb.rates.std(0) / np.sqrt(4000),
    )
    assert np.all(np.abs(vb.rates.mean(0) - g.rates.mean(0)) <= 4 * se + 0.01 * g.rates.mean(0))


# ---------------------------------------------------------------- Bootstrap
def test_bootstrap(k3_problem, seed):
    mock, _ = k3_problem
    with pytest.warns(ApproximationWarning):
        res = Bootstrap(PopulationModel(mock.likelihoods), rng=seed + 1).run(60)
    assert res.samples.shape == (60, 3) and res.approximate
    # At large N the bootstrap sd matches the posterior sd. With B = 200 replicates the
    # sd has relative MC error ~ 1/sqrt(2B) = 5% (ratios 0.91-1.10 over 8 seeds), so
    # |log ratio| < 0.2 is 4 sigma and catches a factor sqrt(2) either way (0.35).
    big = binference.simulate.mock_1d(
        2000, mock.bins, density=np.array([0.5, 0.15, 0.35]), noise=(0.3, 1.2), rng=seed + 12345
    )
    ref = binference.validation.exact_posterior(big.likelihoods)
    with pytest.warns(ApproximationWarning):
        boot = Bootstrap(PopulationModel(big.likelihoods), rng=seed + 1).run(200)
    assert np.all(np.abs(np.log(boot.std / ref.std)) < 0.2)
    with pytest.warns(ApproximationWarning):
        bb = Bootstrap(PopulationModel(mock.likelihoods), scheme="bayesian", rng=seed + 1).run(20)
    assert bb.info["scheme"] == "bayesian" and bb.info["n_not_converged"] == 0
    with pytest.raises(ValueError):
        Bootstrap(PopulationModel(mock.likelihoods), scheme="parametric")


def test_bootstrap_of_a_logistic_prior_refits_each_reweighted_catalogue(seed):
    # a smooth prior has no EM update: each replicate is a weighted L-BFGS fit
    mock = binference.simulate.mock_1d(200, Bins.linear(0, 3, 6), noise=(0.1, 0.3), rng=seed + 9)
    m = PopulationModel(mock.likelihoods, P.LogisticRandomWalk())
    with pytest.warns(ApproximationWarning):
        res = Bootstrap(m, rng=seed + 1).run(20)
    X = res.samples
    assert X.shape == (20, 6) and np.all(np.isfinite(X)) and res.info["n_not_converged"] == 0
    np.testing.assert_allclose(X.sum(1), 1.0)
    assert np.unique(X.round(8), axis=0).shape[0] == 20  # every replicate is its own fit
    assert not np.any(np.all(np.isclose(X, res.info["estimate"]), axis=1))


def test_weighted_label_counts_on_sparse_likelihoods_stay_sparse_and_correct():
    rng = np.random.default_rng(0)
    N, K = 20000, 2500
    cols = rng.integers(0, K, size=(N, 10))
    L = sp.csr_array((rng.random(N * 10), (np.repeat(np.arange(N), 10), cols.ravel())), (N, K))
    L = sp.csr_array(sp.vstack([L, np.ones((1, K))]))  # one flat object must not densify the rest
    N += 1
    rho = np.full(K, 1.0 / K)
    w = rng.integers(1, 1000, size=N).astype(float)
    t = time.perf_counter()
    counts, _ = kern.sample_counts(L, w, rho, rng, unit_weights=False)
    # the dense path took 13 s and 1.5 GB here (audit G-2); a label per object exceeded 2 s
    assert time.perf_counter() - t < 2.0
    assert counts.sum() == w.sum()
    # each count is binomial: its expectation from the responsibilities
    expected = rho * (L.T @ (w / (L @ rho)))  # no N x K array
    assert np.max(np.abs(counts - expected) / np.sqrt(expected + 1)) < 6


def _model(n=200, K=4):
    mock = binference.simulate.mock_1d(n, Bins.linear(0, 3, K), rng=1)
    return PopulationModel(mock.likelihoods)


def test_gibbs_inputs_are_checked_when_the_engine_is_made():
    m = _model()
    Gibbs(m, chunk=None).run(2)  # no chunking, as the kernels allow
    with pytest.raises(ValueError, match="store_labels"):
        Gibbs(m, store_labels=[0, 5000])
    with pytest.raises(ValueError, match="init"):
        Gibbs(m, init=[0.5, 0.5, 0.0, 0.0])
    with pytest.raises(ValueError, match="n_warmup"):
        Gibbs(m, n_warmup=-1)
    compressed = PopulationModel(Likelihoods(np.eye(2)[[0, 0, 1]], Bins.categories(2)).compress())
    for store_labels in (True, [0]):
        with pytest.raises(ValueError, match="store_labels needs unit object weights"):
            Gibbs(compressed, store_labels=store_labels)


def test_gibbs_chains_from_one_init_start_apart():
    eng = Gibbs(_model(), init=[0.25, 0.25, 0.25, 0.25], n_chains=3, rng=0)
    eng._setup()
    starts = np.array(eng._state)
    assert not np.allclose(starts[0], starts[1])  # R-hat needs overdispersed starts


def test_gibbs_starts_correct_for_the_objects_a_selection_missed():
    mock = binference.simulate.mock_1d(
        3000, Bins.linear(0, 3, 3), selection=[1.0, 0.5, 0.2], noise=0.3, rng=3
    )
    m = PopulationModel(mock.likelihoods)
    eng = Gibbs(m, n_chains=20, rng=0)
    eng._setup()
    # starts drawn around the observed composition were 0.1 or more off in a fraction;
    # the start posterior SD is about 0.01, so the mean of 20 is within 0.01
    np.testing.assert_allclose(np.mean(eng._state, axis=0), MAP(m).run().rho, atol=0.02)


def test_bootstrap_counts_replicates_that_did_not_converge():
    m = PopulationModel(
        binference.simulate.mock_1d(500, Bins.linear(0, 3, 6), rng=2).likelihoods,
        binference.priors.Dirichlet(2.0),
    )
    with (
        pytest.warns(binference.ApproximationWarning),  # always: not a posterior
        pytest.warns(binference.ConvergenceWarning) as rec,
    ):
        res = Bootstrap(m, max_iter=1, rng=0).run(5)
    msgs = [str(r.message) for r in rec]
    assert any(s.startswith("EM did not converge") for s in msgs)  # the base fit is not silenced
    assert "5 of 5 bootstrap fits did not converge; raise max_iter." in msgs
    assert res.info["n_not_converged"] == 5


def test_label_draws_do_not_depend_on_the_chunk_size():
    L = np.random.default_rng(0).random((50, 6)) * (np.random.default_rng(1).random((50, 6)) > 0.4)
    L[:, 0] += 0.1  # every object possible
    rho = np.random.default_rng(2).dirichlet(np.ones(6))
    for M in (L, sp.csr_array(L)):
        whole = kern.sample_labels(M, rho, np.random.default_rng(3), chunk=None)
        np.testing.assert_array_equal(
            kern.sample_labels(M, rho, np.random.default_rng(3), chunk=7), whole
        )


def test_gibbs_says_why_a_model_is_unsupported_and_refuses_it_with_that_reason():
    lk = binference.simulate.mock_1d(50, Bins.linear(0, 1, 4), rng=0).likelihoods
    assert Gibbs.unsupported(PopulationModel(lk)) is None
    logistic = PopulationModel(lk, binference.priors.LogisticNormal())
    reason = Gibbs.unsupported(logistic)
    assert "conjugate" in reason
    with pytest.raises(ValueError, match=reason[:30]):
        Gibbs(logistic)


def test_mixsqp_takes_dirichlet_priors_mixing_flat_and_informative_bins():
    # a_k = 0 next to a_k > 0 divided 0 by 0 in the Hessian (warnings are errors here)
    m = PopulationModel(
        small_mock().likelihoods, P.Dirichlet([1, 1, 3, 1, 2, 1, 1, 4, 1, 1, 1, 2])
    )
    sqp = MAP(m).run()
    assert sqp.info["algorithm"] == "mixsqp" and sqp.converged and sqp.kkt_gap < 1e-5
    em = MAP(m, algorithm="em").run()
    np.testing.assert_allclose(sqp.rho, em.rho, atol=1e-4)


@pytest.mark.parametrize(
    ("shape", "rate", "selection"),
    [(1.0, 1e-12, None), (1.0, 0.5, np.linspace(2.0, 0.5, 12)), (2.0, 0.1, None)],
)
def test_gamma_rates_use_mixsqp_and_reach_the_mode(shape, rate, selection):
    sel = None if selection is None else Selection(selection, "on_truth", scale="exposure")
    m = PopulationModel(small_mock().likelihoods, P.GammaRates(shape, rate), selection=sel)
    sqp = MAP(m).run()
    assert sqp.info["algorithm"] == "mixsqp" and sqp.converged and sqp.n_iter < 100
    assert sqp.kkt_gap <= GAP
    # the total is profiled out exactly: Lambda = W + sum(shape - 1) over the exposure
    e = m._S + rate
    np.testing.assert_allclose(sqp.value @ e, m.W + m.K * (shape - 1.0), rtol=1e-10)
    em = MAP(m, algorithm="em", max_iter=100000).run()
    assert em.converged and em.kkt_gap <= 1e-5
    np.testing.assert_allclose(sqp.value, em.value, rtol=1e-3, atol=1e-3 * m.W / m.K)


def test_em_converges_only_where_the_kkt_gap_is_small():
    # SQUAREM used to stop on a stalled objective far from the mode (audit M-6)
    m = PopulationModel(small_mock(K=40, noise=0.1).likelihoods, P.GammaRates(1.0, 1e-12))
    em = MAP(m, algorithm="em").run()
    assert em.converged and em.kkt_gap <= 1e3 * MAP(m).tol
    # adjacent bins trade mass along a nearly flat direction, so compare the objective
    np.testing.assert_allclose(em.log_posterior, MAP(m).run().log_posterior, rtol=1e-10)


@pytest.mark.parametrize(
    ("prior", "kwargs"),
    [
        (P.Dirichlet(2.0), {}),
        (P.Dirichlet(2.0), {"algorithm": "em"}),
        (P.Dirichlet(2.0), {"space": "unconstrained"}),
        (P.GammaRates(1.0, 1e-3), {}),
    ],
)
def test_every_algorithm_reports_the_same_log_posterior(prior, kwargs):
    lk = small_mock().likelihoods
    m = PopulationModel(lk._copy(weights=7.0 * lk.weights), prior)  # log-scales matter
    est = MAP(m, **kwargs).run()
    np.testing.assert_allclose(est.log_posterior, m.log_posterior(est.value), rtol=1e-12)
    if est.x is not None:
        np.testing.assert_allclose(
            est.info["log_density_unconstrained"], m.log_density_unconstrained(est.x)
        )


@pytest.mark.parametrize("kwargs", [{"algorithm": "em"}, {"space": "unconstrained"}])
def test_a_looser_tolerance_stops_every_algorithm_sooner(kwargs):
    m = PopulationModel(small_mock(K=20, noise=0.3).likelihoods, P.Dirichlet(2.0))
    tight = MAP(m, tol=1e-10, **kwargs).run()
    loose = MAP(m, tol=1e-4, **kwargs).run()
    assert loose.n_iter < tight.n_iter and tight.converged and loose.converged


def test_mixsqp_starts_from_its_init():
    m = PopulationModel(small_mock().likelihoods)
    mode = MAP(m).run()
    again = MAP(m, init=mode.value).run()
    assert again.n_iter <= 2 < mode.n_iter
    np.testing.assert_allclose(again.value, mode.value, atol=1e-8)


def test_map_checks_its_init_and_em_moves_a_zero_in_it():
    m = PopulationModel(small_mock(n=200, K=4).likelihoods)
    for init in (np.full(7, 0.1), [0.5, 0.6, -0.2, 0.1], np.zeros(4)):
        with pytest.raises(ValueError, match="init"):
            MAP(m, init=init)
    em = MAP(m, algorithm="em", init=[0.5, 0.5, 0.0, 0.0]).run()  # the mode's third is 0.26
    assert em.converged
    np.testing.assert_allclose(em.value, MAP(m).run().value, atol=1e-8)


def test_a_failed_quadratic_subproblem_hands_mixsqp_over_to_em(monkeypatch):
    from binference.engines import optimize

    def fail(*args, **kwargs):
        raise RuntimeError("Maximum number of iterations reached.")

    monkeypatch.setattr(optimize.optimize, "nnls", fail)
    m = PopulationModel(small_mock().likelihoods)
    with pytest.warns(ConvergenceWarning, match="continuing with EM"):
        est = MAP(m).run()
    assert est.info["algorithm"] == "squarem" and est.converged and est.kkt_gap < 1e-5


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_zero_rate_under_a_gamma_prior_is_no_kkt_violation(seed):
    # the prior gradient at a zero rate is -b, not 0
    sel = Selection(np.linspace(2.0, 0.5, 12), "on_truth", scale="exposure")
    m = PopulationModel(small_mock(seed=seed).likelihoods, P.GammaRates(1.0, 0.5), selection=sel)
    est = MAP(m).run()
    assert est.converged and est.kkt_gap <= GAP


@pytest.mark.parametrize("alpha", [0.5, 0.9])
def test_priors_unbounded_at_the_boundary_converge_on_the_simplex(alpha):
    m = PopulationModel(small_mock(K=20, noise=0.2).likelihoods, P.Dirichlet(alpha))
    with pytest.warns(binference.InputWarning, match="unbounded"):
        est = MAP(m, space="simplex").run()  # a ConvergenceWarning is an error here
    assert est.converged and np.any(est.value == 0.0)


def test_em_started_at_its_fixed_point_converges_at_once():
    m = PopulationModel(small_mock().likelihoods, P.Dirichlet(2.0))
    mode = MAP(m, algorithm="em").run()
    again = MAP(m, algorithm="em", init=mode.value, max_iter=5).run()
    assert again.converged and again.n_iter < 5


def test_an_explicit_simplex_algorithm_never_silently_runs_lbfgs():
    lk = small_mock(n=300, K=8).likelihoods
    # a non-conjugate prior, or the unconstrained space, cannot run EM or mix-SQP
    with pytest.raises(ValueError, match="algorithm='em' cannot run here: the simplex needs"):
        MAP(PopulationModel(lk, P.LogisticRandomWalk()), algorithm="em")
    with pytest.raises(ValueError, match="algorithm='mixsqp'"):
        MAP(PopulationModel(lk, P.Dirichlet(1.0)), algorithm="mixsqp", space="unconstrained")
    # and mix-SQP refuses a prior it cannot take before warning about anything
    with pytest.raises(ValueError, match="concentrations >= 1"):
        MAP(PopulationModel(lk, P.Dirichlet(0.5)), algorithm="mixsqp")
    # below one, auto picks L-BFGS, but asking for EM runs EM on the simplex (and warns)
    with pytest.warns(binference.InputWarning, match="unbounded"):
        est = binference.infer(lk, P.Dirichlet(0.5), method="em")
    assert est.info == {"space": "simplex", "algorithm": "squarem"}


def _one_object_bin(n=40, K=5, seed=0):
    # n objects spread over bins 0..K-2 and one object alone in bin K-1
    rng = np.random.default_rng(seed)
    L = rng.uniform(0.0, 0.1, size=(n + 1, K))
    L[np.arange(n), rng.integers(0, K - 1, size=n)] = 1.0
    L[:n, K - 1] = 0.0
    L[n] = 0.0
    L[n, K - 1] = 1.0
    return L


def test_a_zero_weight_row_is_the_same_as_no_row():
    L = _one_object_bin()
    w = np.ones(L.shape[0])
    w[-1] = 0.0
    m = PopulationModel(Likelihoods(L, Bins.categories(5)), P.Dirichlet(1.0))
    with_zero = MAP(m._with_weights(w), algorithm="em").run().value
    dropped = PopulationModel(Likelihoods(L[:-1], Bins.categories(5)), P.Dirichlet(1.0))
    without = MAP(dropped, algorithm="em").run().value
    np.testing.assert_allclose(with_zero, without, atol=1e-6)
    assert with_zero[-1] < 1e-6  # not reset to the uniform vector


def test_a_rate_bootstrap_never_collapses_to_zero():
    L = _one_object_bin()
    m = PopulationModel(Likelihoods(L, Bins.categories(5)), P.GammaRates(1.0, 1e-3))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = Bootstrap(m, rng=3).run(60)
    totals = res.rates.sum(axis=1)
    assert np.all(totals > 0) and np.all(np.isfinite(res.samples))


def test_em_stops_with_a_warning_where_an_object_becomes_impossible():
    L = _one_object_bin(n=8, K=4)
    w = np.ones(L.shape[0])
    w[-1] = 0.3  # alone in its bin, clipped to zero by a concentration below one
    with pytest.warns(binference.InputWarning, match="pseudo-likelihood"):
        lk = Likelihoods(L, Bins.categories(4), weights=w, fractional_weights=True)
    m = PopulationModel(lk, P.Dirichlet(0.5))
    for accelerate in (True, False):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # also warns that the prior is unbounded
            est = MAP(m, space="simplex", accelerate=accelerate).run()
        assert any(
            issubclass(w.category, ConvergenceWarning) and "zero likelihood" in str(w.message)
            for w in caught
        )
        assert np.isfinite(est.log_posterior) and not np.allclose(est.value, 0.25)


def test_the_rates_start_is_the_observed_composition_over_the_exposure():
    from binference.engines.base import interior_start

    # broad likelihoods, so the composition must be fitted on the columns of L over S: for
    # seeds 0-19 of this mock the start's log posterior is 0.2-2.2 below the mode's (26-47
    # when fitted on L)
    lk = binference.simulate.mock_1d(1000, Bins.linear(0, 3, 6), noise=0.8, rng=3).likelihoods
    S = np.linspace(0.9, 0.3, 6)
    for mode in ("on_truth", "on_measurement"):
        m = PopulationModel(lk, P.GammaRates(1.0, 1e-9), selection=Selection(S, mode))
        start, mle = interior_start(m), MAP(m).run().value
        np.testing.assert_allclose(start.sum(), mle.sum(), rtol=0.1)
        assert m.log_posterior(mle) - m.log_posterior(start) < 8.0


def test_squarem_rejects_an_extrapolation_that_makes_an_object_impossible():
    m = PopulationModel(Likelihoods(np.eye(3) + 0.1, Bins.categories(3)))
    target = np.array([0.2, 0.3, 0.5])
    calls = []

    def F(x, s):  # an EM-like contraction; every extrapolated trial meets an impossible object
        calls.append(x)
        return None if len(calls) > 2 else 0.5 * (x + target)

    v = np.array([0.6, 0.3, 0.1])
    new, f, s = MAP(m)._squarem_step(m, F, v, m._L @ v, fractions=True)
    r2 = 0.5 * (0.5 * (v + target) + target)  # the two plain EM steps
    np.testing.assert_allclose(new, r2)
    assert np.isfinite(f) and len(calls) > 3  # trials were made and rejected
    np.testing.assert_allclose(s, m._L @ r2)  # the evidences at the point returned


def _gp_model(n=3000, K=12, seed=2):
    mock = binference.simulate.mock_1d(n, Bins.linear(0, 3, K), noise=(0.05, 0.1), rng=seed)
    return PopulationModel(mock.likelihoods, P.LogisticRandomWalk(order=1))


def test_nuts_refuses_a_model_whose_prior_is_already_centred():
    m = _gp_model()
    centred = m._with_bound_prior(m.prior.with_centering(np.zeros(m.prior.n_active)))
    with pytest.raises(ValueError, match="non-centred"):
        NUTS(centred)


def test_positions_from_sample_are_in_the_model_coordinates():
    m = _gp_model()
    eng = NUTS(m, n_chains=2, n_warmup=150, rng=1)
    draws = eng.sample(1)
    x = next(draws)
    draws.close()  # records the draw
    assert not np.all(eng.model.prior.gamma == 1.0)  # sampled partly centred
    np.testing.assert_allclose(m.to_value(x[0]), eng.result.chains[0, -1])


def test_nuts_arguments_are_checked():
    m = _gp_model(n=200, K=6)
    for kwargs in ({"max_depth": 0}, {"n_warmup": -1}, {"jitter": -0.1}):
        with pytest.raises(ValueError):
            NUTS(m, **kwargs)
    with pytest.raises(ValueError, match="init_x"):
        NUTS(m, init_x=np.full(m.dim, np.nan))


def test_an_error_in_a_worker_is_raised_not_hidden_by_a_serial_rerun():
    eng = NUTS(_gp_model(n=200, K=6), n_chains=2, n_warmup=0, n_jobs=2, rng=0)
    eng._setup()

    class Pool:  # a pool whose workers report the model's own error
        def map(self, f, payloads):
            return [ValueError("from the model")] * len(payloads)

        def shutdown(self):
            pass

    eng._pool = Pool()
    with pytest.raises(ValueError, match="from the model"):
        eng._parallel(1, adapt=False, record=False)
    assert eng.n_jobs == 2


def test_a_dense_metric_draws_momenta_with_the_inverse_covariance():
    inv_m = np.array([[2.0, 0.6, 0.0], [0.6, 1.0, 0.3], [0.0, 0.3, 0.5]])
    rng = np.random.default_rng(0)
    p = np.array([Metric(inv_m).draw_momentum(rng) for _ in range(40000)])
    # entries of a sample covariance from 40,000 draws are within about 2% of the truth
    np.testing.assert_allclose(np.cov(p.T), np.linalg.inv(inv_m), atol=0.08)


def _dirichlet_model():
    mock = binference.simulate.mock_1d(500, Bins.linear(0, 3, 4), rng=3)
    return PopulationModel(mock.likelihoods)


def test_an_uncorrected_laplace_returns_every_draw_asked_for():
    with pytest.warns(binference.ApproximationWarning):
        res = Laplace(_dirichlet_model(), importance=False, n_proposals=100, rng=0).run(400)
    assert res.n_draws == 400


def test_laplace_draws_a_few_samples_from_enough_proposals_to_smooth():
    # 2 * n_samples = 4 proposals by default; Pareto smoothing needs 5, too few to vouch
    with pytest.warns(binference.ApproximationWarning):
        res = Laplace(_dirichlet_model(), rng=0).run(2)
    assert res.n_draws == 2 and res.approximate
    with pytest.raises(ValueError, match="n_samples"):
        Laplace(_dirichlet_model(), rng=0).run(0)


def test_the_laplace_k_hat_threshold_tightens_for_few_proposals():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", binference.ApproximationWarning)
        res = Laplace(_dirichlet_model(), rng=0).run(100)  # 200 proposals
    assert res.info["k_hat_threshold"] == pytest.approx(1 - 1 / np.log10(200))
    assert res.approximate == (res.info["k_hat"] >= res.info["k_hat_threshold"])


def test_warm_up_alone_leaves_no_worker_processes():
    eng = NUTS(_gp_model(n=200, K=6), n_chains=2, n_warmup=30, n_jobs=2, rng=0)
    eng.warmup()
    assert eng._pool is None


def _standard_gaussian(x):
    return -0.5 * float(x @ x), -x


def test_a_step_that_overflows_the_energy_is_a_silent_divergence():
    # a momentum of 1e200 overflows p @ M^-1 p; the tree marks the leaf divergent, and the
    # overflow warning is not raised (it failed the joint-selection SBC as an error)
    x, p = np.zeros(2), np.full(2, 1e200)
    m = Metric(np.ones(2))
    t = _leaf(x, p, -x, 1, 1e-300, 0.0, m, _standard_gaussian, _MAX_ENERGY_ERROR)
    assert t.diverging


def test_adaptation_windows_match_stan_except_one_window_for_151_to_199_iterations():
    assert adaptation_windows(1000) == [(75, 100), (100, 150), (150, 250), (250, 450), (450, 950)]
    assert adaptation_windows(175) == [(75, 125)]  # Stan: (75, 100) and (100, 125)
    # at the edges of that range, which pin the comparisons of the buffers and the window
    assert adaptation_windows(150) == [(75, 100)]
    assert adaptation_windows(200) == [(75, 100), (100, 150)]


def test_step_size_adaptation_reaches_the_target_acceptance_on_a_standard_gaussian():
    acc = []
    for rng in np.random.default_rng(0).spawn(4):
        chain = _Chain(rng.standard_normal(10), _standard_gaussian, rng, 1000, 0.8, dense=False)
        for _ in range(1000):
            chain.step(10, adapt=True)
        acc.append([chain.step(10, adapt=False)[0] for _ in range(1000)])
    # Stan's averaged step size accepts more often than the target (0.87-0.90 with seeds
    # 0-19), so the bound is one-sided, at four Monte Carlo errors of the mean accept
    # statistic of each chain: pooled, the error grows when broken adaptation splits them
    mcse = binference.diagnostics.mcse
    assert all(np.mean(a) >= 0.8 - 4 * mcse(np.asarray(a)) for a in acc)


def test_above_the_dense_limit_the_warm_up_curvature_is_taken_at_one_chain(monkeypatch):
    from binference.engines import nuts

    calls = []
    for name in ("fd_hessian_diag", "fd_hessian"):
        real = getattr(nuts, name)
        monkeypatch.setattr(
            nuts, name, lambda *a, _f=real, _n=name, **k: calls.append(_n) or _f(*a, **k)
        )
    model = _gp_model(n=200, K=6)
    counts = []
    for limit in (nuts._DENSE_MAX_DIM, 0):  # 6 bins, then as if above the limit
        monkeypatch.setattr(nuts, "_DENSE_MAX_DIM", limit)
        calls.clear()
        eng = NUTS(model, n_chains=3, n_warmup=0, rng=0)
        eng._setup()
        eng._adapt_geometry()
        counts.append((calls.count("fd_hessian_diag"), calls.count("fd_hessian")))
    # the field's diagonal, then the hyperparameters' and the whole model's Hessians, at
    # every chain's position and then at one
    assert counts == [(3, 6), (1, 2)]


def _bootstrap(model, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ApproximationWarning)
        return Bootstrap(model, **kw).run(150)


# -------------------------------------------------------------- bootstrap (1-3)
@pytest.mark.parametrize("scheme", ["nonparametric", "bayesian"])
def test_bootstrap_resamples_objects_not_rows(scheme):
    rng = np.random.default_rng(0)
    labels = rng.choice(4, size=2000, p=[0.4, 0.1, 0.4, 0.1])
    L = np.eye(4)[labels] * 0.9 + 0.1 / 4  # only four distinct rows
    raw = Likelihoods(L)
    packed = raw.compress()
    assert packed.N == 4
    sd_raw = _bootstrap(PopulationModel(raw), scheme=scheme, rng=1).std
    sd_packed = _bootstrap(PopulationModel(packed), scheme=scheme, rng=1).std
    np.testing.assert_allclose(sd_packed, sd_raw, rtol=0.3)
    assert np.all(sd_packed < 0.05)  # was ~0.35 when rows were resampled


def test_bootstrap_rates_have_a_random_total():
    mock = binference.simulate.mock_1d(1000, Bins.linear(0, 3, 6), rng=2)
    res = _bootstrap(PopulationModel(mock.likelihoods, P.GammaRates(1.0, 1e-6)), rng=1)
    totals = res.rates.sum(axis=1)
    assert totals.std() == pytest.approx(np.sqrt(1000), rel=0.3)


def test_bootstrap_replicates_are_optimal_for_their_own_weights(monkeypatch):
    """A sparse MLE must not trap replicates: each refit is certified optimal (audit T-3)."""
    mock = binference.simulate.mock_1d(150, Bins.linear(0, 3, 20), noise=(0.05, 0.1), rng=3)
    # selection with a non-flat Dirichlet routes MAP to EM, which cannot revive the zeros of
    # the sparse full-data mode (mix-SQP, the default without selection, can)
    alpha = np.ones(20)
    alpha[5:9] = 2.0
    S = np.linspace(1.0, 0.5, 20)
    m = PopulationModel(
        mock.likelihoods, P.Dirichlet(alpha), selection=S, selection_mode="on_truth"
    )
    assert MAP(m).run().info["algorithm"] == "squarem"
    weights = []
    draw = Bootstrap._replicate_weights

    def record(self, w, integer, rates):
        weights.append(draw(self, w, integer, rates))
        return weights[-1]

    monkeypatch.setattr(Bootstrap, "_replicate_weights", record)
    with pytest.warns(ApproximationWarning):
        res = Bootstrap(m, rng=2).run(60)
    assert res.info["n_not_converged"] == 0
    gaps = [m._with_weights(wb).kkt_gap(v) for wb, v in zip(weights, res.samples, strict=True)]
    # converged means gap <= 1e3 * tol (DECISIONS #71); from the full-data MAP: up to 6e-2
    assert max(gaps) < 1e-5


# -------------------------------------------------------------- priors (8, 9, 15)
def test_polya_tree_map_avoids_boundary_zeros():
    bins = Bins.linear(0, 1, 16)
    rho = np.r_[np.full(12, 1 / 12), np.zeros(4)]
    mock = binference.simulate.mock_1d(300, bins, density=rho, noise=0.02, rng=8)
    m = PopulationModel(mock.likelihoods, P.PolyaTree(c=1.0))
    assert m.prior.unbounded_density()
    est = MAP(m).run()
    assert est.info["space"] == "unconstrained"
    assert np.all(est.rho > 0)
    with pytest.warns(InputWarning, match="unbounded"):
        MAP(m, space="simplex").run()


@pytest.mark.parametrize("K", [8, 32])
def test_map_of_a_bounded_polya_tree_converges_when_half_its_bins_are_empty(K):
    # c = K/2 gives the root's branches exponent zero; as the empty half goes to zero the
    # KKT gap in rho cancels terms of order 1/rho (a gap of 1e5 at K = 8), so it is
    # checked in branch coordinates
    bins = Bins.linear(0, 1, K)
    rho = np.r_[np.full(K // 2, 2 / K), np.zeros(K // 2)]
    mock = binference.simulate.mock_1d(300, bins, density=rho, noise=0.02, rng=8)
    m = PopulationModel(mock.likelihoods, P.PolyaTree(c=K / 2))
    est = MAP(m).run()  # a ConvergenceWarning fails the test
    assert est.info["space"] == "simplex" and est.converged
    assert m.kkt_gap(est.rho) < 1e-12


# ------------------------------------------------------------------- NUTS (7, 11)
def test_weak_data_smoothness_fits_stay_non_centred():
    """Poorly constrained scales: partial centring would create a funnel."""
    bins = Bins.linear(0.0, 3.0, 6)
    density = binference.simulate.population(bins, "bimodal")
    mock = binference.simulate.mock_1d(30, bins, density=density, noise=(0.2, 0.4), rng=6)
    # the centring is decided once, at warm-up iteration 25 (from 150 warm-up iterations,
    # as by default; shorter warm-ups decide earlier), so one draw after it suffices
    model = PopulationModel(mock.likelihoods, P.LogisticRandomWalk(order=2))
    eng = NUTS(model, rng=1, n_chains=2, n_warmup=150)
    eng.run(1)
    penalised = ~eng.model.prior.null
    assert np.all(eng.model.prior.gamma[penalised] == 1.0)


def test_parallel_chains_fall_back_when_the_model_cannot_be_pickled():
    mock = binference.simulate.mock_1d(100, Bins.linear(0, 3, 8), rng=1)
    model = PopulationModel(mock.likelihoods, P.PolyaTree(base=lambda z: np.exp(-z)))
    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        pickle.dumps(model)
    eng = NUTS(model, n_chains=2, n_warmup=30, rng=1, n_jobs=2)
    with pytest.warns(RuntimeWarning, match="serially"):
        res = eng.run(10)
    assert res.chains.shape == (2, 10, 8)
    assert eng.n_jobs == 1
    assert eng.run(5).chains.shape == (2, 15, 8)  # still usable


def test_em_convergence_warning_reports_missing_information():
    mock = binference.simulate.mock_1d(500, Bins.linear(0, 3, 15), noise=(0.4, 0.6), rng=1)
    S = np.linspace(1.0, 0.5, 15)
    m = PopulationModel(mock.likelihoods, P.Dirichlet(2.0), selection=S, selection_mode="on_truth")
    with pytest.warns(binference.ConvergenceWarning, match="missing information"):
        MAP(m, max_iter=5).run()


def test_parallel_chain_workers_share_blas_threads():
    from concurrent.futures import ProcessPoolExecutor

    from threadpoolctl import threadpool_info, threadpool_limits

    from binference.engines.nuts import _init_worker, _worker_threads

    with ProcessPoolExecutor(1, initializer=_init_worker, initargs=(None, 1)) as ex:
        info = ex.submit(threadpool_info).result()
    assert all(lib["num_threads"] == 1 for lib in info)
    with threadpool_limits(1):  # a limit set by the user is never raised (audit N-3)
        if threadpool_info():  # nothing to respect where no thread pool is visible
            assert _worker_threads(1) == 1


def test_data_rich_gp_fits_are_partially_centred():
    """A weakly identified length scale must not switch centring off (20x slower)."""
    bins = Bins.linear(0.0, 3.0, 40)
    lk = binference.simulate.mock_1d(400, bins, noise=(0.05, 0.15), rng=21).likelihoods[:300]
    eng = NUTS(PopulationModel(lk, P.LogisticGP()), rng=2, n_warmup=40)
    eng.run(1)
    assert np.mean(eng.model.prior.gamma < 0.5) > 0.1
