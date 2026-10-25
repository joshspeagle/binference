"""Engine tests on small problems with exact references (test plan 2.6)."""

import warnings

import numpy as np
import pytest
from scipy import optimize

import binference
from binference import Bins, Likelihoods, PopulationModel, Selection
from binference import _kernels as kern
from binference import priors as P
from binference.samplers import MAP, NUTS, Bootstrap, Gibbs, Laplace, MeanFieldVB
from binference.warnings import ApproximationWarning, InputWarning

from .conftest import assert_matches_reference


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
def test_bin_without_likelihood_support_is_not_frozen(method):
    """A bin no object can be in keeps its prior-driven mass alpha_k / (A + N) (frankenz F11)."""
    rng = np.random.default_rng(3)
    N, K = 300, 5
    L = rng.random((N, K)) + 0.1
    L[:, -1] = 0.0
    lk = Likelihoods(L, Bins.linear(0, 1, K))
    res = binference.infer(lk, method=method, rng=1, n_samples=1000, check=False)
    exact_mean = 1.0 / (K + N)  # rho_K | D ~ Beta(1, K - 1 + N) under Dir(1)
    assert np.all(res.samples[:, -1] > 0)
    assert res.mean[-1] == pytest.approx(exact_mean, rel=0.15)


def test_nuts_matches_exact(k3_problem):
    mock, exact = k3_problem
    res = binference.infer(
        mock.likelihoods, rng=1, n_chains=2, n_warmup=400, n_samples=1500, check=False
    )
    assert res.method == "nuts" and not res.approximate
    assert_matches_reference(res, exact.mean, exact.std)
    for k in range(3):
        q = exact.quantile([0.05, 0.95], k)
        np.testing.assert_allclose(np.quantile(res.samples[:, k], [0.05, 0.95]), q, atol=0.02)


@pytest.mark.parametrize("metric", ["diag", "dense"])
def test_nuts_onehot_is_conjugate(onehot, metric):
    lk, counts = onehot
    alpha = np.array([1.0, 2.0, 0.5, 1.5])
    res = binference.infer(
        lk,
        P.Dirichlet(alpha),
        rng=2,
        n_chains=2,
        n_warmup=300,
        n_samples=1500,
        metric=metric,
        check=False,
    )
    mean, sd = dirichlet_moments(alpha + counts)
    assert_matches_reference(res, mean, sd)


def test_nuts_reproducible_resumable_and_generator(onehot):
    lk, _ = onehot
    m = PopulationModel(lk)
    a = NUTS(m, n_chains=2, n_warmup=50, rng=11).run(30)
    b = NUTS(m, n_chains=2, n_warmup=50, rng=11).run(30)
    np.testing.assert_array_equal(a.samples, b.samples)
    c = NUTS(m, n_chains=2, n_warmup=50, rng=12).run(30)
    assert not np.array_equal(a.samples, c.samples)
    eng = NUTS(m, n_chains=2, n_warmup=50, rng=3)
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
def test_resuming_is_bitwise_identical_to_one_long_run(engine, onehot):
    m = PopulationModel(onehot[0])
    e1 = engine(m, n_chains=2, n_warmup=30, rng=4)
    e1.run(20)
    split = e1.run(15)
    whole = engine(m, n_chains=2, n_warmup=30, rng=4).run(35)
    np.testing.assert_array_equal(split.chains, whole.chains)


def test_parallel_chains_match_serial(onehot):
    m = PopulationModel(onehot[0], P.LogisticRandomWalk(order=1))
    serial = NUTS(m, n_chains=2, n_warmup=40, rng=5).run(15)
    eng = NUTS(m, n_chains=2, n_warmup=40, rng=5, n_jobs=2)
    parallel = eng.run(15)
    np.testing.assert_array_equal(serial.chains, parallel.chains)
    np.testing.assert_array_equal(serial.info["n_leapfrog"], parallel.info["n_leapfrog"])
    assert eng.run(5).chains.shape == (2, 20, 4)  # resuming works across processes
    with pytest.raises(ValueError):
        NUTS(m, n_jobs=0)


def test_hessian_helpers():
    from binference.samplers._nuts_core import covariance_from_hessian, fd_hessian

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
    assert covariance_from_hessian(np.full((2, 2), np.nan)) is None


def test_nuts_reparametrises_logistic_priors_without_touching_the_model():
    mock = binference.simulate.mock_1d(5000, Bins.linear(0, 3, 8), noise=(0.05, 0.1), rng=2)
    m = PopulationModel(mock.likelihoods, P.LogisticRandomWalk(order=1))
    eng = NUTS(m, n_chains=2, n_warmup=150, rng=1)
    assert eng.metric == "dense" and eng.target_accept == 0.9
    res = eng.run(40)
    assert np.all(m.prior.lam == 1.0)  # the user's model keeps its parametrisation
    assert eng.model is not m
    assert not np.all(eng.model.prior.lam == 1.0)  # strong data: some modes centred
    assert res.model is eng.model
    # draws map to the same fractions under the engine's parametrisation
    np.testing.assert_allclose(eng.model.to_value(res.x[0, -1]), res.chains[0, -1])
    plain = NUTS(m, n_chains=2, n_warmup=60, rng=1, reparametrize=False, metric="diag")
    assert plain.metric == "diag" and plain.run(5).model.prior is m.prior
    dirichlet = NUTS(PopulationModel(mock.likelihoods), n_chains=1)
    assert dirichlet.metric == "diag" and dirichlet.target_accept == 0.8


def test_nuts_validation():
    m = PopulationModel(np.ones((3, 2)))
    with pytest.raises(ValueError):
        NUTS(m, n_chains=0)
    with pytest.raises(ValueError):
        NUTS(m, metric="full")
    with pytest.raises(ValueError):
        NUTS(m, target_accept=1.5)
    with pytest.raises(ValueError):
        NUTS(m, init=np.zeros(5)).run(1)
    with pytest.raises(RuntimeError):
        _ = NUTS(m).result


def test_nuts_logistic_prior_runs():
    mock = binference.simulate.mock_1d(300, Bins.linear(0, 3, 8), rng=4)
    res = binference.infer(
        mock.likelihoods,
        P.LogisticRandomWalk(order=2),
        rng=1,
        n_chains=2,
        n_warmup=200,
        n_samples=200,
        check=False,
    )
    assert res.samples.shape == (400, 8)
    assert "scale" in res.info["hyperparameters"]


# -------------------------------------------------------------------- Gibbs
def test_gibbs_matches_exact(k3_problem):
    mock, exact = k3_problem
    res = binference.infer(
        mock.likelihoods, method="gibbs", rng=3, n_samples=3000, n_warmup=200, check=False
    )
    assert_matches_reference(res, exact.mean, exact.std)


def test_gibbs_onehot_is_conjugate(onehot):
    lk, counts = onehot
    res = binference.infer(lk, method="gibbs", rng=4, n_samples=3000, n_warmup=10, check=False)
    mean, sd = dirichlet_moments(1.0 + counts)
    assert_matches_reference(res, mean, sd)


def test_gibbs_selection_matches_exact(k3_selection_problem):
    mock, exact = k3_selection_problem
    res = binference.infer(
        mock.likelihoods, method="gibbs", rng=5, n_samples=3000, n_warmup=200, check=False
    )
    assert_matches_reference(res, exact.mean, exact.std)
    nuts = binference.infer(
        mock.likelihoods, rng=5, n_chains=2, n_warmup=300, n_samples=1500, check=False
    )
    assert_matches_reference(nuts, exact.mean, exact.std)


@pytest.mark.parametrize("mode", ["on_truth", "on_measurement"])
def test_fraction_selection_only_depends_on_ratios(k3_selection_problem, mode):
    """S and 3 S (values above one) give the same posterior for fractions."""
    mock, _ = k3_selection_problem
    lk = mock.likelihoods.with_selection(None)
    S = np.array([0.9, 0.5, 0.25])
    fits = {}
    for scale in (1.0, 3.0):
        m = PopulationModel(lk, selection=Selection(scale * S, mode, scale="relative"))
        est = MAP(m).run()
        gibbs = Gibbs(m, n_chains=2, n_warmup=200, rng=3).run(1500)
        fits[scale] = (est.rho, gibbs.mean, gibbs.std)
        assert np.all(gibbs.samples > 0)
    np.testing.assert_allclose(fits[3.0][0], fits[1.0][0], rtol=1e-6)
    np.testing.assert_allclose(fits[3.0][1], fits[1.0][1], atol=0.02)
    exact_m = PopulationModel(lk, selection=S, selection_mode=mode)
    nuts = NUTS(exact_m, n_chains=2, n_warmup=300, rng=4).run(800)
    np.testing.assert_allclose(fits[3.0][1], nuts.mean, atol=0.02)


def test_gibbs_rates_agree_with_nuts(k3_selection_problem):
    mock, _ = k3_selection_problem
    prior = P.GammaRates(1.0, 1e-3)
    g = binference.infer(
        mock.likelihoods, prior, method="gibbs", rng=6, n_samples=3000, check=False
    )
    n = binference.infer(
        mock.likelihoods, prior, rng=6, n_chains=2, n_warmup=300, n_samples=1500, check=False
    )
    assert g.rates is not None and n.rates is not None
    np.testing.assert_allclose(g.rates.mean(0), n.rates.mean(0), rtol=0.06)
    np.testing.assert_allclose(g.rates.std(0), n.rates.std(0), rtol=0.15)
    np.testing.assert_allclose(g.samples.sum(1), 1.0)


@pytest.mark.parametrize("prior", [P.GeneralizedDirichlet(2.0), P.PolyaTree(1.0)], ids=repr)
def test_gibbs_tree_priors_agree_with_nuts(prior):
    mock = binference.simulate.mock_1d(200, Bins.linear(0, 3, 4), noise=(0.2, 0.6), rng=9)
    g = binference.infer(
        mock.likelihoods, prior, method="gibbs", rng=7, n_samples=3000, check=False
    )
    n = binference.infer(
        mock.likelihoods, prior, rng=7, n_chains=2, n_warmup=300, n_samples=1500, check=False
    )
    assert_matches_reference(g, n.mean, n.std, z=6, sd_rtol=0.2)


def test_gibbs_validation_and_labels(onehot):
    lk, _ = onehot
    with pytest.raises(TypeError):
        Gibbs(PopulationModel(lk, P.LogisticNormal()))
    with pytest.warns(InputWarning, match="pseudo-likelihood"):
        half = lk.with_weights(np.full(lk.N, 0.5), fractional=True)
    with pytest.raises(ValueError, match="integer"):
        Gibbs(PopulationModel(half))
    g = Gibbs(PopulationModel(lk), n_chains=2, n_warmup=5, store_labels=[0, 3], rng=1)
    res = g.run(50)
    assert res.labels.shape == (2, 50, 2)
    np.testing.assert_array_equal(res.label_index, [0, 3])
    a = Gibbs(PopulationModel(lk), n_warmup=5, rng=2).run(10)
    b = Gibbs(PopulationModel(lk), n_warmup=5, rng=2).run(10)
    np.testing.assert_array_equal(a.samples, b.samples)


def test_gibbs_integer_weights_compressed(rng):
    L = np.array([[1.0, 0.5, 0.1], [0.2, 1.0, 0.3], [0.1, 0.4, 1.0]])
    rows = rng.integers(0, 3, size=300)
    lk_full = Likelihoods(L[rows])
    lk_comp = lk_full.compress()
    assert lk_comp.N == 3
    a = binference.infer(lk_full, method="gibbs", rng=1, n_samples=4000, check=False)
    b = binference.infer(lk_comp, method="gibbs", rng=1, n_samples=4000, check=False)
    assert_matches_reference(b, a.mean, a.std, z=6, sd_rtol=0.2)


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
    assert est.converged and est.kkt_gap < 1e-8
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


def test_map_squarem_faster_than_em():
    mock = binference.simulate.mock_1d(1000, Bins.linear(0, 3, 15), noise=(0.3, 0.6), rng=1)
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
def test_mixsqp_matches_em_and_is_certified(alpha, sparse):
    mock = binference.simulate.mock_1d(1500, Bins.linear(0, 3, 20), noise=(0.2, 0.5), rng=3)
    lk = mock.likelihoods.to_sparse(1e-8) if sparse else mock.likelihoods
    m = PopulationModel(lk, P.Dirichlet(alpha))
    sqp = MAP(m).run()
    assert sqp.info["algorithm"] == "mixsqp"
    # mix-SQP stops on round-off with the KKT gap below 1e3 times its tolerance of
    # 100 * tol = 1e-8 (macOS BLAS stops there at 2.5e-7 where Linux reaches 1e-13)
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


def test_mixsqp_low_rank_path_and_fallbacks():
    mock = binference.simulate.mock_1d(6000, Bins.linear(0, 3, 30), noise=(0.1, 0.3), rng=4)
    m = PopulationModel(mock.likelihoods)  # tall and dense: low-rank Hessian
    est = MAP(m).run()
    assert est.info["algorithm"] == "mixsqp" and est.kkt_gap < 1e-7
    # informative priors with selection fall back to EM; asking for mix-SQP there is an error
    sel = PopulationModel(
        mock.likelihoods,
        P.Dirichlet(2.0),
        selection=np.linspace(1, 0.5, 30),
        selection_mode="on_truth",
    )
    assert MAP(sel).run().info["algorithm"] == "squarem"
    with pytest.raises(ValueError, match="mix-SQP"):
        MAP(sel, algorithm="mixsqp").run()
    with pytest.raises(ValueError):
        MAP(m, algorithm="newton")


def test_mixsqp_stops_at_the_round_off_floor():
    # the KKT gap of this optimum cannot be pushed much below 1e-8 in floating point
    mock = binference.simulate.mock_1d(800, Bins.linear(0, 3, 15), rng=6)
    est = MAP(PopulationModel(mock.likelihoods)).run()  # warnings are errors here
    assert est.converged and est.n_iter < 50 and est.kkt_gap < 1e-6


@pytest.mark.parametrize("mode", ["on_truth", "on_measurement"])
def test_mixsqp_with_selection_solves_for_observed_fractions(mode):
    mock = binference.simulate.mock_1d(3000, Bins.linear(0, 3, 20), noise=(0.1, 0.3), rng=2)
    S = np.linspace(0.2, 1.0, 20) ** 2
    m = PopulationModel(mock.likelihoods, selection=Selection(3.0 * S, mode, scale="relative"))
    sqp = MAP(m).run()
    assert sqp.info["algorithm"] == "mixsqp"
    assert sqp.converged and sqp.n_iter < 50 and sqp.kkt_gap < 1e-7
    em = MAP(m, algorithm="em", max_iter=200000, tol=1e-14).run()
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
    pts = exact.points
    logp = np.array([m.log_posterior(p) for p in pts[::7]])
    best = pts[::7][np.argmax(logp)]
    np.testing.assert_allclose(est.rho, best, atol=5e-3)
    assert est.kkt_gap < 1e-7


def test_map_tree_prior_matches_grid():
    mock = binference.simulate.mock_1d(150, Bins.linear(0, 3, 3), noise=(0.3, 0.8), rng=2)
    m = PopulationModel(mock.likelihoods, P.GeneralizedDirichlet([3.0, 2.0], [4.0, 2.5]))
    est = MAP(m).run()
    ex = binference.validation.exact_posterior(m, n_grid=600)
    best = ex.points[np.argmax(ex.weights)]
    np.testing.assert_allclose(est.rho, best, atol=3e-3)


def test_map_unconstrained_and_logistic():
    mock = binference.simulate.mock_1d(300, Bins.linear(0, 3, 8), rng=3)
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
def test_laplace_with_importance(k3_problem):
    mock, exact = k3_problem
    res = binference.infer(mock.likelihoods, method="laplace", rng=4, n_samples=4000)
    assert "k_hat" in res.info
    np.testing.assert_allclose(res.mean, exact.mean, atol=0.01)
    np.testing.assert_allclose(res.std, exact.std, rtol=0.15)


def test_laplace_uncorrected_is_flagged(k3_problem):
    mock, _ = k3_problem
    with pytest.warns(ApproximationWarning):
        res = Laplace(PopulationModel(mock.likelihoods), importance=False, rng=1).run(500)
    assert res.approximate


# ----------------------------------------------------------------------- VB
def test_vb_exact_for_onehot_and_overconfident_otherwise(onehot, k3_problem):
    lk, counts = onehot
    with pytest.warns(ApproximationWarning):
        res = MeanFieldVB(PopulationModel(lk), rng=1).run(20000)
    mean, sd = dirichlet_moments(1.0 + counts)
    np.testing.assert_allclose(res.mean, mean, atol=3e-3)
    np.testing.assert_allclose(res.std, sd, rtol=0.05)
    mock, exact = k3_problem
    with pytest.warns(ApproximationWarning):
        vb = MeanFieldVB(PopulationModel(mock.likelihoods), rng=1).run(4000)
    assert vb.approximate
    assert np.all(vb.std < 0.8 * exact.std)
    with pytest.raises(TypeError):
        MeanFieldVB(PopulationModel(lk, P.LogisticNormal()))


# ---------------------------------------------------------------- Bootstrap
def test_bootstrap(k3_problem):
    mock, exact = k3_problem
    with pytest.warns(ApproximationWarning):
        res = Bootstrap(PopulationModel(mock.likelihoods), rng=1).run(60)
    assert res.samples.shape == (60, 3) and res.approximate
    assert np.all(res.std < 2.5 * exact.std) and np.all(res.std > 0.3 * exact.std)
    with pytest.warns(ApproximationWarning):
        bb = Bootstrap(PopulationModel(mock.likelihoods), kind="bayesian", rng=1).run(20)
    assert bb.info["kind"] == "bayesian"
    with pytest.raises(ValueError):
        Bootstrap(PopulationModel(mock.likelihoods), kind="parametric")
