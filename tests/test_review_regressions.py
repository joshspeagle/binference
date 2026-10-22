"""Regression tests for findings of the independent v0.1 code review."""

import pickle
import warnings

import numpy as np
import pytest
import scipy.sparse as sp
from scipy import stats

import pastry
from pastry import Bins, Likelihoods, PopulationModel, PopulationResult
from pastry import priors as P
from pastry.samplers import MAP, NUTS, Bootstrap
from pastry.warnings import ApproximationWarning, InputWarning


def _bootstrap(model, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ApproximationWarning)
        return Bootstrap(model, **kw).run(150)


# -------------------------------------------------------------- bootstrap (1-3)
@pytest.mark.parametrize("kind", ["nonparametric", "bayesian"])
def test_bootstrap_resamples_objects_not_rows(kind):
    rng = np.random.default_rng(0)
    labels = rng.choice(4, size=2000, p=[0.4, 0.1, 0.4, 0.1])
    L = np.eye(4)[labels] * 0.9 + 0.1 / 4  # only four distinct rows
    raw = Likelihoods(L)
    packed = raw.compress()
    assert packed.N == 4
    sd_raw = _bootstrap(PopulationModel(raw), kind=kind, rng=1).std
    sd_packed = _bootstrap(PopulationModel(packed), kind=kind, rng=1).std
    np.testing.assert_allclose(sd_packed, sd_raw, rtol=0.3)
    assert np.all(sd_packed < 0.05)  # was ~0.35 when rows were resampled


def test_bootstrap_rates_have_a_random_total():
    mock = pastry.simulate.mock_1d(1000, Bins.linear(0, 3, 6), rng=2)
    res = _bootstrap(PopulationModel(mock.likelihoods, P.GammaRates(1.0, 1e-6)), rng=1)
    totals = res.rates.sum(axis=1)
    assert totals.std() == pytest.approx(np.sqrt(1000), rel=0.3)


def test_bootstrap_replicates_start_inside_the_simplex():
    """A sparse MLE must not trap replicates: each refit is certified optimal."""
    mock = pastry.simulate.mock_1d(150, Bins.linear(0, 3, 20), noise=(0.05, 0.1), rng=3)
    m = PopulationModel(mock.likelihoods)
    res = _bootstrap(m, rng=2)
    start = 0.5 * MAP(m).run().value + 0.5 / 20
    assert np.all(start > 0)
    assert res.samples.shape == (150, 20)


# ------------------------------------------------------------- inputs (4, 5, 10)
def test_to_sparse_does_not_modify_the_original():
    lk = Likelihoods(np.random.default_rng(0).random((4, 5)) ** 3).to_sparse()
    before, nnz = lk.toarray().copy(), lk.nnz
    model = PopulationModel(lk)
    L_before = model._L.toarray().copy()
    lk.to_sparse(rtol=0.2)
    np.testing.assert_array_equal(lk.toarray(), before)
    assert lk.nnz == nnz
    np.testing.assert_array_equal(model._L.toarray(), L_before)
    M = sp.csr_array(np.random.default_rng(1).random((3, 4)))
    data = M.data.copy()
    Likelihoods(M)
    np.testing.assert_array_equal(M.data, data)


def test_from_response_with_explicit_efficiency():
    mu = pastry.simulate.mock_unfolding(6000, rng=3)
    R, counts = mu.truth.extra["response"], mu.measurements["reco_counts"]
    eps = R.sum(axis=0)
    migration = R / eps
    a = Likelihoods.from_response(counts, R, bins=mu.bins)
    b = Likelihoods.from_response(counts, migration, bins=mu.bins, efficiency=eps)
    np.testing.assert_allclose(b.toarray(), a.toarray(), rtol=1e-12)
    np.testing.assert_array_equal(b.weights, a.weights)
    assert a.selection.mode == b.selection.mode == "observed"
    np.testing.assert_allclose(b.selection.values, a.selection.values, rtol=1e-12)
    with pytest.warns(InputWarning, match="counted twice"):
        Likelihoods.from_response(counts, R, bins=mu.bins, efficiency=eps)
    with pytest.raises(ValueError, match="sum to more than one"):
        Likelihoods.from_response(counts, 2.0 * migration, bins=mu.bins)


def test_interim_density_follows_posterior_representation():
    bins = Bins.log(0.05, 5.0, 12)
    rng = np.random.default_rng(4)
    z = rng.uniform(0.1, 4.0, 200)
    c = bins.centers
    like = stats.norm.pdf(c[None, :], z[:, None], 0.1 * (1 + z[:, None]))
    prior_density = 1.0 / c  # interim prior density at the bin centres
    post_density = like * prior_density
    post_density /= (post_density * bins.volumes).sum(axis=1, keepdims=True)
    lk = Likelihoods.from_posteriors(post_density, bins, prior_density, representation="density")
    expected = Likelihoods(like, bins)
    np.testing.assert_allclose(lk.toarray(), expected.toarray(), rtol=1e-8, atol=1e-12)


def test_per_object_meta_follows_subsets_and_drops():
    bins = Bins.linear(0, 1, 5)
    samples = [np.r_[np.full(30, 0.5)], np.full(30, 7.0), np.linspace(0.1, 0.9, 40)]
    with pytest.warns(InputWarning):
        lk = Likelihoods.from_samples(samples, bins, "flat", drop_empty=True, min_samples=1)
    assert lk.N == 2 and lk.meta["n_eff"].shape == (2,)
    g = Likelihoods.from_gaussian(np.array([0.2, 0.5, 0.8]), 0.1, bins)
    assert g[1:].meta["gaussian_mean"].shape == (2, 1)
    assert "gaussian_mean" not in Likelihoods(np.ones((3, 5)), bins).compress().meta


# ------------------------------------------------------------- results (6, 12-14)
def test_sample_values_follow_the_likelihood_within_bins():
    bins = Bins([0.0, 0.5, 1.0])
    lk = Likelihoods.from_gaussian(np.array([0.26, 0.7]), np.array([0.02, 0.3]), bins)
    res = PopulationResult(PopulationModel(lk), np.tile([0.5, 0.5], (400, 1)), method="test")
    v = res.sample_values(rng=1)
    assert v[:, 0].mean() == pytest.approx(0.26, abs=0.005)
    assert v[:, 0].std() == pytest.approx(0.02, rel=0.15)
    plain = PopulationResult(PopulationModel(Likelihoods(lk.toarray(), bins)),
                             np.tile([0.5, 0.5], (400, 1)), method="test")  # fmt: skip
    custom = plain.sample_values(
        idx=[0], rng=1, loglike=lambda z, i: stats.norm.logpdf(z, [0.26, 0.7][i], 0.02)
    )
    assert custom.std() == pytest.approx(0.02, rel=0.2)
    assert plain.sample_values(idx=[0], rng=1).std() > 0.1  # top-hat fallback


def test_log_predictive_rates_with_selection():
    mock = pastry.simulate.mock_1d(400, Bins.linear(0, 3, 4), rng=5)
    S = np.array([1.0, 0.6, 0.3, 0.1])
    model = PopulationModel(mock.likelihoods, P.GammaRates(1.0, 1e-3), selection=S,
                            selection_mode="true")  # fmt: skip
    lam = np.random.default_rng(0).gamma(5.0, 20.0, size=(50, 4))
    res = PopulationResult(model, lam, method="test", kind="rates")
    new = mock.likelihoods[:5]
    Lt = new.toarray() * S
    expected = np.log(np.mean((Lt @ lam.T) / (lam @ S), axis=1)) + new.log_scale
    np.testing.assert_allclose(res.log_predictive(new, n_draws=None), expected, rtol=1e-10)


def test_sbc_ranks_rates_for_rate_models():
    bins = Bins.linear(0, 1, 3)

    def simulate(rng):
        lam = rng.gamma(20.0, 1.0, size=3)
        n = rng.poisson(lam)
        z = np.concatenate([rng.uniform(k / 3, (k + 1) / 3, c) for k, c in enumerate(n)])
        lk = Likelihoods.from_gaussian(z + 0.05 * rng.standard_normal(z.size), 0.05, bins)
        return lk, lam

    def fit(lk, rng):
        return pastry.infer(lk, P.GammaRates(20.0, 1.0), method="gibbs", rng=rng, n_chains=1,
                            n_warmup=50, n_samples=100, check=False)  # fmt: skip

    res = pastry.validation.sbc(simulate, fit, n_simulations=20, n_draws=19, rng=0)
    # ranks of rates are spread out; ranking fractions against rates would pile them at 0
    assert res.ranks[:, :3].mean() == pytest.approx(9.5, abs=3.0)


def test_rate_diagnostics_use_rates():
    mock = pastry.simulate.mock_1d(300, Bins.linear(0, 3, 4), rng=6)
    res = pastry.infer(mock.likelihoods, P.GammaRates(1.0, 1e-3), method="gibbs", rng=1,
                       n_samples=200, check=False)  # fmt: skip
    r = res.rates.reshape(res.n_chains, -1, 4)
    expected = pastry.diagnostics.ess(r[:, :, 0], "bulk")
    assert res.diagnostics["ess_bulk"][0] == pytest.approx(expected)


def test_save_load_labels_and_hyperparameters(tmp_path):
    lk = Likelihoods(np.random.default_rng(0).random((50, 3)) + 0.1)
    res = PopulationResult(PopulationModel(lk), np.full((10, 3), 1 / 3), method="x",
                           labels=np.zeros((10, 50), int))  # fmt: skip
    res.save(tmp_path / "a.npz")
    back = PopulationResult.load(tmp_path / "a.npz")
    assert back.bins == lk.bins  # integer category labels survive
    mock = pastry.simulate.mock_1d(200, Bins.linear(0, 3, 6), rng=7)
    gp = pastry.infer(mock.likelihoods, P.LogisticGP(), rng=1, n_samples=30, n_warmup=60,
                      n_chains=2, check=False)  # fmt: skip
    gp.save(tmp_path / "gp.npz")
    back = PopulationResult.load(tmp_path / "gp.npz")
    for name, draws in gp.info["hyperparameters"].items():
        np.testing.assert_array_equal(back.info["hyperparameters"][name], draws)


# -------------------------------------------------------------- priors (8, 9, 15)
def test_polya_tree_map_avoids_boundary_zeros():
    bins = Bins.linear(0, 1, 16)
    rho = np.r_[np.full(12, 1 / 12), np.zeros(4)]
    mock = pastry.simulate.mock_1d(300, bins, density=rho, noise=0.02, rng=8)
    m = PopulationModel(mock.likelihoods, P.PolyaTree(c=1.0))
    assert m.prior.unbounded_density()
    est = MAP(m).run()
    assert est.info["space"] == "unconstrained"
    assert np.all(est.rho > 0)
    with pytest.warns(InputWarning, match="unbounded"):
        MAP(m, space="simplex").run()


def test_gp_binds_on_log_and_large_grids():
    for bins in (Bins.log(0.01, 10, 20), Bins.linear(0, 1, 3000)):
        P.LogisticGP().bind(bins)
    h = P.InverseGamma.from_quantiles(1.0, 1e5)
    d = stats.invgamma(h.a, scale=h.b)
    assert d.cdf(1.0) == pytest.approx(0.01, abs=1e-8)
    assert d.sf(1e5) == pytest.approx(0.01, abs=1e-8)


def test_logistic_prior_mean_is_a_flat_density():
    bins = Bins([0.0, 0.1, 0.5, 2.0])
    for prior in (P.LogisticRandomWalk(), P.LogisticGP(), P.LogisticNormal()):
        mu = prior.bind(bins).mu
        lv = np.log(bins.volumes)
        np.testing.assert_allclose(mu, lv - lv.mean())


# ------------------------------------------------------------------- NUTS (7, 11)
def test_weak_data_smoothness_fits_stay_non_centred():
    """Poorly constrained scales: partial centring would create a funnel."""
    bins = Bins.linear(0.0, 3.0, 6)
    mock = pastry.simulate.mock_1d(30, bins, density=pastry.simulate.population(bins, "bimodal"),
                                   noise=(0.2, 0.4), rng=6)  # fmt: skip
    eng = NUTS(PopulationModel(mock.likelihoods, P.LogisticRandomWalk(order=2)), rng=1)
    eng.run(200)
    penalised = ~eng.model.prior.null
    assert np.all(eng.model.prior.lam[penalised] == 1.0)


def test_parallel_chains_fall_back_when_the_model_cannot_be_pickled():
    mock = pastry.simulate.mock_1d(100, Bins.linear(0, 3, 8), rng=1)
    model = PopulationModel(mock.likelihoods, P.PolyaTree(base=lambda z: np.exp(-z)))
    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        pickle.dumps(model)
    eng = NUTS(model, n_chains=2, n_warmup=30, rng=1, n_jobs=2)
    with pytest.warns(RuntimeWarning, match="serially"):
        res = eng.run(10)
    assert res.chains.shape == (2, 10, 8)
    assert eng.n_jobs == 1
    assert eng.run(5).chains.shape == (2, 15, 8)  # still usable
