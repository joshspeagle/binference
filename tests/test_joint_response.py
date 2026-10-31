"""A response tallied from events, inferred jointly with the population (DECISIONS #28, #81)."""

import pickle

import numpy as np
import pytest

import binference
from binference import Bins, Likelihoods, Response
from binference.results import _draws_rng
from binference.samplers import MAP, Gibbs, MeanFieldVB
from binference.validation import SBCResult

from .conftest import assert_draws_agree, assert_uniform_ranks


def _unfolding(n_sim=2000, n_obs=400, seed=1):
    rng = np.random.default_rng(seed)
    truth, measured = Bins.linear(0, 1, 4), Bins.linear(-0.2, 1.2, 7)
    zt = rng.uniform(0, 1, n_sim)
    zm = zt + 0.1 * rng.standard_normal(n_sim)
    zm[rng.random(n_sim) < 0.2] = np.nan  # lost events
    resp = Response.from_simulation(zt, zm, truth, measured)
    obs = rng.uniform(0, 1, n_obs) + 0.1 * rng.standard_normal(n_obs)
    return Likelihoods.from_binned(resp, values=np.clip(obs, -0.19, 1.19)), resp


def test_a_tallied_response_has_a_dirichlet_posterior_and_a_fixed_plug_in():
    _, resp = _unfolding()
    assert resp.uncertain and not resp.fixed().uncertain
    np.testing.assert_array_equal(resp.fixed().matrix, resp.matrix)
    a = resp._dirichlet()
    J = resp.matrix.shape[0]
    np.testing.assert_allclose(a.sum(axis=0), resp.generated + (J + 1) * resp.prior)
    np.testing.assert_allclose(a[:-1], resp.counts + resp.prior)
    draws = resp.draw(4000, rng=0)
    mean = (a / a.sum(axis=0))[:-1]
    sd = np.sqrt(mean * (1 - mean) / (a.sum(axis=0) + 1))
    np.testing.assert_array_less(np.abs(draws.mean(0) - mean), 5 * sd / np.sqrt(4000) + 1e-12)
    assert pickle.loads(pickle.dumps(resp)).prior == resp.prior
    with pytest.raises(ValueError, match="prior"):
        Response.from_counts(resp.counts, resp.generated, resp.truth_bins, prior=0.0)


def test_the_joint_density_has_the_gradient_of_its_values():
    lk, _ = _unfolding()
    rng = np.random.default_rng(3)
    for prior in (None, binference.priors.LogisticRandomWalk()):
        m = binference.PopulationModel(lk, prior)
        assert m.joint and m.dim == m.prior.dim + 4 * 7
        x = m.from_value(np.full(4, 0.25)) + 0.3 * rng.standard_normal(m.dim)
        _, g = m.log_density_unconstrained_and_grad(x)
        h = 1e-5
        fd = [
            (m.log_density_unconstrained(x + h * e) - m.log_density_unconstrained(x - h * e))
            / (2 * h)
            for e in np.eye(m.dim)
        ]
        np.testing.assert_allclose(g, fd, rtol=1e-6, atol=1e-6)


def test_engines_refuse_or_infer_the_response_and_report_its_draws():
    lk, resp = _unfolding()
    m = binference.PopulationModel(lk)
    with pytest.raises(ValueError, match=r"response\.fixed\(\)"):
        MeanFieldVB(m)
    with pytest.raises(ValueError, match="selection"):
        binference.PopulationModel(lk, selection=np.ones(4), selection_mode="on_truth")
    with pytest.raises(ValueError, match="fractions only"):
        binference.PopulationModel(lk, binference.priors.GammaRates(1.0, 1e-3))
    assert binference.inference.Auto(m).candidates()[0][0] == "gibbs"
    est = MAP(m).run()
    R = est.info["response_draws"][0, 0]
    assert R.shape == (7, 4) and est.info["space"] == "unconstrained"
    assert est.total == pytest.approx(m.W / (R.sum(axis=0) @ est.rho))
    a = resp._dirichlet()
    P = np.vstack([R, 1.0 - R.sum(axis=0)])
    log_dir = np.sum((a - 1.0) * np.log(P))
    expect = m.log_likelihood(est.rho, response=R) + m.log_prior(est.rho) + log_dir
    assert est.log_posterior == pytest.approx(expect)
    with pytest.raises(ValueError, match="shape"):
        m.log_likelihood(est.rho, response=R[:-1])
    with pytest.raises(ValueError, match="at most one"):
        m.log_likelihood(est.rho, response=2 * R)
    fixed = Likelihoods.from_binned(resp.fixed(), values=np.full(3, 0.5))
    assert not binference.PopulationModel(fixed).joint
    # the rows and the selection must be those from_binned built
    for changed in (lk.with_selection(np.full(4, 0.5), "on_measurement"), lk.compress()):
        with pytest.raises(ValueError, match=r"response\.fixed\(\)"):
            binference.PopulationModel(changed)


def test_per_object_methods_and_the_total_use_each_draws_response():
    lk, _ = _unfolding()
    res = Gibbs(binference.PopulationModel(lk), n_warmup=30, rng=0).run(40)
    Rd = res.info["response_draws"].reshape(-1, 7, 4)
    rows = lk.meta["measured_bin"]
    expect = np.mean(
        [R[rows] * r / (R[rows] @ r)[:, None] for r, R in zip(res.samples, Rd, strict=True)],
        axis=0,
    )
    np.testing.assert_allclose(res.assignment_probabilities(n_draws=None), expect)
    eff = res.info["response_draws"].sum(axis=-2)  # (chain, draw, K), as the draws
    total = res.model.draw_total(res._params, _draws_rng(res._params), eff)
    np.testing.assert_allclose(res.total, total.ravel())
    assert "ess_bulk_response" in res.diagnostics  # the efficiencies'
    assert res.sample_assignments(n_draws=5, rng=1).shape == (5, lk.N)
    try:
        import xarray  # noqa: F401
    except ImportError:
        return
    assert res.to_datatree()["posterior"]["response"].shape == (4, 40, 7, 4)


def test_the_predictive_check_with_response_draws_leaves_out_the_tallies_covariance():
    lk, resp = _unfolding()
    n = np.bincount(lk.meta["measured_bin"], weights=lk.weights, minlength=7)
    rho = np.full((3, 4), 0.25)
    same = np.broadcast_to(resp.matrix, (3, 7, 4))
    p_draws = resp.predictive_pvalue(n, rho, same)
    known = Response(resp.matrix, resp.truth_bins, resp.measured_bins)  # no tallies
    assert p_draws == pytest.approx(known.predictive_pvalue(n, rho))
    assert p_draws != pytest.approx(resp.predictive_pvalue(n, rho))
    # the plug-in keeps the tallies' covariance (DECISIONS #68)
    assert resp.fixed().predictive_pvalue(n, rho) == pytest.approx(resp.predictive_pvalue(n, rho))
    with pytest.raises(ValueError, match="response_draws"):
        resp.predictive_pvalue(n, rho, same[:2])


@pytest.mark.slow
def test_nuts_agrees_with_joint_gibbs_on_population_and_response():
    lk, _ = _unfolding()
    m = binference.PopulationModel(lk)
    fits = [binference.samplers.NUTS(m, rng=1).run(1500), Gibbs(m, rng=1).run(5000)]
    # the fractions and each truth bin's efficiency, the response summed over measured bins
    draws = [np.concatenate([r.chains, r.info["response_draws"].sum(-2)], -1) for r in fits]
    assert_draws_agree(*draws, z=4)


@pytest.mark.slow
def test_joint_inference_of_population_and_response_is_calibrated():
    # SBC over (rho, R): rho ~ Dir(1); each column of R with its lost share ~ the
    # response's Dirichlet prior; 40 simulated events per truth bin; 150 observed objects
    K, J = 3, 5
    truth, measured = Bins.categories(K), Bins.categories(J)
    rng = np.random.default_rng(2028)
    n_draws, ranks = 49, []
    for _ in range(150):
        rho = rng.dirichlet(np.ones(K))
        cols = rng.dirichlet(np.ones(J + 1), size=K).T  # (J + 1, K): the default beta = 1
        counts = np.stack([rng.multinomial(40, cols[:, k]) for k in range(K)], axis=1)[:J]
        resp = Response.from_counts(counts, np.full(K, 40), truth, measured)
        R = cols[:J]
        p = (R * rho).ravel()  # observed objects in (measured, truth) cells
        cells = rng.multinomial(150, p / p.sum()).reshape(J, K)
        lk = Likelihoods.from_binned(resp, counts=np.maximum(cells.sum(axis=1), 0))
        res = Gibbs(binference.PopulationModel(lk), n_chains=2, n_warmup=200, rng=rng).run(500)
        idx = np.linspace(0, res.n_draws - 1, n_draws).round().astype(int)
        eff = res.info["response_draws"].sum(axis=-2).reshape(-1, K)[idx]
        ranks.append(
            np.concatenate([(res.samples[idx] < rho).sum(0), (eff < R.sum(axis=0)).sum(0)])
        )
    names = [f"rho[{k}]" for k in range(K)] + [f"eff[{k}]" for k in range(K)]
    assert_uniform_ranks(SBCResult(np.array(ranks), n_draws, names))


def test_objects_in_a_measured_bin_without_tallied_events_are_kept():
    # an inferred response gives every measured bin some probability; the SBC met this
    truth, measured = Bins.categories(2), Bins.categories(3)
    resp = Response.from_counts([[5, 0], [3, 6], [0, 0]], [10, 10], truth, measured)
    lk = Likelihoods.from_binned(resp, counts=[4, 5, 2])
    assert lk.N == 3 and np.all(lk.toarray()[2] > 0)
    with pytest.raises(ValueError, match=r"Measured bin\(s\) \[2\].*uncertain"):
        Likelihoods.from_binned(resp.fixed(), counts=[4, 5, 2])


def test_a_response_tallied_without_losses_has_no_lost_share():
    # a classifier's open-ended score bins observe every object: efficiency one
    mc = binference.simulate.mock_categorical(300, 3, rng=3)
    resp = mc.response
    J, K = resp.matrix.shape
    assert resp._dirichlet().shape == (J, K)
    lk = Likelihoods.from_binned(resp, values=mc.measurements["feature"])
    m = binference.PopulationModel(lk)
    assert lk.selection is None and m.dim == m.prior.dim + K * (J - 1)
    res = Gibbs(m, n_warmup=20, rng=0).run(20)
    np.testing.assert_allclose(res.info["response_draws"].sum(axis=-2), 1.0)
    ones = Likelihoods.from_binned(resp, counts=np.ones(J))  # unit weights: labels allowed
    with pytest.raises(ValueError, match="store_labels"):
        Gibbs(binference.PopulationModel(ones), store_labels=True)


def test_new_binned_objects_take_each_draws_response():
    lk, resp = _unfolding()
    res = Gibbs(binference.PopulationModel(lk), n_warmup=30, rng=0).run(40)
    new = Likelihoods.from_binned(resp, values=[0.1, 0.5, 0.5, 0.9])
    Rd = res.info["response_draws"].reshape(-1, 7, 4)
    rows = new.meta["measured_bin"]
    like = np.array(
        [R[rows] @ r / (R.sum(axis=0) @ r) for r, R in zip(res.samples, Rd, strict=True)]
    )
    expect = np.log(like.mean(axis=0))
    np.testing.assert_allclose(res.log_predictive(new, n_draws=None), expect)
    probs = np.mean(
        [R[rows] * r / (R[rows] @ r)[:, None] for r, R in zip(res.samples, Rd, strict=True)],
        axis=0,
    )
    np.testing.assert_allclose(res.predict_proba(new, n_draws=None), probs)


def test_dropping_rows_keeps_each_rows_measured_bin():
    lk, _ = _unfolding()
    m = binference.PopulationModel(lk)
    w = m._w.copy()
    w[0] = 0.0  # the first measured bin's row is dropped
    new = m._with_weights(w)
    assert new._mb.shape == (new.likelihoods.N,)
    np.testing.assert_array_equal(new._mb, m._mb[1:])
    x = new.from_value(np.full(m.K, 1 / m.K))
    assert np.isfinite(new.log_density_unconstrained(x))
