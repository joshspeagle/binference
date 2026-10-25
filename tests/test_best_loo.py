"""The best fit with every result, `_best` outputs and leave-one-out posteriors
(DECISIONS #35, #51, #52)."""

import numpy as np
import pytest

import binference
from binference import Bins, Likelihoods, Selection
from binference.diagnostics import ess, psis
from binference.samplers import MAP, Gibbs

BINS = Bins.linear(0, 1, 5)


def _lk():
    # 30 objects in the first three bins and one (object 0) alone in the last
    rng = np.random.default_rng(0)
    z = np.concatenate([[0.92], rng.uniform(0.0, 0.6, 29)])
    return Likelihoods.from_gaussian(z, 0.12, BINS, warn_outside=False)


def test_every_result_carries_its_best_fit_and_best_outputs():
    lk = _lk()
    res = Gibbs(binference.PopulationModel(lk), n_warmup=50, rng=0).run(100)
    est = MAP(binference.PopulationModel(lk)).run()
    np.testing.assert_allclose(res.best.value, est.value)
    assert res.best is res.best and est.best is est
    np.testing.assert_allclose(
        res.assignment_probabilities_best(idx=[0, 1]), est.assignment_probabilities(idx=[0, 1])
    )
    np.testing.assert_allclose(res.predict_proba_best(lk[:2]), est.predict_proba(lk[:2]))


def _mc_se(r, chains, w=None):
    # Monte Carlo error per bin of a (weighted) mean of the draws r, with their
    # autocorrelation: the importance-sampling variance times n / ESS
    w = np.full(len(r), 1 / len(r)) if w is None else w
    p = w @ r
    tau = [
        len(r) / ess(r[:, k].reshape(chains, -1)) if r[:, k].std() > 0 else 1
        for k in range(r.shape[1])
    ]
    return np.sqrt((w[:, None] ** 2 * (r - p) ** 2).sum(0) * tau)


def test_leave_one_out_posteriors_match_a_refit_without_the_object():
    lk = _lk()
    L0 = lk.values[0]
    full = Gibbs(binference.PopulationModel(lk), n_chains=4, n_warmup=200, rng=0).run(2000)
    p_loo, k_hat = full.assignment_probabilities_loo(idx=[0], n_draws=None)
    rest = Gibbs(binference.PopulationModel(lk[1:]), n_chains=4, n_warmup=200, rng=1).run(2000)

    def resp(rho):  # object 0's posterior over bins under each draw
        return rho * L0 / (rho @ L0)[:, None]

    r_ref = resp(rest.samples)
    p_ref = r_ref.mean(0)
    # the same importance weights, to give leave-one-out its Monte Carlo error
    r_in = resp(full.samples)
    w = np.exp(psis(-np.log(full.samples @ L0))[0])
    se = np.hypot(_mc_se(r_ref, 4), _mc_se(r_in, 4, w))
    assert k_hat[0] < 0.7
    # the in-sample answer differs from the refit; leave-one-out recovers it
    p_in = full.assignment_probabilities(idx=[0], n_draws=None)[0]
    assert abs(p_in[-1] - p_ref[-1]) > 8 * se[-1]
    assert np.all(np.abs(p_loo[0] - p_ref) <= 4 * se + 1e-12)


def test_leave_one_out_needs_draws_and_flags_unreliable_objects():
    lk = _lk()
    with pytest.raises(ValueError, match="needs posterior draws"):
        MAP(binference.PopulationModel(lk)).run().assignment_probabilities_loo()
    # a single object dominates a short fit: its weights are heavy-tailed
    res = Gibbs(binference.PopulationModel(lk[:3]), n_warmup=20, rng=0).run(40)
    with pytest.warns(binference.ConvergenceWarning, match="k-hat"):
        res.assignment_probabilities_loo()


def test_the_best_fit_of_a_loaded_result_needs_the_fits_model(tmp_path):
    lk = _lk()
    model = binference.PopulationModel(lk, prior=binference.priors.Dirichlet(3.0))
    res = Gibbs(model, n_warmup=20, rng=0).run(40)
    res.save(tmp_path / "r.npz")
    loaded = binference.PopulationResult.load(tmp_path / "r.npz").attach(lk)
    with pytest.raises(ValueError, match="fit's prior"):
        _ = loaded.best  # bare likelihoods get the default prior, not Dirichlet(3)
    np.testing.assert_allclose(loaded.attach(model).best.value, res.best.value)


def test_leave_one_out_of_a_rate_model_includes_the_total():
    # the object's Poisson factor is Lambda (L_i . rho); with unequal exposures Lambda
    # and rho are dependent, so leaving Lambda out of the weights biases the answer
    rng = np.random.default_rng(0)
    z = np.concatenate([[0.92], rng.uniform(0.0, 0.6, 8)])
    lk = Likelihoods.from_gaussian(z, 0.15, BINS, warn_outside=False)
    exposure = Selection(np.array([0.05, 0.05, 1.0, 1.0, 0.05]), "on_truth", scale="exposure")

    def fit(lk, seed):
        prior = binference.priors.GammaRates(1.0, 0.01)
        model = binference.PopulationModel(lk, prior=prior, selection=exposure)
        return Gibbs(model, n_chains=4, n_warmup=200, rng=seed).run(5000)

    full, rest = fit(lk, 0), fit(lk[1:], 1)
    p_loo, _ = full.assignment_probabilities_loo(idx=[0], n_draws=None)
    L0 = full.model._L[0]  # with the exposures applied

    def resp(lam):
        return lam * L0 / (lam @ L0)[:, None]

    r_ref, r_in = resp(rest.rates), resp(full.rates)
    w = np.exp(psis(-np.log(full.rates @ L0))[0])
    se = np.hypot(_mc_se(r_ref, 4), _mc_se(r_in, 4, w))
    assert np.all(np.abs(p_loo[0] - r_ref.mean(0)) <= 4 * se + 1e-12)
    # the fractions weights, without the total, miss by more than that
    rho = full.rates / full.rates.sum(1, keepdims=True)
    w_frac = np.exp(psis(-np.log(rho @ L0))[0])
    assert abs(w_frac @ r_in[:, -1] - r_ref[:, -1].mean()) > 4 * se[-1]


def test_leave_one_out_refuses_bootstrap_draws_and_too_few_draws():
    lk = _lk()
    with pytest.warns(binference.ApproximationWarning):
        boot = binference.infer(lk, method="bootstrap", n_samples=20, rng=0)
    with pytest.raises(ValueError, match="bootstrap"):
        boot.assignment_probabilities_loo()
    res = Gibbs(binference.PopulationModel(lk), n_warmup=20, rng=0).run(40)
    with pytest.raises(ValueError, match="n_draws must be at least 5"):
        res.assignment_probabilities_loo(n_draws=3)


def test_an_object_impossible_under_some_draw_is_flagged():
    lk = Likelihoods(np.array([[0, 0, 0, 1, 1], [1, 1, 1, 1, 1.0]]), BINS)
    draws = np.random.default_rng(0).dirichlet(np.ones(5), 50)
    draws[:10, -2:] = 0.0  # object 0 lives in the last bins
    draws /= draws.sum(1, keepdims=True)
    res = binference.PopulationResult(binference.PopulationModel(lk), draws, method="gibbs")
    with pytest.warns(binference.ConvergenceWarning, match="k-hat"):
        p, k_hat = res.assignment_probabilities_loo(idx=[0, 1])
    assert k_hat[0] == np.inf and np.all(p[0] == 0)
    assert k_hat[1] == -np.inf  # a flat likelihood: equal weights
