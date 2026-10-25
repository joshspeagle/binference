"""The best fit with every result, `_best` outputs and leave-one-out posteriors
(DECISIONS #35, #51, #52)."""

import numpy as np
import pytest

import binference
from binference import Bins, Likelihoods
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
    tau = [len(r) / ess(r[:, k].reshape(chains, -1)) if r[:, k].std() > 0 else 1 for k in range(5)]
    return np.sqrt((w[:, None] ** 2 * (r - p) ** 2).sum(0) * tau)


def test_leave_one_out_posteriors_match_a_refit_without_the_object():
    lk = _lk()
    L0 = lk.values[0]
    full = Gibbs(binference.PopulationModel(lk), n_chains=4, n_warmup=200, rng=0).run(2000)
    p_loo, k_hat = full.assignment_probabilities_loo(idx=[0], n_draws=None)
    rest = Gibbs(binference.PopulationModel(lk[1:]), n_chains=4, n_warmup=200, rng=1).run(2000)
    r_ref = rest.samples * L0 / (rest.samples @ L0)[:, None]  # object 0 under each draw
    p_ref = r_ref.mean(0)
    # the same importance weights, to give leave-one-out its Monte Carlo error
    r_in = full.samples * L0 / (full.samples @ L0)[:, None]
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
