"""Tests for the NUTS and Laplace fixes of audit N-1 to N-15 (DECISIONS #74)."""

import warnings

import numpy as np
import pytest

import binference
from binference import Bins, PopulationModel
from binference import priors as P
from binference.samplers import NUTS, Laplace
from binference.samplers._nuts_core import Metric, _leaf


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
    assert not np.all(eng.model.prior.lam == 1.0)  # sampled partly centred
    np.testing.assert_allclose(m.to_value(x[0]), eng.result.chains[0, -1])


def test_nuts_arguments_are_checked():
    m = _gp_model(n=200, K=6)
    for kwargs in ({"max_depth": 0}, {"n_warmup": -1}, {"jitter": -0.1}):
        with pytest.raises(ValueError):
            NUTS(m, **kwargs)
    with pytest.raises(ValueError, match="init"):
        NUTS(m, init=np.full(m.dim, np.nan)).run(1)


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


def test_the_psis_threshold_needs_five_weights():
    with pytest.raises(ValueError, match="at least 5"):
        binference.diagnostics.psis_threshold(1)
    assert binference.diagnostics.psis_threshold(10_000) == 0.7


def test_a_step_that_overflows_the_energy_is_a_silent_divergence():
    # a momentum of 1e200 overflows p @ M^-1 p; the tree marks the leaf divergent, and the
    # overflow warning is not raised (it failed the joint-selection SBC as an error)

    def fn(x):
        return -0.5 * float(x @ x), -x

    x, p = np.zeros(2), np.full(2, 1e200)
    t = _leaf(x, p, -x, 1, 1e-300, 0.0, Metric(np.ones(2)), fn, 1000.0)
    assert t.diverging
