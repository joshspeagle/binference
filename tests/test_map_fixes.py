"""Tests for the MAP fixes of audit M-5, M-6, M-8, M-9, M-13 and T-4."""

import numpy as np
import pytest

import binference
from binference import Bins, ConvergenceWarning, PopulationModel, Selection
from binference import priors as P
from binference.samplers import MAP


def _mock(n=2000, K=12, noise=0.15, seed=1):
    return binference.simulate.mock_1d(n, Bins.linear(0, 3, K), noise=noise, rng=seed)


def test_mixsqp_takes_dirichlet_priors_mixing_flat_and_informative_bins():
    # a_k = 0 next to a_k > 0 divided 0 by 0 in the Hessian (warnings are errors here)
    m = PopulationModel(_mock().likelihoods, P.Dirichlet([1, 1, 3, 1, 2, 1, 1, 4, 1, 1, 1, 2]))
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
    m = PopulationModel(_mock().likelihoods, P.GammaRates(shape, rate), selection=sel)
    sqp = MAP(m).run()
    assert sqp.info["algorithm"] == "mixsqp" and sqp.converged and sqp.n_iter < 100
    assert sqp.kkt_gap < 1e-6
    # the total is profiled out exactly: Lambda = W + sum(shape - 1) over the exposure
    e = m._S + rate
    np.testing.assert_allclose(sqp.value @ e, m.W + m.K * (shape - 1.0), rtol=1e-10)
    em = MAP(m, algorithm="em", max_iter=100000).run()
    assert em.converged and em.kkt_gap <= 1e-5
    np.testing.assert_allclose(sqp.value, em.value, rtol=1e-3, atol=1e-3 * m.W / m.K)


def test_em_converges_only_where_the_kkt_gap_is_small():
    # SQUAREM used to stop on a stalled objective far from the mode (audit M-6)
    m = PopulationModel(_mock(K=40, noise=0.1).likelihoods, P.GammaRates(1.0, 1e-12))
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
    lk = _mock().likelihoods
    m = PopulationModel(lk._reweighted(7.0 * lk.weights), prior)  # log-scales matter
    est = MAP(m, **kwargs).run()
    np.testing.assert_allclose(est.log_posterior, m.log_posterior(est.value), rtol=1e-12)
    if est.x is not None:
        np.testing.assert_allclose(
            est.info["log_density_unconstrained"], m.log_density_unconstrained(est.x)
        )


@pytest.mark.parametrize("kwargs", [{"algorithm": "em"}, {"space": "unconstrained"}])
def test_a_looser_tolerance_stops_every_algorithm_sooner(kwargs):
    m = PopulationModel(_mock(K=20, noise=0.3).likelihoods, P.Dirichlet(2.0))
    tight = MAP(m, tol=1e-10, **kwargs).run()
    loose = MAP(m, tol=1e-4, **kwargs).run()
    assert loose.n_iter < tight.n_iter and tight.converged and loose.converged


def test_mixsqp_starts_from_its_init():
    m = PopulationModel(_mock().likelihoods)
    mode = MAP(m).run()
    again = MAP(m, init=mode.value).run()
    assert again.n_iter <= 2 < mode.n_iter
    np.testing.assert_allclose(again.value, mode.value, atol=1e-8)


def test_a_failed_quadratic_subproblem_ends_mixsqp_unconverged(monkeypatch):
    from binference.samplers import optimize

    def fail(*args, **kwargs):
        raise RuntimeError("Maximum number of iterations reached.")

    monkeypatch.setattr(optimize.optimize, "nnls", fail)
    m = PopulationModel(_mock().likelihoods)
    with pytest.warns(ConvergenceWarning, match="mix-SQP"):
        est = MAP(m).run()
    assert not est.converged and np.all(np.isfinite(est.value))
