"""Tests for the MAP fixes of audit M-5, M-6, M-8, M-9, M-13 and T-4."""

import warnings

import numpy as np
import pytest

import binference
from binference import Bins, ConvergenceWarning, PopulationModel, Selection
from binference import priors as P
from binference.samplers import MAP

GAP = 1e3 * 1e-8  # MAP's guarantee: tol = 1e-8, or 1e3 * tol once round-off stops it


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
    assert sqp.kkt_gap <= GAP
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
    m = PopulationModel(lk._copy(weights=7.0 * lk.weights), prior)  # log-scales matter
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


def test_map_checks_its_init_and_em_moves_a_zero_in_it():
    m = PopulationModel(_mock(n=200, K=4).likelihoods)
    for init in (np.full(7, 0.1), [0.5, 0.6, -0.2, 0.1], np.zeros(4)):
        with pytest.raises(ValueError, match="init"):
            MAP(m, init=init)
    em = MAP(m, algorithm="em", init=[0.5, 0.5, 0.0, 0.0]).run()  # the mode's third is 0.26
    assert em.converged
    np.testing.assert_allclose(em.value, MAP(m).run().value, atol=1e-8)


def test_a_failed_quadratic_subproblem_hands_mixsqp_over_to_em(monkeypatch):
    from binference.samplers import optimize

    def fail(*args, **kwargs):
        raise RuntimeError("Maximum number of iterations reached.")

    monkeypatch.setattr(optimize.optimize, "nnls", fail)
    m = PopulationModel(_mock().likelihoods)
    with pytest.warns(ConvergenceWarning, match="continuing with EM"):
        est = MAP(m).run()
    assert est.info["algorithm"] == "squarem" and est.converged and est.kkt_gap < 1e-5


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_zero_rate_under_a_gamma_prior_is_no_kkt_violation(seed):
    # the prior gradient at a zero rate is -b, not 0
    sel = Selection(np.linspace(2.0, 0.5, 12), "on_truth", scale="exposure")
    m = PopulationModel(_mock(seed=seed).likelihoods, P.GammaRates(1.0, 0.5), selection=sel)
    est = MAP(m).run()
    assert est.converged and est.kkt_gap <= GAP


def test_the_rates_gap_is_one_sided_at_a_rate_near_zero():
    m = PopulationModel(_mock(K=20, noise=0.05).likelihoods, P.GammaRates(1.0, 1e-12))
    lam = MAP(m).run().value
    assert np.any(lam == 0.0)
    near = np.where(lam == 0.0, 1e-12, lam)  # where EM leaves a rate it drives to zero
    assert m.kkt_gap(near) < 1e-6


@pytest.mark.parametrize("alpha", [0.5, 0.9])
def test_priors_unbounded_at_the_boundary_converge_on_the_simplex(alpha):
    m = PopulationModel(_mock(K=20, noise=0.2).likelihoods, P.Dirichlet(alpha))
    with pytest.warns(binference.InputWarning, match="unbounded"):
        est = MAP(m, space="simplex").run()  # a ConvergenceWarning is an error here
    assert est.converged and np.any(est.value == 0.0)


def test_the_prior_density_is_reported_as_it_is_on_the_boundary():
    lk = _mock(K=4).likelihoods
    at_zero = np.array([0.0, 0.2, 0.3, 0.5])
    assert PopulationModel(lk, P.Dirichlet(2.0)).log_prior(at_zero) == -np.inf
    assert PopulationModel(lk, P.Dirichlet(0.5)).log_prior(at_zero) == np.inf
    assert np.isnan(PopulationModel(lk, P.LogisticGP()).log_prior(at_zero))
    tree = PopulationModel(lk, P.PolyaTree(c=1.0))
    assert np.isnan(tree.log_prior(np.array([0.0, 0.0, 0.5, 0.5])))  # inf - inf, quietly


def test_em_started_at_its_fixed_point_converges_at_once():
    m = PopulationModel(_mock().likelihoods, P.Dirichlet(2.0))
    mode = MAP(m, algorithm="em").run()
    again = MAP(m, algorithm="em", init=mode.value, max_iter=5).run()
    assert again.converged and again.n_iter < 5


def test_an_explicit_simplex_algorithm_never_silently_runs_lbfgs():
    lk = _mock(n=300, K=8).likelihoods
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


def test_the_ess_message_never_rounds_the_per_chain_value_up_to_the_limit():
    with warnings.catch_warnings():  # a short run; only the message's format is tested
        warnings.simplefilter("ignore", ConvergenceWarning)
        res = binference.infer(_mock(n=100, K=3).likelihoods, method="gibbs", n_samples=20, rng=0)
    for ess, shown in ((399.0, "399 (99.7"), (399.9, "399 (99.9")):  # 99.75, 99.975 per chain
        res._cache["diagnostics"] = dict(res.diagnostics, ess_tail=np.full(3, ess))
        msgs = [m for m in res._mixing_problems() if "tail ESS" in m]
        assert msgs and f"= {shown} per chain < 100)" in msgs[0]
