"""Slow calibration tests of the paths beyond the fixed-prior Dirichlet (audit T-2).

* Simulation-based calibration (SBC) of smoothness priors with inferred
  hyperparameters, Gibbs with a known selection in both modes, rates with an
  exposure, the Pólya tree and the Laplace approximation. Each case uses
  :func:`binference.simulate.prior_predictive` and :func:`binference.validation.sbc`
  and asserts that every fit's smallest bulk ESS is at least ``n_draws``, so the
  thinned draws are effectively independent and autocorrelation cannot pass for
  miscalibration.
* Geweke "getting it right" tests of the Gibbs conditionals of the generalised
  Dirichlet, a Dirichlet tree and Gamma rates with an exposure.
"""

import numpy as np
import pytest

import binference
from binference import Bins
from binference import priors as P

from .test_slow import _fit, _geweke_z_scores

pytestmark = pytest.mark.slow

BINS4 = Bins.linear(0.0, 1.0, 4)
GIBBS = {"n_chains": 4, "n_warmup": 100, "n_samples": 400}
GIBBS_SEL = {**GIBBS, "n_samples": 600}  # selection slows mixing (smallest ESS 22 at 400)
DETECT = (1.0, 0.6, 0.35, 0.2)  # detection probabilities per bin


def _sbc(prior, method, kw, *, n_simulations, n_draws, rng, n_objects=40, noise=0.2, **sim):
    simulate = binference.simulate.prior_predictive(prior, BINS4, n_objects, noise=noise, **sim)
    res = binference.validation.sbc(
        simulate,
        _fit(method, prior=prior, **kw),
        n_simulations=n_simulations,
        n_draws=n_draws,
        rng=rng,
    )
    assert np.all(res.min_ess >= n_draws), np.sort(res.min_ess)[:5]
    return res


# (prior, method, engine settings, simulator settings, n_simulations, n_draws)
CASES = {
    "nuts-randomwalk2-inferred-scale": (
        P.LogisticRandomWalk(order=2),
        "nuts",
        # a short warmup leaves some near-flat populations at a step size of 0.002 and
        # an ESS of 6; 500 warmup iterations keep every fit's ESS above 19
        {"n_chains": 2, "n_warmup": 500, "n_samples": 150},
        {},
        50,
        19,
    ),
    "gibbs-selection-on-truth": (
        P.Dirichlet(1.0),
        "gibbs",
        GIBBS_SEL,
        {"noise": 0.1, "selection": DETECT, "selection_mode": "on_truth"},
        200,
        19,
    ),
    "gibbs-selection-on-measurement": (
        P.Dirichlet(1.0),
        "gibbs",
        GIBBS_SEL,
        {"noise": 0.1, "selection": DETECT, "selection_mode": "on_measurement"},
        200,
        19,
    ),
    "gibbs-rates-exposure": (
        P.GammaRates(2.0, 0.1),
        "gibbs",
        GIBBS,
        {"n_objects": None, "selection": (1.5, 1.0, 0.6, 0.3), "selection_mode": "on_truth"},
        200,
        19,
    ),
    "gibbs-polyatree": (P.PolyaTree(), "gibbs", GIBBS, {}, 200, 49),
    # importance-corrected (the default); calibrated at K = 4 with 40 objects
    "laplace-dirichlet": (P.Dirichlet(1.0), "laplace", {"n_samples": 400}, {}, 200, 49),
}


@pytest.mark.parametrize("case", list(CASES))
def test_sbc_ranks_are_uniform(case):
    prior, method, kw, sim, n_simulations, n_draws = CASES[case]
    res = _sbc(prior, method, kw, n_simulations=n_simulations, n_draws=n_draws, rng=2026, **sim)
    n_bins = 10 if n_simulations >= 150 else 5  # at least 10 expected ranks per bin
    p = res.pvalues(n_bins=n_bins)
    assert res.passed(n_bins=n_bins), dict(zip(res.names, p, strict=True))


@pytest.mark.parametrize(
    ("prior", "selection"),
    [
        (P.GeneralizedDirichlet([1.5, 0.8], [2.0, 1.2]), (0.9, 0.5, 0.25)),
        (P.DirichletTree([0, [1, 2]], alpha=[[1.5, 2.0], [0.7, 1.3]]), (0.9, 0.5, 0.25)),
        (P.GammaRates([2.0, 3.0, 1.5], 0.5), (1.5, 1.0, 0.5)),
    ],
    ids=["generalized-dirichlet", "dirichlet-tree", "gamma-rates-exposure"],
)
def test_geweke_getting_it_right_beyond_the_dirichlet(prior, selection):
    z_scores = _geweke_z_scores(selection, prior=prior)
    assert np.all(np.abs(z_scores) < 4.5), z_scores
