"""The binning-bias check and its planning number (DECISIONS #6, #20, #86)."""

import warnings

import numpy as np
import pytest

import binference
from binference import Bins, Likelihoods
from binference.samplers import Gibbs

BINS = Bins.linear(0, 1, 8)
N = 5_000


def _fit(z):
    mu = z + np.random.default_rng(0).normal(0, 0.1, z.size)
    lk = Likelihoods.from_gaussian(mu, 0.1, BINS, warn_outside=False)
    return Gibbs(binference.PopulationModel(lk), n_chains=2, n_warmup=100, rng=0).run(200)


def test_a_steep_population_is_flagged_and_a_flat_one_is_not():
    rng = np.random.default_rng(1)
    steep = rng.exponential(0.1, 3 * N)
    res = _fit(steep[steep < 1][:N])
    with pytest.warns(binference.InputWarning, match="Binning bias"):
        bb = res.binning_bias(rng=1)  # Gaussian likelihoods rebuild themselves
    big = np.abs(bb.z) > 2
    assert np.max(np.abs(bb.z)) > 4
    np.testing.assert_allclose(bb.n_crit[big] * bb.z[big] ** 2, N)  # n_crit = W / z^2
    assert np.all(np.isinf(bb.n_crit[~big]))
    flat = _fit(rng.uniform(0, 1, N)).binning_bias(rng=1)
    assert np.max(np.abs(flat.z)) < 2


def test_inputs_without_their_measurements_need_build():
    lk = Likelihoods(np.ones((5, 8)), BINS)
    res = Gibbs(binference.PopulationModel(lk), n_warmup=5, rng=0).run(10)
    with pytest.raises(ValueError, match="pass build"):
        res.binning_bias()
    with warnings.catch_warnings():  # other data than the fit's: the shift may be flagged
        warnings.simplefilter("ignore", binference.InputWarning)
        check = res.binning_bias(lambda b: Likelihoods.from_gaussian(np.full(5, 0.5), 0.1, b))
    assert check.z.shape == (8,)
    cat = Gibbs(binference.PopulationModel(Likelihoods(np.ones((5, 3)))), n_warmup=5, rng=0)
    with pytest.raises(ValueError, match="grid bins"):
        cat.run(10).binning_bias(lambda b: lk)


def test_the_check_refuses_what_it_cannot_compare():
    lk = Likelihoods.from_gaussian(np.linspace(0.05, 0.95, 40), 0.1, BINS, warn_outside=False)
    res = Gibbs(binference.PopulationModel(lk), n_warmup=10, rng=0).run(20)
    with pytest.raises(ValueError, match="on the basis it is given"):
        res.binning_bias(lambda b: lk)  # the closure ignores the basis
    with pytest.raises(ValueError, match="nothing to compare"):
        res.binning_bias(order=0)
    with pytest.raises(ValueError, match="engine that draws"):
        res.binning_bias(method="map")
    assert res.binning_bias(method="Gibbs", rng=0).z.shape == (8,)  # any case
    inj = binference.Selection.from_injections(
        np.linspace(0.01, 0.99, 80), np.ones(80, bool), BINS, "on_truth"
    )
    joint = Gibbs(binference.PopulationModel(lk, selection=inj), n_warmup=5, rng=0).run(10)
    with pytest.raises(ValueError, match="injections"):
        joint.binning_bias()


def test_bins_without_spread_are_not_compared_and_inf_prints_as_inf():
    lk = Likelihoods.from_gaussian(np.linspace(0.05, 0.6, 40), 0.05, BINS, warn_outside=False)
    draws = np.random.default_rng(0).dirichlet(np.ones(8), 100)
    draws[:, -1] = 0.0  # no spread in the last bin
    draws /= draws.sum(1, keepdims=True)
    res = binference.PopulationResult(binference.PopulationModel(lk), draws, method="gibbs")
    with warnings.catch_warnings():  # these draws are not this data's posterior: may warn
        warnings.simplefilter("ignore", binference.InputWarning)
        check = res.binning_bias(rng=0, n_warmup=20)
    assert np.isnan(check.z[-1]) and np.isinf(check.n_crit[-1])
    assert "inf" in repr(check)
