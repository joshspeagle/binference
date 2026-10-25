"""The binning-bias check and its planning number (DECISIONS #6, #20, #86)."""

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
    check = res.binning_bias(lambda b: Likelihoods.from_gaussian(np.full(5, 0.5), 0.1, b))
    assert check.z.shape == (8,)
    cat = Gibbs(binference.PopulationModel(Likelihoods(np.ones((5, 3)))), n_warmup=5, rng=0)
    with pytest.raises(ValueError, match="grid bins"):
        cat.run(10).binning_bias(lambda b: lk)
