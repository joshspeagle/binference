"""The binning-bias check and its planning number (DECISIONS #6, #20, #86)."""

import numpy as np
import pytest

import binference
from binference import Bins, Likelihoods
from binference.samplers import Gibbs

BINS = Bins.linear(0, 1, 8)
N = 20_000


def _check(z):
    rng = np.random.default_rng(0)
    mu = z + rng.normal(0, 0.1, z.size)

    def build(b):
        return Likelihoods.from_gaussian(mu, 0.1, b, warn_outside=False)

    res = Gibbs(binference.PopulationModel(build(BINS)), n_chains=2, n_warmup=100, rng=0).run(200)
    return res.binning_bias(build, rng=1)


@pytest.mark.slow
def test_a_steep_population_is_flagged_and_a_flat_one_is_not():
    rng = np.random.default_rng(1)
    steep = rng.exponential(0.15, 3 * N)
    with pytest.warns(binference.InputWarning, match="Binning bias"):
        bb = _check(steep[steep < 1][:N])
    assert np.max(np.abs(bb.z)) > 4
    np.testing.assert_allclose(bb.n_crit * bb.z**2, N)  # n_crit = W (sd / shift)^2
    flat = _check(rng.uniform(0, 1, N))
    assert np.max(np.abs(flat.z)) < 2


def test_the_check_needs_grid_bins_without_a_basis():
    lk = Likelihoods(np.ones((5, 3)), Bins.categories(3))
    res = Gibbs(binference.PopulationModel(lk), n_warmup=5, rng=0).run(10)
    with pytest.raises(ValueError, match="grid bins"):
        res.binning_bias(lambda b: lk)
