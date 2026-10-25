"""Tests for diagnostics (R-DIAG-1)."""

import numpy as np
import pytest

from pastry import diagnostics as d


def ar1(phi, n, chains, rng):
    x = np.empty((chains, n))
    x[:, 0] = rng.normal(size=chains) / np.sqrt(1 - phi**2)
    eps = rng.normal(size=(chains, n))
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + eps[:, t]
    return x


@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9])
def test_autocorr_time_and_ess_ar1(phi, rng):
    x = ar1(phi, 20000, 4, rng)
    tau_true = (1 + phi) / (1 - phi)
    # estimator noise grows with tau (roughly 7% sd at phi = 0.9 for these lengths)
    assert d.autocorr_time(x) == pytest.approx(tau_true, rel=0.25)
    n = x.size
    assert d.ess(x, "mean") == pytest.approx(n / tau_true, rel=0.3)
    assert d.ess(x, "bulk") == pytest.approx(n / tau_true, rel=0.3)
    assert np.isfinite(d.ess(x, "tail"))
    sd = x.std()
    assert d.mcse(x) == pytest.approx(sd / np.sqrt(n / tau_true), rel=0.2)


def test_rhat(rng):
    iid = rng.normal(size=(4, 2000))
    assert d.rhat(iid) < 1.01
    shifted = iid + np.array([0.0, 0.0, 0.0, 1.0])[:, None]
    assert d.rhat(shifted) > 1.1
    # heavy tails are handled by rank normalisation
    cauchy = rng.standard_cauchy(size=(4, 2000))
    assert d.rhat(cauchy) < 1.02
    # different scales are caught by the folded version
    scaled = iid * np.array([1.0, 1.0, 1.0, 3.0])[:, None]
    assert d.rhat(scaled) > 1.05
    assert np.isnan(d.rhat(rng.normal(size=(1, 3))))


def test_ess_validation_and_degenerate():
    with pytest.raises(ValueError):
        d.ess(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        d.ess(np.ones(10), kind="median")
    assert np.isnan(d.ess(np.ones((2, 100))))


def test_psis(rng):
    good = rng.normal(scale=0.3, size=4000)
    lw, k = d.psis(good)
    assert k < 0.5
    np.testing.assert_allclose(np.exp(lw).sum(), 1.0)
    heavy = np.log(rng.pareto(1.0, size=4000) + 1)  # Pareto tail with index 1 -> k ~ 1
    _, k2 = d.psis(heavy)
    assert k2 > 0.7
    with pytest.raises(ValueError):
        d.psis([0.0, 1.0])


def test_autocorr_function():
    x = np.sin(np.linspace(0, 50, 1000))
    f = d.autocorr_function(x)
    assert f[0] == pytest.approx(1.0)
    assert f.shape == (1000,)
