"""Tests for diagnostics (R-DIAG-1)."""

import numpy as np
import pytest

from binference import diagnostics as d


def ar1(phi, n, chains, rng):
    x = np.empty((chains, n))
    x[:, 0] = rng.normal(size=chains) / np.sqrt(1 - phi**2)
    eps = rng.normal(size=(chains, n))
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + eps[:, t]
    return x


# Relative tolerances are at least 4x the replicate sd of each estimator (60 seeds,
# 4 x 20000 draws): tau 1.2%, 3.1%, 5.8% and ESS 1.3%, 2.3%, 4.8% at phi = 0, 0.5, 0.9.
# The MCSE is sd / sqrt(ESS), so its relative error is half the ESS's: 10% covers 4x.
@pytest.mark.parametrize(
    ("phi", "tol_tau", "tol_ess"), [(0.0, 0.05, 0.06), (0.5, 0.13, 0.1), (0.9, 0.25, 0.2)]
)
def test_autocorr_time_and_ess_ar1(phi, tol_tau, tol_ess, rng):
    x = ar1(phi, 20000, 4, rng)
    tau_true = (1 + phi) / (1 - phi)
    assert d.autocorr_time(x) == pytest.approx(tau_true, rel=tol_tau)
    n = x.size
    assert d.ess(x, "mean") == pytest.approx(n / tau_true, rel=tol_ess)
    assert d.ess(x, "bulk") == pytest.approx(n / tau_true, rel=tol_ess)
    assert np.isfinite(d.ess(x, "tail"))
    sd = x.std()
    assert d.mcse(x) == pytest.approx(sd / np.sqrt(n / tau_true), rel=0.1)


def test_ess_rhat_and_mcse_match_arviz_on_a_fixed_array():
    """Golden values for one fixed array (AR(1), phi = 0.6, 4 x 400, one chain offset).

    Reference numbers from ArviZ 0.23.4 (az.ess with method "bulk", "tail" and "mean",
    az.rhat with method "rank", az.mcse with method "mean") on this same array.
    """
    e = np.random.default_rng(2021).standard_normal((4, 400))
    x = np.empty_like(e)
    x[:, 0] = e[:, 0]
    for t in range(1, 400):
        x[:, t] = 0.6 * x[:, t - 1] + e[:, t]
    x += np.array([0.0, 0.0, 0.0, 0.3])[:, None]
    assert d.ess(x, "bulk") == pytest.approx(335.3651712242715, rel=1e-10)
    assert d.ess(x, "tail") == pytest.approx(777.3811513039336, rel=1e-10)
    assert d.ess(x, "mean") == pytest.approx(336.3372057270772, rel=1e-10)
    assert d.rhat(x) == pytest.approx(1.0109863729663382, rel=1e-10)
    assert d.mcse(x) == pytest.approx(0.06708114691315269, rel=1e-10)


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


@pytest.mark.parametrize("value", [0.25, 0.3])
def test_a_constant_chain_has_no_autocorrelation_time(value):
    # 0/0 became a zero autocorrelation: a chain whose mean is exact (0.25) looked
    # independent, time 1, and one whose mean is off by round-off (0.3) gave time n
    x = np.full((2, 500), value)
    assert np.isnan(d.autocorr_time(x)) and np.isnan(d.autocorr_function(x)).all()


def test_psis_refuses_an_infinite_log_weight(rng):
    # +inf was mapped to -inf with NaN: an infinite ratio got weight zero and a k-hat that
    # looked fine
    with pytest.raises(ValueError, match=r"log_weights must not be \+inf"):
        d.psis(np.r_[rng.normal(size=99), np.inf])
    lw, _ = d.psis(np.r_[rng.normal(size=99), np.nan])  # NaN still weighs zero
    assert lw[-1] == -np.inf and np.isfinite(lw[:-1]).all()


def test_psis_of_equal_weights_has_no_tail():
    lw, k = d.psis(np.full(100, 3.0) + 1e-14 * np.arange(100))  # equal up to round-off
    np.testing.assert_allclose(np.exp(lw), 0.01)
    assert k == -np.inf


def test_importance_ess_is_kish_and_ignores_the_normalisation():
    w = np.array([1.0, 1.0, 2.0, 4.0])
    assert d.importance_ess(w) == pytest.approx(64.0 / 22.0)
    assert d.importance_ess(w / w.sum()) == pytest.approx(64.0 / 22.0)
    assert d.importance_ess(np.ones(7)) == pytest.approx(7.0)


def test_psis_of_rows_is_psis_of_each_row(rng):
    # rows with tails of different lengths, fitted in blocks of a few rows
    for n in (5, 7, 300):
        rows = [
            rng.standard_t(3, n),
            np.log(rng.pareto(1.0, n) + 1),
            np.full(n, 2.0) + 1e-14 * np.arange(n),  # equal weights: k = -inf
            np.r_[rng.normal(size=n - 2), -np.inf, np.nan],
            np.round(rng.normal(size=n), 1),  # ties
            np.repeat(rng.normal(size=n // 5 + 1), 5)[:n],
        ]
        lw, k = d._psis_rows(np.array(rows), block=300)
        for row, lw_i, k_i in zip(rows, lw, k, strict=True):
            lw_ref, k_ref = d.psis(row)
            np.testing.assert_allclose(lw_i, lw_ref, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(k_i, k_ref, rtol=1e-12)


def test_psis_fits_only_the_weights_above_the_cut():
    # weights tied with the cut have no exceedance; with log(exp(cut)) < cut they entered
    # the tail and the fit divided by zero
    cut = next(c for c in -np.linspace(0.5, 0.6, 1000) if np.log(np.exp(c)) < c)
    above = np.linspace(cut + 0.01, 0.0, 95)
    lw = np.r_[above, np.full(60, cut), np.linspace(-3.0, cut - 0.01, 845)]
    k_ref, _ = d._gpdfit(np.exp(above) - np.exp(cut))
    assert d.psis(lw)[1] == pytest.approx(k_ref)


def test_a_degenerate_tail_fit_gives_an_infinite_k_hat(monkeypatch):
    # a NaN k-hat compares False with any threshold, so it would never be flagged
    def nan_fit(x):
        return np.full(x.shape[0], np.nan), np.full(x.shape[0], np.nan)

    monkeypatch.setattr(d, "_gpdfit", nan_fit)
    lw, k = d.psis(np.random.default_rng(0).normal(size=200))
    assert k == np.inf and np.isfinite(lw).all()
