"""Shared fixtures and helpers for the binference test suite."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import binference
from binference import diagnostics as diag

# Optional seed sweeping for robustness checks: BINFERENCE_TEST_SEEDS="0,1,2" pytest ...
_SEEDS = [int(s) for s in os.environ.get("BINFERENCE_TEST_SEEDS", "0").split(",")]


@pytest.fixture(scope="session")
def matrix():
    """The benchmark-matrix script, ``benchmarks/matrix.py``, as a module."""
    path = Path(__file__).parents[1] / "benchmarks" / "matrix.py"
    if not path.parent.exists():
        pytest.skip("benchmarks/ is not next to the tests (sdist)")
    spec = importlib.util.spec_from_file_location("matrix", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["matrix"] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=_SEEDS)
def seed(request) -> int:
    return request.param


@pytest.fixture
def rng(seed) -> np.random.Generator:
    return np.random.default_rng(seed)


@pytest.fixture(scope="session")
def k3_problem():
    """Three bins, heteroscedastic Gaussian likelihoods comparable to the bin width."""
    bins = binference.Bins([0.0, 1.0, 2.0, 3.0])
    mock = binference.simulate.mock_1d(
        250, bins, density=np.array([0.5, 0.15, 0.35]), noise=(0.3, 1.2), rng=12345
    )
    exact = binference.validation.exact_posterior(mock.likelihoods, n_grid=500)
    return mock, exact


@pytest.fixture(scope="session")
def k3_selection_problem():
    bins = binference.Bins([0.0, 1.0, 2.0, 3.0])
    S = np.array([0.9, 0.5, 0.25])
    mock = binference.simulate.mock_1d(
        250, bins, density=np.array([0.4, 0.35, 0.25]), noise=(0.3, 0.8), selection=S, rng=777
    )
    exact = binference.validation.exact_posterior(mock.likelihoods, n_grid=500)
    return mock, exact


@pytest.fixture(scope="session")
def small_1d():
    """A moderate 1-D problem used by API and results tests."""
    return binference.simulate.mock_1d(400, binference.Bins.linear(0.0, 3.0, 12), rng=2024)


def sd_mcse(x) -> float:
    """Monte Carlo error of the sd of draws ``x`` (chains, draws): the MCSE of the squared
    deviations, by the delta method."""
    return diag.mcse((x - x.mean()) ** 2) / (2 * x.std())


def mc_errors(draws):
    """Monte Carlo errors of the means and sds of ``draws`` (chains, draws, K), by ``mcse``
    and :func:`sd_mcse`."""
    cols = np.moveaxis(draws, -1, 0)
    return np.array([diag.mcse(x) for x in cols]), np.array([sd_mcse(x) for x in cols])


def assert_matches_reference(result, mean, sd, z: float = 5.0, ref_errors=(0.0, 0.0)):
    """Compare posterior means and sds with a reference within z combined Monte Carlo errors,
    plus 1e-3 of the reference sd. The result's errors are its chains', or for resampled
    draws the importance ESS's; ``ref_errors`` are the reference's own."""
    if "ess_importance" in result.info:
        n = result.info["ess_importance"]
        kurt = stats.kurtosis(result.samples, axis=0, fisher=False)
        errors = result.std / np.sqrt(n), result.std * np.sqrt((kurt - 1) / (4 * n))
    else:
        errors = mc_errors(result.chains)
    pairs = zip((result.mean, result.std), (mean, sd), errors, ref_errors, strict=True)
    for est, ref, err, ref_err in pairs:
        tol = z * np.hypot(err, ref_err) + 1e-3 * np.asarray(sd)
        assert np.all(np.abs(est - np.asarray(ref)) <= tol), (est, ref, tol)


def assert_draws_agree(a, b, z: float = 5.0):
    """Compare the means and sds of two runs' draws ``a`` and ``b`` (chains, draws, K)
    within z combined Monte Carlo errors."""
    for f, err_a, err_b in zip((np.mean, np.std), mc_errors(a), mc_errors(b), strict=True):
        x, y = f(a, axis=(0, 1)), f(b, axis=(0, 1))
        assert np.all(np.abs(x - y) <= z * np.hypot(err_a, err_b)), (f.__name__, x, y)


def sbc_pvalues(res, n_bins=10):
    # the chi-square is the more sensitive to width errors, the ECDF test to bias
    return {"chisquare": res.pvalues(n_bins), "ecdf": res.pvalues(method="ecdf")}


def assert_uniform_ranks(res, alpha=1e-3, n_bins=10):
    # both tests, each at the Bonferroni-corrected level of SBCResult.passed (DECISIONS #87)
    for method, p in sbc_pvalues(res, n_bins).items():
        assert np.all(p > alpha / p.size), (method, dict(zip(res.names, p, strict=True)))
