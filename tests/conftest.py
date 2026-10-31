"""Shared fixtures and helpers for the binference test suite."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

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


def assert_matches_reference(result, mean, sd, z: float = 5.0):
    """Compare posterior means and sds with an exact reference within z Monte Carlo errors
    (``mcse`` and :func:`sd_mcse`), plus 1e-3 of the reference sd."""
    C = result.chains
    for k in range(C.shape[-1]):
        x = C[:, :, k]
        err = diag.mcse(x)
        if not np.isfinite(err):
            err = x.std() / np.sqrt(x.size)
        assert abs(x.mean() - mean[k]) <= z * err + 1e-3 * sd[k], (k, x.mean(), mean[k], err)
        err = sd_mcse(x)
        assert abs(x.std() - sd[k]) <= z * err + 1e-3 * sd[k], (k, x.std(), sd[k], err)


def assert_draws_agree(a, b, z: float = 5.0):
    """Compare the means and sds of two runs' draws ``a`` and ``b`` (chains, draws, K)
    within z combined Monte Carlo errors."""
    for k in range(a.shape[-1]):
        x, y = a[..., k], b[..., k]
        err = np.hypot(diag.mcse(x), diag.mcse(y))
        assert abs(x.mean() - y.mean()) <= z * err, (k, x.mean(), y.mean(), err)
        err = np.hypot(sd_mcse(x), sd_mcse(y))
        assert abs(x.std() - y.std()) <= z * err, (k, x.std(), y.std(), err)


def sbc_pvalues(res, n_bins=10):
    # the chi-square is the more sensitive to width errors, the ECDF test to bias
    return {"chisquare": res.pvalues(n_bins), "ecdf": res.pvalues(method="ecdf")}


def assert_uniform_ranks(res, alpha=1e-3, n_bins=10):
    # both tests, each at the Bonferroni-corrected level of SBCResult.passed (DECISIONS #87)
    for method, p in sbc_pvalues(res, n_bins).items():
        assert np.all(p > alpha / p.size), (method, dict(zip(res.names, p, strict=True)))
