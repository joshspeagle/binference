"""Tests for internal simplex helpers."""

import numpy as np
import pytest

from pastry._simplex import clr, helmert_basis, ilr, ilr_inv, log_softmax, softmax


@pytest.mark.parametrize("K", [2, 3, 7, 20])
def test_helmert_orthonormal_zero_sum(K):
    V = helmert_basis(K)
    assert V.shape == (K, K - 1)
    np.testing.assert_allclose(V.T @ V, np.eye(K - 1), atol=1e-12)
    np.testing.assert_allclose(V.sum(axis=0), 0.0, atol=1e-12)


def test_helmert_needs_two_bins():
    with pytest.raises(ValueError):
        helmert_basis(1)


def test_ilr_round_trip(rng):
    for K in (2, 5, 13):
        rho = rng.dirichlet(np.ones(K))
        np.testing.assert_allclose(ilr_inv(ilr(rho)), rho, rtol=1e-12)
        np.testing.assert_allclose(clr(rho).sum(), 0.0, atol=1e-12)


def test_softmax_stable():
    x = np.array([1000.0, 999.0, -1000.0])
    s = softmax(x)
    assert np.all(np.isfinite(s))
    np.testing.assert_allclose(s.sum(), 1.0)
    np.testing.assert_allclose(log_softmax(x)[:2], np.log(s[:2]), atol=1e-12, rtol=0)
    np.testing.assert_allclose(log_softmax(x)[2], -2000.0 - np.log1p(np.exp(-1.0)), rtol=1e-12)


@pytest.mark.parametrize("K", [2, 3, 4, 6])
def test_ilr_log_jacobian(K, rng):
    """log|det d rho[:K-1] / d eta| = 0.5 log K + sum(log rho) (used for prior densities)."""
    eta = rng.normal(size=K - 1)
    h = 1e-6
    J = np.empty((K - 1, K - 1))
    for j in range(K - 1):
        e = np.zeros(K - 1)
        e[j] = h
        J[:, j] = (ilr_inv(eta + e)[: K - 1] - ilr_inv(eta - e)[: K - 1]) / (2 * h)
    rho = ilr_inv(eta)
    sign, logdet = np.linalg.slogdet(J)
    assert sign != 0
    np.testing.assert_allclose(logdet, 0.5 * np.log(K) + np.log(rho).sum(), atol=1e-6)
