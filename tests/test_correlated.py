"""Tests for correlated Gaussian likelihoods (DECISIONS #2, #25, #66)."""

import numpy as np
import pytest
from scipy import stats

import binference
from binference import Bins, Likelihoods

GRID = Bins.grid(np.linspace(0, 1, 6), np.linspace(0, 1, 6))
MU = np.array([[0.42, 0.55], [0.3, 0.7]])
COV = np.array([[[0.01, 0.006], [0.006, 0.0064]], [[0.02, -0.01], [-0.01, 0.01]]])


def test_correlated_bins_match_rectangle_probabilities():
    lk = Likelihoods.from_gaussian(MU, bins=GRID, cov=COV, tol=1e-7)
    e = GRID.edges
    for n in range(2):
        mv = stats.multivariate_normal(MU[n], COV[n])
        ref = np.array(
            [
                mv.cdf([e[0][i + 1], e[1][j + 1]])
                - mv.cdf([e[0][i], e[1][j + 1]])
                - mv.cdf([e[0][i + 1], e[1][j]])
                + mv.cdf([e[0][i], e[1][j]])
                for i, j in zip(*GRID.cell_index, strict=True)
            ]
        )
        np.testing.assert_allclose(lk.toarray()[n], ref / ref.max(), atol=1e-6)


def test_a_diagonal_covariance_equals_independent_sigmas():
    d = Likelihoods.from_gaussian(MU, bins=GRID, cov=np.diag([0.01, 0.0064]), tol=1e-8)
    s = Likelihoods.from_gaussian(MU, [[0.1, 0.08]], GRID)
    np.testing.assert_allclose(d.toarray(), s.toarray(), atol=1e-8)


def test_a_one_dimensional_covariance_is_a_variance():
    bins = Bins.linear(0, 1, 5)
    a = Likelihoods.from_gaussian([0.3], bins=bins, cov=np.array([[[0.04]]]))
    np.testing.assert_allclose(a.toarray(), Likelihoods.from_gaussian([0.3], 0.2, bins).toarray())


def test_covariances_are_validated():
    with pytest.raises(ValueError, match="exactly one of sigma and cov"):
        Likelihoods.from_gaussian(MU, 0.1, GRID, cov=COV)
    with pytest.raises(ValueError, match="positive definite"):
        Likelihoods.from_gaussian(MU, bins=GRID, cov=np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(ValueError, match="cov must have shape"):
        Likelihoods.from_gaussian(MU, bins=GRID, cov=np.eye(3))


def test_correlated_likelihoods_support_values_and_outside_fractions():
    lk = Likelihoods.from_gaussian(MU, bins=GRID, cov=COV)
    assert np.all(lk.meta["outside_fraction"] < 0.05)
    res = binference.infer(lk, method="map")
    vals = res.sample_values(n_draws=5, rng=0)
    assert vals.shape[1:] == (2, 2) and np.all((vals >= 0) & (vals <= 1))
