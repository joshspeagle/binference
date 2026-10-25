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
    s = Likelihoods.from_gaussian(MU, [[0.1, 0.08]], GRID)
    d = Likelihoods.from_gaussian(MU, bins=GRID, cov=np.diag([0.01, 0.0064]))
    np.testing.assert_allclose(d.toarray(), s.toarray(), rtol=1e-14)  # the exact path
    tiny = np.array([[0.01, 1e-12], [1e-12, 0.0064]])  # correlated in name only: cubature
    c = Likelihoods.from_gaussian(MU, bins=GRID, cov=tiny, tol=1e-8)
    np.testing.assert_allclose(c.toarray(), s.toarray(), atol=1e-7)


def test_a_one_dimensional_covariance_is_a_variance():
    bins = Bins.linear(0, 1, 5)
    a = Likelihoods.from_gaussian([0.3], bins=bins, cov=np.array([[[0.04]]]))
    np.testing.assert_allclose(a.toarray(), Likelihoods.from_gaussian([0.3], 0.2, bins).toarray())


def test_covariances_are_validated():
    with pytest.raises(ValueError, match="exactly one of sigma and cov"):
        Likelihoods.from_gaussian(MU, 0.1, GRID, cov=COV)
    with pytest.raises(ValueError, match="positive definite"):
        Likelihoods.from_gaussian(MU, bins=GRID, cov=np.array([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(ValueError, match="symmetric"):  # small entries, relative check
        Likelihoods.from_gaussian(MU, bins=GRID, cov=np.array([[1e-8, 3e-9], [-3e-9, 1e-8]]))
    with pytest.raises(ValueError, match="strictly positive variances"):
        Likelihoods.from_gaussian(MU, bins=GRID, cov=np.diag([-0.01, 0.01]))
    with pytest.raises(ValueError, match="cov must have shape"):
        Likelihoods.from_gaussian(MU, bins=GRID, cov=np.eye(3))


def test_correlated_likelihoods_support_values_and_outside_fractions():
    lk = Likelihoods.from_gaussian(MU, bins=GRID, cov=COV)
    assert np.all(lk.meta["outside_fraction"] < 0.05)
    res = binference.infer(lk, method="map")
    with pytest.warns(binference.InputWarning, match="flat within each bin"):
        vals = res.sample_values(n_draws=5, rng=0)
    assert vals.shape[1:] == (2, 2) and np.all((vals >= 0) & (vals <= 1))


def test_a_thin_ridge_is_resolved_rather_than_missed():
    mu, sd, rho = np.array([0.43, 0.47]), 0.1, 0.9999
    cov = sd**2 * np.array([[1.0, rho], [rho, 1.0]])
    lk = Likelihoods.from_gaussian(mu[None], bins=GRID, cov=cov)
    mv = stats.multivariate_normal(mu, cov)
    e = GRID.edges
    ref = np.array(
        [
            mv.cdf([e[0][i + 1], e[1][j + 1]])
            - mv.cdf([e[0][i], e[1][j + 1]])
            - mv.cdf([e[0][i + 1], e[1][j]])
            + mv.cdf([e[0][i], e[1][j]])
            for i, j in zip(*GRID.cell_index, strict=True)
        ]
    )
    np.testing.assert_allclose(lk.toarray()[0], ref / ref.max(), atol=1e-5)
    assert lk.meta["outside_fraction"][0] < 1e-4


def test_a_three_dimensional_correlated_gaussian_matches_its_rectangle_masses():
    grid = Bins.grid(*[np.linspace(0, 1, 4)] * 3)
    mu = np.array([0.48, 0.52, 0.5])
    cov = 0.08**2 * np.array([[1.0, 0.9, 0.8], [0.9, 1.0, 0.85], [0.8, 0.85, 1.0]])
    lk = Likelihoods.from_gaussian(mu[None], bins=grid, cov=cov, tol=1e-6)
    mv = stats.multivariate_normal(mu, cov)
    e = grid.edges
    corners = np.array(np.meshgrid(*[[0, 1]] * 3, indexing="ij")).reshape(3, -1).T
    sign = (-1.0) ** (3 - corners.sum(axis=1))
    ref = np.array(
        [
            sign @ mv.cdf([[e[d][idx[d] + c[d]] for d in range(3)] for c in corners])
            for idx in np.array(grid.cell_index).T
        ]
    )
    np.testing.assert_allclose(lk.toarray()[0], ref / ref.max(), atol=2e-4)  # QMC error
