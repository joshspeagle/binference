"""C3c experiment (DECISIONS #25): bin averages of correlated 3-D Gaussians.

Compares binference's adaptive cubature (Gauss-Legendre 5 vs 3, split while the
difference exceeds tol, in log space) with SciPy's multivariate normal CDF (Genz
quasi-Monte Carlo) used through rectangle inclusion-exclusion. Reference: the
cubature at tol=1e-10. Reports the largest error relative to each object's
largest bin, the error on the smallest bins, and the time. The covariances are
A A^T + 0.002 I with A ~ N(0, 0.08^2): moderately correlated, with no thin ridges
(the ridge rule added in code review is tested in tests/test_correlated.py).
"""

import time

import numpy as np
from scipy import stats

from binference import Bins, Likelihoods

rng = np.random.default_rng(0)
grid = Bins.grid(*[np.linspace(0, 1, 7)] * 3)  # 6^3 = 216 bins
N = 20
mu = rng.uniform(0.2, 0.8, (N, 3))
A = rng.normal(size=(N, 3, 3)) * 0.08
cov = A @ np.swapaxes(A, 1, 2) + 0.002 * np.eye(3)

ref = Likelihoods.from_gaussian(mu, bins=grid, cov=cov, tol=1e-10).toarray()


def qmc_rows():
    e, out = grid.edges, np.empty((N, grid.K))
    corners = np.array(np.meshgrid(*[[0, 1]] * 3, indexing="ij")).reshape(3, -1).T
    sign = (-1.0) ** (3 - corners.sum(axis=1))
    for n in range(N):
        mv = stats.multivariate_normal(mu[n], cov[n])
        for k, idx in enumerate(np.array(grid.cell_index).T):
            pts = np.array([[e[d][idx[d] + c[d]] for d in range(3)] for c in corners])
            out[n, k] = max(sign @ mv.cdf(pts), 0.0)
    return out / out.max(axis=1, keepdims=True)


for name, fn in [
    ("cubature tol=1e-3", lambda: Likelihoods.from_gaussian(mu, bins=grid, cov=cov).toarray()),
    ("cubature tol=1e-5", lambda: Likelihoods.from_gaussian(mu, bins=grid, cov=cov, tol=1e-5).toarray()),
    ("Genz QMC (scipy)", qmc_rows),
]:
    t = time.perf_counter()
    got = fn()
    dt = time.perf_counter() - t
    err = np.abs(got - ref)
    small = ref < 1e-3
    print(f"{name:20s} max err {err.max():.1e}  max err on bins <1e-3 of max {err[small].max():.1e}  time {dt:.2f} s")
