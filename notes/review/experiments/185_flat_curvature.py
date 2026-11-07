"""Issue #185: NUTS time and adapted metric under variants of the curvature floor.

LogisticGP() on 4 bins, 10 objects, one chain, 300 warm-up plus 200 draws, 20 simulations.
Run with one BLAS thread; argument: number of worker processes.
"""

import multiprocessing as mp
import sys
import time
import warnings

import numpy as np

import binference
from binference import Bins, priors as P
from binference.engines import _curvature, nuts
from binference.simulate import prior_predictive


def clipped(floor):
    """The rule before #185: eigenvalues clipped below at ``floor`` times the largest."""

    def f(H):
        eig = _curvature.neg_hessian_eigen(H)
        if eig is None or not eig[0].max() > 0:
            return None
        lam, U = eig
        lam = np.maximum(lam, floor * lam.max())
        cov = (U / lam) @ U.T
        return 0.5 * (cov + cov.T)

    return f


def largest_seen(floor):
    """The rule of #185 at another threshold."""
    return lambda H: _curvature.covariance_from_hessian(H, floor)


def capped(max_var):
    def f(H):
        eig = _curvature.neg_hessian_eigen(H)
        if eig is None or not eig[0].max() > 0:
            return None
        lam, U = eig
        lam = np.maximum(lam, np.maximum(1e-6 * lam.max(), 1.0 / max_var))
        cov = (U / lam) @ U.T
        return 0.5 * (cov + cov.T)

    return f


VARIANTS = {
    "clip 1e-6 (before)": clipped(1e-6),
    "largest seen 1e-6 (shipped)": largest_seen(1e-6),
    "clip 1e-4": clipped(1e-4),
    "clip 1e-2": clipped(1e-2),
    "largest seen 1e-2": largest_seen(1e-2),
    "cap var 25": capped(25.0),
    "cap var 100": capped(100.0),
}


def one(args):
    name, seed = args
    warnings.simplefilter("ignore")
    nuts.covariance_from_hessian = VARIANTS[name]
    rng = np.random.default_rng([185, seed])
    bins = Bins.linear(0, 1, 4)
    prior = P.LogisticGP()
    lk, _ = prior_predictive(prior, bins, 10, noise=0.2)(rng)
    t = time.perf_counter()
    r = binference.infer(
        lk, prior, method="nuts", n_chains=1, n_warmup=300, n_samples=200, rng=rng, check=False
    )
    dt = time.perf_counter() - t
    return name, dt, float(np.min(r.diagnostics["ess_bulk"]))


if __name__ == "__main__":
    jobs = [(n, s) for n in VARIANTS for s in range(20)]
    with mp.get_context("spawn").Pool(int(sys.argv[1])) as pool:
        res = pool.map(one, jobs, chunksize=1)
    for n in VARIANTS:
        t = np.array([r[1] for r in res if r[0] == n])
        e = np.array([r[2] for r in res if r[0] == n], dtype=float)
        print(f"{n:28s} total {t.sum():7.1f} s  max {t.max():6.1f}  over 5 s {(t > 5).sum():2d}  min ESS {np.nanmin(e):.0f}")
