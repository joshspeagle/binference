"""C7c: where sparse storage beats dense for one log-likelihood + expected counts.

Banded likelihoods (each row nonzero on a contiguous window) at several fill
fractions; times kern.loglike_and_counts for dense float64 and CSR storage with the
environment's BLAS threads.
"""

import time

import numpy as np
import scipy.sparse as sp

from binference import _kernels as kern

rng = np.random.default_rng(0)


def bench(L, w, rho, reps=10):
    kern.loglike_and_counts(L, w, rho)
    t = time.perf_counter()
    for _ in range(reps):
        kern.loglike_and_counts(L, w, rho)
    return (time.perf_counter() - t) / reps


for N, K in [(200000, 100), (50000, 1000)]:
    w, rho = np.ones(N), np.full(K, 1.0 / K)
    for fill in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        width = max(1, int(fill * K))
        start = rng.integers(0, K - width + 1, N)
        cols = (start[:, None] + np.arange(width)).ravel()
        rows = np.repeat(np.arange(N), width)
        S = sp.csr_array((rng.random(N * width) + 0.1, (rows, cols)), shape=(N, K))
        D = S.toarray()
        td, ts = bench(D, w, rho), bench(S, w, rho)
        print(f"N={N:>7} K={K:>5} fill {fill:4.0%}: dense {1e3 * td:7.2f} ms  sparse {1e3 * ts:7.2f} ms  ratio {ts / td:5.2f}")
