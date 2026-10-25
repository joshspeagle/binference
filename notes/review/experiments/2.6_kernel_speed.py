"""Step 2.6: cost of one log-likelihood + gradient (two passes over L).

Times kern.loglike_and_counts for dense float64, dense float32 and sparse CSR
storage, single-threaded BLAS (OPENBLAS_NUM_THREADS=1), and reports the memory
bandwidth implied (bytes of L read per second).
"""
import time

import numpy as np
import scipy.sparse as sp

from pastry import _kernels as kern

rng = np.random.default_rng(0)


def bench(L, w, rho, reps=20):
    kern.loglike_and_counts(L, w, rho)
    t = time.perf_counter()
    for _ in range(reps):
        kern.loglike_and_counts(L, w, rho)
    return (time.perf_counter() - t) / reps


for N, K in [(20000, 80), (200000, 100), (1000000, 50)]:
    z = rng.random(N)
    c = (np.arange(K) + 0.5) / K
    sig = 0.02 + 0.08 * rng.random(N)
    L = np.exp(-0.5 * ((z[:, None] - c[None, :]) / sig[:, None]) ** 2)
    L /= L.max(1, keepdims=True)
    w = np.ones(N)
    rho = np.full(K, 1.0 / K)
    t64 = bench(L, w, rho)
    L32 = L.astype(np.float32)
    t32 = bench(L32, w, rho.astype(np.float32))
    Ls = L.copy()
    Ls[Ls < 1e-8] = 0
    S = sp.csr_array(Ls)
    ts = bench(S, w, rho)
    print(f"N={N:>8} K={K:>4} dense64 {1e3*t64:8.2f} ms ({2*L.nbytes/t64/1e9:5.1f} GB/s)  "
          f"dense32 {1e3*t32:8.2f} ms  sparse(nnz {S.nnz/(N*K):.0%}) {1e3*ts:8.2f} ms")
