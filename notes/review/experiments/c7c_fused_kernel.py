"""C7c: one log-likelihood + expected counts, fused vs two passes (DECISIONS #54, #55).

Times, for dense L of the 2.6 experiment's shapes:
  two-pass   the current kernel (L @ rho, then L.T @ (w / s)), NumPy + BLAS
  chunked    the same two products on row blocks that stay in cache, NumPy + BLAS
  threads    those row blocks in four Python threads, one BLAS thread each
  numba-1    a fused single pass (each row read once), one thread
  numba-P    the same with prange over row blocks, all cores
in float64 and float32 (rho, w and the accumulators in L's dtype; log-likelihood summed
in float64), and reports the relative errors of the log-likelihood (ll) and of the
expected counts (n) against float64 two-pass. BLAS threads
are as set by the environment (run with OPENBLAS_NUM_THREADS=1 and without).
"""

import os
import time

import numba
import numpy as np

rng = np.random.default_rng(0)


def two_pass(L, w, rho):
    s = L @ rho
    return float(np.dot(w, np.log(s.astype(np.float64)))), rho * (L.T @ (w / s))


def chunked(L, w, rho, rows=4096):
    ll, acc = 0.0, np.zeros(rho.size)  # counts accumulated across blocks in float64
    for a in range(0, L.shape[0], rows):
        B = L[a : a + rows]
        s = B @ rho
        ll += float(np.dot(w[a : a + rows], np.log(s.astype(np.float64))))
        acc += B.T @ (w[a : a + rows] / s)
    return ll, rho * acc


@numba.njit(cache=True, fastmath=False)
def _fused_block(L, w, rho, lo, hi, acc):
    ll = 0.0
    K = L.shape[1]
    for i in range(lo, hi):
        s = L.dtype.type(0)
        for k in range(K):
            s += L[i, k] * rho[k]
        ll += w[i] * np.log(np.float64(s))
        f = w[i] / s
        for k in range(K):
            acc[k] += f * L[i, k]
    return ll


@numba.njit(cache=True)
def fused1(L, w, rho):
    acc = np.zeros_like(rho)
    ll = _fused_block(L, w, rho, 0, L.shape[0], acc)
    return ll, rho * acc


@numba.njit(cache=True, parallel=True)
def fusedP(L, w, rho):
    nb = numba.get_num_threads()
    N = L.shape[0]
    step = (N + nb - 1) // nb
    accs = np.zeros((nb, rho.size), dtype=rho.dtype)
    lls = np.zeros(nb)
    for b in numba.prange(nb):
        lls[b] = _fused_block(L, w, rho, b * step, min(N, (b + 1) * step), accs[b])
    return lls.sum(), rho * accs.sum(axis=0)


_pool = None


def threaded(L, w, rho, rows=2048, n_threads=4):
    # row blocks that stay in cache, four Python threads, one BLAS thread each
    global _pool
    from concurrent.futures import ThreadPoolExecutor

    from threadpoolctl import threadpool_limits

    _pool = _pool or ThreadPoolExecutor(n_threads)
    N = L.shape[0]
    step = -(-N // n_threads)
    with threadpool_limits(1, "blas"):
        parts = list(_pool.map(lambda t: chunked(L[t * step : (t + 1) * step], w[t * step : (t + 1) * step], rho, rows), range(n_threads)))
    return sum(p[0] for p in parts), sum(p[1] for p in parts)


def bench(f, L, w, rho, reps=10, repeats=3):
    # the best of several repeats, against the noise of a shared container
    out = f(L, w, rho)
    best = np.inf
    for _ in range(repeats):
        t = time.perf_counter()
        for _ in range(reps):
            out = f(L, w, rho)
        best = min(best, (time.perf_counter() - t) / reps)
    return best, out


print(f"threads: BLAS {os.environ.get('OPENBLAS_NUM_THREADS', 'default')}, numba {numba.get_num_threads()}")
for N, K in [(20000, 80), (200000, 100), (1000000, 50)]:
    z = rng.random(N)
    c = (np.arange(K) + 0.5) / K
    sig = 0.02 + 0.08 * rng.random(N)
    L = np.exp(-0.5 * ((z[:, None] - c[None, :]) / sig[:, None]) ** 2)
    L /= L.max(1, keepdims=True)
    w = np.ones(N)
    rho = rng.dirichlet(np.ones(K))
    ref = two_pass(L, w, rho)
    for dtype in (np.float64, np.float32):
        Ld, wd, rd = L.astype(dtype), w.astype(dtype), rho.astype(dtype)
        row = []
        for name, f in [("two-pass", two_pass), ("chunked", chunked), ("threads", threaded), ("numba-1", fused1), ("numba-P", fusedP)]:
            t, (ll, n) = bench(f, Ld, wd, rd)
            e_ll, e_n = abs(ll / ref[0] - 1), float(np.max(np.abs(n / ref[1] - 1)))
            row.append(f"{name} {1e3 * t:7.2f} ms (ll {e_ll:.0e}, n {e_n:.0e})")
        print(f"N={N:>8} K={K:>4} {np.dtype(dtype).name}: " + "  ".join(row))
