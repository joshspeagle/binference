"""Low-level numerical kernels shared by models and engines.

Every function accepts the likelihood matrix ``L`` (shape ``(N, K)``) either as
a dense ``numpy.ndarray`` or as a ``scipy.sparse`` CSR array/matrix. The hot
paths are two matrix-vector products:

    s     = L @ rho                       (per-object marginal likelihoods)
    n_hat = rho * (L.T @ (w / s))          (expected counts / EM "stack")
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp

from ._utils import chunk_slices

__all__ = [
    "loglike",
    "loglike_and_counts",
    "matvec",
    "possible",
    "responsibilities",
    "rmatvec",
    "sample_counts",
    "sample_labels",
]

DEFAULT_CHUNK = 65536


def matvec(L, x: np.ndarray) -> np.ndarray:
    """``L @ x`` for dense or sparse ``L``."""
    return np.asarray(L @ x).ravel()


def rmatvec(L, y: np.ndarray) -> np.ndarray:
    """``L.T @ y`` for dense or sparse ``L``."""
    return np.asarray(L.T @ y).ravel()


def possible(s: np.ndarray) -> bool:
    """Whether every object's evidence ``s_i = L_i . rho`` is positive.

    Rows are normalised to a maximum of one, so an evidence below 1e-250
    (log-likelihood < -575) is effectively impossible; the one test used by the
    log-likelihood, EM and the KKT gap (it also keeps ``w / s`` finite).
    """
    return bool(s.min() > 1e-250)  # the comparison also catches NaN


def loglike(L, w: np.ndarray, rho: np.ndarray) -> float:
    """``sum_i w_i log(L_i . rho)`` (``-inf`` if some object is impossible), one pass."""
    s = matvec(L, rho)
    return float(np.dot(w, np.log(s))) if possible(s) else -np.inf


def loglike_and_counts(L, w: np.ndarray, rho: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Weighted marginal log-likelihood, expected counts and per-object evidences.

    Parameters
    ----------
    L : (N, K) array or sparse matrix
        Bin-averaged likelihoods (rows may carry arbitrary positive scales).
    w : (N,) array
        Object weights.
    rho : (K,) array
        Population fractions (or rates); must be non-negative.

    Returns
    -------
    loglike : float
        ``sum_i w_i log(L_i . rho)`` (``-inf`` if any object has zero evidence).
    n_hat : (K,) array
        ``rho_k * sum_i w_i L_ik / (L_i . rho)``.
    s : (N,) array
        ``L_i . rho``.
    """
    s = matvec(L, rho)
    if not possible(s):
        return -np.inf, np.zeros_like(rho), s
    ll = float(np.dot(w, np.log(s)))
    if not math.isfinite(ll):
        return -np.inf, np.zeros_like(rho), s
    n_hat = rho * rmatvec(L, w / s)
    return ll, n_hat, s


def responsibilities(
    L, rho: np.ndarray, idx=None, chunk: int | None = DEFAULT_CHUNK
) -> np.ndarray:
    """Per-object posterior over bins given ``rho``: ``r_ik = L_ik rho_k / (L_i . rho)``.

    Returns a dense ``(n, K)`` array for the selected objects ``idx`` (all by
    default). Memory scales as ``n * K``; select a subset for large catalogues.
    """
    if idx is not None:
        L = L[np.atleast_1d(idx)]
    N = L.shape[0]
    out = np.empty((N, rho.size))
    for sl in chunk_slices(N, chunk):
        block = L[sl]
        block = block.toarray() if sp.issparse(block) else np.asarray(block)
        P = block * rho
        tot = P.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[sl] = np.where(tot > 0, P / tot, 0.0)
    return out


def _sample_labels_dense(Lc: np.ndarray, rho: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    P = Lc * rho
    c = np.cumsum(P, axis=1)
    # u in (0, total]: the first index with c_k >= u always has P_k > 0
    u = (1.0 - rng.random(Lc.shape[0])) * c[:, -1]
    idx = np.sum(c < u[:, None], axis=1)
    return np.minimum(idx, Lc.shape[1] - 1)


def _sample_labels_csr(Lc, rho: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    Lc = sp.csr_array(Lc)
    indptr, indices = Lc.indptr, Lc.indices
    data = Lc.data * rho[indices]
    n = Lc.shape[0]
    # normalise each row so that cumulative sums stay O(n): avoids cancellation
    row_tot = np.asarray(Lc @ rho).ravel()
    counts = np.diff(indptr)
    data = data / np.repeat(np.where(row_tot > 0, row_tot, 1.0), counts)
    cs = np.cumsum(data)
    start = np.concatenate(([0.0], cs))[indptr[:-1]]
    u = start + (1.0 - rng.random(n))
    pos = np.searchsorted(cs, u, side="left")
    pos = np.clip(pos, indptr[:-1], np.maximum(indptr[1:] - 1, indptr[:-1]))
    return indices[pos]


def sample_labels(L, rho: np.ndarray, rng: np.random.Generator, chunk: int | None = DEFAULT_CHUNK):
    """Draw one bin label per object from ``Cat(r_i(rho))`` (vectorised, chunked)."""
    N = L.shape[0]
    out = np.empty(N, dtype=np.int64)
    sparse = sp.issparse(L)
    for sl in chunk_slices(N, chunk):
        block = L[sl]
        if sparse:
            out[sl] = _sample_labels_csr(block, rho, rng)
        else:
            out[sl] = _sample_labels_dense(np.asarray(block), rho, rng)
    return out


def _multinomial_csr(Lc, m: np.ndarray, rho: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draws of ``Mult(m_i, r_i)`` on the nonzeros of each CSR row, one per stored entry.

    Conditional binomials position by position within rows, rows sorted by
    their number of nonzeros: the cost follows the nonzeros, not K or ``m_i``.
    """
    nnz = np.diff(Lc.indptr)
    order = np.argsort(-nnz, kind="stable")
    n_sorted = nnz[order]
    start = Lc.indptr[:-1][order]
    p = Lc.data * rho[Lc.indices]
    mass = np.asarray(Lc @ rho, dtype=float).ravel()[order]
    left = m[order].astype(np.int64)
    out = np.zeros(p.size)
    for j in range(int(n_sorted[0]) if n_sorted.size else 0):
        a = int(np.searchsorted(-n_sorted, -j, side="left"))  # rows with more than j nonzeros
        pos = start[:a] + j
        q = np.divide(p[pos], mass[:a], out=np.ones(a), where=mass[:a] > 0)
        q = np.where(n_sorted[:a] == j + 1, 1.0, np.clip(q, 0.0, 1.0))  # last entry takes the rest
        d = rng.binomial(left[:a], q)
        out[pos] = d
        left[:a] -= d
        mass[:a] -= p[pos]
    return out


def sample_counts(
    L,
    w_int: np.ndarray,
    rho: np.ndarray,
    rng: np.random.Generator,
    chunk: int | None = DEFAULT_CHUNK,
    unit_weights: bool | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Draw latent bin counts ``n_k = sum_i w_i 1[z_i = k]``.

    For unit weights a label is drawn per object (and returned). Integer weights
    ``m_i`` (compressed duplicate rows) draw one ``Mult(m_i, r_i)`` per row and
    return ``labels=None``; for sparse ``L`` only over the row's nonzeros, so
    the cost follows the nonzeros whatever K and the ``m_i`` (audit G-2).

    Returns
    -------
    counts : (K,) float array
    labels : (N,) int array or None
    """
    K = rho.size
    if unit_weights is None:
        unit_weights = bool(np.all(w_int == 1))
    if unit_weights:
        labels = sample_labels(L, rho, rng, chunk)
        return np.bincount(labels, minlength=K).astype(float), labels
    counts = np.zeros(K)
    N = L.shape[0]
    m_all = np.asarray(np.round(w_int), dtype=np.int64)
    if sp.issparse(L):
        for sl in chunk_slices(N, chunk):
            Lc = sp.csr_array(L[sl])
            n = _multinomial_csr(Lc, m_all[sl], rho, rng)
            counts += np.bincount(Lc.indices, weights=n, minlength=K)
        return counts, None
    for sl in chunk_slices(N, chunk):
        R = responsibilities(L[sl], rho, chunk=None)
        # guard tiny negative round-off and renormalise
        R = np.clip(R, 0.0, None)
        R /= R.sum(axis=1, keepdims=True)
        counts += rng.multinomial(m_all[sl], R).sum(axis=0)
    return counts, None
