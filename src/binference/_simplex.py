"""Simplex helpers: softmax, isometric log-ratio (ILR) and centred log-ratio maps.

The ILR map uses the orthonormal Helmert basis ``Q`` (shape ``(K, K-1)``,
columns orthonormal and summing to zero):

    x = Q.T @ log(rho)          (ILR)
    rho = softmax(Q @ x)        (inverse ILR)

The Jacobian of ``x -> rho[:K-1]`` has determinant ``sqrt(K) * prod(rho)``,
so the log-density of ``x`` is ``log p(rho) + sum(log rho) + 0.5 log K``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from ._utils import frozen


@lru_cache(maxsize=64)
def _helmert(K: int) -> np.ndarray:
    V = np.zeros((K, K - 1))
    for j in range(1, K):
        V[:j, j - 1] = 1.0 / j
        V[j, j - 1] = -1.0
        V[:, j - 1] *= np.sqrt(j / (j + 1.0))
    return frozen(V)


def helmert_basis(K: int) -> np.ndarray:
    """Orthonormal ``(K, K-1)`` basis of the zero-sum subspace of R^K."""
    if K < 2:
        raise ValueError("The simplex needs at least two bins (K >= 2).")
    return _helmert(int(K))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along ``axis``."""
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax along ``axis``."""
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))
