"""Tensor-product Gauss-Legendre points inside every bin."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .bins import Bins


def _gauss_legendre_unit(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes in [0, 1] and weights summing to one."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * (x + 1.0), 0.5 * w


@lru_cache(maxsize=16)
def _subbin_points(bins: Bins, n_sub: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature points inside every bin and their within-bin weights (cached, read-only).

    Returns ``points`` with shape ``(K * M,)`` (1-D) or ``(K * M, ndim)`` and
    ``weights`` with shape ``(M,)``, where ``M = n_sub ** ndim``; points for
    bin ``k`` occupy rows ``k * M : (k + 1) * M``.
    """
    u, wu = _gauss_legendre_unit(n_sub)
    edges = bins.edges
    per_axis = [e[:-1, None] + np.diff(e)[:, None] * u[None, :] for e in edges]  # (K_d, n)
    cell_idx = bins.cell_index  # per axis, (K,)
    if bins.ndim == 1:
        return _frozen(per_axis[0][cell_idx[0]].ravel()), _frozen(wu)
    # tensor-product points per cell, in the order of the (active) bins
    sub = np.meshgrid(*[np.arange(n_sub)] * bins.ndim, indexing="ij")
    sub_idx = [s.ravel() for s in sub]  # (M,) per axis
    pts = np.stack(
        [per_axis[d][cell_idx[d][:, None], sub_idx[d][None, :]] for d in range(bins.ndim)],
        axis=-1,
    )  # (K, M, ndim)
    w = np.ones(len(sub_idx[0]))
    for d in range(bins.ndim):
        w = w * wu[sub_idx[d]]
    return _frozen(pts.reshape(-1, bins.ndim)), _frozen(w)


def _frozen(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a
