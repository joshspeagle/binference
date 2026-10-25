"""Quadrature inside bins: fixed Gauss-Legendre points and adaptive bin averages."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from itertools import product
from typing import Literal

import numpy as np
from scipy import special

from .bins import Bins


def _gauss_legendre_unit(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes in [0, 1] and weights summing to one."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * (x + 1.0), 0.5 * w


@lru_cache(maxsize=4)
def _subbin_points(bins: Bins, n_sub: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature points inside every bin and their within-bin weights.

    Cached for repeated calls on the same bins, so the arrays are read-only;
    pass a copy to code that may modify them.

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


def _identity(x: np.ndarray) -> np.ndarray:
    return x


def _tensor_rule(n: int, ndim: int) -> tuple[np.ndarray, np.ndarray]:
    # nodes (M, ndim) in the unit cube and log weights (M,) summing to one
    t, w = _gauss_legendre_unit(n)
    nodes = np.stack([g.ravel() for g in np.meshgrid(*[t] * ndim, indexing="ij")], axis=-1)
    weights = np.prod(np.meshgrid(*[w] * ndim, indexing="ij"), axis=0).ravel()
    return nodes, np.log(weights)


def bin_log_average(
    logf: Callable[[np.ndarray], np.ndarray],
    bins: Bins,
    *,
    coordinate: Literal["measure", "z"] = "measure",
    tol: float = 1e-3,
    max_depth: int = 12,
    n_sub: int | None = None,
    max_elements: int = 2**24,
) -> tuple[np.ndarray, np.ndarray]:
    """Log of the average of ``exp(logf)`` over each active bin, shape ``(N, K)``.

    ``logf(points)`` returns ``(N, M)`` log-values at ``M`` points of shape
    ``(M,)`` (1-D) or ``(M, ndim)``. The average is taken uniformly in the
    measure coordinate ``u`` (the within-bin population shape) or in ``z``.

    Each bin starts as one cell. A cell's integral is estimated with 5 and 3
    Gauss-Legendre points per axis; a cell whose difference, relative to each
    row's largest bin average, exceeds ``tol`` is split in half along every
    axis, up to ``max_depth`` times. With ``n_sub`` a fixed ``n_sub``-point rule
    is used and nothing is refined. Returns the averages and a ``(K,)``
    boolean array marking bins that did not reach ``tol``.
    """
    nd = bins.ndim
    if coordinate == "measure":
        edges, inverse = bins.u_edges, [m.inverse for m in bins.measures]
    else:
        edges, inverse = bins.edges, [_identity] * nd
    j = bins.cell_index
    lo = np.stack([e[:-1][i] for e, i in zip(edges, j, strict=True)], axis=-1)
    hi = np.stack([e[1:][i] for e, i in zip(edges, j, strict=True)], axis=-1)
    rules = (
        [_tensor_rule(n_sub, nd)]
        if n_sub is not None
        else [_tensor_rule(5, nd), _tensor_rule(3, nd)]
    )

    def integrals(lo, hi):
        # per rule, the (N, C) log averages over each cell
        out = []
        for nodes, logw in rules:
            M = nodes.shape[0]
            parts, start, size = [], 0, 64
            while start < lo.shape[0]:
                sl = slice(start, start + size)
                u = lo[sl, None, :] + (hi[sl] - lo[sl])[:, None, :] * nodes[None]
                z = np.stack(
                    [
                        u[..., d] if inverse[d] is None else inverse[d](u[..., d])
                        for d in range(nd)
                    ],
                    axis=-1,
                ).reshape(-1, nd)
                v = np.asarray(logf(z[:, 0] if nd == 1 else z), dtype=float)
                C = u.shape[0]
                if v.ndim != 2 or v.shape[1] != C * M:
                    raise ValueError("The function must return an array of shape (N, n_points).")
                parts.append(special.logsumexp(v.reshape(-1, C, M) + logw, axis=2))
                start += C
                size = max(1, max_elements // max(1, v.shape[0] * M))
            out.append(np.concatenate(parts, axis=1))
        return out

    K = bins.K
    owner = np.arange(K)
    logfrac = np.zeros(K)
    fine, *coarse = integrals(lo, hi)
    if n_sub is not None:
        return fine, np.zeros(K, dtype=bool)
    rowmax = fine.max(axis=1, keepdims=True)
    done_owner, done_val = [], []
    unconverged = np.zeros(K, dtype=bool)
    corners = np.array(list(product((0.0, 0.5), repeat=nd)))
    for depth in range(max_depth + 1):
        with np.errstate(invalid="ignore"):
            diff = np.abs(np.exp(fine - rowmax) - np.exp(coarse[0] - rowmax))
        err = np.exp(logfrac) * np.nan_to_num(diff).max(axis=0)
        ok = err <= tol
        if depth == max_depth:
            unconverged[owner[~ok]] = True
            ok[:] = True
        done_owner.append(owner[ok])
        done_val.append(fine[:, ok] + logfrac[ok])
        if ok.all():
            break
        lo, hi, owner, logfrac = lo[~ok], hi[~ok], owner[~ok], logfrac[~ok]
        half = 0.5 * (hi - lo)
        lo = (lo[:, None, :] + 2 * half[:, None, :] * corners[None]).reshape(-1, nd)
        hi = lo + np.repeat(half, len(corners), axis=0)
        owner = np.repeat(owner, len(corners))
        logfrac = np.repeat(logfrac, len(corners)) - nd * np.log(2.0)
        fine, *coarse = integrals(lo, hi)
    owner = np.concatenate(done_owner)
    vals = np.concatenate(done_val, axis=1)
    order = np.argsort(owner, kind="stable")
    starts = np.searchsorted(owner[order], np.arange(K))
    return np.logaddexp.reduceat(vals[:, order], starts, axis=1), unconverged
