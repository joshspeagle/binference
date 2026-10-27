"""Adaptive bin averages, in the measure coordinate or in ``z`` (DECISIONS #23)."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from typing import Literal

import numpy as np
from scipy import special

from ._utils import chunk_slices, warn
from .bins import Bins
from .measure import _identity
from .warnings import InputWarning

#: Most function values a quadrature evaluates in one call.
MAX_ELEMENTS = 2**22


class _Rule:
    # a tensor-product Gauss-Legendre rule on the unit cube, kept per axis
    def __init__(self, n: int, ndim: int) -> None:
        x, w = np.polynomial.legendre.leggauss(int(n))
        self.t = 0.5 * (x + 1.0)
        idx = np.array(list(product(range(int(n)), repeat=ndim)))  # (M, ndim)
        self.idx = idx.T
        self.logw = np.log(0.5 * w)[idx].sum(axis=1)  # (M,)

    def points(self, lo: np.ndarray, hi: np.ndarray, inverse) -> np.ndarray:
        # (C, M, ndim) points in z for cells [lo, hi] given in the integration coordinate
        axes = [
            inv(lo[:, d, None] + (hi - lo)[:, d, None] * self.t)[:, i]
            for d, (inv, i) in enumerate(zip(inverse, self.idx, strict=True))
        ]
        return np.stack(axes, axis=-1)


def bin_log_average(
    logf: Callable[..., np.ndarray],
    bins: Bins,
    *,
    rows: int | None = None,
    coordinate: Literal["measure", "z"] = "measure",
    tol: float = 1e-3,
    max_depth: int = 12,
    n_sub: int | None = None,
    must_split: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Log of the average of ``exp(logf)`` over each active bin, shape ``(N, K)``.

    The average is taken uniformly in the measure coordinate ``u`` (the
    within-bin population shape) or in ``z``. Two calling conventions:

    * ``rows=None``: ``logf(z)`` receives points of shape ``(M,)`` (1-D) or
      ``(M, ndim)`` and returns ``(N, M)`` log-values for all ``N`` rows; a
      cell is refined for every row at once.
    * ``rows=N``: ``logf(z, r)`` receives points of shape ``(P, M)`` or
      ``(P, M, ndim)`` and row indices ``r`` of shape ``(P,)``, and returns
      ``(P, M)``: row ``r[p]`` at the points ``z[p]``. Cells are refined per row.

    Each bin starts as one cell. A cell is estimated with 5 and 3 Gauss-Legendre
    points per axis; while their difference exceeds ``tol`` relative to the row's
    largest bin average (in ``u``) or bin mass (in ``z``), it is split in half
    along every axis, at most ``max_depth`` times, after which an
    :class:`~binference.InputWarning` is raised. With ``n_sub`` a fixed
    ``n_sub``-point rule is used instead.

    ``must_split(z_lo, z_hi, r)`` (per-row mode) marks cells, given by their
    bounds in ``z``, that must be split whatever the error estimate says: a
    function narrower than a cell can fall between the points of both rules,
    which then agree on a wrong value.
    """
    if not tol > 0 or max_depth < 0:
        raise ValueError(
            f"tol must be positive and max_depth non-negative; got {tol}, {max_depth}."
        )
    if rows is not None:
        # refine blocks of rows separately, so the per-row state stays bounded
        block = max(1, MAX_ELEMENTS // (8 * bins.K))
        if rows > block:
            kw = {"coordinate": coordinate, "tol": tol, "max_depth": max_depth, "n_sub": n_sub}
            return np.concatenate(
                [
                    bin_log_average(
                        lambda z, r, start=sl.start: logf(z, r + start),
                        bins,
                        rows=sl.stop - sl.start,
                        must_split=None
                        if must_split is None
                        else (lambda a, b, r, start=sl.start: must_split(a, b, r + start)),
                        **kw,
                    )
                    for sl in chunk_slices(rows, block)
                ]
            )
    nd = bins.ndim
    if coordinate == "measure":
        edges, inverse = bins.u_edges, [m.inverse for m in bins.measures]
    else:
        edges, inverse = bins.edges, [_identity] * nd
    j = bins.cell_index
    lo = np.stack([e[:-1][i] for e, i in zip(edges, j, strict=True)], axis=-1)
    hi = np.stack([e[1:][i] for e, i in zip(edges, j, strict=True)], axis=-1)
    owner = np.arange(bins.K)
    rules = [_Rule(n_sub, nd)] if n_sub is not None else [_Rule(5, nd), _Rule(3, nd)]

    if rows is None:

        def integrate(rule, lo, hi, row):
            # (N, C) log averages over the cells, chunked to bound memory
            z = rule.points(lo, hi, inverse)
            C, M = z.shape[:2]
            first = _call(logf, z[:1], M)
            if first.shape[0] == 0:
                raise ValueError("The function returned no rows.")
            size = max(1, MAX_ELEMENTS // (first.shape[0] * M))
            parts = [special.logsumexp(first.reshape(-1, 1, M) + rule.logw, axis=2)]
            for sl in chunk_slices(C - 1, size):
                v = _call(logf, z[1:][sl], M)
                parts.append(special.logsumexp(v.reshape(v.shape[0], -1, M) + rule.logw, axis=2))
            return np.concatenate(parts, axis=1)

        row = None
    else:

        def integrate(rule, lo, hi, row):
            # (P,) log averages, one row per cell
            out = np.empty(lo.shape[0])
            for sl in chunk_slices(lo.shape[0], max(1, MAX_ELEMENTS // len(rule.logw))):
                z = rule.points(lo[sl], hi[sl], inverse)
                v = np.asarray(logf(z[..., 0] if nd == 1 else z, row[sl]), dtype=float)
                out[sl] = special.logsumexp(v + rule.logw, axis=1)
            return out

        K = bins.K
        row = np.repeat(np.arange(rows), K)
        owner = np.tile(owner, rows)
        lo, hi = np.tile(lo, (rows, 1)), np.tile(hi, (rows, 1))

    logfrac = np.zeros(owner.size)
    fine, *coarse = (integrate(rule, lo, hi, row) for rule in rules)
    N = fine.shape[0] if rows is None else rows
    out = np.full((N, bins.K), -np.inf)
    if n_sub is not None:
        return fine if rows is None else fine.reshape(N, bins.K)
    # errors are measured on averages (in u) or on masses (in z), relative to the
    # row's largest; the reference grows as refinement resolves narrow peaks
    logvol = np.zeros(bins.K) if coordinate == "measure" else np.log(bins.volumes)
    rowmax = np.full(N, -np.inf)
    corners = np.array(list(product((0.0, 0.5), repeat=nd)))
    unconverged = np.zeros(bins.K, dtype=bool)
    for depth in range(max_depth + 1):
        shift = logfrac + logvol[owner]
        if rows is None:
            rowmax = np.maximum(rowmax, (fine + shift).max(axis=1))
        else:
            np.maximum.at(rowmax, row, fine + shift)
        rowmax = np.maximum(rowmax, (out + logvol).max(axis=1))
        scale = (rowmax[:, None] if rows is None else rowmax[row]) - logvol[owner]
        with np.errstate(invalid="ignore"):
            diff = np.nan_to_num(np.abs(np.exp(fine - scale) - np.exp(coarse[0] - scale)))
        err = np.exp(logfrac) * (diff.max(axis=0) if rows is None else diff)
        ok = err <= tol
        if must_split is not None:
            z_lo = np.stack([inv(lo[:, d]) for d, inv in enumerate(inverse)], axis=-1)
            z_hi = np.stack([inv(hi[:, d]) for d, inv in enumerate(inverse)], axis=-1)
            ok &= ~must_split(z_lo, z_hi, row)
        if depth == max_depth:
            unconverged[owner[~ok]] = True
            ok[:] = True
        if rows is None:
            np.logaddexp.at(out.T, owner[ok], (fine[:, ok] + logfrac[ok]).T)
        else:
            np.logaddexp.at(out, (row[ok], owner[ok]), fine[ok] + logfrac[ok])
        if ok.all():
            break
        keep = ~ok
        half = 0.5 * (hi[keep] - lo[keep])
        lo = (lo[keep][:, None, :] + 2 * half[:, None, :] * corners[None]).reshape(-1, nd)
        hi = lo + np.repeat(half, len(corners), axis=0)
        owner = np.repeat(owner[keep], len(corners))
        logfrac = np.repeat(logfrac[keep], len(corners)) - nd * np.log(2.0)
        if row is not None:
            row = np.repeat(row[keep], len(corners))
        fine, *coarse = (integrate(rule, lo, hi, row) for rule in rules)
    if unconverged.any():
        warn(
            f"Bin averages did not reach tol={tol:g} in {int(unconverged.sum())} bin(s) after "
            f"{max_depth} refinements; raise max_depth or tol.",
            InputWarning,
        )
    return out


def _call(logf: Callable, z: np.ndarray, M: int) -> np.ndarray:
    # logf at the points of the cells in z (C, M, ndim), as (N, C * M)
    pts = z.reshape(-1, z.shape[-1])
    v = np.asarray(logf(pts[:, 0] if z.shape[-1] == 1 else pts), dtype=float)
    if v.ndim != 2 or v.shape[1] != pts.shape[0]:
        raise ValueError("The function must return an array of shape (N, n_points).")
    return v
