"""Adaptive bin averages, in the measure coordinate or in ``z``."""
# DECISIONS #23

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
#: How far inside a cell, as a fraction of its width, the check rule reads its ends: a jump
#: closer to an edge goes unseen, and one exactly on an edge reads as the inside value.
EDGE_INSET = 1e-9


def _gauss(n: int) -> tuple[np.ndarray, np.ndarray]:
    # Gauss-Legendre nodes and weights on [0, 1]
    x, w = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * (x + 1.0), 0.5 * w


# Lobatto's 5-point rule on [0, 1]
_LOBATTO = (
    np.array([EDGE_INSET, 0.5 - np.sqrt(21) / 14, 0.5, 0.5 + np.sqrt(21) / 14, 1 - EDGE_INSET]),
    np.array([9.0, 49.0, 64.0, 49.0, 9.0]) / 180,
)


class _Rule:
    # a tensor-product rule on the unit cube from nodes t and weights w on [0, 1], kept per axis
    def __init__(self, t: np.ndarray, w: np.ndarray, ndim: int) -> None:
        self.t = t
        idx = np.array(list(product(range(t.size), repeat=ndim)))  # (M, ndim)
        self.idx = idx.T
        self.logw = np.log(w)[idx].sum(axis=1)  # (M,)

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
    jumps: bool = False,
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
    largest bin average (in ``u``) or bin mass (in ``z``) as then estimated, it is split
    in half along every axis, at most ``max_depth`` times, after which an
    :class:`~binference.InputWarning` is raised. The estimate is the sum over the current
    cells, so it falls as well as rises, and a cell accepted earlier is split if it falls:
    a narrow peak that a coarse cell read and its children miss leaves a low estimate,
    against which the peak's tails are followed back up to it. Above a broad background
    the tails do not stand out, so such a spike can still be lost. With ``n_sub`` a fixed
    ``n_sub``-point rule is used instead.

    The Gauss-Legendre points leave the outer 4.7% of a cell at each end unread, so
    a jump there goes unseen. With ``jumps`` each cell is also checked against
    Lobatto's 5-point rule with its ends :data:`EDGE_INSET` inside the cell. That
    suits a bounded function such as a detection probability: at an integrable
    singularity on an edge the check would refine to ``max_depth``. A row whose
    5-point estimates are all zero gives no scale for the error: where a check rule
    reads something there, its cells are split until they are below ``tol`` of their
    bin, and keep what the 5-point rule then reads, with an
    :class:`~binference.InputWarning` where that is still nothing. The same warning is
    raised for a bin that a coarser cell read and that ends with nothing read.

    ``must_split(z_lo, z_hi, r)`` (per-row mode) marks cells, given by their
    bounds in ``z``, that must be split whatever the error estimate says: a
    function narrower than a cell can fall between the points of every rule,
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
            return np.concatenate(
                [
                    bin_log_average(
                        lambda z, r, start=sl.start: logf(z, r + start),
                        bins,
                        rows=sl.stop - sl.start,
                        must_split=None
                        if must_split is None
                        else (lambda a, b, r, start=sl.start: must_split(a, b, r + start)),
                        coordinate=coordinate,
                        tol=tol,
                        max_depth=max_depth,
                        n_sub=n_sub,
                        jumps=jumps,
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
    if n_sub is not None:
        rules = [_Rule(*_gauss(n_sub), nd)]
    else:
        rules = [_Rule(*_gauss(5), nd), _Rule(*_gauss(3), nd)]
        if jumps:
            rules.append(_Rule(*_LOBATTO, nd))

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

    def marked(lo, hi, row):
        # the cells that must_split says to split whatever the error estimate says
        if must_split is None:
            return np.zeros(lo.shape[0], dtype=bool)
        z_lo = np.stack([inv(lo[:, d]) for d, inv in enumerate(inverse)], axis=-1)
        z_hi = np.stack([inv(hi[:, d]) for d, inv in enumerate(inverse)], axis=-1)
        return must_split(z_lo, z_hi, row)

    logfrac = np.zeros(owner.size)
    fine, *coarse = (integrate(rule, lo, hi, row) for rule in rules)
    N = fine.shape[0] if rows is None else rows
    if n_sub is not None:
        return fine if rows is None else fine.reshape(N, bins.K)
    logerr = _log_disagreement(fine, coarse)
    # errors are measured on averages (in u) or on masses (in z), relative to the row's
    # largest as now estimated. Every cell stays in play, so one accepted against an
    # estimate that later falls, such as a coarse cell's reading of a narrow peak, reopens
    logvol = np.zeros(bins.K) if coordinate == "measure" else np.log(bins.volumes)
    log_tol = np.log(tol)
    corners = np.array(list(product((0.0, 0.5), repeat=nd)))
    cell_depth = np.zeros(owner.size, dtype=int)
    forced = marked(lo, hi, row)
    seen = np.zeros((N, bins.K), dtype=bool)
    while True:
        out = np.full((N, bins.K), -np.inf)
        if rows is None:
            np.logaddexp.at(out.T, owner, (fine + logfrac).T)
        else:
            np.logaddexp.at(out, (row, owner), fine + logfrac)
        seen |= out > -np.inf
        ref = (out + logvol).max(axis=1)
        blind = ref == -np.inf  # no scale for the error: stop below tol of the bin
        scale = (ref[:, None] if rows is None else ref[row]) - logvol[owner]
        with np.errstate(invalid="ignore"):  # NaN: nothing read on either side, which passes
            excess = logerr + logfrac - scale > log_tol
        stopped = None
        if blind.any():
            small = logfrac < log_tol
            mask = blind[:, None] & small if rows is None else blind[row] & small
            stopped = logerr > -np.inf
            stopped = (stopped & mask).any(axis=0) if rows is None else stopped & mask
            excess = excess & ~mask
        split = excess.any(axis=0) if rows is None else excess
        split |= forced
        deep = cell_depth >= max_depth
        if not (split & ~deep).any():
            break
        keep = split & ~deep
        half = 0.5 * (hi[keep] - lo[keep])
        kid_lo = (lo[keep][:, None, :] + 2 * half[:, None, :] * corners[None]).reshape(-1, nd)
        kid_hi = kid_lo + np.repeat(half, len(corners), axis=0)
        kid_owner = np.repeat(owner[keep], len(corners))
        kid_row = None if row is None else np.repeat(row[keep], len(corners))
        kid_fine, *kid_coarse = (integrate(rule, kid_lo, kid_hi, kid_row) for rule in rules)
        stay = ~keep
        lo, hi = np.concatenate([lo[stay], kid_lo]), np.concatenate([hi[stay], kid_hi])
        owner = np.concatenate([owner[stay], kid_owner])
        logfrac = np.concatenate(
            [logfrac[stay], np.repeat(logfrac[keep], len(corners)) - nd * np.log(2.0)]
        )
        cell_depth = np.concatenate(
            [cell_depth[stay], np.repeat(cell_depth[keep], len(corners)) + 1]
        )
        forced = np.concatenate([forced[stay], marked(kid_lo, kid_hi, kid_row)])
        logerr = np.concatenate(
            [logerr[..., stay], _log_disagreement(kid_fine, kid_coarse)], axis=-1
        )
        fine = np.concatenate([fine[..., stay], kid_fine], axis=-1)
        if row is not None:
            row = np.concatenate([row[stay], kid_row])
    unconverged = np.zeros(bins.K, dtype=bool)
    unconverged[owner[split]] = True
    unread = (seen & ~(out > -np.inf)).any(axis=0)  # read at a coarser level, then lost
    if stopped is not None:
        unread[owner[stopped & ~split]] = True
    if unconverged.any():
        warn(
            f"Bin averages did not reach tol={tol:g} in {int(unconverged.sum())} bin(s) after "
            f"{max_depth} refinements; raise max_depth or tol.",
            InputWarning,
        )
    if unread.any():
        warn(
            f"Bin averages read zero in {int(unread.sum())} bin(s) where the function is "
            f"non-zero only in a feature too narrow to resolve at tol={tol:g}; lower tol, or "
            "put bin edges at it.",
            InputWarning,
        )
    return out


def _log_disagreement(fine: np.ndarray, coarse: list[np.ndarray]) -> np.ndarray:
    # log of the largest |exp(fine) - exp(c)| over the check rules c
    with np.errstate(divide="ignore", invalid="ignore"):
        d = [np.maximum(fine, c) + np.log1p(-np.exp(-np.abs(fine - c))) for c in coarse]
    return np.fmax.reduce(d, axis=0)  # NaN where a rule and fine both read nothing


def _call(logf: Callable, z: np.ndarray, M: int) -> np.ndarray:
    # logf at the points of the cells in z (C, M, ndim), as (N, C * M)
    pts = z.reshape(-1, z.shape[-1])
    v = np.asarray(logf(pts[:, 0] if z.shape[-1] == 1 else pts), dtype=float)
    if v.ndim != 2 or v.shape[1] != pts.shape[0]:
        raise ValueError("The function must return an array of shape (N, n_points).")
    return v
