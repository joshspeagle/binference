"""Values drawn inside bins from each input's own within-bin information.

``sample_values``, on a ``PopulationResult`` or a ``PointEstimate``, draws a value for each
object inside the bin it is assigned to; :func:`_value_drawer` chooses how from what the
likelihoods kept: the per-object arrays of ``Likelihoods.meta``
(``_PER_OBJECT_KEYS`` and ``_WithinBin`` in ``likelihoods``).
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
from scipy import special

from . import _kernels as kern
from .density import _log_nonneg
from .likelihoods import _log_interim, _rows_logpdf
from .measure import Measure

#: Most values a density function returns per call when values are drawn (2**24 floats
#: are 128 MB).
EVAL_BUDGET = 2**24


def _value_drawer(model, loglike, n_sub: int, n_sweeps: int) -> tuple[Callable, bool]:
    """``draw(objects, labels, lo, hi, gen, height) -> (z, keep)``: values in the boxes
    ``[lo, hi]`` (in ``u``, ``(n, ndim)``) from the input's own likelihood times the
    population, flat in ``u`` times the hat's ``height(labels, d, u)`` per axis (``None``
    when flat); and whether the input has within-bin information."""
    # DECISIONS #50
    meta = model.likelihoods.meta
    measures = model.bins.measures
    flat = partial(_draw_subpoints, measures, n_sub, None)
    if loglike is not None:
        return partial(_draw_subpoints, measures, n_sub, _per_object(loglike)), True
    if "gaussian_mean" in meta:
        return partial(_draw_gauss, measures, n_sub, n_sweeps, meta), True
    within = meta.get("within")
    if within is None or within.data is None or "value_row" not in meta:
        return flat, False  # e.g. rows merged by compress, which drops value_row
    kind, *data = within.data
    row = np.asarray(meta["value_row"])
    if kind == "samples":
        # labels are the bins the samples are in
        direct = model.likelihoods.basis.is_identity and data[0] == model.K
        return partial(_draw_samples, measures, flat, direct, row, *data), True
    if kind == "dists":  # one family's parameters are packed once, not per call
        dists, interim = data
        evaluate = partial(_dists_logl, row, _rows_logpdf(dists, 1), interim)
    else:
        evaluate = partial(_rows_logl, row, *data)
    return partial(_draw_subpoints, measures, n_sub, evaluate), True


def _dists_logl(row, logpdf, interim, objects, labels, zz):
    # distributions of from_density: one vectorised call for the rows' own points
    with np.errstate(invalid="ignore"):  # -inf - -inf where the interim vanishes: NaN
        return logpdf(zz, row[objects]) - _log_interim(interim, zz)


def _rows_logl(row, n_in, fn, log, interim, objects, labels, zz):
    # a density function of from_density returns every input row at common points: it
    # is called on the sub-points of a few bins at a time, at most EVAL_BUDGET values
    out = np.empty(zz.shape)
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    per = max(1, EVAL_BUDGET // (n_in * zz.shape[1]))
    for c in range(0, len(first), per):
        at = np.flatnonzero((inverse >= c) & (inverse < c + per))
        pts = zz[first[c : c + per]]
        v = fn(pts.ravel())
        v = np.asarray(v if log else _log_nonneg(v), dtype=float).reshape(n_in, *pts.shape)
        out[at] = v[row[objects[at]], inverse[at] - c]
    with np.errstate(invalid="ignore"):  # -inf - -inf where the interim vanishes: NaN
        return out - _log_interim(interim, zz)


def _per_object(loglike: Callable[[np.ndarray, int], np.ndarray]) -> Callable:
    """An evaluator ``(objects, labels, zz) -> log-likelihoods`` calling ``loglike(z, i)``
    once per object, at the sub-points of all its bins."""

    def evaluate(objects, labels, zz):
        key = objects.astype(np.int64) * (int(labels.max()) + 1) + labels  # by object
        _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
        vals = np.empty((len(first), zz.shape[1]))
        ids, starts = np.unique(objects[first], return_index=True)
        for i, a, b in zip(ids, starts, [*starts[1:], len(first)], strict=True):
            pts = zz[first[a:b]]
            vals[a:b] = np.asarray(loglike(pts.ravel(), int(i)), dtype=float).reshape(pts.shape)
        return vals[inverse.ravel()]

    return evaluate


def _subpoints(zlo, zhi, n_sub):
    # midpoints of n_sub equal cells in z of each box, (n, n_sub)
    return zlo[:, None] + (zhi - zlo)[:, None] * ((np.arange(n_sub) + 0.5) / n_sub)


def _subpoint_draw(zz, width, logw, gen, objects) -> np.ndarray:
    """One value per row by inverse-CDF sampling on the sub-points ``zz`` (cells of
    ``width / n_sub``) with log-weights ``logw``."""
    n, n_sub = logw.shape
    logw = np.where(np.isnan(logw), -np.inf, logw)  # e.g. a posterior over a zero prior
    empty = ~np.isfinite(logw).any(axis=1)
    if empty.any():  # it drew from the box's first sub-point before (audit O-11)
        raise ValueError(
            f"The likelihood is zero at all {n_sub} sub-points of a bin with posterior "
            f"mass (object {int(objects[empty][0])}); raise n_sub or check loglike."
        )
    cell = kern._sample_labels_dense(np.exp(logw - logw.max(axis=1, keepdims=True)), 1.0, gen)
    return zz[np.arange(n), cell] + width / n_sub * (gen.random(n) - 0.5)


def _base_logw(m, zz, labels, d, height):
    # flat in u (the Jacobian in z) times the hat's height along axis d
    with np.errstate(divide="ignore"):
        return m.log_jacobian(zz) + (
            0.0 if height is None else np.log(height(labels, d, m.forward(zz)))
        )


def _draw_subpoints(measures, n_sub, evaluate, objects, labels, lo, hi, gen, height):
    # each axis on n_sub sub-points: the population, times evaluate(objects, labels, zz)
    z = np.empty(lo.shape)
    for d, m in enumerate(measures):
        zlo, zhi = m.inverse(lo[:, d]), m.inverse(hi[:, d])
        zz = _subpoints(zlo, zhi, n_sub)
        logw = _base_logw(m, zz, labels, d, height)
        if evaluate is not None:
            logw = logw + evaluate(objects, labels, zz)
        z[:, d] = _subpoint_draw(zz, zhi - zlo, logw, gen, objects)
    return z, np.ones(len(z), dtype=bool)


def _truncnorm(a, b, gen) -> np.ndarray:
    """Standard normal draws truncated to ``[a, b]``, by the inverse CDF in log space
    (intervals above zero are reflected, so tails keep their precision)."""
    flip = a > 0
    lo, hi = np.where(flip, -b, a), np.where(flip, -a, b)
    la, lb = special.log_ndtr(lo), special.log_ndtr(hi)
    u = gen.random(np.shape(lo))
    with np.errstate(divide="ignore"):  # u = 0 at a lower bound of -inf
        x = special.ndtri_exp(lb + np.log(u + (1 - u) * np.exp(la - lb)))
    x = np.clip(x, lo, hi)
    return np.where(flip, -x, x)


def _draw_gauss(measures, n_sub, n_sweeps, meta, objects, labels, lo, hi, gen, height):
    """Gaussian likelihoods: exact truncated draws per axis on linear axes (the hat's
    height as an acceptance test), sub-points elsewhere; with a covariance, ``n_sweeps``
    Gibbs sweeps over the axes' conditionals from that start (sub-points inside hats)."""
    n, nd = lo.shape
    mu = np.asarray(meta["gaussian_mean"])[objects]
    P = None
    if "gaussian_cov" in meta:
        ids, inverse = np.unique(objects, return_inverse=True)  # one inverse per object
        C = np.asarray(meta["gaussian_cov"])[ids]
        P = np.linalg.inv(C)[inverse]
        sig = np.sqrt(np.diagonal(C, axis1=1, axis2=2))[inverse]
    else:
        sig = np.asarray(meta["gaussian_sigma"])[objects]
    zlo = np.stack([m.inverse(lo[:, d]) for d, m in enumerate(measures)], axis=1)
    zhi = np.stack([m.inverse(hi[:, d]) for d, m in enumerate(measures)], axis=1)
    keep = np.ones(n, dtype=bool)

    def axis(d, m, mu_d, sig_d, reject):
        if m == Measure.linear() and (height is None or reject):
            x = mu_d + sig_d * _truncnorm(
                (zlo[:, d] - mu_d) / sig_d, (zhi[:, d] - mu_d) / sig_d, gen
            )
            if height is not None:
                keep[:] &= gen.random(n) < height(labels, d, m.forward(x))
            return x
        zz = _subpoints(zlo[:, d], zhi[:, d], n_sub)
        logw = (
            _base_logw(m, zz, labels, d, height)
            - 0.5 * ((zz - mu_d[:, None]) / sig_d[:, None]) ** 2
        )
        return _subpoint_draw(zz, zhi[:, d] - zlo[:, d], logw, gen, objects)

    z = np.empty((n, nd))
    for d, m in enumerate(measures):
        z[:, d] = axis(d, m, mu[:, d], sig[:, d], reject=P is None)
    for _ in range(n_sweeps if P is not None else 0):
        for d, m in enumerate(measures):
            s2 = 1.0 / P[:, d, d]
            rest = np.einsum("nj,nj->n", P[:, d], z - mu) - P[:, d, d] * (z[:, d] - mu[:, d])
            z[:, d] = axis(d, m, mu[:, d] - s2 * rest, np.sqrt(s2), reject=False)
    return z, keep


def _draw_samples(
    measures, flat, direct, row, K, key, zs, G, objects, labels, lo, hi, gen, height
):
    """An object's kept samples inside the box, drawn with their weights (``G`` their
    cumulative sum, each row's summing to one; ``key`` = row * K + bin, sorted); in
    a box no sample reached, flat in ``u`` as the ``floor`` is."""
    n, nd = lo.shape
    z, keep = np.empty((n, nd)), np.ones(n, dtype=bool)
    r = row[objects]
    G0 = np.concatenate(([0.0], G))
    if direct:  # the label is the bin each sample was counted in: one segment each
        s0 = np.searchsorted(key, r * K + labels, side="left")
        s1 = np.searchsorted(key, r * K + labels, side="right")
        a, b = G0[s0], G0[s1]
        hit = b > a  # samples with weight in the bin
        u = a[hit] + (1.0 - gen.random(hit.sum())) * (b - a)[hit]
        z[hit] = zs[np.clip(np.searchsorted(G, u, side="left"), s0[hit], s1[hit] - 1)]
    else:  # components: the samples of the row inside the box, times the hat's height
        hit = np.zeros(n, dtype=bool)
        s0 = np.searchsorted(key, r * K, side="left")
        s1 = np.searchsorted(key, (r + 1) * K, side="left")
        order = np.lexsort((labels, objects))
        cut = np.flatnonzero(np.diff(objects[order]) | np.diff(labels[order])) + 1
        last = -1
        for at in np.split(order, cut):  # the draws of one object in one box
            i = at[0]
            seg = slice(s0[i], s1[i])
            if objects[i] != last:  # the object's samples in u, once for all its boxes
                us = np.stack([m.forward(zs[seg, d]) for d, m in enumerate(measures)], axis=1)
                last = objects[i]
            w = np.diff(G0[s0[i] : s1[i] + 1]) * np.all((us >= lo[i]) & (us <= hi[i]), axis=1)
            for d in range(nd if height is not None else 0):
                w = w * height(labels[i], d, us[:, d])
            c = np.cumsum(w)
            if c.size and c[-1] > 0:
                z[at] = zs[seg][np.searchsorted(c, (1.0 - gen.random(at.size)) * c[-1])]
                hit[at] = True
    miss = ~hit
    if miss.any():
        z[miss], keep[miss] = flat(objects[miss], labels[miss], lo[miss], hi[miss], gen, height)
    return z, keep
