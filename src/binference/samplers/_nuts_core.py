"""Core of the No-U-Turn Sampler (pure functions).

Multinomial NUTS with the generalised U-turn criterion and Stan's additional
checks across merged subtrees; diagonal or dense Euclidean metrics.
References: Hoffman & Gelman (2014); Betancourt (2017, arXiv:1701.02434).

The tree stores, at each end, the position, momentum, velocity
(``inverse metric @ momentum``) and gradient, so that the U-turn checks need no
extra metric products. Scalar arithmetic uses :mod:`math`, which is much
faster than NumPy scalars in this hot loop.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy import linalg

LogpGrad = Callable[[np.ndarray], tuple[float, np.ndarray]]

_NEG_INF = -math.inf


class Metric:
    """Euclidean metric: inverse mass matrix ``inv_m`` (diagonal vector or dense matrix)."""

    def __init__(self, inv_m: np.ndarray) -> None:
        self.inv_m = np.asarray(inv_m, dtype=float)
        self.dense = self.inv_m.ndim == 2
        if self.dense:
            # p ~ N(0, inv_m^{-1}) with inv_m = C C^T is p = C^{-T} z; C^{-T} once (audit N-9)
            chol = np.linalg.cholesky(self.inv_m)
            self._draw = linalg.solve_triangular(chol, np.eye(chol.shape[0]), lower=True).T
        else:
            self._draw = 1.0 / np.sqrt(self.inv_m)

    def mv(self, p: np.ndarray) -> np.ndarray:
        return self.inv_m @ p if self.dense else self.inv_m * p

    def kinetic(self, p: np.ndarray) -> float:
        return 0.5 * float(p @ self.mv(p))

    def draw_momentum(self, rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_normal(self.inv_m.shape[0])
        return self._draw @ z if self.dense else self._draw * z


def _logaddexp(a: float, b: float) -> float:
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class _Tree:
    __slots__ = (
        "diverging",
        "g_m",
        "g_p",
        "g_prop",
        "log_w",
        "lp_prop",
        "n_leaf",
        "p_m",
        "p_p",
        "p_prop",
        "rho",
        "sum_acc",
        "turning",
        "v_m",
        "v_p",
        "x_m",
        "x_p",
        "x_prop",
    )


def leapfrog(x, p, g, eps, metric: Metric, fn: LogpGrad):
    p_half = p + (0.5 * eps) * g
    x_new = x + eps * metric.mv(p_half)
    lp_new, g_new = fn(x_new)
    p_new = p_half + (0.5 * eps) * g_new
    return x_new, p_new, g_new, lp_new


def _no_uturn(va, vb, rho) -> bool:
    """Generalised criterion with velocities ``va``, ``vb`` at the two ends."""
    return float(va @ rho) > 0.0 and float(vb @ rho) > 0.0


def _accept(rng, log_ratio: float) -> bool:
    # u < exp(log_ratio) with u in (0, 1]: log1p(-u) never takes log(0) (audit N-8)
    return math.log1p(-rng.random()) < log_ratio


def _merge(t1: _Tree, t2: _Tree, v: int, rng, old_log_w: float) -> bool:
    """Append ``t2`` to ``t1`` in direction ``v``; returns whether the merged tree is valid.

    ``t2``'s proposal replaces ``t1``'s with probability ``exp(t2.log_w - old_log_w)``:
    multinomial within subtrees (``old_log_w`` is the merged weight) and biased
    progressive at the top level (the old tree's weight). Then Stan's U-turn checks
    across the merged tree and the two extra sub-checks.
    """
    if _accept(rng, t2.log_w - old_log_w):
        t1.x_prop, t1.p_prop, t1.lp_prop, t1.g_prop = t2.x_prop, t2.p_prop, t2.lp_prop, t2.g_prop
    t1.log_w = _logaddexp(t1.log_w, t2.log_w)
    rho = t1.rho + t2.rho
    if v > 0:
        p_ie, p_fb = t1.p_p, t2.p_m
        v_ib, v_ie, v_fb, v_fe = t1.v_m, t1.v_p, t2.v_m, t2.v_p
    else:
        p_ie, p_fb = t1.p_m, t2.p_p
        v_ib, v_ie, v_fb, v_fe = t1.v_p, t1.v_m, t2.v_p, t2.v_m
    ok = (
        _no_uturn(v_ib, v_fe, rho)
        and _no_uturn(v_ib, v_fb, t1.rho + p_fb)
        and _no_uturn(v_ie, v_fe, t2.rho + p_ie)
    )
    t1.rho = rho
    if v > 0:
        t1.x_p, t1.p_p, t1.v_p, t1.g_p = t2.x_p, t2.p_p, t2.v_p, t2.g_p
    else:
        t1.x_m, t1.p_m, t1.v_m, t1.g_m = t2.x_m, t2.p_m, t2.v_m, t2.g_m
    return ok


def _leaf(x, p, g, v, eps, H0, metric, fn, max_dE):
    x1, p1, g1, lp1 = leapfrog(x, p, g, v * eps, metric, fn)
    vel = metric.mv(p1)
    H1 = float(lp1) - 0.5 * float(p1 @ vel)  # NaN in g1 propagates into p1 and H1
    finite = math.isfinite(H1)
    t = _Tree()
    t.x_m = t.x_p = t.x_prop = x1
    t.p_m = t.p_p = t.p_prop = p1
    t.v_m = t.v_p = vel
    t.g_m = t.g_p = t.g_prop = g1
    t.lp_prop = lp1
    t.rho = p1
    t.turning = False
    t.n_leaf = 1
    if finite:
        dH = H1 - H0
        t.log_w = dH
        t.diverging = -dH > max_dE
        t.sum_acc = math.exp(dH) if dH < 0.0 else 1.0
    else:
        t.log_w = _NEG_INF
        t.diverging = True
        t.sum_acc = 0.0
    return t


def _build(x, p, g, v, depth, eps, H0, metric, fn, rng, max_dE):
    if depth == 0:
        return _leaf(x, p, g, v, eps, H0, metric, fn, max_dE)
    t1 = _build(x, p, g, v, depth - 1, eps, H0, metric, fn, rng, max_dE)
    if t1.turning or t1.diverging:
        return t1
    if v > 0:
        t2 = _build(t1.x_p, t1.p_p, t1.g_p, v, depth - 1, eps, H0, metric, fn, rng, max_dE)
    else:
        t2 = _build(t1.x_m, t1.p_m, t1.g_m, v, depth - 1, eps, H0, metric, fn, rng, max_dE)
    t1.sum_acc += t2.sum_acc
    t1.n_leaf += t2.n_leaf
    if t2.turning or t2.diverging:
        t1.turning = t1.turning or t2.turning
        t1.diverging = t1.diverging or t2.diverging
        return t1
    t1.turning = not _merge(t1, t2, v, rng, _logaddexp(t1.log_w, t2.log_w))
    return t1


def nuts_step(x, lp, g, eps, metric: Metric, fn: LogpGrad, rng, max_depth=10, max_dE=1000.0):
    """One NUTS transition.

    Returns ``(x, lp, g, accept_stat, depth, diverging, n_leapfrog, energy)``.
    """
    p0 = metric.draw_momentum(rng)
    v0 = metric.mv(p0)
    H0 = float(lp) - 0.5 * float(p0 @ v0)
    tree = _Tree()
    tree.x_m = tree.x_p = x
    tree.p_m = tree.p_p = p0
    tree.v_m = tree.v_p = v0
    tree.g_m = tree.g_p = g
    tree.log_w = 0.0
    tree.rho = p0
    tree.x_prop, tree.lp_prop, tree.g_prop, tree.p_prop = x, lp, g, p0
    sum_acc, n_leaf = 0.0, 0
    diverging = False
    n_doublings = 0
    for depth in range(max_depth):
        v = 1 if rng.random() < 0.5 else -1
        if v > 0:
            t = _build(tree.x_p, tree.p_p, tree.g_p, v, depth, eps, H0, metric, fn, rng, max_dE)
        else:
            t = _build(tree.x_m, tree.p_m, tree.g_m, v, depth, eps, H0, metric, fn, rng, max_dE)
        sum_acc += t.sum_acc
        n_leaf += t.n_leaf
        if t.diverging:
            diverging = True
            break
        if t.turning:
            break
        ok = _merge(tree, t, v, rng, tree.log_w)
        n_doublings = depth + 1
        if not ok:
            break
    accept = sum_acc / max(n_leaf, 1)
    energy = -float(tree.lp_prop) + metric.kinetic(tree.p_prop)
    return tree.x_prop, tree.lp_prop, tree.g_prop, accept, n_doublings, diverging, n_leaf, energy


def find_reasonable_eps(x, lp, g, metric: Metric, fn: LogpGrad, rng) -> float:
    """Heuristic initial step size (Hoffman & Gelman 2014, Algorithm 4).

    Starts from 1 with one momentum draw and an acceptance threshold of 0.5,
    unlike Stan's ``init_stepsize`` (current step size, threshold 0.8).
    """
    eps = 1.0
    p = metric.draw_momentum(rng)
    H0 = lp - metric.kinetic(p)

    def dH(e):
        _, p1, _, lp1 = leapfrog(x, p, g, e, metric, fn)
        if not np.isfinite(lp1):
            return -np.inf
        with np.errstate(over="ignore", invalid="ignore"):  # trial steps may blow up
            h = lp1 - metric.kinetic(p1) - H0
        return h if math.isfinite(h) else -np.inf

    h = dH(eps)
    a = 1.0 if h > np.log(0.5) else -1.0
    for _ in range(100):
        if not (a * h > -a * np.log(2.0)):
            break
        eps *= 2.0**a
        h = dH(eps)
    return float(eps)


def adaptation_windows(
    n_warmup: int, init_buffer: int = 75, term_buffer: int = 50, base_window: int = 25
) -> list[tuple[int, int]]:
    """Stan-style slow adaptation windows ``[(start, end), ...]``.

    As Stan's, except that the first window also extends to the end of the slow
    phase when the next would not fit, where Stan adds a short last window (49
    warm-up lengths, all from 151 to 199 iterations, differ). None below 20.
    """
    if n_warmup < 20:
        return []
    if init_buffer + base_window + term_buffer > n_warmup:
        init_buffer = int(0.15 * n_warmup)
        term_buffer = int(0.1 * n_warmup)
        base_window = n_warmup - init_buffer - term_buffer
    wins = []
    start, size = init_buffer, base_window
    end_slow = n_warmup - term_buffer
    while start < end_slow:
        end = start + size
        if end + 2 * size > end_slow:
            end = end_slow
        wins.append((start, end))
        start = end
        size *= 2
    return wins
