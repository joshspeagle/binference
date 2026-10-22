"""Core of the No-U-Turn Sampler (pure functions).

Multinomial NUTS with the generalised U-turn criterion and Stan's additional
checks across merged subtrees; diagonal or dense Euclidean metrics.
References: Hoffman & Gelman (2014); Betancourt (2017, arXiv:1701.02434).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

LogpGrad = Callable[[np.ndarray], tuple[float, np.ndarray]]


class Metric:
    """Euclidean metric: inverse mass matrix ``inv_m`` (diagonal vector or dense matrix)."""

    def __init__(self, inv_m: np.ndarray) -> None:
        self.inv_m = np.asarray(inv_m, dtype=float)
        self.dense = self.inv_m.ndim == 2
        self.chol = np.linalg.cholesky(self.inv_m) if self.dense else None

    def mv(self, p: np.ndarray) -> np.ndarray:
        return self.inv_m @ p if self.dense else self.inv_m * p

    def kinetic(self, p: np.ndarray) -> float:
        return 0.5 * float(p @ self.mv(p))

    def draw_momentum(self, rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_normal(self.inv_m.shape[0])
        if self.dense:
            # p ~ N(0, inv_m^{-1}) with inv_m = C C^T  =>  p = C^{-T} z
            return np.linalg.solve(self.chol.T, z)
        return z / np.sqrt(self.inv_m)


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
        "x_m",
        "x_p",
        "x_prop",
    )


def leapfrog(x, p, g, eps, metric: Metric, fn: LogpGrad):
    p_half = p + 0.5 * eps * g
    x_new = x + eps * metric.mv(p_half)
    lp_new, g_new = fn(x_new)
    p_new = p_half + 0.5 * eps * g_new
    return x_new, p_new, g_new, lp_new


def _crit(pa, pb, rho, metric: Metric) -> bool:
    return (float(metric.mv(pa) @ rho) > 0.0) and (float(metric.mv(pb) @ rho) > 0.0)


def _merge_ok(rho1, rho2, init_beg, init_end, final_beg, final_end, metric) -> bool:
    return (
        _crit(init_beg, final_end, rho1 + rho2, metric)
        and _crit(init_beg, final_beg, rho1 + final_beg, metric)
        and _crit(init_end, final_end, rho2 + init_end, metric)
    )


def _ends(t: _Tree, v: int):
    return (t.p_m, t.p_p) if v > 0 else (t.p_p, t.p_m)


def _leaf(x, p, g, v, eps, H0, metric, fn, max_dE):
    x1, p1, g1, lp1 = leapfrog(x, p, g, v * eps, metric, fn)
    if np.isfinite(lp1) and np.all(np.isfinite(g1)) and np.all(np.isfinite(p1)):
        H1 = lp1 - metric.kinetic(p1)
    else:
        H1 = -np.inf
    t = _Tree()
    t.x_m = t.x_p = t.x_prop = x1
    t.p_m = t.p_p = t.p_prop = p1
    t.g_m = t.g_p = t.g_prop = g1
    t.lp_prop = lp1
    finite = bool(np.isfinite(H1))
    t.log_w = (H1 - H0) if finite else -np.inf
    t.rho = p1.copy()
    t.turning = False
    t.diverging = (not finite) or (H0 - H1 > max_dE)
    t.sum_acc = float(np.exp(min(0.0, H1 - H0))) if finite else 0.0
    t.n_leaf = 1
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
    log_w = np.logaddexp(t1.log_w, t2.log_w)
    if np.log(rng.random()) < t2.log_w - log_w:
        t1.x_prop, t1.p_prop, t1.lp_prop, t1.g_prop = t2.x_prop, t2.p_prop, t2.lp_prop, t2.g_prop
    t1.log_w = log_w
    ib, ie = _ends(t1, v)
    fb, fe = _ends(t2, v)
    ok = _merge_ok(t1.rho, t2.rho, ib, ie, fb, fe, metric)
    t1.rho = t1.rho + t2.rho
    if v > 0:
        t1.x_p, t1.p_p, t1.g_p = t2.x_p, t2.p_p, t2.g_p
    else:
        t1.x_m, t1.p_m, t1.g_m = t2.x_m, t2.p_m, t2.g_m
    t1.turning = not ok
    return t1


def nuts_step(x, lp, g, eps, metric: Metric, fn: LogpGrad, rng, max_depth=10, max_dE=1000.0):
    """One NUTS transition.

    Returns ``(x, lp, g, accept_stat, depth, diverging, n_leapfrog, energy)``.
    """
    p0 = metric.draw_momentum(rng)
    H0 = lp - metric.kinetic(p0)
    tree = _Tree()
    tree.x_m = tree.x_p = x
    tree.p_m = tree.p_p = p0
    tree.g_m = tree.g_p = g
    tree.log_w = 0.0
    tree.rho = p0.copy()
    x_new, lp_new, g_new, p_new = x, lp, g, p0
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
        if np.log(rng.random()) < t.log_w - tree.log_w:
            x_new, lp_new, g_new, p_new = t.x_prop, t.lp_prop, t.g_prop, t.p_prop
        tree.log_w = np.logaddexp(tree.log_w, t.log_w)
        ib, ie = _ends(tree, v)
        fb, fe = _ends(t, v)
        ok = _merge_ok(tree.rho, t.rho, ib, ie, fb, fe, metric)
        tree.rho = tree.rho + t.rho
        if v > 0:
            tree.x_p, tree.p_p, tree.g_p = t.x_p, t.p_p, t.g_p
        else:
            tree.x_m, tree.p_m, tree.g_m = t.x_m, t.p_m, t.g_m
        n_doublings = depth + 1
        if not ok:
            break
    accept = sum_acc / max(n_leaf, 1)
    energy = -lp_new + metric.kinetic(p_new)
    return x_new, lp_new, g_new, accept, n_doublings, diverging, n_leaf, energy


def find_reasonable_eps(x, lp, g, metric: Metric, fn: LogpGrad, rng) -> float:
    """Heuristic initial step size (Hoffman & Gelman 2014, Algorithm 4)."""
    eps = 1.0
    p = metric.draw_momentum(rng)
    H0 = lp - metric.kinetic(p)

    def dH(e):
        _, p1, _, lp1 = leapfrog(x, p, g, e, metric, fn)
        if not np.isfinite(lp1):
            return -np.inf
        return lp1 - metric.kinetic(p1) - H0

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
    """Stan-style slow adaptation windows ``[(start, end), ...]``."""
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
