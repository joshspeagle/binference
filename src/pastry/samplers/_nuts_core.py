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

LogpGrad = Callable[[np.ndarray], tuple[float, np.ndarray]]

_NEG_INF = -math.inf


class Metric:
    """Euclidean metric: inverse mass matrix ``inv_m`` (diagonal vector or dense matrix)."""

    def __init__(self, inv_m: np.ndarray) -> None:
        self.inv_m = np.asarray(inv_m, dtype=float)
        self.dense = self.inv_m.ndim == 2
        if self.dense:
            self.chol = np.linalg.cholesky(self.inv_m)
            self._sqrt = None
        else:
            self.chol = None
            self._sqrt = np.sqrt(self.inv_m)

    def mv(self, p: np.ndarray) -> np.ndarray:
        return self.inv_m @ p if self.dense else self.inv_m * p

    def kinetic(self, p: np.ndarray) -> float:
        return 0.5 * float(p @ self.mv(p))

    def draw_momentum(self, rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_normal(self.inv_m.shape[0])
        if self.dense:
            # p ~ N(0, inv_m^{-1}) with inv_m = C C^T  =>  p = C^{-T} z
            return np.linalg.solve(self.chol.T, z)
        return z / self._sqrt


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


def _merge_ok(t1: _Tree, t2: _Tree, v: int) -> bool:
    """Stan's U-turn checks across the merged tree and the two extra sub-checks."""
    if v > 0:
        p_ie, p_fb = t1.p_p, t2.p_m
        v_ib, v_ie, v_fb, v_fe = t1.v_m, t1.v_p, t2.v_m, t2.v_p
    else:
        p_ie, p_fb = t1.p_m, t2.p_p
        v_ib, v_ie, v_fb, v_fe = t1.v_p, t1.v_m, t2.v_p, t2.v_m
    return (
        _no_uturn(v_ib, v_fe, t1.rho + t2.rho)
        and _no_uturn(v_ib, v_fb, t1.rho + p_fb)
        and _no_uturn(v_ie, v_fe, t2.rho + p_ie)
    )


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
    log_w = _logaddexp(t1.log_w, t2.log_w)
    if math.log(rng.random()) < t2.log_w - log_w:
        t1.x_prop, t1.p_prop, t1.lp_prop, t1.g_prop = t2.x_prop, t2.p_prop, t2.lp_prop, t2.g_prop
    t1.log_w = log_w
    ok = _merge_ok(t1, t2, v)
    t1.rho = t1.rho + t2.rho
    if v > 0:
        t1.x_p, t1.p_p, t1.v_p, t1.g_p = t2.x_p, t2.p_p, t2.v_p, t2.g_p
    else:
        t1.x_m, t1.p_m, t1.v_m, t1.g_m = t2.x_m, t2.p_m, t2.v_m, t2.g_m
    t1.turning = not ok
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
        if math.log(rng.random()) < t.log_w - tree.log_w:
            x_new, lp_new, g_new, p_new = t.x_prop, t.lp_prop, t.g_prop, t.p_prop
        tree.log_w = _logaddexp(tree.log_w, t.log_w)
        ok = _merge_ok(tree, t, v)
        tree.rho = tree.rho + t.rho
        if v > 0:
            tree.x_p, tree.p_p, tree.v_p, tree.g_p = t.x_p, t.p_p, t.v_p, t.g_p
        else:
            tree.x_m, tree.p_m, tree.v_m, tree.g_m = t.x_m, t.p_m, t.v_m, t.g_m
        n_doublings = depth + 1
        if not ok:
            break
    accept = sum_acc / max(n_leaf, 1)
    energy = -float(lp_new) + metric.kinetic(p_new)
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


def fd_hessian(fn: LogpGrad, x: np.ndarray, h: float = 1e-4, cols=None) -> np.ndarray:
    """Symmetrised central-difference Hessian of a log density from its gradient.

    ``cols`` restricts the computation to some columns (returns ``(dim, len(cols))``).
    """
    x = np.asarray(x, dtype=float)
    d = x.size
    idx = range(d) if cols is None else cols
    H = np.empty((d, len(idx)))
    for c, j in enumerate(idx):
        step = h * max(1.0, abs(x[j]))
        e = np.zeros(d)
        e[j] = step
        H[:, c] = (fn(x + e)[1] - fn(x - e)[1]) / (2.0 * step)
    if cols is None:
        H = 0.5 * (H + H.T)
    return H


def covariance_from_hessian(H: np.ndarray, floor: float = 1e-6) -> np.ndarray | None:
    """A positive-definite covariance from a (possibly indefinite) Hessian of log p.

    Eigenvalues of ``-H`` are clipped below at ``floor`` times the largest, so
    directions with no or negative curvature get the largest variance seen.
    Returns ``None`` if the Hessian is unusable.
    """
    A = -0.5 * (H + H.T)
    if not np.all(np.isfinite(A)):
        return None
    lam, U = np.linalg.eigh(A)
    top = lam.max()
    if not top > 0:
        return None
    lam = np.maximum(lam, floor * top)
    cov = (U / lam) @ U.T
    return 0.5 * (cov + cov.T)
