"""The No-U-Turn Sampler engine (default exact engine)."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np

from .._utils import RNGLike, progress, spawn
from ..model import PopulationModel
from ..results import PopulationResult
from ..warnings import ConvergenceWarning
from ._nuts_core import Metric, adaptation_windows, find_reasonable_eps, nuts_step
from .base import Engine

__all__ = ["NUTS"]


class _Chain:
    """State of one NUTS chain, including its step-size and metric adaptation."""

    def __init__(self, x0, fn, rng, n_warmup, target_accept, dense):
        self.fn = fn
        self.rng = rng
        self.x = np.asarray(x0, dtype=float)
        self.lp, self.g = fn(self.x)
        d = self.x.size
        self.metric = Metric(np.eye(d) if dense else np.ones(d))
        self.dense = dense
        self.eps = find_reasonable_eps(self.x, self.lp, self.g, self.metric, fn, rng)
        self.target = target_accept
        self._restart_dual_averaging()
        self.windows = adaptation_windows(n_warmup)
        self.win_ends = {e for _, e in self.windows}
        self.buf: list[np.ndarray] = []
        self.n_warmup = n_warmup
        self.it = 0

    def _restart_dual_averaging(self):
        self.mu = np.log(10.0 * self.eps)
        self.Hbar = 0.0
        self.log_eps_bar = 0.0
        self.m = 0

    def step(self, max_depth, adapt):
        x, lp, g, acc, depth, div, nl, energy = nuts_step(
            self.x, self.lp, self.g, self.eps, self.metric, self.fn, self.rng, max_depth
        )
        self.x, self.lp, self.g = x, lp, g
        if adapt:
            self._adapt(acc)
        self.it += 1
        return acc, depth, div, nl, energy

    def _adapt(self, acc):
        t0, gamma, kappa = 10.0, 0.05, 0.75
        self.m += 1
        m = self.m
        self.Hbar = (1.0 - 1.0 / (m + t0)) * self.Hbar + (self.target - acc) / (m + t0)
        log_eps = self.mu - np.sqrt(m) / gamma * self.Hbar
        eta = m ** (-kappa)
        self.log_eps_bar = eta * log_eps + (1.0 - eta) * self.log_eps_bar
        self.eps = float(np.exp(log_eps))
        it = self.it
        if any(s <= it < e for s, e in self.windows):
            self.buf.append(self.x.copy())
        if (it + 1) in self.win_ends:
            arr = np.asarray(self.buf)
            n = len(arr)
            d = self.x.size
            if self.dense:
                cov = np.cov(arr.T) if n > 1 else np.eye(d)
                cov = np.atleast_2d(cov)
                inv_m = (n / (n + 5.0)) * cov + 1e-3 * (5.0 / (n + 5.0)) * np.eye(d)
            else:
                var = arr.var(axis=0, ddof=1) if n > 1 else np.ones(d)
                inv_m = (n / (n + 5.0)) * var + 1e-3 * (5.0 / (n + 5.0))
            self.metric = Metric(inv_m)
            self.buf = []
            self.eps = find_reasonable_eps(self.x, self.lp, self.g, self.metric, self.fn, self.rng)
            self._restart_dual_averaging()
        if it == self.n_warmup - 1 and self.m > 0:
            self.eps = float(np.exp(self.log_eps_bar))


class NUTS(Engine):
    """No-U-Turn Sampler on the exact posterior (the default engine).

    Samples the unconstrained coordinates of the model (isometric log-ratios
    for simplex priors, whitened fields and log-hyperparameters for smooth
    priors, log-rates for rates) with multinomial NUTS, dual-averaging step
    size adaptation and Stan-style windowed metric adaptation. Chains run in
    lock-step, so the engine supports progress reporting, a generator
    interface (:meth:`sample`) and resuming (call :meth:`run` again to draw
    more samples without re-running warm-up).

    Parameters
    ----------
    model : PopulationModel
    n_chains : int
        Number of independent chains (default 4; needed for R-hat).
    n_warmup : int
        Warm-up (adaptation) iterations per chain.
    target_accept : float
        Target mean acceptance statistic for step-size adaptation.
    max_depth : int
        Maximum tree depth (at most ``2**max_depth`` gradient evaluations per draw).
    metric : {"diag", "dense"}
        Adapted Euclidean metric.
    init : array_like (dim,) or (n_chains, dim), optional
        Starting points in unconstrained coordinates. By default chains start
        at the posterior mode (in unconstrained coordinates) plus ``jitter``.
    jitter : float
        Standard deviation of the Gaussian jitter around the mode.
    rng : int, Generator or None
        Random number generator (SPEC 7); each chain gets an independent child.
    progress : bool
        Show a progress bar.
    """

    name = "nuts"

    def __init__(
        self,
        model: PopulationModel,
        *,
        n_chains: int = 4,
        n_warmup: int = 1000,
        target_accept: float = 0.8,
        max_depth: int = 10,
        metric: str = "diag",
        init: Any = None,
        jitter: float = 0.1,
        rng: RNGLike = None,
        progress: bool = False,
    ) -> None:
        super().__init__(model, rng=rng, progress=progress)
        if n_chains < 1:
            raise ValueError("n_chains must be at least 1.")
        if metric not in ("diag", "dense"):
            raise ValueError("metric must be 'diag' or 'dense'.")
        if not 0 < target_accept < 1:
            raise ValueError("target_accept must be in (0, 1).")
        self.n_chains = int(n_chains)
        self.n_warmup = int(n_warmup)
        self.target_accept = float(target_accept)
        self.max_depth = int(max_depth)
        self.metric = metric
        self.init = init
        self.jitter = float(jitter)
        self._chains: list[_Chain] | None = None
        self._warm = False
        self._x: list[np.ndarray] = []
        self._lp: list[np.ndarray] = []
        self._stats: dict[str, list[np.ndarray]] = {
            k: [] for k in ("accept_stat", "tree_depth", "diverging", "n_leapfrog", "energy")
        }
        self.warmup_divergences = 0

    # ---------------------------------------------------------- set-up
    def _initial_points(self) -> np.ndarray:
        d = self.model.dim
        child = spawn(self.rng, 1)[0]
        if self.init is not None:
            x0 = np.asarray(self.init, dtype=float)
            if x0.shape == (d,):
                x0 = np.tile(x0, (self.n_chains, 1))
                x0 = x0 + self.jitter * child.standard_normal(x0.shape)
            if x0.shape != (self.n_chains, d):
                raise ValueError(f"init must have shape ({d},) or ({self.n_chains}, {d}).")
            return x0
        from .optimize import MAP

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            mode = MAP(self.model, space="unconstrained").run()
        x_mode = mode.x if mode.x is not None else self.model.from_value(mode.value)
        pts = []
        for _ in range(self.n_chains):
            for scale in (self.jitter, self.jitter / 10, 0.0):
                x = x_mode + scale * child.standard_normal(d)
                if np.isfinite(self.model.log_prob(x)):
                    break
            pts.append(x)
        return np.asarray(pts)

    def _setup(self) -> None:
        if self._chains is not None:
            return
        x0 = self._initial_points()
        rngs = spawn(self.rng, self.n_chains)
        fn = self.model.log_prob_and_grad
        self._chains = [
            _Chain(x0[c], fn, rngs[c], self.n_warmup, self.target_accept, self.metric == "dense")
            for c in range(self.n_chains)
        ]

    # ---------------------------------------------------------- running
    def warmup(self) -> None:
        """Run warm-up (adaptation) if it has not been run yet."""
        self._setup()
        if self._warm:
            return
        assert self._chains is not None
        for _ in progress(range(self.n_warmup), self.n_warmup, "NUTS warm-up", self.progress):
            for ch in self._chains:
                _, _, div, _, _ = ch.step(self.max_depth, adapt=True)
                self.warmup_divergences += int(div)
        self._warm = True

    def sample(self, n_samples: int) -> Iterator[np.ndarray]:
        """Generator over post-warm-up iterations, yielding ``(n_chains, dim)`` positions.

        Draws are also recorded and included in the next :meth:`run` /
        :attr:`result`.
        """
        self.warmup()
        assert self._chains is not None
        C, d = self.n_chains, self.model.dim
        xs = np.empty((C, n_samples, d))
        lps = np.empty((C, n_samples))
        st = {k: np.empty((C, n_samples)) for k in self._stats}
        n_done = 0
        try:
            for i in progress(range(n_samples), n_samples, "NUTS sampling", self.progress):
                for c, ch in enumerate(self._chains):
                    acc, depth, div, nl, energy = ch.step(self.max_depth, adapt=False)
                    xs[c, i] = ch.x
                    lps[c, i] = ch.lp
                    st["accept_stat"][c, i] = acc
                    st["tree_depth"][c, i] = depth
                    st["diverging"][c, i] = div
                    st["n_leapfrog"][c, i] = nl
                    st["energy"][c, i] = energy
                n_done = i + 1
                yield xs[:, i]
        finally:
            if n_done:
                self._x.append(xs[:, :n_done])
                self._lp.append(lps[:, :n_done])
                for k in st:
                    self._stats[k].append(st[k][:, :n_done])

    def run(self, n_samples: int = 1000) -> PopulationResult:
        """Warm up (first call only), draw ``n_samples`` per chain, and return all draws so far."""
        for _ in self.sample(int(n_samples)):
            pass
        return self.result

    @property
    def result(self) -> PopulationResult:
        """A :class:`~pastry.PopulationResult` with every draw recorded so far."""
        if not self._x:
            raise RuntimeError("No draws yet; call run().")
        X = np.concatenate(self._x, axis=1)
        LP = np.concatenate(self._lp, axis=1)
        C, n, d = X.shape
        vals = self.model.transform(X.reshape(-1, d)).reshape(C, n, -1)
        stats = {k: np.concatenate(v, axis=1) for k, v in self._stats.items()}
        info: dict[str, Any] = dict(stats)
        info["n_divergent"] = int(stats["diverging"].sum())
        info["warmup_divergences"] = self.warmup_divergences
        info["max_depth_fraction"] = float(np.mean(stats["tree_depth"] >= self.max_depth))
        info["step_size"] = np.array([ch.eps for ch in self._chains])
        info["inv_metric"] = [ch.metric.inv_m for ch in self._chains]
        E = stats["energy"]
        with np.errstate(invalid="ignore", divide="ignore"):
            ebfmi = np.sum(np.diff(E, axis=1) ** 2, axis=1) / np.sum(
                (E - E.mean(axis=1, keepdims=True)) ** 2, axis=1
            )
        info["ebfmi"] = ebfmi
        hyper = getattr(self.model.prior, "hyper_names", [])
        if hyper:
            u = X[..., self.model.prior.n_active :]
            info["hyperparameters"] = {name: np.exp(u[..., j]) for j, name in enumerate(hyper)}
        return PopulationResult(
            self.model, vals, method="nuts", kind=self.model.kind, x=X, log_prob=LP, info=info
        )
