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
from ._nuts_core import (
    Metric,
    adaptation_windows,
    covariance_from_hessian,
    fd_hessian,
    find_reasonable_eps,
    nuts_step,
)
from .base import Engine

__all__ = ["NUTS"]

# Dimension limits for the warm-up geometry adaptation (see NUTS docstring).
_DENSE_MAX_DIM = 200
_HESSIAN_MAX_DIM = 500


class _Chain:
    """State of one NUTS chain, including its step-size and metric adaptation."""

    def __init__(self, x0, fn, rng, n_warmup, target_accept, dense):
        self.fn = fn
        self.rng = rng
        self.x = np.asarray(x0, dtype=float)
        self.lp, self.g = fn(self.x)
        d = self.x.size
        self.metric = Metric(np.eye(d) if dense else np.ones(d))
        self.informed = False  # metric initialised from curvature rather than the identity
        self.dense = dense
        self.eps = find_reasonable_eps(self.x, self.lp, self.g, self.metric, fn, rng)
        self.target = target_accept
        self._restart_dual_averaging()
        self.windows = adaptation_windows(n_warmup)
        self.win_ends = {e for _, e in self.windows}
        self.buf: list[np.ndarray] = []
        self.n_warmup = n_warmup
        self.it = 0

    def reset_geometry(self, fn, x, inv_m) -> None:
        """Move to a new parametrisation and/or metric mid warm-up."""
        self.fn = fn
        self.x = np.asarray(x, dtype=float)
        self.lp, self.g = fn(self.x)
        if inv_m is not None:
            self.metric = Metric(inv_m)
            self.informed = True
        self.buf = []
        self.eps = find_reasonable_eps(self.x, self.lp, self.g, self.metric, self.fn, self.rng)
        self._restart_dual_averaging()

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
            prev = self.metric.inv_m
            if self.dense:
                cov = np.atleast_2d(np.cov(arr.T)) if n > 1 else prev
                if self.informed:  # shrink toward the curvature-based metric
                    k = d + 5.0
                    inv_m = (n / (n + k)) * cov + (k / (n + k)) * prev
                else:  # Stan's regularisation
                    inv_m = (n / (n + 5.0)) * cov + 1e-3 * (5.0 / (n + 5.0)) * np.eye(d)
            else:
                var = arr.var(axis=0, ddof=1) if n > 1 else prev
                if self.informed:
                    inv_m = (n / (n + 5.0)) * var + (5.0 / (n + 5.0)) * prev
                else:
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
    target_accept : float, optional
        Target mean acceptance statistic for step-size adaptation. Defaults to
        0.8, or 0.9 for logistic (smoothness) priors, whose hyperparameters
        make the geometry harder.
    max_depth : int
        Maximum tree depth (at most ``2**max_depth`` gradient evaluations per draw).
    metric : {"auto", "diag", "dense"}
        Adapted Euclidean metric. ``"auto"`` uses a dense metric for logistic
        (smoothness) priors with at most 200 unconstrained dimensions, whose
        hyperparameters correlate with the field, and a diagonal one otherwise.
    reparametrize : bool
        For logistic priors, switch each basis coefficient between the
        non-centred and centred forms according to how strongly the data
        constrain it (partial non-centring with weights ``1 / -H_jj`` from the
        curvature at the chains' positions early in warm-up). This changes
        the sampling coordinates, not the posterior.
    init : array_like (dim,) or (n_chains, dim), optional
        Starting points in unconstrained coordinates. By default chains start
        at the posterior mode (in unconstrained coordinates) plus ``jitter``.
    jitter : float
        Standard deviation of the Gaussian jitter around the mode.
    rng : int, Generator or None
        Random number generator (SPEC 7); each chain gets an independent child.
    progress : bool
        Show a progress bar.

    Notes
    -----
    At the start of the first slow adaptation window (iteration 75 of the
    default warm-up) the engine optionally reparametrises logistic priors
    (``reparametrize``) and, for at most 500 dimensions, initialises the metric
    from the finite-difference curvature of the log posterior at the chains'
    positions. Later windows refine the metric from the warm-up draws,
    shrinking toward that curvature-based starting point.
    """

    name = "nuts"

    def __init__(
        self,
        model: PopulationModel,
        *,
        n_chains: int = 4,
        n_warmup: int = 1000,
        target_accept: float | None = None,
        max_depth: int = 10,
        metric: str = "auto",
        reparametrize: bool = True,
        init: Any = None,
        jitter: float = 0.1,
        rng: RNGLike = None,
        progress: bool = False,
    ) -> None:
        super().__init__(model, rng=rng, progress=progress)
        if n_chains < 1:
            raise ValueError("n_chains must be at least 1.")
        if metric not in ("auto", "diag", "dense"):
            raise ValueError("metric must be 'auto', 'diag' or 'dense'.")
        logistic = hasattr(model.prior, "with_centering")
        if target_accept is None:
            target_accept = 0.9 if logistic else 0.8
        if not 0 < target_accept < 1:
            raise ValueError("target_accept must be in (0, 1).")
        self.n_chains = int(n_chains)
        self.n_warmup = int(n_warmup)
        self.target_accept = float(target_accept)
        self.max_depth = int(max_depth)
        if metric == "auto":
            metric = "dense" if logistic and model.dim <= _DENSE_MAX_DIM else "diag"
        self.metric = metric
        self.reparametrize = bool(reparametrize)
        self.init = init
        self.jitter = float(jitter)
        self._chains: list[_Chain] | None = None
        self._base_model = model
        self._geometry_iter = 25
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

    def _adapt_geometry(self) -> None:
        """Reparametrise a logistic prior and initialise the metric from curvature.

        Only for priors with a ``with_centering`` method (the logistic family):
        for simplex priors the curvature at single points is a poor guide to
        the posterior scales (sparse bins), so they keep Stan's adaptation.
        """
        assert self._chains is not None
        base = self._base_model
        prior0 = base.prior
        if not hasattr(prior0, "with_centering"):
            return
        chains = self._chains
        d = base.dim
        cur = self.model.prior
        ones = np.ones(prior0.n_active)
        # positions in the base (non-centred) parametrisation
        pts = [cur.recentre(ch.x, ones) for ch in chains]
        # curvature at every chain's position for small problems, one chain otherwise
        n_pts = len(chains) if d <= _DENSE_MAX_DIM else 1
        model = base
        if self.reparametrize and prior0.n_active > 0:
            fn0 = base.log_prob_and_grad
            n_act = prior0.n_active
            diag = np.mean(
                [np.diag(fd_hessian(fn0, x, cols=range(n_act))) for x in pts[:n_pts]], axis=0
            )
            lam = np.clip(1.0 / np.maximum(-diag, 1.0), 0.0, 1.0)
            model = base._with_bound_prior(prior0.with_centering(lam))
            pts = [prior0.recentre(x, lam) for x in pts]
        self.model = model
        fn = model.log_prob_and_grad
        inv_m = None
        if d <= _HESSIAN_MAX_DIM:
            H = np.mean([fd_hessian(fn, x) for x in pts[:n_pts]], axis=0)
            cov = covariance_from_hessian(H)
            if cov is not None:
                inv_m = cov if self.metric == "dense" else np.diag(cov).copy()
        for ch, x in zip(chains, pts, strict=True):
            ch.reset_geometry(fn, x, inv_m)

    # ---------------------------------------------------------- running
    def warmup(self) -> None:
        """Run warm-up (adaptation) if it has not been run yet."""
        self._setup()
        if self._warm:
            return
        assert self._chains is not None
        windows = self._chains[0].windows
        # adapt the geometry once the chains have left the mode for the typical set
        switch = min(self._geometry_iter, windows[0][0]) if windows else -1
        for it in progress(range(self.n_warmup), self.n_warmup, "NUTS warm-up", self.progress):
            if it == switch:
                self._adapt_geometry()
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
