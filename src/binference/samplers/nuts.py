"""The No-U-Turn Sampler engine (default exact engine)."""

from __future__ import annotations

import os
import pickle
import warnings
from collections.abc import Iterator
from typing import Any

import numpy as np

from .._utils import RNGLike, progress, spawn, warn
from ..model import PopulationModel
from ..results import PopulationResult
from ..warnings import ConvergenceWarning
from ._curvature import covariance_from_hessian, fd_hessian, fd_hessian_diag
from ._nuts_core import Metric, adaptation_windows, find_reasonable_eps, nuts_step
from .base import Engine

__all__ = ["NUTS"]

# Dimension limits for the warm-up geometry adaptation (see NUTS docstring).
_DENSE_MAX_DIM = 200
_HESSIAN_MAX_DIM = 500
# Partial centring re-creates a funnel when the field itself pins the scale
# hyperparameters down poorly (few modes per hyperparameter). The prior then stays
# non-centred unless the median spread of the modes' log-scales given the field is
# below this. In validation runs 0.28 separated K <= 6, N <= 40 (0.33-0.62; divergences
# when centred) from K >= 10 (0.11-0.25; 3-20x more ESS/s when centred).
_CENTRING_MAX_SD = 0.28


def _centred(model: PopulationModel, lam: np.ndarray) -> PopulationModel:
    """The model with its logistic prior partially centred by weights ``lam``."""
    return model._with_bound_prior(model.prior.with_centering(lam))


def _conditional_log_scale_sd(model, prior, pts: list[np.ndarray]) -> np.ndarray:
    """Spread of each mode's log-scale given the field, from the hyperparameter curvature.

    In the centred form the field stays put while the hyperparameters move, so the
    curvature of the hyperparameter block alone says how well the field pins them.
    (The marginal spread from the full Hessian is unreliable: weakly identified
    length scales can make it arbitrarily large.) Modes with fixed scales get 0.
    """
    n_act = prior.n_active
    if prior.n_free == 0:
        return np.zeros(n_act)
    zeros = np.zeros(n_act)
    fn = _centred(model, zeros).log_density_unconstrained_and_grad
    cols = range(n_act, model.dim)
    H = np.mean([fd_hessian(fn, prior.recentre(x, zeros), cols=cols)[n_act:] for x in pts], axis=0)
    cov = covariance_from_hessian(H)
    if cov is None:
        return np.full(n_act, np.inf)
    _, dlogs = prior.log_scales(prior.theta_from_x(np.mean(pts, axis=0)))
    G = dlogs[prior.free]  # (n_free, n_active): d log s_j / d log theta
    return np.sqrt(np.maximum(np.einsum("aj,ab,bj->j", G, cov, G), 0.0))


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


_STAT_KEYS = ("accept_stat", "tree_depth", "diverging", "n_leapfrog", "energy")


def _advance_chain(chain: _Chain, n_iter: int, adapt: bool, max_depth: int, record: bool):
    """Advance one chain; returns ``(draws, log_probs, stats, n_divergent)``."""
    d = chain.x.size
    xs = np.empty((n_iter, d)) if record else None
    lps = np.empty(n_iter) if record else None
    st = np.empty((len(_STAT_KEYS), n_iter)) if record else None
    n_div = 0
    for i in range(n_iter):
        acc, depth, div, nl, energy = chain.step(max_depth, adapt)
        n_div += int(div)
        if record:
            xs[i] = chain.x
            lps[i] = chain.lp
            st[:, i] = (acc, depth, div, nl, energy)
    return xs, lps, st, n_div


def _cpu_count() -> int:
    # the cores this process may use (CPU affinity, e.g. under a batch scheduler)
    return os.process_cpu_count() or 1


def _worker_threads(n_workers: int) -> int:
    """BLAS threads per worker: the cores shared between workers, never above the
    limit already in force in this process (e.g. ``OMP_NUM_THREADS=1``; audit N-3)."""
    from threadpoolctl import threadpool_info

    share = max(1, _cpu_count() // n_workers)
    info = threadpool_info()
    # the BLAS pools' limit (another runtime, e.g. a single-threaded OpenMP, does not cap
    # it); where no BLAS pool is visible (e.g. Accelerate), every reported limit counts
    blas = [lib["num_threads"] for lib in info if lib.get("user_api") == "blas"]
    limits = [max(blas)] if blas else [lib["num_threads"] for lib in info]
    return min([share, *limits])


_WORKER_MODEL: PopulationModel | None = None  # the base model, sent once per worker


def _init_worker(model: PopulationModel, n_threads: int) -> None:
    global _WORKER_MODEL
    from threadpoolctl import threadpool_limits

    _WORKER_MODEL = model
    threadpool_limits(limits=n_threads)


def _chain_worker(payload):
    """Run in a worker process: advance one chain and send it back without the model.

    An error from the model comes back as a value, so the parent can tell it from a
    failure of the pool itself (audit N-7).
    """
    chain, lam, n_iter, adapt, max_depth, record = payload
    try:
        model = _WORKER_MODEL if lam is None else _centred(_WORKER_MODEL, lam)
        chain.fn = model.log_density_unconstrained_and_grad
        out = _advance_chain(chain, n_iter, adapt, max_depth, record)
    except Exception as err:
        return err
    chain.fn = None
    return chain, out


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
        Maximum tree depth (at least 1; at most ``2**max_depth`` gradient
        evaluations per draw).
    metric : {"auto", "diag", "dense"}
        Adapted Euclidean metric. ``"auto"`` uses a dense metric for logistic
        (smoothness) priors with at most 200 unconstrained dimensions, whose
        hyperparameters correlate with the field, and a diagonal one otherwise.
    reparametrize : bool
        For logistic priors, switch each basis coefficient between the
        non-centred and centred forms according to how strongly the data
        constrain it (partial non-centring with weights ``1 / -H_jj`` from the
        curvature at the chains' positions early in warm-up). If the field
        pins the hyperparameters down poorly (few basis coefficients per
        hyperparameter, e.g. a handful of bins), the coefficients stay
        non-centred, which avoids funnels. This changes the sampling
        coordinates, not the posterior.
    init : array_like (dim,) or (n_chains, dim), optional
        Starting points in the model's unconstrained coordinates; a single
        point gets ``jitter`` added for each chain. Every start must have a
        finite log density. By default chains start at the posterior mode (in
        unconstrained coordinates) plus ``jitter``.
    jitter : float
        Standard deviation of the Gaussian jitter around the starting point.
    n_jobs : int
        Number of worker processes for the chains (``-1``: one per CPU this
        process may use). The default ``1`` runs the chains in this process.
        The workers share the cores' BLAS threads instead of oversubscribing
        them, never exceeding a thread limit already set. Parallel and serial
        runs use the same random streams, so their draws agree up to rounding
        differences in multithreaded linear algebra. Parallel runs show no
        progress bar.
    rng : int, Generator or None
        Random number generator (SPEC 7); each chain gets an independent child.
    progress : bool
        Show a progress bar.

    Notes
    -----
    For logistic (smoothness) priors, after ``min(25, first window start)``
    warm-up iterations (none when ``n_warmup < 20``, which also skips metric
    adaptation) the engine reparametrises the prior (``reparametrize``) and,
    for at most 500 dimensions, initialises the metric from the
    finite-difference curvature of the log posterior at the chains'
    positions. Each later adaptation window shrinks its estimate toward the
    previous window's metric, starting from that curvature-based one.
    Simplex and rate priors use Stan's windowed adaptation unchanged.

    Results (``x``, ``log_density_unconstrained``, ``model``) are in the
    prior's non-centred coordinates whatever the sampling coordinates;
    ``info["centring"]`` holds the partial-centring weights used, and
    ``info["inv_metric"]`` is in those sampling coordinates.
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
        n_jobs: int = 1,
        rng: RNGLike = None,
        progress: bool = False,
    ) -> None:
        logistic = hasattr(model.prior, "with_centering")
        # results are in the non-centred form, so only engines ever hold a centred prior
        if logistic and np.any(model.prior.lam != 1.0):
            raise ValueError(
                "model: the logistic prior must be in its non-centred form (prior.lam all 1), "
                "as the model is built; NUTS chooses the centring itself."
            )
        super().__init__(model, rng=rng, progress=progress)
        if n_chains < 1:
            raise ValueError("n_chains must be at least 1.")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1.")
        if n_warmup < 0 or jitter < 0:
            raise ValueError("n_warmup and jitter must be non-negative.")
        if n_jobs == -1:
            n_jobs = _cpu_count()
        if n_jobs < 1:
            raise ValueError("n_jobs must be a positive integer or -1.")
        self.n_jobs = int(n_jobs)
        if metric not in ("auto", "diag", "dense"):
            raise ValueError("metric must be 'auto', 'diag' or 'dense'.")
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
        self._lam = None  # partial-centring weights of the sampling coordinates, once chosen
        self._pool = None  # worker processes, kept for one sample() (audit N-13)
        self._geometry_iter = 25
        self._warm = False
        self._x: list[np.ndarray] = []
        self._lp: list[np.ndarray] = []
        self._stats: dict[str, list[np.ndarray]] = {k: [] for k in _STAT_KEYS}
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
            if not all(np.isfinite(self.model.log_density_unconstrained(x)) for x in x0):
                raise ValueError("init has a starting point with zero posterior density.")
            return x0
        from .optimize import MAP

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            mode = MAP(self.model, space="unconstrained").run()
        x_mode = mode.x
        pts = []
        for _ in range(self.n_chains):
            for scale in (self.jitter, self.jitter / 10, 0.0):
                x = x_mode + scale * child.standard_normal(d)
                if np.isfinite(self.model.log_density_unconstrained(x)):
                    break
            pts.append(x)
        return np.asarray(pts)

    def _setup(self) -> None:
        if self._chains is not None:
            return
        x0 = self._initial_points()
        rngs = spawn(self.rng, self.n_chains)
        fn = self.model.log_density_unconstrained_and_grad
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
        base = self._base_model  # non-centred (see __init__)
        prior0 = base.prior
        if not hasattr(prior0, "with_centering"):
            return
        chains = self._chains
        d = base.dim
        pts = [ch.x for ch in chains]
        # curvature at every chain's position for small problems, one chain otherwise
        n_pts = len(chains) if d <= _DENSE_MAX_DIM else 1
        model = base
        if self.reparametrize and prior0.n_active > 0:
            fn0 = base.log_density_unconstrained_and_grad
            n_act = prior0.n_active
            # only the field's diagonal curvature is needed here (audit N-10)
            diag = np.mean([fd_hessian_diag(fn0, x, range(n_act)) for x in pts[:n_pts]], axis=0)
            lam = np.clip(1.0 / np.maximum(-diag, 1.0), 0.0, 1.0)
            # one decision for all modes that share the hyperparameters: mixing centred
            # and non-centred modes under one length scale mixed worst in validation
            sd_c = _conditional_log_scale_sd(base, prior0, pts)
            scaled = sd_c > 0.0
            if np.any(scaled) and np.median(sd_c[scaled]) > _CENTRING_MAX_SD:
                lam[scaled] = 1.0
            model = _centred(base, lam)
            pts = [prior0.recentre(x, lam) for x in pts]
            self._lam = lam
        self.model = model
        fn = model.log_density_unconstrained_and_grad
        inv_m = None
        if d <= _HESSIAN_MAX_DIM:
            H = np.mean([fd_hessian(fn, x) for x in pts[:n_pts]], axis=0)
            cov = covariance_from_hessian(H)
            if cov is not None:
                inv_m = cov if self.metric == "dense" else np.diag(cov).copy()
        for ch, x in zip(chains, pts, strict=True):
            ch.reset_geometry(fn, x, inv_m)

    # ---------------------------------------------------------- running
    def _parallel(self, n_iter: int, adapt: bool, record: bool):
        """Advance every chain ``n_iter`` iterations in worker processes.

        Returns per-chain outputs, or ``None`` if no worker pool could be used
        (the caller then runs serially; the chains are untouched). An error raised
        by the model in a worker comes back as a value and is raised here.
        """
        from concurrent.futures import ProcessPoolExecutor
        from concurrent.futures.process import BrokenProcessPool

        assert self._chains is not None
        chains = self._chains
        fn = self.model.log_density_unconstrained_and_grad
        for ch in chains:
            ch.fn = None  # the model goes to each worker once, not with every chain
        payloads = [(ch, self._lam, n_iter, adapt, self.max_depth, record) for ch in chains]
        try:
            if self._pool is None:
                try:  # e.g. a lambda in a prior cannot be sent to workers
                    pickle.dumps((self._base_model, payloads[0]))
                except (pickle.PicklingError, AttributeError, TypeError) as err:
                    return self._serial_from_now(err)
                n_workers = min(self.n_jobs, len(chains))
                self._pool = ProcessPoolExecutor(
                    max_workers=n_workers,
                    initializer=_init_worker,
                    initargs=(self._base_model, _worker_threads(n_workers)),
                )
            # errors from the model come back as values (see _chain_worker), so only a
            # failure of the pool itself falls back to serial (audit N-7)
            results = list(self._pool.map(_chain_worker, payloads))
        except (OSError, BrokenProcessPool, NotImplementedError) as err:
            return self._serial_from_now(err)
        finally:
            for ch in chains:
                ch.fn = fn
        for r in results:
            if isinstance(r, BaseException):
                raise r
        self._chains = []
        outs = []
        for ch, out in results:
            ch.fn = fn
            self._chains.append(ch)
            outs.append(out)
        return outs

    def _serial_from_now(self, err: BaseException) -> None:
        warn(
            f"Could not run chains in parallel ({type(err).__name__}: {err}); "
            "running them serially.",
            RuntimeWarning,
        )
        self.n_jobs = 1  # do not try again for this engine
        self._close_pool()

    def _close_pool(self) -> None:
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def _advance(self, n_iter: int, adapt: bool, desc: str) -> None:
        """Warm-up iterations for every chain (parallel or interleaved serial)."""
        if n_iter <= 0:
            return
        assert self._chains is not None
        if self.n_jobs > 1 and len(self._chains) > 1:
            outs = self._parallel(n_iter, adapt, record=False)
            if outs is not None:
                self.warmup_divergences += sum(o[3] for o in outs)
                return
        for _ in progress(range(n_iter), n_iter, desc, self.progress):
            for ch in self._chains:
                _, _, div, _, _ = ch.step(self.max_depth, adapt=adapt)
                self.warmup_divergences += int(div)

    def warmup(self) -> None:
        """Run warm-up (adaptation) if it has not been run yet."""
        self._setup()
        if self._warm:
            return
        assert self._chains is not None
        windows = self._chains[0].windows
        logistic = hasattr(self._base_model.prior, "with_centering")
        # adapt the geometry once the chains have left the mode for the typical set
        switch = min(self._geometry_iter, windows[0][0]) if (windows and logistic) else 0
        try:
            self._advance(switch, adapt=True, desc="NUTS warm-up")
            if switch > 0:
                self._adapt_geometry()
            self._advance(self.n_warmup - switch, adapt=True, desc="NUTS warm-up")
        finally:
            self._close_pool()  # warm-up phases share one pool; sampling starts its own
        self._warm = True

    def _record(self, xs, lps, st, n_done: int) -> None:
        if n_done:
            self._x.append(xs[:, :n_done])
            self._lp.append(lps[:, :n_done])
            for j, k in enumerate(_STAT_KEYS):
                self._stats[k].append(st[:, j, :n_done])

    def _report(self, X: np.ndarray, LP: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Draws and log densities from the sampling coordinates to the model's (audit N-1)."""
        if self._lam is None:
            return X, LP
        d = X.shape[-1]
        x, lp = self.model.prior.to_noncentred(X.reshape(-1, d), LP.reshape(-1))
        return x.reshape(X.shape), lp.reshape(LP.shape)

    def sample(self, n_samples: int) -> Iterator[np.ndarray]:
        """Generator over post-warm-up iterations, yielding ``(n_chains, dim)`` positions.

        Positions are in the model's unconstrained coordinates (the prior's
        non-centred form). Draws are also recorded and included in the next
        :meth:`run` / :attr:`result`. With ``n_jobs > 1`` the chains advance in
        worker processes and the positions are yielded once all have finished.
        """
        try:
            self.warmup()
            assert self._chains is not None
            C, d = self.n_chains, self.model.dim
            n_samples = int(n_samples)
            if self.n_jobs > 1 and C > 1 and n_samples > 0:
                outs = self._parallel(n_samples, adapt=False, record=True)
                if outs is not None:
                    xs, lps = self._report(
                        np.stack([o[0] for o in outs]), np.stack([o[1] for o in outs])
                    )
                    self._record(xs, lps, np.stack([o[2] for o in outs]), n_samples)
                    yield from (xs[:, i] for i in range(n_samples))
                    return
            xs = np.empty((C, n_samples, d))
            lps = np.empty((C, n_samples))
            st = np.empty((C, len(_STAT_KEYS), n_samples))
            n_done = 0
            try:
                for i in progress(range(n_samples), n_samples, "NUTS sampling", self.progress):
                    for c, ch in enumerate(self._chains):
                        acc, depth, div, nl, energy = ch.step(self.max_depth, adapt=False)
                        xs[c, i] = ch.x
                        lps[c, i] = ch.lp
                        st[c, :, i] = (acc, depth, div, nl, energy)
                    xs[:, i], lps[:, i] = self._report(xs[:, i], lps[:, i])
                    n_done = i + 1
                    yield xs[:, i]
            finally:
                self._record(xs, lps, st, n_done)
        finally:
            self._close_pool()

    def run(self, n_samples: int = 1000) -> PopulationResult:
        """Warm up (first call only), draw ``n_samples`` per chain, and return all draws so far."""
        for _ in self.sample(int(n_samples)):
            pass
        return self.result

    @property
    def result(self) -> PopulationResult:
        """A :class:`~binference.PopulationResult` with every draw recorded so far."""
        if not self._x:
            raise RuntimeError("No draws yet; call run().")
        X = np.concatenate(self._x, axis=1)
        LP = np.concatenate(self._lp, axis=1)
        C, n, d = X.shape
        base = self._base_model
        info: dict[str, Any] = {} if self._lam is None else {"centring": self._lam}
        vals = base.transform(X.reshape(-1, d)).reshape(C, n, -1)
        stats = {k: np.concatenate(v, axis=1) for k, v in self._stats.items()}
        info.update(stats)
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
        hyper = getattr(base.prior, "hyper_names", [])
        if hyper:
            u = X[..., base.prior.n_active :]
            info["hyperparameters"] = {name: np.exp(u[..., j]) for j, name in enumerate(hyper)}
        return PopulationResult(
            base,
            vals,
            method="nuts",
            kind=base.kind,
            x=X,
            log_density_unconstrained=LP,
            info=info,
        )
