"""The one-call interface: :func:`infer`."""

from __future__ import annotations

import inspect
import warnings
from math import ceil
from typing import Any

import numpy as np

from ._utils import RNGLike, check_positive, warn
from .bins import Bins
from .model import PopulationModel
from .priors import PopulationPrior
from .results import PointEstimate, PopulationResult
from .samplers import MAP, NUTS, Bootstrap, Gibbs, Laplace, MeanFieldVB
from .samplers.base import Engine
from .warnings import ApproximationWarning, BinferenceWarning

__all__ = ["METHODS", "Auto", "infer"]

#: Registry of inference methods accepted by :func:`infer`.
METHODS = {
    "nuts": NUTS,
    "gibbs": Gibbs,
    "map": MAP,
    "em": MAP,
    "laplace": Laplace,
    "vb": MeanFieldVB,
    "bootstrap": Bootstrap,
}


def _options(engine: type) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(engine.__init__).parameters)


def _default_draws(engine: type) -> int:
    """The engine's own default ``n_samples``."""
    return inspect.signature(engine.run).parameters["n_samples"].default


class Auto(Engine):
    r"""Picks Gibbs, Laplace or NUTS from rules set by the benchmark matrix (DECISIONS #88).

    The candidates, in order, keeping only engines that take every option in
    ``engine_kwargs``:

    1. Gibbs for a selection (not a smoothed injection selection) or response inferred
       jointly with a conjugate prior and integer weights.
    2. Gibbs for a conjugate prior with integer weights when the fraction of missing
       information at the MAP, :math:`f`, is at most ``gibbs_max_missing``, and the
       run it needs is at most ``max_gibbs_draws`` per chain. It runs
       :math:`\lceil t\,(1+f)/(1-f) / n_\mathrm{chains} \rceil` draws per chain (at
       least the usual 1000), for a smallest bulk ESS of about :math:`t` =
       ``target_ess``.

       Without ``n_samples``, either Gibbs run is then extended until every bulk ESS
       (of the population, and of an inferred selection or response) reaches
       :math:`t` and the run passes the :math:`\hat R` and ESS limits of
       :meth:`~binference.PopulationResult.check` (runs that store labels keep their
       length). If that is projected to need more than ``max_gibbs_draws`` draws per
       chain, its draws are discarded and the next candidate runs (a jointly inferred
       selection or response can mix much more slowly than :math:`f` predicts); when
       Gibbs is the only candidate it is kept, and the reason says it did not mix.
    3. Importance-corrected Laplace, kept when its Pareto :math:`\hat k` check passes
       and its importance ESS is at least ``target_ess``, or when the likelihoods have
       more than ``nnz_limit`` stored entries, or when it is the only candidate left;
       in those two cases a result that did not pass is flagged approximate (a short
       importance ESS with a warning of its own).
    4. NUTS.

    The result records the choice and the reason in ``info["auto_method"]`` and
    ``info["auto_reason"]``. Where rule 2 computed the MAP without a warning, the
    result keeps it as :attr:`~binference.PopulationResult.best`.

    Parameters
    ----------
    model : PopulationModel
    target_ess : float
        Smallest bulk ESS that Gibbs's run length aims for and that Laplace's
        importance ESS must reach.
    gibbs_max_missing : float
        Largest missing-information fraction for which Gibbs is chosen (0.98: an
        integrated autocorrelation time of 99 iterations).
    nnz_limit : float
        Above this many stored likelihood entries Laplace is kept even when its check
        fails, rather than running NUTS.
    max_gibbs_draws : int
        Most draws per chain a Gibbs run may be extended to before it gives way to the
        next candidate.
    rng, progress
        As for every engine.
    **engine_kwargs
        Options for the chosen engine.
    """

    name = "auto"
    joint = True

    def __init__(
        self,
        model: PopulationModel,
        *,
        target_ess: float = 400.0,
        gibbs_max_missing: float = 0.98,
        nnz_limit: float = 2e7,
        max_gibbs_draws: int = 25_000,
        rng: RNGLike = None,
        progress: bool = False,
        **engine_kwargs: Any,
    ) -> None:
        super().__init__(model, rng=rng, progress=progress)
        self.target_ess = float(check_positive(target_ess, "target_ess"))
        if not 0.0 <= gibbs_max_missing < 1.0:
            raise ValueError(f"gibbs_max_missing must be in [0, 1); got {gibbs_max_missing}.")
        self.gibbs_max_missing = float(gibbs_max_missing)
        self.nnz_limit = float(check_positive(nnz_limit, "nnz_limit"))
        self.max_gibbs_draws = int(check_positive(max_gibbs_draws, "max_gibbs_draws"))
        self.engine_kwargs = engine_kwargs
        self._rng_arg = rng
        self._map: tuple[Any, float] | None = None  # the MAP and its missing information

    def candidates(self, n_samples: int | None = None) -> list[tuple[str, str, int | None]]:
        """(engine, reason, Gibbs draws per chain) in the order they are tried, for runs
        of ``n_samples`` draws per chain (None: the engines' own lengths)."""
        m, kw = self.model, self.engine_kwargs
        out: list[tuple[str, str, int | None]] = []
        if Gibbs.unsupported(m) is None and set(kw) <= set(_options(Gibbs)):
            if m.joint:  # exact and fast (DECISIONS #80, #81)
                out.append(("gibbs", "a joint model with a conjugate prior", None))
            else:
                if self._map is None:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", BinferenceWarning)
                        est = MAP(m).run()
                    # kept as best only when it raised no warning (best would show them)
                    keep = est if not caught else None
                    self._map = keep, m.missing_information(est.value)  # component weights
                f = self._map[1]
                if f <= self.gibbs_max_missing:
                    chains = kw.get("n_chains", _options(Gibbs)["n_chains"].default)
                    n = max(
                        _default_draws(Gibbs),
                        ceil(self.target_ess * (1 + f) / (1 - f) / chains),
                    )
                    if n_samples is not None or n <= self.max_gibbs_draws:
                        out.append(("gibbs", f"conjugate prior, missing information {f:.2f}", n))
        out += [
            (key, "", None)
            for key, engine in (("laplace", Laplace), ("nuts", NUTS))
            if set(kw) <= set(_options(engine))
        ]
        if not out:
            raise ValueError(
                f"None of the engines method='auto' can use for this model (Laplace, NUTS, "
                f"and Gibbs where it applies) takes all of {sorted(kw)}; pass method= "
                "explicitly."
            )
        return out

    def run(self, n_samples: int | None = None) -> PopulationResult:
        """Run the first candidate that applies (see the class docstring).

        ``n_samples`` is the number of draws per chain; Laplace returns as many in
        total as NUTS's default four chains would (4000 by default).
        """
        plan = self.candidates(n_samples)
        rejected: list[str] = []  # why earlier candidates were not kept
        for i, (key, why, n_gibbs) in enumerate(plan):
            last = i == len(plan) - 1
            engine = METHODS[key](
                self.model, rng=self._rng_arg, progress=self.progress, **self.engine_kwargs
            )
            if key == "laplace":
                n = 4 * (n_samples or _default_draws(NUTS))
                result, why = (engine.run(n), "") if last else self._try(engine, n)
                if result is None:
                    rejected.append(why)
                    continue
                if not (result.approximate or self._passed(result)):
                    # k-hat passed (else Laplace flagged and warned) but the ESS fell short
                    result.approximate = True
                    short = f"its importance ESS is below target_ess={self.target_ess:g}"
                    why = "; ".join(filter(None, [why, short]))
                    warn(
                        f"auto kept Laplace draws although {short}; results are approximate.",
                        ApproximationWarning,
                    )
            else:
                n = n_samples or n_gibbs
                result = engine.run() if n is None else engine.run(n)
                # stored labels grow with the run: their length stays as asked
                extend = key == "gibbs" and n_samples is None and engine.label_index is None
                if extend and not self._extend(engine, result):
                    if not last:
                        rejected.append("Gibbs mixed too slowly")
                        continue
                    why += f"; it did not mix within max_gibbs_draws={self.max_gibbs_draws}"
                if extend:
                    result = engine.result
            reasons = [*rejected, why] if why else rejected
            result.info["auto_method"] = key
            result.info["auto_reason"] = (
                "; ".join(reasons) or "the only engine that takes these options"
            )
            if self._map is not None and self._map[0] is not None:  # not solved again
                result._cache["best"] = self._map[0]
            return result
        raise AssertionError("unreachable: the last candidate always returns")  # pragma: no cover

    def _extend(self, engine: Gibbs, result: PopulationResult) -> bool:
        """Extend the Gibbs run until its bulk ESS reaches ``target_ess`` and it passes the
        limits of ``check()``; False once that is projected to need more than
        ``max_gibbs_draws`` draws per chain."""
        while True:
            d = result.diagnostics
            ess = np.concatenate([np.ravel(v) for k, v in d.items() if k.startswith("ess_bulk")])
            ess = float(np.min(ess[np.isfinite(ess)], initial=np.inf))  # constant: no ESS
            if ess >= self.target_ess and not result._mixing_problems():
                return True
            done = result.chains.shape[1]
            need = ceil(done * max(1.2 * self.target_ess / max(ess, 1.0) - 1.0, 0.5))
            if done + need > self.max_gibbs_draws:
                return False
            # at most three times the draws so far, so a poor first projection costs less
            result = engine.run(min(need, 3 * done))

    def _passed(self, res: PopulationResult) -> bool:
        """Whether Laplace draws pass auto's check: Pareto k-hat and importance ESS."""
        return not res.approximate and res.info.get("ess_importance", 0.0) >= self.target_ess

    def _try(self, engine: Laplace, n: int) -> tuple[PopulationResult | None, str]:
        """Laplace's result when it is kept (its warnings re-emitted), else None.

        Laplace is only tried with no engine options, so it has its default 2n
        proposals. A first pass of 1000 of them stops the trial early when the
        importance ESS it projects for the full run is below half of ``target_ess``.
        """
        failed = None, "the Laplace importance check failed"
        big = self.model.likelihoods.nnz > self.nnz_limit  # kept whatever its check
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                if not big and 2 * n > 1000:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        engine.n_proposals = 1000
                        probe = engine.run(500)
                        engine.n_proposals = None
                    if probe.info["ess_importance"] * 2 * n / 1000 < self.target_ess / 2:
                        return failed
                res = engine.run(n)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                return None, "the Laplace approximation failed"
        if self._passed(res):
            reason = "the Laplace importance check passed"
        elif big:
            reason = f"more than {self.nnz_limit:.0e} likelihood entries"
        else:
            return failed
        for w in caught:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        return res, reason


METHODS["auto"] = Auto


def infer(
    likelihoods: Any,
    prior: PopulationPrior | None = None,
    *,
    bins: Bins | None = None,
    selection: Any = None,
    selection_mode: str | None = None,
    method: str = "auto",
    n_samples: int | None = None,
    rng: RNGLike = None,
    progress: bool = False,
    check: bool = True,
    **engine_kwargs: Any,
) -> PopulationResult | PointEstimate:
    r"""Infer a population distribution from per-object likelihoods in one call.

    Parameters
    ----------
    likelihoods : Likelihoods, PopulationModel, or array_like (N, K)
        Per-object likelihoods over bins (see :class:`~binference.Likelihoods` for
        converting posteriors, samples, Gaussian measurements, response
        matrices or classifier outputs). A raw array is treated as
        likelihoods over categorical bins (or ``bins``).
    prior : PopulationPrior, optional
        Population prior (default ``Dirichlet(1.0)``). Use
        :class:`~binference.priors.GammaRates` to infer rates.
    bins : Bins, optional
        Bins for a raw array input.
    selection : array_like (K,), callable or Selection, optional
        Detection probability (fractions) or exposure (rates) per bin.
    selection_mode : {"on_measurement", "on_truth"}, optional
        Required with an array or callable ``selection`` (see
        :class:`~binference.selection.Selection`).
    method : {"auto", "nuts", "gibbs", "map", "em", "laplace", "vb", "bootstrap"}
        Inference engine. ``"auto"`` (default) picks Gibbs, Laplace or NUTS from
        rules set by the benchmark matrix (:class:`Auto`, DECISIONS #88); its
        options ``target_ess``, ``gibbs_max_missing`` and ``nnz_limit`` go in
        ``engine_kwargs``, and the choice and its reason are in
        ``result.info["auto_method"]`` and ``result.info["auto_reason"]``.
        ``"nuts"`` samples the exact posterior with any prior; ``"em"`` is MAP by
        EM on the simplex (``algorithm="em"``) rather than mix-SQP; it needs a
        conjugate prior, and a jointly inferred selection or response runs L-BFGS.
    n_samples : int, optional
        Draws per chain (NUTS, Gibbs) or in total (Laplace, VB, bootstrap).
    rng : int, numpy.random.Generator, or None
        Random number generator or seed (SPEC 7).
    progress : bool
        Show progress bars.
    check : bool
        Warn about convergence problems after sampling (NUTS, Gibbs).
    **engine_kwargs
        Passed to the engine (e.g. ``n_chains``, ``n_warmup``, ``target_accept``
        for NUTS; see :mod:`binference.samplers`).

    Returns
    -------
    PopulationResult or PointEstimate
        Posterior draws, or the MAP estimate for ``method="map"`` and ``"em"``.

    Examples
    --------
    >>> import numpy as np, binference
    >>> rng = np.random.default_rng(1)
    >>> bins = binference.Bins.linear(0.0, 1.0, 5)
    >>> z = rng.uniform(0, 1, 300)
    >>> lk = binference.Likelihoods.from_gaussian(z + rng.normal(0, 0.05, 300), 0.05, bins)
    >>> res = binference.infer(lk, rng=0, n_samples=200, n_warmup=200)
    >>> res.samples.shape
    (800, 5)
    """
    if isinstance(likelihoods, PopulationModel):
        if any(v is not None for v in (prior, selection, selection_mode, bins)):
            raise ValueError("Pass prior/selection/bins when building the model, not to infer().")
        model = likelihoods
    else:
        model = PopulationModel(
            likelihoods, prior, selection=selection, selection_mode=selection_mode, bins=bins
        )
    key = str(method).lower()
    if key not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(METHODS)}.")
    if key == "em" and not model.joint:  # MAP by EM itself, not mix-SQP (audit O-17)
        engine_kwargs.setdefault("algorithm", "em")
    engine = METHODS[key](model, rng=rng, progress=progress, **engine_kwargs)
    if n_samples is None:
        result = engine.run()
    elif isinstance(engine, MAP):
        raise ValueError(f"n_samples does not apply to method={key!r}.")
    else:
        result = engine.run(int(n_samples))
    if check and METHODS[result.method].markov:
        result.check()
    return result
