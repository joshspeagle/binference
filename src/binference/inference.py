"""The one-call interface: :func:`infer`."""

from __future__ import annotations

import inspect
import warnings
from math import ceil
from typing import Any

import numpy as np

from . import diagnostics as diag
from ._utils import RNGLike, check_positive, check_positive_integers, warn
from .bins import Bins
from .engines import MAP, NUTS, Bootstrap, Gibbs, Laplace, MeanFieldVB
from .engines.base import Engine
from .engines.laplace import PROPOSAL_FACTOR
from .model import PopulationModel
from .priors import PopulationPrior
from .results import PointEstimate, PopulationResult, _report
from .warnings import ApproximationWarning, BinferenceWarning, InputWarning

__all__ = ["METHODS", "Auto", "infer"]

#: Registry of inference methods accepted by :func:`infer`; an engine's first key is its
#: name, and a later key is an alias (``"em"``).
METHODS = {
    "nuts": NUTS,
    "gibbs": Gibbs,
    "map": MAP,
    "em": MAP,
    "laplace": Laplace,
    "vb": MeanFieldVB,
    "bootstrap": Bootstrap,
}

#: (key, class) of the engines auto may run, in the order it tries them; each is made
#: from ``METHODS[key]``, and takes the options of the class
_CANDIDATES = (("gibbs", Gibbs), ("laplace", Laplace), ("nuts", NUTS))


def _options(engine: type) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(engine.__init__).parameters)


def _default_draws(engine: type) -> int:
    """The engine's own default ``n_samples``."""
    return inspect.signature(engine.run).parameters["n_samples"].default


class Auto(Engine):
    r"""Picks Gibbs, Laplace or NUTS from rules set by a benchmark of the engines.

    It tries Gibbs where it mixes well, then importance-corrected Laplace where its
    check passes, then NUTS, keeping only engines that take every option in
    ``engine_kwargs``. :doc:`/how-it-works/auto` states the rules and the benchmark.
    The result records the choice and the reason in ``info["auto_method"]`` and
    ``info["auto_reason"]``.

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
    extend_target : float
        An extension of a Gibbs run aims for this many times ``target_ess``.
    extend_max_factor, extend_min_factor : float
        One extension of a Gibbs run adds at most and at least these many times the draws
        so far.
    probe_proposals, probe_draws : int
        Proposals and draws of the first pass that can stop a Laplace trial early.
    probe_ess_share : float
        A Laplace trial stops when its first pass projects an importance ESS below this
        share of ``target_ess``.
    rng, progress
        As for every engine.
    **engine_kwargs
        Options for the chosen engine. Auto makes each engine it may run when it is
        made, so that each checks them at once.
    """

    joint = True

    def __init__(
        self,
        model: PopulationModel,
        *,
        target_ess: float = 400.0,
        gibbs_max_missing: float = 0.98,
        nnz_limit: float = 2e7,
        max_gibbs_draws: int = 25_000,
        extend_target: float = 1.2,
        extend_max_factor: float = 3.0,
        extend_min_factor: float = 0.5,
        probe_proposals: int = 1000,
        probe_draws: int = 500,
        probe_ess_share: float = 0.5,
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
        self.max_gibbs_draws = check_positive_integers(max_gibbs_draws, "max_gibbs_draws")[0]
        self.extend_target = float(check_positive(extend_target, "extend_target"))
        self.extend_max_factor = float(check_positive(extend_max_factor, "extend_max_factor"))
        self.extend_min_factor = float(check_positive(extend_min_factor, "extend_min_factor"))
        self.probe_proposals = check_positive_integers(probe_proposals, "probe_proposals")[0]
        if self.probe_proposals < diag.PSIS_MIN_WEIGHTS:
            raise ValueError(
                f"probe_proposals must be at least {diag.PSIS_MIN_WEIGHTS}; got {probe_proposals}."
            )
        self.probe_draws = check_positive_integers(probe_draws, "probe_draws")[0]
        self.probe_ess_share = float(check_positive(probe_ess_share, "probe_ess_share"))
        self.engine_kwargs = engine_kwargs
        self._rng_arg = rng
        self._map: tuple[Any, float] | None = None  # the MAP and its missing information
        self._may_run = tuple(
            key
            for key, engine in _CANDIDATES
            if engine.unsupported(model) is None and set(engine_kwargs) <= set(_options(engine))
        )
        if not self._may_run:
            raise self._none_takes_the_options()
        for key in self._may_run:  # made once now so that each checks its options (#174)
            self._make(key)

    def _make(self, key: str) -> Engine:
        """The engine ``key`` with auto's options."""
        return METHODS[key](
            self.model, rng=self._rng_arg, progress=self.progress, **self.engine_kwargs
        )

    def _none_takes_the_options(self) -> ValueError:
        return ValueError(
            f"None of the engines method='auto' can use for this model (Laplace, NUTS, "
            f"and Gibbs where it applies) takes all of {sorted(self.engine_kwargs)}; pass "
            "method= explicitly."
        )

    def candidates(self, n_samples: int | None = None) -> list[tuple[str, str, int | None]]:
        """(engine, reason, Gibbs draws per chain) in the order they are tried.

        Parameters
        ----------
        n_samples : int, optional
            Least draws per chain of the runs; ``None`` uses the engines' own lengths.
        """
        return self._plan(n_samples)[0]

    def _plan(self, n_samples: int | None) -> tuple[list[tuple[str, str, int | None]], list[str]]:
        # the candidates, and why the rules leave out Gibbs where it could run
        m, kw = self.model, self.engine_kwargs
        out: list[tuple[str, str, int | None]] = []
        left_out: list[str] = []
        if "gibbs" in self._may_run:
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
                    if n <= self.max_gibbs_draws:
                        n = n if n_samples is None else max(n, n_samples)
                        out.append(("gibbs", f"conjugate prior, missing information {f:.2f}", n))
                    else:
                        left_out.append(
                            f"Gibbs would need {n} draws per chain "
                            f"(> max_gibbs_draws={self.max_gibbs_draws})"
                        )
                else:
                    left_out.append(
                        f"Gibbs would mix too slowly (missing information {f:.3f} > "
                        f"gibbs_max_missing={self.gibbs_max_missing:g})"
                    )
        out += [(key, "", None) for key in self._may_run if key != "gibbs"]
        if not out:  # only Gibbs takes the options, and a rule left it out
            raise ValueError(
                f"Only Gibbs takes all of {sorted(kw)}, but {left_out[0]}; pass "
                "method='gibbs' to run it anyway."
            )
        return out, left_out

    def run(self, n_samples: int | None = None) -> PopulationResult:
        """Run the first candidate that applies (see the class docstring).

        Parameters
        ----------
        n_samples : int, optional
            Least draws per chain; ``None`` lets auto set the run length. Laplace returns as
            many in total as NUTS's default four chains would (4000 by default).
        """
        if n_samples is not None:
            n_samples = check_positive_integers(n_samples, "n_samples")[0]
        plan, rejected = self._plan(n_samples)  # and why earlier engines were not kept
        for i, (key, why, n_gibbs) in enumerate(plan):
            last = i == len(plan) - 1
            engine = self._make(key)
            if key == "laplace":
                per_chain = _default_draws(NUTS) if n_samples is None else n_samples
                n = _options(NUTS)["n_chains"].default * per_chain
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
                # stored labels keep the length asked for, which may be a memory limit
                labels = key == "gibbs" and engine.label_index is not None
                n = n_samples if labels and n_samples is not None else n_gibbs or n_samples
                result = engine.run() if n is None else engine.run(n)
                if key == "gibbs" and engine.label_index is None:
                    result, mixed = self._extend(engine, result)
                    if not mixed:
                        if not last:
                            rejected.append("Gibbs mixed too slowly")
                            continue
                        why += f"; it did not mix within max_gibbs_draws={self.max_gibbs_draws}"
            reasons = [*rejected, why] if why else rejected
            result.info["auto_method"] = key
            result.info["auto_reason"] = (
                "; ".join(reasons) or "the only engine that takes these options"
            )
            if self._map is not None and self._map[0] is not None:  # not solved again
                result._cache["best"] = self._map[0]
            return result
        raise AssertionError("unreachable: the last candidate always returns")  # pragma: no cover

    def _extend(self, engine: Gibbs, result: PopulationResult) -> tuple[PopulationResult, bool]:
        """Extend the Gibbs run until its bulk ESS reaches ``target_ess`` and it passes the
        limits of ``check()``: the result with every draw so far (its diagnostics computed)
        and whether it got there, False once that is projected to need more than
        ``max_gibbs_draws`` draws per chain."""
        while True:
            d = result.diagnostics
            ess = [np.ravel(v) for k, v in d.items() if k.startswith("ess_bulk")]
            # chains too short for an ESS have no value yet; a constant chain has no finite one
            ess = np.concatenate(ess) if ess else np.zeros(1)
            ess = float(np.min(ess[np.isfinite(ess)], initial=np.inf))
            if ess >= self.target_ess and not result._mixing_problems():
                return result, True
            done = result.chains.shape[1]
            need = ceil(
                done
                * max(
                    self.extend_target * self.target_ess / max(ess, 1.0) - 1.0,
                    self.extend_min_factor,
                )
            )
            if done + need > self.max_gibbs_draws:
                return result, False
            # bounded, so a poor first projection costs less
            result = engine.run(min(need, ceil(self.extend_max_factor * done)))

    def _passed(self, res: PopulationResult) -> bool:
        """Whether Laplace draws pass auto's check: Pareto k-hat and importance ESS."""
        return not res.approximate and res.info.get("ess_importance", 0.0) >= self.target_ess

    def _try(self, engine: Laplace, n: int) -> tuple[PopulationResult | None, str]:
        """Laplace's result when it is kept (its warnings re-emitted), else None.

        Laplace is only tried with no engine options, so it has its default
        ``PROPOSAL_FACTOR * n`` proposals. A first pass of ``probe_proposals`` of them
        stops the trial early when the importance ESS it projects for the full run is
        below ``probe_ess_share`` of ``target_ess``.
        """
        failed = None, "the Laplace importance check failed"
        big = self.model.likelihoods.nnz > self.nnz_limit  # kept whatever its check
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                proposals = PROPOSAL_FACTOR * n
                if not big and proposals > self.probe_proposals:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        engine.n_proposals = self.probe_proposals
                        probe = engine.run(self.probe_draws)
                        engine.n_proposals = None
                    projected = probe.info["ess_importance"] * proposals / self.probe_proposals
                    if projected < self.probe_ess_share * self.target_ess:
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
        rules set by a benchmark of the engines (:class:`Auto`); its
        options (:class:`Auto`) go in ``engine_kwargs``, and the choice and its reason are in
        ``result.info["auto_method"]`` and ``result.info["auto_reason"]``.
        ``"nuts"`` samples the exact posterior with any prior; ``"em"`` is MAP by
        EM on the simplex (``algorithm="em"``) rather than mix-SQP; it needs a
        conjugate prior, and a jointly inferred selection or response runs L-BFGS.
    n_samples : int, optional
        Draws per chain (NUTS, Gibbs, and ``"auto"``) or in total (Laplace, VB,
        bootstrap). Under ``"auto"`` it is a minimum: a Gibbs run may be extended past it
        until it mixes, and Gibbs still gives way when it would need more than
        ``max_gibbs_draws`` (see :class:`Auto`).
    rng : int, numpy.random.Generator, or None
        Random number generator or seed (SPEC 7).
    progress : bool
        Show progress bars.
    check : bool
        Warn about problems after the fit: convergence for NUTS and Gibbs, and the inputs
        for every engine: the Monte Carlo noise of sample likelihoods and, for results
        with draws, binned counts the response does not describe.
    **engine_kwargs
        Passed to the engine (e.g. ``n_chains``, ``n_warmup``, ``target_accept``
        for NUTS; see :mod:`binference.engines`).

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
    >>> res = binference.infer(lk, rng=0)
    >>> res.mean.shape
    (5,)
    """
    if n_samples is not None:
        n_samples = check_positive_integers(n_samples, "n_samples")[0]
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
    elif check:  # convergence is the engine's own; the inputs are every engine's concern
        inputs, hints = result._input_problems()
        _report([(inputs, hints, InputWarning)], strict=False)
    return result
