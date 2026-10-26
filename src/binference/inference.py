"""The one-call interface: :func:`infer`."""

from __future__ import annotations

import inspect
import warnings
from typing import Any

import numpy as np

from ._utils import RNGLike, is_integer_array
from .bins import Bins
from .model import PopulationModel
from .priors import PopulationPrior
from .results import PointEstimate, PopulationResult
from .samplers import MAP, NUTS, Bootstrap, Gibbs, Laplace, MeanFieldVB
from .samplers.base import Engine
from .warnings import BinferenceWarning

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

_DEFAULT_SAMPLES = {"nuts": 1000, "gibbs": 1000, "laplace": 4000, "vb": 4000, "bootstrap": 200}


def _options(engine: type) -> dict[str, inspect.Parameter]:
    return dict(inspect.signature(engine.__init__).parameters)


class Auto(Engine):
    r"""Picks Gibbs, Laplace or NUTS from rules set by the benchmark matrix (DECISIONS #88).

    The candidates, in order, keeping only engines that take every option in
    ``engine_kwargs``:

    1. Gibbs for a selection or response inferred jointly with a conjugate prior.
    2. Gibbs for a conjugate prior with integer weights when the fraction of missing
       information at the MAP, :math:`f`, is at most ``gibbs_max_missing``. It runs
       :math:`\lceil t\,(1+f)/(1-f) / n_\mathrm{chains} \rceil` draws per chain (at
       least the usual 1000), for a smallest bulk ESS of about :math:`t` =
       ``target_ess``.
    3. Importance-corrected Laplace, kept when its Pareto :math:`\hat k` check passes
       and its importance ESS is at least ``target_ess``, or when the likelihoods have
       more than ``nnz_limit`` stored entries (then flagged approximate), or when it is
       the only candidate left.
    4. NUTS.

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
        rng: RNGLike = None,
        progress: bool = False,
        **engine_kwargs: Any,
    ) -> None:
        super().__init__(model, rng=rng, progress=progress)
        self.target_ess = float(target_ess)
        self.gibbs_max_missing = float(gibbs_max_missing)
        self.nnz_limit = float(nnz_limit)
        self.engine_kwargs = engine_kwargs
        self._rng_arg = rng

    def candidates(self) -> list[tuple[str, str, int | None]]:
        """(engine, reason, Gibbs draws per chain) in the order they are tried."""
        m, kw = self.model, self.engine_kwargs
        out: list[tuple[str, str, int | None]] = []
        if (
            m.prior.conjugate
            and not (m.selection_uncertain and m.selection_spec.injections.smooth is not None)
            and is_integer_array(m.likelihoods.weights)
            and set(kw) <= set(_options(Gibbs))
        ):
            if m.joint:  # exact and fast (DECISIONS #80, #81)
                out.append(("gibbs", "a joint model with a conjugate prior", None))
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", BinferenceWarning)
                    f = m.missing_information(MAP(m).run().rho)
                if f <= self.gibbs_max_missing:
                    chains = kw.get("n_chains", _options(Gibbs)["n_chains"].default)
                    n = int(np.ceil(self.target_ess * (1 + f) / (1 - f) / chains))
                    reason = f"conjugate prior, missing information {f:.2f}"
                    out.append(("gibbs", reason, max(_DEFAULT_SAMPLES["gibbs"], n)))
        out += [
            (key, "", None)
            for key, engine in (("laplace", Laplace), ("nuts", NUTS))
            if set(kw) <= set(_options(engine))
        ]
        if not out:
            raise ValueError(
                f"No engine that method='auto' chooses from takes all of {sorted(kw)}; "
                "pass method= explicitly."
            )
        return out

    def run(self, n_samples: int | None = None) -> PopulationResult:
        """Run the first candidate that applies (see the class docstring)."""
        plan = self.candidates()
        reason = "the exact default"
        for i, (key, why, n_gibbs) in enumerate(plan):
            n = n_samples if n_samples is not None else n_gibbs or _DEFAULT_SAMPLES[key]
            engine = METHODS[key](
                self.model, rng=self._rng_arg, progress=self.progress, **self.engine_kwargs
            )
            if key == "laplace" and i < len(plan) - 1:
                result, reason = self._try(engine, int(n))
                if result is None:
                    continue
            else:
                result = engine.run(int(n))
                reason = why or (reason if i else "the only engine that takes these options")
            result.info["auto_method"] = key
            result.info["auto_reason"] = reason
            return result
        raise AssertionError("unreachable: NUTS or Laplace is always last")  # pragma: no cover

    def _try(self, engine: Laplace, n: int) -> tuple[PopulationResult | None, str]:
        """Laplace's result when it is kept (its warnings re-emitted), else None."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                res = engine.run(n)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                return None, "the Laplace approximation failed"
        if not res.approximate and res.info.get("ess_importance", 0.0) >= self.target_ess:
            reason = "the Laplace importance check passed"
        elif self.model.likelihoods.nnz > self.nnz_limit:
            reason = f"more than {self.nnz_limit:.0e} likelihood entries"
        else:
            return None, "the Laplace importance check failed"
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
        EM itself (``algorithm="em"``) rather than mix-SQP.
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
        Posterior draws (or the MAP estimate for ``method="map"``).

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
    if key in ("map", "em"):
        if n_samples is not None:
            raise ValueError(f"n_samples does not apply to method={key!r}.")
        return engine.run()
    result = engine.run(_DEFAULT_SAMPLES.get(key) if n_samples is None else int(n_samples))
    if check and result.method in ("nuts", "gibbs"):
        result.check()
    return result
