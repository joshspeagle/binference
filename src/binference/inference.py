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
from .warnings import BinferenceWarning

__all__ = ["METHODS", "infer"]

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

#: method="auto" aims for at least this smallest bulk ESS (Laplace: importance ESS).
AUTO_TARGET_ESS = 400
#: method="auto" uses Gibbs for conjugate priors up to this fraction of missing
#: information, i.e. integrated autocorrelation times (1 + f) / (1 - f) up to 99.
AUTO_GIBBS_MAX_MISSING = 0.98
#: Problems with more stored likelihood entries than this skip NUTS under method="auto".
AUTO_NNZ_LIMIT = 2e7


def _accepts(engine: type, kwargs: dict) -> bool:
    return set(kwargs) <= set(inspect.signature(engine.__init__).parameters)


def _auto_method(
    model: PopulationModel, engine_kwargs: dict | None = None
) -> tuple[str, str, int | None]:
    """The engine for method="auto", the reason, and Gibbs's draws per chain (DECISIONS #88).

    ``"laplace?"`` means: run Laplace and keep it if its importance check passes, else
    NUTS. Gibbs runs long enough for a smallest bulk ESS of about ``AUTO_TARGET_ESS``,
    from the integrated autocorrelation time (1 + f) / (1 - f) at the MAP.
    """
    kw = engine_kwargs or {}
    gibbs_ok = (
        model.prior.conjugate
        and not (model.selection_uncertain and model.selection_spec.injections.smooth is not None)
        and is_integer_array(model.likelihoods.weights)
        and _accepts(Gibbs, kw)
    )
    if gibbs_ok and model.joint:  # exact and fast (DECISIONS #80, #81)
        return "gibbs", "a selection or response inferred jointly with a conjugate prior", None
    if gibbs_ok:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BinferenceWarning)
            f = model.missing_information(MAP(model).run().rho)
        if f <= AUTO_GIBBS_MAX_MISSING:
            n_chains = kw.get("n_chains", 4)
            n = int(np.ceil(AUTO_TARGET_ESS * (1 + f) / (1 - f) / n_chains))
            return "gibbs", f"conjugate prior, missing information {f:.2f}", max(1000, n)
    if _accepts(Laplace, kw):
        return "laplace?", "", None
    return "nuts", "engine options that Laplace does not take", None


def _try_laplace(model, n_samples, rng, progress, engine_kwargs):
    """Laplace's result and its warnings, and whether its importance check passed with
    an ESS of at least ``AUTO_TARGET_ESS``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = Laplace(model, rng=rng, progress=progress, **engine_kwargs).run(n_samples)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            return None, caught, False
    ok = not res.approximate and res.info.get("ess_importance", 0.0) >= AUTO_TARGET_ESS
    return res, caught, ok


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
        Inference engine. ``"auto"`` (default) picks one from rules set by the
        benchmark matrix (DECISIONS #88), in this order:

        1. Gibbs for a selection or response inferred jointly with a conjugate prior.
        2. Gibbs for a conjugate prior with integer weights when the fraction of
           missing information at the MAP, :math:`f`, is at most
           ``AUTO_GIBBS_MAX_MISSING`` (0.98). It runs
           :math:`\lceil 400 (1+f)/(1-f) / n_\mathrm{chains} \rceil` draws per
           chain (at least 1000), for a smallest bulk ESS of about
           ``AUTO_TARGET_ESS`` (400).
        3. The importance-corrected Laplace approximation, kept when its Pareto
           :math:`\hat k` passes and its importance ESS is at least 400.
        4. NUTS, or the Laplace draws flagged approximate above
           ``AUTO_NNZ_LIMIT`` stored likelihood entries.

        Engine options that an engine does not take rule it out (e.g.
        ``target_accept`` leaves NUTS). The choice and its reason are in
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
    auto = key == "auto"
    if auto:
        key, reason, n_gibbs = _auto_method(model, engine_kwargs)
        if key == "gibbs" and n_samples is None:
            n_samples = n_gibbs
        if key == "laplace?":
            n = _DEFAULT_SAMPLES["laplace"] if n_samples is None else int(n_samples)
            res, caught, ok = _try_laplace(model, n, rng, progress, engine_kwargs)
            big = model.likelihoods.nnz > AUTO_NNZ_LIMIT
            if ok or (big and res is not None):
                for w in caught:  # the kept fit's warnings
                    warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
                res.info["auto_method"] = "laplace"
                res.info["auto_reason"] = (
                    "the Laplace importance check passed"
                    if ok
                    else f"more than {AUTO_NNZ_LIMIT:.0e} likelihood entries"
                )
                return res
            key, reason = "nuts", "the Laplace importance check failed"
    if key not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted([*METHODS, 'auto'])}.")
    if key == "em" and not model.joint:  # MAP by EM itself, not mix-SQP (audit O-17)
        engine_kwargs.setdefault("algorithm", "em")
    engine = METHODS[key](model, rng=rng, progress=progress, **engine_kwargs)
    if key in ("map", "em"):
        if n_samples is not None:
            raise ValueError(f"n_samples does not apply to method={key!r}.")
        return engine.run()
    result = engine.run(_DEFAULT_SAMPLES[key] if n_samples is None else int(n_samples))
    if auto:
        result.info["auto_method"] = key
        result.info["auto_reason"] = reason
    if check and key in ("nuts", "gibbs"):
        result.check()
    return result
