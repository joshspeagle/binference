"""The one-call interface: :func:`infer`."""

from __future__ import annotations

from typing import Any

from ._utils import RNGLike
from .bins import Bins
from .model import PopulationModel
from .priors import PopulationPrior
from .results import PointEstimate, PopulationResult
from .samplers import MAP, NUTS, Bootstrap, Gibbs, Laplace, MeanFieldVB

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

#: Problems with more stored likelihood entries than this use Laplace under method="auto".
AUTO_NNZ_LIMIT = 2e7


def _auto_method(model: PopulationModel) -> str:
    nnz = model.likelihoods.nnz
    return "nuts" if nnz <= AUTO_NNZ_LIMIT else "laplace"


def infer(
    likelihoods: Any,
    prior: PopulationPrior | None = None,
    *,
    bins: Bins | None = None,
    selection: Any = None,
    selection_mode: str | None = None,
    method: str = "nuts",
    n_samples: int | None = None,
    rng: RNGLike = None,
    progress: bool = False,
    check: bool = True,
    **engine_kwargs: Any,
) -> PopulationResult | PointEstimate:
    """Infer a population distribution from per-object likelihoods in one call.

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
    selection : array_like (K,) or Selection, optional
        Detection probability (fractions) or exposure (rates) per bin.
    selection_mode : {"observed", "true"}, optional
        Required with an array-valued ``selection`` (see
        :class:`~binference.selection.Selection`).
    method : {"nuts", "gibbs", "map", "laplace", "vb", "bootstrap", "auto"}
        Inference engine. ``"nuts"`` (default) samples the exact posterior.
        ``"auto"`` uses NUTS unless the problem is very large, in which case it
        uses the importance-corrected Laplace approximation.
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
    if key == "auto":
        key = _auto_method(model)
    if key not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted([*METHODS, 'auto'])}.")
    engine = METHODS[key](model, rng=rng, progress=progress, **engine_kwargs)
    if key in ("map", "em"):
        if n_samples is not None:
            raise ValueError("n_samples does not apply to method='map'.")
        return engine.run()
    result = engine.run(_DEFAULT_SAMPLES[key] if n_samples is None else int(n_samples))
    if method == "auto":
        result.info["auto_method"] = key
    if check and key in ("nuts", "gibbs"):
        result.check()
    return result
