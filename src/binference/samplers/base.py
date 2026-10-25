"""Base class and shared helpers for inference engines."""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import _kernels as kern
from .._utils import RNGLike, as_generator
from ..model import PopulationModel

__all__ = ["Engine"]


class Engine:
    """Base class of all inference engines.

    Engines are configured at construction and executed with :meth:`run`,
    which returns a :class:`~binference.PopulationResult` (or a
    :class:`~binference.PointEstimate` for :class:`~binference.samplers.MAP`).
    """

    name: str = "engine"
    approximate: bool = False
    #: whether the engine infers a selection from injections or a tallied response jointly
    #: (DECISIONS #30, #80, #81)
    joint: bool = False

    def __init__(self, model: PopulationModel, *, rng: RNGLike = None, progress: bool = False):
        if not isinstance(model, PopulationModel):
            raise TypeError("Engines operate on a binference.PopulationModel.")
        if model.joint and not self.joint:
            fixed = "selection.at_mean()" if model.selection_uncertain else "response.fixed()"
            raise ValueError(
                f"{type(self).__name__} cannot infer a selection from injections or a tallied "
                "response; use method='nuts', 'laplace', 'map' (space='unconstrained') or "
                f"'gibbs', or pass {fixed} for a fixed one that ignores its uncertainty."
            )
        self.model = model
        self.rng = as_generator(rng)
        self.progress = bool(progress)

    def run(self, *args: Any, **kwargs: Any):  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.model!r})"


def em_update(model: PopulationModel, value: np.ndarray) -> np.ndarray | None:
    """One EM step for conjugate priors (fractions or rates, with selection).

    Returns ``None`` where an object has zero likelihood, rather than taking zero
    counts that would reset the step to the prior's mode (audit M-1).
    """
    ll, nhat, _ = kern.loglike_and_counts(model._L, model._w, value)
    if not np.isfinite(ll):
        return None
    exposure = model._S if model.kind == "rates" else None
    return model.prior.map_from_counts(model._complete_counts(value, nhat), exposure=exposure)


def interior_start(model: PopulationModel, n_iter: int = 50) -> np.ndarray:
    """A robust interior starting value: up to ``n_iter`` regularised EM steps from uniform.

    Returns fractions (or rates for rate models) bounded away from zero.
    """
    K = model.K
    rho = np.full(K, 1.0 / K)
    for _ in range(n_iter):
        nhat = model._complete_counts(rho, model.expected_counts(rho))
        new = (nhat + 0.5) / (nhat.sum() + 0.5 * K)
        converged = np.max(np.abs(new - rho)) < 1e-6
        rho = new
        if converged:
            break
    if model.kind == "rates":
        # rho is the composition of the observed objects, n_k proportional to S_k lambda_k
        # (rates impute nothing), so the MLE-like start is W rho_k / S_k (audit M-7)
        return np.maximum(model.W * rho / np.maximum(model._S, 1e-12), 1e-12)
    return rho
