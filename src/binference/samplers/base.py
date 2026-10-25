"""Base class and shared helpers for inference engines."""

from __future__ import annotations

from typing import Any

import numpy as np

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

    def __init__(self, model: PopulationModel, *, rng: RNGLike = None, progress: bool = False):
        if not isinstance(model, PopulationModel):
            raise TypeError("Engines operate on a binference.PopulationModel.")
        self.model = model
        self.rng = as_generator(rng)
        self.progress = bool(progress)

    def run(self, *args: Any, **kwargs: Any):  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.model!r})"


def em_update(model: PopulationModel, value: np.ndarray, weights: np.ndarray | None = None):
    """One EM step for conjugate priors (fractions or rates, with selection)."""
    prior = model.prior
    if weights is None:
        nhat = model.expected_counts(value)
        W = model.W
    else:
        from .. import _kernels as kern

        _, nhat, _ = kern.loglike_and_counts(model._L, weights, value)
        W = float(weights.sum())
    if model.kind == "rates":
        return prior.map_from_counts(nhat, exposure=model._exposure())
    if model.selection is not None:
        S = model._detect
        rho = value / value.sum()
        nhat = nhat + W * rho * (1.0 - S) / float(S @ rho)
    return prior.map_from_counts(nhat)


def interior_start(model: PopulationModel, n_iter: int = 50) -> np.ndarray:
    """A robust interior starting value: a few regularised EM steps from uniform.

    Returns fractions (or rates for rate models) bounded away from zero.
    """
    K = model.K
    rho = np.full(K, 1.0 / K)
    nhat = None
    for _ in range(n_iter):
        nhat = model.expected_counts(rho)
        if model.selection is not None and model.kind == "fractions":
            S = model._detect
            nhat = nhat + model.W * rho * (1.0 - S) / float(S @ rho)
        new = (nhat + 0.5) / (nhat.sum() + 0.5 * K)
        if np.max(np.abs(new - rho)) < 1e-6:
            rho = new
            break
        rho = new
    if model.kind == "rates":
        S = model._exposure()
        # expected counts per bin divided by exposure (MLE-like scale)
        tot = model.W if model.selection is None else model.W / float(S @ rho)
        return np.maximum(tot * rho / np.maximum(S, 1e-12), 1e-12)
    return rho
