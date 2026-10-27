"""Base class and shared helpers for inference engines."""

from __future__ import annotations

from typing import Any

import numpy as np

from .. import _kernels as kern
from .._utils import RNGLike, as_generator
from ..model import PopulationModel

__all__ = ["Engine"]

#: Most regularised EM steps of the start of Gibbs and of MAP's EM and L-BFGS paths when
#: ``init`` is ``None``.
INTERIOR_START_ITER = 50


class Engine:
    """Base class of all inference engines.

    Engines are configured at construction and executed with :meth:`run`,
    which returns a :class:`~binference.PopulationResult` (or a
    :class:`~binference.PointEstimate` for :class:`~binference.samplers.MAP`).

    Parameters
    ----------
    model : PopulationModel
        The model to infer.
    rng : int, numpy.random.Generator, or None
        Random number generator.
    progress : bool
        Show a progress bar.
    """

    name: str = "engine"
    approximate: bool = False
    #: whether the engine infers a selection from injections or a tallied response jointly
    #: (DECISIONS #30, #80, #81)
    joint: bool = False
    #: whether its draws are Markov chains, described by R-hat and ESS
    markov: bool = False

    def __init__(self, model: PopulationModel, *, rng: RNGLike = None, progress: bool = False):
        if not isinstance(model, PopulationModel):
            raise TypeError("Engines operate on a binference.PopulationModel.")
        reason = self.unsupported(model)
        if reason is not None:
            raise ValueError(reason)
        self.model = model
        self.rng = as_generator(rng)
        self.progress = bool(progress)

    @classmethod
    def unsupported(cls, model: PopulationModel) -> str | None:
        """Why the engine cannot run on ``model``, or ``None`` when it can.

        Parameters
        ----------
        model : PopulationModel
            The model to run on.
        """
        if model.joint and not cls.joint:
            from ..inference import METHODS  # the registry imports the engines

            *names, last = (repr(k) for k, e in METHODS.items() if e.joint)
            fixed = "selection.at_mean()" if model.selection_uncertain else "response.fixed()"
            return (
                f"{cls.__name__} cannot infer a selection from injections or a tallied "
                f"response; use method={', '.join(names)} or {last}, or pass {fixed} for a "
                "fixed one that ignores its uncertainty."
            )
        return None

    def run(self, *args: Any, **kwargs: Any):  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.model!r})"


def em_update(
    model: PopulationModel, value: np.ndarray, s: np.ndarray | None = None
) -> np.ndarray | None:
    """One EM step for conjugate priors (fractions or rates, with selection), given the
    evidences ``s = L @ value`` when they are already computed.

    Returns ``None`` where an object has zero likelihood, rather than taking zero
    counts that would reset the step to the prior's mode (audit M-1).
    """
    s = kern.matvec(model._L, value) if s is None else s
    if not kern.possible(s):
        return None
    nhat = value * kern.rmatvec(model._L, model._w / s)
    counts = model._complete_counts(value, nhat)
    return model.prior.map_from_counts(counts, exposure=model._exposure)


def interior_start(model: PopulationModel, n_iter: int = INTERIOR_START_ITER) -> np.ndarray:
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
