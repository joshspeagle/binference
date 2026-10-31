"""Base class and shared helpers for inference engines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .. import _kernels as kern
from .._utils import RNGLike, as_generator, check_nonnegative, check_positive
from ..model import PopulationModel

if TYPE_CHECKING:
    from ..results import PopulationResult

__all__ = ["Engine"]

#: Most regularised EM steps of the start of Gibbs and of MAP's EM and L-BFGS paths when
#: ``init`` is ``None``.
INTERIOR_START_ITER = 50


class Engine:
    """Base class of all inference engines.

    Engines are configured at construction and executed with :meth:`run`,
    which returns a :class:`~binference.PopulationResult` (or a
    :class:`~binference.PointEstimate` for :class:`~binference.samplers.MAP`).
    Construction checks the options and draws no random numbers, so ``method="auto"``
    makes an engine to check its options and makes it again to run it.

    Parameters
    ----------
    model : PopulationModel
        The model to infer.
    rng : int, numpy.random.Generator, or None
        Random number generator.
    progress : bool
        Show a progress bar.
    """

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

            first: dict[type, str] = {}  # each engine by its first name, without aliases
            for k, e in METHODS.items():
                first.setdefault(e, k)
            *names, last = (repr(k) for e, k in first.items() if e.joint)
            fixed = "selection.fixed()" if model.selection_uncertain else "response.fixed()"
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


class MarkovEngine(Engine):
    """Base class of the engines whose draws are Markov chains, which every
    :meth:`run` after the first continues."""

    markov = True

    def __init__(self, model: PopulationModel, *, rng: RNGLike = None, progress: bool = False):
        super().__init__(model, rng=rng, progress=progress)
        self._warm = False  # whether warm-up has run
        self._draws: list[np.ndarray] = []  # (n_chains, n, ...) per sample() call

    def warmup(self) -> None:
        """Run warm-up (Gibbs's burn-in, NUTS's adaptation) if it has not been run yet."""
        self._setup()
        if not self._warm:
            self._warm_up()
            self._warm = True

    def run(self, n_samples: int = 1000) -> PopulationResult:
        """Warm up (first call only), draw ``n_samples`` per chain, and return all draws so far.

        Parameters
        ----------
        n_samples : int, optional
            Draws per chain after warm-up.
        """
        for _ in self.sample(int(n_samples)):
            pass
        return self.result

    def _all_draws(self) -> np.ndarray:
        """Every draw so far, ``(n_chains, n, ...)``."""
        if not self._draws:
            raise RuntimeError("No draws yet; call run().")
        return np.concatenate(self._draws, axis=1)


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


def check_init(init: Any, model: PopulationModel, *, positive: bool = True) -> np.ndarray | None:
    """``init`` as fractions or rates of shape (K,), checked: positive, or with
    ``positive=False`` non-negative and not all zero; ``None`` stays ``None``."""
    if init is None:
        return None
    init = check_positive(init, "init") if positive else check_nonnegative(init, "init")
    if init.shape != (model.K,):
        raise ValueError(
            f"init must be fractions or rates of shape ({model.K},); got {init.shape}."
        )
    if not positive and not init.any():
        raise ValueError("init must have a positive entry.")
    return init


def interior_start(model: PopulationModel, n_iter: int = INTERIOR_START_ITER) -> np.ndarray:
    """A robust interior starting value: up to ``n_iter`` regularised EM steps from uniform.

    Returns fractions (or rates for rate models) bounded away from zero.
    """
    K = model.K
    # for rates rho is the observed composition, rho_k proportional to S_k lambda_k: its
    # expected counts are those of the rates rho / S, and the start is W rho / S (audit M-7)
    rates = model.kind == "rates"
    rho = np.full(K, 1.0 / K)
    for _ in range(n_iter):
        value = rho / model._S if rates else rho
        nhat = model._complete_counts(rho, model.expected_counts(value))
        new = (nhat + 0.5) / (nhat.sum() + 0.5 * K)
        converged = np.max(np.abs(new - rho)) < 1e-6
        rho = new
        if converged:
            break
    if rates:
        return np.maximum(model.W * rho / model._S, 1e-12)  # a selection's S is positive
    return rho
