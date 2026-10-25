"""Warning classes emitted by binference.

All warnings derive from :class:`BinferenceWarning`, so they can be filtered in one
place, e.g. ``warnings.simplefilter("ignore", binference.BinferenceWarning)``.
"""

from __future__ import annotations

__all__ = [
    "ApproximationWarning",
    "BinferenceWarning",
    "ConvergenceWarning",
    "DiagnosticsError",
    "InputWarning",
]


class BinferenceWarning(UserWarning):
    """Base class for all binference warnings."""


class InputWarning(BinferenceWarning):
    """Inputs look inconsistent with the declared conventions.

    Examples: posterior masses whose rows do not sum to one, values that look
    like densities when masses were declared, or objects with too few samples.
    """


class ConvergenceWarning(BinferenceWarning):
    """A sampler or optimiser shows signs of non-convergence.

    Examples: large R-hat, low effective sample size, divergent transitions, or
    an optimiser that stopped before meeting its tolerance.
    """


class ApproximationWarning(BinferenceWarning):
    """A result comes from an approximation with known failure modes.

    Mean-field variational Bayes, Laplace approximations without importance
    correction (or with an unreliable correction), and bootstrap intervals are
    flagged with this warning.
    """


class DiagnosticsError(RuntimeError):
    """``result.check(strict=True)`` found problems (the messages its warnings carry)."""
