"""Warning classes emitted by pastry.

All warnings derive from :class:`PastryWarning`, so they can be filtered in one
place, e.g. ``warnings.simplefilter("ignore", pastry.PastryWarning)``.
"""

from __future__ import annotations

__all__ = ["ApproximationWarning", "ConvergenceWarning", "InputWarning", "PastryWarning"]


class PastryWarning(UserWarning):
    """Base class for all pastry warnings."""


class InputWarning(PastryWarning):
    """Inputs look inconsistent with the declared conventions.

    Examples: posterior masses whose rows do not sum to one, values that look
    like densities when masses were declared, or objects with too few samples.
    """


class ConvergenceWarning(PastryWarning):
    """A sampler or optimiser shows signs of non-convergence.

    Examples: large R-hat, low effective sample size, divergent transitions, or
    an optimiser that stopped before meeting its tolerance.
    """


class ApproximationWarning(PastryWarning):
    """A result comes from an approximation with known failure modes.

    Mean-field variational Bayes, Laplace approximations without importance
    correction (or with an unreliable correction), and bootstrap intervals are
    flagged with this warning.
    """
