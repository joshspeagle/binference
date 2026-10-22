r"""Selection functions (detection probabilities or exposures per bin).

Selection enters the likelihood in one of two ways, and choosing the wrong one
biases the population (Mandel, Farr & Gair 2019). pastry therefore requires
the mode to be stated explicitly:

``mode="observed"``
    Selection acts on the *observed data* (e.g. a flux or signal-to-noise
    cut), or the per-object likelihoods already include the detection
    probability (e.g. response matrices that fold in efficiencies). The
    likelihood of each detected object is used as is, and only the
    normalisation :math:`(S\cdot\rho)^{-W}` (fractions) or
    :math:`e^{-S\cdot\lambda}` (rates) is added. Here :math:`S_k` is the
    probability that an object in bin *k* would be selected.

``mode="true"``
    Objects are detected with probability :math:`S_k` that depends on their
    *true* bin, independently of their measured data given the truth (e.g. a
    completeness that depends on true planet radius). Each object's
    likelihood row is multiplied by :math:`S_k` *and* the normalisation is
    added.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["SELECTION_MODES", "Selection", "as_selection"]

SELECTION_MODES = ("observed", "true")

_HELP = (
    "A selection function needs an explicit selection_mode: 'observed' if objects are "
    "selected on their observed data (e.g. a flux limit) or the likelihoods already include "
    "the detection probability; 'true' if detection depends on the true bin independently of "
    "the measured data. See the user guide on selection effects."
)


class Selection:
    """Per-bin detection probability (fractions) or exposure (rates), with its mode.

    Parameters
    ----------
    values : array_like (K,)
        Strictly positive detection probabilities (for fractions, only ratios
        between bins matter) or exposures (for rates, the absolute scale
        matters: expected counts are ``values * rates``).
    mode : {"observed", "true"}
        How selection enters the likelihood (see the module documentation).
    """

    __slots__ = ("mode", "values")

    def __init__(self, values: Any, mode: str) -> None:
        if mode not in SELECTION_MODES:
            raise ValueError(f"selection mode must be 'observed' or 'true'; got {mode!r}. {_HELP}")
        v = np.asarray(values, dtype=float).ravel()
        if v.size < 2:
            raise ValueError("selection needs one value per bin.")
        if not np.all(np.isfinite(v)) or np.any(v <= 0):
            raise ValueError(
                "selection values must be finite and strictly positive (bins that can never be "
                "observed are not identifiable; remove them from the bins)."
            )
        v.setflags(write=False)
        self.values = v
        self.mode = mode

    @property
    def K(self) -> int:
        return int(self.values.size)

    def __repr__(self) -> str:
        return f"Selection(K={self.K}, mode={self.mode!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Selection):
            return NotImplemented
        return self.mode == other.mode and np.array_equal(self.values, other.values)

    def __hash__(self) -> int:
        return hash((self.mode, self.values.tobytes()))


def as_selection(selection: Any, mode: str | None, K: int) -> Selection | None:
    """Normalise user input (``None``, a :class:`Selection`, or an array plus mode)."""
    if selection is None:
        if mode is not None:
            raise ValueError("selection_mode was given without a selection function.")
        return None
    if isinstance(selection, Selection):
        if mode is not None and mode != selection.mode:
            raise ValueError("selection_mode conflicts with the mode of the Selection object.")
        sel = selection
    else:
        if mode is None:
            raise ValueError(_HELP)
        sel = Selection(selection, mode)
    if sel.K != K:
        raise ValueError(f"selection must have length K={K}; got {sel.K}.")
    return sel
