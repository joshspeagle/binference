r"""Selection functions (detection probabilities or exposures per bin).

Selection enters the likelihood in one of three ways, and choosing the wrong one
biases the population (Mandel, Farr & Gair 2019). binference therefore requires
the mode to be stated explicitly (DECISIONS #7, #8):

``mode="on_measurement"``
    Objects are selected on their *measured data* (e.g. a flux or
    signal-to-noise cut), or the per-object likelihoods already include the
    detection probability (e.g. responses that fold in efficiencies). The
    likelihood of each detected object is used as is, and only the
    normalisation :math:`(S\cdot\rho)^{-W}` (fractions) or
    :math:`e^{-S\cdot\lambda}` (rates) is added. Here :math:`S_k` is the
    probability that an object in bin *k* would be selected.

``mode="on_truth"``
    Objects are detected with probability :math:`S_k` that depends on their
    *true* bin, independently of their measured data given the truth (e.g. a
    completeness that depends on true planet radius). Each object's
    likelihood row is multiplied by :math:`S_k` *and* the normalisation is
    added.

``Selection.combined(on_truth=C, on_measurement=T)``
    Both at once, :math:`P(\text{obs} \mid z, d) = C(z)\,T(d)`: rows are
    multiplied by :math:`C_k` only, and the normalisation uses
    :math:`S_k`, the bin average of :math:`C(z)\,P(d \in A \mid z)`.

Values are detection probabilities in (0, 1] by default; ``scale="exposure"``
allows any positive absolute values (rates), and ``scale="relative"`` declares
values known only up to a constant, which is enough for fractions but not for
totals or rates (DECISIONS #46).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ._quadrature import bin_log_average
from ._utils import frozen, warn
from .bins import Bins
from .warnings import InputWarning

__all__ = ["SELECTION_MODES", "SELECTION_SCALES", "Selection", "as_selection"]

SELECTION_MODES = ("on_truth", "on_measurement", "combined")
SELECTION_SCALES = ("probability", "exposure", "relative")
_RENAMED = {"true": "on_truth", "observed": "on_measurement"}

#: neighbouring bins whose selection differs by more than this factor raise a warning
STEEP_RATIO = 2.0

_HELP = (
    "A selection function needs an explicit selection_mode: 'on_measurement' if objects are "
    "selected on their measured data (e.g. a flux limit) or the likelihoods already include "
    "the detection probability; 'on_truth' if detection depends on the true bin independently "
    "of the measured data; Selection.combined(...) for both. See the user guide on selection "
    "effects."
)


class Selection:
    """Per-bin detection probability (fractions) or exposure (rates), with its mode.

    Parameters
    ----------
    values : array_like (K,)
        Strictly positive detection probabilities or exposures per bin.
    mode : {"on_measurement", "on_truth"}
        How selection enters the likelihood (see the module documentation);
        :meth:`combined` builds the third mode.
    scale : {"probability", "exposure", "relative"}
        ``"probability"`` (default): values are detection probabilities in
        (0, 1]. ``"exposure"``: absolute positive values, such as the exposure
        of a rate model. ``"relative"``: only ratios between bins are known,
        which suffices for fractions but not for rates or totals.
    averaged : bool
        Whether the values are already the population-weighted averages over
        each bin (from a function, a simulation, or because selection is
        constant within bins). Otherwise a warning is raised where the values
        change steeply between neighbouring bins (DECISIONS #47).

    Attributes
    ----------
    values : ndarray (K,)
        :math:`S_k`, the probability that an object in bin *k* is selected.
    row : ndarray (K,) or None
        The factor applied to the likelihood columns (:math:`S_k` on the
        truth, :math:`C_k` combined, ``None`` on the measurement).
    """

    __slots__ = ("averaged", "mode", "row", "scale", "values")

    def __init__(
        self, values: Any, mode: str, *, scale: str = "probability", averaged: bool = False
    ) -> None:
        if mode in _RENAMED:
            raise ValueError(f"selection mode {mode!r} is now {_RENAMED[mode]!r} (DECISIONS #7).")
        if mode not in ("on_truth", "on_measurement"):
            raise ValueError(
                f"selection mode must be 'on_measurement' or 'on_truth' (or use "
                f"Selection.combined); got {mode!r}. {_HELP}"
            )
        v = _check_values(values, scale, "selection")
        self._set(v, mode, v if mode == "on_truth" else None, scale, averaged)

    def _set(self, values, mode, row, scale, averaged) -> None:
        self.values = frozen(values)
        self.mode = mode
        self.row = None if row is None else frozen(np.array(row, dtype=float))
        self.scale = scale
        self.averaged = bool(averaged)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in Selection.__slots__ and not hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError("Selection objects are immutable.")

    def __reduce__(self) -> tuple:
        return _rebuild, (self.values, self.mode, self.row, self.scale, self.averaged)

    @classmethod
    def from_function(
        cls,
        function: Callable[[np.ndarray], Any],
        bins: Bins,
        mode: str,
        *,
        scale: str = "probability",
    ) -> Selection:
        """A selection given as a function of the true value, averaged over each bin.

        ``function(z)`` returns the detection probability (or exposure) at
        points ``z`` (shape ``(M,)`` in 1-D, ``(M, ndim)`` on grids, labels for
        categories). Each bin takes its average in the measure coordinate, the
        within-bin shape of the population (DECISIONS #10, #47).
        """
        return cls(_bin_average(function, bins), mode, scale=scale, averaged=True)

    @classmethod
    def combined(
        cls,
        on_truth: Any,
        on_measurement: Any,
        bins: Bins | None = None,
        *,
        scale: str = "probability",
        averaged: bool = False,
    ) -> Selection:
        """Selection on the truth and on the measurement at once (DECISIONS #8).

        Parameters
        ----------
        on_truth : array_like (K,) or callable
            :math:`C`, the detection probability given the true value.
        on_measurement : array_like (K,) or callable
            :math:`P(d \\in A \\mid z)`, the probability that the measured data
            pass the cut given the true value.
        bins : Bins, optional
            Needed when either part is a callable; both are then averaged over
            each bin in the measure coordinate, the product as a whole.
        scale, averaged : str, bool
            As for :class:`Selection`, applied to both parts (callables are
            always averaged).
        """
        if callable(on_truth) or callable(on_measurement):
            if bins is None:
                raise ValueError("Selection.combined needs bins when a part is a callable.")
            C = on_truth if callable(on_truth) else _piecewise(on_truth, bins)
            T = on_measurement if callable(on_measurement) else _piecewise(on_measurement, bins)
            row = _check_values(_bin_average(C, bins), scale, "on_truth")
            values = _check_values(_bin_average(lambda z: C(z) * T(z), bins), scale, "selection")
            averaged = True
        else:
            row = _check_values(on_truth, scale, "on_truth")
            T = _check_values(on_measurement, scale, "on_measurement")
            if T.shape != row.shape:
                raise ValueError("on_truth and on_measurement must have one value per bin.")
            values = row * T
        out = cls.__new__(cls)
        out._set(values, "combined", row, scale, averaged)
        return out

    @property
    def K(self) -> int:
        return int(self.values.size)

    def __repr__(self) -> str:
        return f"Selection(K={self.K}, mode={self.mode!r}, scale={self.scale!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Selection):
            return NotImplemented
        return (
            (self.mode, self.scale) == (other.mode, other.scale)
            and np.array_equal(self.values, other.values)
            and (self.row is None) == (other.row is None)
            and (self.row is None or np.array_equal(self.row, other.row))
        )

    def __hash__(self) -> int:
        return hash((self.mode, self.scale, self.values.tobytes()))


def _rebuild(values, mode, row, scale, averaged) -> Selection:
    out = Selection.__new__(Selection)
    out._set(np.array(values, dtype=float), mode, row, scale, averaged)
    return out


def _check_values(values: Any, scale: str, name: str) -> np.ndarray:
    if scale not in SELECTION_SCALES:
        raise ValueError(f"scale must be one of {SELECTION_SCALES}; got {scale!r}.")
    v = np.array(values, dtype=float).ravel()
    if v.size < 2:
        raise ValueError(f"{name} needs one value per bin.")
    if not np.all(np.isfinite(v)) or np.any(v <= 0):
        raise ValueError(
            f"{name} values must be finite and strictly positive (bins that can never be "
            "observed are not identifiable; remove them from the bins)."
        )
    if scale == "probability" and np.any(v > 1.0 + 1e-12):
        raise ValueError(
            f"{name} values are detection probabilities and must not exceed one; pass "
            "scale='exposure' for absolute exposures or scale='relative' for values known "
            "only up to a constant (DECISIONS #46)."
        )
    return v


def _bin_average(function: Callable, bins: Bins) -> np.ndarray:
    # each bin's average of a non-negative function, uniform in the measure coordinate
    if bins.is_categorical:
        return np.asarray(function(np.asarray(bins.labels, dtype=object)), dtype=float).ravel()

    def logf(z):
        v = np.asarray(function(z), dtype=float).ravel()
        if not np.all(np.isfinite(v)) or np.any(v < 0):
            raise ValueError("A selection function must return finite, non-negative values.")
        with np.errstate(divide="ignore"):
            return np.log(v)[None, :]

    return np.exp(bin_log_average(logf, bins, coordinate="measure")[0])


def _piecewise(values: Any, bins: Bins) -> Callable[[np.ndarray], np.ndarray]:
    # an array of per-bin values as a function of z, constant within each bin
    v = np.asarray(values, dtype=float).ravel()
    return lambda z: v[bins.index(z)]


def _warn_if_steep(values: np.ndarray, bins: Bins) -> None:
    # within-bin variation of S biases the fractions when S changes steeply (DECISIONS #47)
    if bins.is_categorical:
        return
    grid = np.full(bins.mask.shape, np.nan)
    grid[bins.cell_index] = np.log(values)
    jumps = [np.abs(np.diff(grid, axis=d)) for d in range(grid.ndim)]
    worst = max(float(np.nanmax(j, initial=0.0)) for j in jumps)
    if worst > np.log(STEEP_RATIO):
        warn(
            f"The selection changes by a factor of {np.exp(worst):.3g} between neighbouring "
            "bins; its variation within a bin then biases the fractions there. Pass S as a "
            "function of the true value (averaged within bins), or use narrower bins.",
            InputWarning,
        )


def as_selection(selection: Any, mode: str | None, bins: Bins) -> Selection | None:
    """Normalise user input: ``None``, a :class:`Selection`, a callable or an array plus mode."""
    if selection is None:
        if mode is not None:
            raise ValueError("selection_mode was given without a selection function.")
        return None
    if isinstance(selection, Selection):
        if mode is not None and mode != selection.mode:
            raise ValueError("selection_mode conflicts with the mode of the Selection object.")
        sel = selection
    elif mode is None:
        raise ValueError(_HELP)
    elif callable(selection):
        sel = Selection.from_function(selection, bins, mode)
    else:
        sel = Selection(selection, mode)
    if sel.K != bins.K:
        raise ValueError(f"selection must have length K={bins.K}; got {sel.K}.")
    if not sel.averaged:
        _warn_if_steep(sel.values, bins)
    return sel
