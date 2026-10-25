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
from ._utils import as_generator, frozen, warn
from .basis import Basis
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

    Attributes
    ----------
    values : ndarray (K,)
        :math:`S_k`, the probability that an object in bin *k* is selected.
    row : ndarray (K,) or None
        The factor applied to the likelihood columns (:math:`S_k` on the
        truth, :math:`C_k` combined, ``None`` on the measurement).
    basis : Basis or None
        Set when the values are per component of a :class:`~binference.Basis`
        (:math:`\\phi_k`-weighted averages) rather than per bin.
    injections : tuple or None
        For a selection measured by injections (:meth:`from_injections`):
        ``(injected, recovered, (a, b))``, counts per bin and the Beta prior.
        The selection is then uncertain: ``values`` are its posterior means,
        and Gibbs sampling infers it jointly with the population (DECISIONS #30).
    """

    __slots__ = ("basis", "injections", "row", "scale", "values")

    def __init__(self, values: Any, mode: str, *, scale: str = "probability") -> None:
        _check_mode(mode)
        v = _check_values(values, scale, "selection")
        self.values = frozen(v)
        self.row = self.values if mode == "on_truth" else None
        self.scale = scale
        self.basis = None
        self.injections = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name in Selection.__slots__ and not hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError("Selection objects are immutable.")

    def _parts(self) -> tuple:
        # the arguments of _rebuild; row as "values" when it is the values (on the truth)
        row = "values" if self.row is self.values else self.row
        return self.values, row, self.scale, self.basis, self.injections

    def __reduce__(self) -> tuple:
        return _rebuild, self._parts()

    @classmethod
    def from_injections(
        cls,
        z: Any,
        detected: Any,
        bins: Bins,
        mode: str,
        *,
        prior: tuple[float, float] = (1.0, 1.0),
    ) -> Selection:
        """A selection measured by injecting objects of known true value ``z``.

        Counts the injections per bin, :math:`n_k`, and those recovered,
        :math:`d_k`; with a :math:`\\mathrm{Beta}(a, b)` prior each :math:`S_k` has the
        posterior :math:`\\mathrm{Beta}(a + d_k, b + n_k - d_k)` given the
        injections alone. The selection is uncertain: Gibbs sampling infers it
        jointly with the population, where the catalogue (and the objects it
        missed) inform it as well (DECISIONS #10, #30); :meth:`at_mean` gives a
        fixed selection at the posterior means, ignoring the uncertainty.

        Parameters
        ----------
        z : array_like, shape (n,) or (n, ndim)
            True values of the injected objects.
        detected : array_like of bool, shape (n,)
            Whether each injected object was recovered.
        bins : Bins
            The bins of the population.
        mode : {"on_truth", "on_measurement"}
            As for :class:`Selection`.
        prior : (float, float)
            Beta prior :math:`(a, b)` of every :math:`S_k`, both positive.
        """
        _check_mode(mode)
        if isinstance(bins, Basis):
            raise ValueError(
                "from_injections needs plain Bins: a component of a Basis is not a set of "
                "injections, so its Beta posterior would not be conjugate."
            )
        d = np.asarray(detected)
        if d.ndim != 1 or not (d.dtype == bool or np.isin(d, (0, 1)).all()):
            raise ValueError("detected must be a 1-D array of booleans (or zeros and ones).")
        k = bins.index(z)
        if k.shape != d.shape:
            raise ValueError(f"z has {k.size} injections but detected has {d.size}.")
        a, b = (float(x) for x in prior)
        if not (a > 0 and b > 0 and np.isfinite(a) and np.isfinite(b)):
            raise ValueError(f"prior must be two positive numbers (a, b); got {prior!r}.")
        inside = k >= 0
        if not inside.all():
            warn(
                f"{int((~inside).sum())} injections fall outside the bins and are ignored.",
                InputWarning,
            )
        n = np.bincount(k[inside], minlength=bins.K).astype(float)
        r = np.bincount(k[inside], weights=d[inside].astype(float), minlength=bins.K)
        sel = cls((a + r) / (a + b + n), mode)
        object.__setattr__(sel, "injections", (frozen(n), frozen(r), (a, b)))
        return sel

    def draw(self, n: int, rng: Any = None) -> np.ndarray:
        """``(n, K)`` posterior draws of :math:`S` given the injections alone."""
        if self.injections is None:
            raise ValueError("Only a selection from injections has draws.")
        inj, rec, (a, b) = self.injections
        return as_generator(rng).beta(a + rec, b + inj - rec, size=(int(n), self.K))

    def at_mean(self) -> Selection:
        """This selection fixed at its values (the posterior means for injections),
        ignoring their uncertainty."""
        return _rebuild(*self._parts()[:4])

    @property
    def mode(self) -> str:
        """``"on_measurement"``, ``"on_truth"`` or ``"combined"``, from :attr:`row`."""
        if self.row is None:
            return "on_measurement"
        return "on_truth" if self.row is self.values else "combined"

    @classmethod
    def from_function(
        cls,
        function: Callable[[np.ndarray], Any],
        bins: Bins,
        mode: str,
        *,
        scale: str = "probability",
        tol: float = 1e-6,
    ) -> Selection:
        """A selection given as a function of the true value, averaged over each bin.

        ``function(z)`` returns the detection probability (or exposure) at
        points ``z`` (shape ``(M,)`` in 1-D, ``(M, ndim)`` on grids, labels for
        categories). Each bin takes its average in the measure coordinate, the
        within-bin shape of the population (DECISIONS #10, #47), by adaptive
        quadrature to ``tol`` relative to the largest bin average. With a
        :class:`~binference.Basis` for ``bins``, the averages are taken on its
        quadrature cells and weighted by each component (DECISIONS #47, #76).
        """
        _check_mode(mode)
        _check_values(np.ones(2), scale, "selection")  # the scale, before any quadrature
        if isinstance(bins, Basis):
            sel = cls.from_function(function, bins.grid, mode, scale=scale, tol=tol)
            return _on_basis(sel, bins, bins.grid_masses)
        return cls(_bin_average(function, bins, tol=tol)[0], mode, scale=scale)

    @classmethod
    def combined(
        cls,
        on_truth: Any,
        on_measurement: Any,
        bins: Bins | None = None,
        *,
        scale: str = "probability",
        tol: float = 1e-6,
    ) -> Selection:
        """Selection on the truth and on the measurement at once (DECISIONS #8).

        Parameters
        ----------
        on_truth : array_like (K,) or callable
            :math:`C`, the detection probability given the true value.
        on_measurement : array_like (K,) or callable
            :math:`P(d \\in A \\mid z)`, the probability that the measured data
            pass the cut given the true value.
        bins : Bins or Basis, optional
            Needed when either part is a callable, which is then averaged over
            each bin in the measure coordinate (when both are, the product is
            averaged as a whole). With a :class:`~binference.Basis` the result
            is per component, as for :meth:`from_function`.
        scale : str
            As for :class:`Selection`, applied to both parts.
        tol : float
            Quadrature tolerance for callable parts (see :meth:`from_function`).
        """
        if isinstance(bins, Basis):  # on the quadrature cells, then per component
            parts = [
                p if callable(p) else _per_bin(p, bins.K)[bins.grid_to_bins]
                for p in (on_truth, on_measurement)
            ]
            sel = cls.combined(*parts, bins.grid, scale=scale, tol=tol)
            return _on_basis(sel, bins, bins.grid_masses)
        if callable(on_truth) and callable(on_measurement):
            if bins is None:
                raise ValueError("Selection.combined needs bins when a part is a callable.")
            C, CT = _bin_average(on_truth, bins, on_measurement, tol=tol)
            C = _check_values(C, scale, "on_truth")
            return _combined(C, CT / C, scale)
        parts = []
        for part in (on_truth, on_measurement):
            if callable(part):
                if bins is None:
                    raise ValueError("Selection.combined needs bins when a part is a callable.")
                part = _bin_average(part, bins, tol=tol)[0]  # the other is constant in bins
            parts.append(part)
        return _combined(*parts, scale)

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
            and (self.row is None or np.array_equal(self.row, other.row))
            and (self.injections is None) == (other.injections is None)
        )

    def __hash__(self) -> int:
        return hash((self.mode, self.scale, self.values.tobytes()))


def _combined(on_truth: Any, on_measurement: Any, scale: str) -> Selection:
    # rows scaled by the truth part, the normaliser by the product
    C = _check_values(on_truth, scale, "on_truth")
    T = _check_values(on_measurement, scale, "on_measurement")
    if T.shape != C.shape:
        raise ValueError("on_truth and on_measurement must have one value per bin.")
    return _rebuild(C * T, C, scale)


def _per_bin(values: Any, K: int) -> np.ndarray:
    v = np.asarray(values, dtype=float).ravel()
    if v.size != K:
        raise ValueError(f"selection must have length K={K}; got {v.size}.")
    return v


def _on_basis(sel: Selection, basis: Basis, M: Any) -> Selection:
    """``sel`` given per cell of a grid (the rows of ``M``), as averages per component."""
    if sel.injections is not None:
        raise ValueError(
            "A selection from injections cannot be moved onto a Basis: its Beta posteriors "
            "are per bin. Use plain bins, or sel.at_mean() for a fixed selection."
        )
    row = None if sel.row is None else "values" if sel.row is sel.values else M.T @ sel.row
    return _rebuild(M.T @ sel.values, row, sel.scale, basis)


def _rebuild(
    values: np.ndarray, row: Any, scale: str, basis: Basis | None = None, injections: Any = None
) -> Selection:
    # a Selection from its arrays; row is None, "values" (on the truth) or an array
    out = Selection.__new__(Selection)
    out.values = frozen(np.array(values, dtype=float))
    out.row = (
        out.values
        if isinstance(row, str)
        else None
        if row is None
        else frozen(np.array(row, dtype=float))
    )
    out.scale = scale
    out.basis = basis
    out.injections = injections
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
            f"{name} values are detection probabilities and must not exceed one; for "
            "absolute exposures pass Selection(S, mode, scale='exposure'), for values known "
            "only up to a constant Selection(S, mode, scale='relative') (DECISIONS #46)."
        )
    return v


def _check_mode(mode: str) -> None:
    if mode in _RENAMED:
        raise ValueError(f"selection mode {mode!r} is now {_RENAMED[mode]!r} (DECISIONS #7).")
    if mode not in ("on_truth", "on_measurement"):
        raise ValueError(
            "selection mode must be 'on_measurement' or 'on_truth' (or use "
            f"Selection.combined); got {mode!r}. {_HELP}"
        )


def _bin_average(
    function: Callable,
    bins: Bins,
    factor: Callable | None = None,
    log_weight: Callable | None = None,
    tol: float = 1e-6,
) -> np.ndarray:
    """Weighted bin averages of a non-negative ``function`` (and of ``function * factor``).

    Returns an array with one row: the average of ``function`` over each bin,
    and a second row with the average of ``function * factor`` when ``factor``
    is given, both from one adaptive quadrature. The weight is uniform in the
    measure coordinate (the within-bin population shape), or ``exp(log_weight(z))``
    per unit ``z`` when given (a population density); a bin where that weight
    vanishes takes the unweighted average.
    """
    if bins.is_categorical:
        labels = np.asarray(bins.labels, dtype=object)
        f = np.asarray(function(labels), dtype=float).ravel()
        return f[None] if factor is None else np.stack([f, f * np.ravel(factor(labels))])

    def logs(f, z):
        v = np.asarray(f(z), dtype=float).ravel()
        if not np.all(np.isfinite(v)) or np.any(v < 0):
            raise ValueError("A selection function must return finite, non-negative values.")
        with np.errstate(divide="ignore"):
            return np.log(v)

    def logf(z):
        lf = logs(function, z)
        rows = [lf] if factor is None else [lf, lf + logs(factor, z)]
        if log_weight is not None:
            lw = np.asarray(log_weight(z), dtype=float).ravel()
            rows = [r + lw for r in rows] + [lw]
        return np.stack(rows)

    if log_weight is None:
        return np.exp(bin_log_average(logf, bins, tol=tol))
    out = bin_log_average(logf, bins, coordinate="z", tol=tol)
    with np.errstate(invalid="ignore"):
        avg = np.exp(out[:-1] - out[-1])  # divide by the bin's weight
    empty = ~np.isfinite(out[-1])
    if empty.any():  # no population in the bin: fall back to the unweighted average
        avg[:, empty] = _bin_average(function, bins, factor, tol=tol)[:, empty]
    return avg


def _warn_if_steep(values: np.ndarray, bins: Bins) -> None:
    # within-bin variation of S biases the fractions when S changes steeply (DECISIONS #47)
    if bins.is_categorical or bins.K < 2:
        return
    pairs = bins.neighbors()
    worst = float(np.abs(np.diff(np.log(values)[pairs], axis=1)).max(initial=0.0))
    if worst > np.log(STEEP_RATIO):
        warn(
            f"The selection changes by a factor of {np.exp(worst):.3g} between neighbouring "
            "bins; its variation within a bin then biases the fractions there. Pass S as a "
            "function of the true value (averaged within bins), use narrower bins, or pass a "
            "Selection built from bin averages.",
            InputWarning,
        )


def as_selection(selection: Any, mode: str | None, bins: Bins | Basis) -> Selection | None:
    """Normalise user input: ``None``, a :class:`Selection`, a callable or an array plus mode.

    A raw array is checked for steep changes between neighbouring bins; a
    :class:`Selection` or a callable is taken as already averaged within bins.
    With a :class:`~binference.Basis`, arrays and selections given per bin are
    mapped to the components (:math:`\\phi_k`-weighted averages) and callables are
    averaged on the basis's quadrature cells.
    """
    if selection is None:
        if mode is not None:
            raise ValueError("selection_mode was given without a selection function.")
        return None
    basis = bins if isinstance(bins, Basis) else None
    report = bins if basis is None else basis.bins
    if isinstance(selection, Selection):
        if mode is not None and mode != selection.mode:
            raise ValueError("selection_mode conflicts with the mode of the Selection object.")
        if selection.basis is not None:
            if selection.basis != basis:
                raise ValueError("The Selection was built for a different basis.")
            return selection
        sel = selection
    elif mode is None:
        raise ValueError(_HELP)
    elif callable(selection):
        return Selection.from_function(selection, bins, mode)
    else:
        sel = Selection(selection, mode)
        if sel.K == report.K:
            _warn_if_steep(sel.values, report)
    if sel.K != report.K:
        raise ValueError(f"selection must have length K={report.K}; got {sel.K}.")
    return sel if basis is None else _on_basis(sel, basis, basis.matrix)
