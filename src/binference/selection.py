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
from typing import Any, NamedTuple

import numpy as np
from scipy import special

from ._quadrature import bin_log_average
from ._utils import SetOnce, as_generator, check_positive, frozen, warn
from .basis import Basis
from .bins import Bins
from .density import _log_nonneg
from .priors.logistic import _rw_precision
from .warnings import ConvergenceWarning, InputWarning

__all__ = ["SELECTION_MODES", "SELECTION_SCALES", "Selection", "as_selection"]

#: The values of ``Selection.mode``: how objects are selected.
SELECTION_MODES = ("on_truth", "on_measurement", "combined")
#: The values of ``Selection.scale``: what the selection values measure.
SELECTION_SCALES = ("probability", "exposure", "relative")

#: A selection that changes by more than this factor between neighbouring bins raises a
#: warning.
STEEP_RATIO = 2.0

# a Beta draw can underflow to zero, which would void its bin
#: Smallest detection probability a draw of an inferred selection takes.
S_FLOOR = 1e-12

_HELP = (
    "A selection function needs an explicit selection_mode: 'on_measurement' if objects are "
    "selected on their measured data (e.g. a flux limit) or the likelihoods already include "
    "the detection probability; 'on_truth' if detection depends on the true bin independently "
    "of the measured data; Selection.combined(...) for both. See the user guide on selection "
    "effects."
)


class Selection(SetOnce):
    """Per-bin detection probability (fractions) or exposure (rates), with its mode.

    Parameters
    ----------
    values : array_like (K,)
        Strictly positive detection probabilities or exposures per bin.
    mode : {"on_measurement", "on_truth"}
        How selection enters the likelihood (see the module documentation);
        :meth:`combined` builds the third mode.
    scale : {"probability", "exposure", "relative"}
        Whether the values are detection probabilities, exposures or relative
        values. ``"probability"`` (default): values are detection probabilities in
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
        For a selection measured by injections (:meth:`from_injections`), a named
        tuple ``(injected, recovered, prior, smooth)``: counts per bin, the Beta prior
        and the step scale of the smooth model (or ``None``). The selection is
        then uncertain, and inferred jointly with the population (DECISIONS #30,
        #80); ``values`` are its posterior means given the injections alone
        (independent Betas) or the mode of :math:`\\mathrm{logit}\\,S` (smooth).
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
        smooth: float | None = None,
    ) -> Selection:
        """A selection measured by injecting objects of known true value ``z``.

        Counts the injections per bin, :math:`n_k`, and those recovered,
        :math:`d_k`; with a :math:`\\mathrm{Beta}(a, b)` prior each :math:`S_k` has the
        posterior :math:`\\mathrm{Beta}(a + d_k, b + n_k - d_k)` given the
        injections alone. With ``smooth`` the :math:`S_k` are tied together by a
        second-order random walk on :math:`y_k = \\mathrm{logit}\\,S_k`, the log density
        gaining :math:`-\\sum_k (y_{k-1} - 2y_k + y_{k+1})^2 / 2\\tau^2`, which lends
        sparsely injected bins strength from their neighbours.

        A selection on the truth is uncertain: NUTS, MAP (``space="unconstrained"``),
        Laplace and, for independent Betas and a conjugate prior, Gibbs infer it
        jointly with the population, where the catalogue informs the ratios
        between the :math:`S_k` (their level comes from the injections; DECISIONS
        #10, #30, #79, #80); :meth:`fixed` gives a fixed selection at
        :attr:`values`, ignoring the uncertainty. A selection on the measurement
        cannot be inferred (the likelihoods' cut already fixes it), and neither can a
        selection with a rates prior; pass :meth:`fixed` for both.

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
        smooth : float, optional
            Step scale :math:`\\tau` of the random walk, in logit per bin (1-D bins
            only); ``None`` (default) keeps the :math:`S_k` independent.
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
        try:
            k = bins.index(z)
        except ValueError as err:
            raise ValueError(f"z: {err}") from None
        if k.shape != d.shape:
            raise ValueError(f"z has {k.size} injections but detected has {d.size}.")
        try:
            a, b = (float(x) for x in prior)
        except (TypeError, ValueError):
            a = b = np.nan
        if not (a > 0 and b > 0 and np.isfinite(a) and np.isfinite(b)):
            raise ValueError(f"prior must be two positive numbers (a, b); got {prior!r}.")
        if smooth is not None:
            if bins.ndim != 1 or bins.is_categorical or bins.K < 3:
                raise ValueError(
                    "smooth needs ordered 1-D bins, at least three: the random walk of "
                    "second differences runs along them."
                )
            smooth = float(check_positive(smooth, "smooth"))
        inside = k >= 0
        if not inside.all():
            warn(
                f"{int((~inside).sum())} injections fall outside the bins and are ignored.",
                InputWarning,
            )
        n = np.bincount(k[inside], minlength=bins.K).astype(float)
        r = np.bincount(k[inside], weights=d[inside].astype(float), minlength=bins.K)
        injections = _Injections(frozen(n), frozen(r), (a, b), smooth)
        s, f = _beta_posterior(injections)
        S = s / (s + f)  # the posterior means, where the smooth model's Newton steps start
        if smooth is not None:
            S = special.expit(_logit_mode(special.logit(S), injections))
        sel = cls(S, mode)
        object.__setattr__(sel, "injections", injections)
        return sel

    def draw(self, n: int, rng: Any = None) -> np.ndarray:
        """``(n, K)`` posterior draws of :math:`S` given the injections alone
        (independent Betas), floored at :data:`S_FLOOR` so that none underflows to zero.

        Parameters
        ----------
        n : int
            Number of draws.
        rng : int, numpy.random.Generator, or None
            Random number generator.
        """
        if self.injections is None:
            raise ValueError("Only a selection from injections has draws.")
        if self.injections.smooth is not None:
            raise ValueError("Draws are available for independent Betas only (smooth=None).")
        s, f = _beta_posterior(self.injections)
        return _draw_beta(s, f, as_generator(rng), size=(int(n), self.K))

    def fixed(self) -> Selection:
        """This selection taken as known at its :attr:`values`, a plug-in that ignores their
        uncertainty."""
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
        max_depth: int = 20,
    ) -> Selection:
        """A selection given as a function of the true value, averaged over each bin.

        ``function(z)`` returns the detection probability (or exposure) at
        points ``z`` (shape ``(M,)`` in 1-D, ``(M, ndim)`` on grids, labels for
        categories). Each bin takes its average in the measure coordinate, the
        within-bin shape of the population (DECISIONS #10, #47), by adaptive
        quadrature to ``tol`` relative to the largest bin average. A single jump inside a
        bin is found wherever it lies (a window narrower than the spacing of the points can
        still fall between them: put bin edges at its ends), and each refinement halves the
        cell that holds it,
        so reaching ``tol`` takes about :math:`\\log_2(1/\\mathrm{tol})` refinements, 20
        for the default ``tol``, which the default ``max_depth`` allows; for a smaller
        ``tol`` raise ``max_depth``, or put a bin edge at the jump. With a
        :class:`~binference.Basis` for ``bins``, the averages are taken on its
        quadrature cells and weighted by each component (DECISIONS #47, #76).

        Parameters
        ----------
        function : callable
            Detection probability or exposure at points ``z``.
        bins : Bins or Basis
            Bins over which ``function`` is averaged.
        mode : {"on_measurement", "on_truth"}
            How selection enters the likelihood, as for :class:`Selection`.
        scale : {"probability", "exposure", "relative"}, optional
            Whether the values are detection probabilities, exposures or relative
            values, as for :class:`Selection`.
        tol : float, optional
            Quadrature tolerance relative to the largest bin average.
        max_depth : int, optional
            Most refinements of the adaptive quadrature; each halves a cell.
        """
        _check_mode(mode)
        _check_values(np.ones(2), scale, "selection")  # the scale, before any quadrature
        quad = {"tol": tol, "max_depth": max_depth}
        if isinstance(bins, Basis):
            sel = cls.from_function(function, bins.grid, mode, scale=scale, **quad)
            return _on_basis(sel, bins, bins.grid_masses)
        return cls(_bin_average(function, bins, **quad)[0], mode, scale=scale)

    @classmethod
    def combined(
        cls,
        on_truth: Any,
        on_measurement: Any,
        bins: Bins | None = None,
        *,
        scale: str = "probability",
        tol: float = 1e-6,
        max_depth: int = 20,
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
            Bins over which callable parts are averaged, in the measure coordinate;
            needed when either part is a callable (when both are, the product is
            averaged as a whole). With a :class:`~binference.Basis` the result
            is per component, as for :meth:`from_function`.
        scale : {"probability", "exposure", "relative"}, optional
            Whether the values of both parts are detection probabilities, exposures
            or relative values, as for :class:`Selection`.
        tol : float, optional
            Quadrature tolerance for callable parts (see :meth:`from_function`).
        max_depth : int, optional
            Most refinements of the adaptive quadrature for callable parts.
        """
        _check_values(np.ones(2), scale, "selection")  # the scale, before any quadrature
        quad = {"tol": tol, "max_depth": max_depth}
        if isinstance(bins, Basis):  # on the quadrature cells, then per component
            parts = [
                p if callable(p) else _per_bin(p, bins.K)[bins.grid_to_bins]
                for p in (on_truth, on_measurement)
            ]
            sel = cls.combined(*parts, bins.grid, scale=scale, **quad)
            return _on_basis(sel, bins, bins.grid_masses)
        if callable(on_truth) and callable(on_measurement):
            if bins is None:
                raise ValueError("Selection.combined needs bins when a part is a callable.")
            C, CT = _bin_average(on_truth, bins, on_measurement, **quad)
            C = _check_values(C, scale, "on_truth")
            return _combined(C, CT / C, scale)
        parts = []
        for part in (on_truth, on_measurement):
            if callable(part):
                if bins is None:
                    raise ValueError("Selection.combined needs bins when a part is a callable.")
                part = _bin_average(part, bins, **quad)[0]  # the other is constant in bins
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
            and _same_injections(self.injections, other.injections)
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


def _same_injections(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    return np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


class _Injections(NamedTuple):
    """Injection counts per bin and the model of :math:`S` given them."""

    injected: np.ndarray
    recovered: np.ndarray
    prior: tuple[float, float]
    smooth: float | None


def _beta_posterior(injections: _Injections) -> tuple[np.ndarray, np.ndarray]:
    """Parameters ``(s, f)`` of the Beta posteriors of :math:`S` given the injections
    alone: :math:`s = a + d`, :math:`f = b + n - d`."""
    n, r, (a, b), _ = injections
    return a + r, b + n - r


def _draw_beta(s: Any, f: Any, rng: np.random.Generator, size: Any = None) -> np.ndarray:
    r"""Draws of :math:`S \sim \mathrm{Beta}(s, f)`, floored at :data:`S_FLOOR`."""
    return np.maximum(rng.beta(s, f, size=size), S_FLOOR)


def _logit_density(y: np.ndarray, injections: _Injections) -> tuple[float, np.ndarray]:
    r"""Log density of :math:`y = \mathrm{logit}\,S` given the injections alone, with
    the Jacobian, and its gradient: :math:`\sum_k s_k\log S_k + f_k\log(1 - S_k)` with
    :math:`s = a + d`, :math:`f = b + n - d`, plus the random walk of the smooth model."""
    s, f = _beta_posterior(injections)
    tau = injections.smooth
    lp = float(s @ special.log_expit(y) + f @ special.log_expit(-y))
    g = s - (s + f) * special.expit(y)
    if tau is not None:
        d2 = np.diff(y, 2)
        lp -= 0.5 * float(d2 @ d2) / tau**2
        g -= np.convolve(d2, [1.0, -2.0, 1.0]) / tau**2  # D2^T d2: the stencil is symmetric
    return lp, g


def _logit_mode(y: np.ndarray, injections: _Injections) -> np.ndarray:
    """The mode of :func:`_logit_density` (strictly concave), by damped Newton steps from ``y``."""
    s, f = _beta_posterior(injections)
    Q = _rw_precision(s.size, 2) / injections.smooth**2
    lp, g = _logit_density(y, injections)
    for _ in range(100):
        S = special.expit(y)
        step = np.linalg.solve(np.diag((s + f) * S * (1.0 - S)) + Q, g)
        t = 1.0
        while t > 1e-10:
            new_lp, new_g = _logit_density(y + t * step, injections)
            if new_lp >= lp:
                break
            t /= 2
        else:
            break
        y, lp, g = y + t * step, new_lp, new_g
        if np.max(np.abs(t * step)) < 1e-10:
            return y
    warn(
        f"The smooth selection's mode did not converge (gradient {np.max(np.abs(g)):.1e}).",
        ConvergenceWarning,
    )
    return y


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
            "are per bin. Use plain bins, or sel.fixed() for a fixed selection."
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
    out.injections = (
        None
        if injections is None
        else _Injections(
            frozen(np.array(injections[0])),
            frozen(np.array(injections[1])),
            tuple(injections[2]),
            injections[3],
        )
    )
    return out


def _check_values(values: Any, scale: str, name: str) -> np.ndarray:
    if scale not in SELECTION_SCALES:
        raise ValueError(f"scale must be one of {SELECTION_SCALES}; got {scale!r}.")
    v = np.array(values, dtype=float).ravel()
    if v.size < 2:
        raise ValueError(f"{name} needs one value per bin.")
    bad = np.flatnonzero(~(np.isfinite(v) & (v > 0)))
    if bad.size:
        raise ValueError(
            f"{name} values must be finite and strictly positive; bin(s) {bad[:5].tolist()} "
            "are not (bins that can never be observed are not identifiable; remove them)."
        )
    if scale == "probability" and np.any(v > 1.0 + 1e-12):
        raise ValueError(
            f"{name} values are detection probabilities and must not exceed one; for "
            "absolute exposures pass Selection(S, mode, scale='exposure'), for values known "
            "only up to a constant Selection(S, mode, scale='relative') (DECISIONS #46)."
        )
    return v


def _check_mode(mode: str) -> None:
    if mode not in SELECTION_MODES[:2]:  # the third, "combined", is Selection.combined's
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
    max_depth: int = 12,
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
        return _log_nonneg(np.ravel(f(z)), "The values of a selection function")

    def logf(z):
        lf = logs(function, z)
        rows = [lf] if factor is None else [lf, lf + logs(factor, z)]
        if log_weight is not None:
            lw = np.asarray(log_weight(z), dtype=float).ravel()
            rows = [r + lw for r in rows] + [lw]
        return np.stack(rows)

    quad = {"tol": tol, "max_depth": max_depth}
    if log_weight is None:  # bounded, so the check for jumps converges (bin_log_average)
        return np.exp(bin_log_average(logf, bins, jumps=True, **quad))
    out = bin_log_average(logf, bins, coordinate="z", **quad)
    with np.errstate(invalid="ignore"):
        avg = np.exp(out[:-1] - out[-1])  # divide by the bin's weight
    empty = ~np.isfinite(out[-1])
    if empty.any():  # no population in the bin: fall back to the unweighted average
        avg[:, empty] = _bin_average(function, bins, factor, **quad)[:, empty]
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

    Parameters
    ----------
    selection : None, Selection, callable or array_like
        The selection as the user gave it.
    mode : str or None
        ``selection_mode``, required with a callable or an array.
    bins : Bins or Basis
        Bins of the model.
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
