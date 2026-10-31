"""Fixed densities over the bin coordinate: interim priors, base measures, simulation truth.

A :class:`Density` is one fixed probability density :math:`\\pi(z)` with a single
meaning wherever it appears (DECISIONS #26). It is resolved against bins with
:meth:`Density.masses`, which returns the normalised mass :math:`\\Pi_k` of each bin.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from ._quadrature import _gauss, bin_log_average
from ._utils import check_nonnegative, frozen
from .bins import Bins

__all__ = ["Density"]

Representation = Literal["mass", "density"]


class Density:
    """A fixed probability density over the bin coordinate.

    Build one with :meth:`flat`, :meth:`from_masses`, :meth:`from_densities`,
    :meth:`from_function` or :meth:`from_distribution`. Wherever a density is
    expected, the string ``"flat"``, a callable, a distribution object and an
    array are also accepted (see :meth:`coerce`).
    """

    # exactly one of _flat ("measure" or "z"), _values (per-bin array, read as
    # masses, densities or, when _per_volume is None, either on equal-volume bins),
    # _func (callable density, _bounded if it is) or _dist (distribution object) is set
    __slots__ = ("_bounded", "_dist", "_flat", "_func", "_per_volume", "_values")

    def __init__(
        self, *, flat=None, values=None, per_volume=False, func=None, dist=None, bounded=False
    ):
        if sum(x is not None for x in (flat, values, func, dist)) != 1:
            raise TypeError(
                "Build a Density with flat, from_masses, from_densities, from_function, "
                "from_distribution or coerce."
            )
        self._flat = flat
        self._values = values
        self._per_volume = per_volume
        self._func = func
        self._bounded = bounded
        self._dist = dist

    # ------------------------------------------------------------ constructors
    @classmethod
    def flat(cls, coordinate: Literal["measure", "z"] = "measure") -> Density:
        """Uniform in the measure coordinate ``u`` of each axis (default) or in ``z``.

        On linear bins the two agree. On log bins, flat in ``u`` gives equal
        mass per decade. On categories, every category has equal mass.

        Parameters
        ----------
        coordinate : {"measure", "z"}, optional
            Flat in the measure coordinate ``u`` or in ``z``.
        """
        if coordinate not in ("measure", "z"):
            raise ValueError(f"coordinate must be 'measure' or 'z'; got {coordinate!r}.")
        return cls(flat=coordinate)

    @classmethod
    def from_masses(cls, masses: Any) -> Density:
        """Probability mass per bin (normalised on use).

        Parameters
        ----------
        masses : array_like (K,)
            Non-negative mass per bin.
        """
        return cls(values=_nonneg(masses, "masses"))

    @classmethod
    def from_densities(cls, densities: Any) -> Density:
        """Density per unit ``z``-volume in each bin; masses are densities times volumes.

        Parameters
        ----------
        densities : array_like (K,)
            Non-negative density per bin.
        """
        return cls(values=_nonneg(densities, "densities"), per_volume=True)

    @classmethod
    def from_function(
        cls, func: Callable[[np.ndarray], np.ndarray], *, bounded: bool = False
    ) -> Density:
        """A callable density of ``z``, averaged over each bin by quadrature.

        On grids ``func`` receives points of shape ``(M,)`` (1-D) or
        ``(M, ndim)``; on categories it receives the category labels.

        Parameters
        ----------
        func : callable
            Non-negative density of ``z``.
        bounded : bool, optional
            ``func`` is bounded, such as a box: each cell is also checked for jumps near
            its edges, as a selection function's is (see :meth:`Likelihoods.from_density
            <binference.Likelihoods.from_density>`).
        """
        if not callable(func):
            raise TypeError("func must be callable.")
        return cls(func=func, bounded=bounded)

    @classmethod
    def from_distribution(cls, dist: Any) -> Density:
        """A distribution object with ``cdf``, ``logpdf`` or ``pdf`` (e.g. frozen ``scipy.stats``).

        On 1-D grids a ``cdf`` gives exact bin masses; otherwise the density is
        averaged over each bin by quadrature.

        Parameters
        ----------
        dist : object
            A distribution with ``cdf``, ``logpdf`` or ``pdf``.
        """
        if not _is_distribution(dist):
            raise TypeError("dist needs a cdf, logpdf or pdf method.")
        return cls(dist=dist)

    @classmethod
    def coerce(
        cls,
        value: Any,
        representation: Representation | None = None,
        name: str = "interim",
        bounded: bool = False,
    ) -> Density:
        """A :class:`Density` from a density, ``"flat"``, a callable, a distribution or an array.

        An array is read as masses or densities by ``representation`` (which
        applies to arrays only). Without it the array is accepted only on bins
        where both readings agree (categories and bins of equal volume);
        :meth:`masses` raises otherwise.

        Parameters
        ----------
        value : Density, "flat", callable, distribution or array_like (K,)
            What to read as a density.
        representation : {"mass", "density"}, optional
            Whether an array holds masses or densities.
        name : str, optional
            Name of the argument, used in error messages.
        bounded : bool, optional
            A callable ``value`` is bounded (see :meth:`from_function`).
        """
        if representation not in (None, "mass", "density"):
            raise ValueError(f"{name} representation must be 'mass' or 'density'.")
        if isinstance(value, Density):
            return value
        if isinstance(value, str):
            if value != "flat":
                raise ValueError(f"{name} must be 'flat', a Density, a callable or an array.")
            return cls.flat()
        if _is_distribution(value):
            return cls.from_distribution(value)
        if callable(value):
            return cls.from_function(value, bounded=bounded)
        per_volume = {None: None, "mass": False, "density": True}[representation]
        return cls(values=_nonneg(value, name), per_volume=per_volume)

    # ----------------------------------------------------------------- methods
    def masses(
        self, bins: Bins, *, tol: float = 1e-3, max_depth: int = 12, n_sub: int | None = None
    ) -> np.ndarray:
        """Normalised probability mass of each bin, shape ``(K,)``.

        Callables and distributions without a usable ``cdf`` are integrated by
        adaptive quadrature to ``tol`` (at most ``max_depth`` refinements), or with
        ``n_sub`` fixed Gauss-Legendre
        points per bin and axis (see :meth:`Likelihoods.from_density
        <binference.Likelihoods.from_density>`).

        Parameters
        ----------
        bins : Bins
            The bins.
        tol : float, optional
            Relative tolerance of the adaptive quadrature over each bin.
        max_depth : int, optional
            Most refinements of the adaptive quadrature.
        n_sub : int, optional
            Fixed Gauss-Legendre points per bin and axis in place of adaptive quadrature.
        """
        m = check_nonnegative(self._bin_masses(bins, tol, max_depth, n_sub), f"{self!r} masses")
        total = m.sum()
        if total <= 0:
            raise ValueError(f"{self!r} has no mass on the bins.")
        return m / total

    def _bin_masses(self, bins: Bins, tol: float, max_depth: int, n_sub: int | None) -> np.ndarray:
        # unnormalised bin masses
        if self._flat is not None:
            if bins.is_categorical or self._flat == "z":
                return bins.volumes
            return bins.u_volumes
        if self._values is not None:
            v = self._values
            if v.size != bins.K:
                raise ValueError(f"{self!r} has {v.size} values but the bins have K={bins.K}.")
            if self._per_volume is None and not _equal_volumes(bins.volumes):
                raise ValueError(
                    "An array interim prior on bins of unequal volume is ambiguous: pass its "
                    "representation ('mass' or 'density'), or Density.from_masses / "
                    "Density.from_densities."
                )
            return v * bins.volumes if self._per_volume else v
        if bins.is_categorical:
            if self._func is None:
                raise ValueError("A distribution needs grid bins, not categories.")
            m = np.asarray(self._func(np.asarray(bins.labels)), dtype=float).ravel()
            if m.size != bins.K:
                raise ValueError(f"{self!r} must return one value per category.")
            return m
        if self._dist is not None and bins.ndim == 1 and _has_cdf(self._dist):
            return np.exp(_dist_log_masses(self._dist, bins.edges[0]))[bins.cell_index[0]]
        if self._dist is not None:
            logf = lambda z: _dist_logpdf(self._dist, z)[None]  # noqa: E731
        else:
            logf = lambda z: _log_nonneg(self._func(z))[None]  # noqa: E731
        if self._dist is not None and not _has_pdf(self._dist):
            raise TypeError(f"{self!r} needs a logpdf or pdf here (or a cdf on 1-D bins).")
        quad = {"tol": tol, "max_depth": max_depth, "n_sub": n_sub, "jumps": self._bounded}
        log_avg = bin_log_average(logf, bins, coordinate="z", **quad)
        return np.exp(log_avg[0]) * bins.volumes

    def _log_pointwise(self) -> Callable[[np.ndarray], np.ndarray] | None:
        # the log density as a function of z, or None when only bin masses are known
        if self._flat == "z":
            return lambda z: np.zeros(np.shape(z)[0])
        if self._func is not None:
            return lambda z: _log_nonneg(self._func(z))
        if self._dist is not None and _has_pdf(self._dist):
            return lambda z: _dist_logpdf(self._dist, z)
        return None

    def __repr__(self) -> str:
        if self._flat is not None:
            return "Density.flat()" if self._flat == "measure" else "Density.flat('z')"
        if self._values is not None:
            kind = "densities" if self._per_volume else "masses"
            return f"Density.from_{kind}(<{self._values.size} values>)"
        if self._dist is not None:
            return f"Density.from_distribution({self._dist!r})"
        return f"Density.from_function({self._func!r}{', bounded=True' if self._bounded else ''})"


def _equal_volumes(vol: np.ndarray) -> bool:
    """Whether all bins have the same volume, so masses and densities read alike."""
    return bool(np.allclose(vol, vol[0], rtol=1e-9, atol=0.0))


def _is_distribution(x: Any) -> bool:
    return any(hasattr(x, m) for m in ("cdf", "logcdf", "logpdf", "pdf"))


def _dist_logpdf(dist: Any, z: np.ndarray) -> np.ndarray:
    if hasattr(dist, "logpdf"):
        return np.asarray(dist.logpdf(z), dtype=float)
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(dist.pdf(z), dtype=float))


def _has_pdf(dist: Any) -> bool:
    return hasattr(dist, "logpdf") or hasattr(dist, "pdf")


def _has_cdf(dist: Any) -> bool:
    return hasattr(dist, "cdf") or hasattr(dist, "logcdf")


def _dist_log_masses(dist: Any, edges: np.ndarray) -> np.ndarray:
    """Log mass between consecutive edges of a distribution with a CDF; for a frozen
    distribution whose parameters are columns, one row per distribution."""
    with np.errstate(divide="ignore"):
        if hasattr(dist, "logcdf"):
            lc = np.asarray(dist.logcdf(edges), dtype=float)
        else:
            lc = np.log(np.asarray(dist.cdf(edges), dtype=float))
        if hasattr(dist, "logsf"):
            ls = np.asarray(dist.logsf(edges), dtype=float)
        else:
            ls = np.log1p(-np.exp(lc))
    out = log_interval_masses(lc, ls)
    # near the median a small interval's CDF difference cancels; the density is then
    # nearly constant across it, so a 5-point quadrature of the pdf keeps full precision
    near = (out < np.log(1e-3)) & (np.abs(np.exp(lc[..., :-1]) - 0.5) < 0.25)
    if near.any() and _has_pdf(dist):
        *row, k = np.nonzero(near)  # only the near intervals: one wide object has many
        t, w = _gauss(5)
        width = np.diff(edges)[k]
        pts = edges[:-1][k, None] + width[:, None] * t
        if row:  # a batch: the near rows' parameters, a row of points each
            args = (np.asarray(a)[row[0]] for a in dist.args)
            dist = dist.dist(*args, **{n: np.asarray(v)[row[0]] for n, v in dist.kwds.items()})
            logpdf = _dist_logpdf(dist, pts)
        else:
            logpdf = _dist_logpdf(dist, pts.ravel()).reshape(pts.shape)
        with np.errstate(divide="ignore"):
            out[near] = np.log((np.exp(logpdf) @ w) * width)
    return out


def log_interval_masses(
    lc: np.ndarray, ls: np.ndarray, centred: np.ndarray | None = None
) -> np.ndarray:
    """Log mass between consecutive edges from the log-CDF ``lc`` and log-survival ``ls``.

    Uses the side that keeps precision: the survival function for intervals
    above the median. Near the median both logs are close to ``log(1/2)`` and
    their difference cancels; ``centred``, the CDF minus one half computed
    without that loss (``erf(x/sqrt 2)/2`` for a Gaussian), is then used for
    intervals inside the central half of the mass. The edges run along the last axis.
    """
    a, b = (..., slice(None, -1)), (..., slice(1, None))
    upper = lc[a] > np.log(0.5)  # the interval lies above the median: use survival functions
    hi = np.where(upper, ls[a], lc[b])
    lo = np.where(upper, ls[b], lc[a])
    near = False
    with np.errstate(divide="ignore", invalid="ignore"):
        out = hi + np.log1p(-np.exp(lo - hi))
        if centred is not None:
            near = (np.abs(centred[a]) < 0.25) & (np.abs(centred[b]) < 0.25)
            out = np.where(near, np.log(centred[b] - centred[a]), out)
    return np.where(near | (hi > lo), out, -np.inf)


def _log_nonneg(values: Any, name: str = "The values of a density") -> np.ndarray:
    v = check_nonnegative(values, name)
    with np.errstate(divide="ignore"):
        return np.log(v)


def _nonneg(a: Any, name: str) -> np.ndarray:
    return frozen(check_nonnegative(np.array(a, dtype=float).ravel(), name))
