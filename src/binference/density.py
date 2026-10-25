"""Fixed densities over the bin coordinate: interim priors, base measures, simulation truth.

A :class:`Density` is one fixed probability density :math:`\\pi(z)` with a single
meaning wherever it appears (DECISIONS #26). It is resolved against bins with
:meth:`Density.masses`, which returns the normalised mass :math:`\\Pi_k` of each bin.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from ._quadrature import _subbin_points
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
    # _func (callable density) or _dist (distribution object) is set
    __slots__ = ("_dist", "_flat", "_func", "_per_volume", "_values")

    def __init__(self, *, flat=None, values=None, per_volume=False, func=None, dist=None):
        if sum(x is not None for x in (flat, values, func, dist)) != 1:
            raise TypeError(
                "Build a Density with flat, from_masses, from_densities, from_function, "
                "from_distribution or coerce."
            )
        self._flat = flat
        self._values = values
        self._per_volume = per_volume
        self._func = func
        self._dist = dist

    # ------------------------------------------------------------ constructors
    @classmethod
    def flat(cls, coordinate: Literal["measure", "z"] = "measure") -> Density:
        """Uniform in the measure coordinate ``u`` of each axis (default) or in ``z``.

        On linear bins the two agree. On log bins, flat in ``u`` gives equal
        mass per decade. On categories, every category has equal mass.
        """
        if coordinate not in ("measure", "z"):
            raise ValueError(f"coordinate must be 'measure' or 'z'; got {coordinate!r}.")
        return cls(flat=coordinate)

    @classmethod
    def from_masses(cls, masses: Any) -> Density:
        """Probability mass per bin (normalised on use)."""
        return cls(values=_nonneg(masses, "masses"))

    @classmethod
    def from_densities(cls, densities: Any) -> Density:
        """Density per unit ``z``-volume in each bin; masses are densities times volumes."""
        return cls(values=_nonneg(densities, "densities"), per_volume=True)

    @classmethod
    def from_function(cls, func: Callable[[np.ndarray], np.ndarray]) -> Density:
        """A callable density of ``z``, averaged over each bin by quadrature.

        On grids ``func`` receives points of shape ``(M,)`` (1-D) or
        ``(M, ndim)``; on categories it receives the category labels.
        """
        if not callable(func):
            raise TypeError("func must be callable.")
        return cls(func=func)

    @classmethod
    def from_distribution(cls, dist: Any) -> Density:
        """A distribution object with ``cdf``, ``logpdf`` or ``pdf`` (e.g. frozen ``scipy.stats``).

        On 1-D grids a ``cdf`` gives exact bin masses; otherwise the density is
        averaged over each bin by quadrature.
        """
        if not _is_distribution(dist):
            raise TypeError("dist needs a cdf, logpdf or pdf method.")
        return cls(dist=dist)

    @classmethod
    def coerce(
        cls, value: Any, representation: Representation | None = None, name: str = "interim_prior"
    ) -> Density:
        """A :class:`Density` from a density, ``"flat"``, a callable, a distribution or an array.

        An array is read as masses or densities by ``representation`` (which
        applies to arrays only). Without it the array is accepted only on bins
        where both readings agree (categories and bins of equal volume);
        :meth:`masses` raises otherwise.
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
            return cls.from_function(value)
        per_volume = {None: None, "mass": False, "density": True}[representation]
        return cls(values=_nonneg(value, name), per_volume=per_volume)

    # ----------------------------------------------------------------- methods
    def masses(self, bins: Bins, n_sub: int = 5) -> np.ndarray:
        """Normalised probability mass of each bin, shape ``(K,)``.

        ``n_sub`` is the number of Gauss-Legendre points per bin and axis for
        callables and distributions without a usable ``cdf``.
        """
        m = self._bin_masses(bins, n_sub)
        total = m.sum()
        if not np.all(np.isfinite(m)) or np.any(m < 0):
            raise ValueError(f"{self!r} must give finite, non-negative masses.")
        if total <= 0:
            raise ValueError(f"{self!r} has no mass on the bins.")
        return m / total

    def _bin_masses(self, bins: Bins, n_sub: int = 5) -> np.ndarray:
        # unnormalised bin masses
        if self._flat is not None:
            if bins.is_categorical or self._flat == "z":
                return bins.volumes
            widths = [np.diff(u)[j] for u, j in zip(bins.u_edges, bins.cell_index, strict=True)]
            return np.prod(widths, axis=0)
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
        if self._dist is not None:
            return _dist_bin_masses(self._dist, bins, n_sub)
        return _func_bin_masses(self._func, bins, n_sub)

    def _pointwise(self) -> Callable[[np.ndarray], np.ndarray] | None:
        # the density as a function of z, or None when only bin masses are known
        if self._flat == "z":
            return lambda z: np.ones(np.shape(z)[0])
        return self._func if self._dist is None else _dist_pdf(self._dist)

    def __repr__(self) -> str:
        if self._flat is not None:
            return "Density.flat()" if self._flat == "measure" else "Density.flat('z')"
        if self._values is not None:
            kind = "densities" if self._per_volume else "masses"
            return f"Density.from_{kind}(<{self._values.size} values>)"
        if self._dist is not None:
            return f"Density.from_distribution({self._dist!r})"
        return f"Density.from_function({self._func!r})"


def _equal_volumes(vol: np.ndarray) -> bool:
    """Whether all bins have the same volume, so masses and densities read alike."""
    return bool(np.allclose(vol, vol[0], rtol=1e-9, atol=0.0))


def _is_distribution(x: Any) -> bool:
    return any(hasattr(x, m) for m in ("cdf", "logpdf", "pdf"))


def _dist_pdf(dist: Any) -> Callable[[np.ndarray], np.ndarray]:
    if hasattr(dist, "logpdf"):
        return lambda z: np.exp(np.asarray(dist.logpdf(z), dtype=float))
    return dist.pdf


def _func_bin_masses(func: Callable, bins: Bins, n_sub: int) -> np.ndarray:
    pts, w = _subbin_points(bins, n_sub)
    dens = np.asarray(func(pts.copy()), dtype=float).ravel()
    if dens.size != pts.shape[0]:
        raise ValueError("A density function must return one value per point.")
    return (dens.reshape(bins.K, -1) @ w) * bins.volumes


def _dist_bin_masses(dist: Any, bins: Bins, n_sub: int) -> np.ndarray:
    """Unnormalised bin masses of a distribution: exact by ``cdf`` on 1-D grids."""
    if bins.ndim == 1 and hasattr(dist, "cdf"):
        c = np.asarray(dist.cdf(bins.edges[0]), dtype=float)
        return np.clip(np.diff(c), 0.0, None)[bins.cell_index[0]]
    return _func_bin_masses(_dist_pdf(dist), bins, n_sub)


def _nonneg(a: Any, name: str) -> np.ndarray:
    arr = np.array(a, dtype=float).ravel()
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"{name} must be finite and non-negative.")
    arr.setflags(write=False)
    return arr
