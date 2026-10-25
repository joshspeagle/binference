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
    expected, the string ``"flat"``, a callable and a distribution object are
    also accepted; a bare array is accepted only where its reading is
    unambiguous (see :meth:`coerce`).
    """

    __slots__ = ("_kind", "_value")

    def __init__(self, kind: str, value: Any) -> None:
        self._kind = kind
        self._value = value

    # ------------------------------------------------------------ constructors
    @classmethod
    def flat(cls, coordinate: Literal["measure", "z"] = "measure") -> Density:
        """Uniform in the measure coordinate ``u`` of each axis (default) or in ``z``.

        On linear bins the two agree. On log bins, flat in ``u`` gives equal
        mass per decade. On categories, every category has equal mass.
        """
        if coordinate not in ("measure", "z"):
            raise ValueError(f"coordinate must be 'measure' or 'z'; got {coordinate!r}.")
        return cls("flat", coordinate)

    @classmethod
    def from_masses(cls, masses: Any) -> Density:
        """Probability mass per bin (normalised on use)."""
        return cls("masses", _nonneg(masses, "masses"))

    @classmethod
    def from_densities(cls, densities: Any) -> Density:
        """Density per unit ``z``-volume in each bin; masses are densities times volumes."""
        return cls("densities", _nonneg(densities, "densities"))

    @classmethod
    def from_function(cls, func: Callable[[np.ndarray], np.ndarray]) -> Density:
        """A callable density of ``z``, averaged over each bin by quadrature.

        On grids ``func`` receives points of shape ``(M,)`` (1-D) or
        ``(M, ndim)``; on categories it receives the category labels.
        """
        if not callable(func):
            raise TypeError("func must be callable.")
        return cls("function", func)

    @classmethod
    def from_distribution(cls, dist: Any) -> Density:
        """A distribution object with ``cdf``, ``logpdf`` or ``pdf`` (e.g. frozen ``scipy.stats``).

        On 1-D grids a ``cdf`` gives exact bin masses; otherwise the density is
        averaged over each bin by quadrature.
        """
        if not any(hasattr(dist, m) for m in ("cdf", "logpdf", "pdf")):
            raise TypeError("dist needs a cdf, logpdf or pdf method.")
        return cls("distribution", dist)

    @classmethod
    def coerce(
        cls,
        value: Any,
        bins: Bins,
        representation: Representation | None = None,
        name: str = "interim_prior",
    ) -> Density:
        """A :class:`Density` from a density, ``"flat"``, a callable, a distribution or an array.

        An array is read as masses or densities by ``representation``. Without
        it the array is accepted only where both readings agree: on categories
        and on bins of equal volume.
        """
        if representation not in (None, "mass", "density"):
            raise ValueError(f"{name} representation must be 'mass' or 'density'.")
        if isinstance(value, Density):
            return value
        if isinstance(value, str):
            if value != "flat":
                raise ValueError(f"{name} must be 'flat', a Density, a callable or an array.")
            return cls.flat()
        if any(hasattr(value, m) for m in ("cdf", "logpdf", "pdf")):
            return cls.from_distribution(value)
        if callable(value):
            return cls.from_function(value)
        if representation is None:
            vol = bins.volumes
            if not np.allclose(vol, vol[0], rtol=1e-12, atol=0.0):
                raise ValueError(
                    f"An array {name} on bins of unequal volume is ambiguous: pass its "
                    "representation ('mass' or 'density'), or Density.from_masses / "
                    "Density.from_densities."
                )
            representation = "mass"
        if representation == "mass":
            return cls.from_masses(value)
        return cls.from_densities(value)

    # ----------------------------------------------------------------- methods
    def masses(self, bins: Bins, n_sub: int = 5) -> np.ndarray:
        """Normalised probability mass of each bin, shape ``(K,)``.

        ``n_sub`` is the number of Gauss-Legendre points per bin and axis for
        callables and distributions without a usable ``cdf``.
        """
        m = self._bin_masses(bins, n_sub)
        if not np.all(np.isfinite(m)) or np.any(m < 0):
            raise ValueError(f"{self!r} must give finite, non-negative masses.")
        if m.sum() <= 0:
            raise ValueError(f"{self!r} has no mass on the bins.")
        return m / m.sum()

    def _bin_masses(self, bins: Bins, n_sub: int = 5, quad: Any = None) -> np.ndarray:
        # unnormalised bin masses; ``quad`` reuses precomputed (points, weights)
        kind, v = self._kind, self._value
        if kind == "flat":
            if bins.is_categorical or v == "z":
                return np.array(bins.volumes, dtype=float)
            widths = [np.diff(u)[j] for u, j in zip(bins.u_edges, bins.cell_index, strict=True)]
            return np.prod(widths, axis=0)
        if kind in ("masses", "densities"):
            if v.size != bins.K:
                raise ValueError(f"{self!r} has {v.size} values but the bins have K={bins.K}.")
            return v * bins.volumes if kind == "densities" else v.copy()
        if bins.is_categorical:
            if kind == "distribution":
                raise ValueError("A distribution needs grid bins, not categories.")
            labels = np.empty(bins.K, dtype=object)
            labels[:] = bins.labels
            return np.asarray(v(labels), dtype=float).ravel()
        if kind == "distribution" and bins.ndim == 1 and hasattr(v, "cdf"):
            c = np.asarray(v.cdf(bins.edges[0]), dtype=float)
            return np.clip(np.diff(c), 0.0, None)[bins.cells]
        pts, w = _subbin_points(bins, n_sub) if quad is None else quad
        if kind == "function":
            dens = np.asarray(v(pts), dtype=float).ravel()
        elif hasattr(v, "logpdf"):
            dens = np.exp(np.asarray(v.logpdf(pts), dtype=float)).ravel()
        else:
            dens = np.asarray(v.pdf(pts), dtype=float).ravel()
        if dens.size != pts.shape[0]:
            raise ValueError(f"{self!r} must return one density per point.")
        return (dens.reshape(bins.K, -1) @ w) * bins.volumes

    def __repr__(self) -> str:
        if self._kind == "flat":
            return "Density.flat()" if self._value == "measure" else "Density.flat('z')"
        if self._kind in ("masses", "densities"):
            return f"Density.from_{self._kind}(<{self._value.size} values>)"
        return f"Density.from_{self._kind}({self._value!r})"


def _nonneg(a: Any, name: str) -> np.ndarray:
    arr = np.array(a, dtype=float).ravel()
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"{name} must be finite and non-negative.")
    arr.setflags(write=False)
    return arr
