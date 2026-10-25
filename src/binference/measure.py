"""Measures: the coordinate in which a population is flat within each bin.

A measure is a strictly increasing map :math:`u = T(z)` of a bin axis. Within a
bin the population is taken to be flat in :math:`u`, so its density in
:math:`z` is proportional to :math:`T'(z)`. Linear bins use :math:`u = z`;
logarithmic bins use :math:`u = \\log z`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["Measure"]

_NAMED = ("linear", "log")


class Measure:
    """A strictly increasing map ``u = T(z)`` of one bin axis.

    Build one with :meth:`linear`, :meth:`log`, :meth:`transform` (a function
    and its inverse) or :meth:`density` (a positive density ``g(z)``, whose
    cumulative integral is the map). Strings ``"linear"`` and ``"log"`` are
    accepted wherever a measure is.
    """

    __slots__ = ("_derivative", "_forward", "_inverse", "_table", "name")

    def __init__(
        self,
        name: str,
        forward: Callable[[np.ndarray], np.ndarray],
        inverse: Callable[[np.ndarray], np.ndarray],
        derivative: Callable[[np.ndarray], np.ndarray],
        table: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.name = name
        self._forward = forward
        self._inverse = inverse
        self._derivative = derivative
        self._table = table

    # ------------------------------------------------------------ constructors
    @classmethod
    def linear(cls) -> Measure:
        """``u = z``: flat within bins in the bin coordinate itself."""
        return cls("linear", _identity, _identity, np.ones_like)

    @classmethod
    def log(cls) -> Measure:
        """``u = log z`` (``z > 0``): flat within bins in the logarithm."""
        return cls("log", np.log, np.exp, np.reciprocal)

    @classmethod
    def transform(
        cls,
        forward: Callable[[np.ndarray], np.ndarray],
        inverse: Callable[[np.ndarray], np.ndarray],
        derivative: Callable[[np.ndarray], np.ndarray] | None = None,
        name: str = "transform",
    ) -> Measure:
        """A user map ``u = forward(z)`` with its inverse.

        ``derivative`` is ``dT/dz``; without it a central finite difference is
        used. All three must accept and return NumPy arrays.
        """
        if not (callable(forward) and callable(inverse)):
            raise TypeError("forward and inverse must be callables.")
        if derivative is None:

            def derivative(z: np.ndarray) -> np.ndarray:
                z = np.asarray(z, dtype=float)
                h = 1e-6 * np.maximum(1.0, np.abs(z))
                return (forward(z + h) - forward(z - h)) / (2 * h)

        return cls(name, forward, inverse, derivative)

    @classmethod
    def density(
        cls, g: Callable[[np.ndarray], np.ndarray], lo: float, hi: float, n: int = 4097
    ) -> Measure:
        """The map whose derivative is the positive density ``g`` on ``[lo, hi]``.

        ``T(z) = integral of g from lo to z``, computed by the trapezoidal rule
        on ``n`` points and interpolated linearly, as is its inverse.
        """
        z = np.linspace(float(lo), float(hi), int(n))
        dens = np.asarray(g(z), dtype=float)
        if dens.shape != z.shape or not np.all(np.isfinite(dens)) or np.any(dens <= 0):
            raise ValueError("g must return finite, strictly positive values on [lo, hi].")
        u = np.concatenate(([0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(z))))
        return cls.tabulated(z, u, name="density")

    @classmethod
    def tabulated(cls, z: Any, u: Any, name: str = "tabulated") -> Measure:
        """A piecewise-linear map through the points ``(z, u)``, both strictly increasing."""
        z = np.array(z, dtype=float)
        u = np.array(u, dtype=float)
        if z.ndim != 1 or z.shape != u.shape or z.size < 2:
            raise ValueError("z and u must be 1-D arrays of the same length (at least 2).")
        if not (np.all(np.diff(z) > 0) and np.all(np.diff(u) > 0)):
            raise ValueError("z and u must both be strictly increasing.")
        z.setflags(write=False)
        u.setflags(write=False)
        slopes = np.diff(u) / np.diff(z)

        def derivative(x: np.ndarray) -> np.ndarray:
            j = np.clip(np.searchsorted(z, x, side="right") - 1, 0, slopes.size - 1)
            return slopes[j]

        return cls(
            name,
            lambda x: np.interp(x, z, u),
            lambda y: np.interp(y, u, z),
            derivative,
            table=(z, u),
        )

    @classmethod
    def coerce(cls, measure: Measure | str) -> Measure:
        """A :class:`Measure` from a measure or one of the names ``"linear"`` and ``"log"``."""
        if isinstance(measure, Measure):
            return measure
        if measure in _NAMED:
            return getattr(cls, measure)()
        raise ValueError(f"measure must be a Measure, 'linear' or 'log'; got {measure!r}.")

    # ----------------------------------------------------------------- methods
    def forward(self, z: Any) -> np.ndarray:
        """``u = T(z)``."""
        return np.asarray(self._forward(np.asarray(z, dtype=float)), dtype=float)

    def inverse(self, u: Any) -> np.ndarray:
        """``z = T^{-1}(u)``."""
        return np.asarray(self._inverse(np.asarray(u, dtype=float)), dtype=float)

    def derivative(self, z: Any) -> np.ndarray:
        """``dT/dz``: the within-bin density shape in ``z``, up to normalisation."""
        return np.asarray(self._derivative(np.asarray(z, dtype=float)), dtype=float)

    @property
    def is_linear(self) -> bool:
        return self.name == "linear"

    def table(self, edges: np.ndarray, n: int = 4097) -> tuple[np.ndarray, np.ndarray]:
        """Points ``(z, u)`` that reproduce the map on ``[edges[0], edges[-1]]``."""
        if self._table is not None:
            return self._table
        z = np.unique(np.concatenate([np.linspace(edges[0], edges[-1], n), edges]))
        return z, self.forward(z)

    def check(self, edges: np.ndarray, axis: int) -> np.ndarray:
        """The edges mapped to ``u``; raises unless they stay finite and increasing."""
        if self.name == "log" and edges[0] <= 0:
            raise ValueError(f"A log measure needs positive edges (axis {axis}).")
        if self._table is not None and (
            edges[0] < self._table[0][0] or edges[-1] > self._table[0][-1]
        ):
            raise ValueError(f"Edges along axis {axis} extend beyond the measure's range.")
        u = self.forward(edges)
        if not (np.all(np.isfinite(u)) and np.all(np.diff(u) > 0)):
            raise ValueError(f"The measure must map the edges of axis {axis} to increasing u.")
        return u

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Measure):
            return NotImplemented
        if self.name in _NAMED or other.name in _NAMED:
            return self.name == other.name
        if self._table is not None and other._table is not None:
            mine, theirs = self._table, other._table
            return np.array_equal(mine[0], theirs[0]) and np.array_equal(mine[1], theirs[1])
        return self._forward is other._forward and self._inverse is other._inverse

    def __hash__(self) -> int:
        if self.name in _NAMED:
            return hash(self.name)
        if self._table is not None:
            return hash(tuple(t.tobytes() for t in self._table))
        return hash((id(self._forward), id(self._inverse)))

    def __reduce__(self):
        if self.name in _NAMED:
            return (Measure.coerce, (self.name,))
        if self._table is not None:
            z, u = (np.array(t) for t in self._table)
            return (Measure.tabulated, (z, u, self.name))
        return (Measure.transform, (self._forward, self._inverse, self._derivative, self.name))

    def __repr__(self) -> str:
        return f"Measure.{self.name}()" if self.name in _NAMED else f"Measure({self.name!r})"


def _identity(x: np.ndarray) -> np.ndarray:
    return x
