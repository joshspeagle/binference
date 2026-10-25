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
from scipy.integrate import cumulative_trapezoid

from ._utils import frozen

__all__ = ["Measure"]


class Measure:
    """A strictly increasing map ``u = T(z)`` of one bin axis.

    Build one with :meth:`linear`, :meth:`log`, :meth:`transform` (a function
    and its inverse), :meth:`density` (a positive density ``g(z)``, whose
    cumulative integral is the map) or :meth:`tabulated`. Strings ``"linear"``
    and ``"log"`` are accepted wherever a measure is.
    """

    __slots__ = ("_forward", "_hash", "_inverse", "_key", "_table", "name")

    def __init__(
        self,
        name: str,
        forward: Callable[[np.ndarray], np.ndarray],
        inverse: Callable[[np.ndarray], np.ndarray],
        key: tuple,
        table: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.name = name
        self._forward = forward
        self._inverse = inverse
        # identity: ("named", name), ("table", z bytes, u bytes) or ("fn", forward, inverse)
        self._key = key
        self._hash = hash(key)
        self._table = table

    # ------------------------------------------------------------ constructors
    @classmethod
    def linear(cls) -> Measure:
        """``u = z``: flat within bins in the bin coordinate itself."""
        return _NAMED["linear"]

    @classmethod
    def log(cls) -> Measure:
        """``u = log z`` (``z > 0``): flat within bins in the logarithm."""
        return _NAMED["log"]

    @classmethod
    def transform(
        cls,
        forward: Callable[[np.ndarray], np.ndarray],
        inverse: Callable[[np.ndarray], np.ndarray],
        name: str = "transform",
    ) -> Measure:
        """A user map ``u = forward(z)`` with its inverse; both take and return arrays."""
        if not (callable(forward) and callable(inverse)):
            raise TypeError("forward and inverse must be callables.")
        return cls(name, forward, inverse, ("fn", forward, inverse))

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
        return cls.tabulated(z, cumulative_trapezoid(dens, z, initial=0.0), name="density")

    @classmethod
    def tabulated(cls, z: Any, u: Any, name: str = "tabulated") -> Measure:
        """A piecewise-linear map through the points ``(z, u)``, both strictly increasing."""
        z = np.array(z, dtype=float)
        u = np.array(u, dtype=float)
        if z.ndim != 1 or z.shape != u.shape or z.size < 2:
            raise ValueError("z and u must be 1-D arrays of the same length (at least 2).")
        if not (np.all(np.diff(z) > 0) and np.all(np.diff(u) > 0)):
            raise ValueError("z and u must both be strictly increasing.")
        z, u = frozen(z), frozen(u)
        return cls(
            name,
            lambda x: np.interp(x, z, u),
            lambda y: np.interp(y, u, z),
            ("table", z.tobytes(), u.tobytes()),
            table=(z, u),
        )

    @classmethod
    def coerce(cls, measure: Measure | str) -> Measure:
        """A :class:`Measure` from a measure or one of the names ``"linear"`` and ``"log"``."""
        if isinstance(measure, Measure):
            return measure
        if isinstance(measure, str) and measure in _NAMED:
            return _NAMED[measure]
        raise ValueError(f"measure must be a Measure, 'linear' or 'log'; got {measure!r}.")

    # ----------------------------------------------------------------- methods
    @property
    def is_named(self) -> bool:
        """Whether this is one of the built-in measures ``"linear"`` and ``"log"``."""
        return self._key[0] == "named"

    def forward(self, z: Any) -> np.ndarray:
        """``u = T(z)``."""
        return np.asarray(self._forward(np.asarray(z, dtype=float)), dtype=float)

    def inverse(self, u: Any) -> np.ndarray:
        """``z = T^{-1}(u)``."""
        return np.asarray(self._inverse(np.asarray(u, dtype=float)), dtype=float)

    def _covers(self, lo: float, hi: float) -> bool:
        # whether [lo, hi] lies inside the measure's domain
        if self is _NAMED["log"]:
            return lo > 0
        if self._table is not None:
            return self._table[0][0] <= lo and hi <= self._table[0][-1]
        return True

    def _tabulate(self, edges: np.ndarray, n: int = 4097) -> tuple[np.ndarray, np.ndarray]:
        # points (z, u) that reproduce the map on [edges[0], edges[-1]]
        if self._table is not None:
            return self._table
        z = np.unique(np.concatenate([np.linspace(edges[0], edges[-1], n), edges]))
        return z, self.forward(z)

    def _check(self, edges: np.ndarray, axis: int) -> np.ndarray:
        # the edges mapped to u; raises unless they stay finite and increasing
        if not self._covers(edges[0], edges[-1]):
            raise ValueError(f"Edges along axis {axis} lie outside the domain of {self!r}.")
        u = self.forward(edges)
        if not (np.all(np.isfinite(u)) and np.all(np.diff(u) > 0)):
            raise ValueError(f"The measure must map the edges of axis {axis} to increasing u.")
        return u

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Measure):
            return NotImplemented
        return self._key == other._key

    def __hash__(self) -> int:
        return self._hash

    def __reduce__(self):
        kind = self._key[0]
        if kind == "named":
            return (Measure.coerce, (self.name,))
        if kind == "table":
            z, u = (np.array(t) for t in self._table)
            return (Measure.tabulated, (z, u, self.name))
        return (Measure.transform, (self._forward, self._inverse, self.name))

    def __repr__(self) -> str:
        return f"Measure.{self.name}()" if self.is_named else f"Measure({self.name!r})"


def _identity(x: np.ndarray) -> np.ndarray:
    return x


_NAMED = {
    "linear": Measure("linear", _identity, _identity, ("named", "linear")),
    "log": Measure("log", np.log, np.exp, ("named", "log")),
}
