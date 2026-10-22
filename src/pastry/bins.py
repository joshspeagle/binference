"""The support of a population: 1-D/N-D grids of bins, or unordered categories."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["Bins"]


def _as_edges(e: Any, axis: int = 0) -> np.ndarray:
    arr = np.array(e, dtype=float).ravel()
    if arr.size < 2:
        raise ValueError(f"Edges along axis {axis} need at least two values; got {arr.size}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Edges along axis {axis} must be finite.")
    if not np.all(np.diff(arr) > 0):
        raise ValueError(f"Edges along axis {axis} must be strictly increasing.")
    arr.setflags(write=False)
    return arr


class Bins:
    """The bins (or categories) over which populations and likelihoods are defined.

    A ``Bins`` object describes the support of the population distribution: a
    one-dimensional set of bins, a rectilinear N-dimensional grid of cells
    (flattened in C order), or a set of unordered categories. It records bin
    *volumes* (widths, areas, ...), which matter whenever bins have unequal
    sizes (see :doc:`/theory`).

    Parameters
    ----------
    edges : array_like
        Strictly increasing 1-D bin edges (``K + 1`` values). Use
        :meth:`grid` for N-dimensional grids and :meth:`categories` for
        unordered categories.

    Examples
    --------
    >>> import pastry
    >>> bins = pastry.Bins([0.0, 0.5, 1.0, 2.0])
    >>> bins.K, bins.volumes.tolist()
    (3, [0.5, 0.5, 1.0])
    >>> grid = pastry.Bins.grid([0, 1, 2], [0, 1, 2, 3])
    >>> grid.K, grid.shape
    (6, (2, 3))
    """

    __slots__ = ("_centers", "_edges", "_labels", "_ordered", "_shape", "_volumes")

    def __init__(self, edges: Any) -> None:
        self._set_grid((_as_edges(edges, 0),))

    # ------------------------------------------------------------------ setup
    def _set_grid(self, edges: tuple[np.ndarray, ...]) -> None:
        self._edges = edges
        self._shape = tuple(len(e) - 1 for e in edges)
        self._ordered = True
        self._labels = None
        widths = [np.diff(e) for e in edges]
        vol = widths[0]
        for w in widths[1:]:
            vol = np.multiply.outer(vol, w)
        vol = np.asarray(vol, dtype=float).ravel()
        vol.setflags(write=False)
        self._volumes = vol
        mids = [0.5 * (e[1:] + e[:-1]) for e in edges]
        if len(edges) == 1:
            centers = mids[0].copy()
        else:
            mesh = np.meshgrid(*mids, indexing="ij")
            centers = np.stack([m.ravel() for m in mesh], axis=-1)
        centers.setflags(write=False)
        self._centers = centers

    @classmethod
    def _from_parts(cls, edges, labels, ordered) -> Bins:
        obj = cls.__new__(cls)
        if edges is not None:
            obj._set_grid(edges)
        else:
            K = len(labels)
            obj._edges = None
            obj._shape = (K,)
            obj._ordered = ordered
            obj._labels = tuple(labels)
            vol = np.ones(K)
            vol.setflags(write=False)
            obj._volumes = vol
            c = np.arange(K, dtype=float)
            c.setflags(write=False)
            obj._centers = c
        return obj

    # ------------------------------------------------------------ constructors
    @classmethod
    def linear(cls, lo: float, hi: float, n: int) -> Bins:
        """``n`` equal-width bins spanning ``[lo, hi]``."""
        if n < 1:
            raise ValueError("n must be at least 1.")
        return cls(np.linspace(lo, hi, int(n) + 1))

    @classmethod
    def log(cls, lo: float, hi: float, n: int) -> Bins:
        """``n`` logarithmically spaced bins spanning ``[lo, hi]`` (``lo > 0``)."""
        if lo <= 0:
            raise ValueError("Logarithmic bins need lo > 0.")
        return cls(np.geomspace(lo, hi, int(n) + 1))

    @classmethod
    def from_centers(cls, centers: Any) -> Bins:
        """1-D bins whose centres are ``centers`` (edges at the midpoints).

        Useful when PDFs are tabulated on a grid of points: the outermost bins
        extend half a spacing beyond the first and last points.
        """
        c = np.asarray(centers, dtype=float).ravel()
        if c.size < 2 or not np.all(np.diff(c) > 0):
            raise ValueError("centers must contain at least two strictly increasing values.")
        mid = 0.5 * (c[1:] + c[:-1])
        edges = np.concatenate(([c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]))
        return cls(edges)

    @classmethod
    def grid(cls, *edges: Any) -> Bins:
        """A rectilinear N-dimensional grid of cells from per-axis edges.

        Cells are flattened in C order (the last axis varies fastest); use
        :meth:`unflatten` to recover the grid shape.
        """
        if len(edges) == 1 and isinstance(edges[0], Sequence) and np.ndim(edges[0][0]) > 0:
            edges = tuple(edges[0])
        if len(edges) == 0:
            raise ValueError("grid() needs at least one array of edges.")
        obj = cls.__new__(cls)
        obj._set_grid(tuple(_as_edges(e, i) for i, e in enumerate(edges)))
        return obj

    @classmethod
    def categories(cls, labels: int | Sequence[Any]) -> Bins:
        """Unordered categories, either ``K`` integer labels or explicit labels."""
        if isinstance(labels, int | np.integer):
            if labels < 2:
                raise ValueError("Need at least two categories.")
            labs = tuple(range(int(labels)))
        else:
            labs = tuple(labels)
            if len(labs) < 2:
                raise ValueError("Need at least two categories.")
            if len(set(labs)) != len(labs):
                raise ValueError("Category labels must be unique.")
        return cls._from_parts(None, labs, ordered=False)

    # -------------------------------------------------------------- properties
    @property
    def K(self) -> int:
        """Number of bins (cells or categories)."""
        return int(np.prod(self._shape))

    @property
    def shape(self) -> tuple[int, ...]:
        """Grid shape (``(K,)`` for 1-D bins and categories)."""
        return self._shape

    @property
    def ndim(self) -> int:
        """Number of grid dimensions (1 for categories)."""
        return len(self._shape)

    @property
    def ordered(self) -> bool:
        """Whether bins have a geometry (grids) as opposed to unordered categories."""
        return self._ordered

    @property
    def is_categorical(self) -> bool:
        """True for unordered categories."""
        return self._edges is None

    @property
    def edges(self) -> tuple[np.ndarray, ...]:
        """Per-axis bin edges (raises for categories)."""
        if self._edges is None:
            raise AttributeError("Categorical bins have no edges.")
        return self._edges

    @property
    def centers(self) -> np.ndarray:
        """Bin centres: ``(K,)`` in 1-D, ``(K, ndim)`` for grids, codes for categories."""
        return self._centers

    @property
    def volumes(self) -> np.ndarray:
        """Bin volumes ``(K,)`` (widths in 1-D; ones for categories)."""
        return self._volumes

    @property
    def widths(self) -> tuple[np.ndarray, ...]:
        """Per-axis bin widths (raises for categories)."""
        return tuple(np.diff(e) for e in self.edges)

    @property
    def labels(self) -> tuple[Any, ...] | None:
        """Category labels (``None`` for grids)."""
        return self._labels

    @property
    def range(self) -> tuple[tuple[float, float], ...]:
        """Per-axis ``(lo, hi)`` extent of the grid."""
        return tuple((float(e[0]), float(e[-1])) for e in self.edges)

    # ----------------------------------------------------------------- methods
    def __len__(self) -> int:
        return self.K

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bins):
            return NotImplemented
        if self.is_categorical != other.is_categorical or self.shape != other.shape:
            return False
        if self.is_categorical:
            return self._labels == other._labels
        return all(np.array_equal(a, b) for a, b in zip(self._edges, other._edges, strict=True))

    def __hash__(self) -> int:
        if self.is_categorical:
            return hash(("categories", self._labels))
        return hash(tuple(e.tobytes() for e in self._edges))

    def __repr__(self) -> str:
        if self.is_categorical:
            labs = list(self._labels)
            shown = labs if len(labs) <= 6 else [*labs[:3], "...", *labs[-2:]]
            return f"Bins.categories({shown}, K={self.K})"
        if self.ndim == 1:
            e = self._edges[0]
            return f"Bins(K={self.K}, range=({e[0]:g}, {e[-1]:g}))"
        rng = ", ".join(f"({lo:g}, {hi:g})" for lo, hi in self.range)
        return f"Bins.grid(shape={self.shape}, range=[{rng}])"

    def __setattr__(self, name: str, value: Any) -> None:
        if name in Bins.__slots__ and not hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError("Bins objects are immutable.")

    def __reduce__(self):
        if self.is_categorical:
            return (Bins.categories, (list(self._labels),))
        return (Bins.grid, tuple(np.array(e) for e in self._edges))

    def index(self, values: Any) -> np.ndarray:
        """Bin index of each value (``-1`` if outside the support).

        Bins are left-closed, except the last bin along each axis, which also
        includes its upper edge. For N-D grids ``values`` has shape ``(n, ndim)``;
        for categories it contains labels.
        """
        if self.is_categorical:
            lookup = {lab: i for i, lab in enumerate(self._labels)}
            vals = np.asarray(values, dtype=object).ravel()
            return np.array([lookup.get(v, -1) for v in vals], dtype=np.int64)
        x = np.asarray(values, dtype=float)
        if self.ndim == 1:
            x = x.reshape(-1, 1)
        elif x.ndim == 1 and x.size == self.ndim:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.ndim:
            raise ValueError(f"values must have shape (n, {self.ndim}).")
        idx = np.empty(x.shape, dtype=np.int64)
        ok = np.ones(x.shape[0], dtype=bool)
        for d, e in enumerate(self._edges):
            j = np.searchsorted(e, x[:, d], side="right") - 1
            j = np.where(x[:, d] == e[-1], len(e) - 2, j)
            ok &= (j >= 0) & (j < len(e) - 1) & np.isfinite(x[:, d])
            idx[:, d] = j
        flat = np.full(x.shape[0], -1, dtype=np.int64)
        if np.any(ok):
            flat[ok] = np.ravel_multi_index(tuple(idx[ok].T), self._shape)
        return flat

    def contains(self, values: Any) -> np.ndarray:
        """Boolean mask: which values fall inside the support."""
        return self.index(values) >= 0

    def unflatten(self, a: Any) -> np.ndarray:
        """Reshape a trailing axis of length ``K`` to the grid ``shape``."""
        a = np.asarray(a)
        if a.shape[-1] != self.K:
            raise ValueError(f"Last axis must have length K={self.K}.")
        return a.reshape(*a.shape[:-1], *self._shape)

    def flatten(self, a: Any) -> np.ndarray:
        """Inverse of :meth:`unflatten`."""
        a = np.asarray(a)
        nd = self.ndim
        if a.shape[-nd:] != self._shape:
            raise ValueError(f"Trailing axes must have shape {self._shape}.")
        return a.reshape(*a.shape[:-nd], self.K)

    def neighbors(self) -> np.ndarray:
        """Pairs of adjacent cells, shape ``(n_pairs, 2)`` (grids only).

        Adjacent means sharing a face (4-neighbour structure in 2-D).
        """
        if self.is_categorical:
            raise ValueError("Unordered categories have no neighbour structure.")
        ids = np.arange(self.K).reshape(self._shape)
        pairs = []
        for d in range(self.ndim):
            a = np.take(ids, range(self._shape[d] - 1), axis=d).ravel()
            b = np.take(ids, range(1, self._shape[d]), axis=d).ravel()
            pairs.append(np.stack([a, b], axis=1))
        return np.concatenate(pairs, axis=0) if pairs else np.empty((0, 2), dtype=int)

    def coarsen(self, factor: int | Sequence[int]) -> tuple[Bins, np.ndarray]:
        """Merge blocks of ``factor`` adjacent bins along each axis.

        Returns the coarse bins and an ``(K,)`` integer map from fine to coarse
        bins, so that coarse fractions are ``np.bincount(map, weights=rho)``.
        """
        if self.is_categorical:
            raise ValueError("Use explicit groups to merge categories.")
        f = np.broadcast_to(np.asarray(factor, dtype=int), (self.ndim,))
        new_edges = []
        for d, e in enumerate(self._edges):
            n = len(e) - 1
            if f[d] < 1 or n % f[d] != 0:
                raise ValueError(f"Axis {d} has {n} bins, not divisible by factor {f[d]}.")
            new_edges.append(e[:: f[d]])
        coarse = Bins.grid(*new_edges)
        fine_idx = np.indices(self._shape).reshape(self.ndim, -1)
        coarse_idx = tuple(fine_idx[d] // f[d] for d in range(self.ndim))
        mapping = np.ravel_multi_index(coarse_idx, coarse.shape)
        return coarse, mapping
