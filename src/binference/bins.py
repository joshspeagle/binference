"""The support of a population: 1-D/N-D grids of bins, or unordered categories."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from .measure import Measure
from .warnings import InputWarning

__all__ = ["Bins"]

MeasureLike = Measure | str


def _as_edges(e: Any, axis: int = 0) -> np.ndarray:
    # adding 0.0 turns -0.0 into 0.0, so equal edges also hash equally
    arr = np.array(e, dtype=float).ravel() + 0.0
    if arr.size < 2:
        raise ValueError(f"Edges along axis {axis} need at least two values; got {arr.size}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Edges along axis {axis} must be finite.")
    if not np.all(np.diff(arr) > 0):
        raise ValueError(f"Edges along axis {axis} must be strictly increasing.")
    arr.setflags(write=False)
    return arr


def _looks_geometric(e: np.ndarray) -> bool:
    if e.size < 4 or e[0] <= 0:
        return False
    d = np.diff(e)
    r = e[1:] / e[:-1]
    return bool(np.allclose(r, r[0], rtol=1e-6) and not np.allclose(d, d[0], rtol=1e-6))


def _frozen(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


class Bins:
    """The bins (or categories) over which populations and likelihoods are defined.

    A ``Bins`` object describes the support of the population distribution: a
    one-dimensional set of bins, a rectilinear N-dimensional grid of cells
    (flattened in C order), or a set of unordered categories. It records bin
    *volumes* (widths, areas, ...), which matter whenever bins have unequal
    sizes (see :doc:`/theory`).

    Each grid axis carries a :class:`~binference.measure.Measure`, the
    coordinate ``u = T(z)`` in which the population is flat within a bin
    (DECISIONS #21): linear by default, logarithmic for :meth:`log`. A boolean
    ``mask`` over the grid removes cells from the support (DECISIONS #19); the
    remaining *active* cells are numbered ``0 .. K - 1`` in C order.

    Parameters
    ----------
    edges : array_like
        Strictly increasing 1-D bin edges (``K + 1`` values). Use
        :meth:`grid` for N-dimensional grids and :meth:`categories` for
        unordered categories.
    measure : Measure or {"linear", "log"}, optional
        The within-bin coordinate. Default linear; hand-built edges that look
        geometric then raise an :class:`~binference.InputWarning`.
    mask : array_like of bool, optional
        Which bins are part of the support (default: all).

    Examples
    --------
    >>> import binference
    >>> bins = binference.Bins([0.0, 0.5, 1.0, 2.0])
    >>> bins.K, bins.volumes.tolist()
    (3, [0.5, 0.5, 1.0])
    >>> grid = binference.Bins.grid([0, 1, 2], [0, 1, 2, 3])
    >>> grid.K, grid.shape
    (6, (2, 3))
    """

    __slots__ = (
        "_active",
        "_cell_index",
        "_centers",
        "_edges",
        "_labels",
        "_lookup",
        "_measures",
        "_ordered",
        "_shape",
        "_u_edges",
        "_volumes",
    )

    def __init__(self, edges: Any, measure: MeasureLike | None = None, mask: Any = None) -> None:
        self._set_grid((_as_edges(edges, 0),), measure, mask)

    # ------------------------------------------------------------------ setup
    def _set_grid(self, edges: tuple[np.ndarray, ...], measure: Any, mask: Any) -> None:
        nd = len(edges)
        if measure is None:
            for d, e in enumerate(edges):
                if _looks_geometric(e):
                    warnings.warn(
                        f"The edges of axis {d} are geometrically spaced but the measure is "
                        "linear, so the population is taken as flat in z within each bin. "
                        "Pass measure='log' (or use Bins.log) if it should be flat in log z.",
                        InputWarning,
                        stacklevel=3,
                    )
            measure = "linear"
        if isinstance(measure, Measure | str):
            measures = (Measure.coerce(measure),) * nd
        else:
            measures = tuple(Measure.coerce(m) for m in measure)
            if len(measures) != nd:
                raise ValueError(f"measure needs one entry per axis ({nd}); got {len(measures)}.")
        u_edges = tuple(
            _frozen(m._check(e, d)) for d, (m, e) in enumerate(zip(measures, edges, strict=True))
        )
        shape = tuple(len(e) - 1 for e in edges)
        n = math.prod(shape)
        # _active (flat cell of each bin) and _lookup (bin of each cell, -1 if masked)
        # are None when nothing is masked
        active = lookup = None
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != shape:
                raise ValueError(f"mask must have the grid's shape {shape}; got {mask.shape}.")
            cells = np.flatnonzero(mask)
            if cells.size == 0:
                raise ValueError("mask leaves no bins in the support.")
            if cells.size < n:
                active = _frozen(cells)
                lookup = np.full(n, -1, dtype=np.int64)
                lookup[active] = np.arange(active.size)
                lookup = _frozen(lookup)
        self._edges = edges
        self._shape = shape
        self._ordered = True
        self._labels = None
        self._measures = measures
        self._u_edges = u_edges
        self._active = active
        self._lookup = lookup
        k = np.arange(n) if active is None else active
        self._cell_index = tuple(_frozen(i) for i in np.unravel_index(k, shape))
        vol = np.ones(())
        for e in edges:
            vol = np.multiply.outer(vol, np.diff(e))
        vol = vol.ravel()
        mids = [m.inverse(0.5 * (u[1:] + u[:-1])) for m, u in zip(measures, u_edges, strict=True)]
        if nd == 1:
            centers = mids[0]
        else:
            mesh = np.meshgrid(*mids, indexing="ij")
            centers = np.stack([m.ravel() for m in mesh], axis=-1)
        if active is not None:
            vol, centers = vol[active], centers[active]
        self._volumes = _frozen(np.ascontiguousarray(vol, dtype=float))
        self._centers = _frozen(np.ascontiguousarray(centers, dtype=float))

    @classmethod
    def _from_parts(cls, edges, labels, ordered, measure, mask=None) -> Bins:
        obj = cls.__new__(cls)
        if edges is not None:
            obj._set_grid(edges, measure, mask)
        else:
            K = len(labels)
            obj._edges = None
            obj._shape = (K,)
            obj._ordered = ordered
            obj._labels = tuple(labels)
            obj._measures = None
            obj._u_edges = None
            obj._active = None
            obj._lookup = None
            obj._cell_index = (_frozen(np.arange(K)),)
            obj._volumes = _frozen(np.ones(K))
            obj._centers = _frozen(np.arange(K, dtype=float))
        return obj

    # ------------------------------------------------------------ constructors
    @classmethod
    def linear(cls, lo: float, hi: float, n: int, mask: Any = None) -> Bins:
        """``n`` equal-width bins spanning ``[lo, hi]``, linear measure."""
        if n < 1:
            raise ValueError("n must be at least 1.")
        return cls(np.linspace(lo, hi, int(n) + 1), measure="linear", mask=mask)

    @classmethod
    def log(cls, lo: float, hi: float, n: int, mask: Any = None) -> Bins:
        """``n`` logarithmically spaced bins spanning ``[lo, hi]`` (``lo > 0``), log measure."""
        if lo <= 0:
            raise ValueError("Logarithmic bins need lo > 0.")
        if n < 1:
            raise ValueError("n must be at least 1.")
        return cls(np.geomspace(lo, hi, int(n) + 1), measure="log", mask=mask)

    @classmethod
    def from_centers(cls, centers: Any, measure: MeasureLike = "linear") -> Bins:
        """1-D bins whose centres in the measure coordinate are ``centers``.

        Edges sit at the midpoints of ``u = T(centers)`` and the outermost bins
        extend half a spacing beyond the first and last points, all in ``u``.
        With ``measure="log"`` the edges are geometric midpoints and stay
        positive. With non-uniform spacing the returned :attr:`centers` (the
        midpoints of the bins in ``u``) need not equal the input values.
        """
        m = Measure.coerce(measure)
        c = np.asarray(centers, dtype=float).ravel()
        if c.size < 2 or not np.all(np.diff(c) > 0):
            raise ValueError("centers must contain at least two strictly increasing values.")
        if not m._covers(c[0], c[-1]):
            raise ValueError(f"centers lie outside the domain of {m!r}.")
        u = m.forward(c)
        mid = 0.5 * (u[1:] + u[:-1])
        u_edges = np.concatenate(([u[0] - (mid[0] - u[0])], mid, [u[-1] + (u[-1] - mid[-1])]))
        edges = m.inverse(u_edges)
        if not np.allclose(m.forward(edges), u_edges):
            raise ValueError(f"The outer bins extend beyond the range of {m!r}.")
        return cls(edges, measure=m)

    @classmethod
    def grid(cls, *edges: Any, measure: Any = None, mask: Any = None) -> Bins:
        """A rectilinear N-dimensional grid of cells from per-axis edges.

        Cells are flattened in C order (the last axis varies fastest); use
        :meth:`unflatten` to recover the grid shape. ``measure`` is one measure
        for every axis or a sequence with one per axis; ``mask`` has the grid's
        shape.
        """
        if len(edges) == 1 and isinstance(edges[0], Sequence) and np.ndim(edges[0][0]) > 0:
            edges = tuple(edges[0])
        if len(edges) == 0:
            raise ValueError("grid() needs at least one array of edges.")
        obj = cls.__new__(cls)
        obj._set_grid(tuple(_as_edges(e, i) for i, e in enumerate(edges)), measure, mask)
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
        return cls._from_parts(None, labs, ordered=False, measure=None)

    # -------------------------------------------------------------- properties
    @property
    def K(self) -> int:
        """Number of active bins (cells or categories)."""
        return self._cell_index[0].size

    @property
    def shape(self) -> tuple[int, ...]:
        """Grid shape, including masked cells (``(K,)`` for categories)."""
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
    def measures(self) -> tuple[Measure, ...]:
        """Per-axis measures (raises for categories)."""
        if self._measures is None:
            raise AttributeError("Categorical bins have no measure.")
        return self._measures

    @property
    def u_edges(self) -> tuple[np.ndarray, ...]:
        """Per-axis edges in the measure coordinate ``u = T(z)``."""
        if self._u_edges is None:
            raise AttributeError("Categorical bins have no edges.")
        return self._u_edges

    @property
    def mask(self) -> np.ndarray:
        """Boolean array of the grid's shape: which cells are in the support."""
        if self._lookup is None:
            return np.ones(self._shape, dtype=bool)
        return (self._lookup >= 0).reshape(self._shape)

    @property
    def is_masked(self) -> bool:
        """Whether some grid cells are excluded from the support."""
        return self._active is not None

    @property
    def cells(self) -> np.ndarray:
        """Flat grid index (C order, masked cells included) of each active bin."""
        return np.arange(self.K) if self._active is None else self._active

    @property
    def cell_index(self) -> tuple[np.ndarray, ...]:
        """Per-axis grid indices of every active bin, each ``(K,)``."""
        return self._cell_index

    @property
    def centers(self) -> np.ndarray:
        """Bin centres in ``z``, the midpoints in ``u``: ``(K,)`` in 1-D, ``(K, ndim)``
        for grids, codes for categories."""
        return self._centers

    @property
    def volumes(self) -> np.ndarray:
        """Bin volumes in ``z`` ``(K,)`` (widths in 1-D; ones for categories)."""
        return self._volumes

    @property
    def widths(self) -> tuple[np.ndarray, ...]:
        """Per-axis bin widths in ``z`` over the whole grid (raises for categories)."""
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
        return (
            all(np.array_equal(a, b) for a, b in zip(self._edges, other._edges, strict=True))
            and self._measures == other._measures
            and (
                np.array_equal(self._active, other._active)
                if self.is_masked and other.is_masked
                else self.is_masked == other.is_masked
            )
        )

    def __hash__(self) -> int:
        if self.is_categorical:
            return hash(("categories", self._labels))
        mask = None if self._active is None else self._active.tobytes()
        return hash((tuple(e.tobytes() for e in self._edges), self._measures, mask))

    def __repr__(self) -> str:
        if self.is_categorical:
            labs = list(self._labels)
            shown = labs if len(labs) <= 6 else [*labs[:3], "...", *labs[-2:]]
            return f"Bins.categories({shown}, K={self.K})"
        extra = ""
        if any(m != Measure.linear() for m in self._measures):
            extra += f", measure={'/'.join(m.name for m in self._measures)}"
        if self.is_masked:
            extra += f", {self._lookup.size - self.K} masked"
        if self.ndim == 1:
            e = self._edges[0]
            return f"Bins(K={self.K}, range=({e[0]:g}, {e[-1]:g}){extra})"
        rng = ", ".join(f"({lo:g}, {hi:g})" for lo, hi in self.range)
        return f"Bins.grid(shape={self.shape}, range=[{rng}]{extra})"

    def __setattr__(self, name: str, value: Any) -> None:
        if name in Bins.__slots__ and not hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError("Bins objects are immutable.")

    def __reduce__(self):
        if self.is_categorical:
            return (Bins.categories, (list(self._labels),))
        edges = tuple(np.array(e) for e in self._edges)
        mask = self.mask if self.is_masked else None
        return (Bins._from_parts, (edges, None, True, self._measures, mask))

    def index(self, values: Any) -> np.ndarray:
        """Bin index of each value (``-1`` outside the support or in a masked cell).

        Bins are left-closed, except the last bin along each axis, which also
        includes its upper edge. For 1-D bins ``values`` has shape ``(n,)`` or
        ``(n, 1)``; for N-D grids ``(n, ndim)``; for categories it contains labels.
        """
        if self.is_categorical:
            lookup = {lab: i for i, lab in enumerate(self._labels)}
            vals = np.asarray(values, dtype=object).ravel()
            return np.array([lookup.get(v, -1) for v in vals], dtype=np.int64)
        x = np.asarray(values, dtype=float)
        if self.ndim == 1 and (x.ndim <= 1 or (x.ndim == 2 and x.shape[1] == 1)):
            x = x.reshape(-1, 1)
        elif x.ndim == 1 and x.size == self.ndim:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.ndim:
            shapes = "(n,) or (n, 1)" if self.ndim == 1 else f"(n, {self.ndim})"
            raise ValueError(f"values must have shape {shapes}; got {np.shape(values)}.")
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
        if self._lookup is not None:
            flat[ok] = self._lookup[flat[ok]]
        return flat

    def contains(self, values: Any) -> np.ndarray:
        """Boolean mask: which values fall inside the support."""
        return self.index(values) >= 0

    def grid_index(self, k: Any) -> tuple[np.ndarray, ...]:
        """Per-axis grid indices of the bins ``k`` (indices into each axis's edges)."""
        k = np.asarray(k)
        return tuple(i[k] for i in self._cell_index)

    def sample_within(self, k: Any, rng: np.random.Generator) -> np.ndarray:
        """Values drawn uniformly in ``u`` inside the bins ``k`` (grids only).

        Returns an array of ``k``'s shape for 1-D bins, with a trailing axis of
        length ``ndim`` for grids.
        """
        k = np.asarray(k)
        vals = []
        for m, u, j in zip(self.measures, self.u_edges, self.grid_index(k), strict=True):
            vals.append(m.inverse(u[j] + (u[j + 1] - u[j]) * rng.random(k.shape)))
        return vals[0] if self.ndim == 1 else np.stack(vals, axis=-1)

    def unflatten(self, a: Any) -> np.ndarray:
        """Reshape a trailing axis of length ``K`` to the grid ``shape``.

        Masked cells are filled with NaN (the result is then floating point).
        """
        a = np.asarray(a)
        if a.shape[-1] != self.K:
            raise ValueError(f"Last axis must have length K={self.K}.")
        if self._active is None:
            return a.reshape(*a.shape[:-1], *self._shape)
        out = np.full((*a.shape[:-1], self._lookup.size), np.nan)
        out[..., self._active] = a
        return out.reshape(*a.shape[:-1], *self._shape)

    def flatten(self, a: Any) -> np.ndarray:
        """Inverse of :meth:`unflatten`: the active cells of trailing grid axes."""
        a = np.asarray(a)
        nd = self.ndim
        if a.shape[-nd:] != self._shape:
            raise ValueError(f"Trailing axes must have shape {self._shape}.")
        flat = a.reshape(*a.shape[:-nd], math.prod(self._shape))
        return flat if self._active is None else flat[..., self._active]

    def neighbors(self) -> np.ndarray:
        """Pairs of adjacent active bins, shape ``(n_pairs, 2)`` (grids only).

        Adjacent means sharing a face (4-neighbour structure in 2-D).
        """
        if self.is_categorical:
            raise ValueError("Unordered categories have no neighbour structure.")
        ids = (np.arange(self.K) if self._lookup is None else self._lookup).reshape(self._shape)
        pairs = []
        for d in range(self.ndim):
            a = np.take(ids, range(self._shape[d] - 1), axis=d).ravel()
            b = np.take(ids, range(1, self._shape[d]), axis=d).ravel()
            keep = (a >= 0) & (b >= 0)
            pairs.append(np.stack([a[keep], b[keep]], axis=1))
        return np.concatenate(pairs, axis=0)

    def coarsen(self, factor: int | Sequence[int]) -> tuple[Bins, np.ndarray]:
        """Merge blocks of ``factor`` adjacent bins along each axis.

        Returns the coarse bins (same measures and a coarsened mask) and a
        ``(K,)`` integer map from fine to coarse bins, so that coarse fractions
        are ``np.bincount(map, weights=rho)``. Each block must be wholly active
        or wholly masked, since a coarse bin's volume is its full block.
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
        coarse_idx = tuple(i // f[d] for d, i in enumerate(self._cell_index))
        coarse_shape = tuple(len(e) - 1 for e in new_edges)
        coarse_flat = np.ravel_multi_index(coarse_idx, coarse_shape)
        mask = None
        if self.is_masked:
            count = np.bincount(coarse_flat, minlength=math.prod(coarse_shape))
            if np.any((count > 0) & (count < math.prod(f))):
                raise ValueError("coarsen needs each block wholly active or wholly masked.")
            mask = (count > 0).reshape(coarse_shape)
        coarse = Bins.grid(*new_edges, measure=self._measures, mask=mask)
        if coarse._lookup is not None:
            coarse_flat = coarse._lookup[coarse_flat]
        return coarse, coarse_flat
