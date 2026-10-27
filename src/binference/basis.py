r"""Mixture bases: the model's components and how they map to the reporting bins.

The population is a mixture :math:`f(z) = \sum_k w_k \phi_k(z)` of fixed
non-negative components of unit mass, so every per-object likelihood stays
linear in the weights, :math:`L_{ik} = \int p(d_i \mid z)\,\phi_k(z)\,du`, and
the engines and priors are unchanged (DECISIONS #4, #5, #76). Results are
reported in the bins: the fraction in bin :math:`j` is
:math:`\rho_j = \sum_k A_{jk} w_k` with :math:`A_{jk} = \int_{B_j} \phi_k\,du`.

The components sit on the reporting bins refined ``resolution`` times per axis
in the measure coordinate ``u``. ``order=0`` makes each component flat in ``u``
on its cell (the histogram; at ``resolution=1`` exactly the plain bins);
``order=1`` makes it a linear B-spline (hat) centred on its cell, linear in
``u`` between neighbouring centres and constant from the outermost centres to
the range edges, so the components form a partition of unity and the density
has no steps at bin edges. Grids use tensor products; order 1 needs an
unmasked grid, and on masked grids flat components cover the active cells.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from ._utils import frozen
from .bins import Bins

__all__ = ["Basis"]

#: quadrature cells per component and axis for order 1, by dimension (likelihoods,
#: selection); about 8 to 16 cells per component
DEFAULT_OVERSAMPLE = {1: 8, 2: 4}


class Basis:
    """Components of the population model over reporting bins.

    Parameters
    ----------
    bins : Bins
        The reporting bins (a grid, not categories).
    order : {0, 1}
        B-spline order of the components: 0 flat in ``u`` on each cell, 1
        linear (hats). See the module documentation.
    resolution : int or sequence of int
        Components per reporting bin along each axis (default 1).
    oversample : int, optional
        Quadrature cells per component and axis. They integrate likelihoods
        and selection functions against the components (default 1 for order 0,
        where the integrals are exact; for order 1, 8 in 1-D, 4 in 2-D and 2
        in more dimensions).

    Attributes
    ----------
    bins : Bins
        The reporting bins.
    components : Bins
        The component cells (the bins refined ``resolution`` times); priors are
        placed on these.
    grid : Bins
        The quadrature cells (the components refined ``oversample`` times).
    grid_to_bins : ndarray (grid.K,)
        The reporting bin of each quadrature cell.
    matrix : sparse array (K, C)
        :math:`A_{jk}`, the mass of component *k* in reporting bin *j*; each
        column sums to one.

    Examples
    --------
    >>> import binference
    >>> basis = binference.Basis(binference.Bins.linear(0, 3, 10), order=1)
    >>> basis.K, basis.C
    (10, 10)
    """

    def __init__(
        self, bins: Bins, order: int = 0, resolution: Any = 1, oversample: int | None = None
    ) -> None:
        if not isinstance(bins, Bins):
            raise TypeError("bins must be a binference.Bins instance.")
        if bins.is_categorical:
            raise ValueError("A basis needs a grid; categories have no within-bin shape.")
        if order not in (0, 1):
            raise ValueError(f"order must be 0 (flat) or 1 (linear); got {order!r}.")
        if order == 1 and bins.is_masked:
            raise ValueError(
                "Linear components need an unmasked grid: next to a masked cell the hats "
                "would not sum to one, biasing the bins there. Use order=0 on masked grids."
            )
        try:
            res = np.broadcast_to(np.asarray(resolution), (bins.ndim,))
            valid = all(float(r).is_integer() and r >= 1 for r in res)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(
                f"resolution must be a positive integer, or one per axis ({bins.ndim}); "
                f"got {resolution!r}."
            )
        if oversample is None:
            oversample = 1 if order == 0 else DEFAULT_OVERSAMPLE.get(bins.ndim, 2)
        if not float(oversample).is_integer() or oversample < 1:
            raise ValueError(f"oversample must be a positive integer; got {oversample!r}.")
        self.bins = bins
        self.order = int(order)
        self.oversample = int(oversample)
        self.components, _ = bins.refine(res.astype(int))
        self.resolution = tuple(
            (len(c) - 1) // (len(b) - 1)
            for c, b in zip(self.components.edges, bins.edges, strict=True)
        )
        fine = tuple(r * self.oversample for r in self.resolution)
        self.grid, grid_to_bins = bins.refine(fine)
        self.grid_to_bins = frozen(grid_to_bins)
        self.matrix = self.masses(bins)
        self._grid_masses: sp.csr_array | None = None
        self._shares: sp.csr_array | None = None

    @property
    def K(self) -> int:
        """Number of reporting bins."""
        return self.bins.K

    @property
    def C(self) -> int:
        """Number of components."""
        return self.components.K

    @property
    def is_identity(self) -> bool:
        """True when the components are the bins themselves (order 0, resolution 1)."""
        return self.order == 0 and self.C == self.K

    @property
    def grid_masses(self) -> sp.csr_array:
        """``masses(grid)``: the components' masses in the quadrature cells (cached)."""
        if self._grid_masses is None:
            self._grid_masses = self.masses(self.grid)
        return self._grid_masses

    @property
    def shares(self) -> sp.csr_array:
        """``(C, K)``: each component's mass in each reporting bin (rows sum to one; cached)."""
        if self._shares is None:
            self._shares = sp.csr_array(self.matrix.T)
        return self._shares

    def axis_masses(self, grid: Bins) -> list[sp.csr_array]:
        """Per-axis masses of the components' factors in ``grid``'s intervals (no mask).

        On an unmasked grid :meth:`masses` is their Kronecker product, so
        separable inputs can be projected one axis at a time.
        """
        comp = self.components
        if grid.is_categorical or grid.ndim != comp.ndim or grid.measures != comp.measures:
            raise ValueError("grid must be a grid with the components' measures.")
        out = []
        for u_grid, u_comp in zip(grid.u_edges, comp.u_edges, strict=True):
            if not np.allclose([u_grid[0], u_grid[-1]], [u_comp[0], u_comp[-1]]):
                raise ValueError("grid must cover the same range as the basis.")
            out.append(_axis_masses(u_grid, u_comp, self.order))
        return out

    def masses(self, grid: Bins) -> sp.csr_array:
        r"""``(grid.K, C)`` sparse masses :math:`M_{qk} = \int_{B_q} \phi_k\,du`.

        ``grid`` must cover the components' range with the same measures (the
        reporting bins, the components or any refinement of them). Each column
        sums to one over the active cells of ``grid``.
        """
        M = None
        for m in self.axis_masses(grid):
            M = m if M is None else sp.kron(M, m, format="csr")
        M = sp.csr_array(M)[grid.cells][:, self.components.cells]
        tot = np.asarray(M.sum(axis=0)).ravel()  # below one where a hat crosses a mask
        M.data /= tot[M.indices]
        for a in (M.data, M.indices, M.indptr):  # shared by every object on this basis
            frozen(a)
        return M

    def support(self, k: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(lo, hi)`` in ``u`` of the supports of components ``k``, each ``(n, ndim)``."""
        k = np.asarray(k).ravel()
        lo, hi = [], []
        for u, j in zip(self.components.u_edges, self.components.grid_index(k), strict=True):
            a, _, b, _, _ = _hat(u, j) if self.order == 1 else (u[j], 0, u[j + 1], 0, 0)
            lo.append(a)
            hi.append(b)
        return np.stack(lo, axis=-1), np.stack(hi, axis=-1)

    def height(self, k: Any, d: int, u: Any) -> np.ndarray:
        """Height (at most one) along axis ``d`` of the unnormalised components ``k`` at
        ``u``; ``u`` has ``k``'s shape, or one more trailing axis of points per component.

        A component's shape is the product of its heights over the axes (one for
        flat components inside their cell).
        """
        k = np.asarray(k)
        x = np.asarray(u, dtype=float)
        if self.order == 0:
            return np.ones(x.shape)
        j = self.components.grid_index(k)[d].reshape(k.shape + (1,) * (x.ndim - k.ndim))
        a, mid, b, first, last = _hat(self.components.u_edges[d], j)
        with np.errstate(divide="ignore", invalid="ignore"):
            up = np.where(first, 1.0, (x - a) / (mid - a))
            down = np.where(last, 1.0, (b - x) / (b - mid))
        return np.clip(np.where(x < mid, up, down), 0.0, 1.0)

    def to_bins(self, weights: Any) -> np.ndarray:
        """Map component weights ``(..., C)`` to reporting-bin fractions ``(..., K)``."""
        w = np.asarray(weights, dtype=float)
        return np.asarray(self.matrix @ w.reshape(-1, self.C).T).T.reshape(*w.shape[:-1], self.K)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Basis):
            return NotImplemented
        return (
            self.bins == other.bins
            and self.order == other.order
            and self.resolution == other.resolution
            and self.oversample == other.oversample
        )

    def __hash__(self) -> int:
        return hash((self.bins, self.order, self.resolution, self.oversample))

    def __reduce__(self):
        return Basis, (self.bins, self.order, self.resolution, self.oversample)

    def __repr__(self) -> str:
        res = self.resolution[0] if len(set(self.resolution)) == 1 else self.resolution
        return f"Basis(order={self.order}, resolution={res}, K={self.K}, C={self.C})"


def _axis_masses(u_grid: np.ndarray, u_comp: np.ndarray, order: int) -> sp.csr_array:
    """Masses of one axis's components in the grid intervals, ``(n_grid, n_comp)``, sparse.

    Each entry is the difference of the component's antiderivative at the
    interval's ends (exact), for the components whose support meets the
    interval; columns are normalised to unit total.
    """
    n = len(u_comp) - 1
    a, b = u_grid[:-1], u_grid[1:]
    j0 = np.clip(np.searchsorted(u_comp, a, side="right") - 1, 0, n - 1)
    j1 = np.clip(np.searchsorted(u_comp, b, side="left") - 1, 0, n - 1)
    lo, hi = (j0, j1) if order == 0 else (np.maximum(j0 - 1, 0), np.minimum(j1 + 1, n - 1))
    span = hi - lo + 1
    rows = np.repeat(np.arange(a.size), span)
    cols = np.arange(span.sum()) - np.repeat(np.cumsum(span) - span, span) + lo[rows]
    m = _antiderivative(u_comp, cols, b[rows], order) - _antiderivative(
        u_comp, cols, a[rows], order
    )
    total = _antiderivative(u_comp, np.arange(n), np.full(n, u_comp[-1]), order)
    keep = m > 1e-14 * total[cols]  # round-off outside the support
    return sp.csr_array((m[keep] / total[cols[keep]], (rows[keep], cols[keep])), shape=(a.size, n))


def _antiderivative(u_comp: np.ndarray, k: np.ndarray, x: np.ndarray, order: int) -> np.ndarray:
    """Integral from the range's start to ``x`` of component ``k``'s unnormalised shape."""
    if order == 0:
        return np.clip(x, u_comp[k], u_comp[k + 1]) - u_comp[k]
    a, mid, b, first, last = _hat(u_comp, k)
    y = np.clip(x, a, mid)
    with np.errstate(divide="ignore", invalid="ignore"):
        up = np.where(first, y - a, (y - a) ** 2 / (2 * (mid - a)))
        y = np.clip(x, mid, b)
        down = np.where(last, y - mid, ((b - mid) ** 2 - (b - y) ** 2) / (2 * (b - mid)))
    return up + down


def _hat(u: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, ...]:
    """Support ``(a, b)``, peak and first/last flags of the hats of an axis's cells ``j``.

    A hat peaks at its cell's centre and falls to zero at the neighbouring
    centres; the first and last stay at one out to the range's edges.
    """
    c, n = 0.5 * (u[1:] + u[:-1]), len(u) - 1
    first, last = j == 0, j == n - 1
    a = np.where(first, u[0], c[np.maximum(j - 1, 0)])
    b = np.where(last, u[-1], c[np.minimum(j + 1, n - 1)])
    return a, c[j], b, first, last
