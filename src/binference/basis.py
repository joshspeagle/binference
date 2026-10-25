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
has no steps at bin edges. Grids use tensor products; on masked grids only
active cells count and each component is renormalised over them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp

from .bins import Bins

__all__ = ["Basis"]

#: quadrature cells per component and axis for order 1 (likelihoods, selection)
DEFAULT_OVERSAMPLE = 8


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
        Quadrature cells per component and axis used to integrate likelihoods
        and selection functions against the components (default 1 for order 0,
        where the integrals are exact, and 8 for order 1).

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
        if oversample is None:
            oversample = 1 if order == 0 else DEFAULT_OVERSAMPLE
        if not float(oversample).is_integer() or oversample < 1:
            raise ValueError(f"oversample must be a positive integer; got {oversample!r}.")
        self.bins = bins
        self.order = int(order)
        self.oversample = int(oversample)
        self.components, _ = bins.refine(resolution)
        self.resolution = tuple(
            (len(c) - 1) // (len(b) - 1)
            for c, b in zip(self.components.edges, bins.edges, strict=True)
        )
        fine = tuple(r * self.oversample for r in self.resolution)
        self.grid, self.grid_to_bins = bins.refine(fine)
        self.matrix = self.masses(bins)
        self._grid_masses: sp.csr_array | None = None

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

    def masses(self, grid: Bins) -> sp.csr_array:
        r"""``(grid.K, C)`` sparse masses :math:`M_{qk} = \int_{B_q} \phi_k\,du`.

        ``grid`` must cover the components' range with the same measures (the
        reporting bins, the components or any refinement of them). Each column
        sums to one over the active cells of ``grid``.
        """
        comp = self.components
        if grid.is_categorical or grid.ndim != comp.ndim or grid.measures != comp.measures:
            raise ValueError("grid must be a grid with the components' measures.")
        M = None
        for u_grid, u_comp in zip(grid.u_edges, comp.u_edges, strict=True):
            if not np.allclose([u_grid[0], u_grid[-1]], [u_comp[0], u_comp[-1]]):
                raise ValueError("grid must cover the same range as the basis.")
            m = _axis_masses(u_grid, u_comp, self.order)
            M = m if M is None else sp.kron(M, m, format="csr")
        M = sp.csr_array(M)[grid.cells][:, comp.cells]
        tot = np.asarray(M.sum(axis=0)).ravel()  # below one where a hat crosses a mask
        return sp.csr_array(M @ sp.diags_array(1.0 / tot))

    def support(self, k: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(lo, hi)`` in ``u`` of the supports of components ``k``, each ``(n, ndim)``."""
        k = np.asarray(k).ravel()
        lo, hi = [], []
        for u, j in zip(self.components.u_edges, self.components.grid_index(k), strict=True):
            if self.order == 0:
                lo.append(u[j])
                hi.append(u[j + 1])
                continue
            c, n = 0.5 * (u[1:] + u[:-1]), len(u) - 1
            lo.append(np.where(j == 0, u[0], c[np.maximum(j - 1, 0)]))
            hi.append(np.where(j == n - 1, u[-1], c[np.minimum(j + 1, n - 1)]))
        return np.stack(lo, axis=-1), np.stack(hi, axis=-1)

    def shape(self, k: Any, u: Any) -> np.ndarray:
        """Heights (at most one) of the unnormalised components ``k`` at points ``u``."""
        k = np.asarray(k).ravel()
        u = np.asarray(u, dtype=float).reshape(k.size, -1)
        out = np.ones(k.size)
        if self.order == 0:
            return out
        for d, (ue, j) in enumerate(
            zip(self.components.u_edges, self.components.grid_index(k), strict=True)
        ):
            c, n, x = 0.5 * (ue[1:] + ue[:-1]), len(ue) - 1, u[:, d]
            prev, nxt = c[np.maximum(j - 1, 0)], c[np.minimum(j + 1, n - 1)]
            with np.errstate(divide="ignore", invalid="ignore"):
                up = np.where(j == 0, 1.0, (x - prev) / (c[j] - prev))
                down = np.where(j == n - 1, 1.0, (nxt - x) / (nxt - c[j]))
            out *= np.clip(np.where(x < c[j], up, down), 0.0, 1.0)
        return out

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
    left, right = u_comp[k], u_comp[k + 1]
    if order == 0:
        return np.clip(x, left, right) - left
    n = len(u_comp) - 1
    c = 0.5 * (u_comp[1:] + u_comp[:-1])
    mid = c[k]
    first, last = k == 0, k == n - 1
    a = np.where(first, u_comp[0], c[np.maximum(k - 1, 0)])  # support of the hat
    b = np.where(last, u_comp[-1], c[np.minimum(k + 1, n - 1)])
    y = np.clip(x, a, mid)
    with np.errstate(divide="ignore", invalid="ignore"):
        up = np.where(first, y - a, (y - a) ** 2 / (2 * (mid - a)))
        y = np.clip(x, mid, b)
        down = np.where(last, y - mid, ((b - mid) ** 2 - (b - y) ** 2) / (2 * (b - mid)))
    return up + down
