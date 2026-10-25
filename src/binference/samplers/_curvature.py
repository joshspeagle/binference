"""Finite-difference curvature of a log density, shared by NUTS and Laplace."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

LogpGrad = Callable[[np.ndarray], tuple[float, np.ndarray]]


def _step(x: np.ndarray, j: int, h: float) -> np.ndarray:
    e = np.zeros(x.size)
    e[j] = h * max(1.0, abs(x[j]))
    return e


def fd_hessian(fn: LogpGrad, x: np.ndarray, h: float = 1e-4, cols=None) -> np.ndarray:
    """Symmetrised central-difference Hessian of a log density from its gradient.

    ``cols`` restricts the computation to some columns (returns ``(dim, len(cols))``).
    """
    x = np.asarray(x, dtype=float)
    idx = range(x.size) if cols is None else cols
    H = np.empty((x.size, len(idx)))
    for c, j in enumerate(idx):
        e = _step(x, j, h)
        H[:, c] = (fn(x + e)[1] - fn(x - e)[1]) / (2.0 * e[j])
    if cols is None:
        H = 0.5 * (H + H.T)
    return H


def fd_hessian_diag(fn: LogpGrad, x: np.ndarray, idx: Sequence[int], h: float = 1e-4) -> np.ndarray:
    """The diagonal entries ``idx`` of the Hessian, without storing its columns."""
    x = np.asarray(x, dtype=float)
    out = np.empty(len(idx))
    for c, j in enumerate(idx):
        e = _step(x, j, h)
        out[c] = (fn(x + e)[1][j] - fn(x - e)[1][j]) / (2.0 * e[j])
    return out


def neg_hessian_eigen(H: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """Eigenvalues (clipped below at ``floor`` times the largest) and vectors of ``-H``.

    Also returns whether any eigenvalue was clipped; ``None`` if ``H`` is unusable
    (not finite, or no direction of positive curvature).
    """
    A = -0.5 * (H + H.T)
    if not np.all(np.isfinite(A)):
        return None
    lam, U = np.linalg.eigh(A)
    top = lam.max()
    if not top > 0:
        return None
    clipped = bool(np.any(lam < floor * top))
    return np.maximum(lam, floor * top), U, clipped


def covariance_from_hessian(H: np.ndarray, floor: float = 1e-6) -> np.ndarray | None:
    """A positive-definite covariance from a (possibly indefinite) Hessian of log p.

    Eigenvalues of ``-H`` are clipped below at ``floor`` times the largest, so
    directions with no or negative curvature get the largest variance seen.
    Returns ``None`` if the Hessian is unusable.
    """
    eig = neg_hessian_eigen(H, floor)
    if eig is None:
        return None
    lam, U, _ = eig
    cov = (U / lam) @ U.T
    return 0.5 * (cov + cov.T)
