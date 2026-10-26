r"""Logistic-normal (smooth) priors: Gaussian priors on log-fractions.

Fractions are :math:`\rho = \mathrm{softmax}(\eta)` with a Gaussian prior on
:math:`\eta`, written in whitened ("non-centred") form

.. math:: \eta = \mu + B\,(s(\theta) \odot \xi), \qquad \xi \sim \mathcal N(0, I),

so that the hyperparameters :math:`\theta` (scales, length scales) are sampled
jointly with :math:`\xi` by gradient-based engines. ``B`` is a fixed basis:

* :class:`LogisticNormal` -- eigenvectors of a fixed covariance (or the
  zero-sum Helmert basis for i.i.d. shrinkage);
* :class:`LogisticRandomWalk` -- eigenvectors of a random-walk / ICAR precision
  on the grid (Kronecker structure for N-D grids);
* :class:`LogisticGP` -- the Hilbert-space Gaussian-process basis of Solin &
  Sarkka (2020) on 1-D/N-D grids.

The softmax is invariant to adding a constant to :math:`\eta`; the constant
mode is dropped wherever it is explicit.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import special

from .._simplex import helmert_basis, softmax
from .._utils import check_positive, warn
from ..bins import Bins
from ..density import Density
from ..warnings import InputWarning
from .base import BoundPrior, PopulationPrior, _per_item
from .hyper import Fixed, HalfNormal, Hyperprior, InverseGamma, LogNormal, as_hyperprior

__all__ = ["LogisticGP", "LogisticNormal", "LogisticRandomWalk"]


def _require_full_grid(bins: Bins) -> None:
    if bins.is_categorical:
        raise ValueError("Smoothness priors need ordered (grid) bins, not categories.")
    if bins.is_masked:
        raise NotImplementedError("Smoothness priors do not yet support masked grids.")


# ----------------------------------------------------------------------------
# bases
# ----------------------------------------------------------------------------
class _KronBasis:
    """Matrix-free product basis ``B = F_1 (x) F_2 (x) ...`` (C-order cells)."""

    def __init__(self, factors: Sequence[np.ndarray]) -> None:
        self.factors = [np.asarray(f, dtype=float) for f in factors]
        self.in_shape = tuple(f.shape[1] for f in self.factors)
        self.out_shape = tuple(f.shape[0] for f in self.factors)
        self.J = int(np.prod(self.in_shape))
        self.K = int(np.prod(self.out_shape))

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """``B @ v`` for ``v`` of shape (J,), or row by row for shape (n, J)."""
        if len(self.factors) == 1:
            return v @ self.factors[0].T
        t = v.reshape(*v.shape[:-1], *self.in_shape)
        lead = v.ndim - 1
        for d, f in enumerate(self.factors):
            t = np.moveaxis(np.tensordot(f, t, axes=([1], [lead + d])), 0, lead + d)
        return t.reshape(*v.shape[:-1], self.K)

    def rmatvec(self, u: np.ndarray) -> np.ndarray:
        if len(self.factors) == 1:
            return self.factors[0].T @ u
        t = u.reshape(self.out_shape)
        for d, f in enumerate(self.factors):
            t = np.moveaxis(np.tensordot(f.T, t, axes=([1], [d])), 0, d)
        return t.reshape(self.J)


def _orthonormal_polys(n: int, degree: int) -> np.ndarray:
    """Orthonormal basis of polynomials of degree < ``degree`` on 0..n-1 (columns)."""
    x = np.arange(n, dtype=float)
    x = (x - x.mean()) / max(1.0, x.std())
    P = np.stack([x**p for p in range(degree)], axis=1)
    q, _ = np.linalg.qr(P)
    # make the constant column positive for reproducibility
    q *= np.sign(q[0, 0]) if q[0, 0] != 0 else 1.0
    return q


def _rw_precision(n: int, order: int) -> np.ndarray:
    """The 1-D random-walk precision ``D_r^T D_r`` of the ``order``-th differences."""
    if n <= order:
        raise ValueError(f"A random walk of order {order} needs more than {order} bins per axis.")
    D = np.diff(np.eye(n), n=order, axis=0)
    return D.T @ D


def _rw_eigen(n: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of the 1-D random-walk precision ``D_r^T D_r``.

    The first ``order`` eigenvectors are replaced by an explicit orthonormal
    polynomial basis of the null space (constant first), with eigenvalue 0.
    """
    Q = _rw_precision(n, order)
    _, U = np.linalg.eigh(Q)
    null = _orthonormal_polys(n, order)
    # project the non-null eigenvectors off the polynomial null space for safety
    U_rest = U[:, order:]
    U_rest = U_rest - null @ (null.T @ U_rest)
    U_rest, _ = np.linalg.qr(U_rest)
    U_rest = U_rest[:, : n - order]
    lam_rest = np.sum(U_rest * (Q @ U_rest), axis=0)  # Rayleigh quotients
    return np.concatenate([np.zeros(order), lam_rest]), np.concatenate([null, U_rest], axis=1)


# ----------------------------------------------------------------------------
# bound class shared by the logistic family
# ----------------------------------------------------------------------------
class _BoundLogistic(BoundPrior):
    kind = "fractions"
    conjugate = False
    has_density = False
    has_centering = True

    def __init__(
        self,
        prior: PopulationPrior,
        bins: Bins,
        basis: _KronBasis,
        active: np.ndarray,
        mu: np.ndarray,
        hypers: list[tuple[str, Hyperprior]],
    ) -> None:
        super().__init__(prior, bins)
        self.basis = basis
        self.active = np.asarray(active, dtype=np.int64)
        self.n_active = int(self.active.size)
        self.mu = np.asarray(mu, dtype=float)
        self.hypers = hypers
        self.free = [i for i, (_, h) in enumerate(hypers) if not h.fixed]
        self.n_free = len(self.free)
        self.dim = self.n_active + self.n_free
        self.hyper_names = [hypers[i][0] for i in self.free]
        # hyperparameter values with the fixed ones in place (the free ones are filled per x)
        self._theta0 = np.array([h.median() if h.fixed else 0.0 for _, h in hypers])
        # Partial non-centring: coefficient j is c_j = s_j**lam_j * x_j with
        # x_j ~ N(0, s_j**(2 (1 - lam_j))); lam = 1 is the whitened
        # (non-centred) form, lam = 0 the centred one.
        self.lam = np.ones(self.n_active)

    def with_centering(self, lam: np.ndarray) -> _BoundLogistic:
        """A copy using partial non-centring weights ``lam`` (1 = non-centred, 0 = centred)."""
        lam = np.clip(np.asarray(lam, dtype=float), 0.0, 1.0)
        if lam.shape != (self.n_active,):
            raise ValueError(f"lam must have shape ({self.n_active},).")
        new = copy.copy(self)
        new.lam = lam
        return new

    def recentre(self, x: np.ndarray, lam: np.ndarray) -> np.ndarray:
        """Map unconstrained ``x`` in this parametrisation to the one with weights ``lam``."""
        x = np.array(x, dtype=float, copy=True)
        log_s, _ = self.log_scales(self._theta(x[self.n_active :]))
        z = x[: self.n_active]
        with np.errstate(divide="ignore", over="ignore"):  # in log space, as the density
            z[:] = np.sign(z) * np.exp(np.log(np.abs(z)) + (self.lam - np.clip(lam, 0, 1)) * log_s)
        return x

    def to_noncentred(self, X: np.ndarray, log_density: np.ndarray):
        """Points ``X`` (rows) in this parametrisation, and their log densities, in the
        non-centred one (all weights 1).

        The log density gains the Jacobian ``sum_j (1 - lam_j) log s_j(theta)``.
        """
        X = np.array(X, dtype=float, copy=True)
        na = self.n_active
        LS = np.array([self.log_scales(self._theta(x[na:]))[0] for x in X]).reshape(-1, na)
        one_m = 1.0 - self.lam
        z = X[:, :na]
        with np.errstate(divide="ignore", over="ignore"):  # in log space, as the density
            z[:] = np.sign(z) * np.exp(np.log(np.abs(z)) - one_m * LS)
        return X, np.asarray(log_density, dtype=float) + LS @ one_m

    # subclasses implement: log_scales(theta) -> (log s, dlogs), dlogs (n_hypers, n_active);
    # logarithms, because the scales of high-frequency modes underflow (audit P-5)
    def log_scales(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    def _theta(self, u: np.ndarray) -> np.ndarray:
        theta = self._theta0.copy()
        if self.n_free:
            # clipping keeps absurd excursions (e.g. in a diverging trajectory) finite
            theta[self.free] = np.exp(np.clip(u, -350.0, 350.0))
        return theta

    def theta_from_x(self, x: np.ndarray) -> np.ndarray:
        """All hyperparameter values (fixed and free) at unconstrained ``x``."""
        return self._theta(np.asarray(x)[self.n_active :])

    def _eta(self, c: np.ndarray) -> np.ndarray:
        """Log-fractions from basis coefficients ``c`` on the active modes (last axis)."""
        v = np.zeros((*c.shape[:-1], self.basis.J))
        v[..., self.active] = c
        return self.mu + self.basis.matvec(v)

    def forward(self, x):
        z = x[: self.n_active]
        u = x[self.n_active :]
        log_s, dlogs = self.log_scales(self._theta(u))
        m = np.exp(self.lam * log_s)
        c = m * z
        return self._eta(c), (z, log_s, m, c, dlogs, u)

    def log_prior_and_grad(self, x, cache, value, d_eta):
        z, log_s, m, c, dlogs, u = cache
        if u.size and np.max(np.abs(u)) > 350.0:  # outside the clipped range of _theta
            return -np.inf, np.zeros(self.dim)
        bt = self.basis.rmatvec(d_eta)[self.active]
        lam = self.lam
        one_m = 1.0 - lam
        # prior z_j ~ N(0, s_j**(2 (1 - lam_j))). From the log scales, so that value and
        # gradient agree however small a scale is (audit P-5), and in log space, since a
        # centred coefficient underflows to zero with its scale
        if one_m.any():
            with np.errstate(divide="ignore", over="ignore"):
                a = np.log(np.abs(z)) - one_m * log_s  # log |z_j| / s_j**(1 - lam_j)
                q = np.exp(2.0 * a)
                z_prec = np.sign(z) * np.exp(a - one_m * log_s)  # z_j / s_j**(2 (1 - lam_j))
        else:  # fully non-centred: the same values, without the logarithms
            q, z_prec = z * z, z
        log_s_term = float(one_m @ log_s)
        lp = -0.5 * float(q.sum()) - log_s_term
        if not np.isfinite(lp):
            return -np.inf, np.zeros(self.dim)
        lp -= 0.5 * self.n_active * np.log(2 * np.pi)
        g_z = m * bt - z_prec
        g_u = np.empty(self.n_free)
        common = lam * bt * c + one_m * (q - 1.0)
        for j, i in enumerate(self.free):
            h = self.hypers[i][1]
            lp += h.logpdf_u(u[j])
            g_u[j] = float(common @ dlogs[i]) + h.dlogpdf_u(u[j])
        return lp, np.concatenate([g_z, g_u])

    def init_from_value(self, value: np.ndarray) -> np.ndarray:
        r = np.asarray(value, dtype=float)
        r = np.maximum(r, 1e-8 * np.max(r))
        target = np.log(r / r.sum())
        target = target - target.mean() - (self.mu - self.mu.mean())
        theta0 = np.array([h.median() for _, h in self.hypers])
        s0 = np.exp(self.log_scales(theta0)[0])

        def A(z):
            out = self._eta(s0 * z) - self.mu
            return out - out.mean()

        def AT(y):
            y = y - y.mean()
            return s0 * self.basis.rmatvec(y)[self.active]

        # conjugate gradients on (A^T A + I) xi = A^T target (prior acts as ridge)
        b = AT(target)
        xi = np.zeros(self.n_active)
        r_ = b - (AT(A(xi)) + xi)
        p = r_.copy()
        rs = float(r_ @ r_)
        for _ in range(200):
            if rs < 1e-20 * max(1.0, float(b @ b)):
                break
            Ap = AT(A(p)) + p
            step = rs / float(p @ Ap)
            xi = xi + step * p
            r_ = r_ - step * Ap
            rs_new = float(r_ @ r_)
            p = r_ + (rs_new / rs) * p
            rs = rs_new
        u0 = np.log(theta0[self.free]) if self.n_free else np.empty(0)
        return np.concatenate([xi * s0 ** (1.0 - self.lam), u0])

    def sample(self, n, rng):
        theta = np.stack([h.sample(n, rng) for _, h in self.hypers], axis=1)
        log_s = np.array([self.log_scales(t)[0] for t in theta]).reshape(n, self.n_active)
        s = np.exp(log_s)
        return softmax(self._eta(s * rng.standard_normal((n, self.n_active))), axis=-1)


def _check_mean(mean: Any) -> Any:
    if mean is not None and not np.all(np.isfinite(np.asarray(mean, dtype=float))):
        raise ValueError("mean must be finite.")
    return mean


def _resolve_mean(mean: Any, bins: Bins) -> np.ndarray:
    """Prior mean of the log-fractions, centred; default: flat in the measure coordinate."""
    K = bins.K
    if mean is None:
        m = np.log(Density.flat().masses(bins))
        return m - m.mean()
    m = np.asarray(mean, dtype=float).ravel()
    if m.shape != (K,):
        raise ValueError(f"mean must have K={K} values; got {m.size}.")
    return m - m.mean()


# ----------------------------------------------------------------------------
# LogisticNormal
# ----------------------------------------------------------------------------
class LogisticNormal(PopulationPrior):
    r"""Logistic-normal prior on the population fractions.

    :math:`\eta \sim \mathcal N(\mu, \Sigma)` and :math:`\rho = \mathrm{softmax}(\eta)`.

    Parameters
    ----------
    mean : array_like (K,), optional
        Prior mean of the log-fractions (up to a constant), e.g. the log of
        reference fractions. Defaults to a density flat in the measure
        coordinate, so the prior acts on the log-density whatever the binning.
    cov : array_like (K, K) or float, optional
        Covariance of :math:`\eta`: a positive variance, or a symmetric matrix
        that is positive semi-definite on zero-sum vectors. Only that zero-sum
        part matters, since adding a constant to :math:`\eta` leaves
        :math:`\rho` unchanged. ``None`` means i.i.d. components with standard
        deviation ``scale``.
    scale : Hyperprior or float
        Multiplies the standard deviations. Defaults to ``HalfNormal(2.0)``
        (inferred) when ``cov`` is ``None`` and to ``Fixed(1.0)`` otherwise.
    """

    def __init__(self, mean: Any = None, cov: Any = None, scale: Any = None) -> None:
        self.mean = _check_mean(mean)
        self.cov = cov
        # basis and scales of a covariance matrix, or None (i.i.d. with a common variance)
        self._eig = None if cov is None else _zero_sum_eigen(cov)
        if scale is None:
            scale = HalfNormal(2.0) if cov is None else Fixed(1.0)
        self.scale = as_hyperprior(scale)

    def bind(self, bins: Bins) -> _BoundLogisticNormal:
        K = bins.K
        mu = _resolve_mean(self.mean, bins)
        if self._eig is None:
            base = 1.0 if self.cov is None else float(np.sqrt(self.cov))
            B = helmert_basis(K)
            s0 = np.full(K - 1, base)
        else:
            B, s0 = self._eig
            if B.shape[0] != K:
                raise ValueError(
                    f"cov must be a (K, K) array with K={K}; got {np.shape(self.cov)}."
                )
        basis = _KronBasis([np.asarray(B)])
        return _BoundLogisticNormal(
            self, bins, basis, np.arange(B.shape[1]), mu, [("scale", self.scale)], s0
        )


def _zero_sum_eigen(cov: Any) -> tuple[np.ndarray, np.ndarray] | None:
    # validates cov; for a matrix, the eigenvectors and square-root eigenvalues of its
    # zero-sum part, on its numerical rank
    C = np.asarray(cov, dtype=float)
    if C.ndim == 0:
        check_positive(C, "A scalar cov (a variance)")
        return None
    if not np.all(np.isfinite(C)):
        raise ValueError("cov must be finite.")
    if C.ndim != 2 or C.shape[0] != C.shape[1] or not np.allclose(C, C.T):
        raise ValueError("cov must be a positive number or a symmetric (K, K) array.")
    K = C.shape[0]
    P = np.eye(K) - 1.0 / K
    lam, U = np.linalg.eigh(P @ C @ P)
    # a constant matrix leaves only round-off, so judge against the size of cov itself
    if lam.max() <= 1e-12 * np.abs(C).max() or lam.min() < -1e-8 * lam.max():
        raise ValueError(
            "cov must be positive semi-definite on zero-sum vectors and not constant."
        )
    keep = lam > 1e-10 * lam.max()
    return U[:, keep], np.sqrt(lam[keep])


class _BoundLogisticNormal(_BoundLogistic):
    def __init__(self, prior, bins, basis, active, mu, hypers, s0):
        super().__init__(prior, bins, basis, active, mu, hypers)
        self.log_s0 = np.log(s0)

    def log_scales(self, theta):
        return np.log(theta[0]) + self.log_s0, np.ones((1, self.log_s0.size))


# ----------------------------------------------------------------------------
# LogisticRandomWalk
# ----------------------------------------------------------------------------
class LogisticRandomWalk(PopulationPrior):
    r"""Random-walk (intrinsic GMRF) smoothness prior on log-fractions over a grid.

    The ``order``-th differences of :math:`\eta = \log\rho` (between adjacent
    bins along each axis) are Gaussian with standard deviation ``scale``:
    ``order=1`` penalises jumps, ``order=2`` penalises curvature. N-D grids
    use the Kronecker-sum (ICAR-type) precision. Polynomial modes not
    penalised by the random walk (e.g. a linear trend for ``order=2``) get a
    broad Gaussian prior set by ``trend_scale``. On N-D grids these are the
    products of each axis's unpenalised polynomials, so for ``order=2`` in 2-D
    the surface :math:`xy` is unpenalised too (and :math:`x^2y^2` for
    ``order=3``).

    Smoothness is measured per bin step, so the natural scale of ``scale``
    depends on the binning.

    Parameters
    ----------
    order : {1, 2, 3}
        Order of the differences.
    scale : Hyperprior or float
        Standard deviation of the differences; inferred with a
        ``HalfNormal(1.0)`` prior by default.
    mean : array_like (K,), optional
        Mean of the log-fractions, e.g. ``log(reference)`` to shrink toward
        reference fractions. Defaults to a density flat in the measure
        coordinate, so differences penalise the log-density.
    trend_scale : float
        Prior standard deviation of the root-mean-square size, across the grid,
        of each unpenalised polynomial mode (a linear trend of root-mean-square
        size ``t`` changes the log-density by about ``3.5 t`` end to end),
        whatever the number of bins.
    """

    def __init__(
        self,
        order: int = 2,
        scale: Any = None,
        mean: Any = None,
        trend_scale: float = 3.0,
    ) -> None:
        if order not in (1, 2, 3):
            raise ValueError("order must be 1, 2 or 3.")
        check_positive(trend_scale, "trend_scale")
        self.order = int(order)
        self.scale = as_hyperprior(HalfNormal(1.0) if scale is None else scale)
        self.mean = _check_mean(mean)
        self.trend_scale = float(trend_scale)

    def bind(self, bins: Bins) -> _BoundRandomWalk:
        _require_full_grid(bins)
        eig = [_rw_eigen(n, self.order) for n in bins.shape]
        basis = _KronBasis([U for _, U in eig])
        lam = np.zeros(1)
        null = np.ones(1, dtype=bool)
        for lam_d, _ in eig:
            nd = np.zeros(lam_d.size, dtype=bool)
            nd[: self.order] = True
            lam = (lam[:, None] + lam_d[None, :]).ravel()
            null = (null[:, None] & nd[None, :]).ravel()
        # drop the global constant mode (index 0: product of constants)
        active = np.arange(1, basis.J)
        mu = _resolve_mean(self.mean, bins)
        return _BoundRandomWalk(
            self,
            bins,
            basis,
            active,
            mu,
            [("scale", self.scale)],
            lam[active],
            null[active],
            # the modes have unit norm, so a root-mean-square size of 1/sqrt(K) (audit P-6)
            null_sd=self.trend_scale * np.sqrt(bins.K),
        )


class _BoundRandomWalk(_BoundLogistic):
    def __init__(self, prior, bins, basis, active, mu, hypers, lam, null, null_sd):
        super().__init__(prior, bins, basis, active, mu, hypers)
        self.null = null
        # log s_j = log(tau) - log(lam_j) / 2 on penalised modes, log(null_sd) on the others
        self._pen = np.where(null, 0.0, 1.0)  # 1 on the modes the scale tau multiplies
        self._log_base = np.where(null, np.log(null_sd), -0.5 * np.log(np.where(null, 1.0, lam)))

    def log_scales(self, theta):
        return self._log_base + self._pen * np.log(theta[0]), self._pen[None, :]


# ----------------------------------------------------------------------------
# LogisticGP (Hilbert-space approximation)
# ----------------------------------------------------------------------------
# kernel: (Matérn smoothness nu, or None for the RBF; f), where the shortest length scale m
# sines on a box of half-width c*S resolve is f * c * S / m: the centred field SD within 5%
# and the covariance within 5% of the variance of the exact kernel, measured for ell/S from
# 0.02 to 0.2 (DECISIONS #73)
_KERNELS = {
    "rbf": (None, 1.3),
    "se": (None, 1.3),
    "matern52": (2.5, 1.8),
    "matern32": (1.5, 2.2),
    "matern12": (0.5, 8.5),
}


class LogisticGP(PopulationPrior):
    r"""Gaussian-process prior on log-fractions over a 1-D or N-D grid.

    Uses the Hilbert-space reduced-rank approximation (Solin & Sarkka 2020):
    a fixed sine basis on a box ``boundary_factor`` times larger than the grid,
    with weights scaled by the kernel's spectral density, so amplitude and
    (per-axis) length scales are inferred jointly with the field. The field
    is stationary in the measure coordinate of the bins (log z for
    :meth:`Bins.log <binference.Bins.log>`), so length scales are in that
    coordinate.

    Parameters
    ----------
    kernel : {"matern32", "matern52", "matern12", "rbf"}
        Covariance kernel (``"se"`` is an alias of ``"rbf"``).
    amplitude : Hyperprior or float
        Marginal standard deviation of the log-density field; inferred with a
        ``HalfNormal(2.0)`` prior by default.
    length_scale : Hyperprior, float, or sequence or array (one per axis)
        Length scale(s) in the measure coordinate. Default: per axis, an
        inverse gamma with 1% prior mass below the shortest length scale the
        grid and the basis resolve (the larger of two bin widths and
        ``f * boundary_factor * S / n_basis``, with ``S`` half the range and
        ``f`` = 1.3, 1.8, 2.2 and 8.5 for the RBF, Matérn-5/2, -3/2 and -1/2
        kernels) and 1% above the axis range, so the field is neither rougher
        than the prior can represent nor flatter than the data can constrain.
        Where that lower bound exceeds a quarter of the range (coarse grids),
        a ``LogNormal`` with standard deviation 1 in log length and median the
        larger of the bound and half the range instead. A fixed length scale
        below the bound raises an :class:`~binference.InputWarning`.
    mean : array_like (K,), optional
        Mean log-fractions (e.g. the log of reference fractions). Defaults to
        a density flat in the measure coordinate: the GP acts on the
        log-density.
    n_basis : int or sequence, optional
        Basis functions per axis. Default: one per bin, up to a total of 1024
        (``min(K_d, floor(1024 ** (1 / D)))``: up to 1024 on 1-D grids, 32 per
        axis on 2-D grids); increase it for shorter length scales, which also
        lowers the default length-scale prior.
    boundary_factor : float
        Ratio of the approximation box to the grid half-width (default 3: the
        centred covariance is within 4% of the exact kernel's for length
        scales up to the range, where 1.5 pulled the edge bins towards the
        mean).
    """

    def __init__(
        self,
        kernel: str = "matern32",
        amplitude: Any = None,
        length_scale: Any = None,
        mean: Any = None,
        n_basis: int | Sequence[int] | None = None,
        boundary_factor: float = 3.0,
    ) -> None:
        if kernel not in _KERNELS:
            raise ValueError(f"kernel must be one of {sorted(_KERNELS)}.")
        if not boundary_factor > 1.0:
            raise ValueError("boundary_factor must exceed 1.")
        if n_basis is not None:
            nb = np.asarray(n_basis)
            if nb.ndim > 1 or nb.dtype.kind not in "iu" or np.any(nb < 1):
                raise ValueError("n_basis must be a positive integer or one per axis.")
        self.kernel = kernel
        self.amplitude = as_hyperprior(HalfNormal(2.0) if amplitude is None else amplitude)
        self.length_scale = length_scale
        per_axis = isinstance(length_scale, list | tuple) or np.ndim(length_scale) == 1
        if length_scale is None:
            self._ls = None
        elif per_axis:
            self._ls = [as_hyperprior(v) for v in length_scale]
        else:
            self._ls = as_hyperprior(length_scale)
        self.mean = _check_mean(mean)
        self.n_basis = n_basis
        self.boundary_factor = float(boundary_factor)

    def bind(self, bins: Bins) -> _BoundGP:
        _require_full_grid(bins)
        D = bins.ndim
        if self.n_basis is None:  # one per bin, at most 1024 modes in all
            nb = [min(n, int(1024 ** (1 / D) + 1e-9)) for n in bins.shape]
        else:
            nb = list(_per_item(self.n_basis, D, "n_basis (one per axis)"))
        factors, omegas = [], []
        for d, u in enumerate(bins.u_edges):  # in the measure coordinate (audit P-1)
            lo, hi = float(u[0]), float(u[-1])
            c, L = 0.5 * (lo + hi), self.boundary_factor * 0.5 * (hi - lo)
            x = 0.5 * (u[1:] + u[:-1])
            j = np.arange(1, nb[d] + 1)
            factors.append(
                np.sin(np.pi * j[None, :] * (x[:, None] - c + L) / (2 * L)) / np.sqrt(L)
            )
            omegas.append(np.pi * j / (2 * L))
        basis = _KronBasis(factors)
        grids = np.meshgrid(*omegas, indexing="ij")
        omega = np.stack([g.ravel() for g in grids], axis=1)  # (J, D)
        nu, f = _KERNELS[self.kernel]
        # the shortest length scale the basis resolves on each axis
        resolved = [
            f * self.boundary_factor * 0.5 * (u[-1] - u[0]) / m
            for u, m in zip(bins.u_edges, nb, strict=True)
        ]
        if self._ls is None:
            ls_h = [
                _default_length_scale(u, r) for u, r in zip(bins.u_edges, resolved, strict=True)
            ]
        elif isinstance(self._ls, list):
            if len(self._ls) != D:
                raise ValueError(f"Need one length scale per axis ({D}).")
            ls_h = self._ls
        else:
            ls_h = [self._ls] * D
        for d, (h, r) in enumerate(zip(ls_h, resolved, strict=True)):
            if h.fixed and h.median() < r:
                warn(
                    f"length_scale {h.median():g} on axis {d} is below the {r:.3g} that "
                    f"{nb[d]} basis functions resolve; the field loses variance. Raise n_basis.",
                    InputWarning,
                )
        hypers = [("amplitude", self.amplitude)] + [
            (f"length_scale_{d}" if D > 1 else "length_scale", h) for d, h in enumerate(ls_h)
        ]
        mu = _resolve_mean(self.mean, bins)
        return _BoundGP(self, bins, basis, np.arange(basis.J), mu, hypers, omega, nu)


class _BoundGP(_BoundLogistic):
    def __init__(self, prior, bins, basis, active, mu, hypers, omega, nu):
        super().__init__(prior, bins, basis, active, mu, hypers)
        self.omega2 = omega**2  # (J, D)
        self.D = omega.shape[1]
        self.nu = nu
        D = self.D
        if nu is None:
            self._logc = 0.5 * D * np.log(2 * np.pi)
        else:
            self._logc = (
                D * np.log(2.0)
                + 0.5 * D * np.log(np.pi)
                + special.gammaln(nu + D / 2)
                - special.gammaln(nu)
                + nu * np.log(2 * nu)
            )

    def log_scales(self, theta):
        amp = theta[0]
        ell = theta[1 : 1 + self.D]
        a2 = self.omega2 * ell[None, :] ** 2  # (J, D)
        r2 = a2.sum(axis=1)
        if self.nu is None:
            log_su = self._logc - 0.5 * r2
            dlog_dr2 = -0.5 * np.ones_like(r2)
        else:
            k = self.nu + self.D / 2
            log_su = self._logc - k * np.log(2 * self.nu + r2)
            dlog_dr2 = -k / (2 * self.nu + r2)
        log_S = 2 * np.log(amp) + np.sum(np.log(ell)) + log_su
        dlogs = np.empty((1 + self.D, log_S.size))
        dlogs[0] = 1.0
        for d in range(self.D):
            dlogs[1 + d] = 0.5 * (1.0 + 2.0 * a2[:, d] * dlog_dr2)
        return 0.5 * log_S, dlogs


def _default_length_scale(u: np.ndarray, resolved: float) -> Hyperprior:
    # 1% below what the grid and the basis resolve, 1% above the range (audit P-2, P-7)
    span = float(u[-1] - u[0])
    lower = max(2.0 * float(np.min(np.diff(u))), resolved)
    if span < 4.0 * lower:  # an inverse gamma between them would all but fix the length
        return LogNormal(np.log(max(lower, 0.5 * span)), 1.0)
    return InverseGamma.from_quantiles(lower, span)
