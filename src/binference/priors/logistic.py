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
from ..bins import Bins
from .base import BoundPrior, Prior
from .hyper import Fixed, HalfNormal, Hyperprior, InverseGamma, as_hyperprior

__all__ = ["LogisticGP", "LogisticNormal", "LogisticRandomWalk"]


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
        if len(self.factors) == 1:
            return self.factors[0] @ v
        t = v.reshape(self.in_shape)
        for d, f in enumerate(self.factors):
            t = np.moveaxis(np.tensordot(f, t, axes=([1], [d])), 0, d)
        return t.reshape(self.K)

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


def _rw_eigen(n: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of the 1-D random-walk precision ``D_r^T D_r``.

    The first ``order`` eigenvectors are replaced by an explicit orthonormal
    polynomial basis of the null space (constant first), with eigenvalue 0.
    """
    if n <= order:
        raise ValueError(f"A random walk of order {order} needs more than {order} bins per axis.")
    D = np.diff(np.eye(n), n=order, axis=0)
    Q = D.T @ D
    _, U = np.linalg.eigh(Q)
    null = _orthonormal_polys(n, order)
    # project the non-null eigenvectors off the polynomial null space for safety
    U_rest = U[:, order:]
    U_rest = U_rest - null @ (null.T @ U_rest)
    U_rest, _ = np.linalg.qr(U_rest)
    U_rest = U_rest[:, : n - order]
    lam_rest = np.einsum("ij,jk,ki->i", U_rest.T, Q, U_rest)
    return np.concatenate([np.zeros(order), lam_rest]), np.concatenate([null, U_rest], axis=1)


# ----------------------------------------------------------------------------
# bound class shared by the logistic family
# ----------------------------------------------------------------------------
class _BoundLogistic(BoundPrior):
    kind = "fractions"
    conjugate = False
    has_density = False

    def __init__(
        self,
        prior: Prior,
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
        s, _ = self.scales(self._theta(x[self.n_active :]))
        x[: self.n_active] *= s ** (self.lam - np.clip(lam, 0.0, 1.0))
        return x

    # subclasses implement: scales(theta) -> (s, dlogs) with dlogs (n_hypers, n_active)
    def scales(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    def _theta(self, u: np.ndarray) -> np.ndarray:
        theta = np.array([h.median() if h.fixed else 0.0 for _, h in self.hypers])
        if self.n_free:
            # clipping keeps absurd excursions (e.g. in a diverging trajectory) finite
            theta[self.free] = np.exp(np.clip(u, -350.0, 350.0))
        return theta

    def theta_from_x(self, x: np.ndarray) -> np.ndarray:
        """All hyperparameter values (fixed and free) at unconstrained ``x``."""
        return self._theta(np.asarray(x)[self.n_active :])

    def _eta(self, xi: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Log-fractions from whitened coefficients ``xi`` and scales ``s``."""
        v = np.zeros(self.basis.J)
        v[self.active] = s * xi
        return self.mu + self.basis.matvec(v)

    def forward(self, x):
        z = x[: self.n_active]
        u = x[self.n_active :]
        theta = self._theta(u)
        s, dlogs = self.scales(theta)
        if self.lam is None or np.all(self.lam == 1.0):
            m = s
        else:
            m = s**self.lam
        c = m * z
        v = np.zeros(self.basis.J)
        v[self.active] = c
        return self.mu + self.basis.matvec(v), (z, s, m, c, dlogs, u)

    def log_prior_and_grad(self, x, cache, value, d_eta):
        z, s, m, c, dlogs, u = cache
        if u.size and np.max(np.abs(u)) > 350.0:  # outside the clipped range of _theta
            return -np.inf, np.zeros(self.dim)
        bt = self.basis.rmatvec(d_eta)[self.active]
        lam = self.lam
        one_m = 1.0 - lam
        # prior z_j ~ N(0, s_j**(2 (1 - lam_j))). Fully non-centred modes (lam_j = 1)
        # need no scale terms; flooring the scales keeps underflowed ones finite.
        if one_m.any():
            s_safe = np.maximum(s, 1e-150)
            prec = s_safe ** (-2.0 * one_m)
            log_s_term = float(one_m @ np.log(s_safe))
        else:
            prec = np.ones_like(s)
            log_s_term = 0.0
        q = z * z * prec
        lp = -0.5 * float(q.sum()) - log_s_term
        if not np.isfinite(lp):
            return -np.inf, np.zeros(self.dim)
        lp -= 0.5 * self.n_active * np.log(2 * np.pi)
        g_z = m * bt - z * prec
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
        s0, _ = self.scales(theta0)

        def A(z):
            v = np.zeros(self.basis.J)
            v[self.active] = s0 * z
            out = self.basis.matvec(v)
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
        out = np.empty((n, self.K))
        draws = {i: h.sample(n, rng) for i, (_, h) in enumerate(self.hypers)}
        for t in range(n):
            theta = np.array([draws[i][t] for i in range(len(self.hypers))])
            s, _ = self.scales(theta)
            xi = rng.standard_normal(self.n_active)
            out[t] = softmax(self._eta(xi, s))
        return out


def _resolve_mean(mean: Any, bins: Bins) -> np.ndarray:
    """Prior mean of the log-fractions, centred; default: a flat density (log volumes)."""
    K = bins.K
    if mean is None:
        m = np.log(np.asarray(bins.volumes, dtype=float))
        return m - m.mean()
    m = np.asarray(mean, dtype=float).ravel()
    if m.shape != (K,) or not np.all(np.isfinite(m)):
        raise ValueError(f"mean must be a finite array of length K={K}.")
    return m - m.mean()


# ----------------------------------------------------------------------------
# LogisticNormal
# ----------------------------------------------------------------------------
class LogisticNormal(Prior):
    r"""Logistic-normal prior on the population fractions.

    :math:`\eta \sim \mathcal N(\mu, \Sigma)` and :math:`\rho = \mathrm{softmax}(\eta)`.

    Parameters
    ----------
    mean : array_like (K,), optional
        Prior mean of the log-fractions (up to a constant), e.g. the log of
        reference fractions. Defaults to the log bin volumes: a flat density,
        so the prior acts on the log-density whatever the binning.
    cov : array_like (K, K) or float, optional
        Covariance of :math:`\eta`. ``None`` means i.i.d. components with
        standard deviation ``scale``.
    scale : Hyperprior or float
        Multiplies the standard deviations. Defaults to ``HalfNormal(2.0)``
        (inferred) when ``cov`` is ``None`` and to ``Fixed(1.0)`` otherwise.
    """

    def __init__(self, mean: Any = None, cov: Any = None, scale: Any = None) -> None:
        self.mean = mean
        self.cov = cov
        if scale is None:
            scale = HalfNormal(2.0) if cov is None else Fixed(1.0)
        self.scale = as_hyperprior(scale)

    def _params(self):
        return {"cov": "iid" if self.cov is None else "full", "scale": self.scale}

    def bind(self, bins: Bins) -> _BoundLogisticNormal:
        K = bins.K
        mu = _resolve_mean(self.mean, bins)
        if self.cov is None or np.ndim(self.cov) == 0:
            base = 1.0 if self.cov is None else float(np.sqrt(self.cov))
            B = helmert_basis(K)
            s0 = np.full(K - 1, base)
        else:
            C = np.asarray(self.cov, dtype=float)
            if C.shape != (K, K) or not np.allclose(C, C.T):
                raise ValueError("cov must be a symmetric (K, K) array.")
            P = np.eye(K) - 1.0 / K
            lam, U = np.linalg.eigh(P @ C @ P)
            keep = lam > 1e-10 * lam.max()
            B = U[:, keep]
            s0 = np.sqrt(lam[keep])
        basis = _KronBasis([np.asarray(B)])
        return _BoundLogisticNormal(
            self, bins, basis, np.arange(B.shape[1]), mu, [("scale", self.scale)], s0
        )


class _BoundLogisticNormal(_BoundLogistic):
    def __init__(self, prior, bins, basis, active, mu, hypers, s0):
        super().__init__(prior, bins, basis, active, mu, hypers)
        self.s0 = s0

    def scales(self, theta):
        s = theta[0] * self.s0
        return s, np.ones((1, self.s0.size))


# ----------------------------------------------------------------------------
# LogisticRandomWalk
# ----------------------------------------------------------------------------
class LogisticRandomWalk(Prior):
    r"""Random-walk (intrinsic GMRF) smoothness prior on log-fractions over a grid.

    The ``order``-th differences of :math:`\eta = \log\rho` (between adjacent
    bins along each axis) are Gaussian with standard deviation ``scale``:
    ``order=1`` penalises jumps, ``order=2`` penalises curvature. N-D grids
    use the Kronecker-sum (ICAR-type) precision. Polynomial modes not
    penalised by the random walk (e.g. a linear trend for ``order=2``) get a
    broad Gaussian prior with standard deviation ``trend_scale``.

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
        reference fractions. Defaults to the log bin volumes (a flat density),
        so differences penalise the log-density.
    trend_scale : float
        Prior standard deviation of the unpenalised polynomial modes.
    """

    def __init__(
        self,
        order: int = 2,
        scale: Any = None,
        mean: Any = None,
        trend_scale: float = 10.0,
    ) -> None:
        if order not in (1, 2, 3):
            raise ValueError("order must be 1, 2 or 3.")
        self.order = int(order)
        self.scale = as_hyperprior(HalfNormal(1.0) if scale is None else scale)
        self.mean = mean
        self.trend_scale = float(trend_scale)

    def _params(self):
        return {"order": self.order, "scale": self.scale}

    def bind(self, bins: Bins) -> _BoundRandomWalk:
        if bins.is_categorical:
            raise ValueError("Smoothness priors need ordered (grid) bins, not categories.")
        if bins.is_masked:
            raise NotImplementedError("Smoothness priors do not yet support masked grids.")
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
            self.trend_scale,
        )


class _BoundRandomWalk(_BoundLogistic):
    def __init__(self, prior, bins, basis, active, mu, hypers, lam, null, trend_scale):
        super().__init__(prior, bins, basis, active, mu, hypers)
        self.null = null
        self.inv_sqrt_lam = np.where(null, 0.0, 1.0 / np.sqrt(np.where(null, 1.0, lam)))
        self.trend_scale = trend_scale

    def scales(self, theta):
        tau = theta[0]
        s = np.where(self.null, self.trend_scale, tau * self.inv_sqrt_lam)
        dlogs = np.where(self.null, 0.0, 1.0)[None, :]
        return s, dlogs


# ----------------------------------------------------------------------------
# LogisticGP (Hilbert-space approximation)
# ----------------------------------------------------------------------------
_KERNELS = {"rbf": None, "se": None, "matern12": 0.5, "matern32": 1.5, "matern52": 2.5}


class LogisticGP(Prior):
    r"""Gaussian-process prior on log-fractions over a 1-D or N-D grid.

    Uses the Hilbert-space reduced-rank approximation (Solin & Sarkka 2020):
    a fixed sine basis on a box ``boundary_factor`` times larger than the grid,
    with weights scaled by the kernel's spectral density, so amplitude and
    (per-axis) length scales are inferred jointly with the field.

    Parameters
    ----------
    kernel : {"matern32", "matern52", "matern12", "rbf"}
        Covariance kernel (``"se"`` is an alias of ``"rbf"``).
    amplitude : Hyperprior or float
        Marginal standard deviation of the log-density field; inferred with a
        ``HalfNormal(2.0)`` prior by default.
    length_scale : Hyperprior, float, or sequence (one per axis)
        Length scale(s) in the units of the bin coordinates. Default: per axis,
        an inverse gamma with 1% prior mass below two (narrowest) bin widths
        and 1% above the axis range, so the field is neither rougher than the
        grid can resolve nor flatter than the data can constrain.
    mean : array_like (K,), optional
        Mean log-fractions (e.g. the log of reference fractions). Defaults to
        the log bin volumes, i.e. a flat density: the GP acts on the
        log-density.
    n_basis : int or sequence, optional
        Basis functions per axis (default ``min(K_d, 48)``); increase for
        length scales shorter than a few bins.
    boundary_factor : float
        Ratio of the approximation box to the grid half-width (default 1.5).
    """

    def __init__(
        self,
        kernel: str = "matern32",
        amplitude: Any = None,
        length_scale: Any = None,
        mean: Any = None,
        n_basis: int | Sequence[int] | None = None,
        boundary_factor: float = 1.5,
    ) -> None:
        if kernel not in _KERNELS:
            raise ValueError(f"kernel must be one of {sorted(_KERNELS)}.")
        if boundary_factor <= 1.0:
            raise ValueError("boundary_factor must exceed 1.")
        self.kernel = kernel
        self.amplitude = as_hyperprior(HalfNormal(2.0) if amplitude is None else amplitude)
        self.length_scale = length_scale
        self.mean = mean
        self.n_basis = n_basis
        self.boundary_factor = float(boundary_factor)

    def _params(self):
        return {
            "kernel": self.kernel,
            "amplitude": self.amplitude,
            "length_scale": self.length_scale,
        }

    def bind(self, bins: Bins) -> _BoundGP:
        if bins.is_categorical:
            raise ValueError("Smoothness priors need ordered (grid) bins, not categories.")
        if bins.is_masked:
            raise NotImplementedError("Smoothness priors do not yet support masked grids.")
        D = bins.ndim
        nb = (
            [min(n, 48) for n in bins.shape]
            if self.n_basis is None
            else list(np.broadcast_to(np.asarray(self.n_basis, int), (D,)))
        )
        factors, omegas = [], []
        for d, e in enumerate(bins.edges):
            lo, hi = float(e[0]), float(e[-1])
            c, L = 0.5 * (lo + hi), self.boundary_factor * 0.5 * (hi - lo)
            x = 0.5 * (e[1:] + e[:-1])
            j = np.arange(1, nb[d] + 1)
            factors.append(
                np.sin(np.pi * j[None, :] * (x[:, None] - c + L) / (2 * L)) / np.sqrt(L)
            )
            omegas.append(np.pi * j / (2 * L))
        basis = _KronBasis(factors)
        grids = np.meshgrid(*omegas, indexing="ij")
        omega = np.stack([g.ravel() for g in grids], axis=1)  # (J, D)
        ls = self.length_scale
        if ls is None:
            ls_h = [
                InverseGamma.from_quantiles(2.0 * float(np.min(np.diff(e))), float(e[-1] - e[0]))
                for e in bins.edges
            ]
        elif isinstance(ls, Sequence) and not isinstance(ls, str):
            if len(ls) != D:
                raise ValueError(f"Need one length scale per axis ({D}).")
            ls_h = [as_hyperprior(v) for v in ls]
        else:
            ls_h = [as_hyperprior(ls) for _ in range(D)]
        hypers = [("amplitude", self.amplitude)] + [
            (f"length_scale_{d}" if D > 1 else "length_scale", h) for d, h in enumerate(ls_h)
        ]
        mu = _resolve_mean(self.mean, bins)
        return _BoundGP(
            self, bins, basis, np.arange(basis.J), mu, hypers, omega, _KERNELS[self.kernel]
        )


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

    def scales(self, theta):
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
        s = np.exp(0.5 * log_S)
        dlogs = np.empty((1 + self.D, s.size))
        dlogs[0] = 1.0
        for d in range(self.D):
            dlogs[1 + d] = 0.5 * (1.0 + 2.0 * a2[:, d] * dlog_dr2)
        return s, dlogs
