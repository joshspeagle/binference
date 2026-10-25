"""MAP / maximum-likelihood estimation: SQUAREM-accelerated EM and L-BFGS."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize

from .._utils import RNGLike, warn
from ..likelihoods import _scale_columns
from ..model import PopulationModel
from ..priors.dirichlet import _BoundDirichlet
from ..results import PointEstimate
from ..warnings import ConvergenceWarning, InputWarning
from .base import Engine, em_update, interior_start

__all__ = ["MAP"]


def _mixsqp(
    L,
    w: np.ndarray,
    a: np.ndarray,
    x0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> tuple[np.ndarray, int, bool]:
    r"""mix-SQP (Kim, Carbonetto, Stephens & Anitescu 2020) for mixture proportions.

    Maximises :math:`\sum_i w_i\log(L_i\cdot x)+\sum_k a_k\log x_k` over the
    simplex (a Dirichlet prior with :math:`a_k=\alpha_k-1\ge0`). The relaxed
    objective :math:`f(x)=-W^{-1}[\dots]+c\sum_kx_k` with :math:`c=(W+\sum a)/W`
    has its minimiser on the simplex, so only :math:`x\ge0` is imposed. Each
    iteration solves a non-negative quadratic program for the Newton step
    (as a non-negative least-squares problem) and backtracks along it. For
    tall dense ``L`` the Hessian uses a fixed low-rank approximation of ``L``;
    gradients and objectives stay exact, so the fixed point is exact.

    Returns ``(x, n_iter, converged)``.
    """
    W = float(w.sum())
    A = float(a.sum())
    c = (W + A) / W
    pos = a > 0
    N, K = L.shape
    dense = not hasattr(L, "toarray")
    low_rank = None
    if dense and N > 10 * K:
        U, sv, Vt = np.linalg.svd(np.asarray(L), full_matrices=False)
        r = int(np.sum(sv > 1e-8 * sv[0]))
        low_rank = (U[:, :r], sv[:r, None] * Vt[:r])  # L ~= U @ B

    def objective(x):
        s = L @ x
        if not np.all(s > 0):
            return np.inf
        val = -float(w @ np.log(s)) / W + c * float(x.sum())
        if A > 0:
            if np.any(x[pos] <= 0):
                return np.inf
            val -= float(a[pos] @ np.log(x[pos])) / W
        return val

    x = np.maximum(np.asarray(x0, dtype=float), 1e-12)
    x = x / x.sum()
    f = objective(x)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):  # noqa: B007 (used after the loop)
        s = np.asarray(L @ x).ravel()
        u = w / s
        g = -np.asarray(L.T @ u).ravel() / W + c
        if A > 0:
            g[pos] -= a[pos] / (W * x[pos])
        if np.min(g) >= -tol:
            converged = True
            break
        d2 = u / s  # w_i / s_i^2
        if low_rank is not None:
            Uh, B = low_rank
            M = (Uh * d2[:, None]).T @ Uh
            H = B.T @ M @ B / W
        elif dense:
            Ls = np.asarray(L) * np.sqrt(d2)[:, None]
            H = Ls.T @ Ls / W
        else:
            H = (L.T @ (L.multiply(d2[:, None]))).toarray() / W
        if A > 0:
            H[np.diag_indices(K)] += np.where(pos, a / (W * np.maximum(x, 1e-300) ** 2), 0.0)
        H[np.diag_indices(K)] += 1e-10 * max(np.trace(H) / K, 1e-300)
        try:
            Lc = np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            H[np.diag_indices(K)] += 1e-6 * max(np.trace(H) / K, 1e-300)
            Lc = np.linalg.cholesky(H)
        q = g - H @ x
        b = -np.linalg.solve(Lc, q)
        y, _ = optimize.nnls(Lc.T, b, maxiter=50 * K)
        step = y - x
        slope = float(g @ step)
        if slope >= 0:  # no descent direction left: at the optimum up to round-off
            converged = bool(np.min(g) >= -1e3 * tol)
            break
        t = 1.0
        while t > 1e-12:
            cand = x + t * step
            fc = objective(cand)
            if fc <= f + 1e-4 * t * slope:
                break
            t *= 0.5
        else:
            converged = bool(np.min(g) >= -1e3 * tol)
            break
        stalled = f - fc <= 1e-15 * max(abs(f), 1.0)
        x, f = cand, fc
        if stalled:  # the objective no longer resolves the remaining KKT violation
            converged = bool(np.min(g) >= -1e3 * tol)
            break
    x = np.maximum(x, 0.0)
    return x / x.sum(), it, converged


class MAP(Engine):
    r"""Maximum a posteriori (or maximum-likelihood) estimate of the population.

    For conjugate priors (Dirichlet family, :class:`~binference.priors.GammaRates`)
    the posterior mode on the simplex is found by EM -- "iterated stacking",
    :math:`\rho_k \leftarrow \sum_i w_i r_{ik}(\rho)/W` plus prior pseudo-counts --
    accelerated with SQUAREM (Varadhan & Roland 2008). For a ``Dirichlet``
    prior with all concentrations >= 1 (exactly 1 with a selection function),
    the default uses mix-SQP (Kim et al. 2020) instead, a second-order method
    that converges in tens of iterations where EM can need thousands (broad
    likelihoods). With selection it solves for the fractions of the observed
    population, :math:`\phi_k \propto S_k\rho_k`, in which the likelihood
    is concave, and maps the mode back. The
    default ``Dirichlet(1)`` prior gives the maximum-likelihood estimate (the
    NPMLE on the grid). Other priors, or ``space="unconstrained"``, maximise
    the density in the unconstrained coordinates with L-BFGS.

    Parameters
    ----------
    model : PopulationModel
    space : {"auto", "simplex", "unconstrained"}
        ``"simplex"`` maximises the density of the fractions (requires a
        conjugate prior); ``"unconstrained"`` the density in unconstrained
        coordinates (always interior); ``"auto"`` picks ``"simplex"`` unless
        the prior density is unbounded at the boundary (concentrations below
        one, Polya trees), where the simplex mode can have exact zeros.
    tol : float
        Relative tolerance on the objective.
    max_iter : int
        Maximum number of (accelerated) EM iterations or L-BFGS iterations.
    accelerate : bool
        Use SQUAREM acceleration (EM only).
    algorithm : {"auto", "mixsqp", "em"}
        Simplex-space algorithm: ``"auto"`` uses mix-SQP where it applies
        (Dirichlet prior with concentrations >= 1, or exactly 1 with
        selection) and EM otherwise.
    init : array_like (K,), optional
        Starting fractions (or rates).
    """

    name = "map"

    def __init__(
        self,
        model: PopulationModel,
        *,
        space: str = "auto",
        tol: float = 1e-10,
        max_iter: int = 20000,
        accelerate: bool = True,
        algorithm: str = "auto",
        init: Any = None,
        rng: RNGLike = None,
        progress: bool = False,
    ) -> None:
        super().__init__(model, rng=rng, progress=progress)
        if space not in ("auto", "simplex", "unconstrained"):
            raise ValueError("space must be 'auto', 'simplex' or 'unconstrained'.")
        if algorithm not in ("auto", "mixsqp", "em"):
            raise ValueError("algorithm must be 'auto', 'mixsqp' or 'em'.")
        self.algorithm = algorithm
        self.space = space
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.accelerate = bool(accelerate)
        self.init = None if init is None else np.asarray(init, dtype=float)

    def _resolve_space(self) -> str:
        m = self.model
        simplex_ok = m.prior.conjugate and m.prior.has_density
        if self.space == "simplex":
            if not simplex_ok:
                raise ValueError("space='simplex' needs a conjugate prior; use 'unconstrained'.")
            if m.prior.unbounded_density():
                warn(
                    "The prior density is unbounded at the boundary of the simplex (e.g. "
                    "concentrations below one, or a Polya tree), so the EM solution can clip "
                    "components to exactly zero; space='unconstrained' gives an interior mode.",
                    InputWarning,
                )
            return "simplex"
        if self.space == "unconstrained":
            return "unconstrained"
        if simplex_ok and not m.prior.unbounded_density():
            return "simplex"
        return "unconstrained"

    def run(self) -> PointEstimate:
        """Find the mode."""
        m = self.model
        space = self._resolve_space()
        if space == "simplex":
            if self._mixsqp_applies():
                return self._run_mixsqp(m)
            if self.algorithm == "mixsqp":
                raise ValueError(
                    "mix-SQP needs a Dirichlet prior with concentrations >= 1 (exactly 1 with "
                    "a selection function)."
                )
            return self._run_em(m)
        return self._run_lbfgs(m)

    #: whether results carry their log-likelihood, KKT gap and convergence diagnostics;
    #: the bootstrap turns them off, as it reads only the values
    _diagnostics = True

    def _estimate(
        self, m, v, n_iter, converged, algorithm, x=None, log_posterior=None, grad_norm=None
    ) -> PointEstimate:
        # a mode with its certificate: on the simplex (or orthant), or in unconstrained x
        simplex = x is None
        info = {"space": "simplex" if simplex else "unconstrained", "algorithm": algorithm}
        if grad_norm is not None:
            info["grad_norm"] = grad_norm
        full = self._diagnostics
        if log_posterior is None and full:
            log_posterior = _objective(m, v)
        return PointEstimate(
            m,
            v,
            method="map",
            x=x,
            log_posterior=np.nan if log_posterior is None else log_posterior,
            log_likelihood=m.log_likelihood(v) if full else np.nan,
            kkt_gap=m.kkt_gap(v) if simplex and full else np.nan,
            n_iter=n_iter,
            converged=converged,
            info=info,
        )

    def _mixsqp_applies(self) -> bool:
        m = self.model
        if not (
            self.algorithm in ("auto", "mixsqp")
            and isinstance(m.prior, _BoundDirichlet)
            and bool(np.all(m.prior.alpha >= 1.0))
        ):
            return False
        # with selection the likelihood is concave in the observed fractions
        # phi = S rho / (S . rho); a flat prior keeps the mode under that map
        return m.selection is None or bool(np.all(m.prior.alpha == 1.0))

    def _run_mixsqp(self, m: PopulationModel) -> PointEstimate:
        x0 = interior_start(m) if self.init is None else self.init.copy()
        a = np.asarray(m.prior.alpha, dtype=float) - 1.0
        tol = max(self.tol, 1e-12) * 100.0  # tolerance on the (relative) KKT conditions
        L = m._L
        if m.selection is not None:
            # p(x_i | observed) = sum_k (L_ik / S_k) phi_k: solve for phi, then map back
            L = _scale_columns(L, 1.0 / m._detect)
            x0 = x0 * m._detect / float(x0 @ m._detect)
        x, n_iter, converged = _mixsqp(L, m._w, a, x0, tol=tol, max_iter=self.max_iter)
        if m.selection is not None:
            x = x / m._detect
            x /= x.sum()
        if not converged:
            warn(f"mix-SQP did not converge in {n_iter} iterations.", ConvergenceWarning)
        return self._estimate(m, x, n_iter, converged, "mixsqp")

    # ------------------------------------------------------------------- EM
    def _run_em(self, m: PopulationModel) -> PointEstimate:
        fractions = m.kind == "fractions"
        v = interior_start(m) if self.init is None else self.init.copy()
        if fractions:
            v = v / v.sum()

        def F(x):
            out = em_update(m, x)
            if out is None:
                return None
            return out / out.sum() if fractions else out

        f_old = _objective(m, v)
        if not np.isfinite(f_old):
            raise ValueError(
                "init gives some object zero likelihood; pass an init that is positive "
                "wherever an object has support."
            )
        converged = False
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):  # noqa: B007 (used after the loop)
            if self.accelerate:
                v_new, f_new = self._squarem_step(m, F, v, fractions)
            else:
                v_new = F(v)
                f_new = -np.inf if v_new is None else _objective(m, v_new)
            if not np.isfinite(f_new):  # keep the last population where every object is possible
                warn(
                    "EM reached a population under which an object has zero likelihood (a "
                    "component clipped to zero, e.g. by concentrations below one); "
                    "space='unconstrained' keeps the mode interior.",
                    ConvergenceWarning,
                )
                break
            dv = np.max(np.abs(v_new - v)) / max(np.max(np.abs(v_new)), 1e-300)
            df = abs(f_new - f_old) / (abs(f_old) + 1.0)
            v, f_old = v_new, f_new
            if df <= self.tol and dv <= 1e-7:
                converged = True
                break
        else:
            msg = f"EM did not converge in {self.max_iter} iterations"
            if self._diagnostics and m.K <= 500:  # EM is slow when information is missing
                f = m.missing_information(v, include_prior=True)
                if np.isfinite(f):
                    msg += f" (missing information {f:.3f})"
            warn(msg + "; raise max_iter.", ConvergenceWarning)
        algorithm = "squarem" if self.accelerate else "em"
        return self._estimate(m, v, n_iter, converged, algorithm, log_posterior=f_old)

    def _squarem_step(self, m, F, v, fractions):
        # one SQUAREM step; (None, -inf) when the first plain EM step meets an impossible object
        r1 = F(v)
        f1 = -np.inf if r1 is None else _objective(m, r1)
        if not np.isfinite(f1):
            return None, -np.inf
        r2 = F(r1)
        f2 = -np.inf if r2 is None else _objective(m, r2)
        if not np.isfinite(f2):
            return r1, f1  # the next step stops there
        d1 = r1 - v
        d2 = r2 - 2.0 * r1 + v
        n2 = float(np.sqrt(np.sum(d2**2)))
        if n2 == 0.0:
            return r2, f2
        step = min(-float(np.sqrt(np.sum(d1**2))) / n2, -1.0)
        for _ in range(30):
            cand = v - 2.0 * step * d1 + step**2 * d2
            if np.all(cand >= 0) and np.all(np.isfinite(cand)) and cand.sum() > 0:
                if fractions:
                    cand = cand / cand.sum()
                cand = F(cand)  # stabilisation step keeps EM's monotonicity guarantees
                # an extrapolation that makes an object impossible is rejected
                if cand is not None:
                    fc = _objective(m, cand)
                    if fc >= f2 - 1e-12 * abs(f2):
                        return cand, fc
            if step == -1.0:
                break
            step = (step - 1.0) / 2.0
            step = min(step, -1.0)
        return r2, f2

    # --------------------------------------------------------------- L-BFGS
    def _run_lbfgs(self, m: PopulationModel) -> PointEstimate:
        v0 = interior_start(m) if self.init is None else self.init
        x0 = m.from_value(v0)

        def fun(x):
            lp, g = m.log_density_unconstrained_and_grad(x)
            if not np.isfinite(lp):
                return 1e300, np.zeros_like(x)
            return -lp, -g

        scale = max(1.0, m.W)
        res = optimize.minimize(
            fun,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "maxcor": 30, "ftol": 1e-15, "gtol": 1e-10 * scale},
        )
        x = res.x
        lp, g = m.log_density_unconstrained_and_grad(x)
        gnorm = float(np.max(np.abs(g))) if g.size else 0.0
        converged = bool(res.success) or gnorm < 1e-5 * scale
        if not converged:
            warn(
                f"L-BFGS did not converge ({res.message}); max |gradient| = {gnorm:.2e}.",
                ConvergenceWarning,
            )
        v = m.to_value(x)
        return self._estimate(
            m, v, int(res.nit), converged, "l-bfgs-b", x=x, log_posterior=lp, grad_norm=gnorm
        )


def _objective(m: PopulationModel, v: np.ndarray) -> float:
    # log posterior on the simplex (or orthant), up to the likelihoods' log-scales;
    # components clipped to zero are active constraints: the prior is evaluated at a tiny
    # positive value so the objective stays finite (and constant) for them
    return m.log_likelihood(v, absolute=False) + float(m.prior.log_density(np.maximum(v, 1e-300)))
