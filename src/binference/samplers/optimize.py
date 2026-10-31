"""MAP / maximum-likelihood estimation: mix-SQP, SQUAREM-accelerated EM and L-BFGS."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize

from .. import _kernels as kern
from .._utils import RNGLike, check_nonnegative, warn
from ..likelihoods import _scale_columns
from ..model import PopulationModel
from ..priors.base import _TINY
from ..results import PointEstimate
from ..warnings import ConvergenceWarning, InputWarning
from .base import Engine, em_update, interior_start

__all__ = ["MAP"]

# DECISIONS #71
#: A MAP run that can no longer improve its objective in floating point is converged
#: when its optimality gap is below this many times ``tol``.
_ROUNDOFF = 1e3
# ten regularised EM steps are the cheapest start measured; from uniform, mix-SQP can
# need over a hundred iterations (audit M-5)
#: Regularised EM steps of mix-SQP's start when ``init`` is ``None``.
_MIXSQP_START_ITER = 10


def _converged(gap: float, tol: float, stalled: bool) -> bool:
    return gap <= tol or (stalled and gap <= _ROUNDOFF * tol)


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
    (as a non-negative least-squares problem) and backtracks along it. It
    stops when the KKT gap :math:`-\min_k g_k` is at most ``tol``, or at most
    ``1e3 * tol`` once the objective no longer resolves progress (round-off).

    Returns ``(x, n_iter, converged)``.
    """
    W = float(w.sum())
    A = float(a.sum())
    c = (W + A) / W
    pos = a > 0
    K = L.shape[1]
    dense = not hasattr(L, "toarray")

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
    gap, stalled = np.inf, False
    it = 0
    for it in range(1, max_iter + 1):  # noqa: B007 (used after the loop)
        s = np.asarray(L @ x).ravel()
        u = w / s
        g = -np.asarray(L.T @ u).ravel() / W + c
        if A > 0:
            g[pos] -= a[pos] / (W * x[pos])
        gap = -float(np.min(g))
        if gap <= tol:
            break
        d2 = u / s  # w_i / s_i^2
        if dense:
            Ls = np.asarray(L) * np.sqrt(d2)[:, None]
            H = Ls.T @ Ls / W
        else:
            H = (L.T @ (L.multiply(d2[:, None]))).toarray() / W
        if A > 0:
            H[pos, pos] += a[pos] / (W * x[pos] ** 2)  # only where a_k > 0 (audit T-4)
        H[np.diag_indices(K)] += 1e-10 * max(np.trace(H) / K, _TINY)
        try:
            try:
                Lc = np.linalg.cholesky(H)
            except np.linalg.LinAlgError:
                H[np.diag_indices(K)] += 1e-6 * max(np.trace(H) / K, _TINY)
                Lc = np.linalg.cholesky(H)
            b = -np.linalg.solve(Lc, g - H @ x)
            y, _ = optimize.nnls(Lc.T, b, maxiter=50 * K)
        except (np.linalg.LinAlgError, RuntimeError):  # no quadratic step (audit M-13)
            break
        step = y - x
        slope = float(g @ step)
        if slope >= 0:  # no descent direction left: at the optimum up to round-off
            stalled = True
            break
        t = 1.0
        while t > 1e-12:
            cand = x + t * step
            fc = objective(cand)
            if fc <= f + 1e-4 * t * slope:
                break
            t *= 0.5
        else:
            stalled = True
            break
        stalled = f - fc <= 1e-15 * max(abs(f), 1.0)
        x, f = cand, fc
        if stalled:  # the objective no longer resolves the remaining KKT violation
            break
    x = np.maximum(x, 0.0)
    return x / x.sum(), it, _converged(gap, tol, stalled)


class MAP(Engine):
    r"""Maximum a posteriori (or maximum-likelihood) estimate of the population.

    For conjugate priors (Dirichlet family, :class:`~binference.priors.GammaRates`)
    the posterior mode on the simplex is found by EM -- "iterated stacking",
    :math:`\rho_k \leftarrow \sum_i w_i r_{ik}(\rho)/W` plus prior pseudo-counts --
    accelerated with SQUAREM (Varadhan & Roland 2008). For a ``Dirichlet``
    prior with all concentrations >= 1 (exactly 1 with a selection function),
    and for ``GammaRates`` with all shapes >= 1, the default uses mix-SQP
    (Kim et al. 2020) instead, a second-order method that converges in tens
    of iterations where EM can need thousands (broad likelihoods). With
    selection it solves for the fractions of the observed population,
    :math:`\phi_k \propto S_k\rho_k`, in which the likelihood is concave,
    and maps the mode back; rates :math:`\lambda = \Lambda\psi` reduce to
    the same problem with :math:`S_k + b_k` in place of :math:`S_k` once the
    total :math:`\Lambda` is profiled out. The
    default ``Dirichlet(1)`` prior gives the maximum-likelihood estimate (the
    NPMLE on the grid). Other priors, or ``space="unconstrained"``, maximise
    the density in the unconstrained coordinates with L-BFGS.

    Parameters
    ----------
    model : PopulationModel
    space : {"auto", "simplex", "unconstrained"}
        Coordinates the mode is found in. ``"simplex"`` maximises the density
        of the fractions (requires a
        conjugate prior); ``"unconstrained"`` the density in unconstrained
        coordinates (always interior); ``"auto"`` picks ``"simplex"`` unless
        the prior density is unbounded at the boundary (concentrations or
        shapes below one, the default Polya tree; ``prior.unbounded_density()``),
        where the simplex mode can have exact zeros, or a selection or response
        is inferred jointly. The choice is in ``info["space"]`` and the solver in
        ``info["algorithm"]``.
    tol : float
        Tolerance on first-order optimality. It bounds the KKT gap
        (:meth:`~binference.PopulationModel.kkt_gap`) for EM, and for mix-SQP
        the same gap of the problem it solves (in the observed fractions under
        selection, the composition for rates); the largest absolute gradient
        divided by ``max(1, W)`` for L-BFGS, whose estimates keep the largest
        absolute gradient in ``info["grad_norm"]`` and have ``kkt_gap`` NaN. A
        run that can no longer improve the objective in floating point counts as
        converged when that measure is below ``1e3 * tol``.
    max_iter : int
        Maximum number of (accelerated) EM iterations or L-BFGS iterations.
    accelerate : bool
        Use SQUAREM acceleration (EM only).
    algorithm : {"auto", "mixsqp", "em"}
        Solver on the simplex. ``"auto"`` uses mix-SQP where it applies
        (Dirichlet prior with concentrations >= 1, or exactly 1 with
        selection; GammaRates with shapes >= 1), EM on the simplex otherwise,
        and L-BFGS in the unconstrained space. ``"mixsqp"`` and ``"em"`` solve on
        the simplex, as ``space="simplex"`` does (not for joint models).
    init : array_like (K,), optional
        Non-negative starting fractions (or rates), not all zero; EM and mix-SQP raise
        zeros to a small positive value, so that they can move them. By default mix-SQP
        starts after 10
        regularised EM steps from uniform, the other algorithms after up to 50.
    rng, progress
        Accepted for a common signature; unused.
    """

    joint = True  # in the unconstrained space

    def __init__(
        self,
        model: PopulationModel,
        *,
        space: str = "auto",
        tol: float = 1e-8,
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
        if init is not None:
            init = check_nonnegative(init, "init")
            if init.shape != (model.K,):
                raise ValueError(f"init must have shape ({model.K},); got {init.shape}.")
            if not init.any():
                raise ValueError("init must have a positive entry.")
        self.init = init
        # an explicit algorithm or space="simplex" solves on the simplex, if it can
        self._on_simplex = algorithm != "auto" or space == "simplex"
        if self._on_simplex:
            asked = f"algorithm={algorithm!r}" if algorithm != "auto" else "space='simplex'"
            if model.joint:
                why = "a joint selection or response is fitted by L-BFGS in space='unconstrained'"
            elif space == "unconstrained":
                why = "space='unconstrained' runs L-BFGS"
            elif not (model.prior.conjugate and model.prior.has_density):
                why = "the simplex needs a conjugate prior"
            elif algorithm == "mixsqp" and not self._mixsqp_applies():
                why = (
                    "mix-SQP needs a Dirichlet prior with concentrations >= 1 (exactly 1 with "
                    "a selection function) or GammaRates with shapes >= 1"
                )
            else:
                why = None
            if why:
                fix = "algorithm='auto'" if algorithm != "auto" else "space='auto'"
                raise ValueError(
                    f"{asked} cannot run here: {why}. Use {fix} (method='map' in infer)."
                )

    def _resolve_space(self) -> str:
        m = self.model
        if self._on_simplex:
            if m.prior.unbounded_density():
                warn(
                    "The prior density is unbounded at the boundary of the simplex (e.g. "
                    "concentrations below one, or a Polya tree), so the EM solution can clip "
                    "components to exactly zero; space='unconstrained' gives an interior mode.",
                    InputWarning,
                )
            return "simplex"
        if self.space == "unconstrained" or m.joint:
            return "unconstrained"
        if m.prior.conjugate and m.prior.has_density and not m.prior.unbounded_density():
            return "simplex"
        return "unconstrained"

    def run(self) -> PointEstimate:
        """Find the mode."""
        m = self.model
        if self._resolve_space() == "simplex":
            return self._run_mixsqp(m) if self._mixsqp_applies() else self._run_em(m)
        return self._run_lbfgs(m)

    #: whether results carry their log-likelihood, KKT gap and convergence diagnostics;
    #: the bootstrap turns them off, as it reads only the values
    _diagnostics = True

    def _estimate(self, m, v, n_iter, converged, algorithm, x=None, **extra) -> PointEstimate:
        # a mode with its certificate: on the simplex (or orthant), or in unconstrained x;
        # log_posterior is the density of the fractions or rates for every algorithm (audit M-8)
        simplex = x is None
        info = {
            "space": "simplex" if simplex else "unconstrained",
            "algorithm": algorithm,
            **extra,
        }
        full = self._diagnostics
        if not simplex:  # what is inferred with the population, at the mode
            info.update(m._joint_info(x[None, None]))
        S = None if simplex else m.selection_at(x)
        R = None if simplex else m.response_at(x)
        ll = m.log_likelihood(v, S=S, response=R) if full else np.nan
        lp = ll + m.log_prior(v) if full else np.nan
        if m.joint and not simplex and full:  # and the density of S or R given the data
            lp += m._joint_log_density(x)
        return PointEstimate(
            m,
            v,
            method="map",
            x=x,
            log_posterior=lp,
            log_likelihood=ll,
            kkt_gap=m.kkt_gap(v) if simplex and full else np.nan,
            n_iter=n_iter,
            converged=converged,
            info=info,
        )

    def _mixsqp_applies(self) -> bool:
        m = self.model
        a = m.prior.pseudocounts  # Dirichlet or GammaRates
        if self.algorithm not in ("auto", "mixsqp") or a is None or np.any(a < 0.0):
            return False
        # with selection the likelihood is concave in the observed fractions
        # phi = S rho / (S . rho); a flat prior keeps the mode under that map
        return m.kind == "rates" or m.selection is None or bool(np.all(a == 0.0))

    def _run_mixsqp(self, m: PopulationModel) -> PointEstimate:
        rates = m.kind == "rates"
        a = m.prior.pseudocounts
        # the likelihood is sum_k (L_ik / e_k) psi_k in psi proportional to e * value: the
        # observed fractions (e = S) or, for rates lambda = Lambda psi / e, the composition
        # with e = S + b, whose total Lambda = W + sum(a) profiles out (audit M-6)
        e = m._S + m.prior.b if rates else m._detect
        L = m._L if e is None else _scale_columns(m._L, 1.0 / e)
        x0 = interior_start(m, n_iter=_MIXSQP_START_ITER) if self.init is None else self.init
        if e is not None:
            x0 = x0 * e
        x, n_iter, converged = _mixsqp(L, m._w, a, x0, tol=self.tol, max_iter=self.max_iter)
        if e is not None:  # back from psi to the fractions or rates
            x = x / e
            if rates:
                x *= m.W + a.sum()
            else:
                x /= x.sum()
        if not converged:  # EM takes over from where mix-SQP stopped (audit M-13)
            warn(
                f"mix-SQP did not converge in {n_iter} iterations; continuing with EM.",
                ConvergenceWarning,
            )
            return self._run_em(m, x)
        return self._estimate(m, x, n_iter, converged, "mixsqp")

    # ------------------------------------------------------------------- EM
    def _run_em(self, m: PopulationModel, start: np.ndarray | None = None) -> PointEstimate:
        fractions = m.kind == "fractions"
        if start is None:
            start = interior_start(m) if self.init is None else self.init
        # EM multiplies each by its update, so a zero never moves: floored as the priors'
        # init_from_value floor theirs
        v = np.maximum(start, 1e-12 * start.max())
        if fractions:
            v = v / v.sum()

        def F(x, s):  # the EM update of x, given its evidences s = L @ x
            out = em_update(m, x, s)
            if out is None:
                return None
            return out / out.sum() if fractions else out

        f_old, s = _objective(m, v)
        converged = False
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            if self.accelerate:
                v_new, f_new, s_new = self._squarem_step(m, F, v, s, fractions)
            else:
                v_new = F(v, s)
                f_new, s_new = (-np.inf, None) if v_new is None else _objective(m, v_new)
            if not np.isfinite(f_new):  # keep the last population where every object is possible
                warn(
                    "EM reached a population under which an object has zero likelihood (a "
                    "component clipped to zero, e.g. by concentrations below one); "
                    "space='unconstrained' keeps the mode interior.",
                    ConvergenceWarning,
                )
                break
            df = (f_new - f_old) / (abs(f_new) + 1.0)  # EM never decreases the objective
            v, s, f_old = v_new, s_new, f_new
            # once the objective has nearly stopped, check optimality every tenth step (two
            # passes over the data each), at a fixed point and at the last step (audit M-6)
            if df <= self.tol and (n_iter % 10 == 0 or df <= 0.0 or n_iter == self.max_iter):
                gap = m.kkt_gap(v)
                stalled = df <= 0.0  # EM is monotone: no gain is the round-off floor
                if _converged(gap, self.tol, stalled):
                    converged = True
                    break
                if stalled:
                    warn(
                        f"EM stopped improving the objective with a KKT gap of {gap:.1e}.",
                        ConvergenceWarning,
                    )
                    break
        else:
            msg = f"EM did not converge in {self.max_iter} iterations"
            if self._diagnostics and m.K <= 500:  # EM is slow when information is missing
                f = m.missing_information(v, include_prior=True)
                if np.isfinite(f):
                    msg += f" (missing information {f:.3f})"
            warn(msg + "; raise max_iter.", ConvergenceWarning)
        algorithm = "squarem" if self.accelerate else "em"
        return self._estimate(m, v, n_iter, converged, algorithm)

    def _squarem_step(self, m, F, v, s, fractions):
        # one SQUAREM step from v with evidences s = L @ v: the new point, its objective
        # and its evidences, so each point visited costs one product L @ x;
        # (None, -inf, None) when the first plain EM step meets an impossible object
        r1 = F(v, s)
        f1, s1 = (-np.inf, None) if r1 is None else _objective(m, r1)
        if not np.isfinite(f1):
            return None, -np.inf, None
        r2 = F(r1, s1)
        f2, s2 = (-np.inf, None) if r2 is None else _objective(m, r2)
        if not np.isfinite(f2):
            return r1, f1, s1  # the next step stops there
        d1 = r1 - v
        d2 = r2 - 2.0 * r1 + v
        n2 = float(np.sqrt(np.sum(d2**2)))
        if n2 == 0.0:
            return r2, f2, s2
        step = min(-float(np.sqrt(np.sum(d1**2))) / n2, -1.0)
        for _ in range(30):
            cand = v - 2.0 * step * d1 + step**2 * d2
            if np.all(cand >= 0) and np.all(np.isfinite(cand)) and cand.sum() > 0:
                if fractions:
                    cand = cand / cand.sum()
                # stabilisation step keeps EM's monotonicity guarantees
                cand = F(cand, kern.matvec(m._L, cand))
                # an extrapolation that makes an object impossible is rejected
                if cand is not None:
                    fc, sc = _objective(m, cand)
                    if fc >= f2:  # monotone, so a fall in the objective means round-off
                        return cand, fc, sc
            if step == -1.0:
                break
            step = (step - 1.0) / 2.0
            step = min(step, -1.0)
        return r2, f2, s2

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
            options={
                "maxiter": self.max_iter,
                "maxcor": 30,
                "ftol": 1e-15,
                "gtol": self.tol * scale,
            },
        )
        x = res.x
        lp, g = m.log_density_unconstrained_and_grad(x)
        gnorm = float(np.max(np.abs(g))) if g.size else 0.0
        # any stop but the iteration limit means the objective stopped improving
        converged = _converged(gnorm / scale, self.tol, stalled=res.status != 1)
        if not converged:
            warn(
                f"L-BFGS did not converge ({res.message}); max |gradient| = {gnorm:.2e}.",
                ConvergenceWarning,
            )
        v = m.to_value(x)
        return self._estimate(
            m,
            v,
            int(res.nit),
            converged,
            "l-bfgs-b",
            x=x,
            grad_norm=gnorm,
            log_density_unconstrained=lp,
        )


def _objective(m: PopulationModel, v: np.ndarray) -> tuple[float, np.ndarray]:
    # log posterior on the simplex (or orthant), up to the likelihoods' log-scales, and the
    # evidences L @ v it uses (m.log_likelihood(v, absolute=False) + the prior); components
    # clipped to zero are active constraints: the prior is evaluated at a tiny positive
    # value so the objective stays finite (and constant) for them
    s = kern.matvec(m._L, v)
    ll = kern.loglike_of_evidences(m._w, s)
    if not np.isfinite(ll):
        return -np.inf, s
    return ll + m._selection_term(v) + m.log_prior(np.maximum(v, _TINY)), s
