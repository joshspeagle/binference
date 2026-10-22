"""MAP / maximum-likelihood estimation: SQUAREM-accelerated EM and L-BFGS."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import optimize

from .._utils import RNGLike
from ..model import PopulationModel
from ..results import PointEstimate
from ..warnings import ConvergenceWarning, InputWarning
from .base import Engine, em_update, interior_start

__all__ = ["MAP"]


def _alphas_below_one(model: PopulationModel) -> bool:
    p = model.prior
    if hasattr(p, "alpha") and np.ndim(getattr(p, "alpha", 1.0)) == 1:
        return bool(np.any(p.alpha < 1.0))
    if hasattr(p, "tree"):
        return bool(np.any(p.tree.slot_alpha < 1.0))
    if hasattr(p, "a") and hasattr(p, "b"):
        return bool(np.any(p.a < 1.0))
    return False


class MAP(Engine):
    r"""Maximum a posteriori (or maximum-likelihood) estimate of the population.

    For conjugate priors (Dirichlet family, :class:`~pastry.priors.GammaRates`)
    the posterior mode on the simplex is found by EM -- "iterated stacking",
    :math:`\rho_k \leftarrow \sum_i w_i r_{ik}(\rho)/W` plus prior pseudo-counts --
    accelerated with SQUAREM (Varadhan & Roland 2008). The default
    ``Dirichlet(1)`` prior gives the maximum-likelihood estimate (the NPMLE on
    the grid). Other priors, or ``space="unconstrained"``, maximise the density
    in the unconstrained coordinates with L-BFGS.

    Parameters
    ----------
    model : PopulationModel
    space : {"auto", "simplex", "unconstrained"}
        ``"simplex"`` maximises the density of the fractions (requires a
        conjugate prior with all concentrations >= 1); ``"unconstrained"`` the
        density in unconstrained coordinates (always interior); ``"auto"``
        picks ``"simplex"`` when possible.
    tol : float
        Relative tolerance on the objective.
    max_iter : int
        Maximum number of (accelerated) EM iterations or L-BFGS iterations.
    accelerate : bool
        Use SQUAREM acceleration (EM only).
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
        init: Any = None,
        rng: RNGLike = None,
        progress: bool = False,
    ) -> None:
        super().__init__(model, rng=rng, progress=progress)
        if space not in ("auto", "simplex", "unconstrained"):
            raise ValueError("space must be 'auto', 'simplex' or 'unconstrained'.")
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
            if _alphas_below_one(m):
                warnings.warn(
                    "Some prior concentrations are < 1, so the density is unbounded at the "
                    "boundary; the EM solution clips those components to zero.",
                    InputWarning,
                    stacklevel=3,
                )
            return "simplex"
        if self.space == "unconstrained":
            return "unconstrained"
        return "simplex" if (simplex_ok and not _alphas_below_one(m)) else "unconstrained"

    def run(self, weights: np.ndarray | None = None) -> PointEstimate:
        """Find the mode. ``weights`` optionally overrides object weights (used by Bootstrap)."""
        space = self._resolve_space()
        if space == "simplex":
            return self._run_em(weights)
        return self._run_lbfgs(weights)

    # ------------------------------------------------------------------- EM
    def _objective(self, v: np.ndarray, weights) -> float:
        m = self.model
        if weights is None:
            ll = m.log_likelihood(v, absolute=False)
        else:
            from .. import _kernels as kern

            ll, _, _ = kern.loglike_and_counts(m._L, weights, v)
            if m.kind == "rates":
                ll -= float(m._exposure() @ v)
            elif m.selection is not None:
                ll -= float(weights.sum()) * np.log(float(m.selection @ v))
        # components clipped to zero are active constraints: evaluate the prior at a tiny
        # positive value so that the objective stays finite (and constant) for them
        lp = float(m.prior.log_density(np.maximum(v, 1e-300)))
        return ll + lp

    def _run_em(self, weights) -> PointEstimate:
        m = self.model
        fractions = m.kind == "fractions"
        v = interior_start(m) if self.init is None else self.init.copy()
        if fractions:
            v = v / v.sum()

        def F(x):
            out = em_update(m, x, weights)
            return out / out.sum() if fractions else out

        f_old = self._objective(v, weights)
        converged = False
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):  # noqa: B007 (used after the loop)
            if self.accelerate:
                v_new, f_new = self._squarem_step(F, v, weights, fractions)
            else:
                v_new = F(v)
                f_new = self._objective(v_new, weights)
            dv = np.max(np.abs(v_new - v)) / max(np.max(np.abs(v_new)), 1e-300)
            df = abs(f_new - f_old) / (abs(f_old) + 1.0)
            v, f_old = v_new, f_new
            if df <= self.tol and dv <= 1e-7:
                converged = True
                break
        if not converged:
            warnings.warn(
                f"EM did not converge in {self.max_iter} iterations.",
                ConvergenceWarning,
                stacklevel=3,
            )
        gap = m.kkt_gap(v) if weights is None else np.nan
        return PointEstimate(
            m,
            v,
            method="map",
            log_posterior=f_old,
            log_likelihood=m.log_likelihood(v) if weights is None else np.nan,
            kkt_gap=gap,
            n_iter=n_iter,
            converged=converged,
            info={"space": "simplex", "algorithm": "squarem" if self.accelerate else "em"},
        )

    def _squarem_step(self, F, v, weights, fractions):
        r1 = F(v)
        r2 = F(r1)
        d1 = r1 - v
        d2 = r2 - 2.0 * r1 + v
        n2 = float(np.sqrt(np.sum(d2**2)))
        f2 = self._objective(r2, weights)
        if n2 == 0.0:
            return r2, f2
        step = min(-float(np.sqrt(np.sum(d1**2))) / n2, -1.0)
        for _ in range(30):
            cand = v - 2.0 * step * d1 + step**2 * d2
            if np.all(cand >= 0) and np.all(np.isfinite(cand)) and cand.sum() > 0:
                if fractions:
                    cand = cand / cand.sum()
                cand = F(cand)  # stabilisation step keeps EM's monotonicity guarantees
                fc = self._objective(cand, weights)
                if fc >= f2 - 1e-12 * abs(f2):
                    return cand, fc
            if step == -1.0:
                break
            step = (step - 1.0) / 2.0
            step = min(step, -1.0)
        return r2, f2

    # --------------------------------------------------------------- L-BFGS
    def _run_lbfgs(self, weights) -> PointEstimate:
        m = self.model
        if weights is not None:
            keep = weights > 0
            lk = m.likelihoods.subset(np.flatnonzero(keep)).with_weights(weights[keep])
            m = PopulationModel(lk, m.prior_spec, selection=m.selection_spec)
        v0 = interior_start(m) if self.init is None else self.init
        x0 = m.from_value(v0)

        def fun(x):
            lp, g = m.log_prob_and_grad(x)
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
        lp, g = m.log_prob_and_grad(x)
        gnorm = float(np.max(np.abs(g))) if g.size else 0.0
        converged = bool(res.success) or gnorm < 1e-5 * scale
        if not converged:
            warnings.warn(
                f"L-BFGS did not converge ({res.message}); max |gradient| = {gnorm:.2e}.",
                ConvergenceWarning,
                stacklevel=3,
            )
        v = m.to_value(x)
        return PointEstimate(
            self.model,
            v,
            method="map",
            x=x,
            log_posterior=lp,
            log_likelihood=m.log_likelihood(v),
            kkt_gap=np.nan,
            n_iter=int(res.nit),
            converged=converged,
            info={"space": "unconstrained", "algorithm": "l-bfgs-b", "grad_norm": gnorm},
        )
