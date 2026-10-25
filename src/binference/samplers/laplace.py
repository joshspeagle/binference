"""Laplace approximation with optional Pareto-smoothed importance resampling."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from .. import diagnostics as diag
from .._utils import RNGLike, warn
from ..model import PopulationModel
from ..results import PopulationResult
from ..warnings import ApproximationWarning, ConvergenceWarning
from .base import Engine
from .optimize import MAP

__all__ = ["Laplace"]


def _systematic_resample(w: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    c = np.cumsum(w)
    c /= c[-1]
    u = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(c, u), w.size - 1)


class Laplace(Engine):
    """Gaussian (Laplace) approximation in unconstrained coordinates.

    The posterior mode in unconstrained coordinates is found with L-BFGS, the
    Hessian is computed from central differences of the exact gradient, and
    draws from the Gaussian are mapped to the simplex (or rates). With
    ``importance=True`` (default) the draws are corrected towards the exact
    posterior by Pareto-smoothed importance resampling; the Pareto shape
    ``k_hat`` (in ``info``) diagnoses whether the correction is reliable
    (``k_hat < 0.7``). Without a reliable correction the result is flagged
    ``approximate=True``.

    The approximation is excellent for well-populated bins with many objects
    and poorer for nearly empty bins, whose log-fractions have long tails.

    Parameters
    ----------
    model : PopulationModel
    importance : bool
        Apply PSIS importance resampling.
    n_proposals : int, optional
        Number of Gaussian proposals (default: ``2 * n_samples`` with
        importance resampling).
    df : float or None
        Degrees of freedom of the multivariate Student-t importance proposal
        (default 5); ``None`` uses the Gaussian itself. Heavier tails make the
        importance correction more robust.
    inflate : float
        Scale applied to the proposal's standard deviations.
    step : float
        Relative step for the finite-difference Hessian.

    Notes
    -----
    In ILR coordinates the posterior of nearly empty bins has a long one-sided
    tail, so for more than a handful of bins the importance correction often
    fails (``k_hat > 0.7``) and the result is flagged approximate: the Laplace
    approximation is then a fast summary of well-populated bins, not a
    replacement for NUTS.
    """

    name = "laplace"

    def __init__(
        self,
        model: PopulationModel,
        *,
        importance: bool = True,
        n_proposals: int | None = None,
        df: float | None = 5.0,
        inflate: float = 1.0,
        step: float = 1e-5,
        rng: RNGLike = None,
        progress: bool = False,
    ) -> None:
        super().__init__(model, rng=rng, progress=progress)
        self.importance = bool(importance)
        self.n_proposals = n_proposals
        if df is not None and df <= 0:
            raise ValueError("df must be positive (or None for a Gaussian proposal).")
        self.df = None if df is None else float(df)
        self.inflate = float(inflate)
        self.step = float(step)

    def mode_and_covariance(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mode, covariance and Hessian in unconstrained coordinates."""
        m = self.model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            est = MAP(m, space="unconstrained").run()
        x0 = est.x
        d = x0.size
        H = np.empty((d, d))
        for j in range(d):
            h = self.step * max(1.0, abs(x0[j]))
            e = np.zeros(d)
            e[j] = h
            gp = m.log_prob_and_grad(x0 + e)[1]
            gm = m.log_prob_and_grad(x0 - e)[1]
            H[:, j] = (gp - gm) / (2.0 * h)
        H = 0.5 * (H + H.T)
        lam, U = np.linalg.eigh(-H)
        if np.any(lam <= 0):
            warn(
                "The Hessian at the mode is not positive definite; clipping eigenvalues.",
                ApproximationWarning,
            )
            lam = np.maximum(lam, 1e-8 * max(1.0, lam.max()))
        cov = (U / lam) @ U.T
        return x0, cov, H

    def run(self, n_samples: int = 4000) -> PopulationResult:
        """Draw ``n_samples`` (approximately) from the posterior."""
        m = self.model
        x0, cov, _ = self.mode_and_covariance()
        d = x0.size
        lam, U = np.linalg.eigh(cov)
        lam = np.maximum(lam, 1e-300)
        sd_mat = U * np.sqrt(lam) * self.inflate
        M = self.n_proposals or (2 * n_samples if self.importance else n_samples)
        Z = self.rng.standard_normal((M, d))
        if self.importance and self.df is not None:
            Z = Z / np.sqrt(self.rng.chisquare(self.df, size=M) / self.df)[:, None]
        X = x0 + Z @ sd_mat.T
        info: dict[str, Any] = {"mode": x0, "cov": cov}
        approximate = True
        if self.importance:
            # log q up to a constant shared by all proposals
            if self.df is None:
                log_q = -0.5 * np.sum(Z**2, axis=1)
            else:
                log_q = -0.5 * (self.df + d) * np.log1p(np.sum(Z**2, axis=1) / self.df)
            log_p = np.array([m.log_prob(x) for x in X])
            lw, k_hat = diag.psis(log_p - log_q)
            w = np.exp(lw)
            info["k_hat"] = float(k_hat)
            info["ess_importance"] = float(1.0 / np.sum(w**2))
            idx = _systematic_resample(w, n_samples, self.rng)
            X = X[idx]
            approximate = not (k_hat < 0.7)
            if approximate:
                warn(
                    f"Pareto k-hat = {k_hat:.2f} >= 0.7: the Laplace approximation is not a good "
                    "importance proposal; results are approximate (consider NUTS).",
                    ApproximationWarning,
                )
        else:
            X = X[:n_samples]
            warn(
                "Uncorrected Laplace approximation: results are approximate.", ApproximationWarning
            )
        vals = m.transform(X)
        return PopulationResult(
            m,
            vals[None],
            method="laplace",
            kind=m.kind,
            x=X[None],
            approximate=approximate,
            info=info,
        )
