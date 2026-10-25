"""Bootstrap distribution of the MAP / maximum-likelihood estimator."""

from __future__ import annotations

import warnings

import numpy as np

from .._utils import RNGLike, progress
from ..model import PopulationModel
from ..results import PopulationResult
from ..warnings import ApproximationWarning, ConvergenceWarning
from .base import Engine
from .optimize import MAP

__all__ = ["Bootstrap"]


class Bootstrap(Engine):
    """Bootstrap over objects of the MAP / maximum-likelihood estimator.

    ``kind="nonparametric"`` resamples objects with replacement;
    ``kind="bayesian"`` reweights them with Dirichlet weights (Rubin 1981).
    Integer weights are multiplicities, so a row of weight ``w`` stands for
    ``w`` objects that are resampled individually. For rate models the number
    of objects is itself random: objects get Poisson(1) counts (or Gamma(1)
    weights), so the total rate varies between replicates. Every replicate is
    fitted from an interior starting point.

    The result is the sampling distribution of an estimator, not a posterior:
    it is flagged ``approximate=True``. Useful for frequentist intervals and
    comparisons with unfolding / empirical-Bayes practice.
    """

    name = "bootstrap"
    approximate = True

    def __init__(
        self,
        model: PopulationModel,
        *,
        kind: str = "nonparametric",
        tol: float = 1e-8,
        max_iter: int = 5000,
        rng: RNGLike = None,
        progress: bool = False,
    ) -> None:
        super().__init__(model, rng=rng, progress=progress)
        if kind not in ("nonparametric", "bayesian"):
            raise ValueError("kind must be 'nonparametric' or 'bayesian'.")
        self.kind = kind
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def _replicate_weights(self, w: np.ndarray, integer: bool, rates: bool) -> np.ndarray:
        rng = self.rng
        N = w.size
        if self.kind == "nonparametric":
            if integer:  # resample the underlying objects, not the rows
                if rates:
                    return rng.poisson(w).astype(float)
                W = round(float(w.sum()))
                return rng.multinomial(W, w / w.sum()).astype(float)
            counts = rng.poisson(1.0, size=N) if rates else rng.multinomial(N, np.full(N, 1 / N))
            return w * counts
        # Bayesian bootstrap: a row of multiplicity w gets a Gamma(w, 1) weight (a sum of
        # w Exp(1) weights); fractions renormalise to the original total
        g = rng.gamma(w, 1.0) if integer else w * rng.gamma(1.0, 1.0, size=N)
        return g if rates else g * (w.sum() / g.sum())

    def run(self, n_samples: int = 200) -> PopulationResult:
        """Return ``n_samples`` bootstrap replicates of the estimate."""
        m = self.model
        w = m.likelihoods.weights
        integer = bool(np.all(np.abs(w - np.round(w)) < 1e-9))
        rates = m.kind == "rates"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            base = MAP(m, tol=self.tol, max_iter=self.max_iter).run()
        # EM cannot revive components that are exactly (or nearly) zero, and maximum
        # likelihood estimates on grids are often sparse: start replicates inside
        start = 0.5 * base.value + 0.5 * base.value.mean()
        out = np.empty((int(n_samples), m.K))
        for b in progress(range(int(n_samples)), n_samples, "Bootstrap", self.progress):
            wb = self._replicate_weights(w, integer, rates)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                est = MAP(m, tol=self.tol, max_iter=self.max_iter, init=start).run(weights=wb)
            out[b] = est.value
        warnings.warn(
            "Bootstrap replicates describe the sampling distribution of the MAP estimator, not "
            "the posterior.",
            ApproximationWarning,
            stacklevel=2,
        )
        return PopulationResult(
            m,
            out[None],
            method="bootstrap",
            kind=m.kind,
            approximate=True,
            info={"kind": self.kind, "estimate": base.value},
        )
