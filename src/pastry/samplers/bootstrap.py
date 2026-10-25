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

    def run(self, n_samples: int = 200) -> PopulationResult:
        """Return ``n_samples`` bootstrap replicates of the estimate."""
        m = self.model
        w = m.likelihoods.weights
        N = m.N
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            base = MAP(m, tol=self.tol, max_iter=self.max_iter).run()
        out = np.empty((int(n_samples), m.K))
        for b in progress(range(int(n_samples)), n_samples, "Bootstrap", self.progress):
            if self.kind == "nonparametric":
                wb = w * self.rng.multinomial(N, np.full(N, 1.0 / N))
            else:
                g = self.rng.gamma(1.0, 1.0, size=N)
                wb = w * g * (N / g.sum())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                est = MAP(m, tol=self.tol, max_iter=self.max_iter, init=base.value).run(weights=wb)
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
