"""Bootstrap distribution of the MAP / maximum-likelihood estimator."""

from __future__ import annotations

import warnings

import numpy as np

from .._utils import RNGLike, is_integer_array, progress, warn
from ..model import PopulationModel
from ..results import PopulationResult
from ..warnings import ApproximationWarning, ConvergenceWarning
from .base import Engine
from .optimize import MAP

__all__ = ["Bootstrap"]


class Bootstrap(Engine):
    """Bootstrap over objects of the MAP / maximum-likelihood estimator.

    ``scheme="nonparametric"`` resamples objects with replacement;
    ``scheme="bayesian"`` reweights them with Dirichlet weights (Rubin 1981).
    Integer weights are multiplicities, so a row of weight ``w`` stands for
    ``w`` objects that are resampled individually. For rate models the number
    of objects is itself random: objects get Poisson(1) counts (or Gamma(1)
    weights), so the total rate varies between replicates. Every replicate is
    fitted from an interior starting point; ``info["n_not_converged"]`` counts
    the replicates whose fit did not converge (a warning names it).

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
        scheme: str = "nonparametric",
        tol: float = 1e-8,
        max_iter: int = 5000,
        rng: RNGLike = None,
        progress: bool = False,
    ) -> None:
        super().__init__(model, rng=rng, progress=progress)
        if scheme not in ("nonparametric", "bayesian"):
            raise ValueError("scheme must be 'nonparametric' or 'bayesian'.")
        self.scheme = scheme
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def _replicate_weights(self, w: np.ndarray, integer: bool, rates: bool) -> np.ndarray:
        rng = self.rng
        N = w.size
        if self.scheme == "nonparametric":
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
        integer = is_integer_array(w)
        rates = m.kind == "rates"
        base = MAP(m, tol=self.tol, max_iter=self.max_iter).run()  # warns if it fails
        # EM cannot revive components that are exactly (or nearly) zero, and maximum
        # likelihood estimates on grids are often sparse: start replicates inside
        start = 0.5 * base.value + 0.5 * base.value.mean()
        out = np.empty((int(n_samples), m.K))
        n_not_converged = 0
        for b in progress(range(int(n_samples)), n_samples, "Bootstrap", self.progress):
            wb = self._replicate_weights(w, integer, rates)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                mw = m._with_weights(wb)  # zero-weight objects dropped (audit M-1)
                engine = MAP(mw, tol=self.tol, max_iter=self.max_iter, init=start)
                engine._diagnostics = False  # only the value is used
                est = engine.run()
            out[b] = est.value
            n_not_converged += not est.converged
        if n_not_converged:  # their warnings were silenced one by one (audit G-8)
            warn(
                f"{n_not_converged} of {int(n_samples)} bootstrap fits did not converge; raise "
                "max_iter.",
                ConvergenceWarning,
            )
        warn(
            "Bootstrap replicates describe the sampling distribution of the MAP estimator, not "
            "the posterior.",
            ApproximationWarning,
        )
        return PopulationResult(
            m,
            out[None],
            method="bootstrap",
            approximate=True,
            info={
                "scheme": self.scheme,
                "estimate": base.value,
                "n_not_converged": n_not_converged,
            },
        )
