"""Mean-field variational Bayes (coordinate ascent) for conjugate priors."""

from __future__ import annotations

import numpy as np

from .. import _kernels as kern
from .._utils import RNGLike, warn
from ..model import PopulationModel
from ..priors.base import _TINY
from ..results import PopulationResult
from ..warnings import ApproximationWarning, ConvergenceWarning
from .base import Engine

__all__ = ["MeanFieldVB"]


class MeanFieldVB(Engine):
    r"""Mean-field variational Bayes: :math:`q(\rho)\prod_i q(z_i)`.

    Coordinate ascent alternates :math:`q(z_i) \propto L_{ik}\exp E_q[\log\rho_k]`
    and the conjugate update of :math:`q(\rho)` (e.g. a Dirichlet). This is the
    algorithm of RAIL's ``VarInfStackSummarizer``. It is fast but **under-
    estimates posterior widths**, severely so when per-object PDFs are broad
    compared with the bins (it ignores the correlation between labels and the
    population). Results are always flagged ``approximate=True`` and emit an
    :class:`~binference.ApproximationWarning`; prefer NUTS for error bars.

    With a selection function on fractions, the expected number of missed
    objects is plugged in at the current mean (an additional approximation).

    No evidence lower bound is computed (with selection the plug-in update does
    not ascend one): iterations stop when the largest change of the mean is
    below ``tol`` times its largest value.

    Parameters
    ----------
    model : PopulationModel
        Must use a conjugate prior.
    tol : float
        Relative tolerance on the change of the mean.
    max_iter : int
        Maximum number of coordinate-ascent iterations.
    rng : int, numpy.random.Generator, or None
        For the draws from the variational posterior.
    progress : bool
        Accepted for a common signature; unused.
    """

    name = "vb"
    approximate = True

    def __init__(
        self,
        model: PopulationModel,
        *,
        tol: float = 1e-9,
        max_iter: int = 10000,
        rng: RNGLike = None,
        progress: bool = False,
    ) -> None:
        super().__init__(model, rng=rng, progress=progress)
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    @classmethod
    def unsupported(cls, model: PopulationModel) -> str | None:
        if model.prior.conjugate:
            return super().unsupported(model)
        return "Mean-field VB needs a conjugate prior; use NUTS or Laplace instead."

    def fit(self):
        """Run coordinate ascent; returns ``(state, n_iter, converged)``."""
        m = self.model
        prior = m.prior
        exposure = m._exposure
        state = prior.vb_update(np.full(m.K, m.W / m.K), exposure=exposure)
        mean_old = prior.vb_mean(state)
        converged = False
        it = 0
        for it in range(1, self.max_iter + 1):  # noqa: B007 (used after the loop)
            elog = prior.vb_expected_log(state)
            t = np.exp(elog - np.max(elog))
            _, nhat, _ = kern.loglike_and_counts(m._L, m._w, t)
            nhat = m._complete_counts(mean_old, nhat)
            state = prior.vb_update(nhat, exposure=exposure)
            mean = prior.vb_mean(state)
            if np.max(np.abs(mean - mean_old)) <= self.tol * max(np.max(np.abs(mean)), _TINY):
                converged = True
                break
            mean_old = mean
        if not converged:
            warn("Mean-field VB did not converge.", ConvergenceWarning)
        return state, it, converged

    def run(self, n_samples: int = 4000) -> PopulationResult:
        """Fit and return ``n_samples`` independent draws from the variational posterior."""
        state, it, converged = self.fit()
        draws = self.model.prior.vb_sample(state, int(n_samples), self.rng)
        warn(
            "Mean-field VB underestimates posterior uncertainty, especially for broad per-object "
            "PDFs; use an exact engine (NUTS or Gibbs) for error bars.",
            ApproximationWarning,
        )
        info = {"n_iter": it, "converged": converged, "state": state}
        return PopulationResult(
            self.model,
            draws[None],
            method="vb",
            approximate=True,
            info=info,
        )
