"""Mean-field variational Bayes (coordinate ascent) for conjugate priors."""

from __future__ import annotations

import warnings

import numpy as np

from .. import _kernels as kern
from .._utils import RNGLike
from ..model import PopulationModel
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
    :class:`~pastry.ApproximationWarning`; prefer NUTS for error bars.

    With a selection function on fractions, the expected number of missed
    objects is plugged in at the current mean (an additional approximation).
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
        if not model.prior.conjugate:
            raise TypeError("Mean-field VB needs a conjugate prior; use NUTS or Laplace instead.")
        self.tol = float(tol)
        self.max_iter = int(max_iter)

    def fit(self):
        """Run coordinate ascent; returns ``(state, n_iter, converged)``."""
        m = self.model
        prior = m.prior
        exposure = m._exposure() if m.kind == "rates" else None
        state = prior.vb_update(np.full(m.K, m.W / m.K), exposure=exposure)
        mean_old = prior.vb_mean(state)
        converged = False
        it = 0
        for it in range(1, self.max_iter + 1):
            elog = prior.vb_expected_log(state)
            t = np.exp(elog - np.max(elog))
            _, nhat, _ = kern.loglike_and_counts(m._L, m._w, t)
            if m.kind == "fractions" and m.selection is not None:
                S = m.selection
                rho = mean_old / mean_old.sum()
                nhat = nhat + m.W * rho * (1.0 - S) / float(S @ rho)
            state = prior.vb_update(nhat, exposure=exposure)
            mean = prior.vb_mean(state)
            if np.max(np.abs(mean - mean_old)) <= self.tol * max(np.max(np.abs(mean)), 1e-300):
                converged = True
                break
            mean_old = mean
        if not converged:
            warnings.warn("Mean-field VB did not converge.", ConvergenceWarning, stacklevel=3)
        return state, it, converged

    def run(self, n_samples: int = 4000) -> PopulationResult:
        """Fit and return ``n_samples`` independent draws from the variational posterior."""
        state, it, converged = self.fit()
        draws = self.model.prior.vb_sample(state, int(n_samples), self.rng)
        warnings.warn(
            "Mean-field VB underestimates posterior uncertainty, especially for broad per-object "
            "PDFs; use NUTS (the default) for error bars.",
            ApproximationWarning,
            stacklevel=2,
        )
        info = {"n_iter": it, "converged": converged, "state": state}
        return PopulationResult(self.model, draws[None], method="vb", kind=self.model.kind,
                                approximate=True, info=info)  # fmt: skip
