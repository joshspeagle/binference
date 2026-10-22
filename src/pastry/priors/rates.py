r"""Priors on Poisson rates (expected counts per bin).

With rates :math:`\lambda_k` (expected number of objects in bin ``k``) the
data form an inhomogeneous Poisson process and selection enters as an exposure
:math:`S_k`:

.. math:: p(\lambda \mid D) \propto p(\lambda)\, e^{-S\cdot\lambda} \prod_i (L_i\cdot\lambda)^{w_i}.

Independent Gamma priors stay conjugate given latent bin labels.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import special

from ..bins import Bins
from .base import BoundPrior, Prior

__all__ = ["GammaRates"]


class GammaRates(Prior):
    r"""Independent Gamma priors on per-bin rates, :math:`\lambda_k \sim \mathrm{Gamma}(a_k, b_k)`.

    Use this to infer expected *counts* (e.g. occurrence rates, or number
    densities with a known exposure) rather than fractions. Results report
    both ``rates`` and the implied fractions ``rho = lambda / sum(lambda)``.
    Without selection effects, ``GammaRates(alpha, b)`` implies
    ``rho ~ Dirichlet(alpha)``.

    Parameters
    ----------
    shape : float or array_like (K,)
        Gamma shape parameters ``a_k`` (default 1).
    rate : float or array_like (K,)
        Gamma rate parameters ``b_k`` (default ``1e-6``, i.e. vague). Use an
        informative value for prior predictive checks or simulation-based
        calibration.
    """

    kind = "rates"

    def __init__(self, shape: Any = 1.0, rate: Any = 1e-6) -> None:
        a = np.asarray(shape, dtype=float)
        b = np.asarray(rate, dtype=float)
        if np.any(a <= 0) or not np.all(np.isfinite(a)):
            raise ValueError("shape must be finite and positive.")
        if np.any(b <= 0) or not np.all(np.isfinite(b)):
            raise ValueError("rate must be finite and positive.")
        self.shape = shape
        self.rate = rate

    def _params(self):
        return {"shape": self.shape, "rate": self.rate}

    def bind(self, bins: Bins) -> _BoundGammaRates:
        K = bins.K
        a = np.broadcast_to(np.asarray(self.shape, dtype=float), (K,)).copy()
        b = np.broadcast_to(np.asarray(self.rate, dtype=float), (K,)).copy()
        return _BoundGammaRates(self, bins, a, b)


class _BoundGammaRates(BoundPrior):
    kind = "rates"
    conjugate = True
    has_density = True

    def __init__(self, prior: Prior, bins: Bins, a: np.ndarray, b: np.ndarray) -> None:
        super().__init__(prior, bins)
        self.a = a
        self.b = b
        self.dim = self.K
        self._const = float(np.sum(a * np.log(b) - special.gammaln(a)))

    def forward(self, x):
        return x, None

    def log_prior_and_grad(self, x, cache, value, d_eta):
        lam = value
        lp = float(np.dot(self.a, x) - np.dot(self.b, lam)) + self._const
        return lp, self.a - self.b * lam + d_eta

    def init_from_value(self, value):
        lam = np.asarray(value, dtype=float)
        lam = np.maximum(lam, 1e-12 * max(np.max(lam), 1e-300))
        return np.log(lam)

    def sample(self, n, rng):
        return rng.gamma(self.a, 1.0 / self.b, size=(n, self.K))

    def log_density(self, value):
        lam = np.asarray(value, dtype=float)
        return np.sum(special.xlogy(self.a - 1.0, lam) - self.b * lam, axis=-1) + self._const

    def grad_log_density(self, value):
        v = np.asarray(value, dtype=float)
        pos = v > 0
        return np.where(pos, (self.a - 1.0) / np.where(pos, v, 1.0) - self.b, 0.0)

    def _exposure(self, exposure):
        return np.ones(self.K) if exposure is None else exposure

    def posterior_sample(self, counts, rng, exposure=None):
        S = self._exposure(exposure)
        return rng.gamma(self.a + counts, 1.0 / (self.b + S))

    def map_from_counts(self, counts, exposure=None):
        S = self._exposure(exposure)
        return np.maximum(self.a + np.asarray(counts, float) - 1.0, 0.0) / (self.b + S)

    def vb_update(self, counts, exposure=None):
        S = self._exposure(exposure)
        return (self.a + counts, self.b + S)

    def vb_expected_log(self, state):
        a, b = state
        return special.digamma(a) - np.log(b)

    def vb_sample(self, state, n, rng):
        a, b = state
        return rng.gamma(a, 1.0 / b, size=(n, self.K))

    def vb_mean(self, state):
        a, b = state
        return a / b
