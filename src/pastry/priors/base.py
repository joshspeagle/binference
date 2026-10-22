"""Base classes for population priors.

A :class:`Prior` is a lightweight configuration object (what the user writes);
``prior.bind(bins)`` validates it against the bins and returns a
:class:`BoundPrior` that implements the numerical interface used by models and
engines:

* an unconstrained parametrisation ``x`` (``forward``) mapping to softmax logits
  (fractions) or log-rates (rates), with the prior log-density in ``x``-space
  (including Jacobians) and its gradient by the chain rule;
* prior draws, and (where defined) the density on the simplex and a
  unit-cube ``prior_transform`` for nested samplers;
* for conjugate priors, posterior draws / modes given latent counts and
  mean-field variational updates.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..bins import Bins

__all__ = ["BoundPrior", "Prior"]


class Prior:
    """Base class for priors on population fractions or rates."""

    #: ``"fractions"`` (a distribution on the simplex) or ``"rates"``
    kind: str = "fractions"

    def bind(self, bins: Bins) -> BoundPrior:  # pragma: no cover - abstract
        raise NotImplementedError

    def _params(self) -> dict[str, Any]:  # pragma: no cover - overridden
        return {}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={_short(v)}" for k, v in self._params().items())
        return f"{type(self).__name__}({args})"


def _short(v: Any) -> str:
    if isinstance(v, np.ndarray):
        if v.size <= 4:
            return np.array2string(v, precision=3, separator=", ")
        return f"array(shape={v.shape})"
    return repr(v)


class BoundPrior:
    """Numerical interface of a prior bound to specific bins (internal)."""

    kind: str = "fractions"
    conjugate: bool = False
    has_density: bool = False
    has_prior_transform: bool = False

    def __init__(self, prior: Prior, bins: Bins) -> None:
        self.prior = prior
        self.bins = bins
        self.K = bins.K

    # -- unconstrained parametrisation --------------------------------------
    dim: int

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, Any]:  # pragma: no cover
        """Map unconstrained ``x`` to logits / log-rates ``eta`` (and a cache)."""
        raise NotImplementedError

    def log_prior_and_grad(
        self, x: np.ndarray, cache: Any, value: np.ndarray, d_eta: np.ndarray
    ) -> tuple[float, np.ndarray]:  # pragma: no cover
        """Prior log-density in ``x``-space and the total gradient.

        ``d_eta`` is the gradient of the log-likelihood with respect to
        ``eta``; the returned gradient is ``d/dx [log prior(x) + loglike]``.
        """
        raise NotImplementedError

    def init_from_value(self, value: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Unconstrained point corresponding (approximately) to fractions/rates."""
        raise NotImplementedError

    # -- draws and densities -------------------------------------------------
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def log_density(self, value: np.ndarray) -> np.ndarray:
        """Log-density of fractions (on the simplex) or rates."""
        raise NotImplementedError(f"{type(self.prior).__name__} has no closed-form density.")

    def grad_log_density(self, value: np.ndarray) -> np.ndarray:
        """Gradient of :meth:`log_density` with respect to the fractions/rates.

        Components with zero value are returned as zero (they are handled as
        active constraints by callers).
        """
        raise NotImplementedError(f"{type(self.prior).__name__} has no closed-form density.")

    def prior_transform(self, u: np.ndarray) -> np.ndarray:
        """Map the unit hypercube ``[0, 1]^(K-1)`` to prior draws (nested sampling)."""
        raise NotImplementedError(f"{type(self.prior).__name__} has no prior transform.")

    # -- conjugate updates ----------------------------------------------------
    def posterior_sample(
        self, counts: np.ndarray, rng: np.random.Generator, exposure: np.ndarray | None = None
    ) -> np.ndarray:
        raise NotImplementedError(f"{type(self.prior).__name__} is not conjugate.")

    def map_from_counts(
        self, counts: np.ndarray, exposure: np.ndarray | None = None
    ) -> np.ndarray:
        raise NotImplementedError(f"{type(self.prior).__name__} is not conjugate.")

    def vb_update(self, counts: np.ndarray, exposure: np.ndarray | None = None) -> Any:
        raise NotImplementedError(f"{type(self.prior).__name__} is not conjugate.")

    def vb_expected_log(self, state: Any) -> np.ndarray:
        raise NotImplementedError

    def vb_sample(self, state: Any, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def vb_mean(self, state: Any) -> np.ndarray:
        raise NotImplementedError
