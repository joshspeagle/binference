"""Base classes for population priors.

A :class:`PopulationPrior` is a lightweight configuration object (what the user writes);
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

import inspect
from typing import Any

import numpy as np
from scipy import special

from ..bins import Bins

__all__ = ["BoundPrior", "PopulationPrior"]

#: floor of drawn fractions and rates, so their logarithms stay finite
_TINY = 1e-300


def _log_gamma_variates(a: Any, rng: np.random.Generator, shape: tuple) -> np.ndarray:
    """Logarithms of Gamma(a, 1) draws, without underflow for small shapes.

    For ``min(a) < 1`` they use ``Gamma(a) = Gamma(a + 1) * U**(1/a)`` in log space;
    ordinary draws underflow to exactly 0 (about 12% of ``Gamma(0.003)`` draws).
    """
    a = np.broadcast_to(np.asarray(a, dtype=float), shape)
    if np.min(a) >= 1.0:
        return np.log(rng.standard_gamma(a))
    return np.log(rng.standard_gamma(a + 1.0)) + np.log(rng.random(shape)) / a


def _stick_break(a: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Dirichlet(a) from uniforms ``u`` (last axis of length ``len(a) - 1``) by stick-breaking."""
    tail = np.cumsum(a[::-1])[::-1]  # tail[k] = sum_{j >= k} a_j
    out = np.empty((*u.shape[:-1], a.size))
    remaining = np.ones(u.shape[:-1])
    for k in range(a.size - 1):
        v = special.betaincinv(a[k], tail[k + 1], u[..., k])
        out[..., k] = remaining * v
        remaining = remaining * (1.0 - v)
    out[..., -1] = remaining
    return out


class PopulationPrior:
    """Base class for priors on population fractions or rates."""

    #: ``"fractions"`` (a distribution on the simplex) or ``"rates"``
    kind: str = "fractions"

    def bind(self, bins: Bins) -> BoundPrior:  # pragma: no cover - abstract
        raise NotImplementedError

    def _params(self) -> dict[str, Any]:
        # every constructor argument, read from the attribute of the same name (audit P-14)
        names = list(inspect.signature(type(self).__init__).parameters)[1:]
        return {k: getattr(self, k) for k in names}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={_short(v)}" for k, v in self._params().items())
        return f"{type(self).__name__}({args})"


def _short(v: Any) -> str:
    if isinstance(v, list | tuple):
        try:
            v = np.asarray(v, dtype=float)
        except (TypeError, ValueError):  # ragged (a tree) or not numbers
            return repr(v)
    if isinstance(v, np.ndarray):
        if v.size <= 4:
            return np.array2string(v, precision=3, separator=", ")
        return f"array(shape={v.shape})"
    return repr(v)


def _per_item(value: Any, n: int, name: str) -> np.ndarray:
    """A scalar broadcast to ``n`` values, or an array of exactly ``n`` values."""
    v = np.asarray(value)
    if v.ndim and v.shape != (n,):
        raise ValueError(f"{name} must be a scalar or have length {n}; got shape {v.shape}.")
    return np.broadcast_to(v, (n,)).copy()


class BoundPrior:
    """Numerical interface of a prior bound to specific bins (internal)."""

    kind: str = "fractions"
    conjugate: bool = False
    has_density: bool = False
    has_prior_transform: bool = False

    def __init__(self, prior: PopulationPrior, bins: Bins) -> None:
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
        """An unconstrained starting point near the given fractions or rates.

        Exact for the Dirichlet family and rates; for the logistic priors a
        ridge-regularised fit that shrinks towards the prior mean.
        """
        raise NotImplementedError

    # -- draws and densities -------------------------------------------------
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def log_density(self, value: np.ndarray) -> np.ndarray:
        """Log-density of fractions (on the simplex) or rates."""
        raise NotImplementedError(f"{type(self.prior).__name__} has no closed-form density.")

    def unbounded_density(self) -> bool:
        """Whether the density of the fractions (or rates) is unbounded at the boundary.

        Then its mode can lie on the boundary (components exactly zero), and
        NUTS meets very long tails in log space for components without data.
        """
        return False

    def grad_log_density(self, value: np.ndarray) -> np.ndarray:
        """Gradient of :meth:`log_density` with respect to the fractions/rates.

        Components with zero value are returned as zero (they are handled as
        active constraints by callers).
        """
        raise NotImplementedError(f"{type(self.prior).__name__} has no closed-form density.")

    def prior_transform(self, u: np.ndarray) -> np.ndarray:
        """Map the unit hypercube ``[0, 1]^dim`` to prior draws (nested sampling).

        ``dim`` is ``K - 1`` for fractions and ``K`` for rates.
        """
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
