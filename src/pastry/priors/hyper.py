"""Scalar hyperpriors for positive hyperparameters (scales, length scales, ...).

Each hyperprior is parametrised on ``u = log(theta)`` for sampling: the
methods return the log-density of ``u`` (including the Jacobian ``theta``) and
its derivative. :class:`Fixed` marks a hyperparameter that is not inferred.
"""

from __future__ import annotations

import numpy as np
from scipy import special

__all__ = ["Fixed", "Gamma", "HalfNormal", "Hyperprior", "InverseGamma", "LogNormal"]


class Hyperprior:
    """Base class for priors on a positive scalar hyperparameter."""

    fixed: bool = False

    def logpdf_u(self, u: float) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def dlogpdf_u(self, u: float) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def median(self) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{type(self).__name__}({args})"


class Fixed(Hyperprior):
    """A hyperparameter held fixed at ``value`` (not inferred)."""

    fixed = True

    def __init__(self, value: float) -> None:
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Fixed hyperparameter values must be positive and finite.")
        self.value = float(value)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.full(n, self.value)

    def median(self) -> float:
        return self.value


class HalfNormal(Hyperprior):
    """Half-normal prior with scale ``scale`` on a positive hyperparameter."""

    def __init__(self, scale: float = 1.0) -> None:
        if not scale > 0:
            raise ValueError("scale must be positive.")
        self.scale = float(scale)

    def logpdf_u(self, u: float) -> float:
        t = np.exp(u)
        s = self.scale
        return float(np.log(2.0) - np.log(s) - 0.5 * np.log(2 * np.pi) - 0.5 * (t / s) ** 2 + u)

    def dlogpdf_u(self, u: float) -> float:
        t = np.exp(u)
        return float(1.0 - (t / self.scale) ** 2)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.abs(rng.normal(0.0, self.scale, size=n))

    def median(self) -> float:
        return float(self.scale * special.ndtri(0.75))


class LogNormal(Hyperprior):
    """Log-normal prior: ``log(theta) ~ Normal(mu, sigma)``."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        if not sigma > 0:
            raise ValueError("sigma must be positive.")
        self.mu = float(mu)
        self.sigma = float(sigma)

    def logpdf_u(self, u: float) -> float:
        z = (u - self.mu) / self.sigma
        return float(-0.5 * z * z - np.log(self.sigma) - 0.5 * np.log(2 * np.pi))

    def dlogpdf_u(self, u: float) -> float:
        return float(-(u - self.mu) / self.sigma**2)

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.exp(rng.normal(self.mu, self.sigma, size=n))

    def median(self) -> float:
        return float(np.exp(self.mu))


class InverseGamma(Hyperprior):
    """Inverse-gamma prior with shape ``a`` and scale ``b``."""

    def __init__(self, a: float = 2.0, b: float = 1.0) -> None:
        if not (a > 0 and b > 0):
            raise ValueError("a and b must be positive.")
        self.a = float(a)
        self.b = float(b)

    def logpdf_u(self, u: float) -> float:
        return float(
            self.a * np.log(self.b) - special.gammaln(self.a) - self.a * u - self.b * np.exp(-u)
        )

    def dlogpdf_u(self, u: float) -> float:
        return float(-self.a + self.b * np.exp(-u))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.b / rng.gamma(self.a, 1.0, size=n)

    def median(self) -> float:
        return float(self.b / special.gammaincinv(self.a, 0.5))


class Gamma(Hyperprior):
    """Gamma prior with shape ``a`` and rate ``b``."""

    def __init__(self, a: float = 2.0, b: float = 1.0) -> None:
        if not (a > 0 and b > 0):
            raise ValueError("a and b must be positive.")
        self.a = float(a)
        self.b = float(b)

    def logpdf_u(self, u: float) -> float:
        return float(
            self.a * np.log(self.b) - special.gammaln(self.a) + self.a * u - self.b * np.exp(u)
        )

    def dlogpdf_u(self, u: float) -> float:
        return float(self.a - self.b * np.exp(u))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.gamma(self.a, 1.0 / self.b, size=n)

    def median(self) -> float:
        return float(special.gammaincinv(self.a, 0.5) / self.b)


def as_hyperprior(value: Hyperprior | float | int) -> Hyperprior:
    """Interpret a number as :class:`Fixed` and pass hyperpriors through."""
    if isinstance(value, Hyperprior):
        return value
    if isinstance(value, int | float | np.floating | np.integer):
        return Fixed(float(value))
    raise TypeError(f"Expected a Hyperprior or a positive number; got {type(value).__name__}.")
