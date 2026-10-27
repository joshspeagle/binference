"""Scalar hyperpriors for positive hyperparameters (scales, length scales, ...).

Each hyperprior is parametrised on ``u = log(theta)`` for sampling: the
methods return the log-density of ``u`` (including the Jacobian ``theta``) and
its derivative. :class:`Fixed` marks a hyperparameter that is not inferred.
"""

from __future__ import annotations

from numbers import Real

import numpy as np
from scipy import optimize, special

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
    """A hyperparameter held fixed at ``value`` (not inferred).

    Parameters
    ----------
    value : float
        The fixed value.
    """

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
    """Half-normal prior with scale ``scale`` on a positive hyperparameter.

    Parameters
    ----------
    scale : float, optional
        Scale of the half-normal.
    """

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
    """Log-normal prior: ``log(theta) ~ Normal(mu, sigma)``.

    Parameters
    ----------
    mu : float, optional
        Mean of the log of the hyperparameter.
    sigma : float, optional
        Standard deviation of the log of the hyperparameter.
    """

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
    """Inverse-gamma prior with shape ``a`` and scale ``b``.

    Parameters
    ----------
    a : float, optional
        Shape.
    b : float, optional
        Scale.
    """

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

    @classmethod
    def from_quantiles(cls, lower: float, upper: float, tail: float = 0.01) -> InverseGamma:
        """The inverse gamma with ``P(< lower) = P(> upper) = tail``.

        Useful for GP length scales: put little mass below the resolution of
        the grid and above the size of the domain (Betancourt 2017, "Robust
        Gaussian processes in Stan").

        Parameters
        ----------
        lower, upper : float
            Values below and above which the prior puts mass ``tail``.
        tail : float, optional
            Prior mass below ``lower`` and above ``upper``.
        """
        if not 0 < lower < upper:
            raise ValueError("Need 0 < lower < upper.")
        if not 0 < tail < 0.5:
            raise ValueError("tail must be in (0, 0.5).")

        # The tail conditions fix b = lower * Qinv(a, tail) = upper * Qinv(a, 1 - tail)
        # (Qinv: inverse regularised upper incomplete gamma). The ratio
        # Qinv(a, tail) / Qinv(a, 1 - tail) falls monotonically from infinity to 1 as a
        # grows, so a follows from a bracketed one-dimensional root.
        target = np.log(upper / lower)

        def h(log_a):
            a = np.exp(log_a)
            with np.errstate(divide="ignore"):  # tiny a: the lower quantile underflows
                return (
                    np.log(special.gammainccinv(a, tail))
                    - np.log(special.gammainccinv(a, 1.0 - tail))
                    - target
                )

        lo_, hi_ = np.log(1e-3), np.log(1e7)
        if not (h(lo_) > 0 > h(hi_)):
            raise ValueError("Cannot match these quantiles with an inverse gamma.")
        log_a = optimize.brentq(h, lo_, hi_, xtol=1e-12)
        a = float(np.exp(log_a))
        b = float(lower * special.gammainccinv(a, tail))
        return cls(a, b)


class Gamma(Hyperprior):
    """Gamma prior with shape ``a`` and rate ``b``.

    Parameters
    ----------
    a : float, optional
        Shape.
    b : float, optional
        Rate.
    """

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
    number = isinstance(value, Real) or (
        isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype.kind in "iuf"
    )
    if number and not isinstance(value, bool | np.bool_):
        return Fixed(float(value))
    raise TypeError(f"Expected a Hyperprior or a positive number; got {type(value).__name__}.")
