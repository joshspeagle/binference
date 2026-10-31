"""Base classes for population priors.

A :class:`PopulationPrior` is a lightweight configuration object (what the user writes);
``prior.bind(bins)`` validates it against the bins and returns a
:class:`BoundPrior` that implements the numerical interface used by models and
engines:

* an unconstrained parametrisation ``x`` (``forward``) mapping to softmax logits
  (fractions) or log-rates (rates), with the prior log-density in ``x``-space
  (including Jacobians) and its gradient by the chain rule;
* prior draws, and (where defined) a closed-form density;
* for conjugate priors, posterior draws / modes given latent counts and
  mean-field variational updates.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Any

import numpy as np

from ..bins import Bins

__all__ = ["BoundPrior", "PopulationPrior"]

#: Floor of drawn fractions and rates, so that their logarithms stay finite.
_TINY = 1e-300


def _log_gamma_variates(a: Any, rng: np.random.Generator, shape: tuple) -> np.ndarray:
    """Logarithms of Gamma(a, 1) draws, without underflow for small shapes.

    For ``min(a) < 1`` they use ``Gamma(a) = Gamma(a + 1) * U**(1/a)`` in log space;
    ordinary draws underflow to exactly 0 (about 12% of ``Gamma(0.003)`` draws).
    """
    a = np.broadcast_to(np.asarray(a, dtype=float), shape)
    if a.size == 0 or np.min(a) >= 1.0:
        return np.log(rng.standard_gamma(a))
    return np.log(rng.standard_gamma(a + 1.0)) + np.log(rng.random(shape)) / a


class PopulationPrior:
    """Base class for priors on population fractions or rates."""

    def bind(self, bins: Bins) -> BoundPrior:  # pragma: no cover - abstract
        raise NotImplementedError

    def _params(self) -> dict[str, Any]:
        # the constructor arguments, read from the attributes of the same name (audit P-14);
        # a subclass that stores an argument under another name simply omits it
        params = list(inspect.signature(type(self).__init__).parameters.values())[1:]
        named = (p.name for p in params if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD))
        return {k: getattr(self, k) for k in named if hasattr(self, k)}

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={_short(v)}" for k, v in self._params().items())
        return f"{type(self).__name__}({args})"

    def _fingerprint(self) -> str:
        """A digest of the type and every argument in full (the repr shortens arrays)."""
        h = hashlib.sha256(type(self).__name__.encode())
        for k, v in self._params().items():
            h.update(k.encode())
            if isinstance(v, np.ndarray):
                h.update(f"{v.dtype}{v.shape}".encode() + np.ascontiguousarray(v).tobytes())
            else:
                h.update((v._fingerprint() if hasattr(v, "_fingerprint") else repr(v)).encode())
        return h.hexdigest()


def _short(v: Any) -> str:
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
    """Numerical interface of a prior bound to specific bins.

    :meth:`PopulationPrior.bind` returns one, and the engines use it through the model:

    * NUTS, Laplace and L-BFGS: :meth:`forward`, :meth:`log_prior_and_grad` and
      :meth:`init_from_value`, in unconstrained coordinates ``x`` of size ``dim``;
    * MAP on the simplex and the KKT gap, when ``has_density``: :meth:`log_density`,
      :meth:`grad_log_density` and :meth:`unbounded_density`;
    * Gibbs, EM and mean-field VB, when ``conjugate``: :meth:`posterior_sample`,
      :meth:`map_from_counts` and the ``vb_`` methods, given the number of objects in
      each bin (drawn by Gibbs, expected by EM and VB);
    * prior draws and simulation: :meth:`sample`.

    A subclass sets ``dim`` and the flags below, and implements the methods its flags
    promise; the others raise ``NotImplementedError``. Values (fractions or rates) have
    shape ``(K,)``, or ``(n, K)`` for ``n`` draws.

    Parameters
    ----------
    prior : PopulationPrior
        The prior.
    bins : Bins
        The bins it is bound to.
    """

    #: ``"fractions"`` (a distribution on the simplex) or ``"rates"``
    kind: str = "fractions"
    #: whether the methods Gibbs, EM and mean-field VB use are implemented
    conjugate: bool = False
    #: whether the methods MAP on the simplex and the KKT gap use are implemented
    has_density: bool = False
    #: whether it has partial non-centring weights (``with_centering``; the logistic family)
    has_centering: bool = False
    #: counts its mode adds to the data's, ``alpha - 1`` (Dirichlet) or ``a - 1``
    #: (GammaRates), as in EM's M-step; ``None`` for other priors
    pseudocounts: np.ndarray | None = None

    def __init__(self, prior: PopulationPrior, bins: Bins) -> None:
        self.prior = prior
        self.bins = bins
        self.K = bins.K

    # -- unconstrained parametrisation --------------------------------------
    #: size of the unconstrained coordinates ``x``
    dim: int

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, Any]:  # pragma: no cover
        """Map unconstrained ``x`` to logits / log-rates ``eta`` (and a cache).

        Parameters
        ----------
        x : ndarray (dim,)
            Unconstrained point.
        """
        raise NotImplementedError

    def log_prior_and_grad(
        self, x: np.ndarray, cache: Any, value: np.ndarray, d_eta: np.ndarray
    ) -> tuple[float, np.ndarray]:  # pragma: no cover
        """Prior log-density in ``x``-space and the total gradient.

        ``d_eta`` is the gradient of the log-likelihood with respect to
        ``eta``; the returned gradient is ``d/dx [log prior(x) + loglike]``.

        Parameters
        ----------
        x : ndarray (dim,)
            Unconstrained point.
        cache : object
            The cache :meth:`forward` returned for ``x``.
        value : ndarray (K,)
            Fractions or rates at ``x``.
        d_eta : ndarray (K,)
            Gradient of the log-likelihood with respect to ``eta``.
        """
        raise NotImplementedError

    def init_from_value(self, value: np.ndarray) -> np.ndarray:  # pragma: no cover
        """An unconstrained starting point near the given fractions or rates.

        Exact for the Dirichlet family and rates; for the logistic priors a
        ridge-regularised fit that shrinks towards the prior mean.

        Parameters
        ----------
        value : ndarray (K,)
            Fractions or rates.
        """
        raise NotImplementedError

    # -- draws and densities -------------------------------------------------
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:  # pragma: no cover
        """``n`` draws of the fractions or rates from the prior, shape ``(n, K)``.

        Parameters
        ----------
        n : int
            Number of draws.
        rng : numpy.random.Generator
            Random number generator.
        """
        raise NotImplementedError

    def log_density(self, value: np.ndarray) -> np.ndarray:
        """Log-density of fractions (on the simplex) or rates.

        Parameters
        ----------
        value : ndarray (..., K)
            Fractions or rates.
        """
        raise NotImplementedError(f"{type(self.prior).__name__} has no closed-form density.")

    def unbounded_density(self) -> bool:
        """Whether the density of the fractions (or rates) is unbounded at the boundary.

        Then its mode can lie on the boundary (components exactly zero).
        """
        return False

    def long_log_tails(self) -> bool:
        r"""Whether the prior gives bins without data long tails in the unconstrained coordinates.

        As a group of bins empties, its total fraction (or rate) :math:`s` goes to zero and
        the density of ``x``, the coordinates NUTS samples, falls like
        :math:`s^{a} = e^{a \log s}`: a tail of scale :math:`1/a` in :math:`\log s`, which
        NUTS needs long trajectories to cross. The tails count as long when some
        :math:`a < 1`, the threshold of a Dirichlet's :math:`\alpha_k < 1`. The exponent
        :math:`a` is the density's exponent at the boundary plus one for each bin in the
        group, from the Jacobian of ``x``: for one bin of a Dirichlet, :math:`\alpha_k - 1`
        becomes :math:`\alpha_k`. Where single bins have the longest tails, as for a
        Dirichlet or Gamma rates, the tails are long exactly when the density is
        unbounded, so this default returns :meth:`unbounded_density`; a prior whose
        groups of several bins have their own exponents, such as a tree, overrides it.
        """
        return self.unbounded_density()

    def grad_log_density(self, value: np.ndarray) -> np.ndarray:
        """Gradient of :meth:`log_density` with respect to the fractions/rates.

        Components with zero value are returned as zero (they are handled as
        active constraints by callers).

        Parameters
        ----------
        value : ndarray (K,)
            Fractions or rates.
        """
        raise NotImplementedError(f"{type(self.prior).__name__} has no closed-form density.")

    def _kkt_violation(self, value, grad, include_prior=True):
        # max_k (g_k - rho . g over the bins with mass) for the gradient g of the MAP
        # objective on the simplex: `grad` from the likelihood, plus this prior's. At a zero
        # the prior's gradient is taken at _TINY, as in the objective: hugely negative where
        # the density is unbounded there (no violation), hugely positive where it vanishes
        # (DECISIONS #71)
        if include_prior and self.has_density:
            grad = grad + self.grad_log_density(np.maximum(value, _TINY))
        pos = value > 0
        return float(np.max(grad - value[pos] @ grad[pos]))

    # -- conjugate updates ----------------------------------------------------
    def posterior_sample(
        self, counts: np.ndarray, rng: np.random.Generator, exposure: np.ndarray | None = None
    ) -> np.ndarray:
        """A draw of the fractions or rates given ``counts``, shape ``(K,)``: the
        conditional Gibbs draws from at every iteration.

        Parameters
        ----------
        counts : ndarray (K,)
            Number of objects in each bin.
        rng : numpy.random.Generator
            Random number generator.
        exposure : ndarray (K,), optional
            Exposure of each bin, for rates (``None``: one); fractions ignore it.
        """
        raise NotImplementedError(f"{type(self.prior).__name__} is not conjugate.")

    def map_from_counts(
        self, counts: np.ndarray, exposure: np.ndarray | None = None
    ) -> np.ndarray:
        """The mode on the simplex (or of the rates) of the fractions given ``counts``,
        shape ``(K,)``: EM's M-step.

        Parameters
        ----------
        counts : ndarray (K,)
            Number of objects in each bin.
        exposure : ndarray (K,), optional
            Exposure of each bin, for rates (``None``: one); fractions ignore it.
        """
        raise NotImplementedError(f"{type(self.prior).__name__} is not conjugate.")

    def vb_update(self, counts: np.ndarray, exposure: np.ndarray | None = None) -> Any:
        """Mean-field VB's state given ``counts``: the parameters of the variational
        posterior, read by the other ``vb_`` methods.

        Parameters
        ----------
        counts : ndarray (K,)
            Expected number of objects in each bin.
        exposure : ndarray (K,), optional
            Exposure of each bin, for rates (``None``: one); fractions ignore it.
        """
        raise NotImplementedError(f"{type(self.prior).__name__} is not conjugate.")

    def vb_expected_log(self, state: Any) -> np.ndarray:
        """Expected logarithms of the fractions or rates under ``state``, shape ``(K,)``.

        Parameters
        ----------
        state : object
            What :meth:`vb_update` returned.
        """
        raise NotImplementedError

    def vb_sample(self, state: Any, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` draws from the variational posterior ``state``, shape ``(n, K)``.

        Parameters
        ----------
        state : object
            What :meth:`vb_update` returned.
        n : int
            Number of draws.
        rng : numpy.random.Generator
            Random number generator.
        """
        raise NotImplementedError

    def vb_mean(self, state: Any) -> np.ndarray:
        """Mean of the fractions or rates under ``state``, shape ``(K,)``.

        Parameters
        ----------
        state : object
            What :meth:`vb_update` returned.
        """
        raise NotImplementedError
