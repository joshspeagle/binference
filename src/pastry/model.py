r"""The hierarchical population model.

For fractions :math:`\rho` (simplex priors and logistic priors):

.. math:: \log p(\rho \mid D) = \log p(\rho) + \sum_i w_i \log(L_i\cdot\rho)
          - W \log(S\cdot\rho) + \text{const},

where the selection term is present only when a selection function
:math:`S` is given. For rates :math:`\lambda` (:class:`~pastry.priors.GammaRates`):

.. math:: \log p(\lambda \mid D) = \log p(\lambda) + \sum_i w_i \log(L_i\cdot\lambda)
          - S\cdot\lambda + \text{const}.

Gradient-based engines work in the unconstrained coordinates of the prior
(isometric log-ratios for simplex priors, whitened fields for logistic priors,
log-rates for rates); the log-likelihood gradient with respect to the softmax
logits (or log-rates) is

.. math:: \partial\ell/\partial\eta_k = \hat n_k - W\rho_k S_k/(S\cdot\rho)
          \quad\text{or}\quad \hat n_k - S_k\lambda_k,

with expected counts :math:`\hat n_k = \rho_k \sum_i w_i L_{ik}/(L_i\cdot\rho)`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import linalg

from . import _kernels as kern
from ._simplex import softmax
from ._utils import RNGLike, as_generator
from .bins import Bins
from .likelihoods import Likelihoods, _scale_columns
from .priors import Dirichlet, Prior
from .priors.base import BoundPrior
from .selection import Selection, as_selection

__all__ = ["PopulationModel"]


class PopulationModel:
    """Likelihoods + prior (+ selection): the object every inference engine works on.

    Parameters
    ----------
    likelihoods : Likelihoods or array_like (N, K)
        Per-object bin-averaged likelihoods. A raw array is wrapped with
        ``Likelihoods(array, bins)``.
    prior : Prior, optional
        Population prior; defaults to ``Dirichlet(1.0)``. Use
        :class:`~pastry.priors.GammaRates` to infer rates instead of fractions.
    selection : array_like (K,) or Selection, optional
        Per-bin detection probability (fractions) or exposure (rates). Overrides
        any selection attached to ``likelihoods``.
    selection_mode : {"observed", "true"}, optional
        Required with an array-valued ``selection``. ``"observed"``: objects are
        selected on their observed data (or the likelihoods already include the
        detection probability); only the normalisation term is added.
        ``"true"``: objects are detected with probability ``S_k`` depending on
        their true bin; likelihood rows are multiplied by ``S_k`` as well.
    bins : Bins, optional
        Only used when ``likelihoods`` is a raw array.

    Notes
    -----
    ``likelihoods`` must be ordinary likelihoods :math:`p(d_i \\mid z)` evaluated at
    the observed data, *not* renormalised over the detectable data space (see
    :doc:`/user-guide/selection`).
    """

    def __init__(
        self,
        likelihoods: Likelihoods | Any,
        prior: Prior | None = None,
        *,
        selection: Any = None,
        selection_mode: str | None = None,
        bins: Bins | None = None,
    ) -> None:
        if not isinstance(likelihoods, Likelihoods):
            likelihoods = Likelihoods(likelihoods, bins)
        elif bins is not None and bins != likelihoods.bins:
            raise ValueError("bins differ from the bins attached to the likelihoods.")
        self.likelihoods = likelihoods
        self.prior_spec = Dirichlet(1.0) if prior is None else prior
        if not isinstance(self.prior_spec, Prior):
            raise TypeError("prior must be a pastry.priors.Prior instance.")
        self.prior: BoundPrior = self.prior_spec.bind(likelihoods.bins)
        if selection is None and selection_mode is None:
            sel = likelihoods.selection
        else:
            sel = as_selection(selection, selection_mode, likelihoods.K)
        self.selection_spec: Selection | None = sel
        #: per-bin selection values (or ``None``)
        self.selection = None if sel is None else sel.values
        self.selection_mode = None if sel is None else sel.mode
        L = likelihoods.values
        if sel is not None and sel.mode == "true":
            L = _scale_columns(L, sel.values)
        # effective per-object likelihoods (selection-weighted for mode "true")
        self._L = L
        self._w = likelihoods.weights
        self.W = float(self._w.sum())
        self._ll_const = float(np.dot(self._w, likelihoods.log_scale))

    # ----------------------------------------------------------- properties
    @property
    def bins(self) -> Bins:
        return self.likelihoods.bins

    @property
    def K(self) -> int:
        return self.likelihoods.K

    @property
    def N(self) -> int:
        return self.likelihoods.N

    @property
    def kind(self) -> str:
        """``"fractions"`` or ``"rates"`` (set by the prior)."""
        return self.prior.kind

    @property
    def dim(self) -> int:
        """Dimension of the unconstrained parameter vector."""
        return self.prior.dim

    def effective_likelihoods(self, likelihoods: Likelihoods):
        """Likelihood matrix of (new) objects with this model's selection applied."""
        L = likelihoods.values
        if self.selection is not None and self.selection_mode == "true":
            L = _scale_columns(L, self.selection)
        return L

    def __repr__(self) -> str:
        sel = "" if self.selection is None else f", selection={self.selection_mode!r}"
        return f"PopulationModel(N={self.N}, K={self.K}, prior={self.prior_spec!r}{sel})"

    # ------------------------------------------------- likelihood internals
    def _exposure(self) -> np.ndarray:
        return np.ones(self.K) if self.selection is None else self.selection

    def _ll_and_deta(self, value: np.ndarray) -> tuple[float, np.ndarray]:
        ll, nhat, _ = kern.loglike_and_counts(self._L, self._w, value)
        if not np.isfinite(ll):
            return -np.inf, np.zeros_like(value)
        if self.kind == "rates":
            S = self._exposure()
            return ll - float(S @ value), nhat - S * value
        if self.selection is None:
            return ll, nhat - self.W * value
        sp_ = float(self.selection @ value)
        return ll - self.W * np.log(sp_), nhat - self.W * value * self.selection / sp_

    def _value(self, eta: np.ndarray) -> np.ndarray:
        if self.kind == "rates":
            with np.errstate(over="ignore"):
                return np.exp(eta)
        return softmax(eta)

    # ------------------------------------------------ unconstrained space
    def log_prob_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """Log posterior density (up to a constant) and gradient in unconstrained space."""
        x = np.asarray(x, dtype=float)
        eta, cache = self.prior.forward(x)
        value = self._value(eta)
        if not np.all(np.isfinite(value)):
            return -np.inf, np.zeros_like(x)
        ll, d_eta = self._ll_and_deta(value)
        if not np.isfinite(ll):
            return -np.inf, np.zeros_like(x)
        lp, g = self.prior.log_prior_and_grad(x, cache, value, d_eta)
        return ll + lp + self._ll_const, g

    def log_prob(self, x: np.ndarray) -> float:
        """Log posterior density (up to a constant) in unconstrained space."""
        return self.log_prob_and_grad(x)[0]

    def to_value(self, x: np.ndarray) -> np.ndarray:
        """Fractions (or rates, for rate priors) at unconstrained ``x``."""
        eta, _ = self.prior.forward(np.asarray(x, dtype=float))
        return self._value(eta)

    def to_rho(self, x: np.ndarray) -> np.ndarray:
        """Population fractions at unconstrained ``x``."""
        v = self.to_value(x)
        return v / v.sum() if self.kind == "rates" else v

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map many unconstrained draws ``(n, dim)`` to values ``(n, K)``."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.array([self.to_value(x) for x in X])

    def from_value(self, value: np.ndarray) -> np.ndarray:
        """An unconstrained point corresponding to fractions (or rates) ``value``."""
        return self.prior.init_from_value(np.asarray(value, dtype=float))

    # ----------------------------------------------- fraction/rate space
    def log_likelihood(self, value: np.ndarray, absolute: bool = True) -> float:
        """Marginal log-likelihood of the population (including the selection term).

        With ``absolute=True`` the per-object log-scales of the likelihoods are
        included, so values are comparable across binnings and models.
        """
        ll, _ = self._ll_and_deta(np.asarray(value, dtype=float))
        return ll + (self._ll_const if absolute else 0.0)

    def log_prior(self, value: np.ndarray) -> float:
        """Prior log-density of fractions (on the simplex) or rates, where defined."""
        return float(self.prior.log_density(np.asarray(value, dtype=float)))

    def log_posterior(self, value: np.ndarray) -> float:
        """Unnormalised log posterior density in fraction (or rate) space."""
        return self.log_likelihood(value) + self.log_prior(value)

    def expected_counts(self, value: np.ndarray) -> np.ndarray:
        r"""Expected latent counts :math:`\hat n_k = \sum_i w_i r_{ik}`."""
        _, nhat, _ = kern.loglike_and_counts(self._L, self._w, np.asarray(value, float))
        return nhat

    def responsibilities(self, value: np.ndarray, idx: Any = None) -> np.ndarray:
        r"""Per-object posteriors over bins given the population: :math:`r_{ik}`."""
        return kern.responsibilities(self._L, np.asarray(value, dtype=float), idx=idx)

    def sample_prior(self, n: int, rng: RNGLike = None) -> np.ndarray:
        """Draws of fractions (or rates) from the prior."""
        return self.prior.sample(int(n), as_generator(rng))

    def prior_transform(self, u: np.ndarray) -> np.ndarray:
        """Unit-hypercube transform to prior draws (for nested samplers)."""
        return self.prior.prior_transform(np.asarray(u, dtype=float))

    def stack(self) -> np.ndarray:
        """Naive stacking: average of per-object posteriors under a flat interim prior.

        This is one EM step from a population that is uniform in the bin
        coordinate. It is biased and too broad when PDFs are broad; it is
        provided for comparison only.
        """
        return self.likelihoods.stack()

    # -------------------------------------------------------- diagnostics
    def kkt_gap(self, value: np.ndarray, include_prior: bool = True) -> float:
        """Violation of the first-order optimality conditions at ``value``.

        Zero (to numerical precision) at the maximum-likelihood / MAP estimate
        on the simplex (or the positive orthant for rates); normalised by the
        total weight. Self-consistency of the EM fixed point is necessary but
        not sufficient for optimality; this certificate checks the rest.
        """
        v = np.asarray(value, dtype=float)
        s = kern.matvec(self._L, v)
        G = kern.rmatvec(self._L, self._w / s)
        pos = v > 0
        if self.kind == "rates":
            G = G - self._exposure()
        elif self.selection is not None:
            G = G - self.W * self.selection / float(self.selection @ v)
        if include_prior and self.prior.has_density:
            G = G + self.prior.grad_log_density(v)
        if self.kind == "rates":
            viol = np.where(pos, np.abs(G), np.maximum(G, 0.0))
            return float(viol.max() / max(self.W, 1.0))
        mu = float(v[pos] @ G[pos])
        return float(max(0.0, np.max(G - mu)) / max(self.W, 1.0))

    def missing_information(
        self, value: np.ndarray, max_objects: int = 20000, rng: RNGLike = 0
    ) -> float:
        """Largest fraction of missing information at ``value`` (0 to 1).

        Close to 0 when per-object PDFs are narrow compared with the bins and
        close to 1 when they are broad. It predicts the convergence rate of EM
        and the lag-one autocorrelation of the slowest mode of the Gibbs sampler
        (integrated autocorrelation time roughly ``(1 + f) / (1 - f)``).
        Computed on a random subset of at most ``max_objects`` objects.
        """
        v = np.asarray(value, dtype=float)
        N = self.N
        if N > max_objects:
            sel = np.sort(as_generator(rng).choice(N, size=max_objects, replace=False))
            L = self._L[sel]
            w = self._w[sel] * (self.W / self._w[sel].sum())
        else:
            L, w = self._L, self._w
        R = kern.responsibilities(L, v, chunk=None)
        nhat = w @ R
        Im = np.diag(nhat) - (R * w[:, None]).T @ R
        if self.kind == "rates":
            Ic = np.diag(np.maximum(nhat, 1e-300))
            M_ic, M_im = Ic, Im
        else:
            r = v / v.sum()
            Ic = self.W * (np.diag(r) - np.outer(r, r))
            from ._simplex import helmert_basis

            V = helmert_basis(self.K)
            M_ic, M_im = V.T @ Ic @ V, V.T @ Im @ V
        M_ic = 0.5 * (M_ic + M_ic.T) + 1e-12 * np.trace(M_ic) / M_ic.shape[0] * np.eye(len(M_ic))
        M_im = 0.5 * (M_im + M_im.T)
        lam = linalg.eigh(M_im, M_ic, eigvals_only=True)
        return float(np.clip(lam.max(), 0.0, 1.0))
