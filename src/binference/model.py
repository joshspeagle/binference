r"""The hierarchical population model.

For fractions :math:`\rho` (simplex priors and logistic priors):

.. math:: \log p(\rho \mid D) = \log p(\rho) + \sum_i w_i \log(L_i\cdot\rho)
          - W \log(S\cdot\rho) + \text{const},

where the selection term is present only when a selection function
:math:`S` is given. For rates :math:`\lambda` (:class:`~binference.priors.GammaRates`):

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

import copy
from typing import Any

import numpy as np
from scipy import linalg

from . import _kernels as kern
from ._simplex import helmert_basis, softmax
from ._utils import RNGLike, as_generator
from .bins import Bins
from .likelihoods import Likelihoods, _scale_columns
from .priors import Dirichlet, PopulationPrior
from .priors.base import BoundPrior
from .priors.dirichlet import _BoundDirichlet
from .priors.rates import _gamma_draw
from .selection import Selection, as_selection

__all__ = ["PopulationModel"]


class PopulationModel:
    """Likelihoods + prior (+ selection): the object every inference engine works on.

    Parameters
    ----------
    likelihoods : Likelihoods or array_like (N, K)
        Per-object bin-averaged likelihoods. A raw array is wrapped with
        ``Likelihoods(array, bins)``.
    prior : PopulationPrior, optional
        Population prior; defaults to ``Dirichlet(1.0)``. Use
        :class:`~binference.priors.GammaRates` to infer rates instead of fractions.
    selection : array_like (K,), callable or Selection, optional
        Per-bin detection probability (fractions) or exposure (rates), or a
        function of the true value averaged over each bin. Overrides any
        selection attached to ``likelihoods``.
    selection_mode : {"on_measurement", "on_truth"}, optional
        Required with an array or callable ``selection``. ``"on_measurement"``:
        objects are selected on their measured data (or the likelihoods already
        include the detection probability); only the normalisation term is
        added. ``"on_truth"``: objects are detected with probability ``S_k``
        depending on their true bin; likelihood rows are multiplied by ``S_k``
        as well. :meth:`Selection.combined` gives both at once.
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
        prior: PopulationPrior | None = None,
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
        if not isinstance(self.prior_spec, PopulationPrior):
            raise TypeError("prior must be a binference.priors.PopulationPrior instance.")
        self.prior: BoundPrior = self.prior_spec.bind(likelihoods.bins)
        if selection is None and selection_mode is None:
            sel = likelihoods.selection
        else:
            sel = as_selection(selection, selection_mode, likelihoods.basis or likelihoods.bins)
        if sel is not None and sel.injections is not None:
            if sel.mode != "on_truth":
                raise ValueError(
                    "Only a selection on the truth is inferred from injections: on the "
                    "measurement the likelihoods' cut already fixes S, and with S free the "
                    "catalogue would add only missed objects, driving S to zero. Pass "
                    "selection.at_mean() as a fixed selection, or mode='on_truth' for a "
                    "completeness."
                )
            if self.prior.kind == "rates":
                raise ValueError(
                    "A selection from injections is inferred jointly for fractions only; for "
                    "rates pass selection.at_mean() or a fixed exposure."
                )
        if sel is not None and sel.scale == "relative" and self.prior.kind == "rates":
            raise ValueError(
                "Rates need an absolute selection (exposures); this one is relative "
                "(DECISIONS #46)."
            )
        self.selection_spec: Selection | None = sel
        #: per-bin selection values (or ``None``)
        self.selection = None if sel is None else sel.values
        self.selection_mode = None if sel is None else sel.mode
        #: factor on the likelihood columns (selection on the truth), or ``None``
        self.selection_row = None if sel is None else sel.row
        # For fractions only ratios of S matter. Data-augmentation steps (Gibbs, EM,
        # VB) impute missed objects with probability 1 - S, so they use S rescaled
        # to a maximum of one, which leaves the posterior unchanged.
        self._detect = None if sel is None else sel.values / sel.values.max()
        self._S = np.ones(self.K) if sel is None else sel.values  # exposure, for rates
        self._set_likelihoods(likelihoods)

    def _set_likelihoods(self, likelihoods: Likelihoods) -> None:
        # the data and what is derived from them, in one place
        self.likelihoods = likelihoods
        # effective per-object likelihoods: columns scaled by the selection on the truth
        self._L = self.effective_likelihoods(likelihoods)
        self._w = likelihoods.weights
        self.W = float(self._w.sum())
        self._ll_const = float(np.dot(self._w, likelihoods.log_scale))

    # ----------------------------------------------------------- properties
    @property
    def total_exposure(self) -> np.ndarray | None:
        r""":math:`S` of the Poisson law of the number of objects, :math:`W \sim
        \mathrm{Poisson}(\Lambda\,S\cdot\rho)`: detection probabilities or exposures
        (ones without a selection); ``None`` for a relative selection, which
        leaves the total unidentified (DECISIONS #46)."""
        sel = self.selection_spec
        return None if sel is not None and sel.scale == "relative" else self._S

    @property
    def selection_uncertain(self) -> bool:
        """Whether the selection comes from injections and is inferred with the population.

        Methods that take a population value (``log_likelihood``, ``kkt_gap``, ...)
        then use the selection's posterior means.
        """
        return self.selection_spec is not None and self.selection_spec.injections is not None

    def draw_total(
        self, value: np.ndarray, rng: np.random.Generator, S: np.ndarray | None = None
    ) -> np.ndarray | None:
        r"""Draws of the total :math:`\Lambda \mid \rho \sim \mathrm{Gamma}(W, S\cdot\rho)`,
        one per row of fractions ``value`` ``(..., K)``: its exact posterior under
        :math:`p(\Lambda) \propto 1/\Lambda` (DECISIONS #9, #78). ``S`` gives a
        selection per row (draws of an uncertain one); by default
        :attr:`total_exposure`, and ``None`` without it."""
        S = self.total_exposure if S is None else np.asarray(S)
        if S is None:
            return None
        return _gamma_draw(self.W, np.sum(np.asarray(value) * S, axis=-1), rng)

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
        return L if self.selection_row is None else _scale_columns(L, self.selection_row)

    def __repr__(self) -> str:
        sel = "" if self.selection is None else f", selection={self.selection_mode!r}"
        return f"PopulationModel(N={self.N}, K={self.K}, prior={self.prior_spec!r}{sel})"

    # ------------------------------------------------- likelihood internals
    def _selection_term(self, value: np.ndarray) -> float:
        # -S.lambda for rates, -W log(S.rho) for fractions with selection
        if self.kind == "rates":
            return -float(self._S @ value)
        return 0.0 if self.selection is None else -self.W * np.log(float(self._S @ value))

    def _ll_and_deta(self, value: np.ndarray) -> tuple[float, np.ndarray]:
        ll, nhat, _ = kern.loglike_and_counts(self._L, self._w, value)
        if not np.isfinite(ll):
            return -np.inf, np.zeros_like(value)
        ll += self._selection_term(value)
        # gradient in the logits (or log-rates) of ll, selection term included
        if self.kind == "rates":
            return ll, nhat - self._S * value
        if self.selection is None:
            return ll, nhat - self.W * value
        return ll, nhat - self.W * value * self._S / float(self._S @ value)

    def _value(self, eta: np.ndarray) -> np.ndarray:
        if self.kind == "rates":
            with np.errstate(over="ignore"):
                return np.exp(eta)
        return softmax(eta)

    # ------------------------------------------------ unconstrained space
    def log_density_unconstrained_and_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        """Log posterior density and its gradient in the unconstrained coordinates ``x``.

        This is the density of ``x`` (the prior's isometric log-ratios, whitened
        fields or log-rates), so it includes the Jacobian of the map from ``x``
        to the fractions or rates (DECISIONS #15); its mode differs from that of
        :meth:`log_posterior`. NUTS, Laplace and L-BFGS use it. Up to a constant.
        """
        x = np.asarray(x, dtype=float)
        eta, cache = self.prior.forward(x)
        value = self._value(eta)
        if not np.isfinite(value.sum()):
            return -np.inf, np.zeros_like(x)
        ll, d_eta = self._ll_and_deta(value)
        if not np.isfinite(ll):
            return -np.inf, np.zeros_like(x)
        lp, g = self.prior.log_prior_and_grad(x, cache, value, d_eta)
        return ll + lp + self._ll_const, g

    def log_density_unconstrained(self, x: np.ndarray) -> float:
        """Log posterior density in the unconstrained coordinates ``x`` (with the Jacobian).

        See :meth:`log_density_unconstrained_and_grad`.
        """
        return self.log_density_unconstrained_and_grad(x)[0]

    def _with_bound_prior(self, prior: BoundPrior) -> PopulationModel:
        """A shallow copy sharing the data but using another parametrisation of the prior."""
        new = copy.copy(self)
        new.prior = prior
        return new

    def _with_weights(self, weights: np.ndarray) -> PopulationModel:
        """A shallow copy with other object weights, keeping only the rows they leave positive.

        Rows of zero weight are dropped, so a bin they alone support cannot make
        the likelihood -inf (audit M-1); the one way engines override weights
        (bootstrap replicates).
        """
        w = np.asarray(weights, dtype=float)
        keep = w > 0
        lk = self.likelihoods if keep.all() else self.likelihoods.subset(np.flatnonzero(keep))
        new = copy.copy(self)
        new._set_likelihoods(lk._reweighted(w[keep]))
        return new

    def _complete_counts(self, value: np.ndarray, nhat: np.ndarray) -> np.ndarray:
        r"""Expected counts with the missed objects imputed (fractions with selection).

        Adds :math:`W\rho_k(1 - S_k)/(S\cdot\rho)`, with :math:`S` rescaled to a
        maximum of one, which leaves the posterior unchanged; used by EM, VB and
        the interior start (DECISIONS #57).
        """
        if self.selection is None or self.kind == "rates":  # Poisson thinning imputes nothing
            return nhat
        S = self._detect
        rho = value / value.sum()
        return nhat + self.W * rho * (1.0 - S) / float(S @ rho)

    def _draw_missed(
        self,
        value: np.ndarray,
        counts: np.ndarray,
        rng: np.random.Generator,
        S: np.ndarray | None = None,
    ) -> np.ndarray:
        r"""Counts with the missed objects drawn given the fractions (fractions with selection).

        Their number is :math:`\mathrm{NegBin}(W, S\cdot\rho)` (failures before
        :math:`W` successes) and their bins :math:`\mathrm{Mult}(\rho\circ(1-S))`, with
        :math:`S` rescaled to a maximum of one (or ``S`` itself, absolute, when an
        uncertain selection is being drawn); Gibbs's counterpart of
        :meth:`_complete_counts` (audit G-7).
        """
        if self.selection is None or self.kind == "rates":
            return counts
        S = self._detect if S is None else S
        rho = value / value.sum()
        n_missed = rng.negative_binomial(self.W, float(np.clip(S @ rho, 1e-300, 1.0)))
        if n_missed == 0:
            return counts
        q = rho * (1.0 - S)
        return counts + rng.multinomial(n_missed, q / q.sum())

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
        v = np.asarray(value, dtype=float)
        ll = kern.loglike(self._L, self._w, v)
        if not np.isfinite(ll):
            return -np.inf
        return ll + self._selection_term(v) + (self._ll_const if absolute else 0.0)

    def log_prior(self, value: np.ndarray) -> float:
        """Prior log-density of fractions (on the simplex) or rates, where defined.

        A density in the fractions (or rates) themselves, without the Jacobian of
        any reparametrisation (DECISIONS #15). On the boundary it is -inf where
        the density is zero, +inf where it is unbounded and NaN where it is
        undefined (a Polya tree); NaN for priors without a closed-form density
        (logistic priors).
        """
        if not self.prior.has_density:
            return np.nan
        with np.errstate(invalid="ignore"):
            return float(self.prior.log_density(np.asarray(value, dtype=float)))

    def log_posterior(self, value: np.ndarray) -> float:
        """Unnormalised log posterior density of fractions (on the simplex) or rates.

        A density in the fractions or rates themselves, without the Jacobian of
        the unconstrained coordinates, so its mode is the MAP estimate on the
        simplex (``MAP(space="simplex")``); compare
        :meth:`log_density_unconstrained` (DECISIONS #15).
        """
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
        """Naive stacking: the mean of per-object posteriors; see :meth:`Likelihoods.stack`.

        The posteriors use the interim prior the likelihoods were built with
        (flat if none). This is one EM step from that prior; it is biased and
        too broad when PDFs are broad, and is provided for comparison only.
        """
        return self.likelihoods.stack()

    # -------------------------------------------------------- diagnostics
    def kkt_gap(self, value: np.ndarray, include_prior: bool = True) -> float:
        r"""Violation of the first-order optimality conditions at ``value``.

        Zero (to numerical precision) at the maximum-likelihood / MAP estimate
        on the simplex (or the positive orthant for rates), and infinite where
        an object has zero likelihood. For fractions it is normalised by the
        total weight; for rates it is the larger of :math:`\max_k G_k/S_k` and
        :math:`|\lambda\cdot G|/(S\cdot\lambda)` for the gradient :math:`G`
        in :math:`\lambda`, so it does not depend on the exposure's units
        (DECISIONS #56, #71). Self-consistency of the
        EM fixed point is necessary but not sufficient for optimality; this
        certificate checks the rest.
        """
        v = np.asarray(value, dtype=float)
        s = kern.matvec(self._L, v)
        if not kern.possible(s):
            return np.inf  # an object is impossible here (audit M-3)
        G = kern.rmatvec(self._L, self._w / s)
        pos = v > 0
        if self.kind == "rates":
            G = G - self._S
        elif self.selection is not None:
            G = G - self.W * self.selection / float(self.selection @ v)
        if include_prior and self.prior.has_density:
            # at a zero, the gradient at 1e-300 (as in the MAP objective): -b for rates of
            # shape one, hugely negative where the density is unbounded there (no violation),
            # hugely positive where it vanishes (DECISIONS #71)
            G = G + self.prior.grad_log_density(np.maximum(v, 1e-300))
        mu = float(v[pos] @ G[pos])
        if self.kind == "rates":
            # G <= 0 everywhere and lambda . G = 0 (the total), relative to the exposure so
            # free of its units (audit M-4); one-sided like the fractions' test, so a rate
            # EM drives towards zero is not a violation (audit M-6)
            total_gap = abs(mu) / float(self._S @ v)
            return max(float(np.max(G / self._S)), total_gap)
        return max(0.0, float(np.max(G - mu))) / max(self.W, 1.0)

    def missing_information(
        self,
        value: np.ndarray,
        max_objects: int = 20000,
        rng: RNGLike = 0,
        include_prior: bool = False,
    ) -> float:
        """Largest fraction of missing information at ``value`` (0 to 1).

        Close to 0 when per-object PDFs are narrow compared with the bins and
        close to 1 when they are broad. It predicts the convergence rate of EM
        and the lag-one autocorrelation of the slowest mode of the Gibbs sampler
        (integrated autocorrelation time roughly ``(1 + f) / (1 - f)``). With a
        selection function the missed objects count as missing data, as EM and
        Gibbs impute them; ``include_prior`` adds a Dirichlet prior's term, as in
        the MAP EM (audit M-2; other priors add none; NaN where a prior with
        concentrations below one outweighs the data). Computed on the support
        of ``value`` and on a random subset of at most ``max_objects`` objects.
        """
        v = np.asarray(value, dtype=float)
        N = self.N
        if N > max_objects:
            idx = np.sort(as_generator(rng).choice(N, size=max_objects, replace=False))
            L = self._L[idx]
            w = self._w[idx] * (self.W / self._w[idx].sum())
        else:
            L, w = self._L, self._w
        # on the support: a value at zero is an active constraint, along which the
        # information is degenerate rather than missing
        on = v > 1e-9 * v.max()
        if on.sum() < 2:
            return 0.0
        R = kern.responsibilities(L, v, chunk=None)[:, on]
        nhat = w @ R
        Im = np.diag(nhat) - (R * w[:, None]).T @ R
        if self.kind == "rates":  # Poisson thinning imputes nothing
            I_c = np.diag(nhat)
            I_obs = I_c - Im
        else:
            rho = v[on] / v[on].sum()
            C = np.diag(rho) - np.outer(rho, rho)
            # a Dirichlet prior adds (A - K) to the complete-data count, as in the MAP EM
            extra = (
                float(np.sum(self.prior.alpha[on]) - on.sum())
                if include_prior and isinstance(self.prior, _BoundDirichlet)
                else 0.0
            )
            # complete data: the observed objects and, with selection, the imputed missed ones
            S = np.ones(on.sum()) if self.selection is None else self._detect[on]
            phi = S * rho / float(S @ rho)
            n_complete = self.W / float(S @ rho) + extra
            if n_complete <= 0:  # a prior with concentrations below one outweighs the data
                return np.nan
            I_c = n_complete * C
            I_obs = self.W * (np.diag(phi) - np.outer(phi, phi)) - Im + extra * C
            V = helmert_basis(rho.size)
            I_c, I_obs = V.T @ I_c @ V, V.T @ I_obs @ V
        I_c = 0.5 * (I_c + I_c.T) + 1e-12 * np.trace(I_c) / len(I_c) * np.eye(len(I_c))
        I_obs = 0.5 * (I_obs + I_obs.T)
        lam = linalg.eigh(I_obs, I_c, eigvals_only=True)
        return float(np.clip(1.0 - lam.min(), 0.0, 1.0))
