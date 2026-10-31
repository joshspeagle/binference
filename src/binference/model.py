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

.. math:: \partial\log p(D\mid\rho)/\partial\eta_k = \hat n_k - W\rho_k S_k/(S\cdot\rho)
          \quad\text{or}\quad
          \partial\log p(D\mid\lambda)/\partial\eta_k = \hat n_k - S_k\lambda_k,

with expected counts :math:`\hat n_k = \rho_k \sum_i w_i L_{ik}/(L_i\cdot\rho)`.

A selection on the truth measured by injections is inferred with the population: the
unconstrained coordinates gain :math:`y = \mathrm{logit}\,S` after the prior's, the
likelihood uses that :math:`S`, and :math:`\partial\log p(D\mid\rho)/\partial y_k =
(1 - S_k)\,\partial\log p(D\mid\rho)/\partial\eta_k` (DECISIONS #80). A response
tallied from events (binned measurements) is inferred the same way: the coordinates
gain the isometric log-ratios of each column of :math:`R` with its lost share, under
their Dirichlet posterior given the tallies, and the likelihood is
:math:`\sum_j c_j\log(R_{j\cdot}\cdot\rho) - W\log(\epsilon\cdot\rho)` with
:math:`\epsilon` the column sums (DECISIONS #28, #81).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from scipy import linalg, special

from . import _kernels as kern
from ._simplex import helmert_basis, log_softmax, softmax
from ._utils import RNGLike, as_generator, check_nonnegative, chunk_slices
from .bins import Bins
from .likelihoods import Likelihoods, _scale_columns
from .priors import Dirichlet, PopulationPrior
from .priors.base import _TINY, BoundPrior
from .priors.rates import _gamma_draw
from .selection import Selection, _logit_density, as_selection

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
        Bins for a raw array of likelihoods; otherwise it must equal the likelihoods' bins.

    Notes
    -----
    ``likelihoods`` must be ordinary likelihoods :math:`p(d_i \\mid z)` evaluated at
    the observed data, *not* renormalised over the detectable data space (see
    :doc:`/guides/selection`).
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
                    "selection.fixed() as a fixed selection, or mode='on_truth' for a "
                    "completeness."
                )
            if self.prior.kind == "rates":
                raise ValueError(
                    "A selection from injections is inferred jointly for fractions only; for "
                    "rates pass selection.fixed() or a fixed exposure."
                )
        if sel is not None and sel.scale == "relative" and self.prior.kind == "rates":
            raise ValueError(
                "Rates need an absolute selection (exposures); this one is relative "
                "(DECISIONS #46)."
            )
        resp = likelihoods.meta.get("response")
        #: a response tallied from events, inferred with the population (or ``None``)
        self.response = resp if resp is not None and resp.uncertain else None
        if self.response is not None:
            self._check_response(likelihoods, selection is None and selection_mode is None)
            #: the columns' Dirichlet parameters (J + 1, K)
            self._alpha = self.response._dirichlet()
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

    def _check_response(self, likelihoods: Likelihoods, no_override: bool) -> None:
        # an inferred response needs the rows from_binned built, with its own selection
        if not (no_override and likelihoods.selection == self.response._selection()):
            raise ValueError(
                "The efficiency of a response inferred from its tallies is the selection; "
                "to set another, build the likelihoods from response.fixed()."
            )
        if "measured_bin" not in likelihoods.meta or likelihoods.basis is not None:
            raise ValueError(
                "An inferred response needs the rows Likelihoods.from_binned builds, one per "
                "measured bin (not compressed or moved onto a Basis); or use response.fixed()."
            )
        if self.prior.kind == "rates":
            raise ValueError(
                "A response is inferred jointly for fractions only; for rates build the "
                "likelihoods from response.fixed()."
            )

    def _set_likelihoods(self, likelihoods: Likelihoods) -> None:
        # the data and what is derived from them, in one place
        self.likelihoods = likelihoods
        # effective per-object likelihoods: columns scaled by the selection on the truth
        self._L = self.effective_likelihoods(likelihoods)
        self._w = likelihoods.weights
        self.W = float(self._w.sum())
        self._ll_const = float(np.dot(self._w, likelihoods.log_scale))
        if self.response is not None:  # each row's measured bin
            self._mb = likelihoods.meta["measured_bin"]

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
    def _exposure(self) -> np.ndarray | None:
        """The exposure of a rate model's conjugate updates (``None`` for fractions)."""
        return self._S if self.kind == "rates" else None

    @property
    def joint(self) -> bool:
        """Whether a selection from injections or a tallied response is inferred with the
        population, as extra unconstrained coordinates (DECISIONS #80, #81)."""
        return self.selection_uncertain or self.response is not None

    @property
    def selection_uncertain(self) -> bool:
        """Whether the selection comes from injections and is inferred with the population.

        Methods that take a population value (``log_likelihood``, ``kkt_gap``, ...)
        then use its :attr:`~Selection.values`.
        """
        return self.selection_spec is not None and self.selection_spec.injections is not None

    def draw_total(
        self, value: np.ndarray, rng: np.random.Generator, S: np.ndarray | None = None
    ) -> np.ndarray | None:
        r"""Draws of the total :math:`\Lambda \mid \rho \sim \mathrm{Gamma}(W, S\cdot\rho)`.

        One draw per row of fractions ``value`` ``(..., K)``: its exact posterior under
        :math:`p(\Lambda) \propto 1/\Lambda`. Returns ``None``
        without an exposure.

        Parameters
        ----------
        value : ndarray (..., K)
            Fractions.
        rng : Generator
            Random number generator.
        S : ndarray (..., K), optional
            Selection per row (draws of an uncertain one); ``None`` uses
            :attr:`total_exposure`.
        """
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
        """Dimension of the unconstrained parameter vector: the prior's, plus ``K`` for
        :math:`\\mathrm{logit}\\,S` when the selection is inferred, or the log-ratios of the
        response's columns (``J - 1`` or ``J`` each, ``J`` with a lost share) when the
        response is inferred."""
        extra = self.K if self.selection_uncertain else 0
        if self.response is not None:
            extra = self.K * (self._alpha.shape[0] - 1)
        return self.prior.dim + extra

    def selection_at(self, X: np.ndarray) -> np.ndarray | None:
        """The selection :math:`S` at unconstrained points ``X`` ``(..., dim)``, or ``None``
        when it is fixed.

        Parameters
        ----------
        X : ndarray (..., dim)
            Unconstrained points.
        """
        if not self.selection_uncertain:
            return None
        return special.expit(np.asarray(X, dtype=float)[..., self.prior.dim :])

    def _response_logits(self, X: np.ndarray) -> np.ndarray:
        # log compositions (..., K, J + 1) of the response's columns from their log-ratios
        C = self._alpha.shape[0]  # J measured bins, and the lost share if it is modelled
        Y = np.asarray(X, dtype=float)[..., self.prior.dim :].reshape(*np.shape(X)[:-1], -1, C - 1)
        return log_softmax(Y @ helmert_basis(C).T)

    def response_at(self, X: np.ndarray) -> np.ndarray | None:
        """The response :math:`R` ``(..., J, K)`` at unconstrained points ``X``
        ``(..., dim)``, or ``None`` when it is fixed.

        Parameters
        ----------
        X : ndarray (..., dim)
            Unconstrained points.
        """
        if self.response is None:
            return None
        J = self.response.measured_bins.K
        return np.swapaxes(np.exp(self._response_logits(X))[..., :J], -1, -2)

    def _joint_info(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Draws of what is inferred with the population at points ``X`` (..., dim)."""
        if self.selection_uncertain:
            return {"selection_draws": self.selection_at(X)}
        if self.response is not None:
            return {"response_draws": self.response_at(X)}
        return {}

    def _joint_log_density(self, x: np.ndarray) -> float:
        """Log density at ``x`` of what is inferred with the population, given the
        injections or tallies alone, in its own space (S or R, without the Jacobian of
        its unconstrained coordinates)."""
        y = np.asarray(x, dtype=float)[self.prior.dim :]
        if self.selection_uncertain:
            ls, _ = _logit_density(y, self.selection_spec.injections)
            return ls - float(np.sum(special.log_expit(y) + special.log_expit(-y)))
        if self.response is not None:
            return float(np.sum((self._alpha.T - 1.0) * self._response_logits(x)))
        return 0.0

    def _response_terms(self, value: np.ndarray, R: np.ndarray):
        r"""Log-likelihood of the rows (measured bins) given ``R``, with its gradients in
        the logits of :math:`\rho` and in :math:`R` ``(J, K)``."""
        rows = R[self._mb]
        q = rows @ value
        eff = R.sum(axis=0)
        se = float(eff @ value)
        if not (np.all(q > 0) and se > 0):
            return -np.inf, None, None
        ll = float(self._w @ np.log(q)) - self.W * np.log(se)
        t = self._w / q
        d_eta = value * (rows.T @ t) - self.W * value * eff / se
        G = np.broadcast_to(-self.W * value / se, R.shape).copy()
        np.add.at(G, self._mb, np.outer(t, value))
        return ll, d_eta, G

    def _response_density(self, logP: np.ndarray, G: np.ndarray) -> tuple[float, np.ndarray]:
        # the columns' Dirichlet density in their log-ratios (with the Jacobian) at the log
        # compositions logP (K, J + 1), and the gradient of it plus the likelihood's (G in R;
        # the lost share has none)
        P = np.exp(logP)
        a = self._alpha.T
        Gf = np.zeros_like(P)
        Gf[:, : G.shape[0]] = G.T
        d = P * (Gf - np.sum(P * Gf, axis=1, keepdims=True)) + a - a.sum(axis=1)[:, None] * P
        return float(np.sum(a * logP)), (d @ helmert_basis(P.shape[1])).ravel()

    def effective_likelihoods(self, likelihoods: Likelihoods):
        """Likelihood matrix of (new) objects with this model's selection applied.

        Parameters
        ----------
        likelihoods : Likelihoods
            Likelihoods of the objects, on this model's bins.
        """
        L = likelihoods.values
        return L if self.selection_row is None else _scale_columns(L, self.selection_row)

    def __repr__(self) -> str:
        sel = "" if self.selection is None else f", selection={self.selection_mode!r}"
        return f"PopulationModel(N={self.N}, K={self.K}, prior={self.prior_spec!r}{sel})"

    # ------------------------------------------------- likelihood internals
    def _selection_term(self, value: np.ndarray, S: np.ndarray | None = None):
        # -S.lambda for rates, -W log(S.rho) for fractions with selection; one per row of
        # value (..., K)
        S = self._S if S is None else S
        if self.kind == "rates":
            return -(value @ S)
        if self.selection is None:
            return 0.0
        with np.errstate(divide="ignore"):
            return -self.W * np.log(value @ S)

    def _rows(self, value: np.ndarray, S: np.ndarray | None = None):
        """The likelihood rows and the column weights for ``value``: the model's rows
        (selection on the truth applied) and ``value``, or, given a value ``S`` of an
        inferred selection, the bare rows and :math:`\\rho\\circ S` (DECISIONS #79, #80)."""
        if S is None:
            return self._L, value
        return self.likelihoods.values, value * S

    def _ll_and_deta(
        self, value: np.ndarray, S: np.ndarray | None = None
    ) -> tuple[float, np.ndarray]:
        L, v = self._rows(value, S)
        ll, nhat, _ = kern.loglike_and_counts(L, self._w, v)
        if ll == -np.inf:
            return ll, np.zeros_like(value)
        ll += self._selection_term(value, S)
        S = self._S if S is None else S
        # gradient in the logits (or log-rates) of ll, selection term included
        if self.kind == "rates":
            return ll, nhat - S * value
        if self.selection is None:
            return ll, nhat - self.W * value
        return ll, nhat - self.W * value * S / float(S @ value)

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
        :meth:`log_posterior`. With an inferred selection ``x`` ends with
        :math:`\\mathrm{logit}\\,S`, and with an inferred response with the log-ratios of
        its columns; their densities given the injections or tallies are included.
        NUTS, Laplace and L-BFGS use it. Up to a constant.

        Parameters
        ----------
        x : ndarray (dim,)
            Unconstrained point.
        """
        x = np.asarray(x, dtype=float)
        d = self.prior.dim
        eta, cache = self.prior.forward(x[:d])
        value = self._value(eta)
        if not np.isfinite(value.sum()):
            return -np.inf, np.zeros_like(x)
        S = self.selection_at(x)
        logP = None if self.response is None else self._response_logits(x)
        R = None if logP is None else np.exp(logP[:, : self.response.measured_bins.K]).T
        if R is None:
            ll, d_eta = self._ll_and_deta(value, S)
            ll += self._ll_const
        else:  # absolute already: the response's own rows
            ll, d_eta, G = self._response_terms(value, R)
        if not np.isfinite(ll):
            return -np.inf, np.zeros_like(x)
        lp, g = self.prior.log_prior_and_grad(x[:d], cache, value, d_eta)
        if S is not None:  # the selection's block (DECISIONS #80)
            ls, g_y = _logit_density(x[d:], self.selection_spec.injections)
            lp += ls
            g = np.concatenate([g, g_y + d_eta * (1.0 - S)])
        if R is not None:  # the response's block (DECISIONS #81)
            lr, g_r = self._response_density(logP, G)
            lp += lr
            g = np.concatenate([g, g_r])
        return ll + lp, g

    def log_density_unconstrained(self, x: np.ndarray) -> float:
        """Log posterior density in the unconstrained coordinates ``x`` (with the Jacobian).

        See :meth:`log_density_unconstrained_and_grad`.

        Parameters
        ----------
        x : ndarray (dim,)
            Unconstrained point.
        """
        return self.log_density_unconstrained_and_grad(x)[0]

    def _log_density_many(self, X: np.ndarray) -> np.ndarray:
        # log_density_unconstrained at each row of X (n, dim), without gradients: the
        # prior point by point, the likelihood of many points per pass over the objects
        if self.joint:
            return np.array([self.log_density_unconstrained(x) for x in X])
        V = np.zeros((len(X), self.K))
        lp = np.full(len(X), -np.inf)
        zero = np.zeros(self.K)
        for j, x in enumerate(X):  # the prior, point by point
            eta, cache = self.prior.forward(x)
            value = self._value(eta)
            if np.isfinite(value.sum()):  # else a zero row: an impossible point
                V[j] = value
                lp[j] = self.prior.log_prior_and_grad(x, cache, value, zero)[0]
        ll = self._log_likelihood_many(V) + self._ll_const
        return np.where(np.isfinite(ll), ll + lp, -np.inf)

    def _log_likelihood_many(self, V: np.ndarray) -> np.ndarray:
        # log_likelihood(v, absolute=False) at each row of V (n, K), many per pass; for a
        # joint model, as there, at the selection's values or the response's posterior means
        return kern.loglike_many(self._L, self._w, V) + self._selection_term(V)

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
        new._set_likelihoods(lk._copy(weights=w[keep]))
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
        if (self.selection is None and S is None) or self.kind == "rates":
            return counts
        S = self._detect if S is None else S
        rho = value / value.sum()
        n_missed = rng.negative_binomial(self.W, float(np.clip(S @ rho, _TINY, 1.0)))
        if n_missed == 0:
            return counts
        q = rho * np.maximum(1.0 - S, 0.0)  # an efficiency may round to above one
        return counts + rng.multinomial(n_missed, q / q.sum())

    def to_value(self, x: np.ndarray) -> np.ndarray:
        """Fractions (or rates, for rate priors) at unconstrained ``x``.

        Parameters
        ----------
        x : ndarray (dim,)
            Unconstrained point.
        """
        eta, _ = self.prior.forward(np.asarray(x, dtype=float)[: self.prior.dim])
        return self._value(eta)

    def to_rho(self, x: np.ndarray) -> np.ndarray:
        """Population fractions at unconstrained ``x``.

        Parameters
        ----------
        x : ndarray (dim,)
            Unconstrained point.
        """
        v = self.to_value(x)
        return v / v.sum() if self.kind == "rates" else v

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Map many unconstrained draws ``(n, dim)`` to values ``(n, K)``.

        Parameters
        ----------
        X : ndarray (n, dim)
            Unconstrained draws.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.array([self.to_value(x) for x in X])

    def from_value(self, value: np.ndarray) -> np.ndarray:
        """An unconstrained point corresponding to fractions (or rates) ``value`` (and to
        the selection's :attr:`~Selection.values`, or the response's posterior means, when
        they are inferred).

        Parameters
        ----------
        value : ndarray (K,)
            Fractions or rates.
        """
        x = self.prior.init_from_value(np.asarray(value, dtype=float))
        if self.selection_uncertain:
            return np.concatenate([x, special.logit(self.selection)])
        if self.response is not None:  # the posterior means of the columns
            P = self.response._mean_columns().T
            return np.concatenate([x, (np.log(P) @ helmert_basis(P.shape[1])).ravel()])
        return x

    # ----------------------------------------------- fraction/rate space
    def log_likelihood(
        self,
        value: np.ndarray,
        absolute: bool = True,
        S: np.ndarray | None = None,
        response: np.ndarray | None = None,
    ) -> float:
        """Marginal log-likelihood of the population (including the selection term).

        Parameters
        ----------
        value : ndarray (K,)
            Fractions or rates.
        absolute : bool, optional
            Include the per-object log-scales of the likelihoods, so values compare
            across binnings and models.
        S : ndarray (K,), optional
            Value of an inferred selection; ``None`` uses its
            :attr:`~Selection.values`.
        response : ndarray (J, K), optional
            Value of an inferred response; ``None`` uses the rows of the likelihoods
            (its posterior means).
        """
        v = self._population(value)
        if not v.any():
            raise ValueError("value must not be all zero.")
        if response is not None:
            if self.response is None:
                raise ValueError("response applies only to a response inferred from tallies.")
            R = check_nonnegative(response, "response")
            if R.shape != self.response.matrix.shape or np.any(R.sum(axis=0) > 1.0 + 1e-6):
                raise ValueError(
                    f"response must have shape {self.response.matrix.shape}, non-negative "
                    "columns that sum to at most one."
                )
            ll = self._response_terms(v, R)[0]
            return ll if absolute else ll - self._ll_const
        if S is not None:
            if not self.selection_uncertain:
                raise ValueError("S applies only to a selection inferred from injections.")
            S = np.asarray(S, dtype=float)
            if S.shape != (self.K,) or not np.all((S > 0) & (S <= 1)):
                raise ValueError(f"S must have shape ({self.K},) with values in (0, 1].")
        L, u = self._rows(v, S)
        ll = kern.loglike(L, self._w, u)
        return ll + self._selection_term(v, S) + (self._ll_const if absolute else 0.0)

    def log_prior(self, value: np.ndarray) -> float:
        """Prior log-density of fractions (on the simplex) or rates, where defined.

        A density in the fractions (or rates) themselves, without the Jacobian of
        any reparametrisation (DECISIONS #15). On the boundary it is -inf where
        the density is zero, +inf where it is unbounded and NaN where it is
        undefined (a Polya tree); NaN for priors without a closed-form density
        (logistic priors).

        Parameters
        ----------
        value : ndarray (K,)
            Fractions or rates.
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

        Parameters
        ----------
        value : ndarray (K,)
            Fractions or rates.
        """
        return self.log_likelihood(value) + self.log_prior(value)

    def expected_counts(self, value: np.ndarray) -> np.ndarray:
        r"""Expected latent counts :math:`\hat n_k = \sum_i w_i r_{ik}`.

        Raises where an object has zero likelihood, whose responsibilities are undefined.

        Parameters
        ----------
        value : ndarray (K,)
            Fractions or rates.
        """
        ll, nhat, s = kern.loglike_and_counts(self._L, self._w, self._population(value))
        if ll == -np.inf:  # an impossible object, or a sum that overflows
            bad = np.flatnonzero(~(s > kern.MIN_EVIDENCE))
            why = f"gives {bad.size} object(s) zero likelihood (first: {bad[:5].tolist()})"
            raise ValueError(
                f"value {why if bad.size else 'overflows the likelihood'}, so the expected "
                "counts are undefined."
            )
        return nhat

    def _population(self, value: Any) -> np.ndarray:
        """``value`` checked: finite, non-negative fractions or rates of shape ``(K,)``."""
        v = check_nonnegative(value, "value")
        if v.shape != (self.K,):
            raise ValueError(f"value must have shape ({self.K},); got {v.shape}.")
        return v

    def responsibilities(self, value: np.ndarray, idx: Any = None) -> np.ndarray:
        r"""Per-object posteriors over bins given the population: :math:`r_{ik}`.

        Parameters
        ----------
        value : ndarray (K,)
            Fractions or rates.
        idx : index, optional
            Objects to use; ``None`` uses all of them.
        """
        return kern.responsibilities(self._L, np.asarray(value, dtype=float), idx=idx)

    def sample_prior(self, n: int, rng: RNGLike = None) -> np.ndarray:
        """Draws of fractions (or rates) from the prior.

        Parameters
        ----------
        n : int
            Number of draws.
        rng : int, numpy.random.Generator, or None
            Random number generator.
        """
        return self.prior.sample(int(n), as_generator(rng))

    # -------------------------------------------------------- diagnostics
    def kkt_gap(self, value: np.ndarray, include_prior: bool = True) -> float:
        r"""Violation of the first-order optimality conditions at ``value``.

        Zero (to numerical precision) at the maximum-likelihood / MAP estimate
        on the simplex (or the positive orthant for rates), and infinite where
        an object has zero likelihood. For fractions it is normalised by the
        total weight; for rates it is the larger of
        :math:`\max_k S_k^{-1}\,\partial\log p/\partial\lambda_k` and
        :math:`|\lambda\cdot\nabla_\lambda\log p|/(S\cdot\lambda)`, so it does not
        depend on the exposure's units (DECISIONS #56, #71). Self-consistency of the
        EM fixed point is necessary but not sufficient for optimality; this
        certificate checks the rest. With a tree prior the conditions are checked on each
        node's branch probabilities, where they keep their precision as a group of bins
        empties.

        Parameters
        ----------
        value : ndarray (K,)
            Fractions or rates.
        include_prior : bool, optional
            Add the gradient of the prior's log density, for priors with a closed-form
            density (none for the logistic priors).
        """
        v = np.asarray(value, dtype=float)
        s = kern.matvec(self._L, v)
        if not kern.possible(s):
            return np.inf  # an object is impossible here (audit M-3)
        G = kern.rmatvec(self._L, self._w / s)
        if self.kind == "fractions":
            if self.selection is not None:
                G = G - self.W * self.selection / float(self.selection @ v)
            return max(0.0, self.prior._kkt_violation(v, G, self.W, include_prior))
        G = self.prior._kkt_gradient(v, G - self._S, include_prior)
        pos = v > 0
        # G <= 0 everywhere and lambda . G = 0 (the total), relative to the exposure so
        # free of its units (audit M-4); one-sided like the fractions' test, so a rate
        # EM drives towards zero is not a violation (audit M-6)
        total_gap = abs(float(v[pos] @ G[pos])) / float(self._S @ v)
        return max(float(np.max(G / self._S)), total_gap)

    def missing_information(
        self,
        value: np.ndarray,
        max_objects: int = 20000,
        rng: RNGLike = 0,
        include_prior: bool = False,
        chunk: int | None = kern.DEFAULT_CHUNK,
    ) -> float:
        """Largest fraction of missing information at ``value`` (0 to 1).

        Close to 0 when per-object PDFs are narrow compared with the bins and
        close to 1 when they are broad. It predicts the convergence rate of EM
        and the lag-one autocorrelation of the slowest mode of the Gibbs sampler
        (integrated autocorrelation time roughly ``(1 + f) / (1 - f)``). With a
        selection function the missed objects count as missing data, as EM and
        Gibbs impute them. Computed on the support of ``value`` and on a random
        subset of at most ``max_objects`` objects, ``chunk`` objects at a time.

        Parameters
        ----------
        value : ndarray (K,)
            Fractions or rates.
        max_objects : int, optional
            Largest random subset of objects used.
        rng : int, Generator or None, optional
            Random number generator for the subset; the default ``0`` makes it
            reproducible.
        include_prior : bool, optional
            Add the pseudo-counts of a Dirichlet or ``GammaRates`` prior, as the MAP EM
            does (audit M-2); other priors add none. NaN where concentrations or shapes
            below one outweigh the data.
        chunk : int, optional
            Objects processed per block, which bounds memory.
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
        nhat, RwR = np.zeros(on.sum()), np.zeros((on.sum(), on.sum()))
        for sl in chunk_slices(L.shape[0], chunk):
            R = kern.responsibilities(L[sl], v, chunk=None)[:, on]
            nhat += w[sl] @ R
            RwR += (R * w[sl, None]).T @ R
        Im = np.diag(nhat) - RwR
        # the prior's pseudo-counts join the complete data, as in the MAP EM
        a = self.prior.pseudocounts if include_prior else None
        a = 0.0 if a is None else a[on]
        if self.kind == "rates":  # Poisson thinning imputes nothing
            n_complete = nhat + a
            if np.any(n_complete < 0):  # shapes below one outweigh the data
                return np.nan
            I_c = np.diag(n_complete)
            I_obs = I_c - Im
        else:
            rho = v[on] / v[on].sum()
            C = np.diag(rho) - np.outer(rho, rho)
            extra = float(np.sum(a))  # the total pseudo-count (A - K for a Dirichlet)
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
