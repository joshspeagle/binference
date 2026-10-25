"""Validation tools: exact reference posteriors, simulation-based calibration, coverage.

* :func:`exact_posterior` computes the posterior on a fine grid over the
  simplex for ``K = 2`` or ``3`` bins (a reference for testing engines).
* :func:`sbc` runs simulation-based calibration (Talts et al. 2018): if the
  inference is calibrated, the rank of the true value among posterior draws is
  uniform. Ranks are computed for every bin fraction and for the joint
  log-likelihood (Modrak et al. 2023).
* :func:`coverage` measures the empirical coverage of credible intervals for
  bin fractions and user-defined functionals.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import special, stats

from ._quadrature import MAX_ELEMENTS
from ._utils import RNGLike, as_generator, chunk_slices, progress, spawn, warn
from .likelihoods import Likelihoods
from .model import PopulationModel
from .priors import PopulationPrior
from .results import PopulationResult
from .warnings import ConvergenceWarning

__all__ = ["CoverageResult", "ExactPosterior", "SBCResult", "coverage", "exact_posterior", "sbc"]


# ----------------------------------------------------------------------------
# exact posterior by quadrature
# ----------------------------------------------------------------------------
@dataclass
class ExactPosterior:
    """Posterior on quadrature nodes over the simplex (``K <= 3``).

    Attributes
    ----------
    points : (n, K) array
        Quadrature nodes (fractions).
    weights : (n,) array
        Normalised posterior weights (quadrature weight times density), summing to one.
    log_density : (n,) array
        Unnormalised log posterior density of the fractions at each node (its
        maximum locates the mode; the weights also carry the node spacing).
    """

    points: np.ndarray
    weights: np.ndarray
    log_density: np.ndarray

    @property
    def mean(self) -> np.ndarray:
        return self.weights @ self.points

    def cov(self) -> np.ndarray:
        d = self.points - self.mean
        return (d * self.weights[:, None]).T @ d

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov()))

    def quantile(self, q: Any, k: int) -> np.ndarray:
        """Quantiles of the marginal posterior of fraction ``k``."""
        x = self.points[:, k]
        o = np.argsort(x)
        c = np.cumsum(self.weights[o])
        return np.interp(q, c - 0.5 * self.weights[o], x[o])


def exact_posterior(
    model: PopulationModel | Likelihoods | Any,
    prior: PopulationPrior | None = None,
    n_grid: int = 200,
    max_elements: int = MAX_ELEMENTS,
) -> ExactPosterior:
    r"""Quadrature posterior of the fractions for ``K = 2`` or ``K = 3`` bins.

    Works for priors with a closed-form simplex density (Dirichlet family).
    Selection effects are included. The fractions are stick-broken,
    :math:`\rho = (x, (1-x)y, (1-x)(1-y))`, each stick coordinate substituted
    as :math:`(1-\cos\pi s)/2` and integrated by Gauss-Legendre with
    ``n_grid`` nodes, which places nodes densely at the boundary and handles
    the integrable singularities of concentrations below one (audit G-3).
    With the default ``n_grid``, means of a known Dirichlet posterior are
    exact to 0.01 posterior SD for concentrations down to 0.2, and tail
    probabilities to 0.005 for concentrations of 0.5 and above (at 0.2 the
    mass below a 6e-9 quantile is 22% off). ``max_elements`` bounds the size of
    the (objects x nodes) blocks evaluated at once (audit G-5).
    """
    if not isinstance(model, PopulationModel):
        model = PopulationModel(model, prior)
    if model.kind != "fractions" or not model.prior.has_density:
        raise ValueError(
            "exact_posterior needs a fractions model with a closed-form prior density."
        )
    if model.K not in (2, 3):
        raise ValueError("exact_posterior supports K = 2 or 3 bins.")
    if model.selection_uncertain:
        raise ValueError(
            "exact_posterior integrates the population only; a selection from injections "
            "would be fixed at its means. Pass selection.at_mean() to compare with that."
        )
    s, ws = np.polynomial.legendre.leggauss(int(n_grid))
    s, ws = 0.5 * (s + 1.0), 0.5 * ws
    x = 0.5 * (1.0 - np.cos(np.pi * s))
    log_dx = np.log(0.5 * np.pi * np.sin(np.pi * s) * ws)  # quadrature weight in x
    if model.K == 2:
        pts = np.stack([x, 1.0 - x], axis=1)
        log_q = log_dx
    else:
        X, Y = np.meshgrid(x, x, indexing="ij")
        pts = np.stack([X.ravel(), ((1 - X) * Y).ravel(), ((1 - X) * (1 - Y)).ravel()], axis=1)
        # d(rho_1, rho_2) = (1 - x) dx dy
        log_q = (log_dx[:, None] + log_dx[None, :] + np.log1p(-X)).ravel()
    L, w = model._L, model._w
    logp = np.empty(pts.shape[0])
    for sl in chunk_slices(pts.shape[0], max(1, int(max_elements) // max(model.N, 1))):
        P = pts[sl]
        with np.errstate(divide="ignore"):
            ll = w @ np.log(np.asarray(L @ P.T))
        if model.selection is not None:
            ll = ll - model.W * np.log(P @ model.selection)
        logp[sl] = ll
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = logp + model.prior.log_density(pts)
    logp = np.where(np.isfinite(logp), logp, -np.inf)
    logw = logp + log_q
    wts = np.exp(logw - special.logsumexp(logw))
    return ExactPosterior(points=pts, weights=wts, log_density=logp)


# ----------------------------------------------------------------------------
# SBC
# ----------------------------------------------------------------------------
Simulator = Callable[[np.random.Generator], tuple[Likelihoods, np.ndarray]]
Fitter = Callable[[Likelihoods, np.random.Generator], PopulationResult]


@dataclass
class SBCResult:
    """Ranks from simulation-based calibration.

    Attributes
    ----------
    ranks : (n_simulations, n_quantities) int array
        Number of posterior draws below the true value.
    n_draws : int
        Draws per simulation used for ranking (ranks lie in ``0..n_draws``).
    names : list of str
        Quantity names (``"rho[k]"``, ``"lambda[k]"`` for rates or ``"w[k]"`` for the
        component weights of a :class:`~binference.Basis`, and ``"loglike"``).
    min_ess : (n_simulations,) array
        Smallest bulk effective sample size of each fit (NaN without chains);
        :func:`sbc` warns when it falls below ``n_draws``.
    """

    ranks: np.ndarray
    n_draws: int
    names: list[str]
    min_ess: np.ndarray = field(default_factory=lambda: np.empty(0))

    def _edges(self, n_bins: int | None) -> np.ndarray:
        n_bins = n_bins or self._default_bins()
        if not 1 <= n_bins <= self.n_draws + 1:
            raise ValueError(f"n_bins must be between 1 and n_draws + 1 = {self.n_draws + 1}.")
        return np.linspace(0, self.n_draws + 1, n_bins + 1)

    def histogram(self, n_bins: int | None = None) -> np.ndarray:
        """Rank histograms, shape ``(n_quantities, n_bins)``."""
        edges = self._edges(n_bins)
        return np.stack([np.histogram(r, bins=edges)[0] for r in self.ranks.T])

    def expected(self, n_bins: int | None = None) -> np.ndarray:
        """Expected count per histogram bin under uniform ranks, ``(n_bins,)``.

        Bins hold different numbers of the ``n_draws + 1`` possible ranks unless
        ``n_bins`` divides ``n_draws + 1`` (audit G-1).
        """
        per_bin = np.histogram(np.arange(self.n_draws + 1), bins=self._edges(n_bins))[0]
        return per_bin * self.ranks.shape[0] / (self.n_draws + 1)

    def _default_bins(self) -> int:
        n = self.ranks.shape[0]
        return int(max(2, min(20, n // 5, self.n_draws + 1)))

    def pvalues(self, n_bins: int | None = None) -> np.ndarray:
        """Chi-square p-values of rank uniformity per quantity."""
        H = self.histogram(n_bins)
        e = self.expected(n_bins)
        return np.array([stats.chisquare(h, f_exp=e).pvalue for h in H])

    def passed(self, alpha: float = 0.01, n_bins: int | None = None) -> bool:
        """True if no quantity rejects uniformity at Bonferroni-corrected level ``alpha``."""
        p = self.pvalues(n_bins)
        return bool(np.all(p > alpha / p.size))

    def plot(self, **kwargs: Any):
        from .plotting import plot_sbc

        return plot_sbc(self, **kwargs)


def _symbol(res: PopulationResult) -> str:
    # component weights with a basis (as the prior draws them), else fractions or rates
    return "w" if res.basis is not None else "lambda" if res.kind == "rates" else "rho"


def _draws_of_value(res: PopulationResult) -> np.ndarray:
    """Draws on the scale the model is parametrised in: rates for rate models, and the
    component weights with a :class:`~binference.Basis` (as the prior draws them)."""
    return res.component_draws.reshape(-1, res.component_draws.shape[-1])


def _thin_draws(res: PopulationResult, n_draws: int) -> np.ndarray:
    X = _draws_of_value(res)
    if X.shape[0] < n_draws:
        raise ValueError(f"Need at least {n_draws} posterior draws per fit; got {X.shape[0]}.")
    idx = np.linspace(0, X.shape[0] - 1, n_draws).round().astype(int)
    return X[idx]


def sbc(
    simulate: Simulator,
    fit: Fitter,
    n_simulations: int = 100,
    *,
    n_draws: int = 99,
    log_likelihood: bool = True,
    rng: RNGLike = None,
    progress_bar: bool = False,
) -> SBCResult:
    """Simulation-based calibration.

    Parameters
    ----------
    simulate : callable
        ``simulate(rng) -> (likelihoods, true_value)`` drawing a population from
        the prior and a catalogue from it (see
        :func:`binference.simulate.prior_predictive`). The true value is the
        fractions, or the rates for rate models.
    fit : callable
        ``fit(likelihoods, rng) -> PopulationResult`` running the inference to
        be validated (with the same prior as the simulator).
    n_simulations : int
        Number of simulated data sets.
    n_draws : int
        Posterior draws used for ranking (thinned evenly from each fit; they
        should be roughly independent, i.e. fewer than the ESS: a warning
        names how many fits have a smallest bulk ESS below it, since
        autocorrelated draws make the rank histograms look over-dispersed).
    log_likelihood : bool
        Also rank the joint log-likelihood of the true population.
    """
    gen = as_generator(rng)
    ranks, ess_min = [], []
    names: list[str] = []
    for _ in progress(range(n_simulations), n_simulations, "SBC", progress_bar):
        r_sim, r_fit = spawn(gen, 2)
        lk, truth = simulate(r_sim)
        res = fit(lk, r_fit)
        D = _thin_draws(res, n_draws)
        row = list((D < truth[None, :]).sum(axis=0))
        if not names:
            names = [f"{_symbol(res)}[{k}]" for k in range(D.shape[1])]
            if log_likelihood:
                names.append("loglike")
        if log_likelihood:
            m = res.model
            ll_true = m.log_likelihood(truth, absolute=False)
            ll_draws = np.array([m.log_likelihood(d, absolute=False) for d in D])
            row.append(int((ll_draws < ll_true).sum()))
        ranks.append(row)
        d = res.diagnostics
        ess_min.append(np.nanmin(d["ess_bulk"]) if "ess_bulk" in d else np.nan)
    ess_min = np.asarray(ess_min)
    low = int(np.sum(ess_min < n_draws))  # NaN (no chains) compares False
    if low:
        warn(
            f"In {low} of {ess_min.size} fits the smallest bulk ESS is below n_draws={n_draws}; "
            "autocorrelated draws make the ranks look over-dispersed.",
            ConvergenceWarning,
        )
    return SBCResult(np.asarray(ranks, dtype=int), n_draws, names, ess_min)


# ----------------------------------------------------------------------------
# coverage
# ----------------------------------------------------------------------------
@dataclass
class CoverageResult:
    """Empirical coverage of central credible intervals."""

    levels: np.ndarray
    covered: np.ndarray  # (n_simulations, n_levels, n_quantities) bool
    names: list[str]

    def rates(self) -> np.ndarray:
        """Coverage rate per level and quantity, ``(n_levels, n_quantities)``."""
        return self.covered.mean(axis=0)

    def stderr(self) -> np.ndarray:
        """Binomial standard error of the nominal coverage, ``(n_levels, 1)``."""
        n = self.covered.shape[0]
        return np.sqrt(self.levels * (1 - self.levels) / n)[:, None]

    def average(self) -> np.ndarray:
        """Coverage per level averaged over quantities, ``(n_levels,)``."""
        return self.rates().mean(axis=1)


def coverage(
    simulate: Simulator,
    fit: Fitter,
    n_simulations: int = 100,
    *,
    levels: Sequence[float] = (0.5, 0.68, 0.9, 0.95),
    functionals: dict[str, Callable[[np.ndarray], float]] | None = None,
    rng: RNGLike = None,
    progress_bar: bool = False,
) -> CoverageResult:
    """Empirical coverage of central credible intervals over simulated data sets.

    ``functionals`` maps names to functions of the fractions, or of the rates
    for rate models (e.g. a mean
    redshift); their intervals are checked alongside every bin fraction.
    """
    gen = as_generator(rng)
    levels_arr = np.asarray(levels, dtype=float)
    functionals = functionals or {}
    out = []
    names: list[str] = []
    for _ in progress(range(n_simulations), n_simulations, "coverage", progress_bar):
        r_sim, r_fit = spawn(gen, 2)
        lk, truth = simulate(r_sim)
        res = fit(lk, r_fit)
        X = _draws_of_value(res)
        Q = [X[:, k] for k in range(X.shape[1])]
        tq = list(truth)
        for f in functionals.values():
            Q.append(np.array([f(r) for r in X]))
            tq.append(f(truth))
        if not names:
            names = [f"{_symbol(res)}[{k}]" for k in range(X.shape[1])] + list(functionals)
        Q = np.stack(Q, axis=1)
        tq = np.asarray(tq)
        cov = np.empty((levels_arr.size, Q.shape[1]), dtype=bool)
        for j, lev in enumerate(levels_arr):
            lo, hi = np.quantile(Q, [0.5 - lev / 2, 0.5 + lev / 2], axis=0)
            cov[j] = (tq >= lo) & (tq <= hi)
        out.append(cov)
    return CoverageResult(levels_arr, np.asarray(out), names)
