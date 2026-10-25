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

from ._utils import RNGLike, as_generator, progress, spawn
from .likelihoods import Likelihoods
from .model import PopulationModel
from .priors import PopulationPrior
from .results import PopulationResult

__all__ = ["CoverageResult", "ExactPosterior", "SBCResult", "coverage", "exact_posterior", "sbc"]


# ----------------------------------------------------------------------------
# exact posterior by quadrature
# ----------------------------------------------------------------------------
@dataclass
class ExactPosterior:
    """Posterior on a grid over the simplex (``K <= 3``)."""

    points: np.ndarray
    weights: np.ndarray

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
    n_grid: int = 400,
    chunk: int = 20000,
) -> ExactPosterior:
    """Grid-quadrature posterior of the fractions for ``K = 2`` or ``K = 3`` bins.

    Works for priors with a closed-form simplex density (Dirichlet family).
    Selection effects are included.
    """
    if not isinstance(model, PopulationModel):
        model = PopulationModel(model, prior)
    if model.kind != "fractions" or not model.prior.has_density:
        raise ValueError(
            "exact_posterior needs a fractions model with a closed-form prior density."
        )
    K = model.K
    u = (np.arange(n_grid) + 0.5) / n_grid
    if K == 2:
        pts = np.stack([u, 1.0 - u], axis=1)
    elif K == 3:
        a, b = np.meshgrid(u, u, indexing="ij")
        keep = a + b < 1.0
        pts = np.stack([a[keep], b[keep], 1.0 - a[keep] - b[keep]], axis=1)
    else:
        raise ValueError("exact_posterior supports K = 2 or 3 bins.")
    L = model._L
    w = model._w
    logp = np.empty(pts.shape[0])
    for start in range(0, pts.shape[0], chunk):
        P = pts[start : start + chunk]
        S = np.asarray(L @ P.T)
        with np.errstate(divide="ignore"):
            ll = w @ np.log(S)
        if model.selection is not None:
            ll = ll - model.W * np.log(P @ model.selection)
        logp[start : start + chunk] = ll
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = logp + model.prior.log_density(pts)
    logp = np.where(np.isfinite(logp), logp, -np.inf)
    wts = np.exp(logp - special.logsumexp(logp))
    return ExactPosterior(points=pts, weights=wts)


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
        Quantity names (``"rho[k]"``, or ``"lambda[k]"`` for rates, and ``"loglike"``).
    """

    ranks: np.ndarray
    n_draws: int
    names: list[str]
    min_ess: np.ndarray = field(default_factory=lambda: np.empty(0))

    def histogram(self, n_bins: int | None = None) -> np.ndarray:
        """Rank histograms, shape ``(n_quantities, n_bins)``."""
        n_bins = n_bins or self._default_bins()
        edges = np.linspace(0, self.n_draws + 1, n_bins + 1)
        return np.stack([np.histogram(r, bins=edges)[0] for r in self.ranks.T])

    def _default_bins(self) -> int:
        n = self.ranks.shape[0]
        return int(max(2, min(20, n // 5, self.n_draws + 1)))

    def pvalues(self, n_bins: int | None = None) -> np.ndarray:
        """Chi-square p-values of rank uniformity per quantity."""
        H = self.histogram(n_bins)
        return np.array([stats.chisquare(h).pvalue for h in H])

    def passed(self, alpha: float = 0.01, n_bins: int | None = None) -> bool:
        """True if no quantity rejects uniformity at Bonferroni-corrected level ``alpha``."""
        p = self.pvalues(n_bins)
        return bool(np.all(p > alpha / p.size))

    def plot(self, **kwargs: Any):
        from .plotting import plot_sbc

        return plot_sbc(self, **kwargs)


def _draws_of_value(res: PopulationResult) -> np.ndarray:
    """Draws on the scale the model is parametrised in: rates for rate models."""
    return res.rates if res.kind == "rates" else res.samples


def _thin_draws(res: PopulationResult, n_draws: int) -> np.ndarray:
    X = _draws_of_value(res)
    if X.shape[0] < n_draws:
        raise ValueError(f"Need at least {n_draws} posterior draws per fit; got {X.shape[0]}.")
    idx = np.linspace(0, X.shape[0] - 1, n_draws).round().astype(int)
    return X[idx]


def _symbol(result) -> str:
    # the quantity ranked: fractions rho or rates lambda (audit G-10)
    return "lambda" if result.kind == "rates" else "rho"


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
        should be roughly independent, i.e. fewer than the ESS).
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
    return SBCResult(np.asarray(ranks, dtype=int), n_draws, names, np.asarray(ess_min))


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
