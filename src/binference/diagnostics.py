"""Convergence and quality diagnostics.

* :func:`ess` and :func:`rhat` follow Vehtari, Gelman, Simpson, Carpenter &
  Burkner (2021): rank-normalised, split-chain statistics.
* :func:`autocorr_time` follows the windowed estimator used by emcee
  (Sokal 1997; Goodman & Weare 2010).
* :func:`psis` implements Pareto-smoothed importance sampling (Vehtari et al.
  2024), returning smoothed log-weights and the shape diagnostic ``k_hat``.
* :func:`kkt_gap` and :func:`missing_information` are model-specific checks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import special, stats

from ._utils import chunk_slices

__all__ = [
    "autocorr_function",
    "autocorr_time",
    "ess",
    "importance_ess",
    "kkt_gap",
    "mcse",
    "missing_information",
    "psis",
    "psis_threshold",
    "rhat",
]


# ----------------------------------------------------------------------------
# autocorrelation
# ----------------------------------------------------------------------------
def _next_pow_two(n: int) -> int:
    i = 1
    while i < n:
        i <<= 1
    return i


def _autocov(x: np.ndarray) -> np.ndarray:
    """Biased autocovariance along the last axis (FFT)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    m = _next_pow_two(2 * n)
    xc = x - x.mean(axis=-1, keepdims=True)
    f = np.fft.rfft(xc, n=m, axis=-1)
    acov = np.fft.irfft(f * np.conjugate(f), n=m, axis=-1)[..., :n]
    return acov / n


def autocorr_function(x: Any) -> np.ndarray:
    """Normalised autocorrelation function of a 1-D chain (or average over chains)."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    acov = _autocov(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        acf = acov / acov[..., :1]
    acf = np.where(np.isfinite(acf), acf, 0.0)
    return acf.mean(axis=0)


def autocorr_time(x: Any, c: float = 5.0) -> float:
    """Integrated autocorrelation time with Sokal's adaptive window.

    ``x`` is a 1-D chain or an ``(n_chains, n_draws)`` array (autocorrelation
    functions are averaged over chains, as in emcee).
    """
    f = autocorr_function(x)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = np.arange(len(taus)) < c * taus
    m = int(np.argmin(window)) if np.any(~window) else len(taus) - 1
    return float(max(taus[m], 1.0))


# ----------------------------------------------------------------------------
# ESS and R-hat (Vehtari et al. 2021)
# ----------------------------------------------------------------------------
def _as_chains(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError("Expected an array of shape (n_draws,) or (n_chains, n_draws).")
    return a


def _split(a: np.ndarray) -> np.ndarray:
    n = a.shape[1]
    if n < 4:
        return a
    half = n // 2
    return np.concatenate([a[:, :half], a[:, n - half :]], axis=0)


def _z_scale(a: np.ndarray) -> np.ndarray:
    flat = a.ravel()
    ranks = stats.rankdata(flat, method="average")
    z = special.ndtri((ranks - 3.0 / 8.0) / (flat.size - 2.0 * 3.0 / 8.0 + 1.0))
    return z.reshape(a.shape)


def _ess_raw(a: np.ndarray) -> float:
    n_chain, n_draw = a.shape
    if n_draw < 4:
        return float("nan")
    if np.all(a == a.flat[0]):
        return float("nan")
    acov = _autocov(a)
    chain_mean = a.mean(axis=1)
    mean_var = np.mean(acov[:, 0]) * n_draw / (n_draw - 1.0)
    var_plus = mean_var * (n_draw - 1.0) / n_draw
    if n_chain > 1:
        var_plus += np.var(chain_mean, ddof=1)
    if var_plus <= 0:
        return float("nan")
    rho = np.zeros(n_draw)
    rho_even = 1.0
    rho[0] = rho_even
    rho_odd = 1.0 - (mean_var - np.mean(acov[:, 1])) / var_plus
    rho[1] = rho_odd
    t = 1
    while t < n_draw - 3 and (rho_even + rho_odd) > 0.0:
        rho_even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        rho_odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if (rho_even + rho_odd) >= 0:
            rho[t + 1] = rho_even
            rho[t + 2] = rho_odd
        t += 2
    max_t = t - 2
    if rho_even > 0:
        rho[max_t + 1] = rho_even
    t = 1
    while t <= max_t - 2:
        if (rho[t + 1] + rho[t + 2]) > (rho[t - 1] + rho[t]):
            rho[t + 1] = (rho[t - 1] + rho[t]) / 2.0
            rho[t + 2] = rho[t + 1]
        t += 2
    total = n_chain * n_draw
    tau = -1.0 + 2.0 * np.sum(rho[: max_t + 1]) + np.sum(rho[max_t + 1 : max_t + 2])
    tau = max(tau, 1.0 / np.log10(total))
    return float(total / tau)


def ess(x: Any, kind: str = "bulk") -> float:
    """Effective sample size of a scalar quantity.

    Parameters
    ----------
    x : array_like, shape (n_draws,) or (n_chains, n_draws)
    kind : {"bulk", "tail", "mean"}
        ``"bulk"``: rank-normalised split-chain ESS; ``"tail"``: minimum of the
        ESS of the 5% and 95% quantile indicators; ``"mean"``: ESS of the raw
        (split) draws, appropriate for the Monte Carlo error of the mean.
    """
    a = _split(_as_chains(x))
    if kind == "bulk":
        return _ess_raw(_z_scale(a))
    if kind == "mean":
        return _ess_raw(a)
    if kind == "tail":
        q05, q95 = np.quantile(a, [0.05, 0.95])
        e1 = _ess_raw((a <= q05).astype(float))
        e2 = _ess_raw((a <= q95).astype(float))
        return float(np.nanmin([e1, e2])) if np.isfinite([e1, e2]).any() else float("nan")
    raise ValueError("kind must be 'bulk', 'tail' or 'mean'.")


def _rhat_raw(a: np.ndarray) -> float:
    n = a.shape[1]
    if a.shape[0] < 2 or n < 2:
        return float("nan")
    chain_mean = a.mean(axis=1)
    between = n * np.var(chain_mean, ddof=1)
    within = np.mean(np.var(a, axis=1, ddof=1))
    if within <= 0:
        return float("nan")
    return float(np.sqrt((between / within + n - 1.0) / n))


def rhat(x: Any) -> float:
    """Rank-normalised split R-hat (maximum of bulk and folded/tail versions).

    Needs ``x`` of shape ``(n_chains, n_draws)``; a single chain is split in
    two. Values above about 1.01 indicate non-convergence.
    """
    a = _split(_as_chains(x))
    if a.shape[0] < 2:
        return float("nan")
    bulk = _rhat_raw(_z_scale(a))
    folded = _rhat_raw(_z_scale(np.abs(a - np.median(a))))
    return float(np.nanmax([bulk, folded]))


def mcse(x: Any) -> float:
    """Monte Carlo standard error of the mean of a scalar quantity."""
    a = _as_chains(x)
    n_eff = ess(a, kind="mean")
    if not np.isfinite(n_eff) or n_eff <= 0:
        return float("nan")
    return float(np.std(a, ddof=1) / np.sqrt(n_eff))


# ----------------------------------------------------------------------------
# Pareto-smoothed importance sampling
# ----------------------------------------------------------------------------
def _gpdfit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zhang & Stephens (2009) generalised-Pareto fit with the PSIS prior on k, of each
    row of the sorted exceedances ``x`` ``(..., n)``."""
    prior_bs, prior_k = 3.0, 10.0
    n = x.shape[-1]
    m_est = 30 + int(n**0.5)
    b = 1.0 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b = b / (prior_bs * x[..., int(n / 4 + 0.5) - 1, None]) + 1.0 / x[..., -1, None]
    k = np.log1p(-b[..., None] * x[..., None, :]).mean(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        len_scale = n * (np.log(-(b / k)) - k - 1.0)
    weights = 1.0 / np.exp(len_scale[..., None, :] - len_scale[..., None]).sum(axis=-1)
    weights = np.where(weights >= 10 * np.finfo(float).eps, weights, 0.0)  # NaN dropped too
    b_post = np.sum(b * weights, axis=-1) / weights.sum(axis=-1)
    k_post = np.log1p(-b_post[..., None] * x).mean(axis=-1)
    sigma = -k_post / b_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    return k_post, sigma


def _gpinv(p: np.ndarray, k: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    # quantiles p of generalised Paretos (k, sigma), broadcast; NaN where sigma <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(
            np.abs(k) < np.finfo(float).eps,
            -sigma * np.log1p(-p),
            sigma * np.expm1(-k * np.log1p(-p)) / k,
        )
    return np.where(sigma > 0, q, np.nan)


def psis_threshold(n: int) -> float:
    """The Pareto ``k_hat`` below which PSIS with ``n`` weights is reliable.

    ``min(1 - 1 / log10(n), 0.7)`` (Vehtari et al. 2024, §3), for ``n >= 5``.
    """
    if n < 5:
        raise ValueError(f"n must be at least 5 importance weights; got {n}.")
    return min(1.0 - 1.0 / float(np.log10(n)), 0.7)


def psis(log_weights: Any) -> tuple[np.ndarray, float]:
    """Pareto-smoothed importance sampling.

    Parameters
    ----------
    log_weights : array_like (n,)
        Unnormalised log importance ratios.

    Returns
    -------
    log_weights_smoothed : (n,) array
        Normalised (``logsumexp == 0``) smoothed log-weights.
    k_hat : float
        Estimated Pareto shape: the estimate is reliable below
        :func:`psis_threshold` of the number of weights (0.7 from about 2,000);
        ``-inf`` for equal weights, which have no tail.
    """
    lw = np.array(log_weights, dtype=float).ravel()
    if lw.size < 5:
        raise ValueError("Need at least 5 importance weights.")
    lw, k = _psis_rows(lw[None])  # lw is a copy
    return lw[0], float(k[0])


def _psis_rows(log_weights: np.ndarray, block: int = 2**22) -> tuple[np.ndarray, np.ndarray]:
    """:func:`psis` of each row of the float array ``log_weights`` ``(r, n)``, ``n >= 5``,
    in place: the smoothed log-weights (that array) and ``k_hat`` ``(r,)``. Rows whose
    tails are equally long are fitted together, about ``block`` values at a time."""
    lw = log_weights
    r, n = lw.shape
    lw[~np.isfinite(lw)] = -np.inf
    lw -= lw.max(axis=1, keepdims=True)
    k = np.full(r, np.inf)  # no tail of more than 4 weights to fit
    equal = np.all(lw > -1e-10, axis=1)  # equal weights (to round-off): exact, no tail
    k[equal] = -np.inf
    tail_len = int(np.ceil(min(0.2 * n, 3.0 * np.sqrt(n))))  # rounded up, as ArviZ and loo
    # the tail_len + 1 largest weights of each row, sorted: the cut and the tail above it
    top = np.argpartition(lw, n - tail_len - 1, axis=1)[:, n - tail_len - 1 :]
    srt = np.take_along_axis(lw, top, axis=1)
    order = np.argsort(srt, axis=1, kind="stable")
    top, srt = np.take_along_axis(top, order, axis=1), np.take_along_axis(srt, order, axis=1)
    cut = np.maximum(srt[:, 0], np.log(np.finfo(float).tiny))  # in logs, as ArviZ
    exp_cut = np.exp(cut)
    size = np.sum(srt > cut[:, None], axis=1)  # weights tied with the cut stay below it
    fit = ~equal & (size > 4)
    for m in np.unique(size[fit]):
        same = np.flatnonzero(fit & (size == m))
        probs = np.arange(0.5, m) / m
        for sl in chunk_slices(same.size, max(1, block // (m * (30 + int(m**0.5))))):
            rows = same[sl]
            x = np.exp(srt[rows, -m:]) - exp_cut[rows, None]
            k[rows], sigma = _gpdfit(x)
            k[rows] = np.where(np.isnan(k[rows]), np.inf, k[rows])  # a degenerate fit: no tail
            smoothed = np.log(_gpinv(probs, k[rows, None], sigma[:, None]) + exp_cut[rows, None])
            ok = np.isfinite(k[rows])
            lw[rows[ok, None], top[rows[ok], -m:]] = np.minimum(smoothed[ok], 0.0)
    mx = lw.max(axis=1, keepdims=True)  # normalised: logsumexp of each row zero
    lse = mx[:, 0] + np.log(np.exp(lw - mx).sum(axis=1))
    lw -= np.where(equal, np.log(n), lse)[:, None]
    return lw, k


# ----------------------------------------------------------------------------
# model-specific diagnostics
# ----------------------------------------------------------------------------
def kkt_gap(model: Any, value: Any, include_prior: bool = True) -> float:
    """First-order optimality violation of a point estimate.

    See :meth:`binference.PopulationModel.kkt_gap`.
    """
    return model.kkt_gap(np.asarray(value, dtype=float), include_prior=include_prior)


def missing_information(model: Any, value: Any, **kwargs: Any) -> float:
    """Largest fraction of missing information.

    See :meth:`binference.PopulationModel.missing_information`.
    """
    return model.missing_information(np.asarray(value, dtype=float), **kwargs)


def importance_ess(weights: Any) -> float:
    r"""Kish's effective sample size of importance weights,
    :math:`(\sum_s w_s)^2 / \sum_s w_s^2` (normalised or not)."""
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w) ** 2 / np.sum(w**2))
