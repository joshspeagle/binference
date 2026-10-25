"""Plotting helpers (require matplotlib: ``pip install "binference[plot]"``)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["plot_corner", "plot_individual", "plot_population", "plot_sbc", "plot_trace"]

# Colour-vision-deficiency-checked categorical colours (first slots of a validated
# palette) and ink colours for reference lines.
POSTERIOR_COLOR = "#2a78d6"
STACK_COLOR = "#eb6834"
INK = "#0b0b0b"


def _tidy(ax) -> None:
    """Recessive axes: no top/right spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:  # pragma: no cover - depends on environment
        raise ImportError("Plotting needs matplotlib: pip install 'binference[plot]'.") from err
    return plt


def _draws(result: Any, which: str) -> np.ndarray:
    from .results import PointEstimate

    if isinstance(result, PointEstimate):
        base = result.density() if which == "density" else result.rho
        return base[None, :]
    if which == "density":
        return result.density()
    if which == "fraction":
        return result.samples
    if which == "rate":
        rates = result.rates
        if rates is None:
            raise ValueError("No rates in this result.")
        return rates
    raise ValueError("which must be 'density', 'fraction' or 'rate'.")


def _to_which(values: np.ndarray, bins: Any, which: str, total: float = 1.0) -> np.ndarray:
    """Overlay values (fractions or rates) on the scale of the plotted posterior.

    Overlays are normalised and multiplied by ``total``, the posterior mean total
    rate for rate models (1 otherwise), so only their shape is compared.
    """
    v = np.asarray(values, dtype=float)
    v = v / v.sum()
    if which == "fraction":
        return v
    v = v * total
    return v / bins.volumes if which == "density" else v


def plot_population(
    result: Any,
    ax: Any = None,
    *,
    which: str = "density",
    levels: Sequence[float] = (0.68, 0.95),
    truth: Any = None,
    stack: Any = None,
    color: str = POSTERIOR_COLOR,
    label: str | None = None,
    alpha: float = 0.25,
    step: bool = True,
):
    """Plot a population posterior: bands for 1-D bins, images for 2-D grids, bars for categories.

    Parameters
    ----------
    result : PopulationResult or PointEstimate
    ax : matplotlib Axes, optional
    which : {"density", "fraction", "rate"}
        Plot densities per unit volume (n(z)-style), fractions per bin, or rates.
    levels : sequence of float
        Central credible levels shown as nested bands.
    truth, stack : array_like (K,), optional
        True fractions and a naive stacked estimate to overlay. Only their
        shape is used: for rate models they are scaled to the posterior mean
        total rate.
    """
    plt = _plt()
    bins = result.bins
    X = _draws(result, which)
    med = np.median(X, axis=0)
    # overlays of rates (and of densities of rate models) are scaled to the mean total
    scaled = which == "rate" or result.kind == "rates"
    total = float(np.mean(result.total)) if scaled else 1.0
    if bins.ndim == 2 and not bins.is_categorical:
        return _plot_2d(result, X, plt, ax)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ylabel = {"density": "density", "fraction": "fraction per bin", "rate": "rate"}[which]
    if bins.is_categorical:
        x = np.arange(bins.K)
        lo, hi = np.quantile(X, [0.5 - levels[0] / 2, 0.5 + levels[0] / 2], axis=0)
        ax.bar(x, med, color=color, alpha=0.6, label=label or "posterior median")
        if X.shape[0] > 1:
            ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="none", ecolor=INK, capsize=3)
        if truth is not None:
            ax.plot(x, _to_which(truth, bins, which, total), "D", color=INK, ms=6, label="truth")
        if stack is not None:
            ax.plot(
                x,
                _to_which(stack, bins, which, total),
                "s",
                color=STACK_COLOR,
                ms=6,
                label="stacked",
            )
        ax.set_xticks(x, [str(lab) for lab in bins.labels], rotation=45, ha="right")
    else:
        edges = bins.edges[0]
        centers = bins.centers
        band_handles = []
        for n_layers, lev in enumerate(sorted(levels, reverse=True), start=1):
            if X.shape[0] < 2:
                break
            # the region inside this band is covered by n_layers translucent fills
            band_handles.append(
                _band_patch(color, 1.0 - (1.0 - alpha) ** n_layers, f"{lev:.0%} interval")
            )
            lo, hi = np.quantile(X, [0.5 - lev / 2, 0.5 + lev / 2], axis=0)
            if step:
                ax.fill_between(
                    edges,
                    _step_y(bins, lo),
                    _step_y(bins, hi),
                    step="post",
                    color=color,
                    alpha=alpha,
                    lw=0,
                )
            else:
                ax.fill_between(centers, lo, hi, color=color, alpha=alpha, lw=0)
        lab = label or "posterior median"
        if step:
            ax.step(edges, _step_y(bins, med), where="post", color=color, lw=1.5, label=lab)
        else:
            ax.plot(centers, med, color=color, lw=1.5, label=lab)
        if truth is not None:
            t = _to_which(truth, bins, which, total)
            ax.step(
                edges, _step_y(bins, t), where="post", color=INK, lw=1.5, ls="--", label="truth"
            )
        if stack is not None:
            s = _to_which(stack, bins, which, total)
            ax.step(
                edges,
                _step_y(bins, s),
                where="post",
                color=STACK_COLOR,
                lw=1.5,
                ls=":",
                label="stacked",
            )
        ax.set_xlim(edges[0], edges[-1])
        ax.set_xlabel("bin coordinate")
        handles, labels = ax.get_legend_handles_labels()
        # median first, then bands (innermost first), then reference lines
        handles = handles[:1] + band_handles[::-1] + handles[1:]
        labels = labels[:1] + [h.get_label() for h in band_handles[::-1]] + labels[1:]
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        _tidy(ax)
        ax.legend(handles, labels, frameon=False)
        return ax
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    _tidy(ax)
    ax.legend(frameon=False)
    return ax


def _step_y(bins, y):
    # per-bin values on the full 1-D grid (NaN, a gap, in masked bins), padded for step="post"
    full = bins.unflatten(np.asarray(y, dtype=float))
    return np.append(full, full[-1])


def _band_patch(color: str, alpha: float, label: str):
    from matplotlib.patches import Patch

    return Patch(facecolor=color, alpha=alpha, lw=0, label=label)


def _plot_2d(result, X, plt, ax):
    bins = result.bins
    med = bins.unflatten(np.median(X, axis=0))
    sd = bins.unflatten(X.std(axis=0)) if X.shape[0] > 1 else None
    ex, ey = bins.edges
    if ax is None:
        ncol = 2 if sd is not None else 1
        _, ax = plt.subplots(1, ncol, figsize=(5 * ncol, 4), squeeze=False)
        ax = ax[0]
    axes = np.atleast_1d(ax)
    im = axes[0].pcolormesh(ex, ey, med.T, shading="flat")
    plt.colorbar(im, ax=axes[0], label="posterior median")
    axes[0].set_xlabel("axis 0")
    axes[0].set_ylabel("axis 1")
    if sd is not None and len(axes) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(med > 0, sd / med, np.nan)
        im2 = axes[1].pcolormesh(ex, ey, rel.T, shading="flat", cmap="magma")
        plt.colorbar(im2, ax=axes[1], label="relative uncertainty (sd / median)")
        axes[1].set_xlabel("axis 0")
    return axes


def plot_corner(
    result: Any,
    idx: Sequence[int] | None = None,
    *,
    max_bins: int = 4,
    n_hist: int = 30,
    color: str = POSTERIOR_COLOR,
    fig: Any = None,
):
    """Corner plot of the fractions in selected bins (default: the most populated)."""
    plt = _plt()
    X = result.samples
    if idx is None:
        idx = np.sort(np.argsort(-X.mean(axis=0))[:max_bins])
    idx = list(idx)
    n = len(idx)
    if fig is None:
        fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n), squeeze=False)
    else:
        axes = np.asarray(fig.subplots(n, n, squeeze=False))
    for i, ki in enumerate(idx):
        for j, kj in enumerate(idx):
            a = axes[i, j]
            if j > i:
                a.set_visible(False)
                continue
            if i == j:
                a.hist(X[:, ki], bins=n_hist, color=color, histtype="step")
                a.set_yticks([])
            else:
                a.hist2d(X[:, kj], X[:, ki], bins=n_hist, cmap="Blues")
            if i == n - 1:
                a.set_xlabel(f"rho[{kj}]")
            if j == 0 and i > 0:
                a.set_ylabel(f"rho[{ki}]")
    fig.tight_layout()
    return fig


def plot_trace(
    result: Any, idx: Sequence[int] | None = None, *, max_bins: int = 4, fig: Any = None
):
    """Trace plots of the fractions in selected bins, one line per chain."""
    plt = _plt()
    C = result.chains
    if idx is None:
        idx = np.sort(np.argsort(-C.reshape(-1, C.shape[-1]).mean(axis=0))[:max_bins])
    idx = list(idx)
    if fig is None:
        fig, axes = plt.subplots(
            len(idx), 1, figsize=(8, 1.8 * len(idx)), sharex=True, squeeze=False
        )
    else:
        axes = np.asarray(fig.subplots(len(idx), 1, sharex=True, squeeze=False))
    for a, k in zip(axes[:, 0], idx, strict=True):
        for c in range(C.shape[0]):
            a.plot(C[c, :, k], lw=0.6, alpha=0.8)
        a.set_ylabel(f"rho[{k}]")
    axes[-1, 0].set_xlabel("iteration")
    fig.tight_layout()
    return fig


def plot_individual(result: Any, i: int, ax: Any = None, n_draws: int = 200):
    """Compare object ``i``'s own likelihood (flat-prior posterior) with its shrunk posterior."""
    plt = _plt()
    model = result._require_model()
    bins = result.bins
    own = model.likelihoods.subset([i]).toarray()[0] * bins.volumes
    own = own / own.sum()
    shrunk = result.assignment_probabilities(idx=[i], n_draws=n_draws)[0]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    if bins.is_categorical or bins.ndim > 1:
        x = np.arange(bins.K)
        ax.bar(x - 0.2, own, width=0.4, color="0.6", label="own likelihood (flat prior)")
        ax.bar(x + 0.2, shrunk, width=0.4, color=POSTERIOR_COLOR, label="hierarchical posterior")
        ax.set_xlabel("bin")
    else:
        edges = bins.edges[0]
        vol = bins.volumes
        ax.step(
            edges,
            _step_y(bins, own / vol),
            where="post",
            color="0.45",
            lw=1.5,
            ls="--",
            label="own likelihood (flat prior)",
        )
        ax.step(
            edges,
            _step_y(bins, shrunk / vol),
            where="post",
            color=POSTERIOR_COLOR,
            lw=1.5,
            label="hierarchical posterior",
        )
        ax.set_xlabel("bin coordinate")
    ax.set_ylabel("posterior density")
    _tidy(ax)
    ax.legend(frameon=False)
    return ax


def plot_sbc(sbc_result: Any, n_bins: int | None = None, max_panels: int = 12, fig: Any = None):
    """Rank histograms from :func:`binference.validation.sbc` with a 99% uniformity band."""
    from scipy import stats

    plt = _plt()
    H = sbc_result.histogram(n_bins)
    names = sbc_result.names
    q = min(len(names), max_panels)
    ncol = min(4, q)
    nrow = int(np.ceil(q / ncol))
    if fig is None:
        fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.4 * nrow), squeeze=False)
    else:
        axes = np.asarray(fig.subplots(nrow, ncol, squeeze=False))
    n_sims = sbc_result.ranks.shape[0]
    nb = H.shape[1]
    # per bin: bins can hold different numbers of the possible ranks (audit G-1)
    p = sbc_result.expected(n_bins) / n_sims
    lo, hi = stats.binom.interval(0.99, n_sims, p)
    for j in range(nrow * ncol):
        a = axes.flat[j]
        if j >= q:
            a.set_visible(False)
            continue
        a.stairs(hi, np.arange(nb + 1) - 0.5, baseline=lo, fill=True, color="0.85")
        a.bar(np.arange(nb), H[j], width=1.0, color=POSTERIOR_COLOR, alpha=0.8)
        a.set_title(names[j], fontsize=9)
        a.set_yticks([])
    fig.tight_layout()
    return fig
