"""Plotting helpers (require matplotlib: ``pip install "binference[plot]"``).

Colours follow the active matplotlib style. The population (medians, bands, bars,
histograms) takes the first colour of ``axes.prop_cycle``, or ``color``; what the objects'
own data say (the stack, an object's own likelihood) takes the second, or
``sample_color``; reference lines, error bars and the truth (dashed) take ``text.color``.
2-D medians ramp from a pale tint of the population colour to the colour, and relative
uncertainties likewise in ``text.color``; masked bins show the axes background.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._utils import warn
from .warnings import InputWarning

__all__ = ["plot_corner", "plot_individual", "plot_population", "plot_sbc", "plot_trace"]


def _style_color(name: str, value: Any, slot: int):
    """``value`` as RGBA (a list of them for per-bar colours), or colour ``slot`` of the
    style's cycle when None.

    Resolved now, not left as "C0": a figure drawn after leaving a style context would
    otherwise colour its lines from another cycle than its bars and bands.
    """
    from matplotlib.colors import is_color_like, to_rgba

    if value is None:
        value = f"C{slot}"
    if is_color_like(value):
        return to_rgba(value)
    if isinstance(value, list | tuple) and value and all(is_color_like(c) for c in value):
        return [to_rgba(c) for c in value]
    raise ValueError(
        f"{name} must be a matplotlib colour, a list of them, or None, got {value!r}."
    )


def _ramp(ax, color):
    """A sequential colormap from a pale tint of ``color`` (over the axes background, so
    zero differs from a masked bin) to ``color``."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgba

    ground, ink = np.array(to_rgba(ax.get_facecolor())), np.array(to_rgba(color))
    return LinearSegmentedColormap.from_list("ramp", [0.85 * ground + 0.15 * ink, ink])


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


#: axis labels of what can be plotted (``which``)
_YLABEL = {
    "density": "density",
    "fractions": "fraction per bin",
    "rates": "rate",
    "selected": "selected fraction per bin",
}


def _overlay(result: Any, values: Any, which: str, total: float) -> np.ndarray:
    """Overlay values (fractions or rates) on the scale of the plotted posterior.

    Overlays are normalised and multiplied by ``total``, the posterior mean total
    when rates are plotted or the model is a rate model (1 otherwise), so only
    their shape is compared; the selected population of an overlay is taken under
    the fit's selection.
    """
    v = np.asarray(values, dtype=float)
    v = v / v.sum()
    if which == "selected":
        return result._selected_of(v[None, :]).mean(axis=0)
    if which == "fractions":
        return v
    v = v * total
    return v / result.bins.volumes if which == "density" else v


def plot_population(
    result: Any,
    ax: Any = None,
    *,
    which: str = "density",
    levels: Sequence[float] = (0.68, 0.95),
    truth: Any = None,
    stack: Any = None,
    color: Any = None,
    sample_color: Any = None,
    label: str | None = None,
    alpha: float = 0.25,
    step: bool = True,
):
    """Plot a population posterior: bands for 1-D bins, images for 2-D grids, bars for categories.

    Parameters
    ----------
    result : PopulationResult or PointEstimate
    ax : matplotlib Axes, optional
    which : {"density", "fractions", "rates", "selected"}
        Plot densities per unit volume (n(z)-style), fractions per bin, rates, or the
        selected population per bin (the singular spellings are accepted too).
    levels : sequence of float
        Central credible levels shown as nested bands.
    truth, stack : array_like (K,), optional
        True fractions and a naive stacked estimate to overlay. Only their
        shape is used: for rates (and rate models) they are scaled to the
        posterior mean total; for the selected population they are reweighted by the
        fit's selection (bins without a basis). 2-D grids show the median and the
        relative uncertainty as images (an array of two axes) and take no overlays.
    color, sample_color : matplotlib colour, optional
        Colours of the posterior and of the ``stack`` overlay (default: the style's;
        see the module docstring).
    label : str, optional
        Legend label of the median (default "posterior median").
    alpha : float
        Opacity of each credible band.
    step : bool
        Draw 1-D bins as steps (True) or as lines through the bin centres.
    """
    from .results import _central, _which_name

    plt = _plt()
    color = _style_color("color", color, 0)
    sample_color = _style_color("sample_color", sample_color, 1)
    ink = plt.rcParams["text.color"]
    bins = result.bins
    which = _which_name(which)
    if bins.ndim > 2 and not bins.is_categorical:
        raise ValueError(
            "plot supports 1-D and 2-D grids and categories; for more axes use "
            "result.density() or result.summary() (audit O-9)."
        )
    X = result._which(which)
    med = np.median(X, axis=0)
    # overlays of rates (and of densities of rate models) are scaled to the mean total
    scaled = which == "rates" or result.kind == "rates"
    total = float(np.mean(result.total)) if scaled else 1.0
    if bins.ndim == 2 and not bins.is_categorical:
        if truth is not None or stack is not None or label is not None:
            warn("2-D plots show no truth, stack or label; they are ignored.", InputWarning)
        return _plot_2d(result, X, plt, ax, color)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ylabel = _YLABEL[which]
    if bins.is_categorical:
        x = np.arange(bins.K)
        lo, hi = _central(X, levels[0])
        ax.bar(x, med, color=color, alpha=0.6, label=label or "posterior median")
        if X.shape[0] > 1:
            ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="none", ecolor=ink, capsize=3)
        if truth is not None:
            ax.plot(x, _overlay(result, truth, which, total), "D", color=ink, ms=6, label="truth")
        if stack is not None:
            ax.plot(
                x,
                _overlay(result, stack, which, total),
                "s",
                color=sample_color,
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
            lo, hi = _central(X, lev)
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
            t = _overlay(result, truth, which, total)
            ax.step(
                edges, _step_y(bins, t), where="post", color=ink, lw=1.5, ls="--", label="truth"
            )
        if stack is not None:
            s = _overlay(result, stack, which, total)
            ax.step(
                edges,
                _step_y(bins, s),
                where="post",
                color=sample_color,
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


def _plot_2d(result, X, plt, ax, color):
    bins = result.bins
    med = bins.unflatten(np.median(X, axis=0))
    sd = bins.unflatten(X.std(axis=0)) if X.shape[0] > 1 else None
    ex, ey = bins.edges
    if ax is None:
        ncol = 2 if sd is not None else 1
        _, ax = plt.subplots(1, ncol, figsize=(5 * ncol, 4), squeeze=False)
        ax = ax[0]
    axes = np.atleast_1d(ax)
    im = axes[0].pcolormesh(ex, ey, med.T, shading="flat", cmap=_ramp(axes[0], color))
    plt.colorbar(im, ax=axes[0], label="posterior median")
    axes[0].set_xlabel("axis 0")
    axes[0].set_ylabel("axis 1")
    if sd is not None and len(axes) > 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(med > 0, sd / med, np.nan)
        grey = _ramp(axes[1], plt.rcParams["text.color"])
        im2 = axes[1].pcolormesh(ex, ey, rel.T, shading="flat", cmap=grey)
        plt.colorbar(im2, ax=axes[1], label="relative uncertainty (sd / median)")
        axes[1].set_xlabel("axis 0")
    return axes


def _subplots(plt, fig, nrow: int, ncol: int, figsize: tuple, **kwargs: Any):
    # a new figure of nrow x ncol axes, or those axes added to fig
    if fig is None:
        return plt.subplots(nrow, ncol, figsize=figsize, squeeze=False, **kwargs)
    return fig, np.asarray(fig.subplots(nrow, ncol, squeeze=False, **kwargs))


def _chosen_bins(result: Any, which: str, idx: Any, max_bins: int):
    """The draws of ``which``, the bins ``idx`` (by default the ``max_bins`` largest by
    mean, in order) and the quantity's symbol."""
    from .results import _SYMBOL, _which_name

    X = result._which(which)
    if idx is None:
        idx = np.sort(np.argsort(-X.mean(axis=0))[:max_bins])
    return X, list(idx), _SYMBOL[_which_name(which)]


def plot_corner(
    result: Any,
    idx: Sequence[int] | None = None,
    *,
    max_bins: int = 4,
    n_hist: int = 30,
    color: Any = None,
    fig: Any = None,
    which: str = "fractions",
):
    """Corner plot of the fractions (or ``which="rates"``, ...) in selected bins
    (default: the most populated), in ``color`` (default: the style's)."""
    plt = _plt()
    color = _style_color("color", color, 0)
    X, idx, name = _chosen_bins(result, which, idx, max_bins)
    n = len(idx)
    fig, axes = _subplots(plt, fig, n, n, (2.2 * n, 2.2 * n))
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
                a.hist2d(X[:, kj], X[:, ki], bins=n_hist, cmap=_ramp(a, color))
            if i == n - 1:
                a.set_xlabel(f"{name}[{kj}]")
            if j == 0 and i > 0:
                a.set_ylabel(f"{name}[{ki}]")
    fig.tight_layout()
    return fig


def plot_trace(
    result: Any,
    idx: Sequence[int] | None = None,
    *,
    max_bins: int = 4,
    fig: Any = None,
    which: str = "fractions",
):
    """Trace plots of the fractions (or ``which="rates"``, ...) in selected bins, one
    line per chain."""
    X, idx, name = _chosen_bins(result, which, idx, max_bins)
    C = X.reshape(result.n_chains, -1, result.K)
    fig, axes = _subplots(_plt(), fig, len(idx), 1, (8, 1.8 * len(idx)), sharex=True)
    for a, k in zip(axes[:, 0], idx, strict=True):
        for c in range(C.shape[0]):
            a.plot(C[c, :, k], lw=0.6, alpha=0.8)
        a.set_ylabel(f"{name}[{k}]")
    axes[-1, 0].set_xlabel("iteration")
    fig.tight_layout()
    return fig


def plot_individual(
    result: Any,
    i: int,
    ax: Any = None,
    n_draws: int = 200,
    *,
    color: Any = None,
    sample_color: Any = None,
):
    """Compare object ``i``'s own likelihood (flat-prior posterior, in ``sample_color``)
    with its shrunk posterior (in ``color``); colours default to the style's."""
    plt = _plt()
    color = _style_color("color", color, 0)
    sample_color = _style_color("sample_color", sample_color, 1)
    model = result._require_model()
    bins = result.bins
    own = model.likelihoods.subset([i]).toarray()[0] * bins.volumes
    own = own / own.sum()
    shrunk = result.assignment_probabilities(idx=[i], n_draws=n_draws)[0]
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3.5))
    if bins.is_categorical or bins.ndim > 1:
        x = np.arange(bins.K)
        ax.bar(x - 0.2, own, width=0.4, color=sample_color, label="own likelihood (flat prior)")
        ax.bar(x + 0.2, shrunk, width=0.4, color=color, label="hierarchical posterior")
        ax.set_xlabel("bin")
    else:
        edges = bins.edges[0]
        vol = bins.volumes
        ax.step(
            edges,
            _step_y(bins, own / vol),
            where="post",
            color=sample_color,
            lw=1.5,
            ls="--",
            label="own likelihood (flat prior)",
        )
        ax.step(
            edges,
            _step_y(bins, shrunk / vol),
            where="post",
            color=color,
            lw=1.5,
            label="hierarchical posterior",
        )
        ax.set_xlabel("bin coordinate")
    ax.set_ylabel("posterior density")
    _tidy(ax)
    ax.legend(frameon=False)
    return ax


def plot_sbc(
    sbc_result: Any,
    n_bins: int | None = None,
    max_panels: int = 12,
    fig: Any = None,
    *,
    color: Any = None,
):
    """Rank histograms from :func:`binference.validation.sbc` (in ``color``, default the
    style's) with a 99% uniformity band."""
    from scipy import stats

    plt = _plt()
    color = _style_color("color", color, 0)
    ink = plt.rcParams["text.color"]
    H = sbc_result.histogram(n_bins)
    names = sbc_result.names
    q = min(len(names), max_panels)
    ncol = min(4, q)
    nrow = int(np.ceil(q / ncol))
    fig, axes = _subplots(plt, fig, nrow, ncol, (3 * ncol, 2.4 * nrow))
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
        a.stairs(hi, np.arange(nb + 1) - 0.5, baseline=lo, fill=True, color=ink, alpha=0.15, lw=0)
        a.bar(np.arange(nb), H[j], width=1.0, color=color, alpha=0.8)
        a.set_title(names[j], fontsize=9)
        a.set_yticks([])
    fig.tight_layout()
    return fig
