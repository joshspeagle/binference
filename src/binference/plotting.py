"""Plotting helpers (require matplotlib: ``pip install "binference[plot]"``).

Colours follow the active matplotlib style. The population (medians, bands, bars,
histograms) takes the first colour of ``axes.prop_cycle``, or ``color``; what the objects'
own data say (the stack, an object's own likelihood) takes the second, or
``sample_color``; reference lines, error bars and the truth (dashed) take ``text.color``.
2-D medians ramp from a pale tint of the population colour to the colour, and relative
uncertainties likewise in ``text.color``; masked bins show the axes background.
``SAMPLE_COLOR`` is the coral the binference docs give single objects, for ``sample_color``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ._utils import warn
from .warnings import InputWarning

__all__ = [
    "SAMPLE_COLOR",
    "plot_corner",
    "plot_individual",
    "plot_population",
    "plot_sbc",
    "plot_trace",
]

#: The coral of single objects and samples in the binference docs (their brand ``sample``
#: colour), to pass as ``sample_color``; ``docs/_scripts/brand.py --check`` keeps the two equal.
SAMPLE_COLOR = "#EC5F3E"


def _style_color(name: str, value: Any, slot: int, n: int | None = None):
    """``value`` as RGBA, or colour ``slot`` of the style's cycle when None; for the ``n``
    bars of categories also a list of ``n`` of them, one per bar.

    Resolved now, not left as "C0": a figure drawn after leaving a style context would
    otherwise colour its lines from another cycle than its bars and bands.
    """
    from matplotlib.colors import is_color_like, to_rgba

    if value is None:
        value = f"C{slot}"
    if is_color_like(value):
        return to_rgba(value)
    listed = isinstance(value, list | tuple) and all(map(is_color_like, value))
    if n and listed and len(value) == n:
        return [to_rgba(c) for c in value]
    many = f", a list of {n} of them (one per category)," if n else ""
    raise ValueError(f"{name} must be a matplotlib colour{many} or None, got {value!r}.")


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
    their shape is compared; the selected population of an overlay (the truth) is
    taken under the fit's selection.
    """
    v = np.asarray(values, dtype=float)
    v = v / v.sum()
    if which == "selected":
        if result.basis is not None:
            raise ValueError(
                "truth cannot be shown on which='selected' of a fit with a Basis, whose "
                "selection weighs components, not bins. Compare it on which='fractions'."
            )
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

    Overlays (``truth``, ``stack``) contribute only their shape: for rates (and rate
    models) they are scaled to the posterior mean total. For the selected population
    the truth is reweighted by the fit's selection (bins without a basis); the stack is
    not, as it averages the observed objects and so estimates the selected population
    already. 2-D grids show the median and the relative uncertainty as images and take
    no overlays.

    Parameters
    ----------
    result : PopulationResult or PointEstimate
    ax : matplotlib Axes, optional
        Axes to draw on; ``None`` makes a new figure.
    which : {"density", "fractions", "rates", "selected"}
        Plot densities per unit volume (n(z)-style), fractions per bin, rates, or the
        selected population per bin (the singular spellings are accepted too).
    levels : sequence of float
        Central credible levels shown as nested bands.
    truth : array_like (K,), optional
        True fractions to overlay.
    stack : array_like (K,), optional
        A naive stacked estimate to overlay (see :meth:`Likelihoods.stack
        <binference.Likelihoods.stack>`).
    color : matplotlib colour, optional
        Colour of the posterior, or for categories a list of colours, one per bar;
        ``None`` takes the style's first colour.
    sample_color : matplotlib colour, optional
        Colour of the ``stack`` overlay; ``None`` takes the style's second colour.
    label : str, optional
        Legend label of the median (default "posterior median").
    alpha : float
        Opacity of each credible band.
    step : bool
        Draw 1-D bins as steps (True) or as lines through the bin centres.

    Returns
    -------
    matplotlib Axes, or an array of Axes for 2-D grids
    """
    from .results import _central, _which_name

    plt = _plt()
    bins = result.bins
    color = _style_color("color", color, 0, n=bins.K if bins.is_categorical else None)
    sample_color = _style_color("sample_color", sample_color, 1)
    ink = plt.rcParams["text.color"]
    which = _which_name(which)
    if bins.ndim > 2 and not bins.is_categorical:
        raise ValueError(
            "plot supports 1-D and 2-D grids and categories; for more axes use "
            "result.density() or result.summary() (audit O-9)."
        )
    X = result._which(which)
    # overlays of rates (and of densities of rate models) are scaled to the mean total
    scaled = which == "rates" or result.kind == "rates"
    total = float(np.mean(result.total)) if scaled else 1.0
    if bins.ndim == 2 and not bins.is_categorical:
        if truth is not None or stack is not None or label is not None:
            warn("2-D plots show no truth, stack or label; they are ignored.", InputWarning)
        return _plot_2d(result, X, plt, ax, color)
    if truth is not None:
        truth = _overlay(result, truth, which, total)
    if stack is not None:  # a selected population already: not reweighted again
        stack = _overlay(result, stack, "fractions" if which == "selected" else which, total)
    med = np.median(X, axis=0)
    if ax is None:
        _, ax = plt.subplots()
    ylabel = _YLABEL[which]
    if bins.is_categorical:
        x = np.arange(bins.K)
        lo, hi = _central(X, levels[0])
        ax.bar(x, med, color=color, alpha=0.6, label=label or "posterior median")
        if X.shape[0] > 1:
            ax.errorbar(x, med, yerr=[med - lo, hi - med], fmt="none", ecolor=ink, capsize=3)
        if truth is not None:
            ax.plot(x, truth, "D", color=ink, ms=6, label="truth")
        if stack is not None:
            ax.plot(x, stack, "s", color=sample_color, ms=6, label="stacked")
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
        for y, c, ls, name in ((truth, ink, "--", "truth"), (stack, sample_color, ":", "stacked")):
            if y is not None:
                ax.step(edges, _step_y(bins, y), where="post", color=c, lw=1.5, ls=ls, label=name)
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
        n = 2 if sd is not None else 1
        # side by side when each panel, with its colour bar, keeps at least 4 in, else stacked
        shape = (1, n) if plt.rcParams["figure.figsize"][0] / n >= 4 else (n, 1)
        fig, ax = plt.subplots(*shape, figsize=_figsize(plt, *shape, 0.8), squeeze=False)
        ax = ax.ravel()
    else:
        fig = None
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
    if fig is not None:
        _tight(fig)
    return axes


def _figsize(plt, nrow: int, ncol: int, aspect: float) -> tuple:
    """The style's figure width, and a height of ``nrow`` panels of height ``aspect`` x
    their width, each with room for two lines of text (labels, titles)."""
    width = plt.rcParams["figure.figsize"][0]
    text = 2 * plt.rcParams["font.size"] / 72
    return width, nrow * (aspect * width / ncol + text)


def _subplots(plt, fig, nrow: int, ncol: int, aspect: float, **kwargs: Any):
    # a new figure of nrow x ncol axes, or those axes added to fig
    if fig is None:
        figsize = _figsize(plt, nrow, ncol, aspect)
        return plt.subplots(nrow, ncol, figsize=figsize, squeeze=False, **kwargs)
    return fig, np.asarray(fig.subplots(nrow, ncol, squeeze=False, **kwargs))


def _tight(fig) -> None:
    # tidy a figure the style left without a layout engine (a docs-style figure has one)
    if fig.get_layout_engine() is None:
        fig.tight_layout()


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
    """Corner plot of the fractions (or ``which="rates"``, ...) in selected bins.

    Parameters
    ----------
    result : PopulationResult
    idx : sequence of int, optional
        Bins to show; ``None`` chooses the ``max_bins`` with the largest mean.
    max_bins : int, optional
        Number of bins shown when ``idx`` is ``None``.
    n_hist : int, optional
        Histogram bins per panel.
    color : matplotlib colour, optional
        Colour of the histograms; ``None`` takes the style's first colour.
    fig : matplotlib Figure, optional
        Figure to draw on; ``None`` makes a new one.
    which : {"fractions", "rates", "density", "selected"}, optional
        Quantity shown: fractions, rates, densities or the selected population.

    Returns
    -------
    matplotlib Figure
    """
    plt = _plt()
    color = _style_color("color", color, 0)
    X, idx, name = _chosen_bins(result, which, idx, max_bins)
    n = len(idx)
    fig, axes = _subplots(plt, fig, n, n, 1.0)
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
    _tight(fig)
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
    line per chain.

    Parameters
    ----------
    result : PopulationResult
    idx : sequence of int, optional
        Bins to show; ``None`` chooses the ``max_bins`` with the largest mean.
    max_bins : int, optional
        Number of bins shown when ``idx`` is ``None``.
    fig : matplotlib Figure, optional
        Figure to draw on; ``None`` makes a new one.
    which : {"fractions", "rates", "density", "selected"}, optional
        Quantity shown: fractions, rates, densities or the selected population.

    Returns
    -------
    matplotlib Figure
    """
    X, idx, name = _chosen_bins(result, which, idx, max_bins)
    C = X.reshape(result.n_chains, -1, result.K)
    fig, axes = _subplots(_plt(), fig, len(idx), 1, 0.4, sharex=True)
    for a, k in zip(axes[:, 0], idx, strict=True):
        for c in range(C.shape[0]):
            a.plot(C[c, :, k], lw=0.6, alpha=0.8)
        a.set_ylabel(f"{name}[{k}]")
    axes[-1, 0].set_xlabel("iteration")
    _tight(fig)
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
    """Compare object ``i``'s own likelihood (flat-prior posterior) with its shrunk posterior.

    With a :class:`~binference.Basis` both are posteriors over its components, shared
    among the bins as in :meth:`~binference.PopulationResult.assignment_probabilities`.

    Parameters
    ----------
    result : PopulationResult or PointEstimate
    i : int
        Index of the object.
    ax : matplotlib Axes, optional
        Axes to draw on; ``None`` makes a new figure.
    n_draws : int, optional
        Population draws averaged for the shrunk posterior.
    color : matplotlib colour, optional
        Colour of the shrunk posterior; ``None`` takes the style's first colour.
    sample_color : matplotlib colour, optional
        Colour of the object's own likelihood; ``None`` takes the style's second colour.

    Returns
    -------
    matplotlib Axes
    """
    plt = _plt()
    color = _style_color("color", color, 0)
    sample_color = _style_color("sample_color", sample_color, 1)
    from .density import Density

    lk = result._require_model().likelihoods  # its columns: the bins or the components
    own = result._to_bins(lk[i].stack(Density.flat("z")))  # a population flat in z
    bins = result.bins
    shrunk = result.assignment_probabilities(idx=[i], n_draws=n_draws)[0]
    if ax is None:
        _, ax = plt.subplots()
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
    """Rank histograms from :func:`binference.validation.sbc` with a 99% uniformity band.

    Parameters
    ----------
    sbc_result : SBCResult
    n_bins : int, optional
        Histogram bins; ``None`` uses the default of :meth:`SBCResult.histogram
        <binference.validation.SBCResult.histogram>`.
    max_panels : int, optional
        Most quantities shown, in order; the joint log-likelihood (``"loglike"``) is
        always among them.
    fig : matplotlib Figure, optional
        Figure to draw on; ``None`` makes a new one.
    color : matplotlib colour, optional
        Colour of the histograms; ``None`` takes the style's first colour.

    Returns
    -------
    matplotlib Figure
    """
    from scipy import stats

    plt = _plt()
    color = _style_color("color", color, 0)
    ink = plt.rcParams["text.color"]
    H = sbc_result.histogram(n_bins)
    names = sbc_result.names
    q = min(len(names), max_panels)
    show = list(range(q))
    if names[-1] == "loglike":  # sbc ranks it last
        show[-1] = len(names) - 1
    ncol = min(4, q)
    nrow = int(np.ceil(q / ncol))
    fig, axes = _subplots(plt, fig, nrow, ncol, 1.2)
    n_sims = sbc_result.ranks.shape[0]
    nb = H.shape[1]
    # per bin: bins can hold different numbers of the possible ranks (audit G-1)
    p = sbc_result.expected(n_bins) / n_sims
    lo, hi = stats.binom.interval(0.99, n_sims, p)
    for a in axes.flat[q:]:
        a.set_visible(False)
    for a, k in zip(axes.flat[:q], show, strict=True):
        a.stairs(hi, np.arange(nb + 1) - 0.5, baseline=lo, fill=True, color=ink, alpha=0.15, lw=0)
        a.bar(np.arange(nb), H[k], width=1.0, color=color, alpha=0.8)
        a.set_title(names[k])
        a.set_yticks([])
    _tight(fig)
    return fig
