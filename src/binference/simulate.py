"""Mock catalogues for tests, tutorials and simulation-based calibration.

Each ``mock_*`` function returns a :class:`Mock` bundling the bins, the
correct per-object :class:`~binference.Likelihoods`, the :class:`Truth`, and
(optionally) posteriors under an interim prior and posterior samples, so that
every input convention can be exercised. ``Mock`` unpacks as
``likelihoods, truth = mock``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import special, stats

from ._utils import RNGLike, as_generator
from .bins import Bins
from .likelihoods import Likelihoods, _subbin_points
from .priors import Prior

__all__ = [
    "Mock",
    "Truth",
    "mock_1d",
    "mock_2d",
    "mock_categorical",
    "mock_unfolding",
    "population",
    "prior_predictive",
]


@dataclass
class Truth:
    """The true population and objects behind a mock catalogue.

    Attributes
    ----------
    rho : (K,) array
        Population fractions the objects were drawn from.
    labels : (N,) int array
        True bin of every observed object.
    values : array or None
        True continuous values of the observed objects (``None`` for categories).
    counts : (K,) array
        Realised histogram of the observed objects' true bins.
    n_drawn : int
        Objects drawn before selection.
    """

    rho: np.ndarray
    labels: np.ndarray
    values: np.ndarray | None
    counts: np.ndarray
    n_drawn: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Mock:
    """A mock catalogue: bins, likelihoods, truth and alternative input forms."""

    bins: Bins
    likelihoods: Likelihoods
    truth: Truth
    measurements: dict[str, np.ndarray] = field(default_factory=dict)
    posteriors: np.ndarray | None = None
    interim_prior: np.ndarray | None = None
    samples: np.ndarray | None = None

    def __iter__(self) -> Iterator[Any]:
        yield self.likelihoods
        yield self.truth


# ----------------------------------------------------------------------------
# densities and populations
# ----------------------------------------------------------------------------
def _density_1d(name: str, lo: float, hi: float) -> Callable[[np.ndarray], np.ndarray]:
    span = hi - lo

    def unit(z):
        return (np.asarray(z, dtype=float) - lo) / span

    if name == "bimodal":
        return lambda z: (
            0.6 * stats.beta.pdf(unit(z), 3.0, 9.0) + 0.4 * stats.norm.pdf(unit(z), 0.55, 0.1)
        )
    if name == "gaussian":
        return lambda z: stats.norm.pdf(unit(z), 0.5, 0.15)
    if name == "uniform":
        return lambda z: np.ones_like(np.asarray(z, dtype=float))
    if name == "steep":
        return lambda z: np.exp(-4.0 * unit(z))
    raise ValueError(f"Unknown density {name!r}; use 'bimodal', 'gaussian', 'uniform' or 'steep'.")


def _density_2d(name: str, bins: Bins) -> Callable[[np.ndarray], np.ndarray]:
    (lx, hx), (ly, hy) = bins.range

    def f(p):
        p = np.asarray(p, dtype=float)
        u = (p[..., 0] - lx) / (hx - lx)
        v = (p[..., 1] - ly) / (hy - ly)
        if name == "banana":
            ridge = 0.2 + 0.55 * u + 0.12 * np.sin(3.0 * u)
            return stats.beta.pdf(u, 2.0, 2.5) * np.exp(-0.5 * ((v - ridge) / 0.06) ** 2)
        if name == "gaussian":
            return np.exp(-0.5 * (((u - 0.5) / 0.15) ** 2 + ((v - 0.5) / 0.2) ** 2))
        raise ValueError(f"Unknown 2-D density {name!r}; use 'banana' or 'gaussian'.")

    return f


def _as_density(density: Any, bins: Bins) -> Callable[[np.ndarray], np.ndarray] | None:
    if callable(density):
        return density
    if isinstance(density, str):
        if bins.ndim == 1:
            lo, hi = bins.range[0]
            return _density_1d(density, lo, hi)
        return _density_2d(density, bins)
    return None


def population(bins: Bins, density: Any = "bimodal", n_sub: int = 8) -> np.ndarray:
    """Population fractions per bin from a density (name, callable) or an array."""
    f = _as_density(density, bins)
    if f is None:
        rho = np.asarray(density, dtype=float).ravel()
        if rho.shape != (bins.K,) or np.any(rho < 0) or rho.sum() <= 0:
            raise ValueError("density array must be non-negative with one value per bin.")
        return rho / rho.sum()
    pts, w = _subbin_points(bins, n_sub)
    dens = np.asarray(f(pts), dtype=float).reshape(bins.K, -1)
    rho = (dens @ w) * bins.volumes
    return rho / rho.sum()


def _draw_values(bins: Bins, density: Any, n: int, rng: np.random.Generator):
    """True values (continuous) and labels for n objects."""
    f = _as_density(density, bins)
    if f is None:
        rho = population(bins, density)
        labels = rng.choice(bins.K, size=n, p=rho)
        multi = np.unravel_index(labels, bins.shape)
        vals = [
            e[j] + np.diff(e)[j] * rng.random(n) for e, j in zip(bins.edges, multi, strict=True)
        ]
        values = vals[0] if bins.ndim == 1 else np.stack(vals, axis=1)
        return values, labels
    # fine grid inverse-CDF (piecewise-constant) sampling
    fine = [np.linspace(e[0], e[-1], 20 * (len(e) - 1) + 1) for e in bins.edges]
    fb = Bins.grid(*fine)
    pts = fb.centers
    dens = np.clip(np.asarray(f(pts), dtype=float).ravel(), 0.0, None) * fb.volumes
    cell = rng.choice(fb.K, size=n, p=dens / dens.sum())
    multi = np.unravel_index(cell, fb.shape)
    vals = [e[j] + np.diff(e)[j] * rng.random(n) for e, j in zip(fb.edges, multi, strict=True)]
    values = vals[0] if bins.ndim == 1 else np.stack(vals, axis=1)
    labels = bins.index(values)
    return values, labels


def _noise(noise: Any, n: int, rng: np.random.Generator, shape=()) -> np.ndarray:
    if np.ndim(noise) == 0:
        return np.full((n, *shape), float(noise))
    lo, hi = noise
    return rng.uniform(lo, hi, size=(n, *shape))


def _select(labels, selection, rng):
    if selection is None:
        return np.ones(labels.size, dtype=bool)
    S = np.asarray(selection, dtype=float)
    return rng.random(labels.size) < S[labels]


# ----------------------------------------------------------------------------
# mock catalogues
# ----------------------------------------------------------------------------
def mock_1d(
    n_objects: int = 2000,
    bins: Bins | None = None,
    *,
    density: Any = "bimodal",
    noise: Any = (0.05, 0.2),
    interim_prior: Any = None,
    n_samples: int = 0,
    selection: Any = None,
    outlier_fraction: float = 0.0,
    rng: RNGLike = None,
) -> Mock:
    """A 1-D mock catalogue with heteroscedastic Gaussian measurement errors.

    Parameters
    ----------
    n_objects : int
        Number of *observed* objects (after selection).
    bins : Bins, optional
        Defaults to 30 equal bins on [0, 3].
    density : str, callable, or array_like (K,)
        True population: a named density ("bimodal", "gaussian", "uniform",
        "steep"), a callable density, or bin fractions (values then uniform
        within bins).
    noise : float or (lo, hi)
        Measurement standard deviation, fixed or drawn uniformly per object.
    interim_prior : str, callable or array_like, optional
        If given, also return per-object posteriors on the bins computed under
        this interim prior (``"gaussian"`` gives a broad Gaussian), as
        ``mock.posteriors`` with masses ``mock.interim_prior``.
    n_samples : int
        If positive, also return ``n_samples`` posterior samples per object
        (under the interim prior, flat if not given) as ``mock.samples``.
    selection : array_like (K,), optional
        Per-bin detection probability depending on the *true* bin; objects are
        thinned accordingly and the selection is attached to
        ``mock.likelihoods`` with ``mode="true"``.
    outlier_fraction : float
        Fraction of measurements replaced by uniform draws over the range
        (the likelihoods ignore this: a misspecification test).
    rng : int, Generator or None

    Notes
    -----
    With a named or callable density the true values follow that continuous
    density, whereas the histogram model assumes the population is flat within
    each bin. For coarse bins, measurement errors comparable to the bin width
    and many objects, this *binning bias* can exceed the statistical errors.
    Pass bin fractions (e.g. ``density=population(bins, "bimodal")``) to draw
    values uniformly within bins, for which the model is exactly right.
    """
    gen = as_generator(rng)
    bins = Bins.linear(0.0, 3.0, 30) if bins is None else bins
    if bins.ndim != 1:
        raise ValueError("mock_1d needs one-dimensional bins.")
    rho = population(bins, density)
    values, labels, n_drawn = _draw_selected(bins, density, n_objects, selection, gen)
    sigma = _noise(noise, n_objects, gen)
    obs = values + sigma * gen.standard_normal(n_objects)
    if outlier_fraction > 0:
        lo, hi = bins.range[0]
        bad = gen.random(n_objects) < outlier_fraction
        obs[bad] = gen.uniform(lo, hi, size=int(bad.sum()))
    lk = Likelihoods.from_gaussian(
        obs, sigma, bins, selection=selection, selection_mode=None if selection is None else "true"
    )
    counts = np.bincount(labels, minlength=bins.K).astype(float)
    truth = Truth(rho=rho, labels=labels, values=values, counts=counts, n_drawn=n_drawn)
    mock = Mock(bins, lk, truth, {"mean": obs, "sigma": sigma})
    lo, hi = bins.range[0]
    pi_dens = None
    if interim_prior is not None:
        if isinstance(interim_prior, str) and interim_prior == "gaussian":
            pi_dens = lambda z: stats.norm.pdf(z, 0.4 * lo + 0.6 * hi, 0.3 * (hi - lo))  # noqa: E731
        elif callable(interim_prior):
            pi_dens = interim_prior
        elif isinstance(interim_prior, str):
            pi_dens = _density_1d(interim_prior, lo, hi)
        if pi_dens is not None:
            Pi = population(bins, pi_dens)
        else:
            Pi = population(bins, np.asarray(interim_prior, dtype=float))
        P = lk.toarray() * Pi[None, :]  # row scales are irrelevant after normalising
        mock.posteriors = P / P.sum(axis=1, keepdims=True)
        mock.interim_prior = Pi
    if n_samples > 0:
        mock.samples = _posterior_samples_1d(
            obs, sigma, bins, pi_dens, mock.interim_prior, n_samples, gen
        )
        if mock.interim_prior is None:
            mock.interim_prior = population(bins, "uniform")
    return mock


def _draw_selected(bins, density, n_objects, selection, gen):
    vals, labs = [], []
    n_drawn = 0
    need = n_objects
    while need > 0:
        batch = max(2 * need, 1000)
        v, lab = _draw_values(bins, density, batch, gen)
        n_drawn_batch = batch
        keep = _select(lab, selection, gen) & (lab >= 0)
        v, lab = v[keep], lab[keep]
        if v.shape[0] > need:
            # count only the draws needed to reach n_objects observed objects
            last = np.flatnonzero(keep)[need - 1]
            n_drawn_batch = last + 1
            v, lab = v[:need], lab[:need]
        vals.append(v)
        labs.append(lab)
        n_drawn += n_drawn_batch
        need -= v.shape[0]
    return np.concatenate(vals), np.concatenate(labs), n_drawn


def _posterior_samples_1d(obs, sigma, bins, pi_dens, pi_masses, n_samples, gen):
    lo, hi = bins.range[0]
    grid = np.linspace(lo, hi, 50 * bins.K + 1)
    mid = 0.5 * (grid[1:] + grid[:-1])
    if pi_dens is not None:
        prior = np.asarray(pi_dens(mid), dtype=float)
    elif pi_masses is not None:
        prior = pi_masses[bins.index(mid)] / bins.volumes[bins.index(mid)]
    else:
        prior = np.ones_like(mid)
    out = np.empty((obs.size, n_samples))
    for i in range(obs.size):
        logp = -0.5 * ((mid - obs[i]) / sigma[i]) ** 2 + np.log(np.maximum(prior, 1e-300))
        p = np.exp(logp - logp.max())
        cdf = np.concatenate(([0.0], np.cumsum(p)))
        cdf /= cdf[-1]
        out[i] = np.interp(gen.random(n_samples), cdf, grid)
    return out


def mock_2d(
    n_objects: int = 5000,
    bins: Bins | None = None,
    *,
    density: Any = "banana",
    noise: Any = (0.02, 0.06),
    selection: Any = None,
    rng: RNGLike = None,
) -> Mock:
    """A 2-D mock catalogue (e.g. a colour-magnitude-like ridge) with Gaussian errors.

    ``noise`` gives the per-axis standard deviation (fixed or a ``(lo, hi)``
    range drawn independently per object and axis). Defaults to a 20 x 20 grid
    on the unit square.
    """
    gen = as_generator(rng)
    if bins is None:
        bins = Bins.grid(np.linspace(0, 1, 21), np.linspace(0, 1, 21))
    if bins.ndim != 2:
        raise ValueError("mock_2d needs a two-dimensional grid.")
    rho = population(bins, density)
    values, labels, n_drawn = _draw_selected(bins, density, n_objects, selection, gen)
    sigma = _noise(noise, n_objects, gen, shape=(2,))
    obs = values + sigma * gen.standard_normal(values.shape)
    lk = Likelihoods.from_gaussian(
        obs, sigma, bins, selection=selection, selection_mode=None if selection is None else "true"
    )
    counts = np.bincount(labels, minlength=bins.K).astype(float)
    truth = Truth(rho=rho, labels=labels, values=values, counts=counts, n_drawn=n_drawn)
    return Mock(bins, lk, truth, {"mean": obs, "sigma": sigma})


def mock_categorical(
    n_objects: int = 2000,
    n_classes: int = 4,
    *,
    prevalence: Any = None,
    train_prior: Any = None,
    separation: float = 1.5,
    rng: RNGLike = None,
) -> Mock:
    """A quantification problem: classifier posteriors over unordered classes.

    Each object has a one-dimensional feature drawn from a unit-variance
    Gaussian whose mean depends on its class (means ``separation`` apart).
    An ideal classifier trained with class prior ``train_prior`` (uniform by
    default) reports posteriors ``mock.posteriors``; ``mock.likelihoods`` is
    built with :meth:`Likelihoods.from_classifier`. The test-set prevalence
    ``prevalence`` (default: a skewed vector) differs from the training prior
    (prior shift).
    """
    gen = as_generator(rng)
    C = int(n_classes)
    prev = (
        (np.arange(1, C + 1, dtype=float) ** 2)
        if prevalence is None
        else np.asarray(prevalence, dtype=float)
    )
    prev = prev / prev.sum()
    pi = np.full(C, 1.0 / C) if train_prior is None else np.asarray(train_prior, dtype=float)
    pi = pi / pi.sum()
    labels = gen.choice(C, size=n_objects, p=prev)
    means = separation * np.arange(C)
    x = means[labels] + gen.standard_normal(n_objects)
    loglik = -0.5 * (x[:, None] - means[None, :]) ** 2
    logpost = loglik + np.log(pi)[None, :]
    post = np.exp(logpost - special.logsumexp(logpost, axis=1, keepdims=True))
    bins = Bins.categories([f"class_{c}" for c in range(C)])
    lk = Likelihoods.from_classifier(post, pi, labels=bins.labels)
    counts = np.bincount(labels, minlength=C).astype(float)
    truth = Truth(rho=prev, labels=labels, values=None, counts=counts, n_drawn=n_objects)
    return Mock(bins, lk, truth, {"feature": x}, posteriors=post, interim_prior=pi)


def mock_unfolding(
    n_events: int = 5000,
    n_true: int = 20,
    n_reco: int = 25,
    *,
    slope: float = 3.0,
    resolution: float = 0.08,
    efficiency: tuple[float, float] = (0.6, 0.95),
    rng: RNGLike = None,
) -> Mock:
    """An unfolding problem: an exponential spectrum smeared by a detector.

    True values on [0, 1] follow ``exp(-slope * z)``; events are detected with
    an efficiency rising linearly across the range, smeared with Gaussian
    ``resolution`` and histogrammed in ``n_reco`` bins on [-0.1, 1.1]. The
    response matrix is computed for a population uniform within each true bin,
    and ``mock.likelihoods`` is built with :meth:`Likelihoods.from_response`
    (its efficiency becomes the selection function).
    """
    gen = as_generator(rng)
    tb = Bins.linear(0.0, 1.0, n_true)
    re = np.linspace(-0.1, 1.1, n_reco + 1)
    density = lambda z: np.exp(-slope * np.asarray(z))  # noqa: E731
    rho = population(tb, density)
    eff = lambda z: efficiency[0] + (efficiency[1] - efficiency[0]) * np.asarray(z)  # noqa: E731
    # response R[j, k] = P(reco j, detected | true bin k), population uniform within bin k
    u, wq = np.polynomial.legendre.leggauss(16)
    e = tb.edges[0]
    R = np.zeros((n_reco, n_true))
    for k in range(n_true):
        zk = e[k] + (e[k + 1] - e[k]) * 0.5 * (u + 1.0)
        wk = 0.5 * wq
        cdf = stats.norm.cdf((re[None, :] - zk[:, None]) / resolution)
        R[:, k] = (np.diff(cdf, axis=1) * (wk * eff(zk))[:, None]).sum(axis=0)
    # simulate events
    z, _ = _draw_values(tb, density, n_events, gen)
    det = gen.random(n_events) < eff(z)
    reco = z + resolution * gen.standard_normal(n_events)
    observed = det & (reco >= re[0]) & (reco <= re[-1])
    counts = np.histogram(reco[observed], bins=re)[0].astype(float)
    lk = Likelihoods.from_response(counts, R, bins=tb)
    true_lab = tb.index(z[observed])
    truth = Truth(
        rho=rho,
        labels=true_lab,
        values=z[observed],
        counts=np.bincount(true_lab, minlength=n_true).astype(float),
        n_drawn=n_events,
        extra={"response": R, "reco_edges": re},
    )
    return Mock(tb, lk, truth, {"reco_counts": counts})


# ----------------------------------------------------------------------------
# prior predictive simulator (for SBC)
# ----------------------------------------------------------------------------
def prior_predictive(
    prior: Prior,
    bins: Bins,
    n_objects: int = 200,
    *,
    noise: Any = (0.05, 0.2),
) -> Callable[[np.random.Generator], tuple[Likelihoods, np.ndarray]]:
    """A simulator ``simulate(rng) -> (likelihoods, rho_true)`` drawing populations from a prior.

    Fractions are drawn from ``prior``, objects uniformly within bins from those
    fractions, and Gaussian measurements with the given ``noise`` (in units of
    the bin coordinate). Intended for :func:`binference.validation.sbc` and
    :func:`binference.validation.coverage`. One-dimensional bins only.
    """
    bound = prior.bind(bins)
    if bound.kind != "fractions":
        raise ValueError("prior_predictive supports priors on fractions.")

    def simulate(rng: np.random.Generator) -> tuple[Likelihoods, np.ndarray]:
        rho = bound.sample(1, rng)[0]
        rho = np.maximum(rho, 0.0)
        rho = rho / rho.sum()
        mock = mock_1d(n_objects, bins, density=rho, noise=noise, rng=rng)
        return mock.likelihoods, rho

    return simulate
