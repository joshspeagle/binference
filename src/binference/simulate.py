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
from .density import Density
from .likelihoods import Likelihoods
from .priors import PopulationPrior
from .response import Response
from .selection import Selection, _bin_average

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
    interim: np.ndarray | None = None
    samples: np.ndarray | None = None
    response: Response | None = None

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


def _as_density(density: Any, bins: Bins) -> Density:
    # a Density from a mock density name, a Density, a callable, a distribution or masses
    if isinstance(density, str) and density != "flat":
        if bins.ndim == 1:
            lo, hi = bins.range[0]
            return Density.from_function(_density_1d(density, lo, hi))
        return Density.from_function(_density_2d(density, bins))
    return Density.coerce(density, "mass", name="density")


def population(bins: Bins, density: Any = "bimodal", tol: float = 1e-6) -> np.ndarray:
    """Population fractions per bin from a density name, :class:`Density`, callable or masses.

    Densities are integrated over each bin by adaptive quadrature to ``tol``.
    """
    return _as_density(density, bins).masses(bins, tol=tol)


def _draw_values(bins: Bins, density: Any, n: int, rng: np.random.Generator):
    """True values (continuous) and labels for n objects."""
    f = _as_density(density, bins)._log_pointwise()
    if f is None:
        rho = population(bins, density)
        labels = rng.choice(bins.K, size=n, p=rho)
        return bins.sample_within(labels, rng), labels
    # inverse-CDF sampling on a grid that splits every bin into 20 equal steps in u,
    # so each fine cell lies inside one bin and inherits its mask
    t = np.linspace(0.0, 1.0, 21)[:-1]
    fine = [
        m.inverse(np.append((u[:-1, None] + np.diff(u)[:, None] * t).ravel(), u[-1]))
        for m, u in zip(bins.measures, bins.u_edges, strict=True)
    ]
    fb = Bins.grid(*fine, measure=bins.measures)
    pts = fb.centers
    dens = np.exp(np.asarray(f(pts), dtype=float).ravel()) * fb.volumes
    if bins.is_masked:
        dens[~bins.contains(pts)] = 0.0  # no objects in masked cells
    if not dens.sum() > 0:
        raise ValueError("density has no mass on the active bins.")
    cell = rng.choice(fb.K, size=n, p=dens / dens.sum())
    values = fb.sample_within(cell, rng)
    labels = bins.index(values)
    return values, labels


def _noise(noise: Any, n: int, rng: np.random.Generator, shape=()) -> np.ndarray:
    if np.ndim(noise) == 0:
        return np.full((n, *shape), float(noise))
    lo, hi = noise
    return rng.uniform(lo, hi, size=(n, *shape))


def _measured_pass(z, noise, lo, hi):
    # P(lo <= z + sigma * e <= hi), averaged over the noise distribution
    if np.ndim(noise) == 0:
        sig, wts = np.array([float(noise)]), np.array([1.0])
    else:
        x, w = np.polynomial.legendre.leggauss(16)
        sig = noise[0] + (noise[1] - noise[0]) * 0.5 * (x + 1.0)
        wts = 0.5 * w
    zz = np.asarray(z, dtype=float)[..., None]
    a, b = (lo - zz) / sig, (hi - zz) / sig
    # take the difference in the tail where it keeps its precision
    p = np.where(a > 0, special.ndtr(-a) - special.ndtr(-b), special.ndtr(b) - special.ndtr(a))
    return p @ wts


# ----------------------------------------------------------------------------
# mock catalogues
# ----------------------------------------------------------------------------
def mock_1d(
    n_objects: int = 2000,
    bins: Bins | None = None,
    *,
    density: Any = "bimodal",
    noise: Any = (0.05, 0.2),
    interim: Any = None,
    n_samples: int = 0,
    selection: Any = None,
    measured_range: tuple[float | None, float | None] | None = None,
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
    interim : str, callable or array_like, optional
        If given, also return per-object posteriors on the bins computed under
        this interim prior (``"gaussian"`` gives a broad Gaussian), as
        ``mock.posteriors`` with masses ``mock.interim``.
    n_samples : int
        If positive, also return ``n_samples`` posterior samples per object
        (under the interim prior, flat if not given) as ``mock.samples``.
    selection : array_like (K,), optional
        Per-bin detection probability depending on the *true* bin (selection on
        the truth): objects are thinned accordingly.
    measured_range : (lo, hi), optional
        Selection on the measurement: only objects whose measured value lies in
        ``[lo, hi]`` are observed (either end may be ``None``).
    outlier_fraction : float
        Fraction of measurements replaced by uniform draws over the range
        (the likelihoods ignore this: a misspecification test).
    rng : int, Generator or None

    Notes
    -----
    The matching :class:`Selection` is attached to ``mock.likelihoods``:
    ``"on_truth"`` with ``selection``, ``"on_measurement"`` with
    ``measured_range`` (the population-weighted bin average of the probability
    that the measured value passes, over the noise distribution), and
    :meth:`Selection.combined` with both (DECISIONS #48).

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
    mock = _gaussian_mock(
        bins, density, n_objects, noise, selection, measured_range, outlier_fraction, gen
    )
    lk, obs, sigma = mock.likelihoods, mock.measurements["mean"], mock.measurements["sigma"]
    log_pi = None
    if interim is not None:
        if isinstance(interim, str) and interim == "gaussian":
            lo, hi = bins.range[0]
            interim = stats.norm(0.4 * lo + 0.6 * hi, 0.3 * (hi - lo))
        interim = _as_density(interim, bins)
        log_pi = interim._log_pointwise()
        Pi = interim.masses(bins, tol=1e-6)
        P = lk.toarray() * Pi[None, :]  # row scales are irrelevant after normalising
        mock.posteriors = P / P.sum(axis=1, keepdims=True)
        mock.interim = Pi
    if n_samples > 0:
        mock.samples = _posterior_samples_1d(
            obs, sigma, bins, log_pi, mock.interim, n_samples, gen
        )
        if mock.interim is None:
            mock.interim = population(bins, "uniform")
    return mock


def _gaussian_mock(bins, density, n_objects, noise, on_truth, measured_range, outliers, gen):
    # draw objects until n_objects pass the selection, measure them, and build the mock
    # with the Selection that describes how they were selected (DECISIONS #48)
    sel = None if on_truth is None else Selection(on_truth, "on_truth")
    if measured_range is not None:
        if len(measured_range) != 2:
            raise ValueError("measured_range must be (lo, hi); either may be None.")
        lo = -np.inf if measured_range[0] is None else float(measured_range[0])
        hi = np.inf if measured_range[1] is None else float(measured_range[1])
        if not lo < hi:
            raise ValueError(f"measured_range must have lo < hi; got {measured_range}.")
        # S_k: the population-weighted bin average of the probability of passing the cut
        log_pop = _as_density(density, bins)._log_pointwise()
        if log_pop is None:  # flat in the measure coordinate within bins
            log_pop = lambda z: bins.measures[0].log_jacobian(z)  # noqa: E731
        passes = _bin_average(lambda z: _measured_pass(z, noise, lo, hi), bins, log_weight=log_pop)
        sel = (
            Selection(passes[0], "on_measurement")
            if on_truth is None
            else Selection.combined(on_truth, passes[0])
        )
    S = None if sel is None else sel.row  # thinning on the true bin
    shape = () if bins.ndim == 1 else (bins.ndim,)
    parts: list[tuple[np.ndarray, ...]] = []
    n_drawn, need = 0, n_objects
    while need > 0:
        batch = max(2 * need, 1000)
        v, lab = _draw_values(bins, density, batch, gen)
        sigma = _noise(noise, batch, gen, shape)
        obs = v + sigma * gen.standard_normal(v.shape)
        if outliers > 0:  # misspecification the likelihoods and the selection ignore
            bad = gen.random(batch) < outliers
            obs[bad] = gen.uniform(*bins.range[0], size=int(bad.sum()))
        keep = lab >= 0
        if S is not None:
            keep &= gen.random(batch) < S[np.maximum(lab, 0)]
        if measured_range is not None:
            keep &= (obs >= lo) & (obs <= hi)
        idx = np.flatnonzero(keep)[:need]
        # count only the draws needed to reach n_objects observed objects
        n_drawn += batch if idx.size < need else int(idx[-1]) + 1
        parts.append((v[idx], lab[idx], sigma[idx], obs[idx]))
        need -= idx.size
    values, labels, sigma, obs = (np.concatenate(a) for a in zip(*parts, strict=True))
    lk = Likelihoods.from_gaussian(
        obs,
        sigma,
        bins,
        selection=sel,
        warn_outside=False,  # observations near the edges spill past them by construction
    )
    counts = np.bincount(labels, minlength=bins.K).astype(float)
    rho = population(bins, density)
    truth = Truth(rho=rho, labels=labels, values=values, counts=counts, n_drawn=n_drawn)
    return Mock(bins, lk, truth, {"mean": obs, "sigma": sigma})


def _posterior_samples_1d(obs, sigma, bins, log_pi, pi_masses, n_samples, gen):
    e = bins.edges[0]  # 50 steps per bin, so each step lies inside one bin
    grid = np.append(
        (e[:-1, None] + np.diff(e)[:, None] * np.linspace(0, 1, 51)[:-1]).ravel(), e[-1]
    )
    mid = 0.5 * (grid[1:] + grid[:-1])
    k = bins.index(mid)
    with np.errstate(divide="ignore"):
        if log_pi is not None:
            log_prior = np.asarray(log_pi(mid), dtype=float)
        elif pi_masses is not None:
            log_prior = np.log(pi_masses[k] / bins.volumes[k])
        else:
            log_prior = np.zeros_like(mid)
    log_prior = np.where(k >= 0, log_prior, -np.inf)  # no mass in masked bins
    out = np.empty((obs.size, n_samples))
    for i in range(obs.size):
        logp = -0.5 * ((mid - obs[i]) / sigma[i]) ** 2 + log_prior
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
    return _gaussian_mock(bins, density, n_objects, noise, selection, None, 0.0, gen)


def mock_categorical(
    n_objects: int = 2000,
    n_classes: int = 4,
    *,
    prevalence: Any = None,
    train_prior: Any = None,
    separation: float = 1.5,
    temperature: float = 1.0,
    n_validation: int = 500,
    rng: RNGLike = None,
) -> Mock:
    """A quantification problem: classifier outputs over unordered classes.

    Each object has a one-dimensional feature drawn from a unit-variance
    Gaussian whose mean depends on its class (means ``separation`` apart). A
    classifier trained with class prior ``train_prior`` (uniform by default)
    reports posteriors ``mock.posteriors``, calibrated when ``temperature`` is
    one (its log-likelihoods are divided by ``temperature``);
    ``mock.likelihoods`` divides them by the training prior (route A,
    :meth:`Likelihoods.from_table`). ``mock.response`` is tallied with
    :meth:`Response.from_labelled` from a validation set of ``n_validation``
    objects drawn with the training prior, scored by their feature; with
    ``mock.measurements["feature"]`` it gives route C
    (:meth:`Likelihoods.from_binned`). The test-set prevalence ``prevalence``
    (default: a skewed vector) differs from the training prior (label shift).
    """
    if not temperature > 0:
        raise ValueError(f"temperature must be positive; got {temperature}.")
    if int(n_validation) < 1:
        raise ValueError(f"n_validation must be at least 1; got {n_validation}.")
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
    means = separation * np.arange(C)
    labels = gen.choice(C, size=n_objects, p=prev)
    x = means[labels] + gen.standard_normal(n_objects)
    val_labels = gen.choice(C, size=int(n_validation), p=pi)
    val_x = means[val_labels] + gen.standard_normal(val_labels.size)
    logpost = -0.5 * (x[:, None] - means[None, :]) ** 2 / temperature + np.log(pi)[None, :]
    post = np.exp(logpost - special.logsumexp(logpost, axis=1, keepdims=True))
    bins = Bins.categories([f"class_{c}" for c in range(C)])
    lk = Likelihoods.from_table(post, bins, interim=pi)
    counts = np.bincount(labels, minlength=C).astype(float)
    truth = Truth(rho=prev, labels=labels, values=None, counts=counts, n_drawn=n_objects)
    missing = np.setdiff1d(np.arange(C), val_labels)
    if missing.size:
        raise ValueError(
            f"The validation set has no objects of class(es) {missing.tolist()}; raise "
            "n_validation or the training prior of those classes."
        )
    response = Response.from_labelled(
        val_x, np.asarray(bins.labels, dtype=object)[val_labels], classes=bins.labels
    )
    return Mock(bins, lk, truth, {"feature": x}, posteriors=post, interim=pi, response=response)


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

    ``n_events`` counts the events *before* detection, so the number observed
    is random (``mock.truth.counts.sum()``), unlike the ``n_objects`` of the
    other mocks. True values on [0, 1] follow ``exp(-slope * z)``; events are detected with
    an efficiency rising linearly across the range, smeared with Gaussian
    ``resolution`` and histogrammed in ``n_reco`` bins on [-0.1, 1.1]. The
    response matrix is computed for a population uniform within each true bin,
    ``mock.response`` is a :class:`Response` and ``mock.likelihoods`` is built
    with :meth:`Likelihoods.from_binned` (its efficiency becomes the selection
    function).
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
    response = Response(R, tb, Bins(re))
    lk = Likelihoods.from_binned(response, values=reco[observed])
    counts = np.bincount(lk.meta["measured_bin"], weights=lk.weights, minlength=n_reco)
    true_lab = tb.index(z[observed])
    truth = Truth(
        rho=rho,
        labels=true_lab,
        values=z[observed],
        counts=np.bincount(true_lab, minlength=n_true).astype(float),
        n_drawn=n_events,
    )
    return Mock(tb, lk, truth, {"reco_counts": counts}, response=response)


# ----------------------------------------------------------------------------
# prior predictive simulator (for SBC)
# ----------------------------------------------------------------------------
def prior_predictive(
    prior: PopulationPrior,
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
