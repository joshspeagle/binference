"""Result objects returned by every inference engine."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import special

from . import _kernels as kern
from . import diagnostics as diag
from ._utils import RNGLike, as_generator, chunk_slices
from .bins import Bins
from .warnings import ConvergenceWarning

if TYPE_CHECKING:  # pragma: no cover
    from .likelihoods import Likelihoods
    from .model import PopulationModel

__all__ = ["PointEstimate", "PopulationResult", "Summary"]


# ----------------------------------------------------------------------------
# Summary table
# ----------------------------------------------------------------------------
def _small_concentration(model: Any) -> bool:
    """Whether a conjugate prior's density is unbounded at the boundary (long log tails)."""
    prior = getattr(model, "prior", None)
    return bool(prior is not None and prior.conjugate and prior.unbounded_density())


class Summary:
    """A small per-bin summary table with a readable ``repr``.

    Columns are available as attributes and via :meth:`to_dict` /
    :meth:`to_pandas`.
    """

    def __init__(self, columns: dict[str, np.ndarray], title: str = "") -> None:
        self.columns = columns
        self.title = title

    def __getitem__(self, key: str) -> np.ndarray:
        return self.columns[key]

    def __getattr__(self, key: str) -> np.ndarray:
        try:
            return self.__dict__["columns"][key]
        except KeyError:
            raise AttributeError(key) from None

    def to_dict(self) -> dict[str, np.ndarray]:
        return dict(self.columns)

    def to_pandas(self):
        import pandas as pd

        return pd.DataFrame(self.columns)

    def __len__(self) -> int:
        return len(next(iter(self.columns.values())))

    def __repr__(self) -> str:
        names = list(self.columns)
        rows = []
        for i in range(len(self)):
            row = []
            for n in names:
                v = self.columns[n][i]
                if isinstance(v, float | np.floating):
                    row.append(f"{v:.4g}" if np.isfinite(v) else "nan")
                else:
                    row.append(str(v))
            rows.append(row)
        widths = [max(len(n), *(len(r[j]) for r in rows)) for j, n in enumerate(names)]
        lines = [self.title] if self.title else []
        lines.append("  ".join(n.rjust(w) for n, w in zip(names, widths, strict=True)))
        max_rows = 40
        shown = rows if len(rows) <= max_rows else [*rows[:20], ["..."] * len(names), *rows[-19:]]
        lines.extend("  ".join(c.rjust(w) for c, w in zip(r, widths, strict=True)) for r in shown)
        return "\n".join(lines)

    _repr_html_ = None


def _bin_labels(bins: Bins) -> np.ndarray:
    if bins.is_categorical:
        return np.array([str(lab) for lab in bins.labels], dtype=object)
    if bins.ndim == 1:
        e = bins.edges[0]
        return np.array([f"[{e[k]:.4g}, {e[k + 1]:.4g})" for k in range(bins.K)], dtype=object)
    idx = np.indices(bins.shape).reshape(bins.ndim, -1).T
    return np.array([str(tuple(int(i) for i in row)) for row in idx], dtype=object)


def _hdi(x: np.ndarray, level: float) -> tuple[float, float]:
    s = np.sort(x)
    n = s.size
    m = int(np.ceil(level * n))
    if m >= n:
        return float(s[0]), float(s[-1])
    widths = s[m - 1 :] - s[: n - m + 1]
    j = int(np.argmin(widths))
    return float(s[j]), float(s[j + m - 1])


# ----------------------------------------------------------------------------
# shared individual-object helpers
# ----------------------------------------------------------------------------
class _IndividualMixin:
    """Methods for per-object outputs that need the likelihoods (via the model)."""

    model: PopulationModel | None

    def _require_model(self) -> PopulationModel:
        if self.model is None:
            raise RuntimeError(
                "This result is not attached to its likelihoods; call result.attach(likelihoods)."
            )
        return self.model

    def _rho_draws(self) -> np.ndarray:  # (n, K) fractions
        raise NotImplementedError

    def _select_draws(self, n_draws: int | None, rng: RNGLike = None) -> np.ndarray:
        R = self._rho_draws()
        if n_draws is None or n_draws >= R.shape[0]:
            return R
        idx = np.linspace(0, R.shape[0] - 1, int(n_draws)).round().astype(int)
        return R[idx]

    def assignment_probabilities(
        self, idx: Any = None, n_draws: int | None = 500, chunk: int = 4096
    ) -> np.ndarray:
        r"""Posterior probability that each object lies in each bin, :math:`P(z_i = k \mid D)`.

        Averages the per-object conditional posteriors
        :math:`r_{ik}(\rho)` over (up to ``n_draws``) population draws: the
        "hierarchical shrinkage" posteriors of the individual objects.

        Returns an ``(n, K)`` array for the objects ``idx`` (all by default).
        """
        model = self._require_model()
        R = self._select_draws(n_draws)
        L = model._L if idx is None else model._L[np.atleast_1d(np.arange(model.N)[idx])]
        n = L.shape[0]
        out = np.zeros((n, model.K))
        for sl in chunk_slices(n, chunk):
            block = L[sl]
            block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
            acc = np.zeros((block.shape[0], model.K))
            for rho in R:
                P = block * rho
                tot = P.sum(axis=1, keepdims=True)
                acc += P / np.where(tot > 0, tot, 1.0)  # rows with no support stay zero
            out[sl] = acc / R.shape[0]
        return out

    def sample_assignments(
        self, idx: Any = None, n_draws: int | None = None, rng: RNGLike = None
    ) -> np.ndarray:
        """Joint draws of the bin labels of objects ``idx`` (all by default).

        For each of (up to ``n_draws``) population draws :math:`\\rho^{(s)}`, labels are
        drawn from :math:`z_i \\sim \\mathrm{Cat}(r_i(\\rho^{(s)}))`; together these
        are exact joint draws from :math:`p(\\rho, z \\mid D)`. Returns an
        ``(n_draws, n_objects)`` integer array.
        """
        model = self._require_model()
        gen = as_generator(rng)
        R = self._select_draws(n_draws)
        L = model._L if idx is None else model._L[np.atleast_1d(np.arange(model.N)[idx])]
        return np.stack([kern.sample_labels(L, rho, gen) for rho in R])

    def realized_counts(self, n_draws: int | None = None, rng: RNGLike = None) -> np.ndarray:
        """Draws of the realised histogram of the observed objects' true bins.

        Unlike the population fractions (which include the shot noise of
        drawing ``N`` objects from a population), this is the distribution of
        ``n_k = sum_i w_i 1[z_i = k]`` for *these* objects. Returns
        ``(n_draws, K)`` weighted counts.
        """
        model = self._require_model()
        gen = as_generator(rng)
        R = self._select_draws(n_draws)
        w = model._w
        unit = bool(np.all(w == 1))
        integer = bool(np.all(np.abs(w - np.round(w)) < 1e-12))
        out = np.empty((R.shape[0], model.K))
        for s, rho in enumerate(R):
            if unit or integer:
                out[s] = kern.sample_counts(model._L, w, rho, gen, unit_weights=unit)[0]
            else:
                labels = kern.sample_labels(model._L, rho, gen)
                out[s] = np.bincount(labels, weights=w, minlength=model.K)
        return out

    def sample_values(
        self,
        idx: Any = None,
        n_draws: int | None = None,
        rng: RNGLike = None,
        loglike: Callable[[np.ndarray, int], np.ndarray] | None = None,
        n_sub: int = 64,
    ) -> np.ndarray:
        """Joint draws of the continuous true values of objects ``idx``.

        Labels are drawn as in :meth:`sample_assignments`. Within the drawn bin
        the population is flat, so a true value follows the object's own
        likelihood there:

        * exactly (truncated Gaussians) for likelihoods built with
          :meth:`Likelihoods.from_gaussian`;
        * by inverse-CDF sampling on ``n_sub`` sub-points per bin (1-D bins)
          if ``loglike(z, i)`` is given, returning the log-likelihood of object
          ``i`` (an index into the likelihoods) at the points ``z``;
        * otherwise uniformly within the bin, the top-hat approximation, which
          is accurate only when likelihoods are nearly flat within bins.

        With weighted (compressed) rows each row is treated as one object.
        Returns ``(n_draws, n_objects)`` (1-D) or ``(n_draws, n_objects, ndim)``.
        """
        model = self._require_model()
        bins = model.bins
        if bins.is_categorical:
            raise ValueError("Categorical bins have no continuous values; use sample_assignments.")
        gen = as_generator(rng)
        rows = np.atleast_1d(np.arange(model.N)[slice(None) if idx is None else idx])
        labels = self.sample_assignments(idx=rows, n_draws=n_draws, rng=gen)
        multi = np.unravel_index(labels, bins.shape)
        meta = model.likelihoods.meta
        if "gaussian_mean" in meta:
            from scipy import stats

            mu = np.asarray(meta["gaussian_mean"])[rows]  # (n, ndim)
            sig = np.asarray(meta["gaussian_sigma"])[rows]
            vals = []
            for d, e in enumerate(bins.edges):
                j = multi[d]
                m_, s_ = mu[None, :, d], sig[None, :, d]
                a, b = (e[j] - m_) / s_, (e[j + 1] - m_) / s_
                vals.append(stats.truncnorm.rvs(a, b, loc=m_, scale=s_, random_state=gen))
            return vals[0] if bins.ndim == 1 else np.stack(vals, axis=-1)
        if loglike is not None:
            if bins.ndim != 1:
                raise ValueError("loglike-based sampling supports 1-D bins only.")
            e = bins.edges[0]
            u = (np.arange(n_sub) + 0.5) / n_sub
            out = np.empty(labels.shape)
            for c, i in enumerate(rows):
                for k in np.unique(labels[:, c]):
                    sel = labels[:, c] == k
                    z = e[k] + (e[k + 1] - e[k]) * u
                    lw = np.asarray(loglike(z, int(i)), dtype=float).ravel()
                    w = np.exp(lw - np.max(lw))
                    cell = gen.choice(n_sub, size=int(sel.sum()), p=w / w.sum())
                    width = (e[k + 1] - e[k]) / n_sub
                    out[sel, c] = z[cell] + width * (gen.random(cell.size) - 0.5)
            return out
        vals = []
        for d, e in enumerate(bins.edges):
            j = multi[d]
            lo, hi = e[j], e[j + 1]
            vals.append(lo + (hi - lo) * gen.random(labels.shape))
        return vals[0] if bins.ndim == 1 else np.stack(vals, axis=-1)

    def predict_proba(self, likelihoods: Likelihoods | Any, n_draws: int | None = 500):
        """Posterior over bins for *new* objects under the inferred population."""
        from .likelihoods import Likelihoods

        model = self._require_model()
        lk = (
            likelihoods
            if isinstance(likelihoods, Likelihoods)
            else Likelihoods(likelihoods, model.bins)
        )
        if lk.K != model.K:
            raise ValueError("New likelihoods must use the same bins.")
        R = self._select_draws(n_draws)
        L = model.effective_likelihoods(lk)
        L = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
        acc = np.zeros_like(L)
        for rho in R:
            P = L * rho
            tot = P.sum(axis=1, keepdims=True)
            acc += P / np.where(tot > 0, tot, 1.0)
        return acc / R.shape[0]

    def log_predictive(self, likelihoods: Likelihoods | Any, n_draws: int | None = 500):
        r"""Log posterior-predictive density of new objects, :math:`\log E[L_i\cdot\rho]`.

        Includes the per-object log-scales of the likelihoods, so values can be
        compared across priors and binnings (e.g. for cross-validation).
        """
        from .likelihoods import Likelihoods

        model = self._require_model()
        lk = (
            likelihoods
            if isinstance(likelihoods, Likelihoods)
            else Likelihoods(likelihoods, model.bins)
        )
        R = self._select_draws(n_draws)  # fractions (rates rescale out of the ratio)
        S = np.asarray(model.effective_likelihoods(lk) @ R.T)  # (N_new, n_draws)
        if model.selection is not None:  # a new *detected* object: divide by S . rho
            S = S / (R @ model.selection)[None, :]
        with np.errstate(divide="ignore"):
            lp = special.logsumexp(np.log(S), axis=1) - np.log(R.shape[0])
        return lp + lk.log_scale


# ----------------------------------------------------------------------------
# PopulationResult
# ----------------------------------------------------------------------------
class PopulationResult(_IndividualMixin):
    """Posterior draws of a population (fractions and, for rate models, rates).

    Every inference engine returns this object (``MAP`` returns a
    :class:`PointEstimate`). Draws are stored per chain; flattened views are
    available as :attr:`samples` (fractions) and :attr:`rates`.

    Attributes
    ----------
    method : str
        Engine that produced the draws.
    approximate : bool
        True for approximations (VB, uncorrected Laplace, bootstrap).
    info : dict
        Engine statistics (e.g. NUTS tree depths, divergences, step sizes).
    """

    def __init__(
        self,
        model: PopulationModel | None,
        draws: np.ndarray,
        *,
        method: str,
        kind: str = "fractions",
        bins: Bins | None = None,
        x: np.ndarray | None = None,
        log_prob: np.ndarray | None = None,
        approximate: bool = False,
        info: dict[str, Any] | None = None,
        labels: np.ndarray | None = None,
        label_index: np.ndarray | None = None,
        prior_repr: str | None = None,
    ) -> None:
        d = np.asarray(draws, dtype=float)
        if d.ndim == 2:
            d = d[None]
        if d.ndim != 3:
            raise ValueError("draws must have shape (n_chains, n_draws, K) or (n_draws, K).")
        self.model = model
        self._draws = d
        self.kind = kind
        self.bins = bins if bins is not None else (model.bins if model is not None else None)
        if self.bins is None:
            self.bins = Bins.categories(d.shape[-1])
        self.x = x
        self.log_prob = log_prob
        self.method = method
        self.approximate = approximate
        self.info = dict(info or {})
        self.labels = labels
        self.label_index = label_index
        self.prior_repr = prior_repr or (repr(model.prior_spec) if model is not None else None)
        self._cache: dict[str, Any] = {}

    # ------------------------------------------------------------- creation
    @classmethod
    def from_samples(
        cls,
        model: PopulationModel,
        samples: Any,
        *,
        method: str = "external",
        approximate: bool = False,
        info: dict[str, Any] | None = None,
    ) -> PopulationResult:
        """Wrap draws from an external sampler (fractions or rates, per chain or flat)."""
        return cls(
            model,
            np.asarray(samples, dtype=float),
            method=method,
            kind=model.kind,
            approximate=approximate,
            info=info,
        )

    def attach(self, likelihoods_or_model: Any) -> PopulationResult:
        """Attach likelihoods (or a model) to a loaded result to enable per-object methods."""
        from .model import PopulationModel

        m = likelihoods_or_model
        if not isinstance(m, PopulationModel):
            m = PopulationModel(m)
        if m.K != self.bins.K:
            raise ValueError("Likelihoods have a different number of bins.")
        self.model = m
        return self

    # ------------------------------------------------------------ properties
    @property
    def n_chains(self) -> int:
        return int(self._draws.shape[0])

    @property
    def n_draws(self) -> int:
        """Total number of draws (over all chains)."""
        return int(self._draws.shape[0] * self._draws.shape[1])

    @property
    def K(self) -> int:
        return int(self._draws.shape[-1])

    @property
    def chains(self) -> np.ndarray:
        """Fraction draws per chain, shape ``(n_chains, n_per_chain, K)``."""
        if self.kind == "rates":
            return self._draws / self._draws.sum(axis=-1, keepdims=True)
        return self._draws

    @property
    def samples(self) -> np.ndarray:
        """Population fractions, shape ``(n_draws, K)`` (chains concatenated)."""
        return self.chains.reshape(-1, self.K)

    @property
    def rates(self) -> np.ndarray | None:
        """Rate draws ``(n_draws, K)`` for rate models, else ``None``."""
        if self.kind != "rates":
            return None
        return self._draws.reshape(-1, self.K)

    def _rho_draws(self) -> np.ndarray:
        return self.samples

    def _which(self, which: str) -> np.ndarray:
        if which == "fractions":
            return self.samples
        if which == "rates":
            if self.rates is None:
                raise ValueError("This result has no rates (fit a GammaRates prior).")
            return self.rates
        if which == "density":
            return self.density()
        raise ValueError("which must be 'fractions', 'rates' or 'density'.")

    @property
    def mean(self) -> np.ndarray:
        """Posterior mean fractions ``(K,)``."""
        return self.samples.mean(axis=0)

    @property
    def median(self) -> np.ndarray:
        """Posterior median fractions ``(K,)``."""
        return np.median(self.samples, axis=0)

    @property
    def std(self) -> np.ndarray:
        """Posterior standard deviation of the fractions ``(K,)``."""
        return self.samples.std(axis=0, ddof=1)

    def cov(self) -> np.ndarray:
        """Posterior covariance of the fractions ``(K, K)``."""
        return np.cov(self.samples, rowvar=False)

    def corr(self) -> np.ndarray:
        """Posterior correlation matrix of the fractions ``(K, K)``."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.corrcoef(self.samples, rowvar=False)

    def quantile(self, q: Any, which: str = "fractions") -> np.ndarray:
        """Posterior quantiles, shape ``(*np.shape(q), K)``."""
        return np.quantile(self._which(which), q, axis=0)

    def interval(
        self, level: float = 0.68, kind: str = "central", which: str = "fractions"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-bin credible intervals ``(lo, hi)``, each ``(K,)``.

        ``kind="central"`` gives equal-tailed intervals, ``"hdi"`` highest
        density intervals.
        """
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1).")
        X = self._which(which)
        if kind == "central":
            lo, hi = np.quantile(X, [0.5 - level / 2, 0.5 + level / 2], axis=0)
            return lo, hi
        if kind == "hdi":
            bounds = np.array([_hdi(X[:, k], level) for k in range(X.shape[1])])
            return bounds[:, 0], bounds[:, 1]
        raise ValueError("kind must be 'central' or 'hdi'.")

    def density(self) -> np.ndarray:
        """Draws of the population density per unit bin volume, ``(n_draws, K)``.

        Fractions divided by bin volumes (rates divided by volumes for rate
        models): the quantity to plot as n(z).
        """
        base = self.rates if self.kind == "rates" else self.samples
        return base / self.bins.volumes

    def functional(self, func: Callable[[np.ndarray], Any], vectorized: bool = False):
        """Draws of a derived quantity ``func(rho)`` (e.g. the mean redshift).

        With ``vectorized=True``, ``func`` receives the full ``(n_draws, K)``
        array at once.

        Examples
        --------
        >>> zbar = result.functional(lambda r: r @ result.bins.centers)  # doctest: +SKIP
        """
        X = self.samples
        if vectorized:
            return np.asarray(func(X))
        return np.array([func(r) for r in X])

    def rebin(self, groups: Any) -> np.ndarray:
        """Fractions aggregated into coarser groups.

        ``groups`` is either a length-``K`` integer array mapping bins to groups
        (e.g. from :meth:`Bins.coarsen`) or a sequence of lists of bin indices.
        Returns ``(n_draws, n_groups)``.
        """
        X = self.samples
        try:
            arr = np.asarray(groups)
        except ValueError:  # ragged sequence of groups
            arr = None
        if (
            arr is not None
            and arr.ndim == 1
            and arr.shape[0] == self.K
            and np.issubdtype(arr.dtype, np.integer)
        ):
            n_groups = int(arr.max()) + 1
            return np.stack([X[:, arr == g].sum(axis=1) for g in range(n_groups)], axis=1)
        return np.stack([X[:, list(grp)].sum(axis=1) for grp in groups], axis=1)

    # ---------------------------------------------------------- diagnostics
    @property
    def diagnostics(self) -> dict[str, Any]:
        """Convergence diagnostics (computed lazily and cached)."""
        if "diagnostics" in self._cache:
            return self._cache["diagnostics"]
        out: dict[str, Any] = {}
        # for rate models diagnose the rates, which also carry the total
        C = self._draws if self.kind == "rates" else self.chains
        if C.shape[1] >= 8:
            out["ess_bulk"] = np.array([diag.ess(C[:, :, k], "bulk") for k in range(self.K)])
            out["ess_tail"] = np.array([diag.ess(C[:, :, k], "tail") for k in range(self.K)])
            if self.n_chains >= 2:
                out["rhat"] = np.array([diag.rhat(C[:, :, k]) for k in range(self.K)])
        for key in (
            "n_divergent",
            "k_hat",
            "kkt_gap",
            "max_depth_fraction",
            "missing_information",
            "converged",
            "ebfmi",
        ):
            if key in self.info:
                out[key] = self.info[key]
        self._cache["diagnostics"] = out
        return out

    def check(self, strict: bool = False, rhat_max: float = 1.01, ess_min: float = 100.0):
        """Check diagnostics and warn (or raise with ``strict=True``) about problems.

        Returns a list of messages (empty if everything looks fine).
        """
        d = self.diagnostics
        msgs = []
        hints = []
        if "rhat" in d and np.nanmax(d["rhat"]) > rhat_max:
            msgs.append(f"max R-hat = {np.nanmax(d['rhat']):.3f} > {rhat_max}")
        if "ess_bulk" in d:
            per_chain = np.nanmin(d["ess_bulk"]) / max(1, self.n_chains)
            if self.method in ("nuts", "gibbs") and per_chain < ess_min:
                msgs.append(
                    f"min bulk ESS = {np.nanmin(d['ess_bulk']):.0f} "
                    f"({per_chain:.0f} per chain < {ess_min:.0f})"
                )
        if msgs and self.method == "gibbs":
            hints.append(
                "Gibbs mixes slowly when likelihoods are broad compared with the bins; "
                "run longer or use method='nuts'"
            )
        if d.get("n_divergent", 0):
            msgs.append(f"{d['n_divergent']} divergent transitions")
            hints.append("try a higher target_accept, e.g. 0.95")
        if d.get("k_hat", 0) > 0.7:
            msgs.append(f"Pareto k-hat = {d['k_hat']:.2f} > 0.7 (importance sampling unreliable)")
        if d.get("max_depth_fraction", 0) > 0.05:
            msgs.append(
                f"{100 * d['max_depth_fraction']:.0f}% of NUTS iterations hit the maximum tree "
                "depth"
            )
            if _small_concentration(self.model):
                hints.append(
                    "this prior gives nearly empty bins very long tails in log space "
                    "(e.g. concentrations below one); method='gibbs' samples such "
                    "conjugate priors efficiently"
                )
        if msgs:
            text = "; ".join(msgs)
            if hints:
                text += ". Hint: " + "; ".join(hints)
            if strict:
                raise RuntimeError(f"Diagnostics failed: {text}")
            warnings.warn(f"Diagnostics: {text}.", ConvergenceWarning, stacklevel=2)
        return msgs

    def summary(self, level: float = 0.68, which: str = "fractions") -> Summary:
        """Per-bin posterior summary table (mean, sd, median, interval, ESS, R-hat)."""
        X = self._which(which)
        lo, hi = np.quantile(X, [0.5 - level / 2, 0.5 + level / 2], axis=0)
        cols: dict[str, np.ndarray] = {
            "bin": _bin_labels(self.bins),
            "mean": X.mean(axis=0),
            "sd": X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.zeros(self.K),
            "median": np.median(X, axis=0),
            f"q{100 * (0.5 - level / 2):g}": lo,
            f"q{100 * (0.5 + level / 2):g}": hi,
        }
        d = self.diagnostics
        if "ess_bulk" in d:
            cols["ess_bulk"] = d["ess_bulk"]
        if "rhat" in d:
            cols["rhat"] = d["rhat"]
        flag = " [approximate]" if self.approximate else ""
        title = f"pastry {which} posterior ({self.method}{flag}, {self.n_draws} draws)"
        return Summary(cols, title)

    def __repr__(self) -> str:
        flag = ", approximate" if self.approximate else ""
        return (
            f"PopulationResult(method={self.method!r}, kind={self.kind!r}, K={self.K}, "
            f"n_chains={self.n_chains}, n_draws={self.n_draws}{flag})"
        )

    # ------------------------------------------------------------- plotting
    def plot(self, **kwargs: Any):
        """Plot the population posterior (see :func:`pastry.plotting.plot_population`)."""
        from .plotting import plot_population

        return plot_population(self, **kwargs)

    def plot_corner(self, idx: Sequence[int] | None = None, **kwargs: Any):
        """Corner plot of selected bins (see :func:`pastry.plotting.plot_corner`)."""
        from .plotting import plot_corner

        return plot_corner(self, idx=idx, **kwargs)

    def plot_trace(self, idx: Sequence[int] | None = None, **kwargs: Any):
        """Trace plots of selected bins (see :func:`pastry.plotting.plot_trace`)."""
        from .plotting import plot_trace

        return plot_trace(self, idx=idx, **kwargs)

    # ---------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        """Save draws, metadata and bins to a compressed ``.npz`` file."""
        from .io import save_result

        save_result(self, path)

    @classmethod
    def load(cls, path: str) -> PopulationResult:
        """Load a result saved with :meth:`save`.

        Re-:meth:`attach` the likelihoods to use the per-object methods.
        """
        from .io import load_result

        return load_result(path)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dictionary representation (arrays and metadata)."""
        from .io import result_to_dict

        return result_to_dict(self)

    def to_datatree(self):
        """Convert to an :class:`xarray.DataTree` (ArviZ >= 1.0 layout; needs xarray)."""
        from .io import to_datatree

        return to_datatree(self)


# ----------------------------------------------------------------------------
# PointEstimate
# ----------------------------------------------------------------------------
class PointEstimate(_IndividualMixin):
    """A MAP / maximum-likelihood estimate of the population.

    Attributes
    ----------
    value : (K,) array
        Fractions (or rates for rate models) at the optimum.
    x : (dim,) array or None
        Unconstrained coordinates (when optimised there).
    kkt_gap : float
        Optimality certificate (zero at the exact optimum on the simplex).
    """

    def __init__(
        self,
        model: PopulationModel | None,
        value: np.ndarray,
        *,
        method: str = "map",
        x: np.ndarray | None = None,
        log_posterior: float = np.nan,
        log_likelihood: float = np.nan,
        kkt_gap: float = np.nan,
        n_iter: int = 0,
        converged: bool = True,
        info: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.value = np.asarray(value, dtype=float)
        self.method = method
        self.x = x
        self.log_posterior = float(log_posterior)
        self.log_likelihood = float(log_likelihood)
        self.kkt_gap = float(kkt_gap)
        self.n_iter = int(n_iter)
        self.converged = bool(converged)
        self.info = dict(info or {})
        self.kind = model.kind if model is not None else "fractions"
        self.bins = model.bins if model is not None else Bins.categories(self.value.size)

    @property
    def rho(self) -> np.ndarray:
        """Population fractions at the optimum."""
        return self.value / self.value.sum() if self.kind == "rates" else self.value

    @property
    def rates(self) -> np.ndarray | None:
        """Rates at the optimum for rate models, else ``None``."""
        return self.value if self.kind == "rates" else None

    def _rho_draws(self) -> np.ndarray:
        return self.rho[None, :]

    def density(self) -> np.ndarray:
        """Population density per unit bin volume at the optimum."""
        base = self.value if self.kind == "rates" else self.rho
        return base / self.bins.volumes

    def laplace(self, n_samples: int = 4000, rng: RNGLike = None, **kwargs: Any):
        """Laplace approximation around the mode (see :class:`pastry.samplers.Laplace`)."""
        from .samplers import Laplace

        return Laplace(self._require_model(), rng=rng, **kwargs).run(n_samples)

    def __repr__(self) -> str:
        return (
            f"PointEstimate(method={self.method!r}, K={self.value.size}, "
            f"log_posterior={self.log_posterior:.6g}, kkt_gap={self.kkt_gap:.2e}, "
            f"converged={self.converged})"
        )

    def plot(self, **kwargs: Any):
        from .plotting import plot_population

        return plot_population(self, **kwargs)
