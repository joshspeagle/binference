"""Result objects returned by every inference engine."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import special

from . import _kernels as kern
from . import diagnostics as diag
from ._utils import RNGLike, as_generator, chunk_slices, warn
from .bins import Bins
from .measure import Measure
from .warnings import ConvergenceWarning, DiagnosticsError, InputWarning

if TYPE_CHECKING:  # pragma: no cover
    from .basis import Basis
    from .likelihoods import Likelihoods
    from .model import PopulationModel
    from .selection import Selection

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
        j = bins.cell_index[0]
        return np.array([f"[{e[k]:.4g}, {e[k + 1]:.4g})" for k in j], dtype=object)
    idx = np.stack(bins.cell_index, axis=1)
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
    """Methods for per-object outputs that need the likelihoods (via the model).

    With a :class:`~binference.Basis` the model's columns are components: these
    methods work on the component weights and report per bin, sharing each
    component among the bins by its mass :math:`A_{jk}` (exact for flat
    components; for hats the likelihood's variation within a component is
    ignored in that share).
    """

    model: PopulationModel | None
    basis: Basis | None
    bins: Bins

    def _require_model(self) -> PopulationModel:
        if self.model is None:
            raise RuntimeError(
                "This result is not attached to its likelihoods; call result.attach(likelihoods)."
            )
        return self.model

    def _rho_draws(self) -> np.ndarray:  # (n, K) fractions in the bins
        raise NotImplementedError

    def _weight_draws(self) -> np.ndarray:  # (n, C) fractions of the model's components
        raise NotImplementedError

    def _select_draws(self, n_draws: int | None, components: bool = False) -> np.ndarray:
        R = self._weight_draws() if components else self._rho_draws()
        return R[_draw_index(R.shape[0], n_draws)]

    def _inferred_selection(self) -> np.ndarray | None:
        """Per-draw selection ``(chain, draw, K)`` when a selection from injections or a
        tallied response (its efficiencies) is inferred with the population."""
        info = getattr(self, "info", {})
        if "response_draws" in info:
            return np.asarray(info["response_draws"]).sum(axis=-2)
        return info.get("selection_draws")

    def _object_draws(self, n_draws: int | None):
        """The likelihoods and per-draw column weights for per-object methods.

        With a selection inferred jointly (DECISIONS #79) these are the bare
        likelihoods and :math:`\\rho\\circ S` with each draw's own :math:`S`, which is
        also returned; with a response inferred jointly (DECISIONS #81), each draw's
        rows ``(n_draws, N, K)`` and :math:`\\rho`; otherwise the model's likelihoods
        (selection on the truth applied) and :math:`\\rho`, and ``None``.
        """
        model = self._require_model()
        R = self._select_draws(n_draws, components=True)
        info = getattr(self, "info", {})
        if "response_draws" in info:
            Rd = _flat_draws(info["response_draws"], 2, n_draws)
            return Rd[:, model.likelihoods.meta["measured_bin"]], R, None
        S = info.get("selection_draws")
        if S is not None:
            S = _flat_draws(S, 1, n_draws)
        return *model._rows(R, S), S

    def _to_bins(self, P: np.ndarray) -> np.ndarray:
        """Per-component values ``(..., C)`` summed into the bins ``(..., K)``."""
        return P if self.basis is None else self.basis.to_bins(P)

    def _labels_to_bins(self, labels: np.ndarray, gen: np.random.Generator) -> np.ndarray:
        """Component labels as bin labels, each drawn from its component's shares."""
        if self.basis is None or self.basis.is_identity:
            return labels
        S = self.basis.shares  # rows sum to one: invert their cumulative sums
        cum = np.cumsum(S.data)
        flat = labels.ravel()
        first, last = S.indptr[flat], S.indptr[flat + 1] - 1
        start = np.concatenate(([0.0], cum))[first]
        pos = np.searchsorted(cum, start + gen.random(flat.size), side="right")
        return S.indices[np.clip(pos, first, last)].reshape(labels.shape)

    def _counts_to_bins(self, counts: np.ndarray, gen: np.random.Generator) -> np.ndarray:
        """Integer counts per component as counts per bin, shared multinomially."""
        if self.basis is None or self.basis.is_identity:
            return counts
        S = self.basis.shares
        n = kern._multinomial_csr(S, np.round(counts).astype(np.int64), np.ones(self.bins.K), gen)
        return np.bincount(S.indices, weights=n, minlength=self.bins.K)

    def _new_likelihoods(self, likelihoods: Any) -> Likelihoods:
        from .likelihoods import Likelihoods

        model = self._require_model()
        basis = model.likelihoods.basis
        lk = (
            likelihoods
            if isinstance(likelihoods, Likelihoods)
            else Likelihoods(likelihoods, basis or model.bins)
        )
        if lk.K != model.K or lk.basis != basis or (basis is None and lk.bins != model.bins):
            raise ValueError("New likelihoods must use the same bins (and basis) as the fit.")
        return lk

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
        L, R, _ = self._object_draws(n_draws)
        L = _objects(L, model.N, idx)
        n = L.shape[-2]
        out = np.zeros((n, model.K))
        for sl in chunk_slices(n, chunk):
            block = L[:, sl] if L.ndim == 3 else L[sl]
            block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
            acc = np.zeros((block.shape[-2], model.K))
            for s, rho in enumerate(R):
                P = _draw_rows(block, s) * rho
                tot = P.sum(axis=1, keepdims=True)
                acc += P / np.where(tot > 0, tot, 1.0)  # rows with no support stay zero
            out[sl] = acc / R.shape[0]
        return self._to_bins(out)

    def sample_assignments(
        self, idx: Any = None, n_draws: int | None = None, rng: RNGLike = None
    ) -> np.ndarray:
        """Joint draws of the bin labels of objects ``idx`` (all by default).

        For each of (up to ``n_draws``) population draws :math:`\\rho^{(s)}`, labels are
        drawn from :math:`z_i \\sim \\mathrm{Cat}(r_i(\\rho^{(s)}))`; together these
        are exact joint draws from :math:`p(\\rho, z \\mid D)`. Returns an
        ``(n_draws, n_objects)`` integer array.
        """
        gen = as_generator(rng)
        return self._labels_to_bins(self._component_labels(idx, n_draws, gen), gen)

    def _component_labels(self, idx: Any, n_draws: int | None, gen: np.random.Generator):
        model = self._require_model()
        L, R, _ = self._object_draws(n_draws)
        L = _objects(L, model.N, idx)
        return np.stack(
            [kern.sample_labels(_draw_rows(L, s), rho, gen) for s, rho in enumerate(R)]
        )

    def realized_counts(self, n_draws: int | None = None, rng: RNGLike = None) -> np.ndarray:
        """Draws of the realised histogram of the observed objects' true bins.

        Unlike the population fractions (which include the shot noise of
        drawing ``N`` objects from a population), this is the distribution of
        ``n_k = sum_i w_i 1[z_i = k]`` for *these* objects. Returns
        ``(n_draws, K)`` weighted counts.
        """
        model = self._require_model()
        gen = as_generator(rng)
        L, R, _ = self._object_draws(n_draws)
        w = model._w
        unit = bool(np.all(w == 1))
        integer = bool(np.all(np.abs(w - np.round(w)) < 1e-12))
        out = np.empty((R.shape[0], self.bins.K))
        for s, rho in enumerate(R):
            Ls = _draw_rows(L, s)
            if unit or integer:
                n = kern.sample_counts(Ls, w, rho, gen, unit_weights=unit)[0]
                out[s] = self._counts_to_bins(n, gen)
            else:
                labels = self._labels_to_bins(kern.sample_labels(Ls, rho, gen), gen)
                out[s] = np.bincount(labels, weights=w, minlength=self.bins.K)
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
        the population is flat in the measure coordinate ``u`` (its density in
        ``z`` carries :math:`T'(z)`), so a true value follows the object's own
        likelihood times that density:

        * exactly (truncated Gaussians) on linear axes for likelihoods built
          with :meth:`Likelihoods.from_gaussian` and independent errors
          (``sigma``), and by inverse-CDF sampling on ``n_sub`` sub-points per
          bin on other axes; correlated errors (``cov``) take the top-hat route;
        * by inverse-CDF sampling on ``n_sub`` sub-points per bin (1-D bins)
          if ``loglike(z, i)`` is given, returning the log-likelihood of object
          ``i`` (an index into the likelihoods) at the points ``z``;
        * otherwise flat in ``u`` within the bin, the top-hat approximation,
          which is accurate only when likelihoods are nearly flat within bins.

        With a :class:`~binference.Basis` the label is a component: flat
        components are sampled as above on their cells; inside a hat the
        population is the hat, so sub-point weights include its height, and
        other draws on its support are kept with that height.

        With weighted (compressed) rows each row is treated as one object.
        Returns ``(n_draws, n_objects)`` (1-D) or ``(n_draws, n_objects, ndim)``.
        """
        model = self._require_model()
        bins = model.bins  # the components' cells with a basis
        if bins.is_categorical:
            raise ValueError("Categorical bins have no continuous values; use sample_assignments.")
        if loglike is not None and bins.ndim != 1:
            raise ValueError("loglike-based sampling supports 1-D bins only.")
        gen = as_generator(rng)
        rows = np.atleast_1d(np.arange(model.N)[slice(None) if idx is None else idx])
        labels = self._component_labels(rows, n_draws, gen)
        basis = model.likelihoods.basis
        hat = basis is not None and basis.order == 1
        objects = np.broadcast_to(rows, labels.shape)
        out = np.empty((*labels.shape, bins.ndim))
        todo = np.ones(labels.shape, dtype=bool)
        while todo.any():  # a single pass unless hats reject a proposal
            lab = labels[todo]
            if hat:
                lo, hi = basis.support(lab)
            else:
                cells = list(zip(bins.u_edges, bins.grid_index(lab), strict=True))
                lo = np.stack([u[j] for u, j in cells], axis=-1)
                hi = np.stack([u[j + 1] for u, j in cells], axis=-1)
            height = basis.height if hat else None
            z, ok = _propose_values(model, objects[todo], lab, lo, hi, gen, loglike, n_sub, height)
            at = np.flatnonzero(todo)[ok]
            out.reshape(-1, bins.ndim)[at] = z[ok]
            todo.ravel()[at] = False
        return out[..., 0] if bins.ndim == 1 else out

    def predict_proba(self, likelihoods: Likelihoods | Any, n_draws: int | None = 500):
        """Posterior over bins for *new* objects under the inferred population."""
        L, R, _, _ = self._new_object_draws(self._new_likelihoods(likelihoods), n_draws)
        L = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
        acc = np.zeros(L.shape[-2:])
        for s, rho in enumerate(R):
            P = _draw_rows(L, s) * rho
            tot = P.sum(axis=1, keepdims=True)
            acc += P / np.where(tot > 0, tot, 1.0)
        return self._to_bins(acc / R.shape[0])

    def log_predictive(self, likelihoods: Likelihoods | Any, n_draws: int | None = 500):
        r"""Log posterior-predictive density of new objects, :math:`\log E[L_i\cdot\rho]`.

        Includes the per-object log-scales of the likelihoods, so values can be
        compared across priors and binnings (e.g. for cross-validation).
        """
        # rates rescale out of the ratio
        L, R, norm, log_scale = self._new_object_draws(self._new_likelihoods(likelihoods), n_draws)
        S = np.einsum("snk,sk->ns", L, R) if L.ndim == 3 else np.asarray(L @ R.T)  # (N, n)
        if norm is not None:  # a new *detected* object: divide by S . rho
            S = S / norm[None, :]
        with np.errstate(divide="ignore"):
            lp = special.logsumexp(np.log(S), axis=1) - np.log(R.shape[0])
        return lp + log_scale

    def _new_object_draws(self, lk: Likelihoods, n_draws: int | None):
        """For new objects ``lk``: rows and column weights per draw as in
        :meth:`_object_draws`, each draw's :math:`S\\cdot\\rho` (``None`` without a
        selection) and the rows' log-scales. New binned objects with the response
        inferred in this fit take its draws (DECISIONS #81)."""
        model = self._require_model()
        info = getattr(self, "info", {})
        if "response_draws" in info and lk.meta.get("response") is model.response:
            R = self._select_draws(n_draws, components=True)
            Rd = _flat_draws(info["response_draws"], 2, n_draws)
            rows = Rd[:, lk.meta["measured_bin"]]  # the response's own rows: scale one
            return rows, R, np.sum(Rd.sum(axis=1) * R, axis=1), np.zeros(lk.N)
        _, R, S = self._object_draws(n_draws)
        L = model.effective_likelihoods(lk) if S is None else lk.values
        if model.selection is None:
            return L, R, None, lk.log_scale
        return L, R, (R.sum(axis=1) if S is not None else R @ model.selection), lk.log_scale


def _objects(L: Any, N: int, idx: Any) -> Any:
    # the rows of objects idx (all by default), of per-draw rows (n_draws, N, K) too
    if idx is None:
        return L
    rows = np.atleast_1d(np.arange(N)[idx])
    return L[:, rows] if L.ndim == 3 else L[rows]


def _draw_rows(L: Any, s: int) -> Any:
    # draw s's rows when they differ by draw (an inferred response), else the rows
    return L[s] if L.ndim == 3 else L


def _flat_draws(A: Any, n_tail: int, n_draws: int | None) -> np.ndarray:
    # per-draw arrays (chain, draw, *tail) as (n, *tail), up to n_draws spread evenly
    A = np.asarray(A)
    A = A.reshape(-1, *A.shape[A.ndim - n_tail :])
    return A[_draw_index(len(A), n_draws)]


def _which_name(which: str) -> str:
    """``"fractions"``, ``"rates"`` or ``"density"``, singular spellings accepted (O-12)."""
    name = {"fraction": "fractions", "rate": "rates"}.get(which, which)
    if name not in ("fractions", "rates", "density"):
        raise ValueError(f"which must be 'fractions', 'rates' or 'density'; got {which!r}.")
    return name


def _mixing(C: np.ndarray, tail: bool = False) -> dict[str, np.ndarray]:
    """Bulk (and tail) ESS and split R-hat of each column of draws ``(chain, draw, K)``."""
    cols = range(C.shape[-1])
    out = {"ess_bulk": np.array([diag.ess(C[:, :, k], "bulk") for k in cols])}
    if tail:
        out["ess_tail"] = np.array([diag.ess(C[:, :, k], "tail") for k in cols])
    out["rhat"] = np.array([diag.rhat(C[:, :, k]) for k in cols])
    return out


def _draw_index(n_total: int, n_draws: int | None) -> np.ndarray | slice:
    # up to n_draws draws spread evenly over all of them
    if n_draws is None or n_draws >= n_total:
        return slice(None)
    return np.linspace(0, n_total - 1, int(n_draws)).round().astype(int)


def _draws_rng(draws: np.ndarray) -> np.random.Generator:
    # a stream seeded by a hash of the draws: reproducible, different for different draws,
    # and independent of any generator a caller holds (DECISIONS #78)
    digest = hashlib.sha256(np.ascontiguousarray(draws).tobytes()).digest()
    return np.random.default_rng(np.frombuffer(digest, dtype=np.uint32))


def _same_layout(a: Basis | None, b: Basis | None) -> bool:
    # equal bases up to the identity of their measures, which do not survive save and load
    if a is None or b is None:
        return a is b
    return (a.order, a.resolution, a.oversample, a.K, a.C) == (
        b.order,
        b.resolution,
        b.oversample,
        b.K,
        b.C,
    ) and all(np.allclose(x, y) for x, y in zip(a.bins.edges, b.bins.edges, strict=True))


def _propose_values(model, objects, labels, lo, hi, gen, loglike, n_sub, height):
    """Values in the boxes ``[lo, hi]`` (in ``u``, ``(n, ndim)``), and whether to keep them.

    Inside a box the population is flat in ``u`` times ``height(labels, d, u)``
    per axis (a hat; ``None`` for flat), so its density in ``z`` carries
    :math:`T'(z)`. Each axis is drawn from the object's likelihood times that
    density: a truncated Gaussian on linear axes (kept with the hat's height),
    else inverse-CDF sampling on ``n_sub`` sub-points per box, with ``loglike``
    or the Gaussian as the likelihood, or the population alone without either.
    """
    meta = model.likelihoods.meta
    gauss = "gaussian_mean" in meta
    n, nd = lo.shape
    z, ok = np.empty((n, nd)), np.ones(n, dtype=bool)
    t = (np.arange(n_sub) + 0.5) / n_sub
    for d, m in enumerate(model.bins.measures):
        zlo, zhi = m.inverse(lo[:, d]), m.inverse(hi[:, d])
        if gauss:
            mu = np.asarray(meta["gaussian_mean"])[objects, d]
            sig = np.asarray(meta["gaussian_sigma"])[objects, d]
        if gauss and m == Measure.linear():
            from scipy import stats

            a, b = (zlo - mu) / sig, (zhi - mu) / sig
            z[:, d] = stats.truncnorm.rvs(a, b, loc=mu, scale=sig, random_state=gen)
            if height is not None:
                ok &= gen.random(n) < height(labels, d, z[:, d])
            continue
        zz = zlo[:, None] + (zhi - zlo)[:, None] * t  # (n, n_sub) sub-points in z
        with np.errstate(divide="ignore"):
            logw = m.log_jacobian(zz) + (
                0.0 if height is None else np.log(height(labels, d, m.forward(zz)))
            )
        if gauss:
            logw = logw - 0.5 * ((zz - mu[:, None]) / sig[:, None]) ** 2
        elif loglike is not None:  # one call per object and box
            key = np.stack([objects, labels], axis=1)
            pairs, inverse = np.unique(key, axis=0, return_inverse=True)
            order = np.argsort(inverse.ravel(), kind="stable")
            for i, members in zip(
                pairs[:, 0],
                np.split(order, np.cumsum(np.bincount(inverse.ravel()))[:-1]),
                strict=True,
            ):
                logw[members] += np.asarray(loglike(zz[members[0]], int(i)), dtype=float).ravel()
        w = np.exp(logw - np.max(logw, axis=1, keepdims=True))
        c = np.cumsum(w, axis=1)
        cell = np.sum(c < gen.random(n)[:, None] * c[:, -1:], axis=1)
        z[:, d] = zz[np.arange(n), cell] + (zhi - zlo) / n_sub * (gen.random(n) - 0.5)
    return z, ok


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
        log_density_unconstrained: np.ndarray | None = None,
        approximate: bool = False,
        info: dict[str, Any] | None = None,
        labels: np.ndarray | None = None,
        label_index: np.ndarray | None = None,
        prior_repr: str | None = None,
        basis: Basis | None = None,
        total: np.ndarray | None = None,
        selection: Selection | None = None,
    ) -> None:
        d = np.asarray(draws, dtype=float)
        if d.ndim == 2:
            d = d[None]
        if d.ndim != 3:
            raise ValueError("draws must have shape (n_chains, n_draws, K) or (n_draws, K).")
        self.model = model
        if basis is None and model is not None:
            basis = model.likelihoods.basis
        #: the :class:`~binference.Basis` of the model's components, or ``None``
        self.basis = basis
        self._params = d  # the model's columns: component weights with a basis
        self._draws = self._to_bins(d)  # per bin (the same array without a basis)
        self.kind = kind
        if kind == "rates":
            total = self._draws.sum(axis=-1)
        # for fractions, drawn on first use (see total)
        self._total = (
            None if total is None else np.asarray(total, dtype=float).reshape(d.shape[:2])
        )
        self.bins = (
            bins
            if bins is not None
            else basis.bins
            if basis is not None
            else model.bins
            if model is not None
            else Bins.categories(d.shape[-1])
        )
        self.x = x
        #: log posterior density of each draw in the unconstrained coordinates
        #: (with the Jacobian; DECISIONS #15), for samplers that compute it
        self.log_density_unconstrained = log_density_unconstrained
        self.method = method
        self.approximate = approximate
        self.info = dict(info or {})
        self.labels = labels
        self.label_index = label_index
        self.prior_repr = prior_repr or (repr(model.prior_spec) if model is not None else None)
        # the fit's selection, kept for attach after a load (audit O-1)
        self._selection = selection if model is None else model.selection_spec
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

    @property
    def selection(self) -> Selection | None:
        """The selection function of the fit, kept through :meth:`save` and :func:`load`."""
        return self._selection if self.model is None else self.model.selection_spec

    def attach(self, likelihoods_or_model: Any) -> PopulationResult:
        """Attach likelihoods (or a model) to a loaded result to enable per-object methods.

        Bare likelihoods take the fit's selection; likelihoods or a model with another
        selection, or on other bins, are refused (audit O-1, O-2).
        """
        from .likelihoods import Likelihoods
        from .model import PopulationModel

        m, saved = likelihoods_or_model, self.selection
        if not isinstance(m, PopulationModel):
            lk = m if isinstance(m, Likelihoods) else Likelihoods(m, self.basis or self.bins)
            given = lk.selection is not None
            m = (
                PopulationModel(lk)
                if saved is None or given
                else PopulationModel(lk, selection=saved)
            )
        if saved is not None and m.selection_spec != saved:
            raise ValueError("The likelihoods (or model) carry another selection than this fit.")
        if not _same_layout(m.likelihoods.basis, self.basis):
            raise ValueError("The likelihoods use a different basis than this result.")
        if self.basis is None and m.bins != self.bins:
            raise ValueError(f"The likelihoods are on other bins ({m.bins}) than this result.")
        self.model = m
        for key in [k for k in self._cache if isinstance(k, tuple) and k[0] == "predictive_p"]:
            del self._cache[key]  # computed from the previous likelihoods
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
    def total(self) -> np.ndarray | None:
        r"""Draws of the total rate :math:`\Lambda` ``(n_draws,)``, or ``None``.

        For fractions, each draw is :math:`\Lambda \sim \mathrm{Gamma}(W, S\cdot\rho)`
        given its :math:`\rho`: the exact posterior of the total under the prior
        :math:`p(\Lambda) \propto 1/\Lambda`, with :math:`W` the number of objects
        and :math:`S` the detection probabilities or exposures (one without a
        selection). ``None`` when the selection is only relative (DECISIONS #9,
        #46, #78) or the result was loaded without its total. For rate models,
        the sum of the rates.

        The Gamma variates come from a stream seeded by the fraction draws
        themselves, so the totals are reproducible for any result (also wrapped
        external draws, and loaded results after :meth:`attach`), independent
        across different draws, and leave the caller's generator untouched.
        """
        if self._total is None and self.model is not None:
            S = self._inferred_selection()  # per draw, when inferred with the population
            self._total = self.model.draw_total(self._params, _draws_rng(self._params), S)
        return None if self._total is None else self._total.ravel()

    @property
    def rates(self) -> np.ndarray | None:
        r"""Rate draws ``(n_draws, K)``: :math:`\lambda = \Lambda\rho` per draw for
        fractions (see :attr:`total`), the sampled rates for rate models; ``None``
        when the total is not available."""
        if self.kind == "rates":
            return self._draws.reshape(-1, self.K)
        total = self.total
        return None if total is None else total[:, None] * self.samples

    def _rho_draws(self) -> np.ndarray:
        return self.samples

    @property
    def component_draws(self) -> np.ndarray:
        """Draws of the model's parameters, ``(n_chains, n_per_chain, C)``: the component
        weights with a :class:`~binference.Basis`, else the same as the per-bin draws."""
        return self._params

    def _weight_draws(self) -> np.ndarray:
        W = self._params.reshape(-1, self._params.shape[-1])
        return W / W.sum(axis=-1, keepdims=True) if self.kind == "rates" else W

    def _which(self, which: str) -> np.ndarray:
        which = _which_name(which)
        if which == "fractions":
            return self.samples
        if which == "rates":
            rates = self.rates
            if rates is None:
                raise ValueError(
                    "This result has no rates: its selection is only relative, so the total "
                    "is not identified (DECISIONS #46), or it has no model (attach the "
                    "likelihoods)."
                )
            return rates
        return self.density()

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
        if C.shape[1] >= 8:  # split R-hat also for one chain (audit O-7)
            out.update(_mixing(C, tail=True))
            Sd = self._inferred_selection()  # a selection, or a response's efficiencies
            if Sd is not None:
                out.update({f"{k}_selection": v for k, v in _mixing(Sd).items()})
        for key in (
            "n_divergent",
            "k_hat",
            "k_hat_threshold",
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

    def _missing_information(self) -> float | None:
        """Fraction of missing information at the posterior mean (cached in ``info``)."""
        if (
            "missing_information" not in self.info
            and self.model is not None
            and self.model.K <= 500
        ):
            value = self._params.reshape(-1, self.model.K).mean(axis=0)
            try:
                self.info["missing_information"] = self.model.missing_information(value)
            except (np.linalg.LinAlgError, ValueError):
                return None
            self._cache.pop("diagnostics", None)
        return self.info.get("missing_information")

    def check(
        self,
        strict: bool = False,
        rhat_max: float = 1.01,
        ess_min: float = 100.0,
        predictive_p_min: float = 1e-3,
        predictive_draws: int = 200,
    ):
        """Check diagnostics and warn (or raise with ``strict=True``) about problems.

        Sampler problems raise a :class:`~binference.ConvergenceWarning`; problems
        with the inputs (Monte Carlo noise of sample likelihoods, binned counts
        the response does not describe) an :class:`~binference.InputWarning`.
        For binned inputs with more measured than truth bins, the counts are
        tested with :meth:`Response.predictive_pvalue` over ``predictive_draws``
        draws, and flagged below ``predictive_p_min``.

        Returns a list of messages (empty if everything looks fine).
        """
        d = self.diagnostics
        msgs = []
        hints = []
        # Markov chains (any engine but the approximations), whose draws are correlated
        mcmc = not self.approximate and self.method not in ("laplace", "vb", "bootstrap")
        for suffix, what in (("", ""), ("_selection", " of the selection")):
            if "rhat" + suffix in d and np.nanmax(d["rhat" + suffix]) > rhat_max:
                msgs.append(f"max R-hat{what} = {np.nanmax(d['rhat' + suffix]):.3f} > {rhat_max}")
            for kind in ("bulk", "tail"):
                key = f"ess_{kind}{suffix}"
                if key in d and mcmc:
                    ess = np.nanmin(d[key])
                    per_chain = ess / max(1, self.n_chains)
                    if per_chain < ess_min:
                        msgs.append(
                            f"min {kind} ESS{what} = {ess:.0f} "
                            f"({per_chain:.0f} per chain < {ess_min:.0f})"
                        )
        if msgs and self.method == "gibbs":
            hint = "Gibbs mixes slowly when likelihoods are broad compared with the bins"
            f = self._missing_information()
            if f is not None:
                tau = (1.0 + f) / max(1.0 - f, 1e-12)
                tau_text = f"~{tau:.0f}" if tau < 1000 else "over 1000"
                hint += f" (missing information {f:.2f}: autocorrelation time {tau_text})"
            hints.append(hint + "; run longer or use method='nuts'")
        if d.get("n_divergent", 0):
            n_div = int(d["n_divergent"])
            share = 100.0 * n_div / max(self.n_draws, 1)
            msgs.append(f"{n_div} divergent transitions ({share:.2g}% of draws)")
            hints.append(
                "try a higher target_accept, e.g. 0.95; divergences that persist mark a "
                "region the sampler cannot explore (see the pitfalls guide)"
            )
        if "ebfmi" in d and np.nanmin(d["ebfmi"]) < 0.3:
            msgs.append(f"E-BFMI = {np.nanmin(d['ebfmi']):.2f} < 0.3 in some chain")
            hints.append(
                "NUTS explores the energy distribution poorly (heavy tails or a funnel); "
                "a longer warm-up or another prior may help"
            )
        if d.get("converged") is False:
            msgs.append("the optimiser did not converge")
        threshold = d.get("k_hat_threshold", 0.7)
        if d.get("k_hat", 0) >= threshold:
            msgs.append(
                f"Pareto k-hat = {d['k_hat']:.2f} >= {threshold:.2f} (importance sampling "
                "unreliable)"
            )
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
        lk = self.model.likelihoods if self.model is not None else None
        inputs, input_hints = [], []
        if lk is not None and "sample_sum" in lk.meta:
            # the population the objects are drawn from, after selection
            rho = self._weight_draws().mean(axis=0)
            if self.model.selection_row is not None:
                rho = rho * self.model.selection_row
            var, _ = lk.sample_variance(rho)
            total = float(np.sum(lk.weights * var))
            if total > 1.0:
                inputs.append(
                    f"likelihoods from samples carry Monte Carlo variance sum_i V_i = {total:.2g} "
                    "> 1 at the fitted population"
                )
                input_hints.append("draw more samples per object, or use wider bins or floor=")
        if lk is not None and "measured_bin" in lk.meta:
            ppp = self._predictive_pvalue(lk, predictive_draws)
            if ppp is not None and ppp < predictive_p_min:
                inputs.append(
                    "the measured counts disagree with the fitted population and the response "
                    f"(posterior predictive p = {ppp:.1g})"
                )
                input_hints.append(
                    "the response may not describe these data; for a classifier, the score "
                    "distribution of a class differs between the validation and the target set "
                    "(label shift violated)"
                )
        texts = []
        for found, why, category in (
            (msgs, hints, ConvergenceWarning),
            (inputs, input_hints, InputWarning),
        ):
            if found:
                text = "; ".join(found) + (". Hint: " + "; ".join(why) if why else "")
                texts.append(text)
                if not strict:
                    warn(f"Diagnostics: {text}.", category)
        if strict and texts:
            raise DiagnosticsError(f"Diagnostics failed: {'; '.join(texts)}")
        return msgs + inputs

    def _predictive_pvalue(self, lk: Any, n_draws: int) -> float | None:
        # label-shift check of binned inputs (DECISIONS #68); needs J > K
        key = ("predictive_p", int(n_draws))
        if key not in self._cache:
            resp = lk.meta["response"]
            J, K = resp.matrix.shape
            n = np.bincount(lk.meta["measured_bin"], weights=lk.weights, minlength=J)
            Rd = self.info.get("response_draws")  # an inferred response, drawn with rho
            if Rd is not None:
                Rd = _flat_draws(Rd, 2, n_draws)
            self._cache[key] = (
                resp.predictive_pvalue(n, self._select_draws(n_draws), Rd) if J > K else None
            )
        return self._cache[key]

    def summary(self, level: float = 0.68, which: str = "fractions") -> Summary:
        """Per-bin posterior summary table (mean, sd, median, interval, ESS, R-hat)."""
        which = _which_name(which)
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
        if self._draws.shape[1] >= 8:  # of the quantity summarised (audit O-8)
            cols.update(_mixing(X.reshape(self.n_chains, -1, self.K)))
        flag = " [approximate]" if self.approximate else ""
        source = None if self.model is None else self.model.likelihoods.meta.get("source")
        inputs = "" if source is None else f"; inputs: {source}"
        title = f"binference {which} posterior ({self.method}{flag}, {self.n_draws} draws{inputs})"
        return Summary(cols, title)

    def __repr__(self) -> str:
        flag = ", approximate" if self.approximate else ""
        return (
            f"PopulationResult(method={self.method!r}, kind={self.kind!r}, K={self.K}, "
            f"n_chains={self.n_chains}, n_draws={self.n_draws}{flag})"
        )

    # ------------------------------------------------------------- plotting
    def plot(self, **kwargs: Any):
        """Plot the population posterior (see :func:`binference.plotting.plot_population`)."""
        from .plotting import plot_population

        return plot_population(self, **kwargs)

    def plot_corner(self, idx: Sequence[int] | None = None, **kwargs: Any):
        """Corner plot of selected bins (see :func:`binference.plotting.plot_corner`)."""
        from .plotting import plot_corner

        return plot_corner(self, idx=idx, **kwargs)

    def plot_trace(self, idx: Sequence[int] | None = None, **kwargs: Any):
        """Trace plots of selected bins (see :func:`binference.plotting.plot_trace`)."""
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
        Fractions (or rates for rate models) at the optimum; with a
        :class:`~binference.Basis`, the component weights (:attr:`rho` and
        :attr:`rates` are per bin).
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
        self.basis = model.likelihoods.basis if model is not None else None
        self.bins = (
            self.basis.bins
            if self.basis is not None
            else model.bins
            if model is not None
            else Bins.categories(self.value.size)
        )

    @property
    def rho(self) -> np.ndarray:
        """Population fractions at the optimum (per bin)."""
        v = self._to_bins(self.value)
        return v / v.sum() if self.kind == "rates" else v

    @property
    def rates(self) -> np.ndarray | None:
        """Rates at the optimum (per bin): for fractions :math:`\\hat\\Lambda\\rho` with
        :math:`\\hat\\Lambda = W / (S\\cdot\\rho)` (``None`` with a relative selection)."""
        if self.kind == "rates":
            return self._to_bins(self.value)
        total = self.total
        return None if total is None else total * self.rho

    @property
    def total(self) -> float | None:
        """Total rate at the optimum (see :attr:`rates`)."""
        if self.kind == "rates":
            return float(self.value.sum())
        if self.model is None:
            return None
        S = self._inferred_selection()  # inferred, at the mode
        S = self.model.total_exposure if S is None else np.ravel(S)
        return None if S is None else float(self.model.W / (S @ self.value))

    def _rho_draws(self) -> np.ndarray:
        return self.rho[None, :]

    def _weight_draws(self) -> np.ndarray:
        v = self.value
        return (v / v.sum() if self.kind == "rates" else v)[None, :]

    def density(self) -> np.ndarray:
        """Population density per unit bin volume at the optimum."""
        base = self.rates if self.kind == "rates" else self.rho
        return base / self.bins.volumes

    def laplace(self, n_samples: int = 4000, rng: RNGLike = None, **kwargs: Any):
        """Laplace approximation around the mode (see :class:`binference.samplers.Laplace`)."""
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
