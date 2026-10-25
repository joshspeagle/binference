"""Per-object likelihoods over bins: the main input to binference."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from scipy import special

from ._quadrature import bin_log_average
from ._utils import check_positive_weights, frozen, is_integer_array, warn
from .bins import Bins
from .density import (
    Density,
    Representation,
    _dist_log_masses,
    _dist_logpdf,
    _equal_volumes,
    _has_cdf,
    _has_pdf,
    _is_distribution,
    _log_nonneg,
    log_interval_masses,
)
from .measure import Measure
from .response import Response
from .selection import Selection, as_selection
from .warnings import InputWarning

__all__ = ["Likelihoods"]

InterimPrior = Any  # anything Density.coerce accepts


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _csr_row_max(L: sp.csr_array) -> np.ndarray:
    n = L.shape[0]
    out = np.zeros(n)
    counts = np.diff(L.indptr)
    nz = counts > 0
    if L.data.size:
        starts = L.indptr[:-1][nz]
        out[nz] = np.maximum.reduceat(L.data, starts)
    return out


def _scale_csr_rows(L: sp.csr_array, factors: np.ndarray) -> sp.csr_array:
    L = L.copy()
    L.data = L.data * np.repeat(factors, np.diff(L.indptr))
    return L


def _scale_columns(values, factors: np.ndarray):
    if sp.issparse(values):
        values = sp.csr_array(values).copy()
        values.data = values.data * factors[values.indices]
        return values
    return np.asarray(values, dtype=float) * factors


def _interim_masses(
    interim: InterimPrior,
    bins: Bins,
    representation: Representation | None = None,
    tol: float = 1e-3,
    max_depth: int = 12,
    n_sub: int | None = None,
) -> np.ndarray:
    """Normalised interim-prior mass per bin."""
    return Density.coerce(interim, representation).masses(
        bins, tol=tol, max_depth=max_depth, n_sub=n_sub
    )


def _check_kind(kind: Any, interim: Any) -> bool:
    """Whether the input is a posterior; exactly one of ``kind`` and ``interim`` must be given."""
    if kind not in (None, "likelihood"):
        raise ValueError(
            f"kind must be 'likelihood' (pass interim= for posteriors); got {kind!r}."
        )
    if (kind is None) == (interim is None):
        raise ValueError(
            "Say what the input is: kind='likelihood' for likelihoods, or interim=<prior> for "
            "posteriors computed under that prior (not both)."
        )
    return interim is not None


def _check_weights(weights: Any, n: int, fractional: bool) -> np.ndarray:
    """Positive integer multiplicities, or any positive weights if ``fractional`` (#1)."""
    w = check_positive_weights(weights, n)
    if not is_integer_array(w):
        if not fractional:
            raise ValueError(
                "weights are integer multiplicities; pass fractional_weights=True "
                "(or with_weights(..., fractional=True)) for a weighted pseudo-likelihood."
            )
        warn(
            "Non-integer weights define a pseudo-likelihood: the posterior is not a proper "
            "posterior and its uncertainties are not calibrated.",
            InputWarning,
        )
    return w


def _divide_interim(P: Any, Pi: np.ndarray, what: str) -> Any:
    """Columns of ``P`` divided by the interim masses ``Pi``; refuses mass where ``Pi`` is zero."""
    zero = Pi <= 0
    if np.any(zero) and np.any(np.asarray(P.sum(axis=0)).ravel()[zero] > 0):
        raise ValueError(
            f"{what} have mass in bins where the interim prior is zero; "
            "the interim prior must be positive wherever they are."
        )
    return _scale_columns(P, np.where(zero, 0.0, 1.0 / np.where(zero, 1.0, Pi)))


def _group_rows(keys: list) -> tuple[np.ndarray, np.ndarray]:
    # (first row of each distinct key, group of every row), groups in order of first appearance
    ids: dict = {}
    first: list[int] = []
    group = np.empty(len(keys), dtype=np.intp)
    for i, k in enumerate(keys):
        group[i] = g = ids.setdefault(k, len(ids))
        if g == len(first):
            first.append(i)
    return np.array(first, dtype=np.intp), group


def _read_only(a: Any) -> np.ndarray:
    # a float array that cannot be changed in place: reused only when it is already
    # read-only and owns its data (a read-only view could still change through its base)
    a = np.asarray(a, dtype=float)
    return a if not a.flags.writeable and a.base is None else frozen(a.copy())


def _common(weights, selection, selection_mode, drop_empty, fractional_weights) -> dict[str, Any]:
    # the keyword arguments every constructor passes on to Likelihoods
    return {
        "weights": weights,
        "selection": selection,
        "selection_mode": selection_mode,
        "drop_empty": drop_empty,
        "fractional_weights": fractional_weights,
    }


def _rows_logpdf(dists: list[Any], ndim: int) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """``f(z, r)``: log density of distribution ``r[p]`` at the points ``z[p]``."""
    family = getattr(dists[0], "dist", None)
    keys = sorted(getattr(dists[0], "kwds", {}))
    same = (
        ndim == 1
        and family is not None
        and all(
            getattr(d, "dist", None) is family
            and len(d.args) == len(dists[0].args)
            and sorted(d.kwds) == keys
            for d in dists
        )
    )
    if same:  # frozen scipy distributions of one family: one call
        args = [np.array(a, dtype=float) for a in zip(*(d.args for d in dists), strict=True)]
        kwds = {k: np.array([d.kwds[k] for d in dists], dtype=float) for k in keys}
        return lambda z, r: family.logpdf(
            z, *(a[r, None] for a in args), **{k: v[r, None] for k, v in kwds.items()}
        )

    def logpdf(z, r):
        # each distribution receives its points as (n,) or (n, ndim), as everywhere else
        out = np.empty(z.shape[:2])
        for i in np.unique(r):
            sel = r == i
            pts = z[sel].reshape(-1, ndim) if ndim > 1 else z[sel].ravel()
            out[sel] = _dist_logpdf(dists[i], pts).reshape(out[sel].shape)
        return out

    return logpdf


def _correlated_gaussian(mu, C, bins, tol, max_depth) -> tuple[np.ndarray, np.ndarray]:
    """Log bin averages (in u) and log bin masses (in z) of multivariate normal likelihoods."""
    try:
        chol = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        raise ValueError("cov must be symmetric positive definite for every object.") from None
    inv_chol = np.linalg.inv(chol)  # whitening: y = L^-1 (z - mu)
    log_norm = -np.log(np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
    log_norm -= 0.5 * bins.ndim * np.log(2 * np.pi)
    logf = partial(_log_mvn_rows, mu=mu, inv_chol=inv_chol, log_norm=log_norm)
    # a strongly correlated Gaussian is a ridge narrower than the bins that both
    # quadrature rules can miss alike; cells that may hold mass are split until they
    # are no wider than twice the narrowest principal standard deviation
    lam, vec = np.linalg.eigh(C)
    split = partial(_ridge_split, mu=mu, vec=vec, sd=np.sqrt(lam), floor=np.log(tol) * 2)
    quad = {"rows": mu.shape[0], "tol": tol, "max_depth": max_depth, "must_split": split}
    log_avg = bin_log_average(logf, bins, **quad)
    if all(m == Measure.linear() for m in bins.measures):
        return log_avg, log_avg + np.log(bins.volumes)
    # the masses only feed the outside fraction, so a looser tolerance is enough
    log_z = bin_log_average(logf, bins, coordinate="z", **{**quad, "tol": max(tol, 1e-2)})
    return log_avg, log_z + np.log(bins.volumes)


def _ridge_split(z_lo, z_hi, r, mu, vec, sd, floor) -> np.ndarray:
    # cells wider than 2 narrowest principal sds that may hold more mass than exp(floor).
    # The cell's mass is at most the probability that the Gaussian's projection on each
    # principal axis falls in the cell's projected interval, which the 1-D CDF gives exactly.
    V, s = vec[r], sd[r]  # (P, d, d) eigenvectors in columns, (P, d) principal sds
    centre = np.einsum("pdk,pd->pk", V, 0.5 * (z_lo + z_hi) - mu[r]) / s
    half = np.einsum("pdk,pd->pk", np.abs(V), 0.5 * (z_hi - z_lo)) / s
    near, far = np.maximum(np.abs(centre) - half, 0.0), np.abs(centre) + half
    with np.errstate(divide="ignore"):
        log_bound = np.log(special.ndtr(-near) - special.ndtr(-far) + (near == 0) * 0.5)
    wide = (z_hi - z_lo).max(axis=1) > 2 * s[:, 0]
    return wide & (log_bound.min(axis=1) > floor)


def _log_mvn_rows(z, r, mu, inv_chol, log_norm) -> np.ndarray:
    # log N(z[p] | mu[r[p]], C[r[p]]) for points z of shape (P, M, ndim)
    y = (z - mu[r][:, None, :]) @ np.swapaxes(inv_chol[r], 1, 2)
    return log_norm[r][:, None] - 0.5 * (y**2).sum(axis=-1)


def _log_gauss_rows(z: np.ndarray, r: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    m, s = mu[r, None], sig[r, None]
    return -0.5 * ((z - m) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)


#: the outside-mass warning fires when more than OUTSIDE_SHARE of the objects have
#: more than OUTSIDE_THRESHOLD of their mass outside the bins (DECISIONS #64)
OUTSIDE_THRESHOLD, OUTSIDE_SHARE = 0.9, 0.1


def _record_outside(lk: Likelihoods, outside: np.ndarray, enabled: bool = True) -> None:
    """Store each object's mass fraction outside the active bins, and warn if many lose most."""
    _attach_per_object(lk, outside_fraction=outside)
    outside, threshold = lk.meta["outside_fraction"], OUTSIDE_THRESHOLD
    many = outside > threshold
    if enabled and outside.size and many.mean() > OUTSIDE_SHARE:
        warn(
            f"{int(many.sum())} of {outside.size} objects have more than {threshold:.0%} of "
            "their likelihood mass outside the bins (see meta['outside_fraction']); "
            "consider wider bins.",
            InputWarning,
        )


def _outside(log_inside: np.ndarray) -> np.ndarray:
    return -np.expm1(np.minimum(log_inside, 0.0))


# ----------------------------------------------------------------------------
# main class
# ----------------------------------------------------------------------------
class Likelihoods:
    r"""Bin-averaged per-object likelihoods :math:`L_{ik} \propto p(d_i \mid z_i \in B_k)`.

    This container holds the ``(N, K)`` matrix of likelihoods of every object
    for every bin, together with the :class:`~binference.Bins`, optional object
    weights, an optional selection function, and provenance metadata. Rows
    are stored normalised to a maximum of one (their scales never affect
    inference on the population); the per-row log-scales are kept so that
    absolute log-likelihoods (evidences, cross-validation scores) can be
    recovered.

    Use the ``from_*`` constructors to convert posteriors (with their interim
    prior), posterior samples, analytic measurement models, response
    matrices or classifier outputs into likelihoods. See :doc:`/user-guide/inputs`
    for the conventions.

    Parameters
    ----------
    values : array_like or sparse matrix, shape (N, K)
        Non-negative likelihood values. Rows with all zeros are invalid.
    bins : Bins, optional
        The support. Defaults to ``Bins.categories(K)``.
    representation : {"density", "mass"}
        ``"density"`` (default): each value is the likelihood evaluated in (or
        averaged over) the bin. ``"mass"``: each value is the likelihood
        *integrated* over the bin (e.g. CDF differences); it is divided by the
        bin volume. The distinction only matters for unequal bin volumes.
    weights : array_like, shape (N,), optional
        Positive integer multiplicities: an object with weight 3 counts as three
        identical objects (as produced by :meth:`compress`).
    fractional_weights : bool
        Allow non-integer weights. They define a weighted pseudo-likelihood,
        not the likelihood of any data set, so posteriors from it are not
        proper posteriors; a warning says so (DECISIONS #1).
    selection : array_like (K,) or Selection, optional
        Detection probability (or exposure) per bin, used by models unless
        overridden; see :doc:`/user-guide/selection`.
    selection_mode : {"observed", "true"}, optional
        Required with an array-valued ``selection``: whether objects are
        selected on their observed data (or the likelihoods already include the
        detection probability), or detected with a probability that depends on
        their true bin. See :class:`~binference.selection.Selection`.
    drop_empty : bool
        Drop all-zero rows with a warning instead of raising.
    log : bool
        If True, ``values`` are log-likelihoods (``-inf`` allowed).

    Examples
    --------
    >>> import numpy as np, binference
    >>> bins = binference.Bins.linear(0, 1, 4)
    >>> lk = binference.Likelihoods(np.array([[1, 2, 1, 0], [0, 1, 3, 1.0]]), bins)
    >>> lk.N, lk.K
    (2, 4)
    """

    def __init__(
        self,
        values: Any,
        bins: Bins | None = None,
        *,
        representation: Representation = "density",
        weights: Any = None,
        selection: Any = None,
        selection_mode: str | None = None,
        drop_empty: bool = False,
        log: bool = False,
        fractional_weights: bool = False,
    ) -> None:
        if representation not in ("density", "mass"):
            raise ValueError(
                f"representation must be 'density' or 'mass'; got {representation!r}."
            )
        meta: dict[str, Any] = {"source": "values", "representation": representation}
        if log:
            norm, log_scale = _normalise_log(values)
        else:
            norm, log_scale = _normalise_linear(values)
        N, K = norm.shape
        bins = Bins.categories(K) if bins is None else bins
        if not isinstance(bins, Bins):
            raise TypeError("bins must be a binference.Bins instance.")
        if bins.K != K:
            raise ValueError(f"values have K={K} columns but bins have K={bins.K}.")
        if representation == "mass":
            vol = np.asarray(bins.volumes, dtype=float)
            norm = _scale_columns(norm, 1.0 / vol)
            norm, extra = _normalise_linear(norm)
            log_scale = log_scale + extra
        weights = _check_weights(weights, N, fractional_weights)
        sel = as_selection(selection, selection_mode, K)
        self._init_from_parts(norm, log_scale, bins, weights, sel, meta, drop_empty)

    # ------------------------------------------------------------- internals
    def _init_from_parts(self, norm, log_scale, bins, weights, selection, meta, drop_empty=False):
        empty = ~np.isfinite(log_scale)
        if np.any(empty):
            bad = np.flatnonzero(empty)
            if not drop_empty:
                raise ValueError(
                    f"{bad.size} object(s) have zero likelihood in every bin (first: {bad[:5]}). "
                    "Check that the bins cover the data, or pass drop_empty=True."
                )
            warn(f"Dropping {bad.size} object(s) with zero likelihood in every bin.", InputWarning)
            keep = ~empty
            norm = _take_rows(norm, keep)
            log_scale = log_scale[keep]
            weights = weights[keep]
            meta = _slice_per_object({**meta, "dropped": bad}, keep)
        if norm.shape[0] == 0:
            raise ValueError("Likelihoods need at least one object.")
        self._values = frozen(norm)
        self._log_scale = _read_only(log_scale)
        self._bins = bins
        self._weights = _read_only(weights)
        if selection is not None and not isinstance(selection, Selection):
            raise TypeError("internal error: selection must be a Selection object")
        self._selection = selection
        self.meta = dict(meta)

    def __setstate__(self, state: dict[str, Any]) -> None:
        # pickling and deepcopy rebuild writable arrays: freeze them again
        self.__dict__.update(state)
        for name in ("_values", "_log_scale", "_weights"):
            frozen(getattr(self, name))

    @classmethod
    def _from_parts(cls, norm, log_scale, bins, weights, selection, meta, drop_empty=False):
        obj = cls.__new__(cls)
        obj._init_from_parts(norm, log_scale, bins, weights, selection, meta, drop_empty)
        return obj

    # ------------------------------------------------------------ properties
    @property
    def values(self):
        """Row-normalised likelihood matrix (dense ndarray or sparse CSR array)."""
        return self._values

    @property
    def log_scale(self) -> np.ndarray:
        """Per-object log normalisation: ``log L_ik = log values_ik + log_scale_i``."""
        return self._log_scale

    @property
    def bins(self) -> Bins:
        """The :class:`~binference.Bins` the likelihoods are defined over."""
        return self._bins

    @property
    def weights(self) -> np.ndarray:
        """Per-object weights ``(N,)``."""
        return self._weights

    @property
    def selection(self) -> Selection | None:
        """Selection function attached to the data, if any.

        See :class:`~binference.Selection`.
        """
        return self._selection

    @property
    def N(self) -> int:
        """Number of objects (rows)."""
        return int(self._values.shape[0])

    @property
    def K(self) -> int:
        """Number of bins (columns)."""
        return int(self._values.shape[1])

    @property
    def W(self) -> float:
        """Total weight ``sum_i w_i`` (the effective number of objects)."""
        return float(self._weights.sum())

    @property
    def is_sparse(self) -> bool:
        """Whether values are stored as a sparse CSR array."""
        return sp.issparse(self._values)

    @property
    def nnz(self) -> int:
        """Number of stored (non-zero) entries."""
        if self.is_sparse:
            return int(self._values.nnz)
        return int(np.count_nonzero(self._values))

    def __len__(self) -> int:
        return self.N

    def __repr__(self) -> str:
        store = "sparse" if self.is_sparse else "dense"
        extra = "" if self._selection is None else f", selection={self._selection.mode!r}"
        w = "" if np.all(self._weights == 1) else f", W={self.W:g}"
        return f"Likelihoods(N={self.N}, K={self.K}, {store}{w}{extra}, bins={self._bins!r})"

    def __getitem__(self, idx: Any) -> Likelihoods:
        return self.subset(idx)

    # --------------------------------------------------------------- methods
    def toarray(self) -> np.ndarray:
        """Dense copy of the row-normalised values."""
        if self.is_sparse:
            return self._values.toarray()
        return np.array(self._values)

    def to_dense(self) -> Likelihoods:
        """Copy with dense storage."""
        return Likelihoods._from_parts(
            self.toarray(), self._log_scale, self._bins, self._weights, self._selection, self.meta
        )

    def to_sparse(self, rtol: float = 0.0) -> Likelihoods:
        """Copy with sparse CSR storage, dropping entries below ``rtol`` (relative to row max).

        ``0 <= rtol <= 1``: rows are normalised to a maximum of one, which is always kept.
        """
        if not 0.0 <= rtol <= 1.0:
            raise ValueError(f"rtol must lie in [0, 1]; got {rtol}.")
        M = sp.csr_array(self._values, copy=True)  # never touch this object's storage
        if rtol > 0:
            M.data[M.data < rtol] = 0.0
        M.eliminate_zeros()
        return Likelihoods._from_parts(
            M, self._log_scale, self._bins, self._weights, self._selection, self.meta
        )

    def subset(self, idx: Any) -> Likelihoods:
        """Rows ``idx`` (any NumPy index) as a new object."""
        rows = np.arange(self.N)[idx]
        rows = np.atleast_1d(rows)
        vals = self._values[rows]
        return Likelihoods._from_parts(
            vals,
            self._log_scale[rows],
            self._bins,
            self._weights[rows],
            self._selection,
            _slice_per_object({**self.meta, "subset_of": self.N}, rows),
        )

    def with_weights(self, weights: Any, fractional: bool = False) -> Likelihoods:
        """Copy with new object weights (``fractional`` as ``fractional_weights``)."""
        return self._reweighted(_check_weights(weights, self.N, fractional))

    def _reweighted(self, w: np.ndarray) -> Likelihoods:
        # copy with validated weights, for internal resampling schemes
        return Likelihoods._from_parts(
            self._values, self._log_scale, self._bins, w, self._selection, self.meta
        )

    def with_selection(self, selection: Any, selection_mode: str | None = None) -> Likelihoods:
        """Copy with a (new) selection function attached (``None`` removes it)."""
        sel = as_selection(selection, selection_mode, self.K)
        return Likelihoods._from_parts(
            self._values, self._log_scale, self._bins, self._weights, sel, self.meta
        )

    def compress(self, decimals: int | None = None) -> Likelihoods:
        """Merge identical rows into weighted rows.

        Rows are compared after row normalisation, optionally rounded to
        ``decimals``. Weights add and log-scales are combined, so total absolute
        log-likelihoods are preserved exactly when ``decimals`` is None; with
        rounding, each group is represented by its first row. Inference on the
        population is unchanged; Gibbs sampling then draws multinomial counts
        per unique row. Sparse storage stays sparse. Always returns a new object.
        """
        V = self._values
        if sp.issparse(V):
            V = sp.csr_array(V)
            K = V
            if decimals is not None:  # entries that round to zero leave the key, as when dense
                K = V.copy()
                K.data = np.round(K.data, decimals)
                K.eliminate_zeros()
            keys = [
                (K.indices[a:b].tobytes(), K.data[a:b].tobytes())
                for a, b in zip(K.indptr[:-1], K.indptr[1:], strict=True)
            ]
        else:
            key = V if decimals is None else np.round(V, decimals)
            keys = [row.tobytes() for row in np.ascontiguousarray(key)]
        first, inverse = _group_rows(keys)
        n = first.size
        w = np.bincount(inverse, weights=self._weights, minlength=n)
        ls = np.bincount(inverse, weights=self._weights * self._log_scale, minlength=n)
        meta = {k: v for k, v in self.meta.items() if k not in _PER_OBJECT_KEYS}
        meta["compressed_from"] = self.N
        return Likelihoods._from_parts(V[first], ls / w, self._bins, w, self._selection, meta)

    def stack(
        self,
        interim: InterimPrior | None = None,
        interim_representation: Representation | None = None,
    ) -> np.ndarray:
        """Naive "stacked" population estimate: the weighted mean of per-object posteriors.

        Posteriors are formed under ``interim``: by default the interim
        prior the likelihoods were built with (so the stack reproduces the
        input posteriors), else :meth:`Density.flat`. Stacking is exactly one
        EM step from the interim prior, not the maximum-likelihood population;
        it is biased toward the interim prior and broadened by the measurement
        noise, and is provided for comparison only. Its distance from the
        optimum is ``PopulationModel(lk).kkt_gap(lk.stack())``.
        """
        if interim is None:
            Pi = self.meta.get("interim_prior_masses")
            if Pi is None:
                Pi = Density.flat().masses(self._bins)
        else:
            Pi = _interim_masses(interim, self._bins, interim_representation)
        P = _scale_columns(self._values, Pi)
        tot = np.asarray(P.sum(axis=1)).ravel()
        if np.any(tot <= 0):
            rows = np.flatnonzero(tot <= 0)[:5].tolist()
            raise ValueError(
                f"Rows {rows} have no support where the interim prior is positive, so their "
                "posteriors are undefined."
            )
        P = (sp.diags_array(1.0 / tot) @ P) if sp.issparse(P) else P / tot[:, None]
        out = np.asarray(P.T @ self._weights).ravel()
        return out / out.sum()

    def log_evidence_terms(self, rho: np.ndarray) -> np.ndarray:
        r"""Per-object absolute log marginal likelihoods :math:`\log(L_i\cdot\rho)`, unweighted."""
        s = np.asarray(self._values @ np.asarray(rho, dtype=float)).ravel()
        with np.errstate(divide="ignore"):
            return np.log(s) + self._log_scale

    # ----------------------------------------------------------- constructors
    @classmethod
    def from_log(cls, log_values: Any, bins: Bins | None = None, **kwargs: Any) -> Likelihoods:
        """Build from log-likelihoods (``-inf`` marks zero likelihood)."""
        lk = cls(log_values, bins, log=True, **kwargs)
        lk.meta["source"] = "log_values"
        return lk

    @classmethod
    def from_table(
        cls,
        values: Any,
        bins: Bins,
        *,
        kind: Literal["likelihood"] | None = None,
        interim: InterimPrior | None = None,
        representation: Representation | None = None,
        interim_representation: Representation | None = None,
        log: bool = False,
        weights: Any = None,
        selection: Any = None,
        selection_mode: str | None = None,
        drop_empty: bool = False,
        fractional_weights: bool = False,
    ) -> Likelihoods:
        r"""Likelihoods from a table of per-object values on the bins.

        Say what the table holds: ``kind="likelihood"`` for likelihoods, or
        ``interim=`` for posteriors computed under that interim prior. Posteriors
        are converted with :math:`L_{ik} \propto P_{ik} / \Pi_k`, where
        :math:`P_{ik}` is the posterior mass of object *i* in bin *k* and
        :math:`\Pi_k` the interim prior's mass; using posteriors as likelihoods
        would count the interim prior twice.

        Parameters
        ----------
        values : array_like or sparse matrix, shape (N, K)
            Non-negative values (rows need not be normalised).
        bins : Bins
            The bins the values are tabulated on.
        kind : {"likelihood"}, optional
            The values are likelihoods. Exactly one of ``kind`` and ``interim``.
        interim : Density, "flat", callable, distribution or array_like (K,), optional
            The prior the posteriors were computed with, read by
            :meth:`Density.coerce`.
        representation : {"mass", "density"}, optional
            Values per bin (``"mass"``) or at or averaged over the bin
            (``"density"``). Defaults to ``"density"`` for likelihoods and
            ``"mass"`` for posteriors, their usual forms.
        interim_representation : {"mass", "density"}, optional
            How an array ``interim`` is read. Required for arrays on bins of
            unequal volume, where the two readings differ.
        log : bool
            The values are logarithms (``-inf`` allowed; dense only).
        """
        posterior = _check_kind(kind, interim)
        if representation is None:
            representation = "mass" if posterior else "density"
        if representation not in ("mass", "density"):
            raise ValueError(
                f"representation must be 'mass' or 'density'; got {representation!r}."
            )
        tail = _common(weights, selection, selection_mode, drop_empty, fractional_weights)
        if not posterior:
            lk = cls(values, bins, representation=representation, log=log, **tail)
            lk.meta["source"] = "table"
            return lk
        P, _ = _normalise_log(values) if log else _normalise_linear(values)
        if P.shape[1] != bins.K:
            raise ValueError(f"values have {P.shape[1]} columns but bins have K={bins.K}.")
        vol = np.asarray(bins.volumes, dtype=float)
        if not log:  # the check reads row sums, so it needs the values at their own scale
            raw = values if sp.issparse(values) else np.asarray(values, dtype=float)
            _check_posterior_representation(raw, vol, representation)
        if representation == "density":
            P = _scale_columns(P, vol)
        Pi = _interim_masses(interim, bins, interim_representation)
        lk = cls(_divide_interim(P, Pi, "Posteriors"), bins, **tail)
        lk.meta.update(source="posteriors", representation=representation, interim_prior_masses=Pi)
        return lk

    @classmethod
    def from_samples(
        cls,
        samples: Any,
        bins: Bins,
        interim: InterimPrior,
        *,
        sample_weights: Any = None,
        interim_representation: Representation | None = None,
        weights: Any = None,
        selection: Any = None,
        selection_mode: str | None = None,
        sparse: bool | None = None,
        drop_empty: bool = False,
        fractional_weights: bool = False,
        min_samples: int = 20,
        floor: float = 0.0,
        warn_outside: bool = True,
    ) -> Likelihoods:
        r"""Build likelihoods from per-object posterior samples (DECISIONS #31).

        One Monte Carlo estimator, :math:`\hat L_{ik} \propto \frac1S \sum_s
        w_s\,\mathbb 1[z_s \in B_k]\,T'(z_s) / \pi(z_s)`, averaged over each bin
        in the measure coordinate:

        * when the interim prior is known only per bin (``"flat"``, masses or
          densities) this is the weighted count divided by the interim mass,
          :math:`L_{ik} \propto c_{ik} / \Pi_k`;
        * when it can be evaluated at points (a function or a distribution) each
          sample is weighted by :math:`T'(z_s)/\pi(z_s)`, which is exact when the
          interim prior varies within bins.

        Samples drawn from a likelihood are samples under ``Density.flat("z")``.
        The per-object effective number of samples inside the bins and the
        fraction outside are stored in ``meta``; the sums behind the estimator
        are kept so that :meth:`sample_variance` can measure its Monte Carlo
        error at a fitted population.

        Parameters
        ----------
        samples : array_like or sequence of arrays
            For 1-D bins: an ``(N, S)`` array or a list of ``N`` 1-D arrays
            (ragged allowed). For N-D grids: ``(N, S, ndim)`` or a list of
            ``(S_i, ndim)`` arrays.
        bins : Bins
            Target bins (grids only).
        interim : Density, "flat", callable, distribution or array_like (K,)
            Prior under which the samples were drawn, read by
            :meth:`Density.coerce`. There is deliberately no default: samples
            drawn under an informative prior must have it divided out.
        interim_representation : {"mass", "density"}, optional
            How an array ``interim`` is read (see :meth:`from_table`).
        sample_weights : array_like or sequence of arrays, optional
            Importance weights for the samples (same layout as ``samples``).
        sparse : bool, optional
            Store the result sparsely. Defaults to sparse when fewer than 30%
            of entries are non-zero.
        min_samples : int
            Warn if more than 10% of the objects have fewer effective samples
            inside the bins than this (audit I-23).
        floor : float
            Pseudo-samples added to every object, spread over the bins in
            proportion to the interim prior: a Dirichlet floor that keeps bins no
            sample reached from having zero likelihood. Off by default.
        warn_outside : bool
            Warn when many objects have most of their samples outside the bins
            (see :meth:`from_gaussian`).
        """
        if bins.is_categorical:
            raise ValueError("from_samples needs ordered (grid) bins.")
        if not floor >= 0:
            raise ValueError(f"floor must be non-negative; got {floor}.")
        flat, obj_ids, idx, sw, n_obj = _bin_samples(samples, bins, sample_weights)
        density = Density.coerce(interim, interim_representation)
        Pi = density.masses(bins)
        du = bins.u_volumes
        inside = idx >= 0
        rows, cols, sw_in = obj_ids[inside], idx[inside], sw[inside]
        # weight each sample by T'(z) / pi(z); a per-bin prior is flat in u within bins
        log_pi = density._log_pointwise()
        if log_pi is None:
            with np.errstate(divide="ignore"):
                log_pi_u = np.log(Pi[cols] / du[cols])
        else:
            zz = flat[inside].reshape(-1, bins.ndim)
            log_pi_u = np.asarray(log_pi(zz[:, 0] if bins.ndim == 1 else zz), dtype=float).ravel()
            log_pi_u -= sum(m.log_jacobian(zz[:, d]) for d, m in enumerate(bins.measures))
        if not np.all(np.isfinite(log_pi_u)):
            raise ValueError("Samples fall where the interim density is zero or undefined.")
        # rows are rescaled to counts below, so shift each object's log weights to max 0
        shift = np.full(n_obj, -np.inf)
        np.maximum.at(shift, rows, -log_pi_u)
        w = sw_in * np.exp(-log_pi_u - shift[rows])
        # sum the weights and their squares per (object, bin) entry with one sort
        entry, inv = np.unique(rows * bins.K + cols, return_inverse=True)
        s1, s2 = np.bincount(inv, weights=w), np.bincount(inv, weights=w**2)
        er, ec = np.divmod(entry, bins.K)
        shape = (n_obj, bins.K)
        n_samples = np.bincount(obj_ids, weights=sw, minlength=n_obj)
        inside_w = np.bincount(rows, weights=sw_in, minlength=n_obj)
        outside = 1.0 - inside_w / np.where(n_samples > 0, n_samples, 1.0)
        n_eff = _ess(
            np.bincount(er, weights=s1, minlength=n_obj),
            np.bincount(er, weights=s2, minlength=n_obj),
        )
        # scale each row to counts: sum_k S1_k Pi_k = number of samples inside, so the
        # floor is in pseudo-samples whatever the scale of sample_weights
        S1 = sp.csr_array((s1 / du[ec], (er, ec)), shape=shape)
        tot = S1 @ Pi
        scale = np.bincount(rows, minlength=n_obj) / np.where(tot > 0, tot, 1.0)
        S1 = sp.diags_array(scale) @ S1
        S2 = sp.diags_array(scale**2) @ sp.csr_array((s2 / du[ec] ** 2, (er, ec)), shape=shape)
        est = S1 if floor == 0 else S1.toarray() + floor  # proportional to the bin averages
        if sparse is None:
            sparse = floor == 0 and S1.nnz < 0.3 * n_obj * bins.K
        values = sp.csr_array(est) if sparse else (est.toarray() if sp.issparse(est) else est)
        short = n_eff < min_samples
        if short.mean() > 0.1:
            warn(
                f"{int(short.sum())} of {n_obj} objects have fewer than {min_samples} effective "
                "samples inside the bins; their likelihoods will be noisy (more samples, wider "
                "bins or floor= help).",
                InputWarning,
            )
        lk = cls(
            values,
            bins,
            weights=weights,
            selection=selection,
            selection_mode=selection_mode,
            drop_empty=drop_empty,
            fractional_weights=fractional_weights,
        )
        lk.meta.update(source="samples", interim_prior_masses=Pi)
        _attach_per_object(
            lk,
            n_eff=n_eff,
            sample_sum=S1,
            sample_sq=S2,
            sample_count=np.bincount(obj_ids, minlength=n_obj),
        )
        lk.meta["sample_floor"] = float(floor)
        _record_outside(lk, outside, warn_outside)
        return lk

    def sample_variance(self, rho: Any) -> tuple[np.ndarray, np.ndarray]:
        r"""Monte Carlo error of likelihoods built by :meth:`from_samples`, at fractions ``rho``.

        Returns the per-object relative variance :math:`V_i(\rho)` of the
        estimate of :math:`L_i\cdot\rho` and the effective number of samples
        under ``rho``. Summed over objects, :math:`\sum_i V_i` is about the bias
        it puts in the total log-likelihood (Jensen); values above about one
        call for more samples (DECISIONS #31).
        """
        if "sample_sum" not in self.meta:
            raise ValueError("sample_variance needs likelihoods built by from_samples.")
        a = np.asarray(rho, dtype=float).ravel()
        if a.shape != (self.K,):
            raise ValueError(f"rho must have length K={self.K}.")
        m = np.asarray(self.meta["sample_sum"] @ a).ravel()
        q = np.asarray(self.meta["sample_sq"] @ a**2).ravel()
        est = (
            m + self.meta.get("sample_floor", 0.0) * a.sum()
        )  # the floor carries no sampling error
        with np.errstate(divide="ignore", invalid="ignore"):
            var = np.where(est > 0, (q - m**2 / self.meta["sample_count"]) / est**2, np.inf)
        return np.maximum(var, 0.0), _ess(m, q)

    @classmethod
    def from_gaussian(
        cls,
        mean: Any,
        sigma: Any = None,
        bins: Bins | None = None,
        *,
        cov: Any = None,
        weights: Any = None,
        selection: Any = None,
        selection_mode: str | None = None,
        drop_empty: bool = False,
        fractional_weights: bool = False,
        tol: float = 1e-3,
        max_depth: int = 12,
        warn_outside: bool = True,
    ) -> Likelihoods:
        r"""Likelihoods for Gaussian measurements :math:`d_i \sim \mathcal N(z, \sigma_i^2)`.

        Bin averages are exact CDF differences (stable in the far tails) on
        axes with a linear measure; on other axes the likelihood is averaged
        uniformly in the measure coordinate by adaptive quadrature to ``tol``
        with at most ``max_depth`` refinements (see :meth:`from_density`). For
        N-D grids, measurement errors are independent per axis (``mean`` and
        ``sigma`` of shape ``(N, ndim)``) unless ``cov`` gives each object's full
        covariance; correlated Gaussians are averaged over each bin by adaptive
        cubature in log space, to ``tol``, in any number of dimensions (DECISIONS #25).
        Each object's fraction of mass outside the active bins is recorded in
        ``meta["outside_fraction"]``; ``warn_outside`` warns when more than 10% of
        the objects have more than 90% of their mass outside.

        Parameters
        ----------
        mean : array_like, shape (N,) or (N, ndim)
            Measured values.
        sigma : float or array_like, shape (N,) or (N, ndim)
            Measurement uncertainties (strictly positive).
        bins : Bins
            Grid bins.
        cov : array_like, shape (ndim, ndim) or (N, ndim, ndim), optional
            Full covariance matrices (symmetric positive definite), in place of
            ``sigma``.
        """
        if not isinstance(bins, Bins):
            raise TypeError("bins must be a binference.Bins instance.")
        if bins.is_categorical:
            raise ValueError("from_gaussian needs ordered (grid) bins.")
        if (sigma is None) == (cov is None):
            raise ValueError("Pass exactly one of sigma and cov.")
        mu = np.asarray(mean, dtype=float)
        if bins.ndim == 1 and (mu.ndim <= 1 or (mu.ndim == 2 and mu.shape[1] == 1)):
            mu = mu.reshape(-1, 1)
        if mu.ndim != 2 or mu.shape[1] != bins.ndim:
            shapes = "(N,) or (N, 1)" if bins.ndim == 1 else f"(N, {bins.ndim})"
            raise ValueError(f"mean must have shape {shapes}; got {np.shape(mean)}.")
        if not np.all(np.isfinite(mu)):
            raise ValueError("mean must be finite.")
        tail = _common(weights, selection, selection_mode, drop_empty, fractional_weights)
        if cov is not None:
            nd, N = bins.ndim, mu.shape[0]
            C = np.asarray(cov, dtype=float)
            if C.shape == (nd, nd):
                C = np.broadcast_to(C, (N, nd, nd))
            if C.shape != (N, nd, nd):
                raise ValueError(
                    f"cov must have shape ({nd}, {nd}) or (N, {nd}, {nd}); got {np.shape(cov)}."
                )
            var = np.diagonal(C, axis1=1, axis2=2)
            if not np.all(np.isfinite(C)) or np.any(var <= 0):
                raise ValueError("cov must be finite with strictly positive variances.")
            scale = np.sqrt(var[:, :, None] * var[:, None, :])
            if np.any(np.abs(C - np.swapaxes(C, 1, 2)) > 1e-10 * scale):
                raise ValueError("cov must be symmetric for every object.")
            if np.any(np.abs(C) * (1 - np.eye(nd)) > 1e-12 * scale):  # correlated: cubature
                logL, log_mass = _correlated_gaussian(mu, C, bins, tol, max_depth)
                lk = cls(logL, bins, log=True, **tail)
                lk.meta.update(source="gaussian")
                _record_outside(lk, _outside(special.logsumexp(log_mass, axis=1)), warn_outside)
                return lk
            sigma = np.sqrt(var)  # independent axes (correlations below 1e-12): exact
        sig = np.asarray(sigma, dtype=float)
        if sig.ndim == 1 and bins.ndim > 1:
            if sig.shape[0] == mu.shape[0] == bins.ndim:
                raise ValueError(
                    f"sigma of length {bins.ndim} is ambiguous with N = ndim = {bins.ndim}: pass "
                    "shape (N, 1) for one sigma per object or (1, ndim) for one per axis."
                )
            if sig.shape[0] == mu.shape[0]:
                sig = sig[:, None]  # one sigma per object (shared across axes)
            elif sig.shape[0] == bins.ndim:
                sig = sig[None, :]  # one sigma per axis (shared across objects)
            else:
                raise ValueError("sigma has an incompatible shape.")
        elif sig.ndim == 1:
            sig = sig[:, None]
        try:
            sig = np.broadcast_to(sig, mu.shape)
        except ValueError:
            raise ValueError(f"sigma cannot be broadcast to shape {mu.shape}.") from None
        if not np.all(np.isfinite(sig)) or np.any(sig <= 0):
            raise ValueError("sigma must be finite and strictly positive.")
        # per axis: exact log masses, and bin averages (in u on non-linear measures),
        # summed over the axes of each active bin
        logL = log_mass = 0.0
        for e, m, j, mu_d, sig_d in zip(
            bins.edges, bins.measures, bins.cell_index, mu.T, sig.T, strict=True
        ):
            x = (e[None, :] - mu_d[:, None]) / sig_d[:, None]
            lm = log_interval_masses(
                special.log_ndtr(x), special.log_ndtr(-x), 0.5 * special.erf(x / np.sqrt(2))
            )
            if m == Measure.linear():
                avg = lm - np.log(np.diff(e))[None, :]
            else:
                avg = bin_log_average(
                    partial(_log_gauss_rows, mu=mu_d, sig=sig_d),
                    Bins(e, measure=m),
                    rows=mu.shape[0],
                    tol=tol,
                    max_depth=max_depth,
                )
            logL = logL + avg[:, j]
            log_mass = log_mass + lm[:, j]
        lk = cls(logL, bins, log=True, **tail)
        lk.meta.update(source="gaussian")
        # kept so that true values can be drawn from the likelihood within bins
        _attach_per_object(lk, gaussian_mean=mu, gaussian_sigma=np.array(sig))
        _record_outside(lk, _outside(special.logsumexp(log_mass, axis=1)), warn_outside)
        return lk

    @classmethod
    def from_density(
        cls,
        density: Callable[[np.ndarray], np.ndarray] | Sequence[Any],
        bins: Bins,
        *,
        kind: Literal["likelihood"] | None = None,
        interim: InterimPrior | None = None,
        interim_representation: Representation | None = None,
        log: bool = False,
        tol: float = 1e-3,
        max_depth: int = 12,
        n_sub: int | None = None,
        weights: Any = None,
        selection: Any = None,
        selection_mode: str | None = None,
        drop_empty: bool = False,
        fractional_weights: bool = False,
        warn_outside: bool = True,
    ) -> Likelihoods:
        """Likelihoods from per-object densities over the bin coordinate.

        ``density`` is either a vectorised function, where ``density(points)``
        returns an ``(N, M)`` array of values (log-values with ``log=True``) at
        ``M`` points of shape ``(M,)`` (1-D) or ``(M, ndim)``, or a sequence of
        distribution objects with ``logpdf``, ``pdf``, ``logcdf`` or ``cdf``
        (e.g. frozen ``scipy.stats``). Say what they are: ``kind="likelihood"``
        or ``interim=`` for posteriors, as in :meth:`from_table`.

        Likelihoods are averaged over each bin uniformly in the measure
        coordinate, the within-bin population shape; posteriors are integrated
        over each bin in ``z`` and divided by the interim prior's bin masses.
        On 1-D bins, distributions with a ``cdf`` are integrated exactly where
        the result is a mass (posteriors, or likelihoods on a linear measure),
        in log space so that far tails keep their precision.

        Other integrals use adaptive quadrature: each bin is refined until the
        estimated error is below ``tol`` relative to each row's largest bin
        average, splitting cells at most ``max_depth`` times; ``n_sub`` instead
        fixes ``n_sub`` Gauss-Legendre points per bin and axis. The fraction of
        each distribution's mass outside the active bins is recorded in
        ``meta["outside_fraction"]`` on 1-D bins (see :meth:`from_gaussian` for
        ``warn_outside``).
        """
        if bins.is_categorical:
            raise ValueError("from_density needs grid bins.")
        posterior = _check_kind(kind, interim)
        quad = {
            "coordinate": "z" if posterior else "measure",
            "tol": tol,
            "max_depth": max_depth,
            "n_sub": n_sub,
        }
        log_inside = None
        if callable(density):
            logf = density if log else (lambda z: _log_nonneg(density(z)))
            log_avg = bin_log_average(logf, bins, **quad)
        else:
            dists = list(density)
            if not dists or not all(_is_distribution(d) for d in dists):
                raise TypeError("density must be a function or a sequence of distributions.")
            if bins.ndim == 1 and all(_has_cdf(d) for d in dists):
                log_mass = np.stack([_dist_log_masses(d, bins.edges[0]) for d in dists])
                log_mass = log_mass[:, bins.cell_index[0]]
                log_inside = special.logsumexp(log_mass, axis=1)
            if log_inside is not None and (posterior or bins.measures[0] == Measure.linear()):
                log_avg = log_mass - np.log(bins.volumes)  # exact
            elif not all(_has_pdf(d) for d in dists):
                raise TypeError(
                    "Distributions need a logpdf or pdf here: a cdf gives exact bin masses only "
                    "for posteriors or likelihoods on 1-D bins with a linear measure."
                )
            else:
                log_avg = bin_log_average(
                    _rows_logpdf(dists, bins.ndim), bins, rows=len(dists), **quad
                )
        if posterior:  # the interim prior gets the same quadrature as the values
            interim = Density.from_masses(
                _interim_masses(interim, bins, interim_representation, tol, max_depth, n_sub)
            )
        lk = cls.from_table(
            log_avg,
            bins,
            kind=kind,
            interim=interim,
            representation="density",
            log=True,
            **_common(weights, selection, selection_mode, drop_empty, fractional_weights),
        )
        lk.meta["source"] = "density"
        if log_inside is not None:
            _record_outside(lk, _outside(log_inside), warn_outside)
        return lk

    @classmethod
    def from_binned(
        cls,
        response: Response,
        *,
        counts: Any = None,
        values: Any = None,
        fractional_weights: bool = False,
    ) -> Likelihoods:
        r"""Binned measurements: counts per measured bin, or measured values, and a response.

        Every object measured in bin :math:`j` has likelihood row
        :math:`R_{j\cdot}` (:class:`Response`), and rows are weighted by the
        counts. The truth bins are the response's. The efficiency (the column
        sums of :math:`R`) becomes the selection function in ``"observed"`` mode,
        because the rows already include it. This covers unfolding (a detector
        response) and classifier outputs binned by score
        (:meth:`Response.from_labelled`, DECISIONS #29).

        Parameters
        ----------
        response : Response
            The response of the measurement.
        counts : array_like (J,), optional
            Observed objects per measured bin (integers).
        values : array_like, optional
            Measured value of each object, binned by :meth:`Response.index`;
            values outside the measured bins are left out with a warning. Give exactly one of
            ``counts`` and ``values``.
        fractional_weights : bool
            Accept non-integer (for example weighted) counts, which give a
            pseudo-likelihood (the counts are the rows' weights).
        """
        if not isinstance(response, Response):
            raise TypeError(
                f"response must be a binference.Response; got {type(response).__name__}."
            )
        if (counts is None) == (values is None):
            raise ValueError("Give exactly one of counts= and values=.")
        J = response.measured_bins.K
        if values is not None:
            j = response.index(values)
            if np.any(j < 0):
                warn(
                    f"{int(np.count_nonzero(j < 0))} values lie outside the measured bins and "
                    "are left out.",
                    InputWarning,
                )
            c = np.bincount(j[j >= 0], minlength=J).astype(float)
        else:
            c = np.asarray(counts, dtype=float).ravel()
            if c.shape != (J,):
                raise ValueError(f"counts must have one entry per measured bin (J={J}).")
            if not np.all(np.isfinite(c)) or np.any(c < 0):
                raise ValueError("counts must be finite and non-negative.")
        keep = c > 0
        if not keep.any():
            raise ValueError("counts are all zero: there are no observed objects.")
        eff = response.efficiency
        sel = None if np.allclose(eff, 1.0) else Selection(eff, "observed")
        lk = cls(
            response.matrix[keep],
            response.truth_bins,
            weights=c[keep],
            selection=sel,
            fractional_weights=fractional_weights,
        )
        lk.meta.update(source=f"binned, response from {response.origin}", response=response)
        _attach_per_object(lk, measured_bin=np.flatnonzero(keep))
        return lk


# ----------------------------------------------------------------------------
# module-level helpers used by the class
# ----------------------------------------------------------------------------
#: ``meta`` entries holding one value per object (row); they follow subsets and
#: dropped rows, and are discarded when rows are merged by ``compress``.
_PER_OBJECT_KEYS = (
    "n_eff",
    "outside_fraction",
    "gaussian_mean",
    "gaussian_sigma",
    "sample_sum",
    "sample_sq",
    "sample_count",
    "measured_bin",
)


def _take_rows(v: Any, rows: Any) -> Any:
    # rows of a dense or sparse per-object array, by index or boolean mask
    return sp.csr_array(v)[np.arange(v.shape[0])[rows]] if sp.issparse(v) else np.asarray(v)[rows]


def _slice_per_object(meta: dict[str, Any], rows: Any) -> dict[str, Any]:
    out = dict(meta)
    for key in _PER_OBJECT_KEYS:
        if key in out:
            out[key] = _take_rows(out[key], rows)
    return out


def _attach_per_object(lk: Likelihoods, **arrays: Any) -> None:
    """Store per-object arrays aligned with the rows that survived ``drop_empty``."""
    dropped = lk.meta.get("dropped")
    for key, arr in arrays.items():
        keep = np.ones(arr.shape[0], dtype=bool)
        if dropped is not None:
            keep[dropped] = False
        lk.meta[key] = _take_rows(arr, keep)


def _normalise_linear(values: Any):
    """Validate non-negative values and normalise rows to max 1."""
    if sp.issparse(values):
        L = sp.csr_array(values, dtype=float, copy=True)  # do not modify the caller's matrix
        L.sum_duplicates()
        if L.data.size and (not np.all(np.isfinite(L.data)) or np.any(L.data < 0)):
            raise ValueError("Likelihood values must be finite and non-negative.")
        L.eliminate_zeros()
        if L.ndim != 2:
            raise ValueError("values must be 2-D.")
        if L.shape[1] < 2:
            raise ValueError("Need at least two bins (K >= 2).")
        rmax = _csr_row_max(L)
        with np.errstate(divide="ignore"):
            log_scale = np.log(rmax)
        L = _scale_csr_rows(L, np.where(rmax > 0, 1.0 / np.where(rmax > 0, rmax, 1.0), 0.0))
        return L, log_scale
    L = np.array(values, dtype=float)
    if L.ndim != 2:
        raise ValueError("values must be a 2-D array of shape (N, K).")
    if L.shape[1] < 2:
        raise ValueError("Need at least two bins (K >= 2).")
    if not np.all(np.isfinite(L)) or np.any(L < 0):
        raise ValueError("Likelihood values must be finite and non-negative.")
    rmax = L.max(axis=1)
    with np.errstate(divide="ignore"):
        log_scale = np.log(rmax)
    safe = np.where(rmax > 0, rmax, 1.0)
    L /= safe[:, None]
    return L, log_scale


def _normalise_log(log_values: Any):
    """Validate log-likelihoods and exponentiate stably (row max -> 1)."""
    if sp.issparse(log_values):
        raise TypeError("Log-likelihoods must be dense; pass linear values for sparse input.")
    lv = np.array(log_values, dtype=float)
    if lv.ndim != 2:
        raise ValueError("log values must be a 2-D array of shape (N, K).")
    if lv.shape[1] < 2:
        raise ValueError("Need at least two bins (K >= 2).")
    if np.any(np.isnan(lv)) or np.any(lv == np.inf):
        raise ValueError("Log-likelihoods must not contain NaN or +inf.")
    m = lv.max(axis=1)
    finite = np.isfinite(m)
    L = np.zeros_like(lv)
    L[finite] = np.exp(lv[finite] - m[finite, None])
    return L, np.where(finite, m, -np.inf)


def _check_posterior_representation(P, vol: np.ndarray, representation: str) -> None:
    """Warn if declared posterior masses look like densities (or vice versa)."""
    if _equal_volumes(vol):
        return  # representation cannot affect inference with equal volumes
    # sums of the values read as masses, and as densities integrated over bins
    if sp.issparse(P):
        as_mass = np.asarray(P.sum(axis=1)).ravel()
        as_dens = np.asarray(P @ vol).ravel()
    else:
        as_mass = P.sum(axis=1)
        as_dens = P @ vol
    if representation == "mass":
        sums, alt, other = as_mass, as_dens, "density"
    else:
        sums, alt, other = as_dens, as_mass, "mass"
    mismatched = ~np.isclose(sums, 1.0, rtol=1e-2)
    looks_other = np.isclose(alt, 1.0, rtol=1e-2)
    if np.mean(mismatched & looks_other) > 0.5:
        warn(
            f"Most posterior rows integrate to one only if interpreted as '{other}' values; "
            f"did you mean representation='{other}'? "
            "(Bins have unequal volumes, so this matters.)",
            InputWarning,
        )


def _ess(tot: np.ndarray, tot2: np.ndarray) -> np.ndarray:
    # effective number of samples from the sums of the weights and of their squares
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tot2 > 0, tot**2 / tot2, 0.0)


def _bin_samples(samples: Any, bins: Bins, sample_weights: Any):
    """Flatten per-object samples and map them to bin indices."""
    ndim = bins.ndim
    if isinstance(samples, np.ndarray) and samples.dtype != object:
        arr = np.asarray(samples, dtype=float)
        if ndim == 1:
            if arr.ndim != 2:
                raise ValueError("For 1-D bins, samples must have shape (N, S).")
            n_obj, S = arr.shape
            flat = arr.reshape(-1)
        else:
            if arr.ndim != 3 or arr.shape[2] != ndim:
                raise ValueError(f"For {ndim}-D grids, samples must have shape (N, S, {ndim}).")
            n_obj, S = arr.shape[:2]
            flat = arr.reshape(-1, ndim)
        obj_ids = np.repeat(np.arange(n_obj), S)
        if sample_weights is None:
            sw = np.ones(n_obj * S)
        else:
            sw = np.asarray(sample_weights, dtype=float).reshape(-1)
            if sw.size != n_obj * S:
                raise ValueError("sample_weights must match the layout of samples.")
    else:
        seq = list(samples)
        n_obj = len(seq)
        arrays = [
            np.asarray(s, dtype=float).reshape(-1)
            if ndim == 1
            else np.asarray(s, dtype=float).reshape(-1, ndim)
            for s in seq
        ]
        sizes = np.array([a.shape[0] for a in arrays])
        if np.any(sizes == 0):
            raise ValueError("Every object needs at least one sample.")
        flat = np.concatenate(arrays, axis=0)
        obj_ids = np.repeat(np.arange(n_obj), sizes)
        if sample_weights is None:
            sw = np.ones(flat.shape[0])
        else:
            sws = [np.asarray(s, dtype=float).reshape(-1) for s in sample_weights]
            if [len(s) for s in sws] != sizes.tolist():
                raise ValueError("sample_weights must match the layout of samples.")
            sw = np.concatenate(sws)
    if not np.all(np.isfinite(sw)) or np.any(sw < 0):
        raise ValueError("sample_weights must be finite and non-negative.")
    return flat, obj_ids, bins.index(flat), sw, n_obj
