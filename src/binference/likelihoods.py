"""Per-object likelihoods over bins: the main input to binference."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from scipy import special

from ._quadrature import bin_log_average
from ._utils import check_positive_weights
from .bins import Bins
from .density import (
    Density,
    Representation,
    _dist_log_masses,
    _dist_logpdf,
    _equal_volumes,
    _has_cdf,
    _is_distribution,
    _log_nonneg,
    log_interval_masses,
)
from .measure import Measure
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
    n_sub: int | None = None,
) -> np.ndarray:
    """Normalised interim-prior mass per bin."""
    return Density.coerce(interim, representation).masses(bins, tol=tol, n_sub=n_sub)


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


def _common(weights, selection, selection_mode, drop_empty) -> dict[str, Any]:
    # the keyword arguments every constructor passes on to Likelihoods
    return {
        "weights": weights,
        "selection": selection,
        "selection_mode": selection_mode,
        "drop_empty": drop_empty,
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
        out = np.empty(z.shape[:2])
        for i in np.unique(r):
            sel = r == i
            out[sel] = _dist_logpdf(dists[i], z[sel])
        return out

    return logpdf


def _log_gauss_rows(z: np.ndarray, r: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    m, s = mu[r, None], sig[r, None]
    return -0.5 * ((z - m) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)


def _record_outside(lk: Likelihoods, log_inside: np.ndarray, warn: bool = True) -> None:
    """Store each object's mass fraction outside the active bins, and warn if many lose most."""
    _attach_per_object(lk, outside_fraction=-np.expm1(np.minimum(log_inside, 0.0)))
    if warn:
        _warn_outside(lk.meta["outside_fraction"])


def _warn_outside(outside: np.ndarray, threshold: float = 0.9, share: float = 0.1) -> None:
    """Warn when over ``share`` of the objects have over ``threshold`` of their mass outside."""
    many = outside > threshold
    if outside.size and many.mean() > share:
        warnings.warn(
            f"{int(many.sum())} of {outside.size} objects have more than {threshold:.0%} of "
            "their likelihood mass outside the bins (see meta['outside_fraction']); "
            "consider wider bins.",
            InputWarning,
            stacklevel=3,
        )


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
        Positive object weights. Integer weights are multiplicities (e.g. from
        :meth:`compress`); non-integer weights define a weighted likelihood.
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
    ) -> None:
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
        elif representation != "density":
            raise ValueError("representation must be 'density' or 'mass'.")
        weights = check_positive_weights(weights, N)
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
            warnings.warn(
                f"Dropping {bad.size} object(s) with zero likelihood in every bin.",
                InputWarning,
                stacklevel=3,
            )
            keep = ~empty
            norm = norm[keep] if not sp.issparse(norm) else sp.csr_array(norm)[keep]
            log_scale = log_scale[keep]
            weights = weights[keep]
            meta = _slice_per_object({**meta, "dropped": bad}, keep)
            if norm.shape[0] == 0:
                raise ValueError("No objects left after dropping empty rows.")
        self._values = norm
        self._log_scale = np.asarray(log_scale, dtype=float)
        self._bins = bins
        self._weights = np.asarray(weights, dtype=float)
        if selection is not None and not isinstance(selection, Selection):
            raise TypeError("internal error: selection must be a Selection object")
        self._selection = selection
        self.meta = dict(meta)

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
        """Copy with sparse CSR storage, dropping entries below ``rtol`` (relative to row max)."""
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

    def with_weights(self, weights: Any) -> Likelihoods:
        """Copy with new object weights."""
        w = check_positive_weights(weights, self.N)
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

        Rows are compared after row normalisation (optionally rounded to
        ``decimals``). Weights add, and log-scales are combined so that total
        absolute log-likelihoods are preserved exactly. Inference on the
        population is unchanged; Gibbs sampling then draws multinomial counts
        per unique row.
        """
        dense = self.toarray()
        key = dense if decimals is None else np.round(dense, decimals)
        uniq, inverse = np.unique(key, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        if uniq.shape[0] == self.N:
            return self
        w = np.bincount(inverse, weights=self._weights, minlength=uniq.shape[0])
        ls = np.bincount(inverse, weights=self._weights * self._log_scale, minlength=uniq.shape[0])
        rows = dense[np.unique(inverse, return_index=True)[1]] if decimals is not None else uniq
        vals = sp.csr_array(rows) if self.is_sparse else rows
        meta = {k: v for k, v in self.meta.items() if k not in _PER_OBJECT_KEYS}
        meta["compressed_from"] = self.N
        return Likelihoods._from_parts(vals, ls / w, self._bins, w, self._selection, meta)

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
        tail = _common(weights, selection, selection_mode, drop_empty)
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
        zero = Pi <= 0
        if np.any(zero):
            col_mass = np.asarray(P.sum(axis=0)).ravel()
            if np.any(col_mass[zero] > 0):
                raise ValueError(
                    "Posteriors have mass in bins where the interim prior is zero; "
                    "the interim prior must be positive wherever posteriors are."
                )
        inv = np.where(zero, 0.0, 1.0 / np.where(zero, 1.0, Pi))
        lk = cls(_scale_columns(P, inv), bins, **tail)
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
        min_samples: int = 20,
    ) -> Likelihoods:
        r"""Build likelihoods from per-object posterior samples.

        Each object's samples are histogrammed onto ``bins`` and divided by the
        interim-prior mass per bin, :math:`L_{ik} \propto c_{ik} / \Pi_k`.
        The per-object effective number of samples and the fraction of samples
        falling outside the bins are stored in ``meta``.

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
            Warn if the median effective number of samples per object is below this.
        """
        if bins.is_categorical:
            raise ValueError("from_samples needs ordered (grid) bins.")
        obj_ids, flat_idx, sw, n_eff, outside, n_obj = _bin_samples(samples, bins, sample_weights)
        inside = flat_idx >= 0
        M = sp.coo_array(
            (sw[inside], (obj_ids[inside], flat_idx[inside])), shape=(n_obj, bins.K)
        ).tocsr()
        M.sum_duplicates()
        Pi = _interim_masses(interim, bins, interim_representation)
        if np.any(Pi <= 0):
            col_mass = np.asarray(M.sum(axis=0)).ravel()
            if np.any(col_mass[Pi <= 0] > 0):
                raise ValueError("Samples fall in bins where the interim prior is zero.")
        inv = np.where(Pi > 0, 1.0 / np.where(Pi > 0, Pi, 1.0), 0.0)
        M = _scale_columns(M, inv)
        density = M.nnz / max(1, n_obj * bins.K)
        if sparse is None:
            sparse = density < 0.3
        values = M if sparse else M.toarray()
        if np.median(n_eff) < min_samples:
            warnings.warn(
                f"Median effective number of samples per object is {np.median(n_eff):.1f}; "
                "histogrammed likelihoods will be noisy (consider more samples or wider bins).",
                InputWarning,
                stacklevel=2,
            )
        lk = cls(
            values,
            bins,
            weights=weights,
            selection=selection,
            selection_mode=selection_mode,
            drop_empty=drop_empty,
        )
        lk.meta.update(source="samples", interim_prior_masses=Pi)
        _attach_per_object(lk, n_eff=n_eff, outside_fraction=outside)
        _warn_outside(outside)
        return lk

    @classmethod
    def from_gaussian(
        cls,
        mean: Any,
        sigma: Any,
        bins: Bins,
        *,
        weights: Any = None,
        selection: Any = None,
        selection_mode: str | None = None,
        drop_empty: bool = False,
        tol: float = 1e-3,
        warn_outside: bool = True,
    ) -> Likelihoods:
        r"""Likelihoods for Gaussian measurements :math:`d_i \sim \mathcal N(z, \sigma_i^2)`.

        Bin averages are exact CDF differences (stable in the far tails) on
        axes with a linear measure; on other axes the likelihood is averaged
        uniformly in the measure coordinate by adaptive quadrature to ``tol``
        (see :meth:`from_density`). For N-D grids, measurement errors are
        independent per axis (``mean`` and ``sigma`` of shape ``(N, ndim)``).
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
        """
        if bins.is_categorical:
            raise ValueError("from_gaussian needs ordered (grid) bins.")
        mu = np.asarray(mean, dtype=float)
        if bins.ndim == 1:
            mu = mu.reshape(-1, 1)
        if mu.ndim != 2 or mu.shape[1] != bins.ndim:
            raise ValueError(f"mean must have shape (N,) or (N, {bins.ndim}).")
        sig = np.asarray(sigma, dtype=float)
        if sig.ndim == 1:
            if bins.ndim == 1 or sig.shape[0] == mu.shape[0]:
                sig = sig[:, None]  # one sigma per object (shared across axes)
            elif sig.shape[0] == bins.ndim:
                sig = sig[None, :]  # one sigma per axis (shared across objects)
            else:
                raise ValueError("sigma has an incompatible shape.")
        try:
            sig = np.broadcast_to(sig, mu.shape)
        except ValueError:
            raise ValueError(f"sigma cannot be broadcast to shape {mu.shape}.") from None
        if not np.all(np.isfinite(mu)):
            raise ValueError("mean must be finite.")
        if not np.all(np.isfinite(sig)) or np.any(sig <= 0):
            raise ValueError("sigma must be finite and strictly positive.")
        # per axis: exact log masses, and bin averages (in u on non-linear measures),
        # summed over the axes of each active bin
        logL = log_mass = 0.0
        for e, m, j, mu_d, sig_d in zip(
            bins.edges, bins.measures, bins.cell_index, mu.T, sig.T, strict=True
        ):
            x = (e[None, :] - mu_d[:, None]) / sig_d[:, None]
            lm = log_interval_masses(special.log_ndtr(x), special.log_ndtr(-x))
            if m == Measure.linear():
                avg = lm - np.log(np.diff(e))[None, :]
            else:
                avg = bin_log_average(
                    partial(_log_gauss_rows, mu=mu_d, sig=sig_d),
                    Bins(e, measure=m),
                    rows=mu.shape[0],
                    tol=tol,
                )
            logL = logL + avg[:, j]
            log_mass = log_mass + lm[:, j]
        lk = cls(
            logL,
            bins,
            log=True,
            weights=weights,
            selection=selection,
            selection_mode=selection_mode,
            drop_empty=drop_empty,
        )
        lk.meta.update(source="gaussian")
        # kept so that true values can be drawn from the likelihood within bins
        _attach_per_object(lk, gaussian_mean=mu, gaussian_sigma=np.array(sig))
        _record_outside(lk, special.logsumexp(log_mass, axis=1), warn_outside)
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
        if tol <= 0 or max_depth < 0:
            raise ValueError("tol must be positive and max_depth non-negative.")
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
            else:
                log_avg = bin_log_average(
                    _rows_logpdf(dists, bins.ndim), bins, rows=len(dists), **quad
                )
        if posterior:  # the interim prior gets the same quadrature as the values
            interim = Density.from_masses(
                _interim_masses(interim, bins, interim_representation, tol, n_sub)
            )
        lk = cls.from_table(
            log_avg,
            bins,
            kind=kind,
            interim=interim,
            representation="density",
            log=True,
            **_common(weights, selection, selection_mode, drop_empty),
        )
        lk.meta["source"] = "density"
        if log_inside is not None:
            _record_outside(lk, log_inside, warn_outside)
        return lk

    @classmethod
    def from_response(
        cls,
        counts: Any,
        response: Any,
        *,
        bins: Bins | None = None,
        efficiency: Literal["from_response"] | Any | None = "from_response",
    ) -> Likelihoods:
        r"""HEP-style unfolding input: observed counts and a response matrix.

        Every event observed in reco bin *j* has likelihood row
        :math:`p(\text{reco } j,\ \text{observed} \mid \text{true } k)`, and rows
        are weighted by the observed counts. Two conventions are supported:

        * ``efficiency="from_response"`` (default): ``response[j, k] =
          P(reco j, observed | true k)`` already includes the efficiency, so
          its columns sum to :math:`\epsilon_k \le 1`.
        * ``efficiency=eps``: ``response`` is a *migration* matrix
          ``P(reco j | true k, observed)`` whose columns sum to one, and ``eps``
          is the efficiency per true bin. The likelihood rows are then
          ``response[j, k] * eps[k]``.

        Either way the efficiency becomes the selection function, in
        ``"observed"`` mode because it is already folded into the rows. Pass
        ``efficiency=None`` to ignore selection altogether.

        Parameters
        ----------
        counts : array_like, shape (J,)
            Observed counts per reco bin (non-negative).
        response : array_like, shape (J, K)
            Response or migration matrix (see above).
        bins : Bins, optional
            True-spectrum bins (default ``Bins.categories(K)``).
        efficiency : "from_response", array_like (K,), or None
            How to obtain the efficiency (see above).
        """
        c = np.asarray(counts, dtype=float).ravel()
        R = np.asarray(response, dtype=float)
        if R.ndim != 2 or R.shape[0] != c.size:
            raise ValueError("response must have shape (len(counts), K).")
        if np.any(c < 0) or not np.all(np.isfinite(c)):
            raise ValueError("counts must be finite and non-negative.")
        if np.any(R < 0) or not np.all(np.isfinite(R)):
            raise ValueError("response must be finite and non-negative.")
        col = R.sum(axis=0)
        if np.any(col > 1.0 + 1e-6):
            raise ValueError(
                "Columns of the response matrix sum to more than one, so they are not "
                "probabilities P(reco j | true k). Normalise each column by the number of "
                "simulated events generated in that true bin."
            )
        keep = c > 0
        eff: np.ndarray | None
        if isinstance(efficiency, str):
            if efficiency != "from_response":
                raise ValueError("efficiency must be 'from_response', an array, or None.")
            eff = None if np.allclose(col, 1.0) else col
        elif efficiency is None:
            eff = None
        else:
            eff = np.asarray(efficiency, dtype=float).ravel()
            if eff.shape != (R.shape[1],):
                raise ValueError("efficiency must have one value per true bin.")
            if not np.allclose(col[col > 0], 1.0, atol=1e-3):
                warnings.warn(
                    "With an explicit efficiency the response should be a migration matrix "
                    "whose columns sum to one; these columns do not, so the efficiency may "
                    "be counted twice.",
                    InputWarning,
                    stacklevel=2,
                )
            R = R * eff[None, :]
        sel = None if eff is None else Selection(eff, "observed")
        lk = cls(R[keep], bins, weights=c[keep], selection=sel)
        lk.meta.update(source="response", reco_bins_used=np.flatnonzero(keep))
        return lk

    @classmethod
    def from_classifier(
        cls,
        probabilities: Any,
        train_prior: Any,
        *,
        labels: Sequence[Any] | None = None,
        weights: Any = None,
    ) -> Likelihoods:
        r"""Quantification input: calibrated classifier posteriors over classes.

        A classifier trained with class prevalences :math:`\pi^{\rm train}`
        outputs posteriors :math:`p_{ic} \propto L_{ic}\,\pi^{\rm train}_c`, so
        likelihoods are recovered as :math:`L_{ic} \propto p_{ic}/\pi^{\rm train}_c`
        (the correction underlying EMQ / SLD and its Bayesian variants).
        """
        p = np.asarray(probabilities, dtype=float)
        if p.ndim != 2:
            raise ValueError("probabilities must have shape (N, C).")
        C = p.shape[1]
        bins = Bins.categories(labels if labels is not None else C)
        if bins.K != C:
            raise ValueError("labels must have one entry per class.")
        lk = cls.from_table(p, bins, interim=np.asarray(train_prior, dtype=float), weights=weights)
        lk.meta["source"] = "classifier"
        return lk


# ----------------------------------------------------------------------------
# module-level helpers used by the class
# ----------------------------------------------------------------------------
#: ``meta`` entries holding one value per object (row); they follow subsets and
#: dropped rows, and are discarded when rows are merged by ``compress``.
_PER_OBJECT_KEYS = ("n_eff", "outside_fraction", "gaussian_mean", "gaussian_sigma")


def _slice_per_object(meta: dict[str, Any], rows: Any) -> dict[str, Any]:
    out = dict(meta)
    for key in _PER_OBJECT_KEYS:
        if key in out:
            out[key] = np.asarray(out[key])[rows]
    return out


def _attach_per_object(lk: Likelihoods, **arrays: np.ndarray) -> None:
    """Store per-object arrays aligned with the rows that survived ``drop_empty``."""
    dropped = lk.meta.get("dropped")
    for key, arr in arrays.items():
        arr = np.asarray(arr)
        if dropped is not None and len(dropped):
            arr = np.delete(arr, dropped, axis=0)
        lk.meta[key] = arr


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
        warnings.warn(
            f"Most posterior rows integrate to one only if interpreted as '{other}' values; "
            f"did you mean representation='{other}'? "
            "(Bins have unequal volumes, so this matters.)",
            InputWarning,
            stacklevel=3,
        )


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
    idx = bins.index(flat)
    tot = np.bincount(obj_ids, weights=sw, minlength=n_obj)
    tot2 = np.bincount(obj_ids, weights=sw**2, minlength=n_obj)
    inside = np.bincount(obj_ids, weights=sw * (idx >= 0), minlength=n_obj)
    with np.errstate(invalid="ignore", divide="ignore"):
        n_eff = np.where(tot2 > 0, tot**2 / tot2, 0.0)
        outside = np.where(tot > 0, 1.0 - inside / tot, 1.0)
    return obj_ids, idx, sw, n_eff, outside, n_obj
