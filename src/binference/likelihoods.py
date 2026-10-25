"""Per-object likelihoods over bins: the main input to binference."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp
from scipy import special

from ._utils import check_positive_weights
from .bins import Bins
from .selection import Selection, as_selection
from .warnings import InputWarning

__all__ = ["Likelihoods"]

Representation = Literal["density", "mass"]
InterimPrior = str | np.ndarray | Sequence[float] | Callable[[np.ndarray], np.ndarray]


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


def _gauss_legendre_unit(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes in [0, 1] and weights summing to one."""
    x, w = np.polynomial.legendre.leggauss(int(n))
    return 0.5 * (x + 1.0), 0.5 * w


def _subbin_points(bins: Bins, n_sub: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature points inside every bin and their within-bin weights.

    Returns ``points`` with shape ``(K * M,)`` (1-D) or ``(K * M, ndim)`` and
    ``weights`` with shape ``(M,)``, where ``M = n_sub ** ndim``; points for
    bin ``k`` occupy rows ``k * M : (k + 1) * M``.
    """
    u, wu = _gauss_legendre_unit(n_sub)
    edges = bins.edges
    per_axis = [e[:-1, None] + np.diff(e)[:, None] * u[None, :] for e in edges]  # (K_d, n)
    cell_idx = bins.cell_index  # per axis, (K,)
    if bins.ndim == 1:
        return per_axis[0][cell_idx[0]].ravel(), wu
    # tensor-product points per cell, in the order of the (active) bins
    sub = np.meshgrid(*[np.arange(n_sub)] * bins.ndim, indexing="ij")
    sub_idx = [s.ravel() for s in sub]  # (M,) per axis
    pts = np.stack(
        [per_axis[d][cell_idx[d][:, None], sub_idx[d][None, :]] for d in range(bins.ndim)],
        axis=-1,
    )  # (K, M, ndim)
    w = np.ones(len(sub_idx[0]))
    for d in range(bins.ndim):
        w = w * wu[sub_idx[d]]
    return pts.reshape(-1, bins.ndim), w


def _interim_masses(
    interim_prior: InterimPrior,
    bins: Bins,
    representation: Representation = "mass",
    n_sub: int = 5,
) -> np.ndarray:
    """Interim-prior probability mass per bin (normalised to sum to one)."""
    if isinstance(interim_prior, str):
        if interim_prior != "flat":
            raise ValueError("interim_prior must be 'flat', an array of length K, or a callable.")
        masses = np.array(bins.volumes, dtype=float)
    elif callable(interim_prior):
        if bins.is_categorical:
            masses = np.asarray(interim_prior(bins.centers), dtype=float).ravel()
        else:
            pts, w = _subbin_points(bins, n_sub)
            dens = np.asarray(interim_prior(pts), dtype=float).ravel()
            if dens.size != pts.shape[0]:
                raise ValueError("Callable interim_prior must return one density per point.")
            masses = (dens.reshape(bins.K, -1) @ w) * bins.volumes
    else:
        masses = np.asarray(interim_prior, dtype=float).ravel()
        if masses.size != bins.K:
            raise ValueError(f"interim_prior must have length K={bins.K}; got {masses.size}.")
        if representation == "density":
            masses = masses * bins.volumes
        elif representation != "mass":
            raise ValueError("interim_representation must be 'mass' or 'density'.")
    if not np.all(np.isfinite(masses)) or np.any(masses < 0):
        raise ValueError("interim_prior must be finite and non-negative.")
    if masses.sum() <= 0:
        raise ValueError("interim_prior has no mass on the bins.")
    return masses / masses.sum()


def _log_gauss_bin_mass(lo: np.ndarray, hi: np.ndarray, mu: np.ndarray, sig: np.ndarray):
    """``log(Phi((hi-mu)/sig) - Phi((lo-mu)/sig))`` computed stably in the tails."""
    a = (lo - mu) / sig
    b = (hi - mu) / sig
    # use the upper tail when the interval lies above the mean (symmetry)
    flip = a > 0
    a2 = np.where(flip, -b, a)
    b2 = np.where(flip, -a, b)
    lb = special.log_ndtr(b2)
    la = special.log_ndtr(a2)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = lb + np.log1p(-np.exp(la - lb))
    return np.where(b2 > a2, out, -np.inf)


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

    def stack(self, interim_prior: InterimPrior = "flat") -> np.ndarray:
        """Naive "stacked" population estimate: the weighted mean of per-object posteriors.

        Posteriors are formed under ``interim_prior`` (default: uniform density
        in the bin coordinate). Stacking is one EM step from the interim prior;
        it is biased and broadened by the measurement noise, and is provided
        for comparison only.
        """
        Pi = _interim_masses(interim_prior, self._bins)
        P = _scale_columns(self._values, Pi)
        tot = np.asarray(P.sum(axis=1)).ravel()
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
    def from_posteriors(
        cls,
        posteriors: Any,
        bins: Bins,
        interim_prior: InterimPrior,
        *,
        representation: Representation = "mass",
        interim_representation: Representation | None = None,
        weights: Any = None,
        selection: Any = None,
        selection_mode: str | None = None,
        drop_empty: bool = False,
    ) -> Likelihoods:
        r"""Convert per-object posteriors computed under an interim prior into likelihoods.

        Uses :math:`L_{ik} \propto P_{ik} / \Pi_k`, where :math:`P_{ik}` is the
        posterior mass of object *i* in bin *k* and :math:`\Pi_k` the interim
        prior mass of bin *k*. Treating posteriors as likelihoods instead would
        count the interim prior twice and bias the population toward it.

        Parameters
        ----------
        posteriors : array_like or sparse matrix, shape (N, K)
            Posterior PDFs on the bins (rows need not be normalised).
        bins : Bins
            The bins the posteriors are tabulated on.
        interim_prior : "flat", array_like (K,), or callable
            The prior the posteriors were computed with. ``"flat"`` means a
            uniform density in the bin coordinate; an array gives masses (or
            densities, see ``interim_representation``) per bin; a callable is a
            density evaluated by sub-bin quadrature. There is deliberately no
            default.
        representation : {"mass", "density"}
            Whether ``posteriors`` are probabilities per bin (``"mass"``,
            default) or densities evaluated at bin centres (``"density"``).
        interim_representation : {"mass", "density"}, optional
            Same for an array-valued ``interim_prior``. Defaults to
            ``representation``: masses with masses, densities with densities.
        """
        P = posteriors
        if sp.issparse(P):
            P = sp.csr_array(P, dtype=float)
            if P.data.size and (not np.all(np.isfinite(P.data)) or np.any(P.data < 0)):
                raise ValueError("posteriors must be finite and non-negative.")
        else:
            P = np.asarray(P, dtype=float)
            if P.ndim != 2:
                raise ValueError("posteriors must be a 2-D array of shape (N, K).")
            if not np.all(np.isfinite(P)) or np.any(P < 0):
                raise ValueError("posteriors must be finite and non-negative.")
        if P.shape[1] != bins.K:
            raise ValueError(f"posteriors have {P.shape[1]} columns but bins have K={bins.K}.")
        vol = np.asarray(bins.volumes, dtype=float)
        if representation not in ("mass", "density"):
            raise ValueError("representation must be 'mass' or 'density'.")
        _check_posterior_representation(P, vol, representation)
        if representation == "density":
            P = _scale_columns(P, vol)
        if interim_representation is None:
            interim_representation = representation
        Pi = _interim_masses(interim_prior, bins, interim_representation)
        zero = Pi <= 0
        if np.any(zero):
            col_mass = (
                np.asarray(abs(P).sum(axis=0)).ravel() if sp.issparse(P) else np.abs(P).sum(0)
            )
            if np.any(col_mass[zero] > 0):
                raise ValueError(
                    "Posteriors have mass in bins where the interim prior is zero; "
                    "the interim prior must be positive wherever posteriors are."
                )
        inv = np.where(zero, 0.0, 1.0 / np.where(zero, 1.0, Pi))
        L = _scale_columns(P, inv)
        lk = cls(
            L,
            bins,
            weights=weights,
            selection=selection,
            selection_mode=selection_mode,
            drop_empty=drop_empty,
        )
        lk.meta.update(source="posteriors", representation=representation, interim_prior_masses=Pi)
        return lk

    @classmethod
    def from_samples(
        cls,
        samples: Any,
        bins: Bins,
        interim_prior: InterimPrior,
        *,
        sample_weights: Any = None,
        interim_representation: Representation = "mass",
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
        interim_prior : "flat", array_like (K,), or callable
            Prior under which the samples were drawn. ``"flat"`` means a
            uniform density in the bin coordinate. There is deliberately no
            default: samples drawn under an informative prior must have it
            divided out.
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
        Pi = _interim_masses(interim_prior, bins, interim_representation)
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
    ) -> Likelihoods:
        r"""Likelihoods for Gaussian measurements :math:`d_i \sim \mathcal N(z, \sigma_i^2)`.

        Bin-averaged likelihoods are computed exactly from CDF differences
        (stable in the far tails). For N-D grids, measurement errors are
        independent per axis (``mean`` and ``sigma`` of shape ``(N, ndim)``).

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
        # per-axis log bin-averaged densities, summed over the axes of each active bin
        logL = 0.0
        for d, (e, j) in enumerate(zip(bins.edges, bins.cell_index, strict=True)):
            lm = _log_gauss_bin_mass(
                e[None, :-1], e[None, 1:], mu[:, d : d + 1], sig[:, d : d + 1]
            )
            logL = logL + (lm - np.log(np.diff(e))[None, :])[:, j]
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
        return lk

    @classmethod
    def from_distributions(
        cls,
        dists: Sequence[Any],
        bins: Bins,
        *,
        kind: Literal["likelihood", "posterior"] = "likelihood",
        interim_prior: InterimPrior | None = None,
        n_sub: int = 5,
        weights: Any = None,
        selection: Any = None,
        selection_mode: str | None = None,
        drop_empty: bool = False,
    ) -> Likelihoods:
        """Likelihoods from per-object distribution objects (e.g. frozen ``scipy.stats``).

        Each object is described by a distribution over the bin coordinate. For
        1-D bins, objects with a ``cdf`` method are integrated exactly over the
        bins; otherwise ``logpdf`` (or ``pdf``) is integrated with ``n_sub``
        Gauss-Legendre points per bin (per axis).

        Parameters
        ----------
        dists : sequence
            One distribution per object.
        kind : {"likelihood", "posterior"}
            Whether each distribution is the likelihood function of the object
            (default) or a posterior computed under ``interim_prior``.
        interim_prior : required when ``kind="posterior"``.
        """
        if bins.is_categorical:
            raise ValueError("from_distributions needs ordered (grid) bins.")
        if kind == "posterior" and interim_prior is None:
            raise ValueError("kind='posterior' requires an explicit interim_prior.")
        if kind not in ("likelihood", "posterior"):
            raise ValueError("kind must be 'likelihood' or 'posterior'.")
        dists = list(dists)
        N = len(dists)
        masses = np.empty((N, bins.K))
        pts = wq = None
        for i, dist in enumerate(dists):
            if bins.ndim == 1 and hasattr(dist, "cdf"):
                e = bins.edges[0]
                c = np.asarray(dist.cdf(e), dtype=float)
                masses[i] = np.clip(np.diff(c), 0.0, None)
                continue
            if pts is None:
                pts, wq = _subbin_points(bins, n_sub)
            if hasattr(dist, "logpdf"):
                dens = np.exp(np.asarray(dist.logpdf(pts), dtype=float))
            elif hasattr(dist, "pdf"):
                dens = np.asarray(dist.pdf(pts), dtype=float)
            else:
                raise TypeError("Each distribution needs a cdf, logpdf or pdf method.")
            masses[i] = (dens.reshape(bins.K, -1) @ wq) * bins.volumes
        if kind == "likelihood":
            lk = cls(
                masses,
                bins,
                representation="mass",
                weights=weights,
                selection=selection,
                selection_mode=selection_mode,
                drop_empty=drop_empty,
            )
        else:
            lk = cls.from_posteriors(
                masses,
                bins,
                interim_prior,
                representation="mass",
                weights=weights,
                selection=selection,
                selection_mode=selection_mode,
                drop_empty=drop_empty,
            )
        lk.meta["source"] = "distributions"
        return lk

    @classmethod
    def from_function(
        cls,
        func: Callable[[np.ndarray], np.ndarray],
        bins: Bins,
        *,
        n_sub: int = 3,
        log: bool = False,
        kind: Literal["likelihood", "posterior"] = "likelihood",
        interim_prior: InterimPrior | None = None,
        weights: Any = None,
        selection: Any = None,
        selection_mode: str | None = None,
        drop_empty: bool = False,
    ) -> Likelihoods:
        """Likelihoods from a vectorised function of the bin coordinate.

        ``func(points)`` must return an ``(N, M)`` array of (log-)likelihood (or
        posterior density) values for all objects at ``M`` points; ``points`` has
        shape ``(M,)`` for 1-D bins and ``(M, ndim)`` for grids. Values are
        averaged over ``n_sub`` Gauss-Legendre points per bin (per axis).
        """
        if bins.is_categorical:
            raise ValueError("from_function needs ordered (grid) bins.")
        if kind == "posterior" and interim_prior is None:
            raise ValueError("kind='posterior' requires an explicit interim_prior.")
        pts, wq = _subbin_points(bins, n_sub)
        vals = np.asarray(func(pts), dtype=float)
        if vals.ndim != 2 or vals.shape[1] != pts.shape[0]:
            raise ValueError("func(points) must return an array of shape (N, n_points).")
        N = vals.shape[0]
        vals = vals.reshape(N, bins.K, -1)
        if log:
            lw = np.log(wq)
            avg = special.logsumexp(vals + lw[None, None, :], axis=2)
        else:
            avg = vals @ wq
        if kind == "likelihood":
            lk = cls(
                avg,
                bins,
                log=log,
                weights=weights,
                selection=selection,
                selection_mode=selection_mode,
                drop_empty=drop_empty,
            )
        else:
            dens = np.exp(avg - avg.max(axis=1, keepdims=True)) if log else avg
            lk = cls.from_posteriors(
                dens,
                bins,
                interim_prior,
                representation="density",
                weights=weights,
                selection=selection,
                selection_mode=selection_mode,
                drop_empty=drop_empty,
            )
        lk.meta["source"] = "function"
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
        lk = cls.from_posteriors(p, bins, np.asarray(train_prior, dtype=float), weights=weights)
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
    if np.allclose(vol, vol[0]):
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
