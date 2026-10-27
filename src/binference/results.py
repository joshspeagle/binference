"""Result objects returned by every inference engine."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import special

from . import _kernels as kern
from . import diagnostics as diag
from ._utils import RNGLike, as_generator, check_nonnegative, chunk_slices, is_integer_array, warn
from .bins import Bins
from .measure import Measure
from .warnings import ConvergenceWarning, DiagnosticsError, InputWarning

if TYPE_CHECKING:  # pragma: no cover
    from .basis import Basis
    from .likelihoods import Likelihoods
    from .model import PopulationModel
    from .selection import Selection

__all__ = ["PointEstimate", "PopulationResult", "Summary"]

# the per-object methods' outputs grow with draws x objects (audit O-18)
#: Population draws the per-object methods use by default.
N_DRAWS = 500
#: Largest split R-hat that :meth:`PopulationResult.check` accepts by default.
RHAT_MAX = 1.01
#: Smallest bulk and tail ESS per chain that :meth:`PopulationResult.check` accepts by
#: default.
ESS_MIN_PER_CHAIN = 100.0
#: Chains shorter than this many draws get no R-hat or ESS.
MIXING_MIN_DRAWS = 8
#: :meth:`PopulationResult.check` warns when a NUTS chain's E-BFMI is below this.
EBFMI_MIN = 0.3
#: :meth:`PopulationResult.check` warns when a larger share of NUTS iterations hit
#: ``max_depth``.
MAX_DEPTH_SHARE = 0.05
#: :meth:`PopulationResult.check` warns when the summed Monte Carlo variance of
#: likelihoods from samples exceeds this.
SAMPLE_VARIANCE_MAX = 1.0


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

    Parameters
    ----------
    columns : dict of str to ndarray
        The table's columns, by name.
    title : str
        A heading shown above the table.
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
                    row.append(f"{v:.4g}")  # inf and nan as themselves
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
    kind: str
    method: str
    info: dict[str, Any]

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

    def _fractions(self, values: np.ndarray) -> np.ndarray:
        """Values ``(..., C)`` as fractions: rates normalised, fractions as they are."""
        return values / values.sum(axis=-1, keepdims=True) if self.kind == "rates" else values

    def plot_individual(self, i: int, **kwargs: Any):
        """Object ``i``'s own likelihood against its hierarchical posterior (see
        :func:`binference.plotting.plot_individual`).

        Parameters
        ----------
        i : int
            Object index.
        **kwargs
            Passed to :func:`binference.plotting.plot_individual`.
        """
        from .plotting import plot_individual

        return plot_individual(self, i, **kwargs)

    def _select_draws(self, n_draws: int | None, components: bool = False) -> np.ndarray:
        R = self._weight_draws() if components else self._rho_draws()
        return R[_draw_index(R.shape[0], n_draws)]

    def _inferred_selection(self) -> np.ndarray | None:
        """Per-draw selection ``(chain, draw, K)`` when a selection from injections or a
        tallied response (its efficiencies) is inferred with the population."""
        if "response_draws" in self.info:
            return np.asarray(self.info["response_draws"]).sum(axis=-2)
        return self.info.get("selection_draws")

    def _joint_selection(self, n_draws: int | None) -> np.ndarray | None:
        """:meth:`_inferred_selection` as ``(n_draws, K)``, thinned as the population."""
        S = self._inferred_selection()
        return None if S is None else _flat_draws(S, 1, n_draws)

    def _response_rows(self, n_draws: int | None) -> np.ndarray | None:
        """An inferred response's draws ``(n_draws, J, K)``, thinned as the population."""
        Rd = self.info.get("response_draws")
        return None if Rd is None else _flat_draws(Rd, 2, n_draws)

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
        Rd = self._response_rows(n_draws)
        if Rd is not None:
            return Rd[:, model.likelihoods.meta["measured_bin"]], R, None
        S = self._joint_selection(n_draws)  # a selection: no response was inferred
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
        if (
            lk.K != model.K
            or lk.basis != basis
            or (basis is None and not _same_bins(lk.bins, model.bins))
        ):
            raise ValueError("New likelihoods must use the same bins (and basis) as the fit.")
        return lk

    best: PointEstimate

    def assignment_probabilities_loo(
        self, idx: Any = None, n_draws: int | None = N_DRAWS, chunk: int = 4096
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Each object's posterior over bins under the population fitted without it,
        :math:`P(z_i = k \mid D_{-i})`: leave-one-out, for calibration checks
        (DECISIONS #52).

        Pareto-smoothed importance sampling over the posterior draws (Vehtari, Gelman &
        Gabry 2017): draw :math:`s` has the weight :math:`p(d_i \mid \rho_s)^{-w_i}`
        (the whole row, with its multiplicity, is left out), smoothed per object by
        :func:`~binference.diagnostics.psis`. Objects whose Pareto :math:`\hat k` is at
        or above :func:`~binference.diagnostics.psis_threshold` are flagged with a
        warning: their estimate is unreliable, and refitting without them is the remedy.

        The draws must come from the posterior or an approximation of it (bootstrap
        draws are refused). In a rate model an object's factor is
        :math:`\Lambda\,L_i\cdot\rho`; with a selection in a fractions model,
        :math:`L_i\cdot\rho / S\cdot\rho`. An object impossible under some draw gets
        :math:`\hat k = \infty` and a zero row.

        Returns ``(probabilities, k_hat)``: ``(n, K)`` and ``(n,)`` for the objects
        ``idx`` (all by default), ``chunk`` objects at a time.

        Parameters
        ----------
        idx : index, optional
            Objects to use; ``None`` uses all of them.
        n_draws : int, optional
            Population draws used; ``None`` uses all of them.
        chunk : int, optional
            Objects processed per block, which bounds memory.
        """
        if n_draws is not None and n_draws < 5:
            raise ValueError(f"n_draws must be at least 5 for leave-one-out; got {n_draws}.")
        if self.method == "bootstrap":
            raise ValueError("Leave-one-out reweights posterior draws; bootstrap draws are not.")
        model = self._require_model()
        L, R, S = self._object_draws(n_draws)
        m = R.shape[0]
        if m < 5:
            raise ValueError(f"Leave-one-out needs posterior draws (at least 5); this has {m}.")
        if self.kind == "rates":  # Poisson: the object's factor is Lambda (L_i . rho)
            log_norm = -np.log(_flat_draws(self._params, 1, n_draws).sum(axis=1))
        else:
            norm = self._norms(R, S, n_draws)
            log_norm = 0.0 if norm is None else np.log(norm)
        rows = np.atleast_1d(np.arange(model.N)[slice(None) if idx is None else idx])
        w, k_hat = model._w[rows], np.empty(rows.size)

        def weigh(sl, T):  # PSIS weights of the objects sl from p(d_i | rho_s) ~ T
            with np.errstate(divide="ignore"):
                lw = np.log(T)
            lw -= log_norm
            lw *= -w[sl, None]
            bad = ~np.all(np.isfinite(lw), axis=1)  # impossible under some draw
            lw[bad] = 0.0
            lw, k_hat[sl] = diag._psis_rows(lw)
            k_hat[sl][bad], lw[bad] = np.inf, -np.inf
            return np.exp(lw, out=lw)

        out = _mean_responsibilities(_objects(L, model.N, idx), R, chunk, weigh)
        bad = k_hat >= diag.psis_threshold(m)
        if bad.any():
            warn(
                f"{int(bad.sum())} of {rows.size} objects have Pareto k-hat >= "
                f"{diag.psis_threshold(m):.2f}: their leave-one-out posteriors are unreliable "
                f"(first: {rows[bad][:5].tolist()}); refit without them, or use more draws.",
                ConvergenceWarning,
            )
        return self._to_bins(out), k_hat

    # per-object outputs at the best fit (DECISIONS #51)
    def assignment_probabilities_best(self, idx: Any = None, **kwargs: Any) -> np.ndarray:
        """:meth:`assignment_probabilities` at the best fit, :attr:`best`.

        Parameters
        ----------
        idx : index, optional
            Objects to use; ``None`` uses all of them.
        **kwargs
            Further arguments of :meth:`assignment_probabilities`.
        """
        return self.best.assignment_probabilities(idx, **kwargs)

    def sample_assignments_best(self, idx: Any = None, **kwargs: Any) -> np.ndarray:
        """:meth:`sample_assignments` at the best fit, :attr:`best`.

        Parameters
        ----------
        idx : index, optional
            Objects to use; ``None`` uses all of them.
        **kwargs
            Further arguments of :meth:`sample_assignments`.
        """
        return self.best.sample_assignments(idx, **kwargs)

    def sample_values_best(self, idx: Any = None, **kwargs: Any) -> np.ndarray:
        """:meth:`sample_values` at the best fit, :attr:`best`.

        Parameters
        ----------
        idx : index, optional
            Objects to use; ``None`` uses all of them.
        **kwargs
            Further arguments of :meth:`sample_values`.
        """
        return self.best.sample_values(idx, **kwargs)

    def realized_counts_best(self, **kwargs: Any) -> np.ndarray:
        """:meth:`realized_counts` at the best fit, :attr:`best`.

        Parameters
        ----------
        **kwargs
            Passed to :meth:`realized_counts`.
        """
        return self.best.realized_counts(**kwargs)

    def predict_proba_best(self, likelihoods: Likelihoods | Any, **kwargs: Any) -> np.ndarray:
        """:meth:`predict_proba` at the best fit, :attr:`best`.

        Parameters
        ----------
        likelihoods : Likelihoods or array_like (N, K)
            Likelihoods of the objects.
        **kwargs
            Passed to :meth:`predict_proba`.
        """
        return self.best.predict_proba(likelihoods, **kwargs)

    def log_predictive_best(self, likelihoods: Likelihoods | Any, **kwargs: Any) -> np.ndarray:
        """:meth:`log_predictive` at the best fit, :attr:`best`.

        Parameters
        ----------
        likelihoods : Likelihoods or array_like (N, K)
            Likelihoods of the objects.
        **kwargs
            Passed to :meth:`log_predictive`.
        """
        return self.best.log_predictive(likelihoods, **kwargs)

    def assignment_probabilities(
        self, idx: Any = None, n_draws: int | None = N_DRAWS, chunk: int = 4096
    ) -> np.ndarray:
        r"""Posterior probability that each object lies in each bin, :math:`P(z_i = k \mid D)`.

        Averages the per-object conditional posteriors
        :math:`r_{ik}(\rho)` over (up to ``n_draws``) population draws: the
        "hierarchical shrinkage" posteriors of the individual objects.

        Returns an ``(n, K)`` array for the objects ``idx`` (all by default).

        Parameters
        ----------
        idx : index, optional
            Objects to use; ``None`` uses all of them.
        n_draws : int, optional
            Population draws used; ``None`` uses all of them.
        chunk : int, optional
            Objects processed per block, which bounds memory.
        """
        model = self._require_model()
        L, R, _ = self._object_draws(n_draws)
        return self._to_bins(_mean_responsibilities(_objects(L, model.N, idx), R, chunk))

    def sample_assignments(
        self, idx: Any = None, n_draws: int | None = N_DRAWS, rng: RNGLike = None
    ) -> np.ndarray:
        """Joint draws of the bin labels of objects ``idx`` (all by default).

        For each of (up to ``n_draws``) population draws :math:`\\rho^{(s)}`, labels are
        drawn from :math:`z_i \\sim \\mathrm{Cat}(r_i(\\rho^{(s)}))`; together these
        are joint draws from :math:`p(\\rho, z \\mid D)`, exact given exact draws of
        :math:`\\rho` (approximate engines carry their approximation over). Returns an
        ``(n_draws, n_objects)`` integer array.

        Parameters
        ----------
        idx : index, optional
            Objects to use; ``None`` uses all of them.
        n_draws : int, optional
            Population draws used; ``None`` uses all of them.
        rng : int, Generator or None
            Random number generator.
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

    def realized_counts(
        self, n_draws: int | None = N_DRAWS, rng: RNGLike = None, full: bool = False
    ) -> np.ndarray:
        """Draws of the realised histogram of the observed objects' true bins.

        Unlike the population fractions (which include the shot noise of
        drawing ``N`` objects from a population), this is the distribution of
        ``n_k = sum_i w_i 1[z_i = k]`` for *these* objects. With ``full=True`` the
        objects the selection missed are added, drawn for each draw as
        :math:`\\mathrm{NegBin}(W, S\\cdot\\rho)` of them in bins
        :math:`\\propto \\rho\\circ(1 - S)`: the counts of the full sample (DECISIONS #53).
        Returns ``(n_draws, K)`` weighted counts.

        Parameters
        ----------
        n_draws : int, optional
            Population draws used; ``None`` uses all of them.
        rng : int, Generator or None
            Random number generator.
        full : bool, optional
            Add the objects the selection missed.
        """
        model = self._require_model()
        gen = as_generator(rng)
        S = self._missed_selection(n_draws) if full else None
        L, R, _ = self._object_draws(n_draws)
        comps = self._select_draws(n_draws, components=True)  # R holds rho*S when S is inferred
        w = model._w
        unit = bool(np.all(w == 1))
        integer = is_integer_array(w)
        out = np.empty((R.shape[0], self.bins.K))
        for s, rho in enumerate(R):
            Ls = _draw_rows(L, s)
            if integer:
                n = kern.sample_counts(Ls, w, rho, gen, unit_weights=unit)[0]
                out[s] = self._counts_to_bins(n, gen)
            else:
                labels = self._labels_to_bins(kern.sample_labels(Ls, rho, gen), gen)
                out[s] = np.bincount(labels, weights=w, minlength=self.bins.K)
            if full:
                Ss = S if S.ndim == 1 else S[s]
                missed = model._draw_missed(comps[s], np.zeros(comps.shape[1]), gen, S=Ss)
                out[s] += self._counts_to_bins(missed, gen)
        return out

    def _missed_selection(self, n_draws: int | None) -> np.ndarray:
        # the detection probabilities: one row, or one per draw when inferred
        model = self._require_model()
        sel = model.selection_spec
        if self.kind == "rates" or (sel is not None and sel.scale == "exposure"):
            raise ValueError("Full-sample counts need detection probabilities, not exposures.")
        S = self._joint_selection(n_draws)
        S = model.total_exposure if S is None else S
        if S is None:
            raise ValueError(
                "Full-sample counts need an absolute selection; this one is relative, so "
                "the number of missed objects is not identified (DECISIONS #46)."
            )
        return S

    def _selected_of(self, W: np.ndarray) -> np.ndarray:
        """The selected population :math:`\\phi = S\\circ w / (S\\cdot w)` per bin, for rows of
        component weights ``W`` (all draws, or the estimate), with each draw's own S
        when the selection is inferred (DECISIONS #49)."""
        S = self._joint_selection(None)  # all draws, as W
        if S is None and self.selection is not None:
            S = self.selection.values
        if S is None:
            return self._to_bins(W)
        P = W * S
        return self._to_bins(P / P.sum(axis=-1, keepdims=True))

    @property
    def data_support(self) -> np.ndarray:
        """Objects' worth of likelihood in each bin, :math:`\\sum_i w_i \\ell_{ik}` with each
        object's likelihood normalised over the bins: below about one, a bin's fraction
        is set by the prior (DECISIONS #41)."""
        if "data_support" not in self._cache:
            model = self._require_model()
            L = model._L  # with a selection on the truth applied, as the fit sees them
            tot = kern.matvec(L, np.ones(L.shape[1]))
            w = np.divide(model._w, tot, out=np.zeros_like(tot), where=tot > 0)
            self._cache["data_support"] = self._to_bins(kern.rmatvec(L, w))
        return self._cache["data_support"]

    def prior_information_ratio(self, h: float = 1e-4) -> float:
        r"""Efron's (2016) ratio of prior to data information,
        :math:`\mathrm{tr}(-H_\text{prior}) / \mathrm{tr}(-H_\text{data})`.

        The Hessians are of the log densities in the prior's unconstrained coordinates
        at the mean of the unconstrained draws (the optimum for point estimates; for
        engines that keep no unconstrained draws, the image of the posterior mean, with
        any jointly inferred selection or response at its injection or tally values),
        by central differences of the gradient with step ``h``; the prior's is of its
        density in those
        coordinates, so a Dirichlet(:math:`\alpha`) counts as :math:`\alpha` objects
        per bin. The traces depend on those coordinates, so compare ratios across fits
        of one prior family, not across families. Small values mean the data dominate
        (DECISIONS #41); NaN when the data show no curvature (e.g. an empty bin at a
        boundary optimum). Costs :math:`4d` gradient evaluations for ``d`` prior
        coordinates, half of them of the full log density.

        Parameters
        ----------
        h : float, optional
            Finite-difference step of the Hessians.
        """
        from .samplers._curvature import fd_hessian_diag

        model = self._require_model()
        x = self._center_x()
        d = model.prior.dim

        def prior_grad(xp):
            eta, cache = model.prior.forward(xp)
            v = model._value(eta)
            return model.prior.log_prior_and_grad(xp, cache, v, np.zeros_like(v))

        def total_grad(xp):
            return model.log_density_unconstrained_and_grad(np.concatenate([xp, x[d:]]))

        idx = range(d)
        Hp = fd_hessian_diag(prior_grad, x[:d], idx, h=h)
        Ht = fd_hessian_diag(total_grad, x[:d], idx, h=h)
        data = float(np.sum(Hp - Ht))
        return float(np.sum(-Hp) / data) if data > 0 else float("nan")

    def sample_values(
        self,
        idx: Any = None,
        n_draws: int | None = N_DRAWS,
        rng: RNGLike = None,
        loglike: Callable[[np.ndarray, int], np.ndarray] | None = None,
        n_sub: int = 64,
        n_sweeps: int = 20,
    ) -> np.ndarray:
        r"""Joint draws of the continuous true values of objects ``idx``.

        Labels are drawn as in :meth:`sample_assignments`. Within the drawn bin
        the population is flat in the measure coordinate ``u`` (its density in
        ``z`` carries :math:`T'(z)`), so a true value follows the object's own
        likelihood times that density, from each input's within-bin information
        (DECISIONS #50):

        * :meth:`Likelihoods.from_gaussian` with independent errors (``sigma``):
          exactly (truncated Gaussians) on linear axes, else by inverse-CDF
          sampling on ``n_sub`` sub-points per bin; with correlated errors
          (``cov``), ``n_sweeps`` Gibbs sweeps over the axes' conditionals, started
          from independent draws;
        * :meth:`Likelihoods.from_density` on 1-D bins: the density (divided by
          the interim prior for posteriors) on ``n_sub`` sub-points per bin;
        * :meth:`Likelihoods.from_samples`: the object's samples inside the bin,
          resampled with their weights :math:`w_s / \pi_u(u_s)` (flat in ``u``
          in a bin no sample reached, as the ``floor`` is);
        * ``loglike(z, i)``, if given, overrides these on 1-D bins: the
          log-likelihood of object ``i`` (an index into the likelihoods) at the
          points ``z``, on ``n_sub`` sub-points per bin;
        * otherwise (tables, binned inputs, compressed rows, densities on grids,
          samples not kept) flat in ``u`` within the bin, with a warning: the
          top-hat approximation, accurate only when likelihoods are nearly flat
          within bins.

        With a :class:`~binference.Basis` the label is a component: flat
        components are sampled as above on their cells; inside a hat the
        population is the hat, so sub-point weights include its height, and
        other draws on its support are kept with that height.

        With weighted (compressed) rows each row is treated as one object.
        Returns ``(n_draws, n_objects)`` (1-D) or ``(n_draws, n_objects, ndim)``.

        Parameters
        ----------
        idx : index, optional
            Objects to use; ``None`` uses all of them.
        n_draws : int, optional
            Population draws used; ``None`` uses all of them.
        rng : int, Generator or None
            Random number generator.
        loglike : callable, optional
            Each object's own log-likelihood within bins, ``loglike(z, i)``; overrides
            the stored information on 1-D bins.
        n_sub : int, optional
            Sub-points per bin for inverse-CDF draws.
        n_sweeps : int, optional
            Gibbs sweeps over the axes for correlated Gaussian errors.
        """
        model = self._require_model()
        bins = model.bins  # the components' cells with a basis
        if bins.is_categorical:
            raise ValueError("Categorical bins have no continuous values; use sample_assignments.")
        if loglike is not None and bins.ndim != 1:
            raise ValueError("loglike-based sampling supports 1-D bins only.")
        gen = as_generator(rng)
        draw, informed = _value_drawer(model, loglike, n_sub, n_sweeps)
        if not informed:
            what = model.likelihoods.meta.get("source", "a table")
            warn(
                f"sample_values draws values from the population alone within each bin: these "
                f"likelihoods ({what}) keep no information within bins; pass loglike= for each "
                "object's own likelihood.",
                InputWarning,
            )
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
            z, ok = draw(objects[todo], lab, lo, hi, gen, basis.height if hat else None)
            at = np.flatnonzero(todo)[ok]
            out.reshape(-1, bins.ndim)[at] = z[ok]
            todo.ravel()[at] = False
        return out[..., 0] if bins.ndim == 1 else out

    def predict_proba(
        self, likelihoods: Likelihoods | Any, n_draws: int | None = N_DRAWS, chunk: int = 4096
    ):
        """Posterior over bins for *new* objects under the inferred population.

        Parameters
        ----------
        likelihoods : Likelihoods or array_like (N, K)
            Likelihoods of the new objects, on the bins of the fit.
        n_draws : int, optional
            Population draws used; ``None`` uses all of them.
        chunk : int, optional
            Objects processed per block, which bounds memory.
        """
        L, R, _, _ = self._new_object_draws(self._new_likelihoods(likelihoods), n_draws)
        return self._to_bins(_mean_responsibilities(L, R, chunk))

    def log_predictive(
        self, likelihoods: Likelihoods | Any, n_draws: int | None = N_DRAWS, chunk: int = 4096
    ):
        r"""Log posterior-predictive density of new objects, :math:`\log E[L_i\cdot\rho]`.

        Includes the per-object log-scales of the likelihoods, so values can be
        compared across priors and binnings (e.g. for cross-validation). Works
        through the objects ``chunk`` at a time.

        Parameters
        ----------
        likelihoods : Likelihoods or array_like (N, K)
            Likelihoods of the new objects, on the bins of the fit.
        n_draws : int, optional
            Population draws used; ``None`` uses all of them.
        chunk : int, optional
            Objects processed per block, which bounds memory.
        """
        # rates rescale out of the ratio
        L, R, norm, log_scale = self._new_object_draws(self._new_likelihoods(likelihoods), n_draws)
        n = L.shape[-2]
        lp = np.empty(n)
        for sl in chunk_slices(n, chunk):
            T = _row_dots(L, R, sl)[1]
            if norm is not None:  # a new *detected* object: divide by S . rho
                T = T / norm[None, :]
            with np.errstate(divide="ignore"):
                lp[sl] = special.logsumexp(np.log(T), axis=1)
        return lp - np.log(R.shape[0]) + log_scale

    def _new_object_draws(self, lk: Likelihoods, n_draws: int | None):
        """For new objects ``lk``: rows and column weights per draw as in
        :meth:`_object_draws`, each draw's :math:`S\\cdot\\rho` (``None`` without a
        selection) and the rows' log-scales. New binned objects with the response
        inferred in this fit take its draws (DECISIONS #81)."""
        model = self._require_model()
        Rd = self._response_rows(n_draws)
        if Rd is not None and lk.meta.get("response") is model.response:
            R = self._select_draws(n_draws, components=True)
            rows = Rd[:, lk.meta["measured_bin"]]  # the response's own rows: scale one
            return rows, R, self._norms(R, None, n_draws), np.zeros(lk.N)
        _, R, S = self._object_draws(n_draws)
        norm = self._norms(R, S, n_draws)
        if S is None and model.selection_row is not None:
            R = R * model.selection_row  # L~ . rho = L . (S o rho): the rows are not copied
        return lk.values, R, norm, lk.log_scale

    def _norms(self, R: np.ndarray, S: np.ndarray | None, n_draws: int | None):
        """Each draw's normaliser of a detected object's likelihood, :math:`S\\cdot\\rho`
        (the response's efficiencies with an inferred response), for the draws ``R`` and
        inferred selection ``S`` of :meth:`_object_draws`; ``None`` without a selection."""
        model = self._require_model()
        if "response_draws" in self.info:
            return np.sum(self._joint_selection(n_draws) * R, axis=1)
        if model.selection is None:
            return None
        return R.sum(axis=1) if S is not None else R @ model.selection


def _rebuild_gaussian(lk: Likelihoods, basis: Basis) -> Likelihoods:
    """Gaussian likelihoods built again on ``basis`` from their kept means and errors."""
    from .likelihoods import Likelihoods

    meta = lk.meta
    if "gaussian_mean" not in meta:
        raise ValueError(
            "Only likelihoods from Likelihoods.from_gaussian rebuild themselves; pass "
            "build(bins_or_basis) returning these data's likelihoods on a basis."
        )
    spread = (
        {"cov": meta["gaussian_cov"]}
        if "gaussian_cov" in meta
        else {"sigma": meta["gaussian_sigma"]}
    )
    w = lk.weights
    return Likelihoods.from_gaussian(
        meta["gaussian_mean"],
        bins=basis,
        weights=w,
        fractional_weights=not is_integer_array(w),
        warn_outside=False,
        **meta.get("gaussian_options", {}),
        **spread,
    )


def _mean_responsibilities(
    L: Any, R: np.ndarray, chunk: int, weigh: Callable | None = None
) -> np.ndarray:
    r"""Per-object posteriors over the columns, averaged over the draws of column weights
    ``R``, ``chunk`` objects at a time (``L`` may hold rows per draw, ``(n, N, K)``):
    equally, or with the weights ``weigh(sl, T)`` returns for the objects ``sl`` from
    their :math:`T_{is} = L_i\cdot R_s` (``(n_sl, draws)``, rows summing to one)."""
    n, m = L.shape[-2], R.shape[0]
    out = np.zeros((n, L.shape[-1]))
    for sl in chunk_slices(n, chunk):
        block, T = _row_dots(L, R, sl)
        block = block.toarray() if hasattr(block, "toarray") else np.asarray(block)
        rows = np.arange(sl.start, sl.stop)
        W = np.full(T.shape, 1.0 / m) if weigh is None else weigh(sl, T)
        if block.ndim == 2:  # rows shared by the draws: sum_s w_s L R_s / (L.R_s), 2 products
            tiny = np.finfo(float).tiny  # 1/T overflows below: those rows go draw by draw
            inv = np.divide(W, T, out=np.zeros_like(T), where=T > tiny)  # no support: zero
            out[sl] = block * (inv @ R)
            slow = np.any((T > 0) & (T <= tiny), axis=1)
            if not slow.any():
                continue
            rows, block = (
                rows[slow],
                np.broadcast_to(block[slow], (m, slow.sum(), out.shape[1])),
            )
            out[rows] = 0.0
        for s, rho in enumerate(R):  # draw by draw
            P = block[s] * rho
            tot = P.sum(axis=1, keepdims=True)
            out[rows] += W[rows - sl.start, s, None] * P / np.where(tot > 0, tot, 1.0)
    return out


def _row_dots(L: Any, R: np.ndarray, sl: slice) -> tuple[Any, np.ndarray]:
    # the rows of the objects sl (per draw when L is (n, N, K)) and their products with
    # the draws R, T_is = L_i . R_s (n_sl, draws)
    block = L[:, sl] if L.ndim == 3 else L[sl]
    T = np.einsum("snk,sk->ns", block, R) if L.ndim == 3 else np.asarray(block @ R.T)
    return block, T


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


def _central(X: np.ndarray, level: float, axis: int = 0) -> np.ndarray:
    """Bounds ``(lo, hi)`` of the central interval of probability ``level`` along ``axis``."""
    return np.quantile(X, [0.5 - level / 2, 0.5 + level / 2], axis=axis)


#: the symbol of each quantity in labels and names (rates as in audit G-10)
_SYMBOL = {"density": "density", "fractions": "rho", "rates": "lambda", "selected": "phi"}


def _which_name(which: str) -> str:
    """``"fractions"``, ``"rates"``, ``"density"`` or ``"selected"``, singular spellings
    accepted (O-12)."""
    name = {"fraction": "fractions", "rate": "rates"}.get(which, which)
    if name not in ("fractions", "rates", "density", "selected"):
        raise ValueError(
            f"which must be 'fractions', 'rates', 'density' or 'selected'; got {which!r}."
        )
    return name


def _mixing(C: np.ndarray, tail: bool = False) -> dict[str, np.ndarray]:
    """Bulk (and tail) ESS and split R-hat of each column of draws ``(chain, draw, K)``;
    none below ``MIXING_MIN_DRAWS`` per chain (split R-hat also for one chain, audit O-7)."""
    if C.shape[1] < MIXING_MIN_DRAWS:
        return {}
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


def _report(groups: list[tuple[list[str], list[str], type]], strict: bool) -> None:
    # warn once per (messages, hints, category) group, or raise all of them together
    texts = []
    for found, why, category in groups:
        if found:
            text = "; ".join(found) + (". Hint: " + "; ".join(why) if why else "")
            texts.append(text)
            if not strict:
                warn(f"Diagnostics: {text}.", category)
    if strict and texts:
        raise DiagnosticsError(f"Diagnostics failed: {'; '.join(texts)}")


def _same_bins(a: Bins, b: Bins) -> bool:
    # equal bins up to the identity of their measures, which do not survive save and load
    # (a transform axis loads as a table)
    if a.is_categorical or b.is_categorical:
        return a == b
    return (
        a.shape == b.shape
        and all(np.array_equal(x, y) for x, y in zip(a.edges, b.edges, strict=True))
        and [m.name for m in a.measures] == [m.name for m in b.measures]
        and a.is_masked == b.is_masked
        and (not a.is_masked or np.array_equal(a.mask, b.mask))
    )


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


def _value_drawer(model, loglike, n_sub: int, n_sweeps: int) -> tuple[Callable, bool]:
    """``draw(objects, labels, lo, hi, gen, height) -> (z, keep)``: values in the boxes
    ``[lo, hi]`` (in ``u``, ``(n, ndim)``) from the input's own likelihood times the
    population, flat in ``u`` times the hat's ``height(labels, d, u)`` per axis (``None``
    when flat); and whether the input has within-bin information (DECISIONS #50)."""
    meta = model.likelihoods.meta
    measures = model.bins.measures
    flat = partial(_draw_subpoints, measures, n_sub, None)
    if loglike is not None:
        return partial(_draw_subpoints, measures, n_sub, _per_object(loglike)), True
    if "gaussian_mean" in meta:
        return partial(_draw_gauss, measures, n_sub, n_sweeps, meta), True
    within = meta.get("within")
    if within is None or within.data is None or "value_row" not in meta:
        return flat, False  # e.g. rows merged by compress, which drops value_row
    kind, *data = within.data
    row = np.asarray(meta["value_row"])
    if kind == "samples":
        direct = model.likelihoods.basis is None  # labels are the bins the samples are in
        return partial(_draw_samples, measures, flat, direct, row, *data), True
    evaluate = partial(_dists_logl if kind == "dists" else _rows_logl, row, *data)
    return partial(_draw_subpoints, measures, n_sub, evaluate), True


#: Most values a density function returns per call when values are drawn (2**24 floats
#: are 128 MB).
EVAL_BUDGET = 2**24


def _dists_logl(row, dists, interim, objects, labels, zz):
    # distributions of from_density: one vectorised call for the rows' own points
    from .likelihoods import _rows_logpdf, log_interim

    with np.errstate(invalid="ignore"):  # -inf - -inf where the interim vanishes: NaN
        return _rows_logpdf(dists, 1)(zz, row[objects]) - log_interim(interim, zz)


def _rows_logl(row, n_in, fn, log, interim, objects, labels, zz):
    # a density function of from_density returns every input row at common points: it
    # is called on the sub-points of a few bins at a time, at most EVAL_BUDGET values
    from .density import _log_nonneg
    from .likelihoods import log_interim

    out = np.empty(zz.shape)
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    per = max(1, EVAL_BUDGET // (n_in * zz.shape[1]))
    for c in range(0, len(first), per):
        at = np.flatnonzero((inverse >= c) & (inverse < c + per))
        pts = zz[first[c : c + per]]
        v = fn(pts.ravel())
        v = np.asarray(v if log else _log_nonneg(v), dtype=float).reshape(n_in, *pts.shape)
        out[at] = v[row[objects[at]], inverse[at] - c]
    with np.errstate(invalid="ignore"):  # -inf - -inf where the interim vanishes: NaN
        return out - log_interim(interim, zz)


def _per_object(loglike: Callable[[np.ndarray, int], np.ndarray]) -> Callable:
    """An evaluator ``(objects, labels, zz) -> log-likelihoods`` calling ``loglike(z, i)``
    once per object, at the sub-points of all its bins."""

    def evaluate(objects, labels, zz):
        key = objects.astype(np.int64) * (int(labels.max()) + 1) + labels  # by object
        _, first, inverse = np.unique(key, return_index=True, return_inverse=True)
        vals = np.empty((len(first), zz.shape[1]))
        ids, starts = np.unique(objects[first], return_index=True)
        for i, a, b in zip(ids, starts, [*starts[1:], len(first)], strict=True):
            pts = zz[first[a:b]]
            vals[a:b] = np.asarray(loglike(pts.ravel(), int(i)), dtype=float).reshape(pts.shape)
        return vals[inverse.ravel()]

    return evaluate


def _subpoints(zlo, zhi, n_sub):
    # midpoints of n_sub equal cells in z of each box, (n, n_sub)
    return zlo[:, None] + (zhi - zlo)[:, None] * ((np.arange(n_sub) + 0.5) / n_sub)


def _subpoint_draw(zz, width, logw, gen, objects) -> np.ndarray:
    """One value per row by inverse-CDF sampling on the sub-points ``zz`` (cells of
    ``width / n_sub``) with log-weights ``logw``."""
    n, n_sub = logw.shape
    logw = np.where(np.isnan(logw), -np.inf, logw)  # e.g. a posterior over a zero prior
    empty = ~np.isfinite(logw).any(axis=1)
    if empty.any():  # it drew from the box's first sub-point before (audit O-11)
        raise ValueError(
            f"The likelihood is zero at all {n_sub} sub-points of a bin with posterior "
            f"mass (object {int(objects[empty][0])}); raise n_sub or check loglike."
        )
    cell = kern._sample_labels_dense(np.exp(logw - logw.max(axis=1, keepdims=True)), 1.0, gen)
    return zz[np.arange(n), cell] + width / n_sub * (gen.random(n) - 0.5)


def _base_logw(m, zz, labels, d, height):
    # flat in u (the Jacobian in z) times the hat's height along axis d
    with np.errstate(divide="ignore"):
        return m.log_jacobian(zz) + (
            0.0 if height is None else np.log(height(labels, d, m.forward(zz)))
        )


def _draw_subpoints(measures, n_sub, evaluate, objects, labels, lo, hi, gen, height):
    # each axis on n_sub sub-points: the population, times evaluate(objects, labels, zz)
    z = np.empty(lo.shape)
    for d, m in enumerate(measures):
        zlo, zhi = m.inverse(lo[:, d]), m.inverse(hi[:, d])
        zz = _subpoints(zlo, zhi, n_sub)
        logw = _base_logw(m, zz, labels, d, height)
        if evaluate is not None:
            logw = logw + evaluate(objects, labels, zz)
        z[:, d] = _subpoint_draw(zz, zhi - zlo, logw, gen, objects)
    return z, np.ones(len(z), dtype=bool)


def _truncnorm(a, b, gen) -> np.ndarray:
    """Standard normal draws truncated to ``[a, b]``, by the inverse CDF in log space
    (intervals above zero are reflected, so tails keep their precision)."""
    flip = a > 0
    lo, hi = np.where(flip, -b, a), np.where(flip, -a, b)
    la, lb = special.log_ndtr(lo), special.log_ndtr(hi)
    u = gen.random(np.shape(lo))
    with np.errstate(divide="ignore"):  # u = 0 at a lower bound of -inf
        x = special.ndtri_exp(lb + np.log(u + (1 - u) * np.exp(la - lb)))
    x = np.clip(x, lo, hi)
    return np.where(flip, -x, x)


def _draw_gauss(measures, n_sub, n_sweeps, meta, objects, labels, lo, hi, gen, height):
    """Gaussian likelihoods: exact truncated draws per axis on linear axes (the hat's
    height as an acceptance test), sub-points elsewhere; with a covariance, ``n_sweeps``
    Gibbs sweeps over the axes' conditionals from that start (sub-points inside hats)."""
    n, nd = lo.shape
    mu = np.asarray(meta["gaussian_mean"])[objects]
    P = None
    if "gaussian_cov" in meta:
        ids, inverse = np.unique(objects, return_inverse=True)  # one inverse per object
        C = np.asarray(meta["gaussian_cov"])[ids]
        P = np.linalg.inv(C)[inverse]
        sig = np.sqrt(np.diagonal(C, axis1=1, axis2=2))[inverse]
    else:
        sig = np.asarray(meta["gaussian_sigma"])[objects]
    zlo = np.stack([m.inverse(lo[:, d]) for d, m in enumerate(measures)], axis=1)
    zhi = np.stack([m.inverse(hi[:, d]) for d, m in enumerate(measures)], axis=1)
    keep = np.ones(n, dtype=bool)

    def axis(d, m, mu_d, sig_d, reject):
        if m == Measure.linear() and (height is None or reject):
            x = mu_d + sig_d * _truncnorm(
                (zlo[:, d] - mu_d) / sig_d, (zhi[:, d] - mu_d) / sig_d, gen
            )
            if height is not None:
                keep[:] &= gen.random(n) < height(labels, d, m.forward(x))
            return x
        zz = _subpoints(zlo[:, d], zhi[:, d], n_sub)
        logw = (
            _base_logw(m, zz, labels, d, height)
            - 0.5 * ((zz - mu_d[:, None]) / sig_d[:, None]) ** 2
        )
        return _subpoint_draw(zz, zhi[:, d] - zlo[:, d], logw, gen, objects)

    z = np.empty((n, nd))
    for d, m in enumerate(measures):
        z[:, d] = axis(d, m, mu[:, d], sig[:, d], reject=P is None)
    for _ in range(n_sweeps if P is not None else 0):
        for d, m in enumerate(measures):
            s2 = 1.0 / P[:, d, d]
            rest = np.einsum("nj,nj->n", P[:, d], z - mu) - P[:, d, d] * (z[:, d] - mu[:, d])
            z[:, d] = axis(d, m, mu[:, d] - s2 * rest, np.sqrt(s2), reject=False)
    return z, keep


def _draw_samples(
    measures, flat, direct, row, K, key, zs, G, objects, labels, lo, hi, gen, height
):
    """An object's kept samples inside the box, drawn with their weights (``G`` their
    cumulative sum, each row's summing to one; ``key`` = row * K + bin, sorted); in
    a box no sample reached, flat in ``u`` as the ``floor`` is."""
    n, nd = lo.shape
    z, keep = np.empty((n, nd)), np.ones(n, dtype=bool)
    r = row[objects]
    G0 = np.concatenate(([0.0], G))
    if direct:  # the label is the bin each sample was counted in: one segment each
        s0 = np.searchsorted(key, r * K + labels, side="left")
        s1 = np.searchsorted(key, r * K + labels, side="right")
        a, b = G0[s0], G0[s1]
        hit = b > a  # samples with weight in the bin
        u = a[hit] + (1.0 - gen.random(hit.sum())) * (b - a)[hit]
        z[hit] = zs[np.clip(np.searchsorted(G, u, side="left"), s0[hit], s1[hit] - 1)]
    else:  # components: the samples of the row inside the box, times the hat's height
        hit = np.zeros(n, dtype=bool)
        s0 = np.searchsorted(key, r * K, side="left")
        s1 = np.searchsorted(key, (r + 1) * K, side="left")
        order = np.lexsort((labels, objects))
        cut = np.flatnonzero(np.diff(objects[order]) | np.diff(labels[order])) + 1
        last = -1
        for at in np.split(order, cut):  # the draws of one object in one box
            i = at[0]
            seg = slice(s0[i], s1[i])
            if objects[i] != last:  # the object's samples in u, once for all its boxes
                us = np.stack([m.forward(zs[seg, d]) for d, m in enumerate(measures)], axis=1)
                last = objects[i]
            w = np.diff(G0[s0[i] : s1[i] + 1]) * np.all((us >= lo[i]) & (us <= hi[i]), axis=1)
            for d in range(nd if height is not None else 0):
                w = w * height(labels[i], d, us[:, d])
            c = np.cumsum(w)
            if c.size and c[-1] > 0:
                z[at] = zs[seg][np.searchsorted(c, (1.0 - gen.random(at.size)) * c[-1])]
                hit[at] = True
    miss = ~hit
    if miss.any():
        z[miss], keep[miss] = flat(objects[miss], labels[miss], lo[miss], hi[miss], gen, height)
    return z, keep


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
        True for approximations: VB, bootstrap, and Laplace draws that are
        uncorrected, fail their Pareto k-hat check, or (under
        ``method="auto"``) have an importance ESS below ``target_ess``.
    info : dict
        Engine statistics (e.g. NUTS tree depths, divergences, step sizes).
    labels : ndarray or None
        Joint label draws stored by Gibbs, ``(chains, draws, objects)`` for the
        objects in ``label_index``: bin indices, or basis-component indices when
        the likelihoods use a :class:`~binference.Basis`;
        :meth:`sample_assignments` returns bins either way.
    """

    def __init__(
        self,
        model: PopulationModel | None,
        draws: np.ndarray,
        *,
        method: str,
        kind: str | None = None,
        bins: Bins | None = None,
        x: np.ndarray | None = None,
        log_density_unconstrained: np.ndarray | None = None,
        approximate: bool = False,
        info: dict[str, Any] | None = None,
        labels: np.ndarray | None = None,
        label_index: np.ndarray | None = None,
        prior_repr: str | None = None,
        prior_key: str | None = None,
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
        self.kind = kind = kind or (model.kind if model is not None else "fractions")
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
        #: a digest of the prior's arguments in full, which attach compares (the repr
        #: shortens arrays)
        self.prior_key = prior_key or (
            model.prior_spec._fingerprint() if model is not None else None
        )
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
        """Wrap draws from an external sampler, ``(n_chains, n_draws, K)`` or ``(n_draws, K)``:
        fractions for a fractions model (rows summing to one), rates for a rate model.

        A model that infers its selection or response with the population needs their
        draws too, as ``info["selection_draws"]`` or ``info["response_draws"]`` (the
        layout the engines store).

        Parameters
        ----------
        model : PopulationModel
            The model the draws are of.
        samples : array_like (n_chains, n_draws, K) or (n_draws, K)
            The draws.
        method : str, optional
            Engine name recorded in the result.
        approximate : bool, optional
            Flag the draws as approximate.
        info : dict, optional
            Extra information stored with the draws.
        """
        key = "selection_draws" if model.selection_uncertain else "response_draws"
        if (model.selection_uncertain or model.response is not None) and key not in (info or {}):
            raise ValueError(
                f"This model infers its {key.split('_')[0]} with the population: pass its "
                f'draws as info["{key}"], or fix it (`.fixed()` or `.at_mean()`) first.'
            )
        X = np.asarray(samples, dtype=float)
        if X.ndim not in (2, 3) or X.shape[-1] != model.K:
            raise ValueError(
                f"samples must have shape (n_chains, n_draws, {model.K}) or (n_draws, "
                f"{model.K}); got {X.shape}."
            )
        check_nonnegative(X, "samples")
        if model.kind == "fractions":
            sums = X.sum(axis=-1)
            if not np.allclose(sums, 1.0, atol=1e-6):
                raise ValueError(
                    "samples of a fractions model must sum to one per draw; rates need a rate "
                    "model (a GammaRates prior)."
                )
            X = X / sums[..., None]
        if (
            key in (info or {})
            and np.shape(info[key])[:2] != (X[None] if X.ndim == 2 else X).shape[:2]
        ):
            raise ValueError(
                f'info["{key}"] must have one draw per sample, (n_chains, n_draws, ...).'
            )
        return cls(model, X, method=method, approximate=approximate, info=info)

    @property
    def selection(self) -> Selection | None:
        """The selection function of the fit, kept through :meth:`save` and :func:`load`."""
        return self._selection

    _fit_model = True  # whether the model is the fit's own (not bare likelihoods attached)

    def _fit_model_only(self, what: str) -> PopulationModel:
        # the fit's model, refused when bare likelihoods were attached after load
        if not self._fit_model:
            raise ValueError(
                f"{what} needs the fit's prior: attach a PopulationModel built with it "
                "(bare likelihoods get the default prior)."
            )
        return self._require_model()

    @property
    def best(self) -> PointEstimate:
        """The best fit: :class:`~binference.samplers.MAP` with its defaults, which is the
        mode of :math:`\\rho` on the simplex for conjugate priors whose density is bounded
        there (e.g. Dirichlet concentrations of at least one), and the mode in the
        unconstrained coordinates otherwise, and always with a jointly inferred selection
        or response (DECISIONS #15, #35). Computed on first use
        and kept; per-object outputs at it carry a ``_best`` suffix (DECISIONS #51).
        After :func:`load`, it needs the fit's prior: attach a model, not bare likelihoods."""
        if "best" not in self._cache:
            from .samplers import MAP

            self._cache["best"] = MAP(self._fit_model_only("The best fit")).run()
        return self._cache["best"]

    def attach(self, likelihoods_or_model: Any) -> PopulationResult:
        """Attach likelihoods (or a model) to a loaded result to enable per-object methods.

        Bare likelihoods take the fit's selection; likelihoods or a model with another
        selection, or on other bins, are refused (audit O-1, O-2).

        Parameters
        ----------
        likelihoods_or_model : Likelihoods or PopulationModel
            The fit's likelihoods, or a model built on them.
        """
        from .likelihoods import Likelihoods
        from .model import PopulationModel
        from .selection import _rebuild as _rebuild_selection

        m, saved = likelihoods_or_model, self.selection
        if not isinstance(m, PopulationModel):
            lk = m if isinstance(m, Likelihoods) else Likelihoods(m, self.basis or self.bins)
            sel = None if lk.selection is not None else saved
            if sel is not None and sel.basis is not None and _same_layout(lk.basis, sel.basis):
                # the same layout; its measure did not survive saving
                sel = _rebuild_selection(*sel._parts()[:3], lk.basis, sel.injections)
            m = PopulationModel(lk, selection=sel)
        if m.selection_spec != saved:
            raise ValueError("The likelihoods (or model) carry another selection than this fit.")
        if not _same_layout(m.likelihoods.basis, self.basis):
            raise ValueError("The likelihoods use a different basis than this result.")
        if self.basis is None and not _same_bins(m.bins, self.bins):
            raise ValueError(f"The likelihoods are on other bins ({m.bins}) than this result.")
        # bare likelihoods get the default prior; a model counts as the fit's with its prior
        self._fit_model = isinstance(m, PopulationModel) and (
            self.prior_key is None or m.prior_spec._fingerprint() == self.prior_key
        )
        self.model = m
        self._selection = m.selection_spec
        # everything computed from the previous likelihoods: the cache, the missing
        # information and a fractions model's total (drawn with its W)
        self._cache.clear()
        self.info.pop("missing_information", None)
        if self.kind == "fractions":
            self._total = None
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
        return self._fractions(self._draws)

    @property
    def samples(self) -> np.ndarray:
        """Population fractions, shape ``(n_draws, K)`` (chains concatenated)."""
        return _flat_draws(self.chains, 1, None)

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
            return _flat_draws(self._draws, 1, None)
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
        return self._fractions(_flat_draws(self._params, 1, None))

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
        if which == "selected":
            return self.selected
        return self.density()

    @property
    def selected(self) -> np.ndarray:
        """Draws ``(n_draws, K)`` of the selected population
        :math:`\\phi = S\\circ\\rho / (S\\cdot\\rho)`, the bins of the observed objects;
        :math:`\\rho` itself without a selection (DECISIONS #49)."""
        if "selected" not in self._cache:
            self._cache["selected"] = self._selected_of(self._weight_draws())
        return self._cache["selected"]

    def _center_x(self) -> np.ndarray:
        # the mean of the unconstrained draws, else the image of the mean columns
        model = self._require_model()
        if self.x is not None and self.x.shape[-1] == model.dim:
            return _flat_draws(self.x, 1, None).mean(axis=0)
        return model.from_value(_flat_draws(self._params, 1, None).mean(axis=0))

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

    def cov(self, which: str = "fractions") -> np.ndarray:
        """Posterior covariance ``(K, K)`` of the fractions (or ``which="rates"``, ...).

        Parameters
        ----------
        which : {"fractions", "rates", "density", "selected"}, optional
            Quantity used: fractions, rates, densities or the selected population.
        """
        return np.cov(self._which(which), rowvar=False)

    def corr(self, which: str = "fractions") -> np.ndarray:
        """Posterior correlation matrix ``(K, K)`` of the fractions (or ``which=...``).

        Parameters
        ----------
        which : {"fractions", "rates", "density", "selected"}, optional
            Quantity used: fractions, rates, densities or the selected population.
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.corrcoef(self._which(which), rowvar=False)

    def quantile(self, q: Any, which: str = "fractions") -> np.ndarray:
        """Posterior quantiles, shape ``(*np.shape(q), K)``.

        Parameters
        ----------
        q : float or array_like
            Quantiles, in ``[0, 1]``.
        which : {"fractions", "rates", "density", "selected"}, optional
            Quantity used: fractions, rates, densities or the selected population.
        """
        return np.quantile(self._which(which), q, axis=0)

    def interval(
        self, level: float = 0.68, kind: str = "central", which: str = "fractions"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-bin credible intervals ``(lo, hi)``, each ``(K,)``.

        Parameters
        ----------
        level : float, optional
            Probability inside each interval.
        kind : {"central", "hdi"}, optional
            Equal-tailed (``"central"``) or highest-density (``"hdi"``) intervals.
        which : {"fractions", "rates", "density", "selected"}, optional
            Quantity used: fractions, rates, densities or the selected population.
        """
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1).")
        X = self._which(which)
        if kind == "central":
            lo, hi = _central(X, level)
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

    def functional(
        self, func: Callable[[np.ndarray], Any], vectorized: bool = False, which: str = "fractions"
    ):
        """Draws of a derived quantity ``func(rho)`` (e.g. the mean redshift), or of
        the rates or densities with ``which``.

        Parameters
        ----------
        func : callable
            Function of one draw ``(K,)``.
        vectorized : bool, optional
            Pass all draws to ``func`` at once, as an ``(n_draws, K)`` array.
        which : {"fractions", "rates", "density", "selected"}, optional
            Quantity used: fractions, rates, densities or the selected population.

        Examples
        --------
        >>> zbar = result.functional(lambda r: r @ result.bins.centers)  # doctest: +SKIP
        """
        X = self._which(which)
        if vectorized:
            return np.asarray(func(X))
        return np.array([func(r) for r in X])

    def rebin(self, groups: Any, which: str = "fractions") -> np.ndarray:
        """Fractions (or rates, with ``which="rates"``) aggregated into coarser groups.

        ``groups`` is either a length-``K`` array of integers mapping bins to groups
        (e.g. from :meth:`Bins.coarsen`) or a sequence of lists of bin indices.
        Returns ``(n_draws, n_groups)``.

        Parameters
        ----------
        groups : array_like (K,) or sequence of lists
            Group of each bin, or the bins of each group.
        which : {"fractions", "rates", "density", "selected"}, optional
            Quantity used: fractions, rates, densities or the selected population.
        """
        if _which_name(which) == "density":
            raise ValueError("Densities per unit volume do not add across bins; rebin fractions.")
        X = self._which(which)
        try:
            arr = np.asarray(groups)
        except ValueError:  # ragged sequence of groups
            arr = None
        if (
            arr is not None
            and arr.ndim == 1
            and arr.shape[0] == self.K
            and arr.dtype.kind in "iuf"
        ):
            if not is_integer_array(arr) or arr.min() < 0:
                raise ValueError("groups must map each bin to a non-negative integer group.")
            arr = arr.astype(np.intp)
            one_hot = np.zeros((self.K, int(arr.max()) + 1))
            one_hot[np.arange(self.K), arr] = 1.0
            return X @ one_hot
        return np.stack([X[:, list(grp)].sum(axis=1) for grp in groups], axis=1)

    # ---------------------------------------------------------- diagnostics
    def _mixing_of(self, which: str) -> dict[str, np.ndarray]:
        """Bulk and tail ESS and split R-hat of the fractions or the rates, cached; a
        density shares its base's (they ignore a positive scale per bin)."""
        base = self.kind if which == "density" else which
        if ("mixing", base) not in self._cache:
            X = self._which(base).reshape(self.n_chains, -1, self.K)
            self._cache["mixing", base] = _mixing(X, tail=True)
        return self._cache["mixing", base]

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Convergence diagnostics (computed lazily and cached)."""
        if "diagnostics" in self._cache:
            return self._cache["diagnostics"]
        out: dict[str, Any] = {}
        # for rate models diagnose the rates, which also carry the total
        out.update(self._mixing_of(self.kind))
        Sd = self._inferred_selection()  # a selection, or a response's efficiencies
        if Sd is not None:
            name = "response" if "response_draws" in self.info else "selection"
            out.update({f"{k}_{name}": v for k, v in _mixing(Sd).items()})
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
            value = _flat_draws(self._params, 1, None).mean(axis=0)
            try:
                self.info["missing_information"] = self.model.missing_information(value)
            except (np.linalg.LinAlgError, ValueError):
                return None
            self._cache.pop("diagnostics", None)
        return self.info.get("missing_information")

    def _mixing_problems(
        self, rhat_max: float = RHAT_MAX, ess_min: float = ESS_MIN_PER_CHAIN
    ) -> list[str]:
        """R-hat above ``rhat_max`` and bulk or tail ESS below ``ess_min`` per chain, of
        the population and of an inferred selection or response, for Markov chains."""
        d, msgs = self.diagnostics, []
        # Markov chains (and external draws), whose draws are correlated; independent
        # draws have no chain for R-hat and ESS to describe
        from .inference import METHODS

        engine = METHODS.get(self.method)
        if self.approximate or (engine is not None and not engine.markov):
            return msgs
        for suffix, what in (
            ("", ""),
            ("_selection", " of the selection"),
            ("_response", " of the response's efficiencies"),
        ):
            if "rhat" + suffix in d and np.nanmax(d["rhat" + suffix]) > rhat_max:
                msgs.append(f"max R-hat{what} = {np.nanmax(d['rhat' + suffix]):.3f} > {rhat_max}")
            for kind in ("bulk", "tail"):
                key = f"ess_{kind}{suffix}"
                if key in d:
                    ess = np.nanmin(d[key])
                    per_chain = ess / max(1, self.n_chains)
                    if per_chain < ess_min:
                        # rounded down, so the message never shows the limit as reached
                        msgs.append(
                            f"min {kind} ESS{what} = {math.floor(ess)} "
                            f"({math.floor(per_chain * 10) / 10:.1f} per chain < {ess_min:g})"
                        )
        return msgs

    def check(
        self,
        strict: bool = False,
        rhat_max: float = RHAT_MAX,
        ess_min: float = ESS_MIN_PER_CHAIN,
        predictive_p_min: float = 1e-3,
        predictive_draws: int = 200,
        support_min: float | None = None,
    ):
        """Check diagnostics and warn (or raise with ``strict=True``) about problems.

        Sampler problems raise a :class:`~binference.ConvergenceWarning`; problems
        with the inputs (Monte Carlo noise of sample likelihoods, binned counts
        the response does not describe) an :class:`~binference.InputWarning`.
        For binned inputs with more measured than truth bins, the counts are
        tested with :meth:`Response.predictive_pvalue` over ``predictive_draws``
        draws, and flagged below ``predictive_p_min``. With ``support_min``, bins whose
        :attr:`data_support` is below it (in objects) are flagged as set by the prior.

        Returns a list of messages (empty if everything looks fine).

        Parameters
        ----------
        strict : bool, optional
            Raise :class:`~binference.DiagnosticsError` instead of warning.
        rhat_max : float, optional
            Largest R-hat that passes.
        ess_min : float, optional
            Smallest bulk or tail ESS per chain that passes.
        predictive_p_min : float, optional
            Smallest posterior predictive p-value of binned counts that passes.
        predictive_draws : int, optional
            Draws used for the predictive p-value.
        support_min : float, optional
            Flag bins with less data support than this; ``None`` skips the test.
        """
        d = self.diagnostics
        msgs = self._mixing_problems(rhat_max, ess_min)
        hints = []
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
        if "ebfmi" in d and np.nanmin(d["ebfmi"]) < EBFMI_MIN:
            msgs.append(f"E-BFMI = {np.nanmin(d['ebfmi']):.2f} < {EBFMI_MIN:g} in some chain")
            hints.append(
                "NUTS explores the energy distribution poorly (heavy tails or a funnel); "
                "a longer warm-up or another prior may help"
            )
        if d.get("converged") is False:
            msgs.append("the optimiser did not converge")
        threshold = d.get("k_hat_threshold", diag.K_HAT_MAX)
        if d.get("k_hat", 0) >= threshold:
            msgs.append(
                f"Pareto k-hat = {d['k_hat']:.2f} >= {threshold:.2f} (importance sampling "
                "unreliable)"
            )
        if d.get("max_depth_fraction", 0) > MAX_DEPTH_SHARE:
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
            if total > SAMPLE_VARIANCE_MAX:
                inputs.append(
                    f"likelihoods from samples carry Monte Carlo variance sum_i V_i = {total:.2g} "
                    f"> {SAMPLE_VARIANCE_MAX:g} at the fitted population"
                )
                input_hints.append("draw more samples per object, or use wider bins or floor=")
        if lk is not None and support_min is not None:
            low = np.flatnonzero(self.data_support < support_min)
            if low.size:
                inputs.append(
                    f"{low.size} bins have data support below {support_min:g} objects "
                    f"(first: {low[:5].tolist()}): their fractions are set by the prior"
                )
                input_hints.append("merge sparse bins, or read their fractions as the prior's")
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
        _report([(msgs, hints, ConvergenceWarning), (inputs, input_hints, InputWarning)], strict)
        return msgs + inputs

    def _predictive_pvalue(self, lk: Any, n_draws: int) -> float | None:
        # label-shift check of binned inputs (DECISIONS #68); needs J > K
        key = ("predictive_p", int(n_draws))
        if key not in self._cache:
            resp = lk.meta["response"]
            J, K = resp.matrix.shape
            n = np.bincount(lk.meta["measured_bin"], weights=lk.weights, minlength=J)
            Rd = self._response_rows(n_draws)  # an inferred response, drawn with rho
            self._cache[key] = (
                resp.predictive_pvalue(n, self._select_draws(n_draws), Rd) if J > K else None
            )
        return self._cache[key]

    def summary(self, level: float = 0.68, which: str = "fractions") -> Summary:
        """Per-bin posterior summary table (mean, sd, median, interval, ESS, R-hat).

        Parameters
        ----------
        level : float, optional
            Probability inside the central intervals shown.
        which : {"fractions", "rates", "density", "selected"}, optional
            Quantity used: fractions, rates, densities or the selected population.
        """
        which = _which_name(which)
        X = self._which(which)
        lo, hi = _central(X, level)
        cols: dict[str, np.ndarray] = {
            "bin": _bin_labels(self.bins),
            "mean": X.mean(axis=0),
            "sd": X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.zeros(self.K),
            "median": np.median(X, axis=0),
            f"q{100 * (0.5 - level / 2):g}": lo,
            f"q{100 * (0.5 + level / 2):g}": hi,
        }
        mixing = self._mixing_of(which)  # of the quantity summarised (audit O-8)
        cols.update({k: v for k, v in mixing.items() if k != "ess_tail"})
        if self.model is not None:  # objects' worth of likelihood per bin (DECISIONS #41)
            cols["support"] = self.data_support
        flag = " [approximate]" if self.approximate else ""
        source = None if self.model is None else self.model.likelihoods.meta.get("source")
        inputs = "" if source is None else f"; inputs: {source}"
        title = f"binference {which} posterior ({self.method}{flag}, {self.n_draws} draws{inputs})"
        return Summary(cols, title)

    def binning_bias(
        self,
        build: Callable[[Any], Likelihoods] | None = None,
        *,
        order: int = 1,
        resolution: int = 1,
        threshold: float = 2.0,
        prior: Any = None,
        method: str | None = None,
        n_samples: int | None = None,
        rng: RNGLike = None,
        **engine_kwargs: Any,
    ) -> Summary:
        r"""Check the flat-within-bin assumption: refit the same data on a B-spline basis
        over these bins and compare the bin masses (DECISIONS #6, #20).

        ``build(bins_or_basis)`` returns the likelihoods of the fit's data on what it is
        given, e.g. ``lambda b: Likelihoods.from_gaussian(mean, sigma, b)``; likelihoods
        from :meth:`Likelihoods.from_gaussian` rebuild themselves without it. The
        reference is fitted on ``Basis(self.bins, order, resolution)`` with this fit's
        prior (or ``prior``), selection, engine (or ``method``), chains and draws (or
        ``n_samples``), and a Gibbs fit's warm-up and thinning when the engine is the
        same; ``engine_kwargs`` go to the engine.

        The fit's selection keeps only its bin averages, so the reference sees it flat
        within bins; to test a selection's shape within bins as well, attach it to the
        likelihoods ``build`` returns, which then take precedence (a selection from
        injections must be attached that way, fixed).

        Returns a table per bin: ``shift``, the reference's mean fraction minus this
        fit's; ``sd``, this fit's posterior s.d.; ``z = shift / sd`` (NaN where the
        fit has no spread); and, where
        :math:`|z|` exceeds ``threshold``, ``n_crit`` :math:`= W/z^2`, the number of
        objects at which the shift equals the statistical error (which falls as
        :math:`1/\sqrt{W}`): from a pilot fit, the size above which these bins bias
        the result (``inf`` elsewhere). Warns when :math:`|z|` exceeds ``threshold`` in
        any bin: two models of the same data differ by up to about one s.d. even when
        the population is flat within bins (DECISIONS #86), so the default is 2.

        Parameters
        ----------
        build : callable, optional
            Function that rebuilds the likelihoods on a basis; ``None`` works for
            Gaussian inputs.
        order : int, optional
            B-spline order of the reference fit's basis.
        resolution : int, optional
            Components per bin of the reference fit's basis.
        threshold : float, optional
            Shift, in posterior standard deviations, above which a bin is flagged.
        prior : PopulationPrior, optional
            Prior of the reference fit; ``None`` reuses this fit's.
        method : str, optional
            Engine of the reference fit; ``None`` reuses this fit's.
        n_samples : int, optional
            Draws of the reference fit; ``None`` reuses this fit's, except that
            ``method="auto"`` sets its own run length.
        rng : int, Generator or None
            Random number generator.
        **engine_kwargs
            Options for the reference fit's engine.
        """
        from .basis import Basis
        from .inference import METHODS, infer

        model = self._fit_model_only("binning_bias")
        if self.basis is not None or self.bins.is_categorical:
            raise ValueError("binning_bias compares a fit on grid bins with a basis over them.")
        key = str(self.method if method is None else method).lower()
        if key not in ("auto", "nuts", "gibbs", "laplace", "vb", "bootstrap"):
            raise ValueError(
                f"method must be an engine that draws ('auto', 'nuts', 'gibbs', 'laplace', "
                f"'vb' or 'bootstrap'); got {key!r}."
            )
        basis = Basis(self.bins, order=order, resolution=resolution)
        if basis.is_identity:
            raise ValueError(
                "order=0 with resolution=1 is the bins themselves: nothing to compare."
            )
        if build is None:
            lk = _rebuild_gaussian(model.likelihoods, basis)
        else:
            lk = build(basis)
            if lk.basis != basis or lk.N != model.N or not np.array_equal(lk.weights, model._w):
                raise ValueError(
                    "build(basis) must return the fit's objects (same rows and weights) on "
                    "the basis it is given."
                )
        selection = self.selection if lk.selection is None else None  # build's own wins
        if selection is not None and selection.injections is not None:
            raise ValueError(
                "A selection from injections does not move onto a basis: attach a fixed one "
                "(selection.at_mean()) to the likelihoods build returns."
            )
        same = key == self.method
        if METHODS[key].markov:  # draws per chain, as the fit had them
            engine_kwargs.setdefault("n_chains", self.n_chains)
            for name in ("n_warmup", "thin") if same and key == "gibbs" else ():
                if name in self.info:
                    engine_kwargs.setdefault(name, self.info[name])
            n = self.n_draws // self.n_chains
        elif key == "auto":  # its own defaults, as for a new fit
            n = None
        else:
            n = self.n_draws
        ref = infer(
            lk,
            model.prior_spec if prior is None else prior,
            selection=selection,
            method=key,
            n_samples=n if n_samples is None else n_samples,
            rng=rng,
            check=False,
            **engine_kwargs,
        )
        sd, shift = self.std, ref.mean - self.mean
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(sd > 0, shift / sd, np.nan)  # no spread: nothing to compare with
        big = np.abs(z) > threshold
        n_crit = np.where(big, model.W / np.where(big, z, 1.0) ** 2, np.inf)
        if big.any():
            warn(
                f"Binning bias: {big.sum()} bins move by more than {threshold:g} posterior s.d. "
                f"on a basis of order {order} (first: {np.flatnonzero(big)[:5].tolist()}); "
                f"fit with binference.Basis(bins, order={order}) or narrower bins.",
                InputWarning,
            )
        cols = {"bin": _bin_labels(self.bins), "shift": shift, "sd": sd, "z": z, "n_crit": n_crit}
        return Summary(cols, f"binference binning bias (basis of order {order}, {key})")

    def __repr__(self) -> str:
        flag = ", approximate" if self.approximate else ""
        return (
            f"PopulationResult(method={self.method!r}, kind={self.kind!r}, K={self.K}, "
            f"n_chains={self.n_chains}, n_draws={self.n_draws}{flag})"
        )

    # ------------------------------------------------------------- plotting
    def plot(self, **kwargs: Any):
        """Plot the population posterior (see :func:`binference.plotting.plot_population`).

        Parameters
        ----------
        **kwargs
            Passed to :func:`binference.plotting.plot_population`.
        """
        from .plotting import plot_population

        return plot_population(self, **kwargs)

    def plot_corner(self, idx: Sequence[int] | None = None, **kwargs: Any):
        """Corner plot of selected bins (see :func:`binference.plotting.plot_corner`).

        Parameters
        ----------
        idx : sequence of int, optional
            Bins to show; ``None`` chooses the most populated.
        **kwargs
            Further arguments of :func:`~binference.plotting.plot_corner`.
        """
        from .plotting import plot_corner

        return plot_corner(self, idx=idx, **kwargs)

    def plot_trace(self, idx: Sequence[int] | None = None, **kwargs: Any):
        """Trace plots of selected bins (see :func:`binference.plotting.plot_trace`).

        Parameters
        ----------
        idx : sequence of int, optional
            Bins to show; ``None`` chooses the most populated.
        **kwargs
            Further arguments of :func:`~binference.plotting.plot_trace`.
        """
        from .plotting import plot_trace

        return plot_trace(self, idx=idx, **kwargs)

    # ---------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        """Save draws, metadata and bins to a compressed ``.npz`` file.

        Parameters
        ----------
        path : str
            File to write.
        """
        from .io import save_result

        save_result(self, path)

    @classmethod
    def load(cls, path: str) -> PopulationResult:
        """Load a result saved with :meth:`save`.

        Re-:meth:`attach` the likelihoods to use the per-object methods.

        Parameters
        ----------
        path : str
            File written by :meth:`save`.
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
        self._cache: dict[str, Any] = {}
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
        return self._fractions(self._to_bins(self.value))

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

    def _which(self, which: str) -> np.ndarray:
        """The estimate of ``which`` as one draw ``(1, K)``, as for posterior draws."""
        which = _which_name(which)
        if which == "density":
            return self.density()[None, :]
        if which == "rates" and self.rates is None:
            raise ValueError("No rates in this result (a relative selection, or no model).")
        if which == "selected":
            return self.selected[None, :]
        return (self.rho if which == "fractions" else self.rates)[None, :]

    @property
    def selection(self) -> Selection | None:
        """The selection function of the fit (a point estimate is not saved, so it is
        always the model's)."""
        return None if self.model is None else self.model.selection_spec

    @property
    def best(self) -> PointEstimate:
        """The best fit of a point estimate is itself."""
        return self

    @property
    def selected(self) -> np.ndarray:
        """The selected population at the estimate (see :attr:`PopulationResult.selected`)."""
        return self._selected_of(self._weight_draws())[0]

    def _center_x(self) -> np.ndarray:
        model = self._require_model()
        if self.x is not None and np.shape(self.x)[-1] == model.dim:
            return np.asarray(self.x, dtype=float).ravel()
        return model.from_value(self.value)

    def check(self, kkt_max: float = 1e-4, strict: bool = False) -> list[str]:
        """Warn (or raise with ``strict=True``) when the optimiser did not converge or the
        KKT gap (per object; see :meth:`PopulationModel.kkt_gap`) exceeds ``kkt_max``.

        Estimates found by L-BFGS in unconstrained coordinates have no KKT gap (NaN); for
        them the check is ``converged`` (see ``tol`` of :class:`~binference.samplers.MAP`).

        The default is ten times the round-off floor at which MAP stops (DECISIONS #12).
        Returns the messages (empty if the estimate looks fine).

        Parameters
        ----------
        kkt_max : float, optional
            Largest KKT gap that passes.
        strict : bool, optional
            Raise :class:`~binference.DiagnosticsError` instead of warning.
        """
        msgs = []
        if not self.converged:
            msgs.append("the optimiser did not converge")
        if self.kkt_gap > kkt_max:  # inf (an impossible object) fails; NaN passes
            msgs.append(f"KKT gap {self.kkt_gap:.1e} > {kkt_max:.0e}: not at the optimum")
        _report([(msgs, [], ConvergenceWarning)], strict)
        return msgs

    def _weight_draws(self) -> np.ndarray:
        return self._fractions(self.value)[None, :]

    def density(self) -> np.ndarray:
        """Population density per unit bin volume at the optimum."""
        base = self.rates if self.kind == "rates" else self.rho
        return base / self.bins.volumes

    def laplace(self, n_samples: int = 4000, rng: RNGLike = None, **kwargs: Any):
        """Laplace approximation around the mode (see :class:`binference.samplers.Laplace`).

        Parameters
        ----------
        n_samples : int, optional
            Draws of the Laplace approximation.
        rng : int, Generator or None
            Random number generator.
        **kwargs
            Options of :class:`~binference.samplers.Laplace`.
        """
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
