"""Data-augmentation Gibbs sampler for conjugate priors."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from .. import _kernels as kern
from .._utils import RNGLike, check_nonnegative, is_integer_array, progress, spawn
from ..model import PopulationModel
from ..priors.dirichlet import _dirichlet_draw
from ..results import PopulationResult
from ..selection import _beta_posterior, _draw_beta
from .base import MarkovEngine, check_init, interior_start

__all__ = ["Gibbs"]


class Gibbs(MarkovEngine):
    r"""Data-augmentation Gibbs sampler (the "hierarchical" sampler).

    Alternates (i) drawing every object's bin label
    :math:`k_i \sim \mathrm{Cat}(r_i(\rho))` and (ii) drawing the population from
    its conjugate conditional given the counts (Dirichlet family:
    :math:`\rho \mid n \sim \mathrm{Dir}(\alpha + n)`; rates:
    :math:`\lambda_k \mid n \sim \mathrm{Gamma}(a_k + n_k, b_k + S_k)`). With a
    selection function on fractions, the number of undetected objects and their
    bins are imputed as well (negative-binomial augmentation), keeping every
    step conjugate.

    The marginal posterior of the population is the same as NUTS's; Gibbs
    additionally yields joint draws of the labels. It mixes slowly when
    per-object PDFs are broad compared with the bins (a large fraction of
    missing information; see :meth:`PopulationModel.missing_information`), so
    check the effective sample sizes.

    Parameters
    ----------
    model : PopulationModel
        Must use a conjugate prior and integer object weights.
    n_chains : int
        Independent chains (default 4).
    n_warmup : int
        Burn-in iterations per chain.
    thin : int
        Keep every ``thin``-th draw.
    store_labels : bool or array_like of int
        Store joint draws of the objects' bins. ``True`` stores them for all
        objects, an array for the given object indices (sorted and deduplicated,
        as ``result.label_index`` records). Either needs unit object weights
        (uncompressed likelihoods).
    init : array_like (K,), optional
        Positive fractions or rates around which the chains start. Each chain
        starts from its own draw of the population's conditional given the
        expected counts there, as by default around a few EM steps.
    chunk : int or None
        Objects processed per block when drawing labels from dense likelihoods, which
        bounds memory; ``None`` processes all at once. Sparse likelihoods are processed at once,
        in memory proportional to their nonzeros.
    rng : int, numpy.random.Generator, or None
        Random number generator.
    progress : bool
        Show a progress bar.
    """

    joint = True

    def __init__(
        self,
        model: PopulationModel,
        *,
        n_chains: int = 4,
        n_warmup: int = 500,
        thin: int = 1,
        store_labels: bool | Any = False,
        init: Any = None,
        chunk: int | None = kern.DEFAULT_CHUNK,
        rng: RNGLike = None,
        progress: bool = False,
    ) -> None:
        super().__init__(model, rng=rng, progress=progress)
        self._w_int = np.round(model.likelihoods.weights)
        self._unit = bool(np.all(self._w_int == 1))
        if store_labels is True:
            self.label_index = np.arange(model.N)
        elif store_labels is False or store_labels is None:
            self.label_index = None
        else:
            self.label_index = np.unique(np.atleast_1d(np.asarray(store_labels, dtype=int)))
            if self.label_index.size and not (
                0 <= self.label_index[0] and self.label_index[-1] < model.N
            ):
                raise ValueError(f"store_labels must be object indices in 0..{model.N - 1}.")
        if self.label_index is not None and model.response is not None:
            raise ValueError(
                "store_labels is not available with an inferred response (objects are counts "
                "per measured bin); use result.sample_assignments()."
            )
        # with multiplicities m_i each row draws Mult(m_i, r_i) counts, and no labels
        if self.label_index is not None and not self._unit:
            raise ValueError("store_labels needs unit object weights (uncompressed likelihoods).")
        if n_chains < 1 or thin < 1:
            raise ValueError("n_chains and thin must be positive.")
        self.n_chains = int(n_chains)
        self.n_warmup = int(check_nonnegative(n_warmup, "n_warmup"))
        self.thin = int(thin)
        self.chunk = None if chunk is None else int(chunk)
        self.init = check_init(init, model)
        self._state: list[np.ndarray] | None = None
        self._rngs: list[np.random.Generator] | None = None
        self._labels: list[np.ndarray] = []
        # a selection or response inferred with the population: its current value per chain
        self._extra: list[np.ndarray] | None = None
        self._extra_draws: list[np.ndarray] = []

    @classmethod
    def unsupported(cls, model: PopulationModel) -> str | None:
        if not model.prior.conjugate:
            return (
                f"Gibbs sampling needs a conjugate prior; {model.prior_spec!r} is not. "
                "Use NUTS instead."
            )
        if model.selection_uncertain and model.selection_spec.injections.smooth is not None:
            return (
                "Gibbs sampling infers independent Betas only; a smooth injection model is not "
                "conjugate. Use NUTS instead."
            )
        if not is_integer_array(model.likelihoods.weights):
            return (
                "Gibbs sampling needs integer object weights (multiplicities); "
                "use NUTS for non-integer weights."
            )
        return super().unsupported(model)

    def _setup(self) -> None:
        if self._state is not None:
            return
        m = self.model
        self._rngs = spawn(self.rng, self.n_chains)
        # overdispersed starts: each chain draws from the conditional given expected counts
        base = interior_start(m) if self.init is None else self.init
        nhat = m._complete_counts(base, m.expected_counts(base))
        self._state = [m.prior.posterior_sample(nhat, r, exposure=m._exposure) for r in self._rngs]
        # each chain starts from a draw given the injections or the tallies
        if m.selection_uncertain:
            self._extra = [m.selection_spec.draw(1, r)[0] for r in self._rngs]  # floored
        elif m.response is not None:
            self._extra = [m.response.draw(1, r)[0] for r in self._rngs]

    def _iterate(self, c: int) -> tuple[np.ndarray, np.ndarray | None]:
        m = self.model
        rng = self._rngs[c]
        value = self._state[c]
        if m.selection_uncertain:
            return self._iterate_joint(c, m, rng, value)
        if m.response is not None:
            return self._iterate_response(c, m, rng, value)
        counts, labels = kern.sample_counts(
            m._L, self._w_int, value, rng, chunk=self.chunk, unit_weights=self._unit
        )
        counts = m._draw_missed(value, counts, rng)
        self._state[c] = m.prior.posterior_sample(counts, rng, exposure=m._exposure)
        return self._state[c], labels

    def _iterate_joint(self, c, m, rng, value) -> tuple[np.ndarray, np.ndarray | None]:
        # labels with the current S, the missed objects with the absolute S, then rho and
        # S from their conjugate conditionals (DECISIONS #79)
        S = self._extra[c]  # on the truth: rows of L_i scaled by S, detections are successes
        counts, labels = kern.sample_counts(
            m.likelihoods.values,
            self._w_int,
            value * S,
            rng,
            chunk=self.chunk,
            unit_weights=self._unit,
        )
        missed = m._draw_missed(value, counts, rng, S=S) - counts  # with the absolute S
        s, f = _beta_posterior(m.selection_spec.injections)
        self._extra[c] = _draw_beta(s + counts, f + missed, rng)  # floored
        self._state[c] = m.prior.posterior_sample(counts + missed, rng)
        return self._state[c], labels

    def _iterate_response(self, c, m, rng, value) -> tuple[np.ndarray, None]:
        # each measured bin's objects split over the truth bins, the missed objects with the
        # current efficiency, then rho and the columns of R (with their lost shares) from
        # their Dirichlet conditionals (DECISIONS #81)
        R = self._extra[c]
        P = R[m._mb] * value  # one row per measured bin
        split = rng.multinomial(self._w_int.astype(np.int64), P / P.sum(axis=1)[:, None])
        counts = split.sum(axis=0)
        J = R.shape[0]
        A = m._alpha.copy()  # (J + 1, K) with the lost share last, or (J, K) without one
        np.add.at(A, m._mb, split)
        missed = np.zeros_like(counts)
        if A.shape[0] > J:
            missed = m._draw_missed(value, counts, rng, S=R.sum(axis=0)) - counts
            A[-1] += missed
        self._extra[c] = _dirichlet_draw(A.T, rng).T[:J]
        self._state[c] = m.prior.posterior_sample(counts + missed, rng)
        return self._state[c], None

    def warmup(self) -> None:
        """Run burn-in if it has not been run yet."""
        self._setup()
        if self._warm:
            return
        for _ in progress(range(self.n_warmup), self.n_warmup, "Gibbs burn-in", self.progress):
            for c in range(self.n_chains):
                self._iterate(c)
        self._warm = True

    def sample(self, n_samples: int) -> Iterator[np.ndarray]:
        """Generator over kept iterations, yielding ``(n_chains, K)`` values.

        Parameters
        ----------
        n_samples : int
            Kept iterations per chain.
        """
        self.warmup()
        C, K = self.n_chains, self.model.K
        draws = np.empty((C, n_samples, K))
        joint = self._extra is not None
        extra = np.empty((C, n_samples, *self._extra[0].shape)) if joint else None
        store = self.label_index is not None
        labs = np.empty((C, n_samples, len(self.label_index)), dtype=np.int32) if store else None
        n_done = 0
        try:
            for i in progress(range(n_samples), n_samples, "Gibbs sampling", self.progress):
                for c in range(C):
                    for _ in range(self.thin):
                        value, labels = self._iterate(c)
                    draws[c, i] = value
                    if joint:
                        extra[c, i] = self._extra[c]
                    if store:
                        labs[c, i] = labels[self.label_index]
                n_done = i + 1
                yield draws[:, i]
        finally:
            if n_done:
                self._draws.append(draws[:, :n_done])
                if joint:
                    self._extra_draws.append(extra[:, :n_done])
                if store:
                    self._labels.append(labs[:, :n_done])

    @property
    def result(self) -> PopulationResult:
        D = self._all_draws()
        labels = np.concatenate(self._labels, axis=1) if self._labels else None
        info = {"thin": self.thin, "n_warmup": self.n_warmup}
        if self._extra_draws:  # inferred with the population (DECISIONS #79, #81)
            key = "selection_draws" if self.model.selection_uncertain else "response_draws"
            info[key] = np.concatenate(self._extra_draws, axis=1)
        return PopulationResult(
            self.model,
            D,
            method="gibbs",
            labels=labels,
            label_index=self.label_index,
            info=info,
        )
