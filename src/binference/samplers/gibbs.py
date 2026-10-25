"""Data-augmentation Gibbs sampler for conjugate priors."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from .. import _kernels as kern
from .._utils import RNGLike, check_positive, is_integer_array, progress, spawn
from ..model import PopulationModel
from ..results import PopulationResult
from .base import Engine, interior_start

__all__ = ["Gibbs"]

#: smallest detection probability a chain holds (Beta draws can underflow to zero)
S_FLOOR = 1e-12


class Gibbs(Engine):
    r"""Data-augmentation Gibbs sampler (the "hierarchical" sampler).

    Alternates (i) drawing every object's bin label
    :math:`z_i \sim \mathrm{Cat}(r_i(\rho))` and (ii) drawing the population from
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
        Store joint label draws for all objects (``True``; unit weights only)
        or for the given object indices (sorted and deduplicated, as
        ``result.label_index`` records).
    init : array_like (K,), optional
        Positive fractions or rates around which the chains start: each chain
        starts from its own draw of the population's conditional given the
        expected counts there, as by default around a few EM steps.
    chunk : int or None
        Objects processed per block when drawing labels (bounds memory);
        ``None`` processes all at once.
    """

    name = "gibbs"
    joint_selection = True

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
        if not model.prior.conjugate:
            raise TypeError(
                f"Gibbs sampling needs a conjugate prior; {model.prior_spec!r} is not. "
                "Use NUTS instead."
            )
        if model.selection_uncertain and model.selection_spec.injections[3] is not None:
            raise TypeError(
                "Gibbs sampling infers independent Betas only; a smooth injection model is not "
                "conjugate. Use NUTS instead."
            )
        w = model.likelihoods.weights
        if not is_integer_array(w):
            raise ValueError(
                "Gibbs sampling needs integer object weights (multiplicities); "
                "use NUTS for non-integer weights."
            )
        self._w_int = np.round(w)
        self._unit = bool(np.all(self._w_int == 1))
        if store_labels is True:
            if not self._unit:
                raise ValueError("store_labels=True needs unit weights; pass object indices.")
            self.label_index = np.arange(model.N)
        elif store_labels is False or store_labels is None:
            self.label_index = None
        else:
            self.label_index = np.unique(np.atleast_1d(np.asarray(store_labels, dtype=int)))
            if not self._unit:
                raise ValueError("Storing labels needs unit object weights.")
            if self.label_index.size and not (
                0 <= self.label_index[0] and self.label_index[-1] < model.N
            ):
                raise ValueError(f"store_labels must be object indices in 0..{model.N - 1}.")
        if init is not None:
            init = check_positive(init, "init")
            if init.shape != (model.K,):
                raise ValueError(f"init must have shape ({model.K},); got {init.shape}.")
        if n_chains < 1 or thin < 1:
            raise ValueError("n_chains and thin must be positive.")
        self.n_chains = int(n_chains)
        self.n_warmup = int(n_warmup)
        self.thin = int(thin)
        self.chunk = None if chunk is None else int(chunk)
        self.init = init
        self._state: list[np.ndarray] | None = None
        self._rngs: list[np.random.Generator] | None = None
        self._warm = False
        self._draws: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []
        self._sel_draws: list[np.ndarray] = []

    def _setup(self) -> None:
        if self._state is not None:
            return
        m = self.model
        self._rngs = spawn(self.rng, self.n_chains)
        # overdispersed starts: each chain draws from the conditional given expected counts
        base = interior_start(m) if self.init is None else self.init
        nhat = m._complete_counts(base, m.expected_counts(base))
        exposure = m._S if m.kind == "rates" else None
        self._state = [m.prior.posterior_sample(nhat, r, exposure=exposure) for r in self._rngs]
        if m.selection_uncertain:  # each chain starts from a draw given the injections
            self._sel = [np.maximum(m.selection_spec.draw(1, r)[0], S_FLOOR) for r in self._rngs]

    def _iterate(self, c: int) -> tuple[np.ndarray, np.ndarray | None]:
        m = self.model
        rng = self._rngs[c]
        value = self._state[c]
        if m.selection_uncertain:
            return self._iterate_joint(c, m, rng, value)
        counts, labels = kern.sample_counts(
            m._L, self._w_int, value, rng, chunk=self.chunk, unit_weights=self._unit
        )
        counts = m._draw_missed(value, counts, rng)
        exposure = m._S if m.kind == "rates" else None
        self._state[c] = m.prior.posterior_sample(counts, rng, exposure=exposure)
        return self._state[c], labels

    def _iterate_joint(self, c, m, rng, value) -> tuple[np.ndarray, np.ndarray | None]:
        # labels with the current S, the missed objects with the absolute S, then rho and
        # S from their conjugate conditionals (DECISIONS #79)
        S = self._sel[c]  # on the truth: rows of L_i scaled by S, detections are successes
        counts, labels = kern.sample_counts(
            m.likelihoods.values,
            self._w_int,
            value * S,
            rng,
            chunk=self.chunk,
            unit_weights=self._unit,
        )
        missed = m._draw_missed(value, counts, rng, S=S) - counts  # with the absolute S
        injected, recovered, (a, b), _ = m.selection_spec.injections
        S = rng.beta(a + recovered + counts, b + injected - recovered + missed)
        self._sel[c] = np.maximum(S, S_FLOOR)  # an underflow to zero would void its bin
        self._state[c] = m.prior.posterior_sample(counts + missed, rng)
        return self._state[c], labels

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
        """Generator over kept iterations, yielding ``(n_chains, K)`` values."""
        self.warmup()
        C, K = self.n_chains, self.model.K
        draws = np.empty((C, n_samples, K))
        joint = self.model.selection_uncertain
        sel = np.empty((C, n_samples, K)) if joint else None
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
                        sel[c, i] = self._sel[c]
                    if store:
                        labs[c, i] = labels[self.label_index]
                n_done = i + 1
                yield draws[:, i]
        finally:
            if n_done:
                self._draws.append(draws[:, :n_done])
                if joint:
                    self._sel_draws.append(sel[:, :n_done])
                if store:
                    self._labels.append(labs[:, :n_done])

    def run(self, n_samples: int = 1000) -> PopulationResult:
        """Burn in (first call only), draw ``n_samples`` per chain, and return all draws so far."""
        for _ in self.sample(int(n_samples)):
            pass
        return self.result

    @property
    def result(self) -> PopulationResult:
        if not self._draws:
            raise RuntimeError("No draws yet; call run().")
        D = np.concatenate(self._draws, axis=1)
        labels = np.concatenate(self._labels, axis=1) if self._labels else None
        info = {"thin": self.thin, "n_warmup": self.n_warmup}
        if self._sel_draws:  # the selection inferred with the population (DECISIONS #79)
            info["selection_draws"] = np.concatenate(self._sel_draws, axis=1)
        return PopulationResult(
            self.model,
            D,
            method="gibbs",
            kind=self.model.kind,
            labels=labels,
            label_index=self.label_index,
            info=info,
        )
