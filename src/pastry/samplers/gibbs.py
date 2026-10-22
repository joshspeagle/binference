"""Data-augmentation Gibbs sampler for conjugate priors."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from .. import _kernels as kern
from .._utils import RNGLike, is_integer_array, progress, spawn
from ..model import PopulationModel
from ..results import PopulationResult
from .base import Engine, interior_start

__all__ = ["Gibbs"]


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
        or for the given object indices.
    chunk : int
        Objects processed per block when drawing labels (bounds memory).
    """

    name = "gibbs"

    def __init__(
        self,
        model: PopulationModel,
        *,
        n_chains: int = 4,
        n_warmup: int = 500,
        thin: int = 1,
        store_labels: bool | Any = False,
        init: Any = None,
        chunk: int = kern.DEFAULT_CHUNK,
        rng: RNGLike = None,
        progress: bool = False,
    ) -> None:
        super().__init__(model, rng=rng, progress=progress)
        if not model.prior.conjugate:
            raise TypeError(
                f"Gibbs sampling needs a conjugate prior; {model.prior_spec!r} is not. "
                "Use NUTS instead."
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
        if n_chains < 1 or thin < 1:
            raise ValueError("n_chains and thin must be positive.")
        self.n_chains = int(n_chains)
        self.n_warmup = int(n_warmup)
        self.thin = int(thin)
        self.chunk = int(chunk)
        self.init = init
        self._state: list[np.ndarray] | None = None
        self._rngs: list[np.random.Generator] | None = None
        self._warm = False
        self._draws: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []

    def _setup(self) -> None:
        if self._state is not None:
            return
        m = self.model
        self._rngs = spawn(self.rng, self.n_chains)
        if self.init is not None:
            v0 = np.asarray(self.init, dtype=float)
            starts = [v0.copy() for _ in range(self.n_chains)]
        else:
            base = interior_start(m)
            nhat = m.expected_counts(base)
            exposure = m._exposure() if m.kind == "rates" else None
            starts = [m.prior.posterior_sample(nhat, r, exposure=exposure) for r in self._rngs]
        self._state = starts

    def _iterate(self, c: int) -> tuple[np.ndarray, np.ndarray | None]:
        m = self.model
        rng = self._rngs[c]
        value = self._state[c]
        counts, labels = kern.sample_counts(
            m._L, self._w_int, value, rng, chunk=self.chunk, unit_weights=self._unit
        )
        if m.kind == "fractions" and m.selection is not None:
            S = m.selection
            rho = value / value.sum()
            p_det = float(np.clip(S @ rho, 1e-300, 1.0))
            n_missed = rng.negative_binomial(m.W, p_det) if p_det < 1.0 else 0
            if n_missed > 0:
                q = rho * (1.0 - S)
                counts = counts + rng.multinomial(n_missed, q / q.sum())
        exposure = m._exposure() if m.kind == "rates" else None
        self._state[c] = m.prior.posterior_sample(counts, rng, exposure=exposure)
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
        store = self.label_index is not None
        labs = np.empty((C, n_samples, len(self.label_index)), dtype=np.int32) if store else None
        n_done = 0
        try:
            for i in progress(range(n_samples), n_samples, "Gibbs sampling", self.progress):
                for c in range(C):
                    for _ in range(self.thin):
                        value, labels = self._iterate(c)
                    draws[c, i] = value
                    if store:
                        labs[c, i] = labels[self.label_index]
                n_done = i + 1
                yield draws[:, i]
        finally:
            if n_done:
                self._draws.append(draws[:, :n_done])
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
        return PopulationResult(
            self.model,
            D,
            method="gibbs",
            kind=self.model.kind,
            labels=labels,
            label_index=self.label_index,
            info=info,
        )
