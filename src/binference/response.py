r"""Responses of binned measurements to the true value (DECISIONS #27).

A binned measurement records, for each object, the bin :math:`j` of a measured
quantity (a reconstructed energy, a classifier score, a predicted label). Its
response is the matrix

.. math:: R_{jk} = P(\text{measured in bin } j,\ \text{observed} \mid \text{true bin } k),

whose column sums are the efficiencies :math:`\epsilon_k \le 1`. Every object
measured in bin :math:`j` has likelihood row :math:`R_{j\cdot}`
(:meth:`Likelihoods.from_binned`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._utils import frozen, warn
from .bins import Bins
from .warnings import InputWarning

__all__ = ["Response"]


class Response:
    """The probability of each measured bin, and of being observed, given the true bin.

    Parameters
    ----------
    matrix : array_like (J, K)
        :math:`R_{jk} = P(\\text{measured } j, \\text{observed} \\mid \\text{true } k)`;
        each column sums to the efficiency of its true bin, at most one. With
        ``efficiency`` it is instead a migration matrix
        :math:`P(\\text{measured } j \\mid \\text{true } k, \\text{observed})` whose
        columns sum to one.
    truth_bins : Bins
        The bins of the true value, with their real geometry (``K`` bins).
    measured_bins : Bins, optional
        The bins of the measured quantity (``J`` bins); ``Bins.categories(J)`` by
        default.
    efficiency : array_like (K,), optional
        Probability that an object in each true bin is observed; multiplies the
        columns of a migration matrix.

    Responses tallied from simulated or labelled events
    (:meth:`from_simulation`, :meth:`from_labelled`) keep the tallies
    (:attr:`counts`, :attr:`generated`), which measure how well each column is
    known.
    """

    __slots__ = (
        "_counts",
        "_generated",
        "_matrix",
        "_open",
        "measured_bins",
        "origin",
        "truth_bins",
    )

    def __init__(
        self,
        matrix: Any,
        truth_bins: Bins,
        measured_bins: Bins | None = None,
        *,
        efficiency: Any = None,
    ) -> None:
        R = np.array(matrix, dtype=float)
        if R.ndim != 2 or R.shape[1] != truth_bins.K:
            raise ValueError(
                f"matrix must have shape (J, K) with K={truth_bins.K} truth bins; got {R.shape}."
            )
        if not np.all(np.isfinite(R)) or np.any(R < 0):
            raise ValueError("matrix must be finite and non-negative.")
        col = R.sum(axis=0)
        if efficiency is not None:
            eff = np.asarray(efficiency, dtype=float).ravel()
            if eff.shape != (truth_bins.K,) or np.any(eff < 0) or np.any(eff > 1):
                raise ValueError("efficiency must give one probability per truth bin.")
            if not np.allclose(col[col > 0], 1.0, atol=1e-3):
                warn(
                    "With an explicit efficiency the matrix should be a migration matrix whose "
                    "columns sum to one; these columns do not, so the efficiency may be counted "
                    "twice.",
                    InputWarning,
                )
            R *= eff
            col = R.sum(axis=0)
        if np.any(col > 1.0 + 1e-6):
            raise ValueError(
                "Columns of the response sum to more than one, so they are not probabilities "
                "P(measured j | true k). Divide each column by the number of events generated "
                "in its true bin."
            )
        if measured_bins is None:
            measured_bins = Bins.categories(R.shape[0])
        elif measured_bins.K != R.shape[0]:
            raise ValueError(
                f"measured_bins has {measured_bins.K} bins but matrix has {R.shape[0]} rows."
            )
        object.__setattr__(self, "_matrix", frozen(R))
        object.__setattr__(self, "truth_bins", truth_bins)
        object.__setattr__(self, "measured_bins", measured_bins)
        object.__setattr__(self, "_counts", None)
        object.__setattr__(self, "_generated", None)
        object.__setattr__(self, "origin", "matrix")
        object.__setattr__(self, "_open", False)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("Response objects are immutable.")

    def __setstate__(self, state: Any) -> None:
        for name, value in state[1].items():
            object.__setattr__(
                self, name, frozen(value) if isinstance(value, np.ndarray) else value
            )

    @classmethod
    def from_simulation(
        cls,
        true_values: Any,
        measured_values: Any,
        truth_bins: Bins,
        measured_bins: Bins,
        *,
        weights: Any = None,
    ) -> Response:
        """Tally the response from simulated events.

        Parameters
        ----------
        true_values : array_like
            True value of every generated event, observed or not.
        measured_values : array_like
            Measured value of each event, aligned with ``true_values``; NaN (or a
            value outside ``measured_bins``) marks an event that was not observed.
        truth_bins, measured_bins : Bins
            Bins of the true and the measured quantity.
        weights : array_like, optional
            Simulation weight of each event.

        Events whose true value lies outside ``truth_bins`` are left out; a
        warning says how many of them were observed, since measurements coming
        from outside the truth range are not modelled.
        """
        k = truth_bins.index(true_values)
        j = measured_bins.index(measured_values)
        if j.shape != k.shape:
            raise ValueError("true_values and measured_values must describe the same events.")
        if weights is None:
            w = np.ones(k.size)
        else:
            w = np.asarray(weights, dtype=float).ravel()
            if w.shape != k.shape or not np.all(np.isfinite(w)) or np.any(w < 0):
                raise ValueError("weights must give one finite, non-negative value per event.")
        stray = int(np.count_nonzero((k < 0) & (j >= 0)))
        if stray:
            warn(
                f"{stray} observed events have true values outside the truth bins and are left "
                "out; measurements from outside the truth range are not modelled.",
                InputWarning,
            )
        inside = k >= 0
        K, J = truth_bins.K, measured_bins.K
        generated = np.bincount(k[inside], weights=w[inside], minlength=K)
        seen = inside & (j >= 0)
        counts = np.bincount(j[seen] * K + k[seen], weights=w[seen], minlength=J * K).reshape(J, K)
        empty = np.flatnonzero(generated <= 0)
        if empty.size:
            raise ValueError(
                f"No events were generated in truth bin(s) {empty[:5].tolist()}, so their "
                "response is unknown; simulate events there or remove those bins."
            )
        out = cls(counts / generated, truth_bins, measured_bins)
        object.__setattr__(out, "_counts", frozen(counts))
        object.__setattr__(out, "_generated", frozen(generated))
        object.__setattr__(out, "origin", "simulation")
        return out

    @classmethod
    def from_labelled(
        cls,
        scores: Any,
        labels: Any,
        measured_bins: Bins | None = None,
        *,
        classes: Any = None,
        n_bins: int = 10,
    ) -> Response:
        """Tally the response of a classifier from a labelled validation set.

        Under label shift (the score distribution of each class is the same in
        the validation and the target set) the likelihood of a score given the
        class is valid for any score, calibrated or not (DECISIONS #29, route C).

        Parameters
        ----------
        scores : array_like (n,)
            A real-valued score per validation object (for example the
            classifier's probability of one class), or a predicted label.
        labels : array_like (n,)
            True class of each validation object.
        measured_bins : Bins, optional
            Bins of the score. By default real-valued scores get ``n_bins`` bins
            holding equal numbers of validation objects, the outer two open-ended
            (they take every score below or above the inner cuts); predicted
            labels (integers or strings) get one category each, which gives the
            confusion matrix.
        classes : sequence, optional
            The classes, in order (default: the sorted distinct labels).
        n_bins : int
            Number of equal-count score bins when ``measured_bins`` is not given.
        """
        y = np.asarray(labels).ravel()
        s = np.asarray(scores)
        if s.ndim != 1 or s.size != y.size:
            raise ValueError("scores must be one value per labelled object, like labels.")
        truth = Bins.categories(sorted(set(y.tolist())) if classes is None else classes)
        open_ended = measured_bins is None and s.dtype.kind == "f"
        if measured_bins is None:
            if open_ended:
                if int(n_bins) < 2:
                    raise ValueError(f"n_bins must be at least 2; got {n_bins}.")
                edges = np.unique(np.quantile(s, np.linspace(0.0, 1.0, int(n_bins) + 1)))
                if edges.size < 3:
                    raise ValueError("scores take too few distinct values to form score bins.")
                measured_bins = Bins(edges)
            else:
                measured_bins = Bins.categories(sorted(set(s.tolist())))
        out = cls.from_simulation(y, s, truth, measured_bins)
        object.__setattr__(out, "origin", "labelled")
        object.__setattr__(out, "_open", open_ended)
        return out

    # -------------------------------------------------------------- properties
    @property
    def matrix(self) -> np.ndarray:
        """The response :math:`R_{jk}`, shape ``(J, K)``."""
        return self._matrix

    @property
    def efficiency(self) -> np.ndarray:
        """Probability that an object in each true bin is observed (column sums)."""
        return self._matrix.sum(axis=0)

    @property
    def counts(self) -> np.ndarray | None:
        """Tallied events per measured and true bin ``(J, K)``, or ``None`` for a given matrix."""
        return self._counts

    @property
    def generated(self) -> np.ndarray | None:
        """Tallied events generated per true bin ``(K,)``, or ``None`` for a given matrix."""
        return self._generated

    def index(self, values: Any) -> np.ndarray:
        """Measured bin of each value (``-1`` outside the measured bins)."""
        if not self._open:
            return self.measured_bins.index(values)
        x = np.asarray(values, dtype=float).ravel()
        j = np.clip(np.searchsorted(self.measured_bins.edges[0][1:-1], x, side="right"), 0, None)
        return np.where(np.isfinite(x), j, -1)

    def measured_fractions(self, rho: Any) -> np.ndarray:
        """Expected share of observed objects in each measured bin, for true fractions ``rho``."""
        q = self._matrix @ np.asarray(rho, dtype=float)
        return q / q.sum()

    def __repr__(self) -> str:
        J, K = self._matrix.shape
        return f"Response(J={J} measured bins, K={K} truth bins, from {self.origin})"
