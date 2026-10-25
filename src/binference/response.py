r"""Responses of binned measurements to the true value (DECISIONS #27).

A binned measurement records, for each object, the bin :math:`j` of a measured
quantity (a reconstructed energy, a classifier score, a predicted label). Its
response is the matrix

.. math:: R_{jk} = P(\text{measured in bin } j,\ \text{observed} \mid \text{true bin } k),

whose column sums are the efficiencies :math:`\epsilon_k \le 1`. Every object
measured in bin :math:`j` has likelihood row :math:`R_{j\cdot}`
(:meth:`Likelihoods.from_binned`).

A response tallied from events is uncertain: each column, with its lost share
:math:`1 - \epsilon_k`, has a Dirichlet posterior given the tallies, and the engines
infer it jointly with the population (DECISIONS #28, #81).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from ._utils import as_generator, check_positive, frozen, warn
from .bins import Bins
from .priors.dirichlet import _dirichlet_draw
from .selection import Selection
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
    open_ended : bool
        Whether the outer two of 1-D ``measured_bins`` also take every value
        below or above them (:meth:`index`).

    A response tallied from events (:meth:`from_counts` and the constructors
    built on it) records the events generated per true bin, :attr:`generated`,
    which measure how well each column is known. It is :attr:`uncertain` and is
    inferred jointly with the population; :meth:`fixed` takes it as known.
    """

    __slots__ = (
        "_matrix",
        "generated",
        "measured_bins",
        "open_ended",
        "origin",
        "prior",
        "truth_bins",
    )

    def __init__(
        self,
        matrix: Any,
        truth_bins: Bins,
        measured_bins: Bins | None = None,
        *,
        efficiency: Any = None,
        open_ended: bool = False,
    ) -> None:
        self._setup(matrix, truth_bins, measured_bins, efficiency, open_ended, None, "matrix")

    def _setup(
        self,
        matrix,
        truth_bins,
        measured_bins,
        efficiency,
        open_ended,
        generated,
        origin,
        prior=None,
    ):
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
            if eff.shape != (truth_bins.K,) or not np.all((eff >= 0) & (eff <= 1)):
                raise ValueError("efficiency must give one probability per truth bin.")
            if not np.allclose(col[col > 0], 1.0, atol=1e-3):
                warn(
                    "With an explicit efficiency the matrix should be a migration matrix whose "
                    "columns sum to one; these columns do not, so the efficiency may be counted "
                    "twice.",
                    InputWarning,
                )
            R *= eff
            col *= eff
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
        if open_ended and (measured_bins.is_categorical or measured_bins.ndim != 1):
            raise ValueError("open_ended needs 1-D measured_bins.")
        self._matrix = frozen(R)
        self.truth_bins = truth_bins
        self.measured_bins = measured_bins
        self.open_ended = bool(open_ended)
        self.generated = None if generated is None else frozen(np.array(generated, dtype=float))
        self.origin = str(origin)
        self.prior = None if prior is None else float(prior)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in Response.__slots__ and not hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError("Response objects are immutable.")

    def __reduce__(self) -> tuple:
        return _rebuild, (
            self._matrix,
            self.truth_bins,
            self.measured_bins,
            self.open_ended,
            self.generated,
            self.origin,
            self.prior,
        )

    @classmethod
    def from_counts(
        cls,
        counts: Any,
        generated: Any,
        truth_bins: Bins,
        measured_bins: Bins | None = None,
        *,
        open_ended: bool = False,
        origin: str = "counts",
        prior: float = 1.0,
    ) -> Response:
        """A response tallied from events: ``counts[j, k]`` of ``generated[k]`` measured in ``j``.

        Parameters
        ----------
        counts : array_like (J, K)
            Events generated in true bin ``k`` and observed in measured bin ``j``.
        generated : array_like (K,)
            Events generated in each true bin, observed or not.
        truth_bins, measured_bins, open_ended
            As for :class:`Response`.
        origin : str
            Where the tallies come from, as :meth:`PopulationResult.summary` reports it.
        prior : float
            Concentration :math:`\\beta` of the Dirichlet prior on each cell of a column (the
            ``J`` measured bins and the lost share); column :math:`k` has the posterior
            :math:`\\mathrm{Dir}(\\beta + \\text{counts}_{\\cdot k},\\ \\beta + \\text{lost}_k)`.
            The default 1 is uniform; below 1, cells with no events give NUTS long tails
            (DECISIONS #81).
        """
        prior = float(check_positive(prior, "prior"))
        g = np.asarray(generated, dtype=float).ravel()
        if g.shape != (truth_bins.K,) or not np.all(np.isfinite(g)):
            raise ValueError("generated must give one count per truth bin.")
        empty = np.flatnonzero(g <= 0)
        if empty.size:
            raise ValueError(
                f"No events were generated in truth bin(s) {empty[:5].tolist()}, so their "
                "response is unknown; simulate events there or remove those bins."
            )
        out = cls.__new__(cls)
        out._setup(
            np.asarray(counts, dtype=float) / g,
            truth_bins,
            measured_bins,
            None,
            open_ended,
            g,
            origin,
            prior,
        )
        return out

    @classmethod
    def from_simulation(
        cls,
        true_values: Any,
        measured_values: Any,
        truth_bins: Bins,
        measured_bins: Bins,
        *,
        weights: Any = None,
        open_ended: bool = False,
        origin: str = "simulation",
        prior: float = 1.0,
    ) -> Response:
        r"""Tally the response from simulated events.

        Parameters
        ----------
        true_values : array_like
            True value of every generated event, observed or not.
        measured_values : array_like
            Measured value of each event, aligned with ``true_values``; NaN (or,
            unless ``open_ended``, a value outside ``measured_bins``) marks an
            event that was not observed.
        truth_bins, measured_bins, open_ended
            As for :class:`Response`.
        weights : array_like, optional
            Simulation weight of each event. The tallies are then expressed in
            effective events, :math:`(\sum w)^2/\sum w^2` per true bin, so that
            they measure how well each column is known whatever the scale of the
            weights.
        origin, prior : str, float
            As for :meth:`from_counts`.

        Events whose true value lies outside ``truth_bins`` are left out; a
        warning says how many of them were observed, since measurements coming
        from outside the truth range are not modelled.
        """
        k = truth_bins.index(true_values)
        if open_ended:  # the same rule as Response.index
            e = measured_bins.edges[0]
            measured_values = np.clip(np.asarray(measured_values, dtype=float), e[0], e[-1])
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
        counts = np.bincount(j[seen] * K + k[seen], weights=w[seen], minlength=J * K)
        counts = counts.reshape(J, K)
        if weights is not None:  # rescale the tallies to effective events
            w2 = np.bincount(k[inside], weights=w[inside] ** 2, minlength=K)
            n_eff = np.divide(generated**2, w2, out=np.zeros(K), where=w2 > 0)
            counts = counts * np.divide(n_eff, generated, out=np.zeros(K), where=generated > 0)
            generated = n_eff
        return cls.from_counts(
            counts,
            generated,
            truth_bins,
            measured_bins,
            open_ended=open_ended,
            origin=origin,
            prior=prior,
        )

    @classmethod
    def from_labelled(
        cls,
        scores: Any,
        labels: Any,
        measured_bins: Bins | None = None,
        *,
        classes: Any = None,
        n_bins: int = 10,
        prior: float = 1.0,
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
            holding equal numbers of validation objects, the outer two
            open-ended; predicted labels (integers or strings) get one category
            each, which gives the confusion matrix.
        classes : sequence, optional
            The classes, in order (default: the sorted distinct labels).
        n_bins : int
            Number of equal-count score bins when ``measured_bins`` is not given.
        prior : float
            As for :meth:`from_counts`.
        """
        y = np.asarray(labels).ravel()
        s = np.asarray(scores)
        if s.ndim != 1 or s.size != y.size:
            raise ValueError("scores must be one value per labelled object, like labels.")
        truth = Bins.categories(np.unique(y).tolist() if classes is None else classes)
        open_ended = measured_bins is None and s.dtype.kind == "f"
        if open_ended:
            if int(n_bins) < 2:
                raise ValueError(f"n_bins must be at least 2; got {n_bins}.")
            edges = np.unique(np.quantile(s, np.linspace(0.0, 1.0, int(n_bins) + 1)))
            if edges.size < 3:
                raise ValueError("scores take too few distinct values to form score bins.")
            measured_bins = Bins(edges)
        elif measured_bins is None:
            measured_bins = Bins.categories(np.unique(s).tolist())
        return cls.from_simulation(
            y, s, truth, measured_bins, open_ended=open_ended, origin="labelled data", prior=prior
        )

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
        return None if self.generated is None else self._matrix * self.generated

    @property
    def uncertain(self) -> bool:
        """Whether the response is tallied from events and inferred with the population."""
        return self.prior is not None

    def fixed(self) -> Response:
        """This response taken as known at its tallied matrix, a plug-in that ignores the
        tallies' uncertainty in the fit (the predictive check still uses them)."""
        return _rebuild(
            self._matrix,
            self.truth_bins,
            self.measured_bins,
            self.open_ended,
            self.generated,
            self.origin,
        )

    def _dirichlet(self) -> np.ndarray:
        """Dirichlet parameters of the columns given the tallies: ``(J + 1, K)`` with the
        lost share last, or ``(J, K)`` when the tallies record no lost event, so that every
        event is observed by construction (DECISIONS #81)."""
        counts = self.counts
        lost = np.maximum(self.generated - counts.sum(axis=0), 0.0)
        if np.all(lost <= 1e-9 * self.generated):
            return counts + self.prior
        return np.vstack([counts, lost]) + self.prior

    def _mean_columns(self) -> np.ndarray:
        """Posterior means of the columns given the tallies (see :meth:`_dirichlet`)."""
        a = self._dirichlet()
        return a / a.sum(axis=0)

    def _rows(self) -> np.ndarray:
        """The matrix whose rows :meth:`Likelihoods.from_binned` uses: the tallied one, or
        for an inferred response its posterior means, positive wherever events may land."""
        return self._mean_columns()[: self._matrix.shape[0]] if self.uncertain else self._matrix

    def _selection(self):
        """The on-measurement selection of :meth:`_rows`: its efficiency, unless it is one."""
        eff = self._rows().sum(axis=0)
        # column sums may exceed one by rounding (Response allows 1e-6)
        return None if np.allclose(eff, 1.0) else Selection(np.minimum(eff, 1.0), "on_measurement")

    def draw(self, n: int, rng: Any = None) -> np.ndarray:
        """``(n, J, K)`` posterior draws of the matrix given the tallies alone."""
        if not self.uncertain:
            raise ValueError("Only a response tallied from events has draws.")
        draws = _dirichlet_draw(self._dirichlet().T, as_generator(rng), size=n)
        return np.swapaxes(draws, -1, -2)[:, : self._matrix.shape[0]]

    def index(self, values: Any) -> np.ndarray:
        """Measured bin of each value (``-1`` outside the measured bins, or for NaN)."""
        if self.open_ended:
            e = self.measured_bins.edges[0]
            values = np.clip(np.asarray(values, dtype=float), e[0], e[-1])
        return self.measured_bins.index(values)

    def measured_fractions(self, rho: Any) -> np.ndarray:
        """Expected share of observed objects in each measured bin, for true fractions ``rho``."""
        q = self._matrix @ np.asarray(rho, dtype=float)
        return q / q.sum()

    def predictive_pvalue(self, counts: Any, rho_draws: Any, response_draws: Any = None) -> float:
        r"""Posterior predictive p-value of counts per measured bin, over draws of the fractions.

        For each draw, the discrepancy :math:`D = d^\top \Sigma^{-} d` of the
        counts from those the draw predicts, :math:`d = n - N q`, under the
        multinomial covariance of the counts plus, for a tallied response, that
        of its estimated columns; the p-value is the mean of
        :math:`P(\chi^2_{J-1} \ge D)` over draws (DECISIONS #68). A small value
        means the response does not describe the counts; for a classifier, that
        label shift is violated. With ``response_draws`` ``(n_draws, J, K)``, drawn
        with the fractions when the response is inferred jointly, each draw uses its
        own response and the tallies' covariance is left out (DECISIONS #81).
        """
        n = np.asarray(counts, dtype=float).ravel()
        draws = np.atleast_2d(np.asarray(rho_draws, dtype=float))
        tallied = self.generated is not None and response_draws is None  # their covariance
        if response_draws is not None:
            response_draws = np.asarray(response_draws, dtype=float)
            if response_draws.shape != (len(draws), *self._matrix.shape):
                raise ValueError(
                    f"response_draws must have shape ({len(draws)}, *{self._matrix.shape}), "
                    "one response per draw of the fractions."
                )
        if response_draws is None:  # drawn responses give every measured bin some probability
            live = self._matrix.sum(axis=1) > 0
            if np.any(n[~live] > 0):
                return 0.0  # counts where the response allows none
            n, response_draws = n[live], self._matrix[None, live]
        N, J = n.sum(), n.size
        D = np.empty(len(draws))
        for s, rho in enumerate(draws):
            R = response_draws[s if len(response_draws) > 1 else 0]
            eff = R.sum(axis=0)
            scale = eff @ rho
            q = R @ rho / scale
            # covariance diag(a) + V S V^T: the multinomial counts, plus the columns of a
            # tallied response propagated through q = R rho / (eff . rho); the counts sum
            # to N, so the bin with the largest q is dropped and the rest inverted by
            # Woodbury, O(J K^2) per draw
            a, V, S = N * q, q[:, None], np.array([[-N]])
            if tallied:
                c = rho**2 / self.generated / scale**2
                Rc, sc = R @ c, np.sqrt(c)
                a = a + N**2 * Rc
                V = np.column_stack([q, Rc, R * sc - np.outer(q, eff * sc)])
                S = np.zeros((V.shape[1],) * 2)
                S[0, 0], S[0, 1], S[1, 0] = N**2 * Rc.sum() - N, -(N**2), -(N**2)
                S[2:, 2:] = -(N**2) * np.eye(V.shape[1] - 2)
            keep = np.arange(J) != np.argmax(q)  # drop the best-determined bin
            d, a, V = (n - N * q)[keep], a[keep], V[keep]
            if np.any((a <= 0) & (d != 0)):
                D[s] = np.inf
                continue
            a = np.where(a > 0, a, 1.0)
            ad, aV = d / a, V / a[:, None]
            M = np.eye(V.shape[1]) + S @ (V.T @ aV)
            D[s] = d @ ad - (ad @ V) @ np.linalg.solve(M, S @ (V.T @ ad))
        return float(np.mean(stats.chi2.sf(D, J - 1)))

    def __repr__(self) -> str:
        J, K = self._matrix.shape
        return f"Response(J={J} measured bins, K={K} truth bins, from {self.origin})"


def _rebuild(
    matrix, truth_bins, measured_bins, open_ended, generated, origin, prior=None
) -> Response:
    out = Response.__new__(Response)
    out._setup(matrix, truth_bins, measured_bins, None, open_ended, generated, origin, prior)
    return out
