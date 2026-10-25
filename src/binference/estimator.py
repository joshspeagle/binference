"""A scikit-learn style estimator wrapper around :func:`binference.infer`."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from ._utils import RNGLike, as_generator
from .bins import Bins
from .inference import infer
from .likelihoods import Likelihoods
from .priors import PopulationPrior
from .results import PointEstimate, PopulationResult

__all__ = ["PopulationEstimator"]


class PopulationEstimator:
    """Population inference with the scikit-learn estimator conventions.

    ``fit(X)`` infers the population from per-object likelihoods ``X`` (a
    :class:`~binference.Likelihoods` or an ``(N, K)`` array) and stores the result
    in ``result_``. ``predict_proba(X)`` gives per-object posteriors over bins
    for (new) objects under the inferred population, and ``score(X)`` the mean
    log posterior-predictive density per object, which can be used to compare
    priors or binnings by cross-validation. The estimator follows the
    scikit-learn ``get_params``/``set_params`` contract without depending on
    scikit-learn.

    Parameters
    ----------
    prior : PopulationPrior, optional
        Population prior (default ``Dirichlet(1.0)``).
    method : str
        Inference engine (see :func:`binference.infer`).
    selection : array_like (K,), callable or Selection, optional
        Selection function.
    selection_mode : {"on_measurement", "on_truth"}, optional
        Required with an array or callable ``selection``.
    bins : Bins, optional
        Bins for raw array inputs.
    n_samples : int, optional
        Draws per chain / in total.
    rng : int, Generator, or None
        Random number generator.
    engine_kwargs : dict, optional
        Extra keyword arguments for the engine.
    """

    def __init__(
        self,
        prior: PopulationPrior | None = None,
        *,
        method: str = "nuts",
        selection: Any = None,
        selection_mode: str | None = None,
        bins: Bins | None = None,
        n_samples: int | None = None,
        rng: RNGLike = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.prior = prior
        self.method = method
        self.selection = selection
        self.selection_mode = selection_mode
        self.bins = bins
        self.n_samples = n_samples
        self.rng = rng
        self.engine_kwargs = engine_kwargs

    # -------------------------------------------------- sklearn contract
    @classmethod
    def _get_param_names(cls) -> list[str]:
        sig = inspect.signature(cls.__init__)
        return [p for p in sig.parameters if p != "self"]

    def __sklearn_tags__(self):
        # scikit-learn >= 1.6 asks every estimator for its tags; X is all it needs
        from sklearn.utils import Tags, TargetTags

        return Tags(estimator_type=None, target_tags=TargetTags(required=False))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Constructor parameters (scikit-learn convention)."""
        return {name: getattr(self, name) for name in self._get_param_names()}

    def set_params(self, **params: Any) -> PopulationEstimator:
        """Set constructor parameters (scikit-learn convention)."""
        valid = set(self._get_param_names())
        for k, v in params.items():
            if k not in valid:
                raise ValueError(f"Invalid parameter {k!r} for {type(self).__name__}.")
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items() if v is not None)
        return f"{type(self).__name__}({args})"

    # ------------------------------------------------------------- API
    def fit(self, X: Any, y: Any = None) -> PopulationEstimator:
        """Infer the population from likelihoods ``X``; ``y`` is ignored."""
        kwargs = dict(self.engine_kwargs or {})
        self.result_: PopulationResult | PointEstimate = infer(
            X,
            self.prior,
            bins=self.bins if not isinstance(X, Likelihoods) else None,
            selection=self.selection,
            selection_mode=self.selection_mode,
            method=self.method,
            n_samples=self.n_samples,
            rng=self.rng,
            **kwargs,
        )
        res = self.result_
        self.population_ = res.rho if isinstance(res, PointEstimate) else res.mean
        self.n_features_in_ = int(self.population_.size)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "result_"):
            raise RuntimeError(f"This {type(self).__name__} is not fitted yet; call fit().")

    def predict_proba(self, X: Any) -> np.ndarray:
        """Per-object posterior over bins under the inferred population."""
        self._check_fitted()
        return self.result_.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        """Most probable bin of each object under the inferred population."""
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: Any, y: Any = None) -> float:
        """Mean log posterior-predictive density per object (higher is better)."""
        self._check_fitted()
        lp = self.result_.log_predictive(X)
        w = X.weights if isinstance(X, Likelihoods) else np.ones(lp.size)
        return float(np.dot(w, lp) / w.sum())

    def sample(self, n: int = 1, rng: RNGLike = None) -> np.ndarray:
        """Draws of the population fractions from the fitted posterior."""
        self._check_fitted()
        res = self.result_
        if isinstance(res, PointEstimate):
            return np.tile(res.rho, (n, 1))
        idx = as_generator(rng).integers(0, res.n_draws, size=n)
        return res.samples[idx]
