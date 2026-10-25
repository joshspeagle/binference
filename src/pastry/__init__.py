"""pastry: population and hierarchical inference with PDFs.

Infer population distributions -- bin fractions or rates, with honest
uncertainties -- from noisy, heteroscedastic per-object PDFs or likelihoods
over bins or categories, together with consistent posteriors for every
individual object.

Quick start
-----------
>>> import pastry
>>> mock = pastry.simulate.mock_1d(500, rng=1)
>>> result = pastry.infer(mock.likelihoods, rng=2, n_samples=200, n_warmup=200)
>>> result.summary()  # doctest: +SKIP

Main entry points
-----------------
* :class:`Bins` -- the support (1-D/N-D grids or categories).
* :class:`Likelihoods` -- per-object likelihoods, with converters from
  posteriors (+ interim priors), samples, Gaussian measurements, response
  matrices and classifier outputs.
* :func:`infer` -- one-call inference (NUTS by default).
* :mod:`pastry.priors` -- Dirichlet family, smoothness priors, rates.
* :mod:`pastry.samplers` -- inference engines.
* :class:`PopulationResult` -- draws, summaries, per-object outputs, plots.
* :mod:`pastry.simulate` and :mod:`pastry.validation` -- mocks, SBC, coverage.
"""

from . import diagnostics, priors, samplers, simulate, validation
from ._version import __version__
from .bins import Bins
from .estimator import PopulationEstimator
from .inference import infer
from .likelihoods import Likelihoods
from .model import PopulationModel
from .results import PointEstimate, PopulationResult, Summary
from .selection import Selection
from .warnings import ApproximationWarning, ConvergenceWarning, InputWarning, PastryWarning

__all__ = [
    "ApproximationWarning",
    "Bins",
    "ConvergenceWarning",
    "InputWarning",
    "Likelihoods",
    "PastryWarning",
    "PointEstimate",
    "PopulationEstimator",
    "PopulationModel",
    "PopulationResult",
    "Selection",
    "Summary",
    "__version__",
    "diagnostics",
    "infer",
    "priors",
    "samplers",
    "simulate",
    "validation",
]
