"""Priors on population fractions (the simplex) and rates.

Conjugate simplex priors (Gibbs, EM and VB all closed-form):

* :class:`Dirichlet` (the default, ``alpha=1``)
* :class:`GeneralizedDirichlet`
* :class:`DirichletTree`
* :class:`PolyaTree`

Smooth logistic-normal priors (sampled with NUTS, hyperparameters inferred):

* :class:`LogisticNormal`
* :class:`LogisticRandomWalk`
* :class:`LogisticGP`

Rates (Poisson process, conjugate):

* :class:`GammaRates`

Hyperpriors for scale parameters: :class:`HalfNormal`, :class:`LogNormal`,
:class:`InverseGamma`, :class:`Gamma`, :class:`Fixed`.
"""

from .base import BoundPrior, Prior
from .dirichlet import Dirichlet, DirichletTree, GeneralizedDirichlet, PolyaTree
from .hyper import Fixed, Gamma, HalfNormal, Hyperprior, InverseGamma, LogNormal
from .logistic import LogisticGP, LogisticNormal, LogisticRandomWalk
from .rates import GammaRates

__all__ = [
    "BoundPrior",
    "Dirichlet",
    "DirichletTree",
    "Fixed",
    "Gamma",
    "GammaRates",
    "GeneralizedDirichlet",
    "HalfNormal",
    "Hyperprior",
    "InverseGamma",
    "LogNormal",
    "LogisticGP",
    "LogisticNormal",
    "LogisticRandomWalk",
    "PolyaTree",
    "Prior",
]
