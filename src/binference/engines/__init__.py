"""Inference engines.

* :class:`NUTS` -- No-U-Turn Sampler on the exact posterior (any prior).
* :class:`Gibbs` -- data-augmentation Gibbs sampler (conjugate priors; joint labels).
* :class:`MAP` -- posterior mode / maximum likelihood (mix-SQP, SQUAREM-EM or L-BFGS).
* :class:`Laplace` -- Gaussian approximation with PSIS importance correction.
* :class:`MeanFieldVB` -- mean-field variational Bayes (approximate).
* :class:`Bootstrap` -- bootstrap of the MAP estimator (approximate).
"""

from .base import Engine
from .bootstrap import Bootstrap
from .gibbs import Gibbs
from .laplace import Laplace
from .nuts import NUTS
from .optimize import MAP
from .vb import MeanFieldVB

__all__ = ["MAP", "NUTS", "Bootstrap", "Engine", "Gibbs", "Laplace", "MeanFieldVB"]
