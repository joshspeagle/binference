# API reference

Everything a typical analysis needs is importable from the top-level `pastry`
namespace; priors, engines and helpers live in sub-modules.

## Inputs and the model

```{eval-rst}
.. currentmodule:: pastry

.. autosummary::
   :toctree: generated
   :nosignatures:

   Bins
   Likelihoods
   Selection
   PopulationModel
```

## Running inference

```{eval-rst}
.. currentmodule:: pastry

.. autosummary::
   :toctree: generated
   :nosignatures:

   infer
   PopulationEstimator
```

## Results

```{eval-rst}
.. currentmodule:: pastry

.. autosummary::
   :toctree: generated
   :nosignatures:

   PopulationResult
   PointEstimate
   Summary
```

## Priors

```{eval-rst}
.. currentmodule:: pastry.priors

.. autosummary::
   :toctree: generated
   :nosignatures:

   Dirichlet
   GeneralizedDirichlet
   DirichletTree
   PolyaTree
   LogisticNormal
   LogisticRandomWalk
   LogisticGP
   GammaRates
```

Hyperpriors for the hyperparameters of the logistic priors are parametrised on
the log scale internally; pass a plain number to fix a hyperparameter.

```{eval-rst}
.. currentmodule:: pastry.priors

.. autosummary::
   :toctree: generated
   :nosignatures:

   HalfNormal
   LogNormal
   InverseGamma
   Gamma
   Fixed
```

## Engines

`pastry.infer(..., method=...)` constructs these for you; use them directly for
fine control, generator-style sampling or resuming a run.

```{eval-rst}
.. currentmodule:: pastry.samplers

.. autosummary::
   :toctree: generated
   :nosignatures:

   NUTS
   Gibbs
   MAP
   Laplace
   MeanFieldVB
   Bootstrap
```

## Diagnostics

```{eval-rst}
.. currentmodule:: pastry.diagnostics

.. autosummary::
   :toctree: generated
   :nosignatures:

   rhat
   ess
   mcse
   autocorr_time
   autocorr_function
   psis
   kkt_gap
   missing_information
```

## Simulation and validation

```{eval-rst}
.. currentmodule:: pastry.simulate

.. autosummary::
   :toctree: generated
   :nosignatures:

   mock_1d
   mock_2d
   mock_categorical
   mock_unfolding
   prior_predictive
   population
   Mock
   Truth
```

```{eval-rst}
.. currentmodule:: pastry.validation

.. autosummary::
   :toctree: generated
   :nosignatures:

   exact_posterior
   sbc
   coverage
   ExactPosterior
   SBCResult
   CoverageResult
```

## Plotting and input/output

```{eval-rst}
.. currentmodule:: pastry.plotting

.. autosummary::
   :toctree: generated
   :nosignatures:

   plot_population
   plot_corner
   plot_trace
   plot_individual
   plot_sbc
```

```{eval-rst}
.. currentmodule:: pastry.io

.. autosummary::
   :toctree: generated
   :nosignatures:

   save_result
   load_result
   to_datatree
```

## Warnings

```{eval-rst}
.. currentmodule:: pastry.warnings

.. autosummary::
   :toctree: generated
   :nosignatures:

   PastryWarning
   InputWarning
   ConvergenceWarning
   ApproximationWarning
```
