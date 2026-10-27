# API reference

Everything a typical analysis needs is importable from the top-level `binference`
namespace; priors, engines and helpers live in sub-modules.

## Inputs and the model

```{eval-rst}
.. currentmodule:: binference

.. autosummary::
   :toctree: generated
   :nosignatures:

   Bins
   Basis
   Measure
   Density
   Likelihoods
   Response
   Selection
   PopulationModel
```

## Running inference

```{eval-rst}
.. currentmodule:: binference

.. autosummary::
   :toctree: generated
   :nosignatures:

   infer
   PopulationEstimator
```

## Results

```{eval-rst}
.. currentmodule:: binference

.. autosummary::
   :toctree: generated
   :nosignatures:

   PopulationResult
   PointEstimate
   Summary
```

## Priors

```{eval-rst}
.. currentmodule:: binference.priors

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
.. currentmodule:: binference.priors

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

`binference.infer(..., method=...)` constructs these for you; use them directly for
fine control, generator-style sampling or resuming a run.

```{eval-rst}
.. currentmodule:: binference.samplers

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

`method="auto"`, the default, is an engine that picks one of these (DECISIONS #88):

```{eval-rst}
.. currentmodule:: binference.inference

.. autosummary::
   :toctree: generated
   :nosignatures:

   Auto
```

## Diagnostics

```{eval-rst}
.. currentmodule:: binference.diagnostics

.. autosummary::
   :toctree: generated
   :nosignatures:

   rhat
   ess
   mcse
   autocorr_time
   autocorr_function
   psis
   psis_threshold
   importance_ess
   kkt_gap
   missing_information
```

## Simulation and validation

```{eval-rst}
.. currentmodule:: binference.simulate

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
.. currentmodule:: binference.validation

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
.. currentmodule:: binference.plotting

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
.. currentmodule:: binference.io

.. autosummary::
   :toctree: generated
   :nosignatures:

   save_result
   load_result
   result_to_dict
   to_datatree
```

## Warnings and errors

```{eval-rst}
.. currentmodule:: binference.warnings

.. autosummary::
   :toctree: generated
   :nosignatures:

   BinferenceWarning
   InputWarning
   ConvergenceWarning
   ApproximationWarning
   DiagnosticsError
```
