# API reference

Everything a typical analysis needs is importable from the top-level `binference`
namespace; priors, engines and helpers live in sub-modules. The names are grouped by
what you are doing.

## Inferring a population

`infer` fits a population in one call; the estimator wraps the same fit for
scikit-learn, and the model holds the inputs of a fit for reuse.

```{eval-rst}
.. currentmodule:: binference

.. autosummary::
   :toctree: generated
   :nosignatures:

   infer
   PopulationEstimator
   PopulationModel
```

## Inputs and bins

The bins, how volume is measured in them, and the per-object likelihoods and response
that a fit takes.

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
```

## Priors

Priors on the population fractions or rates; pass one to `infer` as `prior=`.

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

A new prior subclasses one of these base classes; `bind` returns its numerical form.

```{eval-rst}
.. currentmodule:: binference.priors

.. autosummary::
   :toctree: generated
   :nosignatures:

   PopulationPrior
   BoundPrior
   Hyperprior
```

## Selection

Which objects enter the sample, given their true or measured values; pass a
`Selection` to `infer` as `selection=`.

```{eval-rst}
.. currentmodule:: binference

.. autosummary::
   :toctree: generated
   :nosignatures:

   Selection
```

```{eval-rst}
.. currentmodule:: binference.selection

.. autosummary::
   :toctree: generated
   :nosignatures:

   as_selection
   SELECTION_MODES
   SELECTION_SCALES
```

## Engines

`infer(..., method=...)` builds one of these for you; use them directly for fine
control, generator-style sampling or resuming a run.

```{eval-rst}
.. currentmodule:: binference.samplers

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/engine.rst

   NUTS
   Gibbs
   MAP
   Laplace
   MeanFieldVB
   Bootstrap
   Engine
```

`method="auto"`, the default, is an engine that picks one of the others; `METHODS`
maps each `method=` name to its engine.

```{eval-rst}
.. currentmodule:: binference.inference

.. autosummary::
   :toctree: generated
   :nosignatures:

   Auto
   METHODS
```

## Results and summaries

What a fit returns: posterior draws or a point estimate, and their summary table.

```{eval-rst}
.. currentmodule:: binference

.. autosummary::
   :toctree: generated
   :nosignatures:

   PopulationResult
   PointEstimate
   Summary
```

## Diagnostics and validation

Checks of a single fit: convergence of draws, importance sampling for approximations,
and optimality of a point estimate.

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

Checks of a method: exact posteriors for small problems, simulation-based calibration
and coverage.

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

## Simulation

Mock data sets with a known truth, for trying a method before it meets real data.

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

## Plotting

Figures of a population, its draws, the per-object posteriors and calibration.

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
   SAMPLE_COLOR
```

## Saving and loading

Write a result to disk, read it back, or convert it to an xarray `DataTree` for ArviZ.

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

## Warnings

The warnings binference emits, and the error a strict convergence check raises.

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
