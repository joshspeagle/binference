# API reference

Everything a typical analysis needs is importable from the top-level `binference`
namespace; priors, engines and helpers live in sub-modules. The names are grouped by
what you are doing.

## Inferring a population

```{py:module} binference
```

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

```{py:module} binference.priors
```

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

```{py:module} binference.selection
```

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

   SELECTION_MODES
   SELECTION_SCALES
```

## Engines

```{py:module} binference.engines
```

`infer(..., method=...)` builds one of these for you; use them directly for fine
control, generator-style sampling or resuming a run.

```{eval-rst}
.. currentmodule:: binference.engines

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

```{py:module} binference.inference
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

```{py:module} binference.diagnostics
```

Checks of a single fit: convergence of draws and importance sampling for
approximations. The optimality of a point estimate,
{meth}`~binference.PopulationModel.kkt_gap`, and the missing information,
{meth}`~binference.PopulationModel.missing_information`, are methods of the model.

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
```

```{py:module} binference.validation
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

```{py:module} binference.simulate
```

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

```{py:module} binference.plotting
```

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

```{py:module} binference.io
```

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

```{py:module} binference.warnings
```

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
