# API reference

Everything a typical analysis needs is importable from the top-level `binference`
namespace; priors, engines and helpers live in sub-modules. The names are grouped by
what you are doing.

(result-task-index)=
## Find a method by task

| Task | Methods | Example |
|---|---|---|
| Convert per-object measurements or posteriors | {meth}`~binference.Likelihoods.from_gaussian`, {meth}`~binference.Likelihoods.from_table`, {meth}`~binference.Likelihoods.from_samples`, {meth}`~binference.Likelihoods.from_binned` | {doc}`Input guides </guides/inputs/index>` |
| Evaluate a proposed population | {meth}`~binference.PopulationModel.log_likelihood`, {meth}`~binference.PopulationModel.responsibilities`, {meth}`~binference.PopulationModel.expected_counts` | {doc}`Three-bin catalogue </tutorials/three-bin-catalogue>` |
| Summarise the population or propagate its uncertainty | {meth}`~binference.PopulationResult.summary`, {meth}`~binference.PopulationResult.interval`, {meth}`~binference.PopulationResult.functional`, {meth}`~binference.PopulationResult.rebin` | [Population results](../../guides/results.md#the-population) |
| Recover true bins and values for the observed objects | {meth}`~binference.PopulationResult.assignment_probabilities`, {meth}`~binference.PopulationResult.sample_assignments`, {meth}`~binference.PopulationResult.sample_values`, {meth}`~binference.PopulationResult.realized_counts` | [Individual objects](../../guides/results.md#individual-objects) |
| Inspect diagnostics or save a fit | {meth}`~binference.PopulationResult.check`, {meth}`~binference.PopulationResult.save`, {meth}`~binference.PopulationResult.load`, {meth}`~binference.PopulationResult.attach` | {doc}`Checking </guides/checking>` and [saving](../../guides/results.md#saving-and-sharing) |

### Shapes of common results

Here $K$ counts true bins, $J$ counts measured bins, and $C$ counts basis
components. An object axis counts the rows selected by `idx`.

| Result | Shape | Meaning |
|---|---|---|
| {attr}`~binference.PopulationResult.samples` | `(n_draws, K)` | Population fractions with chains concatenated. The rows sum to one, including for rate models. |
| {attr}`~binference.PopulationResult.chains` | `(n_chains, n_per_chain, K)` | The same fractions with the chain axis retained. |
| {attr}`~binference.PopulationResult.component_draws` | `(n_chains, n_per_chain, C)` | Model parameters before projection to the bins. With the identity basis, $C=K$. |
| {attr}`~binference.PopulationResult.rates` | `(n_draws, K)` | Expected rates per bin in the model's units. Returns `None` when the total is unavailable. |
| {meth}`~binference.PopulationResult.assignment_probabilities` | `(n_objects, K)` | Probabilities averaged over population draws. Each object's row sums to one. |
| {meth}`~binference.PopulationResult.sample_assignments` | `(n_draws, n_objects)` | One integer bin label per object and draw. Use uncompressed rows for object-level draws. |
| {meth}`~binference.PopulationResult.realized_counts` | `(n_draws, K)` | Draws of the counts of the catalogue's objects in true bins. |
| {meth}`~binference.PopulationResult.response_draws` | `(n_draws, J, K)` | Rows are measured bins. Each column sums to that true bin's efficiency. Requires a jointly inferred response. |

The {doc}`results guide </guides/results>` explains which quantities retain
posterior uncertainty and which also account for latent labels in this catalogue.

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

   Summary
```

```{eval-rst}
.. currentmodule:: binference

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/result.rst

   PopulationResult
   PointEstimate
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
