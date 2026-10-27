# Why binference

## What it is for

Many objects are each measured with noise, and the question is about the
population they come from: the distribution of redshifts of a galaxy sample,
the true spectrum behind a detector's counts, the distribution of effects
behind thousands of noisy estimates, the class proportions of an unlabelled
set, the mix of source populations in a catch. Each object carries a
likelihood over bins of the quantity of interest, and the population is a set
of bin fractions $\rho$ (or rates). binference infers $\rho$ with its full
uncertainty, and each object's posterior under it ([theory](../how-it-works/theory.md)).

The aim is that binference is as easy to pick up as scikit-learn: a newcomer
gets a correct, calibrated answer from one call without knowing the
internals, while an expert can reach and tune every part. Doing the analysis
properly should be easier than the shortcuts it replaces, such as adding up
per-object posteriors or counting each object's most likely class.

The model itself is not new; it has been implemented many times, mostly as
paper code for one field's input format ([background](../fields/index.md)).
What binference adds is the design below, applied to every field's inputs.

## Design principles

1. **Correct by construction.** The constructors ask what is passed:
   likelihood or posterior, which interim prior, mass or density, which
   measure ({doc}`../guides/inputs`). `from_table` refuses a table whose
   kind is not stated, and warns when rows do not look like the declared
   representation. The plain `Likelihoods(values, bins)`, and a bare array
   passed to `infer`, are read as likelihood densities.
2. **Easy things in one line, hard things possible.** `binference.infer` for
   the common case; the model and engine objects beneath it; raw log
   densities and gradients for external samplers.
3. **Exact by default, approximations labelled.** The default
   `method="auto"` returns Gibbs or NUTS draws, or importance-corrected
   Laplace draws that passed their check; `result.approximate` flags the
   approximate results, and `result.info["auto_reason"]` says why each
   engine was chosen ({doc}`../guides/engines`). Approximations of the model
   are measured too: `result.binning_bias()` for the flat-within-bin
   assumption, and `result.check()` for the Monte Carlo noise of sample
   inputs.
4. **Self-validating.** `infer` runs `result.check()` after every Gibbs and
   NUTS fit; other results, and point estimates with their optimality gap,
   are checked by calling `check()`. Simulation-based calibration and
   coverage tools ship in the package ({doc}`../guides/validation`).
5. **A light core** ({doc}`install`); plotting and export are optional extras.
6. **One way to read a result.** Every engine that draws returns a
   {class}`~binference.PopulationResult` with the same methods, and
   `result.best` gives the best fit with the same outputs.
7. **Everything exposed, with sensible defaults.** Modelling and engine
   settings are arguments.
8. **Defaults chosen from evidence and reported**, as `method="auto"` is.
9. **The user's bins define the problem.** A finer B-spline basis
   ({class}`~binference.Basis`) or a measure is defined on them, and results
   are reported on them.
10. **Uncertain inputs modelled jointly by default.** A selection function
    estimated from injections, or a response tallied from a labelled
    calibration set, is inferred with the population; fixing it is an
    explicit choice. Monte Carlo noise in sample inputs is not modelled;
    `check()` reports it.
11. **Fast**, as measured by the benchmarks ({doc}`../guides/performance`).

## What it does not do

binference starts where each object already has a likelihood, posterior,
samples or measured bin. Computing those from raw data (fitting photometry,
training a classifier, simulating a detector) is outside its scope.
