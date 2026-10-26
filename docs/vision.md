# Why binference

## What it is for

Many objects are each measured with noise, and the question is about the
population they come from: the distribution of redshifts of a galaxy sample,
the true spectrum behind a detector's counts, the distribution of effects
behind thousands of noisy estimates, the class proportions of an unlabelled
set, the mix of source populations in a catch. Each object carries a
likelihood over bins of the quantity of interest, and the population is a set
of bin fractions $\rho$ (or rates). binference infers $\rho$ with its full
uncertainty, and each object's posterior under it ([theory](theory.md)).

The aim is that binference is as easy to pick up as scikit-learn: a newcomer
gets a correct, calibrated answer from one call without knowing the
internals, while an expert can reach and tune every part. Doing the analysis
properly should be easier than the shortcuts it replaces, such as adding up
per-object posteriors or counting each object's most likely class.

The model itself is not new; it has been implemented many times, mostly as
paper code for one field's input format ([background](background/index.md)).
What binference adds is the design below, applied to every field's inputs.

## Design principles

1. **Correct by construction.** Users state what they pass: likelihood or
   posterior, which interim prior, mass or density, which measure
   ({doc}`user-guide/inputs`). A table whose kind cannot be told is refused.
   The one shortcut: a bare array passed to `infer` is read as likelihoods.
2. **Easy things in one line, hard things possible.** `binference.infer` for
   the common case; the model and engine objects beneath it; raw log
   densities and gradients for external samplers.
3. **Exact by default, approximations labelled.** The default
   `method="auto"` returns Gibbs or NUTS draws, or importance-corrected
   Laplace draws that passed their check; `result.approximate` flags the
   rest, and `result.info["auto_reason"]` says why each engine was chosen
   ({doc}`user-guide/engines`). Approximations of the model are measured on
   request: `result.binning_bias()` for the flat-within-bin assumption, and
   `result.check()` for the Monte Carlo noise of sample inputs.
4. **Self-validating.** `infer` checks convergence after every Gibbs and NUTS
   fit, point estimates carry their optimality gap, and simulation-based
   calibration and coverage tools ship in the package
   ({doc}`user-guide/validation`).
5. **A light core** ({doc}`install`); plotting and export are optional extras.
6. **One way to read a result.** Every engine that draws returns a
   {class}`~binference.PopulationResult` with the same methods, and
   `result.best` gives the best fit with the same outputs.
7. **Everything exposed, with sensible defaults.** Modelling and engine
   settings are arguments.
8. **Defaults chosen from evidence and reported**, as `method="auto"` is.
9. **The user's bins define the problem.** Internals such as a finer B-spline
   basis ({class}`~binference.Basis`) or a measure adapt to them, and results
   are reported on them.
10. **Uncertain inputs modelled jointly by default.** A selection function
    estimated from injections, or a response tallied from a labelled
    calibration set, is inferred with the population; fixing it is an
    explicit choice. Monte Carlo noise in sample inputs is checked, not
    modelled.
11. **Fast**, as measured by the benchmarks ({doc}`user-guide/performance`).

## What it does not do

binference starts where each object already has a likelihood, posterior,
samples or measured bin. Computing those from raw data (fitting photometry,
training a classifier, simulating a detector) is outside its scope.
