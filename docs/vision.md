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

## What it adds

The model itself is not new; it has been implemented many times, mostly as
paper code for one field's input format ([background](background/index.md)).
binference adds:

- **Correct input conventions.** Constructors for likelihoods, posteriors
  under a stated interim prior, samples, functions and distributions, binned
  measurements with a response, and classifier outputs; masses and densities
  are told apart, and bins carry a measure.
- **Calibrated uncertainty.** Exact samplers (NUTS and Gibbs) by default;
  approximations (Laplace, variational, bootstrap) are labelled, and the
  default `method="auto"` keeps the Laplace approximation only when its
  importance check passes.
- **Uncertain inputs propagated.** A selection function estimated from
  injections, or a response tallied from a labelled calibration set, is
  inferred jointly with the population.
- **Beyond flat bins.** A B-spline basis ({class}`~binference.Basis`) removes
  the assumption that the population is flat within each bin, and
  {meth}`~binference.PopulationResult.binning_bias` measures what that
  assumption costs.
- **Validation built in.** Convergence and optimality checks after every fit,
  data-support flags, and simulation-based calibration and coverage tools.
- **Speed.** Sparse likelihoods, chunked kernels, mix-SQP for the
  maximum-likelihood population, and engines chosen from measured rules.

## Design principles

1. **Correct by construction.** Users state what they pass: likelihood or
   posterior, which interim prior, mass or density, which measure. The API
   refuses ambiguity rather than guess.
2. **Easy things in one line, hard things possible.** `binference.infer` for
   the common case; the model and engine objects beneath it; raw log
   densities and gradients for external samplers.
3. **Exact by default, every approximation labelled**, including
   approximations of the model (flat within bins, Monte Carlo inputs) and not
   only of the sampler.
4. **Self-validating.** Checks of the sampler, the inputs and the
   approximations run automatically.
5. **A light core.** NumPy, SciPy and threadpoolctl; plotting and export are
   optional extras.
6. **One way to read a result.** Every engine that draws returns a
   {class}`~binference.PopulationResult` with the same methods, and
   `result.best` gives the best fit with the same outputs.
7. **Everything exposed, with sensible defaults.** Every tunable is an
   argument.
8. **Defaults chosen from evidence and reported.** The engine choice and its
   reason are recorded on the result
   ([how `method="auto"` chooses](user-guide/engines.md)).
9. **The user's bins define the problem.** Internals such as a finer basis or
   a measure adapt to them, and results are reported on them.
10. **Uncertain inputs are modelled jointly by default.** Fixing them is an
    explicit choice.
11. **Fast.** Performance is a feature, measured by the benchmarks.

## What it does not do

binference starts where each object already has a likelihood, posterior,
samples or measured bin. Computing those from raw data (fitting photometry,
training a classifier, simulating a detector) is outside its scope.
