# What binference promises

What can you expect of binference's answers, and of its interface?

binference aims to be as easy to pick up as scikit-learn. A newcomer gets a correct,
calibrated answer from one call without knowing the internals. An expert can reach and
tune every part. Doing the analysis properly should be easier than the shortcuts it
replaces, such as {term}`stacking` per-object posteriors or counting each object's most
likely class. Eleven principles follow from that aim.

1. **Correct by construction.** The constructors ask what you pass: a likelihood or a
   posterior, under which {term}`interim prior`, as a {term}`mass` or a {term}`density`,
   in which {term}`measure` ({doc}`../guides/inputs`). `Likelihoods.from_table` refuses
   a table whose kind is not stated. It warns when the rows do not look like the
   declared {term}`representation`. A plain `Likelihoods(values, bins)`, or a bare array
   passed to `infer`, is read as likelihoods.
2. **Easy things in one line, hard things possible.** `binference.infer` covers the
   common case. The model and {term}`engine` objects beneath it are public, and so are
   the log densities and gradients that external samplers need.
3. **Exact by default, approximations labelled.** The default `method="auto"` returns
   Gibbs or NUTS draws, or importance-corrected Laplace draws that passed their check.
   `result.approximate` flags an {term}`approximate` result, and
   `result.info["auto_reason"]` says why the engine was chosen
   ({doc}`../guides/engines`). Approximations of the model are measured too:
   `result.binning_bias()` tests the flat-within-bin assumption, and `result.check()`
   reports the Monte Carlo noise of sample inputs.
4. **Self-validating.** `infer` runs `result.check()` after every Gibbs and NUTS fit.
   Other results, and point estimates with their optimality gap, are checked by calling
   `check()`. Simulation-based calibration and coverage tools ship in the package
   ({doc}`../guides/validation`).
5. **A light core.** The core needs only NumPy, SciPy and threadpoolctl; plotting and
   export are optional extras ({doc}`install`).
6. **One way to read a result.** Every engine that draws returns a
   {class}`~binference.PopulationResult` with the same methods. `result.best` gives the
   best fit, with the same outputs.
7. **Everything exposed, with sensible defaults.** Every modelling and engine setting is
   an argument.
8. **Defaults chosen from evidence and reported.** The engine choice of
   `method="auto"` follows rules measured on a benchmark, and the result says which
   rule applied. The {doc}`../reference/defaults` page lists every default.
9. **Your bins define the problem.** A finer B-spline {class}`~binference.Basis`, or a
   measure, is defined on your bins, and results are reported on them.
10. **Uncertain inputs modelled jointly by default.** A selection function estimated
    from {term}`injections`, or a {term}`response matrix` tallied from a labelled
    calibration set, is inferred with the population. Fixing it is an explicit choice.
    Monte Carlo noise in sample inputs is not modelled; `check()` reports it.
11. **Fast**, as measured by the benchmarks ({doc}`../guides/performance`).
