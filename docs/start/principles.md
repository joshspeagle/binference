# What binference promises

What can you expect of binference's answers, and of its interface?

binference aims to be as easy to pick up as scikit-learn. A newcomer gets a correct,
calibrated answer from one call without knowing the internals. An expert can reach and
tune every part. Doing the analysis properly should be easier than the shortcuts it
replaces, such as {term}`stacking` per-object posteriors or counting each object's most
likely class. Eleven principles follow from that aim.

1. **Correct by construction.** The table and density constructors ask whether you
   pass a likelihood or a posterior, and posteriors and samples need their
   {term}`interim prior`. Only a plain array is read as likelihoods without asking
   ({doc}`../guides/inputs/index`).
2. **Easy things in one line, hard things possible.** `binference.infer` covers the
   common case. The model and {term}`engine` objects beneath it are public, and so are
   the log densities and gradients that external samplers need.
3. **Exact by default, approximations labelled.** The default `method="auto"` returns
   exact draws, or importance-corrected draws that passed their check. Any other
   result is flagged {term}`approximate` (`result.approximate`;
   {doc}`../guides/engines`).
4. **Self-validating.** `infer` checks every Gibbs and NUTS fit for convergence. For
   other results, including point estimates, call `check()`
   ({doc}`../guides/checking`).
5. **A light core.** The core needs only NumPy, SciPy and threadpoolctl; plotting and
   export are optional extras ({doc}`install`).
6. **One way to read a result.** Every engine that draws returns a
   {class}`~binference.PopulationResult` with the same methods. `result.best` gives the
   {term}`point estimate`, with the same outputs.
7. **Everything exposed, with sensible defaults.** Every modelling and engine setting is
   an argument.
8. **Defaults chosen from evidence and reported.** The engine choice of
   `method="auto"` follows rules measured on a benchmark, and the result says which
   rule applied. The {doc}`../reference/defaults` page lists the thresholds and
   constants; each parameter's default is in its signature in the API reference.
9. **Your bins define the problem.** A finer B-spline {class}`~binference.Basis` or a
   {term}`measure` refines your bins, and binference reports results on them.
10. **Uncertain inputs modelled jointly by default.** binference infers a
    selection function estimated from {term}`injections`, or a {term}`response matrix`
    tallied from labelled data, jointly with the population unless you fix it.
    It does not model the Monte Carlo noise of sample inputs; `check()` reports it.
11. **Fast**, as measured by the benchmarks ({doc}`../guides/performance`).
