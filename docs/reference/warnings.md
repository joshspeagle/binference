# Warnings and errors

binference reports a questionable input, a fit that did not converge, and an
approximate result with three warning classes. All three derive from
`binference.BinferenceWarning`, so one filter covers them:

```python
import warnings
import binference

warnings.simplefilter("ignore", binference.BinferenceWarning)
```

Each warning points at the line of your code that called binference. This page lists
every message, grouped by class. Text in angle brackets, such as `<n>`, stands for a
value filled in at run time. The thresholds named here are on the {doc}`defaults`
page.

## `InputWarning`

The inputs look inconsistent with the declared conventions. The fit still runs, but its
result may be biased.

`The edges of axis <d> are geometrically spaced but the measure is linear, ...`
: **When.** `Bins(edges)` gets geometrically spaced edges and no `measure`.
  **Cause.** The population is then flat in $z$ within each bin, not in $\log z$.
  **Fix.** Pass `measure="log"` or use `Bins.log` if the population should be flat in
  $\log z$; pass `measure="linear"` to keep flat in $z$ without the warning.

`Non-integer weights define a pseudo-likelihood: the posterior is not a proper posterior ...`
: **When.** Weights are not integers and `fractional_weights=True`.
  **Cause.** Weights are multiplicities, so fractional ones do not give a proper
  posterior.
  **Fix.** Use integer multiplicities where possible. Otherwise read the uncertainties
  as uncalibrated.

`<n> of <N> objects have more than <share> of their likelihood mass outside the bins ...`
: **When.** `from_gaussian`, `from_density` or `from_samples` find that many objects lie
  mostly outside the bins.
  **Cause.** The bins do not cover the measurements.
  **Fix.** Widen the bins, or check `lk.meta["outside_fraction"]`. Pass
  `warn_outside=False` if the loss is intended.

`Dropping <n> object(s) with zero likelihood in every bin.`
: **When.** `drop_empty=True` and some rows are all zero.
  **Cause.** Those objects lie outside the bins or have no support there.
  **Fix.** Widen the bins, or accept that these objects are left out.

`<n> of <N> objects have fewer than <min_samples> effective samples inside the bins; ...`
: **When.** `from_samples` finds that many objects have few samples inside the bins.
  **Cause.** Their histogrammed likelihoods are noisy.
  **Fix.** Draw more samples per object, widen the bins, or pass `floor=`.

`<n> values lie outside the measured bins and are left out.`
: **When.** `from_binned(response, values=...)` gets values outside the measured bins.
  **Fix.** Use `open_ended=True` on the response, or widen its measured bins.

`Most posterior rows integrate to one only if interpreted as '<other>' values; ...`
: **When.** Posteriors on bins of unequal volume sum to one only under the other
  representation.
  **Cause.** `representation` is probably wrong, which biases the fractions toward wide
  bins.
  **Fix.** Pass `representation="<other>"`. See
  [masses versus densities](../how-it-works/model.md#masses-versus-densities).

`Bin averages did not reach tol=<tol> in <n> bin(s) after <max_depth> refinements; ...`
: **When.** The adaptive quadrature of a density or selection function stops early.
  **Cause.** The function is sharp or discontinuous within a bin.
  **Fix.** Raise `max_depth` or `tol`, or put a bin edge at the jump; `from_density`
  also takes `n_sub=` for fixed quadrature points.

`Bin averages read zero in <n> bin(s) where the function is non-zero only in a feature too narrow to resolve at tol=<tol>; ...`
: **When.** None of the quadrature's main points reads the function anywhere, the
  checks near cell edges read something, and the cells there have been split to below
  `tol` of their bin.
  **Cause.** The function is non-zero only in a feature thinner than the quadrature
  resolves at `tol`, such as a thin box on a bin edge.
  **Fix.** Lower `tol`, or put bin edges at the feature.

`<n> injections fall outside the bins and are ignored.`
: **When.** `Selection.from_injections` gets injections outside the bins.
  **Fix.** Widen the bins, or drop those injections first.

`The selection changes by a factor of <x> between neighbouring bins; ...`
: **When.** A selection array changes by more than `STEEP_RATIO` between neighbouring
  bins.
  **Cause.** The selection varies within bins too, which biases their fractions.
  **Fix.** Pass the selection as a function of the true value, use narrower bins, or
  pass a `Selection` built from bin averages. See
  [selection within bins](../guides/selection.md#selection-that-varies-within-bins).

`With an explicit efficiency the matrix should be a migration matrix whose columns sum to one; ...`
: **When.** `Response(matrix, ..., efficiency=...)` gets columns that do not sum to one.
  **Cause.** The matrix may already include the efficiency, which is then counted twice.
  **Fix.** Pass a migration matrix with the efficiency, or the full matrix without it.

`<n> observed events have true values outside the truth bins and are left out; ...`
: **When.** `Response.from_simulation` gets observed events outside the truth bins.
  **Cause.** Events from outside the truth range are not modelled.
  **Fix.** Widen the truth bins to cover every event that can be observed.

`The prior density is unbounded at the boundary of the simplex ...`
: **When.** `space="simplex"` or an explicit `algorithm` asks `MAP` to solve on the
  simplex. The prior's density is unbounded at its edge, as with concentrations below
  one or a Pólya tree.
  **Cause.** EM can then set fractions to exactly zero.
  **Fix.** Pass `space="unconstrained"` for an interior mode.

`length_scale <x> on axis <d> is below the <r> that <m> basis functions resolve; ...`
: **When.** `LogisticGP` gets a fixed length scale shorter than its sine basis
  resolves.
  **Cause.** The field then loses variance.
  **Fix.** Raise `n_basis`, or use a longer length scale.

`sample_values draws values from the population alone within each bin: ...`
: **When.** `result.sample_values()` is called on likelihoods that keep no information
  within bins, such as tables or binned inputs.
  **Cause.** The values are then drawn flat within each bin.
  **Fix.** Pass `loglike=` with each object's own log-likelihood.

`Binning bias: <n> bins move by more than <threshold> posterior s.d. on a basis of order <order> ...`
: **When.** `result.binning_bias()` finds bins whose fraction shifts on a finer basis.
  **Cause.** The population is not flat within those bins.
  **Fix.** Fit with `binference.Basis(bins, order=...)` or narrower bins. See
  [check the binning](../guides/bins.md#check-the-binning)

`2-D plots show no truth, stack or label; they are ignored.`
: **When.** `plot_population` draws a 2-D grid and gets `truth`, `stack` or `label`.
  **Fix.** Leave those arguments out for 2-D grids.

`Diagnostics: <problems>. Hint: <hints>.`
: **When.** `result.check()` finds problems with the inputs. `infer` calls it after NUTS
  and Gibbs. The problems are:
  - `likelihoods from samples carry Monte Carlo variance sum_i V_i = <x> > 1 at the fitted population`:
    the sample likelihoods are noisy enough to bias the fit. Draw more samples, widen
    the bins, or pass `floor=`.
  - `<n> bins have data support below <support_min> objects ...: their fractions are set by the prior`:
    raised only with `support_min=`. Merge sparse bins, or read their fractions as the
    prior's.
  - `the measured counts disagree with the fitted population and the response (posterior predictive p = <p>)`:
    the response may not describe the data. For a classifier, label shift may be
    violated.

## `ConvergenceWarning`

A sampler or optimiser shows signs of non-convergence. The result may be wrong, not
just noisy.

`Diagnostics: <problems>. Hint: <hints>.`
: **When.** `result.check()` finds sampler problems. `infer` calls it after NUTS and
  Gibbs; `check(strict=True)` raises `DiagnosticsError` instead. The problems are:
  - `max R-hat = <x> > <rhat_max>`: the chains disagree. Run longer.
  - `min bulk ESS = <n> (<x> per chain < <ess_min>)`, and the same for the tail ESS: too
    few effective draws. Run longer.
  - the same two messages `of the selection` or `of the response's efficiencies`, for a
    selection or response inferred with the population.
  - `chains of <n> draws are too short to diagnose (R-hat and ESS need 8)`: no R-hat or
    ESS can be computed. Run longer.
  - `<n> divergent transitions (<x>% of draws)`: NUTS met a region it cannot explore.
    Try `target_accept=0.95`; see [diagnostics](../guides/checking.md#diagnostics).
  - `E-BFMI = <x> < 0.3 in some chain`: NUTS explores the energy distribution poorly.
    A longer warm-up or another prior may help.
  - `<x>% of NUTS iterations hit the maximum tree depth`: trajectories are cut short.
    With a concentration below one, in a Dirichlet or on a branch of a tree prior, or a
    shape below one in `GammaRates`, `method="gibbs"` is faster, and the hint says so
    when Gibbs can fit the model.
  - `the optimiser did not converge`.
  - `Pareto k-hat = <k> >= <threshold> (importance sampling unreliable)`.

  After Gibbs, the hint gives the {term}`missing information` and the autocorrelation
  time it implies. Run longer or use `method="nuts"`.

`Diagnostics: <problems>.` from a point estimate
: **When.** `PointEstimate.check()` finds problems with a MAP estimate. The problems
  are `the optimiser did not converge` and `KKT gap <g> > <kkt_max>: not at the optimum`.
  **Fix.** Raise `max_iter` of `MAP`.

`mix-SQP did not converge in <n> iterations; continuing with EM.`
: **When.** mix-SQP stops before its tolerance. EM then continues from its last point.
  **Fix.** Usually none: EM finishes the job. Raise `max_iter` if EM also warns.

`EM did not converge in <max_iter> iterations (missing information <f>); raise max_iter.`
: **When.** EM reaches `max_iter`. The missing information appears when it can be
  computed.
  **Cause.** EM is slow when the {term}`missing information` is near one.
  **Fix.** Raise `max_iter`, or use `algorithm="auto"` so mix-SQP runs where it applies.

`EM stopped improving the objective with a KKT gap of <g>.`
: **When.** EM can no longer improve in floating point, but its gap is above `tol`.
  **Fix.** Loosen `tol`, or check the {term}`KKT gap` of the result.

`EM reached a population under which an object has zero likelihood ...`
: **When.** EM sets a fraction to zero that some object needs.
  **Cause.** Usually concentrations below one.
  **Fix.** Pass `space="unconstrained"` to keep the mode interior.

`L-BFGS did not converge (<message>); max |gradient| = <g>.`
: **When.** The unconstrained optimiser stops before its tolerance.
  **Fix.** Raise `max_iter` of `MAP`.

`Mean-field VB did not converge.`
: **When.** Coordinate ascent reaches `max_iter`.
  **Fix.** Raise `max_iter`, or loosen `tol`.

`<n> of <N> bootstrap fits did not converge; raise max_iter.`
: **When.** Some bootstrap replicates stop before their tolerance.
  **Fix.** Raise `max_iter` of `Bootstrap`.

`The smooth selection's mode did not converge (gradient <g>).`
: **When.** The mode of a smooth injection selection, `from_injections(smooth=...)`, is
  not found.
  **Fix.** Check that the bins hold injections, or pass `smooth=None` for independent
  bins.

`<n> of <N> objects have Pareto k-hat >= <threshold>: their leave-one-out posteriors are unreliable ...`
: **When.** `result.assignment_probabilities_loo()` finds objects whose importance
  weights have heavy tails.
  **Cause.** Leaving these objects out changes the population a lot.
  **Fix.** Refit without them, or use more draws.

`In <n> of <N> fits the smallest bulk ESS is below n_draws=<n_draws>; ...`
: **When.** `validation.sbc` ranks more draws than a fit's effective sample size.
  **Cause.** Correlated draws make the ranks look over-dispersed.
  **Fix.** Run each fit longer, or rank fewer draws.

## `ApproximationWarning`

A result comes from an approximation with known failure modes. Its
`result.approximate` is `True`.

`Mean-field VB underestimates posterior uncertainty, ...`
: **When.** Every `method="vb"` run.
  **Fix.** Use NUTS or Gibbs for error bars.

`Bootstrap replicates describe the sampling distribution of the MAP estimator, not the posterior.`
: **When.** Every `method="bootstrap"` run.
  **Fix.** Use NUTS or Gibbs for a posterior.

`Uncorrected Laplace approximation: results are approximate.`
: **When.** `Laplace(importance=False)`.
  **Fix.** Keep `importance=True` to correct the draws.

`Pareto k-hat = <k> >= <threshold>: the Laplace approximation is not a good importance proposal; ...`
: **When.** The importance check of `Laplace` fails.
  **Cause.** The posterior is far from Gaussian, often through nearly empty bins.
  **Fix.** Use NUTS.

`The Hessian at the mode is not positive definite; clipping eigenvalues.`
: **When.** `Laplace` finds a mode without positive curvature in every direction.
  **Cause.** The log posterior is not curved downward in some direction at the mode.
  **Fix.** Use NUTS, or a more informative prior.

`auto kept Laplace draws although its importance ESS is below target_ess=<t>; ...`
: **When.** `method="auto"` keeps Laplace for a very large input, or because no other
  engine takes the options given.
  **Fix.** Pass `method="nuts"`, or drop the options that rule NUTS out.

`The within-bin data cannot be pickled (e.g. a lambda density), so the copy's value draws are flat within bins.`
: **When.** Likelihoods from a function that cannot be pickled are copied, saved, or
  sent to NUTS worker processes.
  **Fix.** Define the density as a module-level function instead of a lambda.

## `RuntimeWarning`

`Could not run chains in parallel (<error>); running them serially.`
: **When.** `NUTS(n_jobs=...)` cannot start its worker processes. The chains then run in
  this process, with the same draws.
  **Fix.** None needed. On some platforms, run the script under
  `if __name__ == "__main__":`.

## `DiagnosticsError`

`Diagnostics failed: <problems>`
: **When.** `result.check(strict=True)` or `PointEstimate.check(strict=True)` finds any
  of the problems above. The message joins them all.
  **Fix.** As for the matching warning.

## Other errors

binference validates its inputs early. A `ValueError` or `TypeError` names the argument
and says what was expected. These are the errors met most often:

`A selection function needs an explicit selection_mode: ...`
: A selection array or function was passed without `selection_mode`. Say whether the
  selection acts on the truth or on the measurement; see
  [say how the selection acts](../guides/selection.md#say-how-the-selection-acts).

`Say what the input is: kind='likelihood' for likelihoods, or interim=<prior> for posteriors ...`
: `from_table` or `from_density` needs exactly one of `kind` and `interim`.

`weights are integer multiplicities; pass fractional_weights=True for a weighted pseudo-likelihood.`
: Non-integer weights were passed without `fractional_weights=True`.

`<what> have mass in bins where the interim prior is zero; ...`
: The interim prior must be positive wherever a posterior has mass.

`Gibbs sampling needs a conjugate prior; <prior> is not. Use NUTS instead.`
: Gibbs also refuses non-integer weights and a smooth injection selection. Mean-field VB
  also needs a conjugate prior.

`<Engine> cannot infer a selection from injections or a tallied response; ...`
: That engine cannot infer a selection or response jointly. Use one of the engines the
  message names, or pass `selection.fixed()` or `response.fixed()` to fix it.

`Unknown method <name>; choose from [...]`, `n_samples does not apply to method=<name>.`
: `infer` got an unknown engine name, or a draw count for a point estimate.

`None of the engines method='auto' can use for this model ... takes all of [...]; pass method= explicitly.`
: The engine options given rule out every engine auto can choose.

`This result is not attached to its likelihoods; call result.attach(likelihoods).` (`RuntimeError`)
: A loaded result needs its likelihoods for per-object outputs.

`No draws yet; call run().` and `This PopulationEstimator is not fitted yet; call fit().` (`RuntimeError`)
: The engine or estimator has not run yet.

`Plotting needs matplotlib: ...` and `to_datatree needs xarray: ...` (`ImportError`)
: Install the extra: `pip install 'binference[plot]'` or `'binference[xarray]'`.

`Legacy numpy.random.RandomState objects are not supported; ...` (`TypeError`)
: Pass an integer seed or a `numpy.random.Generator`.
