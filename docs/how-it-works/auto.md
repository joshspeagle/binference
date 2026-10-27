# How auto chooses

How does `method="auto"` pick an engine, and what evidence sets its rules? This page
states the rules and describes the benchmark behind them. {class}`~binference.inference.Auto`
is the reference for the rules and the defaults of their options. The
{doc}`engines guide <../guides/engines>` covers when to override the choice.

## The rules

{term}`Auto` tries three engines in order and keeps the first one that passes its
check. Engine options passed to `infer` rule out the engines that do not take them.

1. **Gibbs, where it mixes well.** Gibbs needs a {term}`conjugate prior` and integer
   weights. It is chosen in two cases.
   - A selection or response is inferred with the population. Gibbs is exact and fast
     for these joint models. A selection smoothed across bins is not conjugate, so it
     goes to NUTS.
   - The [missing information](point-estimates.md#missing-information) $f$ at the most
     probable population is at most `gibbs_max_missing`. Auto computes that population
     first, with {class}`~binference.samplers.MAP`.

   In the second case Gibbs runs
   $\lceil n_{\rm ESS}\,(1+f)/(1-f)/n_{\rm chains}\rceil$ draws per chain, and at least
   its usual length. Here $n_{\rm ESS}$ is `target_ess`, the smallest bulk {term}`ESS` the
   run aims for. If
   that would exceed `max_gibbs_draws` per chain, Gibbs is skipped.
2. **Laplace, if its importance check passes.** The importance-corrected Laplace result
   is kept when its {term}`Pareto k-hat` passes and its importance ESS reaches `target_ess`. A
   first pass with fewer proposals stops the trial early when it projects too small an
   ESS ([defaults](../reference/defaults.md#automatic-engine-choice)).
3. **NUTS** otherwise.

Without `n_samples`, a Gibbs run is extended until every bulk ESS reaches `target_ess` and the run
passes the $\hat R$ and ESS limits of {meth}`~binference.PopulationResult.check`. Each
extension aims a little above `target_ess` and is bounded in size
([defaults](../reference/defaults.md#automatic-engine-choice)). If the extensions are
projected to need more than `max_gibbs_draws` draws per chain, auto discards the run and
tries the next engine. A jointly inferred selection or response can mix much more slowly
than $f$ predicts, which this catches. An explicit `n_samples` fixes the length instead:
Gibbs is then kept whatever length it would need, and is not extended.

Laplace is kept even when its check fails in two cases. One is likelihoods with more
than `nnz_limit` stored entries. The other is options that no later engine takes. Such a
result is flagged {term}`approximate`. The result records the choice and its reason in
`result.info["auto_method"]` and `result.info["auto_reason"]`. Where auto computed the
most probable population without a warning, the result keeps it as `result.best`.

## The benchmark behind the rules

The rules come from `benchmarks/matrix.py`, which compares every engine with a long
reference run on a matrix of mock catalogues.

**Catalogues.** There are 18 catalogues, each varying one factor around a baseline. The
baseline has one dimension, 20 bins and 1,000 objects with Gaussian errors one bin wide.
Its bimodal population is drawn uniformly within bins. The variations are:

- error widths of 0.3 and 3 bins;
- 100 and 10,000 objects;
- 10 and 50 bins;
- a sparse population (half the bins empty) and a spiky one (half the population in one
  bin);
- selection on the truth and on the measurement;
- posteriors with the correct and with a wrong interim prior divided out;
- 20 and 200 posterior samples per object;
- an 8×8 grid, and classifier outputs for five classes;
- a photometric-redshift-like set: 30 bins on $[0,3]$, 5,000 objects, errors of
  $0.05(1+z)$, a second peak for 20% of objects and a uniform floor for 3%.

**Fits.** Each catalogue was fitted with the priors that apply to it, 43 combinations in
all. The priors are the flat and the $\alpha=0.1$ Dirichlet, the random walk, the
Gaussian process and the Pólya tree. Each engine ran at its defaults. The reference was NUTS with four
chains of 5,000 draws after 2,000 warm-up iterations. Errors are measured in reference
standard deviations: the largest error of a bin's mean, and the ratio of each bin's
standard deviation to the reference's.

**Setup.** binference 0.1.0.dev0, NumPy 2.5.4 and Python 3.13.16 on x86-64, run on
26 October 2026. Each fit used one thread, with four fits at a time. The results are in
`benchmarks/results/`.

| engine | largest mean error (reference s.d.) | s.d. ratio | smallest bulk ESS per second |
|---|---|---|---|
| NUTS (43 combinations) | 0.08 | 0.69–1.19 | 0.1–339, median 80 |
| Gibbs (26) | 0.94 | 0.24–1.75 | 0.6–791, median 14 |
| Laplace, check passed (4) | 0.05 | 0.96–1.04 | 1751–6607 |
| Laplace, check failed (39) | 8.1 | 0.05–7.3 | |
| mean-field VB (26) | 2.6 | 0.00–1.05 | |

What the table shows:

- **NUTS** was within 0.08 reference standard deviations in every mean. Its ratios
  outside 0.9–1.1 belong to nearly empty bins under the random-walk prior, whose
  posteriors are heavy-tailed. It is slow with Dirichlet concentrations below one: at
  0.1 it managed 2 to 5 ESS per second, and took 859 s on the photometric-redshift set.
- **Gibbs** beat NUTS in all 5 combinations where $f$ was at most 0.976, at 103–791 ESS
  per second against 2–336. It beat NUTS in only 3 of the 21 where $f$ was 0.994 or
  more, at 0.6–67 against 0.1–339. The default of `gibbs_max_missing` lies in that gap.
  Above it, the run length computed from $f$ makes up for the autocorrelation.
- **Laplace** passed its check only with the Pólya tree. There it agreed with the
  reference to 0.05 standard deviations, in 0.6 to 1.0 s. Where the check failed, its
  errors reached 8 standard deviations. The ESS requirement also rejected the two fits
  whose $\hat k$ passed with fewer effective draws than `target_ess`; one of them had quantiles
  0.9 standard deviations off.
- **Mean-field VB** was fast but its standard deviations were too small, down to almost
  zero, so auto never chooses it.

The slow test suite re-checks four of these combinations against stored reference
posteriors, including the engine auto picks in each
([the test suite](validation.md#the-statistical-tests)).
