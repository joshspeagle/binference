# Validation

How do we know a result is right? Two kinds of check answer two questions. Diagnostics
ask whether one run computed its posterior well. Calibration asks whether the whole
pipeline, from likelihoods and prior to engine, gives posteriors that are right on
average. This page defines both, and lists the statistical checks binference's own test
suite runs. The {doc}`checking guide <../guides/checking>` shows how to run them on
your own analysis.

## Diagnostics of one run

- **Convergence.** {term}`R-hat` compares the spread within chains with the spread
  between them. binference uses the rank-normalised split $\hat R$ {cite:p}`vehtari2021`,
  the larger of its bulk and folded versions.
- **Effective sample size.** The bulk {term}`ESS` measures how well the draws estimate
  the centre of a distribution. The tail ESS uses the indicators of two quantiles, which
  measures how well they estimate the tails ([defaults](../reference/defaults.md#results-and-diagnostics)).
- **Monte Carlo error.** The {term}`Monte Carlo standard error` of a posterior mean is
  its standard deviation divided by the square root of its ESS (`diagnostics.mcse`).
- **NUTS health.** A {term}`divergence` marks a region the sampler cannot explore.
  Iterations that hit the maximum {term}`tree depth` explore slowly. The {term}`E-BFMI`
  of a chain with energies $\mathcal E_s$ is
  $\sum_s(\mathcal E_s-\mathcal E_{s-1})^2/\sum_s(\mathcal E_s-\bar{\mathcal E})^2$; low
  values mean heavy tails or a funnel.
- **Importance sampling.** The Pareto shape $\hat k$ of the importance weights says
  whether the Laplace correction is reliable ([Laplace](engines.md#laplace-with-importance-sampling)).
- **Optimisation.** The [KKT gap](point-estimates.md#the-certificate) says how far a
  point estimate is from the optimum.

`result.check()` turns these into warnings; the
[checking guide](../guides/checking.md#diagnostics) lists what it flags, and the
[defaults](../reference/defaults.md#results-and-diagnostics) hold its thresholds.

These diagnostics cannot tell whether the posterior itself is right: whether the
likelihoods, the prior and the engine together give calibrated answers. Calibration
checks that.

## Calibration

### Exact posteriors

For two or three bins, {func}`~binference.validation.exact_posterior` integrates
{eq}`eq-P` by quadrature over the simplex. It works for priors with a closed-form
density, with or without a selection. Any engine can then be compared with it.

### Simulation-based calibration

{term}`Simulation-based calibration <SBC>` {cite:p}`talts2018,modrak2025` repeats three
steps $n_{\rm sim}$ times:

1. draw a population from the prior;
2. simulate a catalogue from that population;
3. fit it, and record the rank of each true value among $n_{\rm draw}$ posterior draws.

If the pipeline is calibrated, every rank is uniform on $0,\dots,n_{\rm draw}$.
{func}`~binference.validation.sbc` ranks each fraction (or rate), and also the joint
log-likelihood of the true population. That last quantity is sensitive to errors in the
correlations between bins. The draws should be nearly independent: `sbc` warns when a
fit's smallest bulk ESS is below $n_{\rm draw}$, since correlated draws make the ranks
look over-dispersed.

Two tests check the ranks of each quantity for uniformity:

- **chi-square:** Pearson's test of the rank histogram, the more sensitive to errors of
  width;
- **ECDF** {cite:p}`sailynoja2022`: let $C_m$ be the number of ranks at most rank
  $m$. Each $C_m$ is binomial with $n_{\rm sim}$ trials and probability
  $(m+1)/(n_{\rm draw}+1)$. The statistic is the smallest two-sided binomial tail
  probability over $m$. Its p-value is computed exactly from the joint law of the
  counts; when its Bonferroni bound, $n_{\rm draw}$ times the statistic, is below
  $10^{-12}$, that bound is returned instead. This test is the more sensitive to bias.

`SBCResult.passed` applies a Bonferroni correction over the quantities.

### Coverage

{func}`~binference.validation.coverage` measures how often central
{term}`credible intervals <credible interval>` contain the truth, for each bin and for
any functional. With data simulated from the prior, Bayesian intervals have exactly their
nominal {term}`coverage` on average over the prior. `CoverageResult.stderr` gives the
binomial standard error of the nominal rate.

## The statistical tests

binference's slow tests (`pytest -m slow`, mostly in
[tests/test_slow.py](https://github.com/joshspeagle/binference/blob/main/tests/test_slow.py))
run these checks on the engines themselves. The {doc}`contributing guide
<../contributing>` says when CI runs them.

- **Simulation-based calibration.** SBC runs for NUTS and Gibbs with Dirichlet-family,
  random-walk and rate priors. It covers known and inferred selections, an inferred
  response, and the importance-corrected Laplace approximation. Most cases use four
  bins on $[0,1]$ and 40 objects with Gaussian errors of 0.2. Both rank tests must pass
  at a Bonferroni-corrected level of $10^{-3}$.
- **Negative controls.** These show that the checks can fail. SBC rejects a Gibbs
  pipeline whose posteriors are about 18% too narrow, and mean-field VB fails SBC.
- **Coverage.** The 68% and 95% Gibbs intervals of every bin and of the mean value
  cover the truth at their nominal rates. The tolerance is four binomial standard
  errors.
- **Geweke tests.** "Getting it right" tests {cite:p}`geweke2004` check that the Gibbs
  conditionals are exact, for the Dirichlet family and Gamma rates, with and without a
  selection. A Gibbs kernel that forgets the missed objects fails them.
- **Agreement between engines.** NUTS and Gibbs agree within a few Monte Carlo errors,
  also with an inferred selection or response. Four
  [benchmark](auto.md#the-benchmark-behind-the-rules) combinations are refitted against
  stored reference posteriors, and auto must pick the benchmark's engine.
- **Recovery and known biases.** Fits with an interim prior divided out, and with a
  selection on the truth, recover the true population with calibrated errors. Using
  posteriors as likelihoods is biased by more than eight posterior standard deviations
  ([likelihoods versus posteriors](model.md#likelihoods-versus-posteriors)). Mean-field
  VB's standard deviations are less than half those of NUTS when $f$ exceeds 0.9.

### Binning bias

One recovery test shows the [binning bias](model.md#binning-bias). It fits a continuous
bimodal population on six bins, with errors of 0.6 to 1.2 bin widths. At
$N=20\,000$ the error of the worst fraction exceeds four posterior standard deviations.
