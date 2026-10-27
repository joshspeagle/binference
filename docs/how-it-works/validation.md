# Validation

How do we know a result is right? Two kinds of check answer two questions. Diagnostics
ask whether one run computed its posterior well. Calibration asks whether the whole
pipeline, from likelihoods and prior to engine, gives posteriors that are right on
average. This page defines both, and lists the statistical checks binference's own test
suite runs. The {doc}`validation guide <../guides/validation>` shows how to run them on
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

`result.check()` turns these into warnings, each against a threshold in the
[defaults](../reference/defaults.md#results-and-diagnostics). It flags:

- $\hat R$ or ESS per chain out of range;
- divergences, a low E-BFMI, and too many iterations at the maximum tree depth;
- a large $\hat k$, and an optimiser that did not converge.

It also checks the inputs. Likelihoods from samples whose summed Monte Carlo variance is
large are flagged.
For binned counts with more measured than true bins, a {term}`posterior predictive
p-value` tests the counts against the response. `infer` runs the check after NUTS and
Gibbs by default. `PointEstimate.check()` checks the KKT gap and convergence of a point
estimate.

These diagnostics cannot tell whether the posterior itself is right: whether the
likelihoods, the prior and the engine together give calibrated answers. That is the job
of calibration.

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
- **ECDF** {cite:p}`sailynoja2022`: with $C_j$ the number of ranks at most $j$, each
  $C_j$ is binomial with $n_{\rm sim}$ trials and probability $(j+1)/(n_{\rm draw}+1)$.
  The statistic is the smallest two-sided binomial tail probability over $j$. Its
  p-value is computed exactly from the joint law of the counts. This test is the more
  sensitive to bias.

`SBCResult.passed` applies a Bonferroni correction over the quantities.

### Coverage

{func}`~binference.validation.coverage` measures how often central
{term}`credible intervals <credible interval>` contain the truth, for each bin and for
any functional. With data simulated from the prior, Bayesian intervals have exactly their
nominal {term}`coverage` on average over the prior. `CoverageResult.stderr` gives the
binomial standard error of the nominal rate.

## The statistical tests

binference's slow tests (`pytest -m slow`) run these checks on the engines themselves.
CI runs them every week, by hand, and on pull requests labelled `slow`. Unless stated,
the SBC tests use four bins on $[0,1]$ and 40 objects per catalogue, with Gaussian errors
of 0.2. Both rank tests must pass at a Bonferroni-corrected level of $10^{-3}$.

**Simulation-based calibration.**

- NUTS with the flat Dirichlet, and with a first-order random walk of fixed scale (200
  simulations).
- NUTS with a second-order random walk whose scale is inferred (50 simulations).
- Gibbs with the flat Dirichlet (1,000 simulations) and a generalised Dirichlet (500).
- Gibbs with a known selection on the truth and on the measurement, with Gamma rates and
  an exposure, and with the Pólya tree (200 each).
- The importance-corrected Laplace approximation with the flat Dirichlet (200).
- A selection inferred from injections, with the population: Gibbs with the flat
  Dirichlet (200 simulations) and NUTS with a random walk (60). These rank the
  fractions and the detection probabilities.
- A response inferred from tallied events, with the population, by Gibbs (150
  simulations). This ranks the fractions and the efficiencies.

**Negative controls.** Each shows that a check can fail.

- SBC rejects a Gibbs pipeline whose posteriors are about 18% too narrow, made by
  counting half the objects twice.
- Mean-field VB fails SBC, as expected; the test is marked to fail.
- The Geweke test below detects a Gibbs kernel that forgets the missed objects under a
  selection.

**Coverage.** Gibbs with the flat Dirichlet is run on 1,000 simulations. The 68% and 95%
intervals of every bin and of the mean value cover the truth at their nominal rates,
within four binomial standard errors.

**Exactness of the Gibbs conditionals.** "Getting it right" tests {cite:p}`geweke2004`
compare two ways of simulating the population and the data jointly. One draws from the
prior and the data model; the other alternates the Gibbs conditionals with draws of the
data. Their means must agree within 4.5 standard errors. They run for the Dirichlet with
and without a selection, a generalised Dirichlet, a Dirichlet tree, and Gamma rates with
an exposure.

**Agreement between engines.**

- Gibbs and NUTS agree in every bin's mean and standard deviation, within five combined
  Monte Carlo errors, on 5,000 objects in 15 bins.
- NUTS and Gibbs agree within four combined errors on a selection inferred from
  injections, and on a response inferred from tallies.
- Four combinations of the [benchmark](auto.md#the-benchmark-behind-the-rules) are
  refitted and compared with stored reference posteriors of 20,000 NUTS draws. Every
  mean and standard deviation must agree within four combined Monte Carlo errors.
  `method="auto"` must pick the engine the benchmark chose.

**Recovery and known biases.**

- Posteriors with an interim prior divided out (80 catalogues of 800 objects), and
  catalogues with a selection on the truth (100 of 600), recover the true population:
  the errors of the posterior means, whitened by the posterior covariance, have unit
  variance within four standard errors.
- Posteriors used as likelihoods, without dividing out the interim prior, are biased
  by more than eight posterior standard deviations on 20,000 objects; the correct fit
  stays within four ([likelihoods versus posteriors](model.md#likelihoods-versus-posteriors)).
- A continuous population on coarse bins shows a [binning bias](model.md#binning-bias)
  of more than four posterior standard deviations at 20,000 objects.
- Mean-field VB on 3,000 objects in 15 bins, where $f$ exceeds 0.9, gives standard
  deviations less than half those of NUTS, while its means stay within one NUTS
  standard deviation.
