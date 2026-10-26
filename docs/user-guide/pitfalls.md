# Pitfalls

Each mistake below has been made in published work. Most of them bias the
population silently: the fit converges, the error bars look reasonable, and
the answer is wrong.

## Feeding posteriors in as likelihoods

**Symptom.** The population looks suspiciously like the prior your
photometric-redshift code, classifier or per-object fitter used.

Posteriors already contain an interim prior $\pi$. Treating them as
likelihoods replaces $L_i\cdot\rho$ by $L_i\cdot(\pi\odot\rho)$. The data
then constrain $\pi\odot\rho$ rather than $\rho$, and an extra factor
$(\pi\cdot\rho)^W$ drags the population toward where $\pi$ is large
([theory](../theory.md#likelihoods-versus-posteriors)). binference's slow tests
show biases of more than eight standard deviations.

**Fix.** Use `Likelihoods.from_table(P, bins, interim=interim)` or
`from_samples(..., interim)`. binference deliberately has no default
interim prior for these.

## Masses versus densities with unequal bins

**Symptom.** Wide bins get too much mass, or too little.

A likelihood integrated over a bin (a CDF difference, a histogram count) is
a *mass*. binference needs the bin *average*, the mass divided by the bin
volume. With equal bins the difference is a constant and harmless. With
unequal bins, logarithmic bins or N-D grids with varying cell sizes it biases
$\rho$, by up to 28 standard deviations in the frankenz audit.

**Fix.** Declare `representation="mass"` or `"density"` explicitly when bins
are unequal. binference warns when posterior rows look inconsistent with the
declared representation.

## Stacking

**Symptom.** A smooth, confident $n(z)$ with no error bars, broader than the
truth.

The average of per-object posteriors (`lk.stack()`) is one EM step from the
interim prior. It convolves the population with the typical measurement
error, and resampling it (bootstrap, Poisson, "PDF draws") understates the
uncertainty, often several-fold.

**Fix.** Use any engine; `method="map"` already runs the iteration to
convergence.

## Ignoring selection, or modelling it twice

**Symptom.** Well-detected bins are over-represented, or the correction
overshoots.

**Fix.** State how selection acts: `selection_mode="on_measurement"` when it acts
on the measured data, `"on_truth"` when it depends on the true value. Never
renormalise each object's likelihood over the detectable region *and* pass a
selection function. See {doc}`selection`.

## Bins that are too coarse

**Symptom.** With many objects, the posterior excludes the truth even though
everything else is right.

The histogram model assumes the population is flat within each bin. With
bins comparable to the measurement errors and a population that varies
within bins, the bin-averaged likelihood weights the wrong parts of each bin.
This **binning bias** does not shrink with $N$, while the error bars do. On
a six-bin mock it exceeds four standard deviations at $N=20\,000$.

**Fix.** Use bins narrower than the features you care about, add a smoothness
prior to keep fine bins well-behaved, and report coarse summaries as
functionals of the fine posterior (`result.rebin`, `result.functional`).

## Trusting approximate error bars

**Symptom.** Error bars that look too good to be true.

- **Mean-field VB** ignores the coupling between objects' bins and the
  population, which is strongest exactly when likelihoods are broad. Its
  intervals can be several times too narrow.
- **The Laplace approximation** is unreliable when the Pareto $\hat k$ reaches its
  threshold ($0.7$ with the default number of proposals),
  which is typical with ten or more bins.
- **Bootstrap and "PDF-draw" intervals** describe an estimator's sampling
  variability, not the posterior.

binference flags approximate results (`result.approximate`), and the summary
says so.

**Fix.** Use an exact engine (NUTS, or Gibbs where it mixes) for error bars,
or check an approximation against it on a subset. `method="auto"`, the
default, keeps the Laplace approximation only when its importance check
passes.

## Too many bins for a flat Dirichlet

**Symptom.** With few objects per bin, every bin seems to have some
population, and the estimate is pulled toward uniform.

`Dirichlet(1.0)` adds one pseudo-count per bin, which is $K$ in total. With
$K$ comparable to $N$, that is a strong prior toward uniform.

**Fix.** Use `Dirichlet.reference()` ($\alpha=1/K$, one pseudo-count in
total), a smoothness prior, or fewer bins.

## Reading per-bin error bars as independent

**Symptom.** Error bars on sums, means or ratios that are too large or too
small.

Neighbouring bins are strongly anti-correlated whenever likelihoods are
broader than the bins. Adding per-bin variances overestimates the error on a
sum; ignoring correlations misstates any ratio.

**Fix.** Compute derived quantities per draw: `result.functional(...)` and
`result.rebin(...)`.

## Confusing the population with the sample

**Symptom.** Error bars on "the redshift distribution of my lens sample" that
include shot noise those particular objects do not have.

$\rho$ is the population the objects were drawn from.
`result.realized_counts()` gives the histogram of *these* objects, which is
what lensing kernels and similar sample-specific quantities need.

## Smoothness priors that the data want to switch off

**Symptom.** Divergent transitions with a random-walk prior, and a smoothness
scale whose posterior piles up near zero.

An order-$r$ random walk does not penalise polynomials of degree below $r$;
order 2 ignores straight lines in $\log\rho$. When the true population is
close to such a trend, for example an exponential spectrum, the data push the
scale toward zero. The posterior then has a funnel that is hard for any
sampler.

**Fix.** Use `LogisticGP()`, raise `target_accept` to 0.95, or give the scale
a hyperprior bounded away from zero, such as `priors.LogNormal(np.log(0.1), 1)`.

## Not looking at the diagnostics

**Symptom.** Results change between runs.

$\hat R>1.01$, low effective sample sizes or NUTS divergences mean the draws
do not yet represent the posterior. `binference.infer` warns by default, so do not
silence `ConvergenceWarning` without reading it. Run longer, or switch to NUTS
if Gibbs mixes slowly.

A few isolated divergences among thousands of draws are usually harmless if
$\hat R$ and the effective sample sizes are fine. Divergences that persist
with a higher `target_accept` mark a region the sampler cannot explore, such
as the funnel described above: change the prior or its hyperprior rather
than ignore them.
