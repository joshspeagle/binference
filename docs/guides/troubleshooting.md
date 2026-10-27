# Troubleshooting

Find your symptom, then follow its link to the message on the
{doc}`warnings page <../reference/warnings>` and the fix. The second half lists the
mistakes that bias a result *without* a warning.

## Symptoms

| Symptom | Likely cause | Where to look |
|---|---|---|
| `ConvergenceWarning` with R-hat or ESS | the chains have not mixed; with Gibbs, high {term}`missing information` | [`ConvergenceWarning`](../reference/warnings.md#convergencewarning); run longer or pass `method="nuts"` |
| divergent transitions | a funnel in a smoothness prior | [Smoothness priors that the data want to switch off](#smoothness-priors-that-the-data-want-to-switch-off) |
| `ApproximationWarning` with Pareto k-hat | the Laplace approximation does not fit the posterior | [`ApproximationWarning`](../reference/warnings.md#approximationwarning); pass `method="nuts"` |
| `InputWarning`: likelihood mass outside the bins | the bins do not cover the data | [`InputWarning`](../reference/warnings.md#inputwarning); {doc}`bins` |
| `InputWarning`: posterior rows sum to one only as the other representation | masses read as densities, or the reverse | [Masses versus densities](#masses-versus-densities-with-unequal-bins) |
| `InputWarning`: few effective samples, or Monte Carlo variance above one | too few samples per object | {doc}`inputs/samples` |
| `InputWarning`: binning bias | the population is not flat within bins | {doc}`bins` |
| `InputWarning`: the selection changes steeply between bins | a selection array evaluated at bin centres | [Selection within bins](selection.md#selection-that-varies-within-bins) |
| `InputWarning`: posterior predictive p | the response does not describe the counts; for a classifier, no label shift | {doc}`inputs/binned` |
| the population looks like your code's prior | posteriors passed as likelihoods | [Feeding posteriors in as likelihoods](#feeding-posteriors-in-as-likelihoods) |
| a jagged population | errors as wide as the bins | {doc}`priors`: a smoothness prior |
| a population pulled toward uniform | too many bins for a flat Dirichlet | [Too many bins](#too-many-bins-for-a-flat-dirichlet) |
| a slow fit | many objects or bins, or a slow engine | {doc}`performance` |
| an error naming an argument | an input of the wrong kind | [Other errors](../reference/warnings.md#other-errors) |

Do not silence `ConvergenceWarning` without reading it. To silence the others after
reading them, filter `binference.BinferenceWarning`.

## Pitfalls

Each mistake below has been made in published work. Most of them bias the population
silently: the fit converges, the error bars look reasonable, and the answer is wrong.

### Feeding posteriors in as likelihoods

**Symptom.** The population looks suspiciously like the prior your
photometric-redshift code, classifier or per-object fitter used.

Posteriors already contain an {term}`interim prior`, which then pulls the population
toward itself ([why](../how-it-works/theory.md#likelihoods-versus-posteriors)).

**Fix.** Use `Likelihoods.from_table(P, bins, interim=interim)` or
`from_samples(..., interim)` ({doc}`inputs/posteriors`). binference deliberately has
no default interim prior for these.

### Masses versus densities with unequal bins

**Symptom.** Wide bins get too much mass, or too little.

A likelihood integrated over a bin (a CDF difference, a histogram count) is
a *mass*. binference needs the bin *average*, the mass divided by the bin
volume. With equal bins the difference is a constant and harmless. With
unequal bins, logarithmic bins or grids with cells of different sizes, it biases
$\rho$. See {doc}`inputs/tables`.

**Fix.** Declare `representation="mass"` or `"density"` explicitly when bins
are unequal. binference warns when posterior rows look inconsistent with the
declared representation.

### Stacking

**Symptom.** A smooth, confident population with no error bars, broader than the
truth.

The average of per-object posteriors, `lk.stack()`, is the population blurred by the
measurement errors. Resampling it understates the uncertainty
([why](../how-it-works/theory.md#point-estimates-em-is-iterated-stacking)).

**Fix.** Use any engine; `method="map"` already runs the iteration to
convergence.

### Ignoring selection, or modelling it twice

**Symptom.** Well-detected bins are over-represented, or the correction overshoots.

{cite:t}`mandel2019` warn about two mistakes:

- **Correcting twice.** Do not renormalise each object's likelihood over the detectable
  region *and* pass a selection. Pass the ordinary likelihood $p(d_i\mid z)$ at the
  observed data.
- **Correcting nothing.** Multiplying likelihoods by $S$ without telling binference
  about the selection omits the normalisation.

**Fix.** Pass the ordinary likelihoods and a selection, and state how the selection
acts: `selection_mode="on_measurement"` when it acts on the measured data,
`"on_truth"` when it depends on the true value ({doc}`selection`).

### Bins that are too coarse

**Symptom.** With many objects, the posterior excludes the truth even though
everything else is right.

The model takes the population as flat within each bin. If it is not, the
{term}`binning bias` does not shrink with $N$, while the error bars do
([theory](../how-it-works/theory.md#binning-is-a-modelling-choice)).

**Fix.** Run `result.binning_bias()`, and follow {doc}`bins`.

### Trusting approximate error bars

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

### Too many bins for a flat Dirichlet

**Symptom.** With few objects per bin, every bin seems to have some
population, and the estimate is pulled toward uniform.

`Dirichlet(1.0)` adds one pseudo-count per bin, which is $K$ in total. With
$K$ comparable to $N$, that is a strong prior toward uniform.

**Fix.** Use `Dirichlet.reference()` ($\alpha=1/K$, one pseudo-count in
total), a smoothness prior, or fewer bins.

### Reading per-bin error bars as independent

**Symptom.** Error bars on sums, means or ratios that are too large or too
small.

Neighbouring bins are strongly anti-correlated whenever likelihoods are
broader than the bins. Adding per-bin variances overestimates the error on a
sum; ignoring correlations misstates any ratio.

**Fix.** Compute derived quantities per draw: `result.functional(...)` and
`result.rebin(...)`.

### Confusing the population with the sample

**Symptom.** Error bars on "the redshift distribution of my lens sample" that
include shot noise those particular objects do not have.

$\rho$ is the population the objects were drawn from.
`result.realized_counts()` gives the histogram of *these* objects, which is
what lensing kernels and similar sample-specific quantities need.

### Smoothness priors that the data want to switch off

**Symptom.** Divergent transitions with a random-walk prior, and a smoothness
scale whose posterior piles up near zero.

An order-$r$ random walk does not penalise polynomials of degree below $r$;
order 2 ignores straight lines in $\log\rho$. When the true population is
close to such a trend, for example an exponential spectrum, the data push the
scale toward zero. The posterior then has a funnel that is hard for any
sampler.

**Fix.** Use `LogisticGP()`, raise `target_accept` to 0.95, or give the scale
a {term}`hyperprior` bounded away from zero, such as `priors.LogNormal(np.log(0.1), 1)`.

### Not looking at the diagnostics

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

A low E-BFMI (below 0.3) means that NUTS moves slowly through the energy
distribution, so it may under-explore the tails of the posterior. A longer
warm-up or another prior may help: a raised `target_accept` does not change
the energy geometry that causes it. Otherwise compare the fit with a longer
run before you rely on its tails.
