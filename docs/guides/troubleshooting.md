# Troubleshooting

Find your symptom, then follow its link to the message on the
{doc}`warnings page <../reference/warnings>` and the fix. The second half lists the
mistakes that bias a result *without* a warning.

## Symptoms

| Symptom | Likely cause | Where to look |
|---|---|---|
| `ConvergenceWarning` with R-hat or ESS | the chains have not mixed; with Gibbs, high {term}`missing information` | [`ConvergenceWarning`](../reference/warnings.md#convergencewarning); run longer or pass `method="nuts"` |
| divergent transitions | a funnel in a smoothness prior | [A smoothness scale near zero](#a-smoothness-scale-near-zero) |
| `ApproximationWarning` with Pareto k-hat | the Laplace approximation does not fit the posterior | [`ApproximationWarning`](../reference/warnings.md#approximationwarning); pass `method="nuts"` |
| `InputWarning`: likelihood mass outside the bins | the bins do not cover the data | [`InputWarning`](../reference/warnings.md#inputwarning); {doc}`bins` |
| `InputWarning`: posterior rows sum to one only as the other representation | masses read as densities, or the reverse | [Masses versus densities](#masses-versus-densities-with-unequal-bins) |
| `InputWarning`: few effective samples, or too much Monte Carlo variance | too few samples per object | {doc}`inputs/samples` |
| `InputWarning`: binning bias | the population is not flat within bins | {doc}`bins` |
| `InputWarning`: the selection changes steeply between bins | a selection array evaluated at bin centres | [Selection within bins](selection.md#selection-that-varies-within-bins) |
| `InputWarning`: posterior predictive p | the response does not describe the counts; for a classifier, no label shift | {doc}`inputs/binned` |
| the population looks like your code's prior | posteriors passed as likelihoods | [Passing posteriors as likelihoods](#passing-posteriors-as-likelihoods) |
| a jagged population | errors as wide as the bins | {doc}`priors`: a smoothness prior |
| fractions biased toward equal values | too many bins for a flat Dirichlet | [Too many bins](#too-many-bins-for-a-flat-dirichlet) |
| a slow fit | many objects or bins, or a slow engine | {doc}`performance` |
| an error naming an argument | an input of the wrong kind | [Other errors](../reference/warnings.md#other-errors) |

Do not silence `ConvergenceWarning` without reading it. To silence the others after
reading them, filter `binference.BinferenceWarning`.

## Pitfalls

Most of these mistakes bias the population silently: the fit converges, the error bars
look reasonable, and the answer is wrong.

### Passing posteriors as likelihoods

**Symptom.** The population looks like the prior your photometric-redshift code,
classifier or per-object fitter used.

**Cause.** Posteriors contain an {term}`interim prior`, and the fit is biased toward it.
**Fix:** {doc}`inputs/posteriors`.

### Masses versus densities with unequal bins

**Symptom.** Wide bins get too much mass, or too little.

**Cause.** A likelihood integrated over each bin was read as a bin average, or the
reverse. **Fix:** {doc}`inputs/tables`.

### Stacking

**Symptom.** A smooth, confident population with no error bars, broader than the
population, because each object adds its measurement error.

**Cause.** `lk.stack()` averages the per-object posteriors instead of fitting the
population. **Fix:** [stacking is not the answer](inputs/posteriors.md#stacking-is-not-the-answer).

### Ignoring selection, or modelling it twice

**Symptom.** Well-detected bins are over-represented, or the correction overshoots.

{cite:t}`mandel2019` warn about two mistakes:

- **Correcting twice.** Do not renormalise each object's likelihood over the detectable
  region *and* pass a selection. Pass the ordinary likelihood $p(d_i\mid z)$ at the
  observed data.
- **Correcting nothing.** Multiplying likelihoods by $S$ without telling binference
  about the selection omits the normalisation.

**Fix.** Pass the ordinary likelihoods and a selection with its {term}`selection mode`
({doc}`selection`).

### Bins that are too coarse

**Symptom.** With many objects, the posterior excludes the truth even though
everything else is right.

**Cause.** The population is not flat within the bins, and the {term}`binning bias`
does not shrink with $N$. **Fix:** {doc}`bins`.

### Trusting approximate error bars

**Symptom.** Error bars narrower than an exact engine gives.

**Cause.** Mean-field VB, a Laplace fit whose check failed, or the bootstrap made
them. **Fix:** [approximations](engines.md#approximations).

### Too many bins for a flat Dirichlet

**Symptom.** With few objects per bin, every bin seems to hold some of the
population, and the fractions are biased toward equal values.

**Cause.** `Dirichlet(1.0)` adds one pseudo-count per bin. **Fix:** {doc}`priors`.

### Reading per-bin error bars as independent

**Symptom.** Error bars on sums, means or ratios that are too large or too small.

**Cause.** Neighbouring bins are correlated. **Fix:**
[use the draws](results.md#the-population).

### Confusing the population with the sample

**Symptom.** Error bars on the histogram of your own objects that include shot noise
those objects do not have.

**Cause.** $\rho$ describes the population, not these objects. **Fix:**
[the realised sample](results.md#population-versus-realised-sample).

### A smoothness scale near zero

**Symptom.** Divergent transitions with a random-walk prior, and a smoothness scale
whose posterior concentrates near zero.

An order-$r$ random walk does not penalise polynomials of degree below $r$; order 2
ignores straight lines in $\log\rho$. When the true population is close to such a
trend, for example an exponential spectrum, the data push the scale toward zero. The
posterior then has a funnel that is hard for any sampler.

**Fix.** Use `LogisticGP()`, raise `target_accept` to 0.95, or give the scale a
{term}`hyperprior` bounded away from zero, such as `priors.LogNormal(np.log(0.1), 1)`.

### Not looking at the diagnostics

**Symptom.** Results change between runs.

**Cause.** The draws do not yet represent the posterior. **Fix:**
[diagnostics](checking.md#diagnostics).
