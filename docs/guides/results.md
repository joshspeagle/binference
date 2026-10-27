# Read a result

Sampling engines return a {class}`~binference.PopulationResult`; `method="map"`
returns a {class}`~binference.PointEstimate`. A result holds posterior {term}`draws
<draw>` of the population, the model they came from, and diagnostics. Everything else
is computed from the draws, so uncertainties carry over into any derived quantity.
Before you use a result, check it ({doc}`checking`).

```python
import numpy as np
import binference

mock = binference.simulate.mock_1d(800, binference.Bins.linear(0, 3, 15), rng=6)
lk, bins = mock.likelihoods, mock.bins
result = binference.infer(lk, method="nuts", rng=1)
print(result)
```

## The population

```python
X = result.samples              # (n_draws, K) draws of the fractions, chains pooled
C = result.chains               # (n_chains, n_draws_per_chain, K)
mean, sd = result.mean, result.std
lo, hi = result.interval(0.68)               # equal-tailed; kind="hdi" for HDIs
dens = result.density()                      # fractions per unit bin volume
cov = result.cov()                           # (K, K): neighbouring bins anti-correlate
print(X.shape, C.shape, np.round(result.corr()[5, 4:7], 2))
print(result.summary())
```

The summary has one row per bin:

- `mean`, `sd`, `median`: of the bin's fraction over the draws;
- `q16`, `q84`: the bounds of the central 68% {term}`credible interval`
  (`summary(level=...)` changes it);
- `ess_bulk`, `rhat`: the bin's {term}`effective sample size` and {term}`R-hat`
  ({doc}`checking`);
- `support`: the bin's objects' worth of data ({doc}`bins`).

Its title names the engine, the number of draws and the kind of input. An approximate
result says so there.

**Use the draws, not the marginals.** Bins are correlated, and neighbours often
strongly anti-correlated. Error bars on sums or ratios of bins therefore cannot be built
from per-bin standard deviations. Compute any derived quantity per draw instead:

```python
zbar = result.functional(lambda rho: rho @ bins.centers)          # mean of the population
frac_high = result.functional(lambda rho: rho[bins.centers > 2].sum())
print(f"mean z = {zbar.mean():.3f} +/- {zbar.std():.3f}")
coarse_bins, mapping = bins.coarsen(3)
coarse = result.rebin(mapping)          # (n_draws, 5): fractions in 5 coarse bins
print(coarse.mean(axis=0).round(3), coarse.std(axis=0).round(3))
```

`result.rates` holds draws of the {term}`rates`, the expected counts per bin, and
`result.total` their sum. Rate models sample them. For fractions, they are drawn
exactly from each draw of $\rho$, given the number of objects and the selection
({doc}`selection`). Pass `which="rates"` to `quantile`, `interval` and
`summary`. Convergence diagnostics for rate models are computed on the rates,
so they also cover the total.

With a selection function, `result.selected` holds draws of the
{term}`selected population`, $\phi=S\circ\rho/(S\cdot\rho)$: the bins of the objects
that are observed. When the selection is inferred, each draw uses its own $S$.
`which="selected"` works wherever `which=` does.

Two numbers say how much of $\rho$ the data determine. `result.data_support` gives
each bin's objects' worth of data ({doc}`bins`). `result.prior_information_ratio()`
compares the prior's information with the data's ({doc}`priors`).

```python
print(result.data_support.round(1), round(result.prior_information_ratio(), 3))
```

## Individual objects

The model also says something about each object: its {term}`hierarchical posterior`
over bins, given **its own data and the population**. It is narrower than the object's
likelihood wherever the population is informative ({term}`shrinkage`).

```python
p = result.assignment_probabilities(idx=[0, 1, 2])       # (3, K), rows sum to one
labels = result.sample_assignments(idx=[0, 1, 2], n_draws=100, rng=0)   # joint label draws
values = result.sample_values(idx=[0], n_draws=1000, rng=0)              # true-value draws
print(p.round(2)[0], labels.shape, values.mean().round(3), mock.truth.values[0].round(3))
```

`sample_assignments` gives joint draws of $\rho$ and the objects' bins $k_i$. They are
exact when the draws of $\rho$ are exact. Draws from an approximation (VB, Laplace,
the bootstrap) carry its error over. `sample_values` then draws a true value inside
the drawn bin from the object's own likelihood, since the population is flat within a
bin. It uses what each input knows within bins:

- Gaussians (`from_gaussian`): exact truncated draws, and Gibbs sweeps over the axes
  for correlated errors (`n_sweeps`, default 20);
- densities (`from_density`, 1-D bins): the density on a fine sub-grid of the bin, with
  the interim prior divided out for posteriors;
- samples (`from_samples`): the object's own samples inside the bin, resampled with
  the interim prior divided out (`keep_samples=False` saves their copy and drops this);
- `loglike(z, i)` (the log-likelihood of object `i` at points `z`), when passed, for
  any 1-D input;
- otherwise (tables, binned inputs, compressed rows, densities on grids or without a
  pdf, samples not kept, or a density function lost when the likelihoods were pickled)
  uniformly within the bin, with an `InputWarning`: accurate only when the likelihoods
  are nearly flat across a bin.

Both methods treat each row of the likelihoods as one object, so use unit
weights (uncompressed rows) when you need draws for individual objects.

For **new** objects that were not used in the fit:

```python
new = binference.Likelihoods.from_gaussian(np.array([1.0, 2.5]), np.array([0.3, 0.3]), bins)
print(result.predict_proba(new).round(2)[0])
print(result.log_predictive(new).round(3))     # log p(d_new | D), for model comparison
```

Every result also carries its **best fit**, `result.best`: the estimate of
`binference.samplers.MAP` with its defaults, computed on first use
([point estimates](engines.md#point-estimates)). After `load`, attach a model built
with the fit's prior to get it.
Per-object answers at it carry a `_best` suffix: `assignment_probabilities_best`,
`sample_assignments_best`, `sample_values_best`, `realized_counts_best`,
`predict_proba_best` and `log_predictive_best`. They ignore the uncertainty of the
population, which the methods without the suffix include.

To check the per-object posteriors against held-out truth, use the
{term}`leave-one-out` posteriors: each object's posterior under the population fitted
without it. `assignment_probabilities_loo` gets them from the posterior draws by
Pareto-smoothed importance sampling, and returns the {term}`Pareto k-hat` per object.
It warns about objects whose $\hat k$ is too large to trust; refit without them.

```python
p_loo, k_hat = result.assignment_probabilities_loo(idx=[0, 1, 2])
print(p_loo.round(2)[0], k_hat.round(2))
```

## Population versus realised sample

$\rho$ describes the population the objects came from. For the histogram of **these**
objects, the {term}`realised sample`, draw it
([why they differ](../how-it-works/theory.md#population-versus-realised-sample)):

```python
counts = result.realized_counts(n_draws=500, rng=0)     # (500, K) draws of the realised histogram
population_counts = result.samples * lk.W                # population fractions as counts
print(counts.sum(axis=1)[:3])
print(counts.std(axis=0)[:5].round(1), population_counts.std(axis=0)[:5].round(1))
```

With an absolute selection,
`realized_counts(full=True)` adds the objects the selection missed
([selection and rates](selection.md#the-objects-the-catalogue-missed)).

## Plots

```python
ax = result.plot(truth=mock.truth.rho, stack=lk.stack())   # bands, median, overlays
fig = result.plot_corner(idx=[4, 5, 6])                     # pairwise posteriors
axes = result.plot_trace(idx=[5])                           # chains
```

`result.plot` shows densities per unit volume for 1-D bins (`which="fractions"` or
`"rates"`; singular spellings work too, as everywhere `which=` appears), images for
2-D grids and bars for categories. `plot_corner`, `plot_trace`, `functional`, `rebin`,
`cov` and `corr` take the same `which=`. `result.plot_individual(i)` compares one
object's likelihood with its shrunken posterior.

## Saving and sharing

```python
import tempfile, os

path = os.path.join(tempfile.mkdtemp(), "fit.npz")
result.save(path)
back = binference.PopulationResult.load(path)       # draws, bins, info; no likelihoods
back.attach(lk)                                 # re-attach to use per-object methods
tree = result.to_datatree()                     # xarray DataTree in ArviZ layout
print(back.samples.shape, list(tree.children))
```

External draws, for example from your own sampler, can be wrapped with
`PopulationResult.from_samples(model, draws)` to use all of the above.

## Point estimates

```python
est = binference.infer(lk, method="map")
print(est.rho.round(3)[:5], est.kkt_gap, round(est.log_likelihood, 2))
print(est.assignment_probabilities(idx=[0]).round(2))
```

A {class}`~binference.PointEstimate` offers the same per-object methods, evaluated at
the point. `est.laplace()` turns it into an approximate posterior. {doc}`checking` says
how to check it.
