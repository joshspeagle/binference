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
```

## The population

```python
X = result.samples              # (n_draws, K) draws of the fractions, chains pooled
C = result.chains               # (n_chains, n_draws_per_chain, K)
mean, sd = result.mean, result.std
lo, hi = result.interval(0.68)               # equal-tailed; kind="hdi" for HDIs
dens = result.density()                      # fractions per unit bin volume
cov = result.cov()                           # (K, K): neighbouring bins anti-correlate
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
`result.total` their sum. Rate models sample them. With a prior on fractions, binference
draws the rates from each draw of $\rho$, given $W$ and the selection
({doc}`selection`). Pass `which="rates"` to `quantile`, `interval` and
`summary`. Convergence diagnostics for rate models are computed on the rates,
so they also cover the total.

With a selection function, `result.selected` holds draws of the
{term}`selected population`, $\phi_k=S_k\rho_k/\sum_{k'}S_{k'}\rho_{k'}$: the bins of the objects
that are observed. When the selection is inferred, each draw uses its own $S$.
`which="selected"` works wherever `which=` does.

To see how much of each bin's fraction the data determine, read its {term}`data support`
({doc}`bins`).

## Individual objects

The model also says something about each object: its {term}`hierarchical posterior`
over bins, given **its own data and the population**. It is narrower than the object's
likelihood wherever the population is informative ({term}`shrinkage`).

```python
p = result.assignment_probabilities(idx=[0, 1, 2])       # (3, K), rows sum to one
labels = result.sample_assignments(idx=[0, 1, 2], n_draws=100, rng=0)   # joint label draws
values = result.sample_values(idx=[0], n_draws=1000, rng=0)              # true-value draws
print(p[0].round(2))
```

`sample_assignments` gives joint draws of $\rho$ and the objects' bins $k_i$. They are
exact when the draws of $\rho$ are exact. Draws from an approximation (VB, Laplace,
the bootstrap) carry its error over. `sample_values` then draws a true value inside
the drawn bin from the object's own likelihood, since the population is flat within a
bin. What it uses within the bin depends on the input; see
{meth}`~binference.PopulationResult.sample_values`. For tables and binned inputs it
draws flat within the bin and warns; pass `loglike=` to do better.

Both methods treat each row of the likelihoods as one object, so use unit
weights (uncompressed rows) when you need draws for individual objects.

For **new** objects that were not used in the fit:

```python
new = binference.Likelihoods.from_gaussian(np.array([1.0, 2.5]), np.array([0.3, 0.3]), bins)
print(result.log_predictive(new).round(3))     # log p(d_new | D), for model comparison
```

Every result also carries its MAP estimate, `result.best`. Each per-object method has a
twin with a `_best` suffix that uses it and ignores the uncertainty of the population
({attr}`~binference.PopulationResult.best`).

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
([why they differ](../how-it-works/model.md#population-versus-realised-sample)):

```python
counts = result.realized_counts(n_draws=500, rng=0)     # (500, K) draws of the histogram
print(counts.mean(axis=0)[:5].round(1))
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

`result.plot` shows densities per unit volume for 1-D bins, images for 2-D grids and
bars for categories. It takes `which="fractions"` or `"rates"`, as do `plot_corner`,
`plot_trace`, `functional`, `rebin`, `cov` and `corr`. `result.plot_individual(i)`
compares one object's likelihood with its hierarchical posterior.

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

To use all of the above with draws from your own sampler, wrap them with
`PopulationResult.from_samples(model, draws)`.

## Point estimates

```python
est = binference.infer(lk, method="map")
print(est.rho.round(3)[:5])
```

A {class}`~binference.PointEstimate` offers the same per-object methods, evaluated at
the point. `est.laplace()` turns it into an approximate posterior. {doc}`checking` says
how to check it.
