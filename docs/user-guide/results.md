# Working with results

Sampling engines return a {class}`~binference.PopulationResult`; `method="map"`
returns a {class}`~binference.PointEstimate`. A result holds posterior draws of the
population, the model they came from, and diagnostics. Everything else is
computed from the draws, so uncertainties propagate correctly into any
derived quantity.

```python
import numpy as np
import binference

mock = binference.simulate.mock_1d(800, binference.Bins.linear(0, 3, 15), rng=6)
lk, bins = mock.likelihoods, mock.bins
result = binference.infer(lk, rng=1, n_samples=500, n_warmup=500)
print(result)
```

## The population

```python
X = result.samples              # (n_draws, K) draws of the fractions, chains pooled
C = result.chains               # (n_chains, n_draws_per_chain, K)
mean, sd = result.mean, result.std
lo, hi = result.interval(0.68)               # equal-tailed; kind="hdi" for HDIs
dens = result.density()                      # per unit bin volume (an n(z))
cov = result.cov()                           # (K, K): neighbouring bins anti-correlate
print(X.shape, C.shape, np.round(result.corr()[5, 4:7], 2))
print(result.summary())
```

**Use the draws, not the marginals.** Bins are correlated, often strongly
negatively for neighbours, so error bars on sums or ratios of bins cannot be
built from per-bin standard deviations. Compute any derived quantity per draw
instead:

```python
zbar = result.functional(lambda rho: rho @ bins.centers)          # mean of the population
frac_high = result.functional(lambda rho: rho[bins.centers > 2].sum())
print(f"mean z = {zbar.mean():.3f} +/- {zbar.std():.3f}")
coarse_bins, mapping = bins.coarsen(3)
coarse = result.rebin(mapping)          # (n_draws, 5): fractions in 5 coarse bins
print(coarse.mean(axis=0).round(3), coarse.std(axis=0).round(3))
```

For rate models, `result.rates` holds draws of the expected counts and
`result.samples` the implied fractions. Pass `which="rates"` to `quantile`,
`interval` and `summary`. Convergence diagnostics for rate models are
computed on the rates, so they also cover the total.

## Individual objects

The hierarchical model also says something about each object: its posterior
over bins given **its own data and the population**. This is narrower than its
likelihood wherever the population is informative (shrinkage).

```python
p = result.assignment_probabilities(idx=[0, 1, 2])       # (3, K), rows sum to one
labels = result.sample_assignments(idx=[0, 1, 2], n_draws=100, rng=0)   # joint label draws
values = result.sample_values(idx=[0], n_draws=1000, rng=0)              # true-value draws
print(p.round(2)[0], labels.shape, values.mean().round(3), mock.truth.values[0].round(3))
```

`sample_assignments` gives exact joint draws of $(\rho, z)$, whatever engine
produced $\rho$. `sample_values` then draws a true value inside the drawn
bin from the object's own likelihood, since the population is flat within a
bin:

- exactly, for likelihoods built with `Likelihoods.from_gaussian`;
- for other 1-D likelihoods, pass `loglike(z, i)` (the log-likelihood of
  object `i` at points `z`) to sample on a fine sub-grid of each bin;
- otherwise uniformly within the bin, which is accurate only when the
  likelihoods are nearly flat across a bin.

Both methods treat each row of the likelihoods as one object, so use unit
weights (uncompressed rows) when you need draws for individual objects.

For **new** objects that were not used in the fit:

```python
new = binference.Likelihoods.from_gaussian(np.array([1.0, 2.5]), np.array([0.3, 0.3]), bins)
print(result.predict_proba(new).round(2)[0])
print(result.log_predictive(new).round(3))     # log p(d_new | D), for model comparison
```

## Population versus realised sample

$\rho$ describes the population the objects came from, including the "shot
noise" of having only $N$ of them. Sometimes you need the histogram of
**these** objects instead, for example the redshifts of a particular lens
sample:

```python
counts = result.realized_counts(n_draws=500, rng=0)     # (500, K) draws of the realised histogram
population_counts = result.samples * lk.W                # population fractions as counts
print(counts.sum(axis=1)[:3])
print(counts.std(axis=0)[:5].round(1), population_counts.std(axis=0)[:5].round(1))
```

With precise measurements the realised histogram is almost known, while the
population fractions still carry multinomial uncertainty.

## Diagnostics

```python
d = result.diagnostics                     # dict: rhat, ess_bulk, ess_tail, divergences, ...
print(d["rhat"].max().round(3), d["ess_bulk"].min().round(0), d.get("n_divergent"))
problems = result.check()                  # warns (ConvergenceWarning) and returns messages
print(problems)
```

`check()` flags $\hat R>1.01$, bulk ESS below 100 per chain, NUTS
divergences, saturated trees and unreliable importance weights.
`check(strict=True)` raises instead. Where it can, the warning ends with a
hint: a higher `target_accept` after divergences, `method="gibbs"` for
conjugate priors with very small concentrations, and, when Gibbs mixes
slowly, the fraction of missing information that explains why. A result
from an approximate engine is marked `result.approximate`, and its summary
says so.

## Plots

```python
ax = result.plot(truth=mock.truth.rho, stack=lk.stack())   # bands, median, overlays
fig = result.plot_corner(idx=[4, 5, 6])                     # pairwise posteriors
axes = result.plot_trace(idx=[5])                           # chains
```

`result.plot` shows densities per unit volume for 1-D bins (`which="fraction"`
for fractions), images for 2-D grids and bars for categories.
{func}`binference.plotting.plot_individual` compares one object's likelihood with
its shrunken posterior.

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

A {class}`~binference.PointEstimate` offers the same per-object methods,
evaluated at the point. `est.laplace()` turns it into an approximate
posterior.
