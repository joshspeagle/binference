# Posterior samples

Use this page if each object comes as a set of samples: MCMC chains, nested-sampling
output, or draws from a likelihood. Gravitational-wave catalogues publish their events
this way. {meth}`~binference.Likelihoods.from_samples` turns the samples into
likelihoods over the bins.

```python
import numpy as np
import binference

rng = np.random.default_rng(1)
bins = binference.Bins.linear(0.0, 3.0, 30)
z_obs = rng.uniform(0.2, 2.8, size=100)
sigma = rng.uniform(0.05, 0.3, size=100)

samples = [z + s * rng.standard_normal(200) for z, s in zip(z_obs, sigma)]
lk = binference.Likelihoods.from_samples(samples, bins, kind="likelihood")
print(lk.N, lk.is_sparse, np.median(lk.meta["n_eff"]).round(0))
```

## The interim prior is required

{term}`Posterior samples <posterior samples>` were drawn under some prior: the
{term}`interim prior`, $\pi(z)$. binference divides it out, so `interim` has no default.
Describe it as on the {doc}`posteriors page <posteriors>`. Samples drawn from a
likelihood are samples under `binference.Density.flat("z")`; `kind="likelihood"` says so.

How binference divides it out depends on what you pass:

- **A function or a distribution.** Each sample is weighted by $T'(z)/\pi(z)$, with
  $T$ the bins' {term}`measure`. This is exact even when the interim prior varies
  within a bin.
- **Values per bin** (`"flat"`, masses or densities). The interim prior is taken as
  flat in the measure coordinate within each bin. Each object's likelihood is then its
  count of samples in the bin divided by the bin's interim mass.

Ragged lists, with a different number of samples per object, are accepted.
`sample_weights=` gives each sample an importance weight $\omega_s$, in the same
layout as `samples`.

## Enough samples

A histogram of a few samples is noisy, and so are the likelihoods built from it.
binference measures this in three places:

- `lk.meta["n_eff"]` is each object's effective number of samples inside the bins, and
  `lk.meta["outside_fraction"]` its share outside. binference warns when many objects
  have fewer than `min_samples` effective samples
  ({doc}`defaults <../../reference/defaults>`).
- `lk.sample_variance(rho)` gives, at a population `rho`, each object's relative Monte
  Carlo variance $V_i(\rho)$ and its effective sample size.
- `result.check()` warns when $\sum_i V_i$ at the fitted population exceeds its limit
  ({doc}`defaults <../../reference/defaults>`). The sample noise biases the total
  log-likelihood low by about half that sum. Each $V_i$ falls like $1/S$ with $S$
  samples per object, so the sum grows like $N/S$.

Bins that no sample reached get zero likelihood. If an object's samples miss bins its
likelihood should cover, pass `floor=a`. It adds `a` pseudo-samples per object, spread
over the bins in proportion to the interim prior.

```python
V, n_eff = lk.sample_variance(binference.infer(lk, method="map").rho)
print(V.sum().round(3))
```

If the variance is too large, draw more samples per object, or widen the bins.
`from_samples` stores the likelihoods sparsely when few entries are non-zero
({doc}`advanced`).

## Smoothing the samples

`smooth="kde"` spreads each sample over the bins by a Gaussian kernel. The kernel lives in
the measure coordinate and is reflected at the ends of the range. `bandwidth=` sets its
width: Silverman's rule per object by default, or one number for every object. It works
on 1-D bins.

It pays when the bins are much narrower than the posteriors. In a test, Gaussian
posteriors had width 0.08 on 100 bins over $[0, 1]$. With 30 samples, the per-object
error of the bin masses fell from 1.11 to 0.30; with 1000 samples, from 0.19 to 0.08. On
20 bins with 100 samples it was 0.42 against 0.32. On bins wider than the posteriors the
two tied.

It has three costs.

- **A bias.** The kernel widens each posterior by its bandwidth: by about 10% with 30
  samples and 3% with 1000. That bias falls more slowly than the histogram's noise.
  `sample_variance` and `check()` measure only the noise.
- **Several peaks.** Silverman's rule takes its width from the whole sample, so a
  posterior with two peaks is blurred. In the test, posteriors with peaks at $d$ and
  $1-d$ gave a fitted population spread of 0.15 to 0.18 against a true 0.21.
- **Time.** On one thread, 10,000 objects with 1000 samples each on 20 bins took 11.5 s,
  against 2.3 s for the histogram. With 30 samples on 100 bins the ratio was about 16.

The scripts are in the repository's `notes/review/experiments/233`.
