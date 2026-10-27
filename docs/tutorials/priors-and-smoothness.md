---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Priors and smoothness

When the data are sparse, the prior shapes the answer. Sparse means few objects per bin,
or errors as wide as the bins. This tutorial puts four of binference's priors side by side
on a small catalogue. It shows how the smoothness {term}`hyperparameters <hyperparameter>`
are learned, and uses held-out objects to choose between priors.

```{code-cell} python
:tags: [remove-cell]

# figures 6 in wide with 17 pt text keep text above 12 px on a 390 px phone screen
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.figsize": (6, 3.4), "font.size": 17,
                     "figure.constrained_layout.use": True})
```

```{code-cell} python
import warnings

import matplotlib.pyplot as plt
import numpy as np

import binference
from binference import priors

bins = binference.Bins.linear(0, 3, 40)
mock = binference.simulate.mock_1d(400, bins, noise=(0.05, 0.15), rng=21)
lk = mock.likelihoods
print(f"{lk.N} objects in {lk.K} bins: {lk.N / lk.K:.0f} per bin on average")
```

## Four priors, one data set

```{code-cell} python
candidates = {
    "Dirichlet(1)": priors.Dirichlet(1.0),
    "Pólya tree": priors.PolyaTree(c=1.0),
    "random walk (order 2)": priors.LogisticRandomWalk(order=2),
    "Gaussian process": priors.LogisticGP(),
}
fits = {}
for name, prior in candidates.items():
    fits[name] = binference.infer(lk, prior, rng=1, n_samples=500, n_warmup=500)
```

```{code-cell} python
---
mystnb:
  figure:
    caption: The population posterior (median, 68% and 95% bands) of the same 400 objects
      under four priors. The truth is dashed.
  image:
    alt: Four stacked step plots of density against z from 0 to 3. Under the Dirichlet
      prior the bands are wide and the median is ragged. Under the Pólya tree they are
      narrower but still ragged. Under the random walk and the Gaussian process they are
      narrow and smooth, and follow the dashed truth with its two peaks.
---
fig, axes = plt.subplots(4, 1, figsize=(6, 11), sharex=True, sharey=True)
for ax, (name, res) in zip(axes, fits.items()):
    res.plot(ax=ax, truth=mock.truth.rho, label="median")
    ax.set_title(name, loc="left")
    ax.set_xlabel("")
    ax.get_legend().remove()
axes[-1].set_xlabel("z")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2);
```

```{code-cell} python
:tags: [remove-cell]

width = {name: np.median(res.std) for name, res in fits.items()}
smooth_widths = [width["random walk (order 2)"], width["Gaussian process"]]
assert max(smooth_widths) < min(width["Dirichlet(1)"], width["Pólya tree"])
```

- **`Dirichlet(1)`** adds one pseudo-count to each of the 40 bins: 40 in all, against 400
  objects. That props up the tails. The bulk is noisy, because neighbouring bins trade
  mass.
- **The {term}`Pólya tree`** shares information between neighbouring bins through its
  tree of halvings. It is a {term}`conjugate prior <conjugate prior>`, so the
  {term}`Gibbs` sampler can be used too.
- **The random walk and the Gaussian process** are {term}`smoothness priors <smoothness
  prior>`. They put a prior on the shape of $\log\rho$ and infer how smooth it is. They
  give the narrowest bands and still cover the truth.

`Dirichlet.reference()` sets the {term}`concentration` $\alpha_k=1/K$, a single
pseudo-count in all. It suits many bins with few objects. Bins without data then have
very long tails in log space: $\rho_k$ down to $10^{-50}$ is plausible. NUTS explores
such tails slowly. With narrow likelihoods the conjugate Gibbs sampler handles them well,
and binference's diagnostics flag the symptoms.

## Learning the smoothness

binference samples the smoothness hyperparameters $\theta$ along with the population.
Their posteriors show how much structure the data support:

```{code-cell} python
---
mystnb:
  figure:
    caption: Posterior draws of the Gaussian-process prior's amplitude (top) and length
      scale (bottom).
  image:
    alt: Two stacked histograms. The amplitude's draws spread from about 1 to 5, with a
      peak near 2. The length scale's draws spread from about 0.3 to 2.5, with a peak
      near 1.
---
gp = fits["Gaussian process"]
hyper = gp.info["hyperparameters"]
fig, axes = plt.subplots(2, 1, figsize=(6, 5.6))
for ax, (name, draws) in zip(axes, hyper.items()):
    ax.hist(draws.ravel(), bins=40, color="C0", alpha=0.8)
    ax.set(xlabel=name, ylabel="draws")
print({k: np.quantile(v, [0.16, 0.5, 0.84]).round(2) for k, v in hyper.items()})
```

To fix a hyperparameter, pass a number, as in `LogisticGP(length_scale=0.3)`. To change
its prior, pass a {term}`hyperprior`, as in
`LogisticRandomWalk(scale=priors.HalfNormal(0.2))`.

## Choosing a prior with held-out data

Which prior describes the population best is partly a scientific choice. Predictive
performance is an objective guide. Fit each prior to 300 objects, and score the other 100
with the {term}`log predictive density`:

```{code-cell} python
train, test = lk[:300], lk[300:]
scores = {}
for name, prior in candidates.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # shorter runs than the default, for speed
        res = binference.infer(train, prior, rng=2, n_samples=300, n_warmup=300)
    scores[name] = res.log_predictive(test).sum()
best = max(scores.values())
for name, s in sorted(scores.items(), key=lambda kv: -kv[1]):
    print(f"{name:28s} {s - best:7.1f}")
```

Higher is better, and the best prior scores zero. Differences of a few units are
meaningful for 100 held-out objects. For a more careful comparison, cross-validate over
several folds, for example with {class}`~binference.PopulationEstimator` and its `score`
method.

## Priors that encode outside information

- **A reference sample with known values**, such as a spectroscopic subsample, has
  counts $n^{\rm ref}_k$ per bin. They enter as
  `Dirichlet.from_counts(counts, strength=kappa)`, which sets
  $\alpha_k=1+\kappa\,n^{\rm ref}_k$. $\kappa=1$ trusts the sample fully, as a fair draw
  from the same population; smaller values down-weight it.
- **A reference distribution**, from a model or a previous survey, can be the mean
  $\mu$ of a logistic prior: `LogisticGP(mean=np.log(ref))`. The amplitude then says how
  far the population may depart from it, and the data decide.

```{code-cell} python
---
mystnb:
  figure:
    caption: The population posterior under a Gaussian-process prior centred on a
      reference shape (grey) that is deliberately wrong. The truth is dashed.
  image:
    alt: Step plot of density against z. The grey reference is a single peak at 1.5.
      The posterior bands follow the dashed truth with its two peaks, not the reference.
---
ref = binference.simulate.population(bins, "gaussian")   # a deliberately wrong reference
centred = binference.infer(lk, priors.LogisticGP(mean=np.log(ref)), rng=1,
                           n_samples=500, n_warmup=500)
fig, ax = plt.subplots(figsize=(6, 4.2))
centred.plot(ax=ax, truth=mock.truth.rho, label="median")
ax.step(bins.edges[0], np.append(ref, ref[-1]) / bins.volumes[0], where="post",
        color="0.45", lw=1.2, label="reference")
ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1))
ax.set_xlabel("z");
```

The posterior follows the data where they are informative. It falls back toward the
reference only where they are not.

## What to take away

- With sparse data, a smoothness prior gives narrower bands than a Dirichlet prior and
  still covers the truth here.
- Smoothness hyperparameters are inferred; fix them or change their hyperprior if you
  know better.
- The log predictive density of held-out objects compares priors on the data. The
  [priors guide](../guides/priors) lists every prior.
