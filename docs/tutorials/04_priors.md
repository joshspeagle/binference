---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Priors and smoothness

When data are sparse, with few objects per bin or errors as wide as the bins,
the prior shapes the answer. This tutorial puts several of binference's priors
side by side on a small catalogue, shows how smoothness hyperparameters are
learned, and uses held-out data to choose between priors.

```{code-cell} python
import warnings

import matplotlib.pyplot as plt
import numpy as np

import binference
from binference import priors

plt.rcParams["figure.dpi"] = 110
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
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True, constrained_layout=True)
for ax, (name, res) in zip(axes.flat, fits.items()):
    res.plot(ax=ax, truth=mock.truth.rho)
    ax.set_title(name, loc="left", fontsize=10)
    ax.set_xlabel("")
    if ax is not axes.flat[0]:
        ax.get_legend().remove()
fig.supxlabel("z");
```

- **`Dirichlet(1)`** adds one pseudo-count to each of the 40 bins, 40 in
  total against 400 objects. That props up the tails, and the bulk is noisy
  because neighbouring bins trade mass.
- **The Pólya tree** shares information between neighbouring bins through its
  dyadic tree. It is conjugate, so the Gibbs sampler can be used too.
- **The random walk and the GP** put a prior on the *shape* of the
  log-density and infer how smooth it is. They give the tightest bands and
  still cover the truth.

`Dirichlet.reference()` ($\alpha=1/K$) adds a single pseudo-count in total,
which suits many bins with few objects. Bins without data then have
enormously long tails in log space ($\rho_k$ down to $10^{-50}$ is
plausible), which NUTS explores slowly. With narrow likelihoods the conjugate
Gibbs sampler handles it well. binference's diagnostics point this out when they
see the symptoms.

## Learning the smoothness

The smoothness hyperparameters are sampled along with the population. Their
posteriors show how much structure the data support:

```{code-cell} python
gp = fits["Gaussian process"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
for ax, (name, draws) in zip(axes, gp.info["hyperparameters"].items()):
    ax.hist(draws.ravel(), bins=40, color="#2a78d6", alpha=0.8)
    ax.set_xlabel(name)
    ax.spines[["top", "right"]].set_visible(False)
print({k: np.quantile(v, [0.16, 0.5, 0.84]).round(2) for k, v in gp.info["hyperparameters"].items()})
```

Fixing a hyperparameter is a matter of passing a number, e.g.
`LogisticGP(length_scale=0.3)`. Pass a hyperprior to change its prior,
e.g. `LogisticRandomWalk(scale=priors.HalfNormal(0.2))`.

## Choosing a prior with held-out data

Which prior describes the population best is partly a scientific choice, but
predictive performance is an objective guide. Fit each prior on 300 objects
and score the other 100 with the log posterior-predictive density:

```{code-cell} python
train, test = lk[:300], lk[300:]
scores = {}
for name, prior in candidates.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = binference.infer(train, prior, rng=2, n_samples=400, n_warmup=400)
    scores[name] = res.log_predictive(test).sum()
best = max(scores.values())
for name, s in sorted(scores.items(), key=lambda kv: -kv[1]):
    print(f"{name:28s} {s - best:7.1f}")
```

Differences of a few units are meaningful for 100 held-out objects. For a
more careful comparison, cross-validate over several folds, for example with
{class}`~binference.PopulationEstimator` and its `score` method.

## Priors that encode outside information

- **A reference sample with known values**, say a spectroscopic subsample
  with counts `m`, can enter as `Dirichlet.from_counts(m, strength=λ)`.
  `strength=1` trusts it fully as a fair sample of the same population;
  smaller values down-weight it.
- **A reference distribution** (a model or a previous survey) can be the mean
  of a logistic prior, `LogisticGP(mean=np.log(ref))`. The amplitude then
  says how far the population may depart from it, and the data decide.

```{code-cell} python
ref = binference.simulate.population(bins, "gaussian")        # a deliberately wrong reference shape
centred = binference.infer(lk, priors.LogisticGP(mean=np.log(ref)), rng=1, n_samples=500, n_warmup=500)
ax = centred.plot(truth=mock.truth.rho)
ax.step(bins.edges[0], np.append(ref / bins.volumes, ref[-1] / bins.volumes[-1]), where="post",
        color="0.5", lw=1, label="reference (prior mean)")
ax.legend(frameon=False, fontsize=8)
ax.set_xlabel("z");
```

The posterior follows the data where they are informative, and falls back
toward the reference only where they are not.
