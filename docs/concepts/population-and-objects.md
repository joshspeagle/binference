---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Population and objects

*What does binference infer: the population, the objects you observed, or each object's
true value?*

It infers the first two, and a posterior for the third. This page separates them.

```{code-cell} python
:tags: [remove-cell]

import warnings

import matplotlib.pyplot as plt
import numpy as np
from myst_nb import glue
from scipy import stats

import binference

plt.rcParams["font.size"] = 14
CORAL, INK, INK2 = "#EC5F3E", "#1B1D4A", "#5D5F86"
RAMP = ["#B8BBEE", "#8186DE", "#4C53C2", "#2A2F8F"]  # the docs' sequential indigo ramp
```

## The population and the realised sample

The {term}`population` is the distribution the objects are drawn from. binference
describes it by the {term}`population fractions` $\rho_k$, the share of the population
in bin $k$. The fractions are non-negative and sum to one, so they lie on the
{term}`simplex` $\Delta^{K-1}$. On linear bins the population density within bin $k$
is $p(z\mid\rho)=\rho_k/v_k$; with another {term}`measure` it is flat in that coordinate
instead.

Your $N$ objects are one draw from that population. Their histogram of true values is
the {term}`realised sample` $n_k$. A second draw of $N$ objects would give a different
histogram, and each object's data add measurement noise on top ({ref}`the sample figure <fig-realised>`).

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-realised
    caption: |
      Top: a population on 15 bins (indigo). The coral lines are the histograms of
      three samples of 100 objects drawn from it. Bottom: 25 objects of
      one sample, sorted by true value. Each has a true value (open circle) and a
      measured value with its error bar (filled).
  image:
    alt: |
      Top panel: an indigo step histogram with two humps. Three coral step histograms
      scatter around it by up to about half its height. Bottom panel: 25 rows, one per
      object. Each row has an open circle at the true value and a filled coral point
      with a horizontal error bar at the measured value. The point is often displaced
      from the circle.
---
bins = binference.Bins.linear(0, 3, 15)
truth = binference.simulate.population(bins, "bimodal")
fig, axes = plt.subplots(2, 1, layout="constrained", figsize=(4.8, 5.1), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1.2]})
ax = axes[0]
ax.stairs(truth / bins.volumes, bins.edges[0], color="C0", lw=2.2, label=r"population $\rho$")
for seed, ls in zip((1, 2, 3), ("-", "--", ":"), strict=True):
    counts = binference.simulate.mock_1d(100, bins, density=truth, noise=0.15, rng=seed).truth.counts
    ax.stairs(counts / counts.sum() / bins.volumes, bins.edges[0], color=CORAL, lw=1.4, ls=ls,
              label="realised samples" if seed == 1 else None)
ax.set_ylabel("density")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0, fontsize=12, ncol=2)
mock = binference.simulate.mock_1d(25, bins, density=truth, noise=(0.1, 0.3), rng=1)
order = np.argsort(mock.truth.values)
rows = np.arange(order.size)
ax = axes[1]
ax.errorbar(mock.measurements["mean"][order], rows, xerr=mock.measurements["sigma"][order],
            fmt="o", ms=5, color=CORAL, lw=1.2, label="measured value $d_i$")
ax.plot(mock.truth.values[order], rows, "o", ms=6, mfc="none", mec=INK, mew=1.2,
        label="true value $z$")
ax.set_yticks([])
ax.set_ylabel("objects")
ax.set_xlabel("value")
ax.legend(fontsize=12, loc="lower right")
plt.show()
```

A property of the population, such as an
occurrence rate or a luminosity function, needs $\rho$. A property of these particular
objects, such as the redshift distribution of one lens sample, needs $n_k$. The
uncertainty of $n_k$ leaves out the noise of drawing $N$ objects, so it is narrower.
With perfect measurements $n_k$ is known exactly, while $\rho$ is still uncertain.

## Stacking and the hierarchical fit

Each object's posterior over bins, given a population $\rho$, is its set of
{term}`responsibilities`:

$$
r_{ik}(\rho)=\frac{L_{ik}\rho_k}{L_i\cdot\rho}.
$$

Averaging them over the objects gives a new estimate of the population. Starting from a
flat population, the first average is the stack of the objects' posteriors under a flat
prior. {term}`EM` repeats the average under each new estimate
({ref}`the EM figure <fig-em>`, top). Each step removes more of the widening caused by
measurement errors.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-em
    caption: |
      2,000 mock objects on 20 bins, with measurement errors between 0.15 and 0.35.
      Top: EM steps from a flat population. Step 1 is the stack; later steps are
      darker. Bottom: the hierarchical fit, with the stack for comparison. The fit is
      the posterior mean and 68% band from Gibbs, four chains of 1,000 draws. Dashed:
      the true population.
  image:
    alt: |
      Two stacked panels. In the top panel the palest line, the stack, is a smooth
      broad hump. It misses the dip between the two humps of the dashed truth. Darker
      lines for later steps deepen the dip and sharpen the humps. The darkest, step
      200, swings up and down between neighbouring bins. In the bottom panel an indigo
      band contains the dashed truth in most bins. The pale stack line stays too
      smooth.
---
bins = binference.Bins.linear(0, 3, 20)
truth = binference.simulate.population(bins, "bimodal")
lk = binference.simulate.mock_1d(2000, bins, density=truth, noise=(0.15, 0.35), rng=11).likelihoods
model = binference.PopulationModel(lk)
stack = lk.stack()  # one EM step from a flat population
steps, rho = {1: stack}, stack
for t in range(2, 201):
    rho = model.expected_counts(rho) / model.W
    if t in (5, 25, 200):
        steps[t] = rho
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fit = binference.infer(lk, method="gibbs", n_samples=1000, rng=1, check=False)
dens = lambda r: r / bins.volumes  # noqa: E731
fig, axes = plt.subplots(2, 1, layout="constrained", figsize=(4.8, 5.1), sharex=True)
ax = axes[0]
for (t, r), c in zip(steps.items(), RAMP, strict=True):
    ax.stairs(dens(r), bins.edges[0], color=c, lw=2.0,
              label="step 1 (the stack)" if t == 1 else f"step {t}")
ax = axes[1]
lo, hi = fit.interval(0.68)
ax.stairs(dens(hi), bins.edges[0], baseline=dens(lo), fill=True, color="C0", alpha=0.3,
          label="hierarchical fit, 68%")
ax.stairs(dens(fit.mean), bins.edges[0], color="C0", lw=2.0, label="hierarchical fit, mean")
ax.stairs(dens(stack), bins.edges[0], color=RAMP[0], lw=2.0, label="the stack")
for ax in axes:
    ax.stairs(dens(truth), bins.edges[0], color=INK, lw=1.4, ls="--", label="truth")
    ax.set_ylabel("population density")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0, fontsize=12, ncol=2)
axes[1].set_xlabel("true value $z$")
plt.show()
corr = fit.corr()
glue("neighbour_corr", float(np.median(np.diag(corr, 1))), display=False)
```

Run to convergence, EM reaches the most probable population, the {term}`MAP`. A
hierarchical fit goes further: it gives the posterior of the population, with its
uncertainty ({ref}`the EM figure <fig-em>`, bottom). The stack does neither. It is one step
of the iteration, too smooth wherever the likelihoods are broader than the features of
the population. The later EM steps in the figure turn jagged, for the reason the next
section gives.

## Correlations between bins

When likelihoods are broader than the bins, the data constrain the sum of neighbouring
bins better than either bin. Mass can shift between them with little change in the
likelihood, so their fractions are anti-correlated. In the fit of
{ref}`the EM figure <fig-em>`, the median correlation between neighbouring bins is
{glue:text}`neighbour_corr:.2f`. Per-bin error bars therefore do not combine like
independent errors; the pitfalls guide shows
[how to combine them](../guides/troubleshooting.md#reading-per-bin-error-bars-as-independent).

## Shrinkage: each object under the population

The population also informs each object. Let $k_i$ be the bin of object $i$, and $D$
the data of all objects. Object $i$'s {term}`hierarchical posterior` $P(k_i=k\mid D)$
is its responsibilities averaged over the posterior of $\rho$. It is narrower than the object's own likelihood wherever the
population is informative. This is {term}`shrinkage`, and it is strongest when the
population is narrow compared with the measurement error
({ref}`the shrinkage figure <fig-shrinkage>`).

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-shrinkage
    caption: |
      One object measured at 1.9 with error 0.4, on 30 bins, under two populations
      centred at 1.2 (indigo). Its posterior $r_{ik}(\rho)$ (solid coral) is its
      likelihood (dashed) times the population, normalised. Top: a narrow population,
      standard deviation 0.25. Bottom: a broad one, standard deviation 1. Each curve
      has unit area.
  image:
    alt: |
      Two stacked panels. In the top panel a narrow indigo population is centred at
      1.2, with a flat top from 1.1 to 1.3. The dashed likelihood peaks at 1.9. The solid posterior lies between them,
      peaking near 1.4, and is narrower than the likelihood. In the bottom panel the
      population is broad and low. The posterior nearly coincides with the
      likelihood.
---
bins = binference.Bins.linear(0, 3, 30)
one = binference.Likelihoods.from_gaussian([1.9], [0.4], bins)
unit = lambda m: m / (m.sum() * bins.volumes)  # noqa: E731  (masses to unit-area densities)
fig, axes = plt.subplots(2, 1, layout="constrained", figsize=(4.8, 4.5), sharex=True)
for ax, width, name in zip(axes, (0.25, 1.0), ("narrow population", "broad population"),
                           strict=True):
    pop = binference.simulate.population(bins, stats.norm(1.2, width))
    post = binference.PopulationModel(one).responsibilities(pop)[0]
    ax.stairs(unit(pop), bins.edges[0], color="C0", fill=True, alpha=0.25)
    ax.stairs(unit(pop), bins.edges[0], color="C0", lw=1.6, label=r"population $\rho$")
    ax.stairs(unit(one.toarray()[0] * bins.volumes), bins.edges[0], color=CORAL, lw=1.4,
              ls="--", label=r"likelihood $L_{ik}$")
    ax.stairs(unit(post), bins.edges[0], color=CORAL, lw=2.4, label=r"posterior $r_{ik}(\rho)$")
    ax.set_title(name, fontsize=14, loc="left")
    ax.set_ylabel("density")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2,
           fontsize=12)
axes[1].set_xlabel("true value $z$")
plt.show()
```

Shrinkage is not a bias to remove. Each object's own likelihood leaves out where the
other objects lie; the hierarchical posterior adds it.

```{plain-words}
The population is where objects like yours come from, and your catalogue is one draw
from it. binference estimates both, and where each object probably lies given the
population.
```

## Where next

- Do it: the results guide reads [the population](../guides/results.md#the-population),
  [individual objects](../guides/results.md#individual-objects) and
  [the realised sample](../guides/results.md#population-versus-realised-sample) from a fit.
- See it: the tutorial {doc}`../tutorials/01_quickstart`.
- Derivation: [the augmented form](../how-it-works/model.md#the-augmented-form)
  in How it works.
