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
from matplotlib.colors import to_rgb
from myst_nb import glue

import binference
from binference.plotting import SAMPLE_COLOR

CORAL, INK = SAMPLE_COLOR, plt.rcParams["text.color"]
c0, ground = np.array(to_rgb("C0")), np.array(to_rgb(plt.rcParams["axes.facecolor"]))
PALE = 0.35 * c0 + 0.65 * ground  # a pale indigo
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
fig, axes = plt.subplots(2, 1, sharex=True,
                         gridspec_kw={"height_ratios": [1, 1.2]})
fig.set_figheight(5.1)
ax = axes[0]
ax.stairs(truth / bins.volumes, bins.edges[0], color="C0", lw=2.2, label=r"population $\rho$")
for seed, ls in zip((1, 2, 3), ("-", "--", ":"), strict=True):
    counts = binference.simulate.mock_1d(100, bins, density=truth, noise=0.15, rng=seed).truth.counts
    ax.stairs(counts / counts.sum() / bins.volumes, bins.edges[0], color=CORAL, lw=1.4, ls=ls,
              label="realised samples" if seed == 1 else None)
ax.set_ylabel("density")
fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=2)
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
ax.legend(loc="lower right")
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
(explainer {ref}`explainer-v4`). Each step removes more of the widening caused by measurement errors.

```{explainer} v4
800 mock objects on 20 bins, measured with error 0.3. Indigo: the population after the
chosen number of EM steps from a flat one; grey: step 1, the stack; dashed: the true
population. Step forward and the two peaks sharpen; by step 200 the estimate swings
between neighbouring bins.
```

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-em
    caption: |
      2,000 mock objects on 20 bins, with measurement errors between 0.15 and 0.35.
      Indigo: the hierarchical fit, the posterior mean and 68% band from Gibbs, four
      chains of 1,000 draws. Pale: the stack. Dashed: the true population.
  image:
    alt: |
      An indigo band contains the dashed truth, which has two humps, in most bins. The
      pale stack line is a smooth broad hump that misses the dip between them.
---
bins = binference.Bins.linear(0, 3, 20)
truth = binference.simulate.population(bins, "bimodal")
lk = binference.simulate.mock_1d(2000, bins, density=truth, noise=(0.15, 0.35), rng=11).likelihoods
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fit = binference.infer(lk, method="gibbs", n_samples=1000, rng=1, check=False)
dens = lambda r: r / bins.volumes  # noqa: E731
fig, ax = plt.subplots()
fig.set_figheight(3.8)
lo, hi = fit.interval(0.68)
ax.stairs(dens(hi), bins.edges[0], baseline=dens(lo), fill=True, color="C0", alpha=0.3,
          label="hierarchical fit, 68%")
ax.stairs(dens(fit.mean), bins.edges[0], color="C0", lw=2.0, label="hierarchical fit, mean")
ax.stairs(dens(lk.stack()), bins.edges[0], color=PALE, lw=2.0, label="the stack")
ax.stairs(dens(truth), bins.edges[0], color=INK, lw=1.4, ls="--", label="truth")
ax.set(xlabel="true value $z$", ylabel="population density")
fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=2)
plt.show()
corr = fit.corr()
glue("neighbour_corr", float(np.median(np.diag(corr, 1))), display=False)
```

Run to convergence, EM reaches the most probable population, the {term}`MAP`. A
hierarchical fit goes further: it gives the posterior of the population, with its
uncertainty ({ref}`the hierarchical-fit figure <fig-em>`). The stack does neither. It is one step
of the iteration, too smooth wherever the likelihoods are broader than the features of
the population. The later EM steps in explainer {ref}`explainer-v4` turn jagged, for the reason the next
section gives.

## Correlations between bins

When likelihoods are broader than the bins, the data constrain the sum of neighbouring
bins better than either bin. Mass can shift between them with little change in the
likelihood, so their fractions are anti-correlated. In the fit of
{ref}`the hierarchical-fit figure <fig-em>`, the median correlation between neighbouring bins is
{glue:text}`neighbour_corr:.2f`. Per-bin error bars therefore do not combine like
independent errors; the pitfalls guide shows
[how to combine them](../guides/troubleshooting.md#reading-per-bin-error-bars-as-independent).

## Shrinkage: each object under the population

The population also informs each object. Let $k_i$ be the bin of object $i$, and $D$
the data of all objects. Object $i$'s {term}`hierarchical posterior` $P(k_i=k\mid D)$
is its responsibilities averaged over the posterior of $\rho$. It is narrower than the object's own likelihood wherever the
population is informative. This is {term}`shrinkage`, and it is strongest when the
population is narrow compared with the measurement error
(explainer {ref}`explainer-v5`).

```{explainer} v5
One object measured at 1.9 with error 0.4, on 30 bins, under a population centred at 1.2
(indigo). Its posterior $r_{ik}(\rho)$ (solid coral) is its likelihood (dashed) times the
population, normalised; each curve has unit area. Narrow the population and the posterior
moves toward it; widen it and the posterior returns to the likelihood.
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
- See it: the tutorial {doc}`../tutorials/gaussian-quickstart`.
- Derivation: [the augmented form](../how-it-works/model.md#the-augmented-form)
  in How it works.
