---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# What binference does

What can you learn about a population when each of its members is measured with noise?

Suppose you have measured many {term}`objects <object>`: galaxies, merger events,
patients, classified documents. Each measurement has an error, so no single object's
true value is known exactly. binference answers two questions from such data:

- What is the {term}`population` the objects come from? binference infers its
  {term}`population fractions` $\rho$, the share of the population in each {term}`bin`,
  with their full uncertainty.
- What is each object's true value, once the population is known? binference gives
  each object's {term}`hierarchical posterior`, which is narrower than its own
  measurement wherever the population is informative.

## The problem in pictures

The figure follows 2000 simulated objects through binference. Their true values $z$
come from a population with two peaks on $[0, 3]$. Each object is measured with a
Gaussian error between 0.05 and 0.2, and the 15 bins are 0.2 wide.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  image:
    alt: >-
      Three stacked panels sharing a horizontal axis, the true value z from 0 to 3,
      with the bin edges every 0.2. Top: eight objects, each a filled coral point with
      a horizontal error bar for its measurement and an open dark circle at its true
      value; most measurements lie within one or two error bars of the truth. Middle:
      three of these objects' likelihoods over the bins, coral steps that peak in one
      or two bins near each measurement. Bottom: the inferred population density as
      forty pale indigo step curves (posterior draws) around a solid indigo step curve
      (their mean), with two peaks near z = 0.5 and z = 1.7; the true density, a dashed
      dark step curve, lies within the spread of the draws.
  figure:
    caption: >-
      From measurements to a population. (1) Eight of the 2000 objects: each measured
      value with its error (filled), and its true value (open). (2) Three of their
      likelihoods averaged over each bin, scaled to a maximum of one. (3) The
      population density inferred from all 2000 likelihoods: 40 posterior draws, their
      mean, and the true density (dashed). The spread of the draws is the uncertainty
      of the population.
---
import matplotlib.pyplot as plt
import numpy as np

import binference

plt.rcParams["font.size"] = 14
CORAL = "#EC5F3E"
INK = plt.rcParams["text.color"]

bins = binference.Bins.linear(0.0, 3.0, 15)
edges = bins.edges[0]
mock = binference.simulate.mock_1d(2000, bins, rng=42)
lk = mock.likelihoods
result = binference.infer(lk, rng=1)


def steps(ax, y, **kw):
    ax.step(edges, np.append(y, y[-1]), where="post", **kw)


z_obs, sigma = mock.measurements["mean"], mock.measurements["sigma"]
pick = np.random.default_rng(3).choice(lk.N, 8, replace=False)
pick = pick[np.argsort(z_obs[pick])]

fig, axes = plt.subplots(3, 1, figsize=(5.2, 8.4), sharex=True, constrained_layout=True)
for ax in axes[:2]:
    for e in edges:
        ax.axvline(e, color=plt.rcParams["grid.color"], lw=0.8, zorder=0)

ax = axes[0]
y = np.arange(pick.size)
ax.errorbar(z_obs[pick], y, xerr=sigma[pick], fmt="o", color=CORAL, ms=6, lw=1.5, label="measured")
ax.plot(mock.truth.values[pick], y, "o", mfc="none", mec=INK, ms=8, mew=1.5, label="true")
ax.set(yticks=[], ylabel="objects", ylim=(-0.8, pick.size + 1.0))
ax.spines["left"].set_visible(False)
ax.legend(loc="upper left", ncols=2, fontsize=12, handletextpad=0.3, columnspacing=1.0)
ax.set_title("1. Each object is measured with noise", loc="left", fontsize=14)

ax = axes[1]
L = lk.toarray()
for i in pick[[0, 4, 7]]:
    steps(ax, L[i], color=CORAL, lw=1.6)
    ax.fill_between(edges, np.append(L[i], L[i, -1]), step="post", color=CORAL, alpha=0.15, lw=0)
ax.set(yticks=[0, 1], ylabel="likelihood", ylim=(0, 1.1))
ax.set_title("2. Each measurement gives a likelihood", loc="left", fontsize=14)

ax = axes[2]
dens = result.density()
for d in dens[np.random.default_rng(0).choice(len(dens), 40, replace=False)]:
    steps(ax, d, color="C0", lw=0.8, alpha=0.2)
ax.plot([], [], color="C0", lw=0.8, alpha=0.5, label="posterior draws")
steps(ax, dens.mean(axis=0), color="C0", lw=2.2, label="posterior mean")
steps(ax, mock.truth.rho / bins.volumes, color=INK, ls="--", lw=1.6, label="truth")
ax.set(xlim=(0, 3), xlabel="true value z", ylabel="density")
ax.set_ylim(0, None)
ax.legend(loc="upper right", fontsize=12)
ax.set_title("3. Together they give the population", loc="left", fontsize=14)
plt.show()
```

The first panel shows why the problem is hard. Some measurements lie two error bars
from the truth, and a histogram of the measured values would be broader than the
population. The second panel shows what binference works with: each object's
{term}`bin-averaged likelihood`, the probability of its measurement if its true value
lay in each bin. The third panel shows the answer. The population is a set of
posterior draws, not one curve, and their spread is its uncertainty.

binference also returns each object's posterior under the inferred population, the
histogram of the objects actually observed, and any quantity computed from the
population, each with its uncertainty. The {doc}`quickstart` shows these in turn.

## One model under many names

The model is not new. Several fields arrived at it independently and named its pieces
differently. The table maps each field's names onto binference's; the {doc}`Fields
<../fields/index>` pages describe each field's methods and software.

| field | objects | per-object input | the population | methods |
|---|---|---|---|---|
| **binference** | objects | likelihoods over bins, or posteriors with their {term}`interim prior` | population fractions $\rho$ | posterior draws; a {term}`point estimate` |
| {doc}`astronomy <../fields/astronomy>` | galaxies, stars | photo-$z$ likelihoods or $p(z)$ | redshift distribution $n(z)$ | {term}`stacking`; hierarchical inference |
| {doc}`gravitational waves <../fields/gravitational-waves>` | merger events | posterior samples under the PE prior | mass distribution and merger rate | hierarchical inference with injections |
| {doc}`particle physics <../fields/unfolding>` | events | measured bin and response matrix | true spectrum | unfolding, iterative Bayesian unfolding |
| {doc}`statistics <../fields/empirical-bayes>` | units with noisy estimates | likelihoods on a grid | prior $g$, or mixing distribution | empirical Bayes deconvolution, NPMLE |
| {doc}`machine learning <../fields/quantification>` | items | classifier probabilities or scores | class prevalence | quantification under label shift |
| {doc}`genetics <../fields/quantification>` | individuals | genotype likelihoods | mixture proportions | mixed-stock analysis |

The point estimators of these fields are one iteration, which
[the theory](../how-it-works/theory.md#point-estimates-em-is-iterated-stacking)
derives. Stacking is its first step from the interim prior. Iterative Bayesian
unfolding stops it after a few steps, and the NPMLE runs it to convergence. binference
finds the converged point and, by default, samples the posterior of $\rho$ around it.

## Next

- {doc}`is-it-right` matches your data to binference's inputs, and lists what binference
  does not do.
- {doc}`principles` lists what binference promises about its answers.
- {doc}`install`, then the {doc}`quickstart`.
