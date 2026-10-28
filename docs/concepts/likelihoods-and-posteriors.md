---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Likelihoods and posteriors

*Many codes report each object's posterior instead of its likelihood. What is the
difference, and why does it matter for the population?*

```{code-cell} python
:tags: [remove-cell]

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import binference
from binference.plotting import SAMPLE_COLOR

CORAL, INK2 = SAMPLE_COLOR, plt.rcParams["xtick.color"]
INTERIMS = {"broad interim prior": stats.norm(1.5, 0.9), "narrow interim prior": stats.norm(0.9, 0.3)}
```

## A posterior contains a prior

A photometric-redshift code, a classifier or a per-object fitter usually returns a
posterior for each object. It multiplies the object's {term}`likelihood function` by a
prior and normalises:

$$
p_i(z)\propto\ell_i(z)\,\pi(z).
$$

The prior $\pi$ is the {term}`interim prior`. The code chose it before anyone knew the
population. For a classifier, it is the class mix of the training set.

The same likelihood gives different posteriors under different interim priors
({ref}`figure <fig-interim>`). A broad interim prior barely changes the likelihood. A
narrow one moves the posterior toward its own peak.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-interim
    caption: |
      One object's likelihood (dashed) times an interim prior (dotted) gives its
      posterior (solid), on 60 bins. The likelihood is Gaussian with mean 1.9 and
      standard deviation 0.3. Top: a broad interim prior, normal with mean 1.5 and standard deviation 0.9.
      Bottom: a narrow one, normal with mean 0.9 and standard deviation 0.3. Each
      curve has unit area.
  image:
    alt: |
      Two stacked panels. In the top panel the posterior almost coincides with the
      likelihood, which peaks near 1.9, while the broad dotted prior is low and flat.
      In the bottom panel the dotted prior peaks at 0.9 and the posterior sits between
      it and the likelihood, peaking near 1.4.
---
fine = binference.Bins.linear(0, 3, 60)
L = binference.Likelihoods.from_gaussian([1.9], [0.3], fine).toarray()[0]
unit = lambda m: m / (m.sum() * fine.volumes)  # noqa: E731  (masses to unit-area densities)
fig, axes = plt.subplots(2, 1, sharex=True)
fig.set_figheight(4.5)
for ax, (name, pi) in zip(axes, INTERIMS.items(), strict=True):
    Pi = binference.Density.coerce(pi).masses(fine)
    ax.stairs(unit(L * fine.volumes), fine.edges[0], color=CORAL, lw=1.4, ls="--",
              label="likelihood")
    ax.stairs(unit(Pi), fine.edges[0], color=INK2, lw=1.4, ls=":", label="interim prior")
    ax.stairs(unit(L * Pi), fine.edges[0], color=CORAL, lw=2.4, label="posterior")
    ax.set_ylabel("density")
    ax.set_title(name, loc="left")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
axes[1].set_xlabel("true value $z$")
plt.show()
```

## Dividing the interim prior out

The population model needs likelihoods. If $P_{ik}$ is the mass of object $i$'s
posterior in bin $k$, and $\Pi_k$ the mass of the interim prior there, then

$$
L_{ik}\propto\frac{P_{ik}}{\Pi_k},
$$

provided $\pi$ is roughly constant within each bin. binference does this division
for you. It has no default interim prior for posteriors or {term}`posterior samples`,
so you must name the one that was used.

## Posteriors used as likelihoods

Treating posteriors as likelihoods counts the interim prior once per object. The data
then constrain the product of $\pi$ and the population rather than the population
itself. The fit then favours populations that are large where $\pi$ is large. It moves
toward the interim prior, more so the narrower $\pi$ is
(explainer V-3). With uninformative data it equals the interim
prior's peak.

```{explainer} v3
Population fits to 600 mock objects on 15 bins, measured with error 0.3, whose posteriors
use a Gaussian interim prior of width 0.4 (grey). Indigo: EM with the interim prior divided
out. Amber: EM on the posteriors used as likelihoods. Dashed: the true population. Move the
interim prior: the amber fit follows it, the indigo fit does not.
```

## Stacking is one step from the interim prior

{term}`Stacking` averages the objects' posteriors and calls the average the
population. The stack is one step of {term}`EM` started from the interim prior, so it
leans toward $\pi$; [the EM figure](population-and-objects.md#stacking-and-the-hierarchical-fit)
shows the steps that follow.

```{plain-words}
A posterior is a likelihood with someone else's guess about the population already
mixed in. Take that guess out before you ask what the population is.
```

## Where next

- Do it: the inputs guide converts
  [posteriors](../guides/inputs/posteriors.md) with
  their interim prior.
- See it: the tutorial {doc}`../tutorials/posteriors-on-a-grid` repeats
  the comparison of explainer V-3 with a photometric-redshift mock.
- Derivation: [likelihoods versus posteriors](../how-it-works/model.md#likelihoods-versus-posteriors)
  in How it works.
