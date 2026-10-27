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

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import binference

plt.rcParams["font.size"] = 14
CORAL, INK, INK2 = "#EC5F3E", "#1B1D4A", "#5D5F86"
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

The same likelihood gives different posteriors under different interim priors ({ref}`figure <fig-interim>`). A broad interim prior barely changes the likelihood. A narrow
one pulls the posterior toward its own peak.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-interim
    caption: |
      One object's likelihood (dashed), Gaussian with mean 1.9 and standard deviation
      0.3, times an interim prior (dotted) gives its posterior (solid), on 60 bins.
      Top: a broad interim prior, normal with mean 1.5 and standard deviation 0.9.
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
fig, axes = plt.subplots(2, 1, layout="constrained", figsize=(4.8, 4.5), sharex=True)
for ax, (name, pi) in zip(axes, INTERIMS.items(), strict=True):
    Pi = binference.Density.coerce(pi).masses(fine)
    ax.stairs(unit(L * fine.volumes), fine.edges[0], color=CORAL, lw=1.4, ls="--",
              label="likelihood")
    ax.stairs(unit(Pi), fine.edges[0], color=INK2, lw=1.4, ls=":", label="interim prior")
    ax.stairs(unit(L * Pi), fine.edges[0], color=CORAL, lw=2.4, label="posterior")
    ax.set_ylabel("density")
    ax.set_title(name, fontsize=14, loc="left")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3,
           fontsize=12, handlelength=1.5, columnspacing=1.0)
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
itself. An extra factor rewards populations that put their mass where $\pi$ is large.
The fit drifts toward the interim prior, and more so the narrower it is ({ref}`figure <fig-interim-drift>`). With uninformative data, it collapses onto the interim
prior's peak.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-interim-drift
    caption: |
      Population fits to the posteriors of 2,000 mock objects on 15 bins, with
      measurement errors between 0.1 and 0.25, computed under the two interim priors
      above. Indigo: the interim prior divided out. Amber: the posteriors used as
      likelihoods. Each shows its posterior mean and 68% band, from Gibbs sampling
      with 4,000 draws. Dashed: the true population. Dotted: the interim prior.
  image:
    alt: |
      Two stacked panels of population densities. In both, the indigo band follows
      the dashed truth, which has two humps. In the top panel the amber band is close
      to the truth. In the bottom panel the amber band piles up between 0.6 and 1.4,
      under the narrow dotted interim prior, and drops below the truth elsewhere.
---
bins = binference.Bins.linear(0, 3, 15)
truth = binference.simulate.population(bins, "bimodal")
fig, axes = plt.subplots(2, 1, layout="constrained", figsize=(4.8, 4.8), sharex=True)
for ax, (name, pi) in zip(axes, INTERIMS.items(), strict=True):
    mock = binference.simulate.mock_1d(2000, bins, density=truth, noise=(0.1, 0.25),
                                       interim=pi, rng=3)
    fits = {}
    for label, how in (("interim prior divided out", {"interim": mock.interim}),
                       ("posteriors as likelihoods", {"kind": "likelihood"})):
        lk = binference.Likelihoods.from_table(mock.posteriors, bins, **how)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fits[label] = binference.infer(lk, method="gibbs", n_samples=1000, rng=1, check=False)
    for (label, fit), color in zip(fits.items(), ("C0", "C1"), strict=True):
        fit.plot(ax=ax, levels=(0.68,), color=color, label=label)
    ax.stairs(truth / bins.volumes, bins.edges[0], color=INK, lw=1.4, ls="--", label="truth")
    ax.stairs(mock.interim / bins.volumes, bins.edges[0], color=INK2, lw=1.4, ls=":",
              label="interim prior")
    ax.get_legend().remove()
    ax.set_xlabel("")
    ax.set_ylabel("density")
    ax.set_title(name, fontsize=14, loc="left")
handles, labels = axes[0].get_legend_handles_labels()
keep = [i for i, t in enumerate(labels) if "interval" not in t]
fig.legend([handles[i] for i in keep], [labels[i] for i in keep], fontsize=12, ncol=2,
           loc="outside upper center")
for ax in axes:
    ax.set_ylim(0, 1.3)
axes[1].set_xlabel("true value $z$")
plt.show()
```

## Stacking is one step from the interim prior

Stacking averages the objects' posteriors and calls the average the population. Each
posterior is the object's posterior over bins under the interim prior: its
{term}`responsibilities` $r_{ik}(\pi)$. The stack is therefore

$$
\frac1W\sum_iw_i\,r_{ik}(\pi),
$$

with $w_i$ the {term}`weights` and $W$ their sum. This is exactly one step of
{term}`EM` started from the interim prior. EM repeats the step: it recomputes every
object's posterior under the current estimate and averages again, until nothing
changes.

One step is not enough when likelihoods are broad. The stack is then the population
blurred by the measurement errors, and pulled toward $\pi$. The
[population page](population-and-objects.md#stacking-and-the-hierarchical-fit) shows the
steps that follow.

```{plain-words}
A posterior is a likelihood with someone else's guess about the population already
mixed in. Take that guess out before you ask what the population is.
```

## Where next

- The inputs guide converts
  [posteriors](../guides/inputs.md#posteriors-computed-under-an-interim-prior) and
  [posterior samples](../guides/inputs.md#posterior-samples) with their interim prior.
- The tutorial {doc}`../tutorials/02_inputs` repeats {ref}`figure <fig-interim-drift>` with a photometric-redshift mock.
  {doc}`../tutorials/06_categories_unfolding` divides a classifier's training class
  mix out of its probabilities.
- The pitfalls guide describes
  [posteriors used as likelihoods](../guides/pitfalls.md#feeding-posteriors-in-as-likelihoods)
  and [stacking](../guides/pitfalls.md#stacking).
- The theory page derives the
  [bias](../how-it-works/theory.md#likelihoods-versus-posteriors) and shows that
  [EM is iterated stacking](../how-it-works/theory.md#point-estimates-em-is-iterated-stacking).
- Next concept: {doc}`population-and-objects`.
