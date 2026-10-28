---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Priors on the simplex

*What does a prior on the population fractions say, and when does it change the answer?*

```{code-cell} python
:tags: [remove-cell]

import matplotlib.pyplot as plt
import numpy as np

import binference
from binference import priors

INK2 = plt.rcParams["xtick.color"]
```

## Every population is a point on the simplex

The {term}`population fractions` $\rho_1,\dots,\rho_K$ are non-negative and sum to one.
The set of all such vectors is the {term}`simplex` $\Delta^{K-1}$. With three bins it is
a triangle: each corner puts the whole population in one bin, and the centre splits it
equally. A prior on the population is a distribution over this triangle, or over its
higher-dimensional version.

## The Dirichlet prior and pseudo-counts

The default prior is the {term}`Dirichlet prior` $\mathrm{Dir}(\alpha)$, with density
proportional to $\prod_k\rho_k^{\alpha_k-1}$. Its parameters $\alpha_k$ are the
{term}`concentrations <concentration>`.

The concentrations act like objects seen before the data, or pseudo-counts. Suppose
you knew the bin of every object, with $n_k$ objects in bin $k$. The posterior would
then be $\mathrm{Dir}(\alpha+n)$, so $\alpha_k$ adds to the count of bin $k$. A prior
with this property is {term}`conjugate <conjugate prior>`. Gibbs and mean-field
VB need it, and so does EM, one of MAP's solvers.

The size of $\alpha$ sets how strongly the prior moves the estimate
({ref}`figure <fig-dirichlet>`):

- With every $\alpha_k=1$, the default, every population is equally probable. The
  prior is flat on the simplex.
- With large $\alpha_k$, draws crowd around the prior mean, $\alpha_k/\sum_j\alpha_j$.
- With $\alpha_k<1$, draws crowd into the corners and edges: most bins are nearly
  empty.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-dirichlet
    caption: |
      1,500 draws each from symmetric Dirichlet priors on three bins, drawn by
      binference. Every concentration $\alpha_k$ equals the $\alpha$ shown. Each point
      is one population; each corner puts the whole population in one bin.
  image:
    alt: |
      Four triangles. For alpha 0.2 the points sit on the corners and edges. For
      alpha 1 they fill the triangle evenly. For alpha 5 they form a cloud around
      the centre, and for alpha 30 a tight cluster at the centre.
---
no_data = binference.Likelihoods(np.ones((1, 3)))  # one uninformative object
corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
names = (r"$\rho_1=1$", r"$\rho_2=1$", r"$\rho_3=1$")
fig, axes = plt.subplots(2, 2)
fig.set_figheight(5.0)
for ax, a in zip(axes.flat, (0.2, 1.0, 5.0, 30.0), strict=True):
    rho = binference.PopulationModel(no_data, priors.Dirichlet(a)).sample_prior(1500, rng=7)
    ax.plot(*np.vstack([corners, corners[:1]]).T, color=INK2, lw=1)
    ax.plot(*(rho @ corners).T, ".", ms=2.5, color="C0", alpha=0.5)
    for (x, y), name, ha, dy in zip(corners, names, ("right", "left", "center"),
                                    (-0.04, -0.04, 0.04), strict=True):
        ax.text(x, y + dy, name, ha=ha, va="top" if dy < 0 else "bottom")
    ax.set_title(rf"$\alpha = {a:g}$", pad=18)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, 0.98)
plt.show()
```

The flat prior is not free. It adds one pseudo-count per bin, $K$ in all. With $K$
comparable to the number of objects, that moves the estimate toward equal fractions.
The {doc}`priors guide <../guides/priors>` gives the alternatives: the
{term}`reference prior`, concentrations from a base density or a reference sample, and
three conjugate relatives of the Dirichlet.

## Smoothness priors

A Dirichlet treats bins as unordered: shuffling them leaves the prior unchanged. With
broad likelihoods, mass can shift between neighbouring bins with little change in the
likelihood, and the posterior mean can be jagged.

A {term}`smoothness prior <logistic prior>` adds what the Dirichlet leaves out. It puts
a Gaussian distribution on the log-fractions $\eta$, with $\rho=\mathrm{softmax}(\eta)$,
and makes neighbouring values of $\eta$ similar:

- a random walk penalises jumps, or bends, between neighbouring bins;
- a Gaussian process correlates bins closer than a length scale.

How smooth the population is, is not fixed in advance. binference infers the prior's
{term}`hyperparameters <hyperparameter>`, such as the size of the jumps or the length
scale, with the population. By default the prior is centred on a density
that is flat in the {term}`measure` coordinate. Its notion of smooth then does not
depend on how you bin. Smoothness priors are not conjugate, so Gibbs and VB cannot use
them (see {doc}`engines`).

## When the prior matters

The prior matters where the data are weak. A bin's {term}`data support` is the number
of objects' worth of likelihood it holds. Where it is low, the prior sets the answer.
With many objects in every bin, any of these priors gives nearly the same population.

To choose between priors that the data cannot tell apart, compare them on held-out
objects, with the {term}`log predictive density`. For counts per unit time or area
instead of fractions, a prior on the {term}`rates` takes the place of the Dirichlet
(see {doc}`selection`).

```{plain-words}
A prior says which populations you find plausible before you look at the data. The
Dirichlet's numbers behave like objects you pretend to have already seen; a smoothness
prior says that neighbouring bins should look alike.
```

## Where next

- Do it: the {doc}`priors guide <../guides/priors>`.
- See it: the tutorial {doc}`../tutorials/priors-and-smoothness`.
- Derivation: [the Dirichlet family](../how-it-works/priors.md#the-dirichlet-family-as-trees) in How it works.
