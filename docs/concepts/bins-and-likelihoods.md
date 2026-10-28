---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Bins and likelihoods

*What does binference need to know about each object, and why does it work with bins?*

```{code-cell} python
:tags: [remove-cell]

import matplotlib.pyplot as plt
import numpy as np

import binference

plt.rcParams["font.size"] = 14
CORAL, INK, INK2 = "#EC5F3E", "#1B1D4A", "#5D5F86"
```

## Objects and bins

You have $N$ {term}`objects <object>`: galaxies, events, patients or documents. Each
object $i$ has data $d_i$, such as a measured flux or a classifier's score. Each object
also has a true value $z$ that you never see exactly, such as a redshift or a class.

binference divides the range of $z$ into $K$ {term}`bins <bin>` $B_1,\dots,B_K$. A bin
can be an interval, a cell of a grid, or a category. Its {term}`bin volume` $v_k$ is its
width in one dimension, its area on a grid, and one for a category.

## One object's likelihood

The {term}`likelihood function` $\ell_i(z)=p(d_i\mid z)$ says how probable object $i$'s
data are if its true value is $z$. It is not a distribution over $z$, and its overall
scale does not matter: only its shape across $z$ carries information.

The model uses one number per object and bin, the {term}`bin-averaged likelihood`.
On linear bins it is

$$
L_{ik}=\frac{1}{v_k}\int_{B_k}\ell_i(z)\,dz ,
$$

the average of $\ell_i$ over bin $k$ (the measures section below covers other bins). The $N\times K$ table of these numbers is the
{term}`likelihood matrix`. {ref}`The bin-average figure <fig-bin-average>` shows one object's
likelihood and its bin averages for a precise and a noisy measurement.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-bin-average
    caption: |
      One object measured at $d_i=1.37$ with a Gaussian error $\sigma$, on 12 bins.
      The dashed curve is the likelihood $\ell_i(z)$. The steps are its bin averages
      $L_{ik}$, computed by binference. With $\sigma=0.1$ almost all of the likelihood
      lies in three bins; with $\sigma=0.4$ it spreads over eight.
  image:
    alt: |
      Two stacked panels. In the top panel a narrow dashed bell curve peaks at z of
      about 1.4. Solid steps put almost all of its height into the bin from 1.25 to
      1.5. In the bottom panel a broad bell curve is followed closely by steps that
      rise and fall over about eight bins.
---
bins = binference.Bins.linear(0, 3, 12)
z = np.linspace(0, 3, 601)
fig, axes = plt.subplots(2, 1, layout="constrained", figsize=(4.8, 4.5), sharex=True)
for ax, sigma in zip(axes, (0.1, 0.4), strict=True):
    lk = binference.Likelihoods.from_gaussian([1.37], [sigma], bins)
    L = lk.toarray()[0] * np.exp(lk.log_scale[0])  # the bin averages at their true scale
    ell = np.exp(-0.5 * ((z - 1.37) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax.plot(z, ell, color=CORAL, lw=1.2, ls="--", label=r"likelihood $\ell_i(z)$")
    ax.stairs(L, bins.edges[0], color=CORAL, lw=2.2, label=r"bin averages $L_{ik}$")
    ax.stairs(L, bins.edges[0], color=CORAL, alpha=0.15, fill=True)
    for e in bins.edges[0]:
        ax.axvline(e, color=INK2, lw=0.5, alpha=0.4)
    ax.set_ylabel("likelihood")
    ax.text(0.98, 0.9, rf"$\sigma = {sigma}$", transform=ax.transAxes, ha="right", va="top")
axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0, fontsize=12, ncol=2)
axes[1].set_xlabel("true value $z$")
plt.show()
```

When the likelihood is much narrower than the bins, each object places nearly all its
likelihood in one bin.
When it is broader, it spreads over several bins, and the data cannot say which of
them the object lies in. The [population page](population-and-objects.md) shows what
that does to the fit.

## Why bins

The {term}`population` is a histogram: a fraction $\rho_k$ of it lies in bin $k$, spread
evenly within the bin. Under that assumption, the probability of object $i$'s data is a
dot product,

$$
p(d_i\mid\rho)=\sum_k\rho_k L_{ik}=L_i\cdot\rho .
$$

Every input type reduces to such a table. Gaussian measurements, posteriors, posterior
samples, counts in measured bins and classifier outputs all become rows $L_i$. One
model then handles all of them, and the cost of a fit grows with $N\times K$ whatever
the input was.

## Masses versus densities

$L_{ik}$ is an average over the bin, a value per unit volume. Inputs often come as
values integrated over each bin instead, such as differences of a cumulative
distribution or histograms of samples. The first kind is a {term}`density`, the second
a {term}`mass`, and a mass equals the density times the bin volume.

With equal bins the two differ by one constant, which cancels. With unequal bins they
have different shapes. If you pass masses as densities, the peak moves toward the
widest bins ({ref}`the mass figure <fig-mass-density>`). Say which you pass, as the
[inputs guide](../guides/inputs/tables.md#conventions) shows.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-mass-density
    caption: |
      One object's likelihood, Gaussian with mean 1 and standard deviation 0.35. It is
      shown as bin averages (density, solid) and bin integrals (mass, dotted), each
      scaled to a peak of one. Top: unequal bins on a linear axis. Bottom: ten log-spaced bins
      from 0.1 to 10 on a logarithmic axis. The density follows the likelihood; the
      mass peaks in a wide bin.
  image:
    alt: |
      Two stacked panels of steps. In the top panel the solid density steps peak in
      the two narrow bins from 0.9 to 1.1. The dotted mass steps peak in the wider bin from 1.1 to 1.6.
      In the bottom panel the density peaks in the bin just below 1, and the mass in
      the wider bin from 1.0 to 1.585.
---
unequal = binference.Bins([0, 0.2, 0.4, 0.5, 0.9, 1.0, 1.1, 1.6, 3.0])
log_spaced = binference.Bins(np.geomspace(0.1, 10, 11), measure="linear")
fig, axes = plt.subplots(2, 1, layout="constrained", figsize=(4.8, 4.6))
for ax, b in zip(axes, (unequal, log_spaced), strict=True):
    dens = binference.Likelihoods.from_gaussian([1.0], [0.35], b).toarray()[0]
    mass = dens * b.volumes
    ax.stairs(dens / dens.max(), b.edges[0], color=CORAL, lw=2.2, label="density")
    ax.stairs(mass / mass.max(), b.edges[0], color=INK, lw=1.6, ls=":", label="mass")
    ax.set_ylabel("scaled to peak 1")
axes[1].set_xscale("log")
axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0, fontsize=12, ncol=2)
axes[1].set_xlabel("true value $z$")
plt.show()
```

## Measures: flat in which coordinate?

The model takes the population as flat within each bin. Flat needs a coordinate, and
binference calls it the bin axis's {term}`measure`, $u=T(z)$. Linear bins use $u=z$.
Logarithmic bins use $u=\log z$, so the population is flat in $\log z$ within each bin.
A logarithmic bin's centre is then the geometric mean of its edges.

With a measure other than $u=z$, binference averages $\ell_i$ over each bin uniformly
in $u$, and the population is flat in $u$, not in $z$. The measure also
decides what a {term}`smoothness prior <logistic prior>` calls flat (see
[priors on the simplex](priors-on-the-simplex.md)). Choose the coordinate in which the
population changes least across a bin.

When the population is not flat within the bins, the resulting {term}`binning bias`
does not shrink as you add objects; the inputs guide shows how a
[within-bin basis](../guides/inputs/advanced.md#shape-within-bins-basis)
reduces it.

```{plain-words}
binference asks one question of each object: how well does each bin explain its data?
The answers, one per bin, are all it needs.
```

## Where next

- Do it: the {doc}`inputs guide <../guides/inputs/index>`.
- See it: the tutorial {doc}`../tutorials/posteriors-on-a-grid`.
- Derivation: [objects, bins and likelihoods](../how-it-works/model.md#objects-bins-and-likelihoods)
  in How it works.
