---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Selection

*Your catalogue is missing objects. When does that bias the population, and what does
binference need to correct it?*

```{code-cell} python
:tags: [remove-cell]

import matplotlib.pyplot as plt
import numpy as np

import binference

plt.rcParams["font.size"] = 14
CORAL, INK, INK2 = "#EC5F3E", "#1B1D4A", "#5D5F86"
CUT = 1.6  # objects with a measured value above this are missed
completeness = lambda z: 1 / (1 + np.exp((np.asarray(z) - 1.5) / 0.2))  # noqa: E731
```

## Which objects are missing

Few catalogues contain every object. A survey misses faint galaxies, a detector misses
quiet events, a screening misses mild cases. The {term}`selection` $S_k$ is the
probability that an object whose true value lies in bin $k$ enters the catalogue.

If $S_k$ is the same in every bin, the catalogue is a fair sample and the fractions
need no correction. If it varies, the catalogue over-represents the bins that are easy
to observe. Ignoring that biases $\rho$ toward them.

## On the truth or on the measurement

What matters is what the selection acts on, its {term}`selection mode`
({ref}`figure <fig-selection-modes>`):

- **On the truth.** The chance of being kept depends on the true value $z$ alone, not on
  the measurement noise. A completeness that depends on true planet radius is an
  example. Each object's likelihood is then multiplied by $S_k$.
- **On the measurement.** The cut acts on the measured data $d$, such as a limit on
  measured flux. An object near the cut is kept or lost depending on its noise. The
  likelihood of a kept object is used as it is. Here $S_k$ is the chance that an object
  in bin $k$ would pass the cut, averaged over its possible data.

The two modes give different likelihoods for the same $S_k$, so binference asks you to
state the mode with every selection. A selection can also act on both at once.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-selection-modes
    caption: |
      300 mock objects, with true values $z$ and measured values $d$ with Gaussian
      errors of 0.3. Top: each object is kept with a probability that falls from one
      to zero around $z=1.5$. Bottom: objects with $d$ above 1.6 (dashed line) are
      missed. Filled: selected. Open: missed.
  image:
    alt: |
      Two scatter plots of measured value against true value, with the points along a
      rising diagonal band. In the top panel kept and missed points mix along the
      band, with missed points dominating above a true value of about 1.5. In the
      bottom panel every point below a horizontal dashed line at 1.6 is kept and
      every point above it is missed, so the boundary is horizontal.
---
bins = binference.Bins.linear(0, 3, 15)
mock = binference.simulate.mock_1d(300, bins, density="gaussian", noise=0.3, rng=5)
z, d = mock.truth.values, mock.measurements["mean"]
rng = np.random.default_rng(5)
kept = {"selection on the truth": rng.random(z.size) < completeness(z),
        "selection on the measurement": d < CUT}
fig, axes = plt.subplots(2, 1, layout="constrained", figsize=(4.8, 5.1), sharex=True)
for ax, (title, keep) in zip(axes, kept.items(), strict=True):
    ax.plot(z[keep], d[keep], "o", ms=4, color=CORAL, label="selected")
    ax.plot(z[~keep], d[~keep], "o", ms=4, mfc="none", mec=INK2, mew=0.8, label="missed")
    ax.set_ylabel("measured value $d$")
    ax.set_title(title, fontsize=14, loc="left")
axes[1].axhline(CUT, color=INK, lw=1.2, ls="--")
axes[0].legend(fontsize=12, loc="upper left")
axes[1].set_xlabel("true value $z$")
plt.show()
```

## The normaliser

A selected object tells you about the population of objects that could be selected,
not about the whole population. The share of the population that is selected is

$$
S\cdot\rho=\sum_kS_k\rho_k ,
$$

the area under the selected part of the population ({ref}`figure <fig-normaliser>`).
binference divides each object's term by this share. The posterior becomes

$$
p(\rho\mid D)\propto p(\rho)\,\frac{\prod_i(\tilde L_i\cdot\rho)^{w_i}}{(S\cdot\rho)^W},
$$

where $\tilde L_{ik}$ is $S_kL_{ik}$ on the truth and $L_{ik}$ on the measurement, and
$W$ is the total {term}`weight <weights>` of the objects.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-normaliser
    caption: |
      Top: the selection $S_k$ of the two examples above, on 15 bins, computed by
      binference: from the completeness for selection on the truth, and from the cut
      at $d=1.6$ for selection on the measurement. Bottom: the population (outline)
      and its part selected by the cut (shaded). The shaded area is $S\cdot\rho$.
  image:
    alt: |
      Top panel: two step curves near one at low true values that fall to near zero
      above about 2; the dashed curve, for the cut on the measurement, falls more
      gradually. Bottom panel: a single-humped population outline centred at 1.5;
      the shaded selected part matches it at low values and falls away above 1.5.
---
mock = binference.simulate.mock_1d(200, bins, density="gaussian", noise=0.3,
                                   measured_range=(None, CUT), rng=5)
S = {"on the truth": binference.Selection.from_function(completeness, bins, "on_truth").values,
     "on the measurement": mock.likelihoods.selection.values}
rho = binference.simulate.population(bins, "gaussian")
area = S["on the measurement"] @ rho
fig, axes = plt.subplots(2, 1, layout="constrained", figsize=(4.8, 4.6), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1.3]})
ax = axes[0]
for (mode, s), ls in zip(S.items(), ("-", "--"), strict=True):
    ax.stairs(s, bins.edges[0], color=INK, lw=1.8, ls=ls, label=mode)
ax.set_ylabel("selection $S_k$")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0, fontsize=12, ncol=1)
ax = axes[1]
selected = S["on the measurement"] * rho / bins.volumes
ax.stairs(selected, bins.edges[0], color="C0", fill=True, alpha=0.3,
          label=rf"selected: area $S\cdot\rho={area:.2f}$")
ax.stairs(rho / bins.volumes, bins.edges[0], color="C0", lw=2, label="population")
ax.set_ylabel("density")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0, fontsize=12, ncol=1)
axes[1].set_xlabel("true value $z$")
plt.show()
```

Two mistakes are common, and both bias the fractions toward well-observed bins. One
divides by $S\cdot\rho$ and also renormalises each object's likelihood over the data
that could be detected, which corrects twice. The other leaves the normaliser out,
which corrects nothing {cite:p}`mandel2019`.

For the fractions, only the ratios between the $S_k$ matter: scaling every $S_k$ by
the same factor leaves $\rho$ unchanged. A selection known only up to a constant is
enough.

## Rates and totals

The normaliser also answers a second question: how many objects were there in all?
Suppose $W$ objects are selected, a share $S\cdot\rho$ of the population. Then the
population held about $W/(S\cdot\rho)$ objects. binference draws the {term}`total` $\Lambda$, the
expected number of objects including those it missed, for every draw of $\rho$. The
{term}`rates` $\lambda_k=\Lambda\rho_k$ are the expected numbers per bin.

The total needs the absolute $S_k$, not just their ratios. An {term}`exposure`, such as
observing time times area, can take the place of a probability, and then the rates
are numbers per unit exposure. A prior on the rates themselves replaces the prior on
the fractions when you have prior knowledge of the counts.

When $S$ changes quickly within a bin, its bin average depends on the shape of the
population there. Give binference the selection as a function of $z$ in that case. When
$S$ is measured by passing simulated {term}`injections` through the pipeline, binference
can infer it together with the population.

```{plain-words}
If some objects are easier to find, your catalogue has too many of them. Tell
binference how likely each kind of object was to be found, and whether the cut was on
the true value or on the measurement.
```

## Where next

- The {doc}`selection guide <../guides/selection>` attaches a selection in each mode,
  measures it with [injections](../guides/selection.md#selection-measured-by-injections)
  and infers [rates and totals](../guides/selection.md#rates-and-exposure).
- The tutorials {doc}`../tutorials/05_selection` and
  {doc}`../tutorials/11_populations_with_selection` correct incomplete catalogues.
- The theory page derives the [normaliser](../how-it-works/theory.md#selection-effects)
  and the distribution of the missed objects.
- Next concept: {doc}`engines`.
