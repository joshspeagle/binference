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
({ref}`the modes figure <fig-selection-modes>`):

- **On the truth.** The chance of being kept depends on the true value $z$ alone, not on
  the measurement noise. A selection that depends on true planet radius is an example.
  binference then multiplies each object's likelihood by $S_k$.
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
      300 mock objects with true values $z$. Their measured values $d$ have Gaussian
      errors of 0.3. Top: each object is kept with a probability that falls from one
      to zero around $z=1.5$. Bottom: objects with $d$ above 1.6 (dashed line) are
      missed. Filled: selected. Open: missed.
  image:
    alt: |
      Two scatter plots of measured value against true value. The points lie along a
      rising diagonal band. In the top panel kept and missed points mix along the
      band, and missed points dominate above a true value of about 1.5. In the bottom
      panel every point below a horizontal dashed line at 1.6 is kept. Every point
      above it is missed, so the boundary is horizontal.
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

the area under the selected part of the population ({ref}`the normaliser figure <fig-normaliser>`).
binference divides each object's term in the likelihood by this selected share.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-normaliser
    caption: |
      Top: the selection $S_k$ of the two examples above, on 15 bins, computed by
      binference. One comes from a smooth selection on the truth, the other from the
      cut at $d=1.6$. Bottom: the population (outline) and its part selected by the cut
      (shaded). The shaded area is $S\cdot\rho$.
  image:
    alt: |
      Top panel: two step curves near one at low true values fall to near zero above
      about 2. The dashed curve, for the cut on the measurement, lies to the right and falls more
      steeply.
      Bottom panel: a single-humped population outline centred at 1.5. The shaded
      selected part matches it at low values and falls away above 1.5.
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

The pitfalls guide describes the
[two common mistakes](../guides/troubleshooting.md#ignoring-selection-or-modelling-it-twice)
with the normaliser.

For the fractions, only the ratios of the $S_k$ matter. A selection known only up to a
constant is enough.

## Rates and totals

The normaliser also answers a second question: how many objects were there in all?
Suppose $W$ objects are selected, a share $S\cdot\rho$ of the population. Then the
population held about $W/(S\cdot\rho)$ objects. For every draw of $\rho$, binference
draws the {term}`total` $\Lambda$: the expected number of objects, including those the
selection missed. The
{term}`rates` $\lambda_k=\Lambda\rho_k$ are the expected numbers per bin.

The total needs absolute values of $S_k$. An {term}`exposure`, such as
observing time times area, can take the place of a probability, and then the rates
are numbers per unit exposure. A prior on the rates themselves replaces the prior on
the fractions when you have prior knowledge of the counts.

```{plain-words}
If some objects are easier to find, your catalogue has too many of them. Tell
binference how likely each kind of object was to be found, and whether the cut was on
the true value or on the measurement.
```

## Where next

- Do it: the {doc}`selection guide <../guides/selection>`, including selections that
  vary within bins and selections measured by injections.
- See it: the tutorial {doc}`../tutorials/selection-effects`.
- Derivation: [selection effects](../how-it-works/model.md#selection-effects) in How it works.
