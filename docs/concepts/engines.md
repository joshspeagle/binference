---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Engines

*binference offers several engines. What does each one compute, and when is its answer
exact?*

```{code-cell} python
:tags: [remove-cell]

import warnings

import matplotlib.pyplot as plt
import numpy as np
from myst_nb import glue
from scipy import stats

import binference

plt.rcParams["font.size"] = 14
INK = "#1B1D4A"
```

## One posterior, several ways to reach it

The model fixes one posterior for the population, $p(\rho\mid D)$. An {term}`engine` is
an algorithm that describes it. Some engines return {term}`draws <draw>` from it, one
returns its most probable point, and some return draws from an approximation. They
differ in what they need, how fast they are, and whether their answer is exact.

| engine | computes | exact when |
|---|---|---|
| {term}`NUTS` | draws of the posterior | always, once its chains have converged |
| {term}`Gibbs` | draws of the posterior and of every object's bin | always, for a conjugate prior and integer weights, once converged |
| {term}`MAP` | the most probable population, with a certificate | it is a point, not a posterior |
| {term}`Laplace` | a Gaussian fitted at the mode, corrected by importance sampling | its importance check passes |
| {term}`mean-field VB` | an approximation that treats the population and the objects' bins as independent | never; its intervals are too narrow |
| {term}`bootstrap` | the spread of the MAP over resampled objects | never; it is not a posterior |
| {term}`auto` | Gibbs, Laplace or NUTS, chosen for the data | as the engine it chose |

Results whose draws are not from the exact posterior are marked {term}`approximate`.

## The engines one by one

**NUTS.** The No-U-Turn Sampler moves through the posterior along simulated
trajectories guided by its gradient. It works with every prior and mixes well even
when likelihoods are broad. Its draws are exact once its {term}`chains <chain>` have
converged, which {term}`R-hat`, the {term}`effective sample size` and its
{term}`divergences <divergence>` check.

**Gibbs.** Gibbs sampling alternates two exact steps. It draws every object's bin
given the population, then the population given those bins. It needs a
{term}`conjugate prior` and integer weights, and it returns each object's bin with every
draw. When likelihoods are broad compared with the bins, the two steps hold each
other back. The {term}`missing information` $f$ measures this, and successive draws
are then strongly correlated.

**MAP.** A point estimate finds the single most probable population. With the default
flat Dirichlet prior, it is the maximum-likelihood population. Three solvers share the
work: {term}`mix-SQP`, {term}`EM` and {term}`L-BFGS`. A certificate, the
{term}`KKT gap`, says how close the answer is to the optimum. A point carries no
uncertainty, so use it for quick looks and as a starting point.

**Laplace.** The Laplace engine fits a Gaussian at the posterior's mode, in
{term}`unconstrained coordinates`. It then reweights draws from it by
{term}`importance sampling` toward the exact posterior. The {term}`Pareto k-hat` of the
weights says whether that worked. If it did, the draws are exact; if not, the result
is flagged approximate. With many bins the check often fails, because nearly empty
bins have one-sided tails.

**Mean-field VB.** Variational Bayes fits a simple distribution that treats the
population and each object's bin as independent. That independence ignores exactly
the coupling that broad likelihoods create, so its intervals are too narrow. It is
fast and always flagged approximate.

**Bootstrap.** The bootstrap refits the MAP to resampled or reweighted objects. Its
spread describes how the estimate would vary over repeated catalogues. That is a
frequentist answer, not a posterior.

**Auto.** The default engine tries Gibbs where Gibbs mixes well, then Laplace where its
check passes, and NUTS otherwise. The rules, and the benchmark behind them, are in
the {doc}`engines guide <../guides/engines>`.

## Seeing them on one problem

{ref}`Figure <fig-engines>` shows a population on three bins. The fractions
$(\rho_1,\rho_2,\rho_3)$ lie on a triangle, which unconstrained coordinates
$(x_1,x_2)$ open onto the whole plane. Every engine describes the same posterior there.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-engines
    caption: |
      One posterior of a population on three bins, from 40 mock objects with
      measurement errors between 0.4 and 0.8, in unconstrained coordinates $x$.
      Top: 1,000 draws each from NUTS and Gibbs (two chains each). Bottom: the NUTS
      draws, the 95% ellipses of the Laplace Gaussian and of mean-field VB, the mode in
      $x$ where the Laplace Gaussian is centred, and the MAP on the simplex.
  image:
    alt: |
      Two scatter plots with the same axes. In the top panel indigo dots from NUTS
      and amber crosses from Gibbs form one cloud, longer to the lower left. In the
      bottom panel the same cloud carries a teal Laplace ellipse that covers most of
      it, a smaller dashed plum VB ellipse inside it, an open diamond at the mode and
      a star for the MAP close beside it.
---
bins = binference.Bins.linear(0, 3, 3)
lk = binference.simulate.mock_1d(40, bins, density=np.array([0.5, 0.35, 0.15]),
                                 noise=(0.4, 0.8), rng=2).likelihoods
model = binference.PopulationModel(lk)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fits = {m: binference.infer(lk, method=m, rng=1, n_samples=500, n_chains=2, check=False)
            for m in ("nuts", "gibbs")}
    lap = binference.infer(lk, method="laplace", rng=1, n_samples=1000)
    vb = binference.infer(lk, method="vb", rng=1, n_samples=2000)
    est = binference.infer(lk, method="map")
to_x = lambda R: np.array([model.from_value(r) for r in R])  # noqa: E731
xn, xg, xv = to_x(fits["nuts"].samples), to_x(fits["gibbs"].samples), to_x(vb.samples)


def ellipse(ax, mean, cov, **kw):
    t = np.linspace(0, 2 * np.pi, 200)
    w, U = np.linalg.eigh(cov)
    radius = np.sqrt(stats.chi2.ppf(0.95, 2))
    ax.plot(*(mean[:, None] + radius * (U * np.sqrt(w)) @ np.vstack([np.cos(t), np.sin(t)])), **kw)


fig, axes = plt.subplots(2, 1, layout="constrained", figsize=(4.8, 6.1), sharex=True, sharey=True)
ax = axes[0]
ax.plot(*xn.T, ".", ms=3, color="C0", alpha=0.6, label="NUTS draws")
ax.plot(*xg.T, "x", ms=4, mew=0.8, color="C1", alpha=0.6, label="Gibbs draws")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0, fontsize=12, ncol=2, markerscale=2)
ax = axes[1]
ax.plot(*xn.T, ".", ms=2, color="#B8BBEE", label="NUTS")
ellipse(ax, lap.info["mode"], lap.info["cov"], color="C2", lw=2, label="Laplace, 95%")
ellipse(ax, xv.mean(axis=0), np.cov(xv.T), color="C3", lw=2, ls="--", label="VB, 95%")
ax.plot(*lap.info["mode"], "D", ms=7, mfc="none", color=INK, label="mode in $x$")
ax.plot(*model.from_value(est.rho), "*", ms=13, color=INK, label="MAP")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0, fontsize=12, ncol=3, columnspacing=0.8, handlelength=1.5)
for a in axes:
    a.set_ylabel("$x_2$")
    a.set_xlim(-4, 2.5)
    a.set_ylim(-2, 3)
axes[1].set_xlabel("$x_1$")
plt.show()
glue("k_hat", lap.info["k_hat"], display=False)
glue("k_hat_max", lap.info["k_hat_threshold"], display=False)
glue("vb_ratio", float(np.median(vb.std / fits["nuts"].std)), display=False)
glue("missing", float(model.missing_information(est.rho)), display=False)
```

NUTS and Gibbs agree: both draw the exact posterior. The Laplace Gaussian is
symmetric, while the posterior has a longer tail to one side. Here its importance
check passed, with $\hat k=$ {glue:text}`k_hat:.2f` against a limit of
{glue:text}`k_hat_max:.2f`, so its corrected draws are exact. The mean-field VB ellipse
is too small: its standard deviations of the fractions are a median
{glue:text}`vb_ratio:.2f` times those of NUTS. The missing information of this problem
is $f=$ {glue:text}`missing:.2f`.

The MAP and the mode in $x$ differ. They are the peaks of two densities of the same
posterior, one in $\rho$ and one in $x$. The change of coordinates moves the peak.
Neither is the answer: the posterior mean and its intervals are the same in any
coordinates.

```{plain-words}
Every engine looks at the same posterior. NUTS and Gibbs walk through it, MAP finds
its highest point, and Laplace and VB sketch it with a simpler shape; binference tells
you when a sketch is not exact.
```

## Where next

- The {doc}`engines guide <../guides/engines>` sets each engine's options, explains how
  `auto` chooses, and gives the benchmark behind the choice.
- The tutorial {doc}`../tutorials/03_engines` runs the engines on one catalogue and
  compares them.
- The theory page describes [the engines](../how-it-works/theory.md#the-engines),
  [unconstrained coordinates](../how-it-works/theory.md#unconstrained-coordinates) and
  [two densities, two modes](../how-it-works/theory.md#two-densities-two-modes).
- The {doc}`defaults page <../reference/defaults>` lists the thresholds the engines and
  their checks use.
- Back to the {doc}`start of this section <index>`.
