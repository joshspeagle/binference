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

INK = plt.rcParams["text.color"]
```

## One posterior, several ways to reach it

The model fixes one posterior for the population, $p(\rho\mid D)$. An {term}`engine` is
an algorithm that describes it. This table is the summary of what each engine
computes and when its answer is exact.

| engine | computes | exact when |
|---|---|---|
| {term}`NUTS` | draws from the posterior | always, once its chains have converged |
| {term}`Gibbs` | draws from the posterior, with every object's bin | for a conjugate prior and integer weights, once converged |
| {term}`MAP` | the most probable population, with a certificate | never; it is a point, not a posterior |
| {term}`Laplace` | draws from a heavy-tailed version of a Gaussian at the mode, reweighted toward the posterior | when its importance check passes |
| {term}`mean-field VB` | draws from an approximation that treats the population and the objects' bins as independent | never; its intervals are too narrow |
| {term}`bootstrap` | MAP estimates refitted to resampled objects | never; it is not a posterior |
| {term}`auto` | Gibbs, Laplace or NUTS, chosen for the data | when the engine it chose is |

binference marks results whose draws are not from the exact posterior as
{term}`approximate`.

## The engines one by one

**NUTS.** NUTS follows the slope of the log posterior to take long steps through it.
It works with every prior, and its draws stay nearly independent even when likelihoods
are broad. {term}`R-hat`, the {term}`effective sample size` and its
{term}`divergences <divergence>` check that its {term}`chains <chain>` have converged.

**Gibbs.** Gibbs sampling alternates two exact steps. It draws every object's bin
given the population, then the population given those bins. When likelihoods are
broad compared with the bins, each step changes the population only a little, so
successive draws are strongly correlated. The {term}`missing information` $f$ measures
this.

**MAP.** With the default flat Dirichlet prior, the MAP is the maximum-likelihood
population. Three solvers share the work; the
[engines guide](../guides/engines.md#point-estimates) says which runs when. Its
certificate, the {term}`KKT gap`, says how close the answer is to the optimum. A point
carries no uncertainty, so use it for quick looks and as a starting point.

**Laplace.** The Laplace engine fits a Gaussian at the posterior's mode, in
{term}`unconstrained coordinates`, where every vector of real numbers is a valid population.
It then draws from a heavy-tailed version of that Gaussian and reweights the draws so
that they follow the exact posterior ({term}`importance sampling`). The {term}`Pareto k-hat` $\hat k$ of the weights says
whether that worked. If it did not, binference marks the result approximate.

**Mean-field VB.** Treating the population and each object's bin as independent ignores
the coupling that broad likelihoods create. That is why its intervals are too narrow.
It is fast, and binference always marks the result approximate.

**Bootstrap.** The bootstrap refits the MAP to resampled or reweighted objects. It
answers how much the estimate would change with new data, not how probable each
population is.

**Auto.** The default engine tries Gibbs where Gibbs mixes well enough, running it
longer to make up for correlated draws. It then tries Laplace where its check passes,
and NUTS otherwise. {class}`~binference.inference.Auto` gives the thresholds. The rules, and the benchmark
behind them, are in {doc}`how auto chooses <../how-it-works/auto>`.

## Seeing them on one problem

{ref}`Figure <fig-engines>` shows a population on three bins. The fractions
$(\rho_1,\rho_2,\rho_3)$ lie on a triangle, and unconstrained coordinates $(x_1,x_2)$
map that triangle to the whole plane. The samplers and the approximations describe the
same posterior there, and MAP marks its highest point.

```{code-cell} python
---
tags: [remove-input]
mystnb:
  figure:
    name: fig-engines
    caption: |
      One posterior of a population on three bins, in unconstrained coordinates $x$.
      It comes from 40 mock objects with measurement errors between 0.4 and 0.8.
      Top: 1,000 draws each from NUTS and Gibbs, two chains each. Bottom: the NUTS
      draws and the 95% ellipses of the Laplace Gaussian and of mean-field VB. The
      diamond is the mode in $x$, where the Laplace Gaussian is centred. The star is
      the MAP on the simplex.
  image:
    alt: |
      Two scatter plots with the same axes. In the top panel, indigo dots from NUTS
      and amber crosses from Gibbs form one cloud. Its tail runs to the left, and a few
      draws lie far above it. In the
      bottom panel, a teal Laplace ellipse covers most of the same cloud. A smaller
      dashed plum VB ellipse lies inside it. An open diamond marks the mode, and a
      star beside it marks the MAP.
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


fig, axes = plt.subplots(2, 1, figsize=(4.8, 6.6), sharex=True, sharey=True)
ax = axes[0]
ax.plot(*xn.T, ".", ms=3, color="C0", alpha=0.6, label="NUTS draws")
ax.plot(*xg.T, "x", ms=4, mew=0.8, color="C1", alpha=0.6, label="Gibbs draws")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0, ncol=2, markerscale=2)
ax = axes[1]
ax.plot(*xn.T, ".", ms=2, color="#B8BBEE", label="NUTS")
ellipse(ax, lap.info["mode"], lap.info["cov"], color="C2", lw=2, label="Laplace, 95%")
ellipse(ax, xv.mean(axis=0), np.cov(xv.T), color="C3", lw=2, ls="--", label="VB, 95%")
ax.plot(*lap.info["mode"], "D", ms=7, mfc="none", color=INK, label="mode in $x$")
ax.plot(*model.from_value(est.rho), "*", ms=13, color=INK, label="MAP")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0, ncol=2)
for a in axes:
    a.set_ylabel("$x_2$")
    a.set_xlim(-5, 2.5)
    a.set_ylim(-2, 5)
axes[1].set_xlabel("$x_1$")
plt.show()
glue("k_hat", lap.info["k_hat"], display=False)
glue("k_hat_max", lap.info["k_hat_threshold"], display=False)
glue("vb_ratio", float(np.median(vb.std / fits["nuts"].std)), display=False)
glue("missing", float(model.missing_information(est.rho)), display=False)
```

NUTS and Gibbs agree: both draw from the exact posterior. The missing information here
is $f=$ {glue:text}`missing:.2f`, so successive Gibbs draws are correlated, but the
chains still mix. The Laplace Gaussian is symmetric, while the posterior has a longer
tail to one side. Here its importance check passed, with $\hat k=$
{glue:text}`k_hat:.2f` against a limit of {glue:text}`k_hat_max:.2f`, so its reweighted
draws follow the posterior. The mean-field VB ellipse is too small. Its standard
deviations are {glue:text}`vb_ratio:.2f` times those of NUTS (median over the
fractions).

The MAP and the mode in $x$ differ because they are the peaks of two densities of the
same posterior, one in $\rho$ and one in $x$.

```{plain-words}
NUTS and Gibbs draw from the posterior, and Laplace and VB approximate it; binference
marks an approximation that is not exact. MAP finds the posterior's highest point. The
bootstrap answers a different question.
```

## Where next

- Do it: the {doc}`engines guide <../guides/engines>`.
- See it: the tutorial {doc}`../tutorials/choosing-an-engine`.
- Derivation: [the engines](../how-it-works/engines.md) in How it works.
