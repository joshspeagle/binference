---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Empirical Bayes deconvolution

Many problems in statistics have the same shape: a large number of units
$i=1,\dots,n$, each with an unknown effect $\theta_i$ and a noisy estimate
$x_i\sim\mathcal N(\theta_i, s_i^2)$ with a known standard error $s_i$. The
units might be genes in a microarray study, schools, hospitals or baseball
players. Estimating the distribution $g$ of the effects from the $x_i$ is
*deconvolution*; using the estimated $g$ as a prior for each unit is
*empirical Bayes*, and its posterior means shrink each $x_i$ toward where
the other units lie.

On a grid of bins this is exactly binference's model: $g$ is the population
$\rho$ over bins of $\theta$, and each unit's likelihood is its Gaussian
measurement. Efron's g-modelling fits a smooth exponential family for $g$
on such a grid, and the nonparametric maximum-likelihood estimate (NPMLE)
of Kiefer and Wolfowitz, as computed by Koenker's REBayes, maximises the
likelihood over all mixing distributions. This tutorial computes the NPMLE,
the full posterior over $g$, and the per-unit shrinkage estimates that
follow from each.

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import binference

plt.rcParams["figure.dpi"] = 110
rng = np.random.default_rng(81)
```

## Simulated effects

1500 units. 60% of them have no effect, $\theta_i=0$; the others have
effects drawn from $\mathcal N(2.5, 0.7^2)$. Standard errors are uniform
between 0.5 and 1.5, comparable to the gap between the two groups, so the
estimates of null and non-null units overlap.

```{code-cell} python
n = 1500
null = rng.random(n) < 0.6
theta = np.where(null, 0.0, rng.normal(2.5, 0.7, n))
s = rng.uniform(0.5, 1.5, n)
x = theta + s * rng.standard_normal(n)
print(f"{null.sum()} null units; estimates range from {x.min():.2f} to {x.max():.2f}")
```

The grid has 70 bins of width 0.1 on $[-2.05, 4.95]$, with one bin centred
on zero. The true $g$ in these bins is the point mass at zero plus the
Gaussian's mass in each bin:

```{code-cell} python
bins = binference.Bins.linear(-2.05, 4.95, 70)
edges = bins.edges[0]
g_true = 0.4 * np.diff(stats.norm.cdf(edges, 2.5, 0.7))
g_true[bins.index(0.0)] += 0.6
g_true /= g_true.sum()                 # the Gaussian's tails beyond the grid are dropped
lk = binference.Likelihoods.from_gaussian(x, s, bins)
```

## The NPMLE and the posterior over g

With the default flat Dirichlet prior, the MAP estimate is the maximum
likelihood estimate of $\rho$ on the grid, that is, the NPMLE restricted to
the grid. `method="map"` computes it with mix-SQP and reports its KKT gap
as a certificate of optimality:

```{code-cell} python
npmle = binference.infer(lk, method="map")
atoms = npmle.rho > 1e-3
print(f"KKT gap {npmle.kkt_gap:.1e}; {atoms.sum()} bins hold more than 0.1% of the mass")
for c, w in zip(bins.centers[atoms], npmle.rho[atoms]):
    print(f"  theta = {c:5.2f}: {w:.3f}")
```

```{code-cell} python
:tags: [remove-cell]

assert npmle.kkt_gap < 1e-6 and atoms.sum() <= 10
```

The NPMLE is discrete: it puts all its mass on a few grid points, as the
theory of the Kiefer–Wolfowitz estimator says it must. It finds the mass at
zero but represents the Gaussian component by a few atoms, and it gives no
measure of its own uncertainty. The default `method="auto"` samples the
full posterior over $\rho$:

```{code-cell} python
fit = binference.infer(lk, n_samples=500, rng=1)
print(fit.info["auto_method"], "-", fit.info["auto_reason"])
```

```{code-cell} python
fig, ax = plt.subplots(figsize=(8, 3.8))
fit.plot(ax=ax, which="fractions", truth=g_true)
ax.vlines(bins.centers[atoms], 0, npmle.rho[atoms], color="#eb6834", lw=2, label="NPMLE")
ax.set(xlabel=r"effect $\theta$", ylim=(0, 0.35))
ax.legend(frameon=False);
```

The truth's spike at zero runs off the top of the plot. The posterior bands
are wide: with standard errors between 5 and 15 bins wide, the data cannot
tell a point mass at zero from a narrow bump around it, so the posterior
spreads the null mass over neighbouring bins. Sums over wider ranges are
better determined than single bins:

```{code-cell} python
near, mass = {}, {}
for cut in (0.5, 1.25):
    near[cut] = b = np.abs(bins.centers) < cut
    mass[cut] = m = fit.functional(lambda rho: rho[b].sum())
    print(f"mass within {cut} of zero: posterior {m.mean():.3f} ± {m.std():.3f}, "
          f"NPMLE {npmle.rho[b].sum():.3f}, truth {g_true[b].sum():.3f}")
ratio = fit.prior_information_ratio()
print(f"prior information ratio: {ratio:.2f}")
```

```{code-cell} python
:tags: [remove-cell]

assert abs(mass[1.25].mean() - g_true[near[1.25]].sum()) < 3 * mass[1.25].std()
assert g_true[near[0.5]].sum() - mass[0.5].mean() > 3 * mass[0.5].std()
assert abs(npmle.rho[near[0.5]].sum() - g_true[near[0.5]].sum()) < 0.05
assert ratio > 1
```

Within 1.25 of zero, the posterior agrees with the truth. Within 0.5 it
falls well short of it, while the NPMLE does not: the likelihood changes
little when mass moves from the spike into the empty bins beside it, and a
posterior spread over that direction has its mean inside the simplex, away
from the spike. A prior information ratio
({meth}`~binference.PopulationResult.prior_information_ratio`) above one says
that the flat Dirichlet on 70 bins weighs more than the data in this fine
structure. A question about the null fraction should be put
as a sum over a range the data can resolve, or answered with a prior that
expects a spike.

## Shrinkage estimates for each unit

Each unit's posterior under the estimated $g$ is
`assignment_probabilities()`, its probability of lying in each bin. For the
NPMLE this is the usual plug-in empirical Bayes posterior; for the full fit
it is averaged over the posterior draws of $\rho$, so the uncertainty in $g$
is carried into each unit. Averaging the bin centres over it approximates
the posterior mean; `sample_values()` draws the units' true values within
the bins from their own likelihoods, which removes the bin-centre
approximation.

```{code-cell} python
eb_npmle = npmle.assignment_probabilities() @ bins.centers
draws = fit.sample_values(rng=2)                  # (500 draws, units)
eb_full = draws.mean(axis=0)
eb_centres = fit.assignment_probabilities() @ bins.centers
print("posterior mean from bin centres vs from sample_values, largest difference:",
      f"{np.abs(eb_centres - eb_full).max():.3f}")  # includes the draws' Monte Carlo error
```

For comparison, the oracle Bayes rule uses the true $g$, by numerical
integration over $\theta$:

```{code-cell} python
grid = np.linspace(-6, 9, 3001)
dx = grid[1] - grid[0]
slab = 0.4 * stats.norm.pdf(grid, 2.5, 0.7)
like = stats.norm.pdf(x[:, None], grid, s[:, None])
oracle = (like @ (slab * grid)) * dx / ((like @ slab) * dx + 0.6 * stats.norm.pdf(x, 0, s))

mse = {name: np.mean((est - theta) ** 2) for name, est in
       [("raw estimate x", x), ("EB, NPMLE", eb_npmle),
        ("EB, full posterior", eb_full), ("oracle Bayes", oracle)]}
for name, v in mse.items():
    print(f"{name:19s} mean squared error {v:.3f}")
```

```{code-cell} python
:tags: [remove-cell]

eb = max(mse["EB, NPMLE"], mse["EB, full posterior"])
assert eb < 0.5 * mse["raw estimate x"] and eb < 1.1 * mse["oracle Bayes"]
```

Both empirical Bayes estimates more than halve the error of the raw
estimates and come close to the oracle that knows $g$. The plot shows the
shrinkage: estimates near zero are pulled to zero, those near the
non-null component toward 2.5, and noisier units (darker points) are
moved further.

```{code-cell} python
fig, ax = plt.subplots(figsize=(6, 4.5))
sc = ax.scatter(x, eb_full, c=s, s=6, cmap="viridis_r")
ax.plot([-3, 7], [-3, 7], color="0.6", lw=1, ls="--")
ax.set(xlabel="raw estimate $x_i$", ylabel=r"posterior mean of $\theta_i$", xlim=(-3, 7), ylim=(-1, 5))
ax.spines[["top", "right"]].set_visible(False)
plt.colorbar(sc, ax=ax, label="standard error $s_i$");
```

The draws also give per-unit intervals. Their coverage of the true effects
is a check of the whole fit:

```{code-cell} python
lo, hi = np.quantile(draws, [0.05, 0.95], axis=0)
covered = (theta >= lo) & (theta <= hi)
print(f"90% intervals cover {covered.mean():.3f} of the true effects; "
      f"mean width {np.mean(hi - lo):.2f}, against {2 * 1.645 * s.mean():.2f} for x ± 1.645 s")
```

```{code-cell} python
:tags: [remove-cell]

assert covered.mean() > 0.9 - 3 * np.sqrt(0.09 / n) and np.mean(hi - lo) < 2 * 1.645 * s.mean()
```

## What to take away

- Deconvolution with known standard errors is binference's model, with
  `Likelihoods.from_gaussian` on a grid of bins.
- `method="map"` with the default flat Dirichlet prior is the grid NPMLE;
  the full posterior over $g$ shows how much of $g$ the data determine, and
  `assignment_probabilities()` and `sample_values()` give each unit's
  empirical Bayes posterior with that uncertainty included.
- A smooth $g$, as in Efron's g-modelling, corresponds to a smoothness prior
  such as `priors.LogisticGP()` ({doc}`04_priors`). The
  [empirical Bayes background page](../background/empirical-bayes) places
  this tutorial among the packages of that field.
