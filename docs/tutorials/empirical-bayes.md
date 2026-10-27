---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Empirical Bayes deconvolution

Many problems in statistics share one shape. There are many units $i=1,\dots,N$, each
with an unknown effect $z_i$. Each has a noisy estimate $d_i\sim\mathcal N(z_i,
\sigma_i^2)$ with a known standard error $\sigma_i$. The units might be genes, schools,
hospitals or baseball players. Estimating the distribution of the effects from the
$d_i$ is deconvolution. Using that distribution as a prior for each unit is empirical
Bayes. Its posterior means shrink each $d_i$ toward where the other units lie.

On a grid of bins this is binference's model. The distribution of the effects is the
population $\rho$ over bins of $z$; the empirical Bayes literature calls it $g$. Each
unit's likelihood is its Gaussian measurement. The nonparametric maximum-likelihood
estimate (NPMLE) of {cite:t}`kiefer1956` maximises the likelihood over all distributions
of $z$. This tutorial computes the NPMLE, the full posterior of $\rho$, and the
{term}`shrinkage` estimates of the units that follow from each.

```{code-cell} python
:tags: [remove-cell]

# figures 6 in wide with 17 pt text keep text above 12 px on a 390 px phone screen
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.figsize": (6, 3.4), "font.size": 17,
                     "figure.constrained_layout.use": True})
```

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

import binference

rng = np.random.default_rng(81)
```

## Simulated effects

There are 1500 units. 60% of them have no effect, $z_i=0$; the others have effects drawn
from $\mathcal N(2.5, 0.7^2)$. The standard errors are uniform between 0.5 and 1.5. That
is comparable to the gap between the two groups, so the estimates of null and non-null
units overlap.

```{code-cell} python
n = 1500
null = rng.random(n) < 0.6
z = np.where(null, 0.0, rng.normal(2.5, 0.7, n))       # true effects
sigma = rng.uniform(0.5, 1.5, n)                        # standard errors
d = z + sigma * rng.standard_normal(n)                  # estimates
print(f"{null.sum()} null units; estimates range from {d.min():.2f} to {d.max():.2f}")
```

The grid has 70 bins of width 0.1 on $[-2.05, 4.95]$, with one bin centred on zero. The
true $\rho$ in these bins is the point mass at zero plus the Gaussian's mass in each bin:

```{code-cell} python
bins = binference.Bins.linear(-2.05, 4.95, 70)
edges = bins.edges[0]
rho_true = 0.4 * np.diff(stats.norm.cdf(edges, 2.5, 0.7))
rho_true[bins.index(0.0)] += 0.6
rho_true /= rho_true.sum()             # the Gaussian's tails beyond the grid are dropped
lk = binference.Likelihoods.from_gaussian(d, sigma, bins)
```

## The NPMLE and the posterior of the population

With the default flat Dirichlet prior, the {term}`MAP` estimate is the maximum-likelihood
estimate of $\rho$ on the grid: the NPMLE restricted to the grid. `method="map"` computes
it with {term}`mix-SQP`. It reports the {term}`KKT gap` as a certificate of optimality:

```{code-cell} python
npmle = binference.infer(lk, method="map")
atoms = npmle.rho > 1e-3
print(f"KKT gap {npmle.kkt_gap:.1e}; {atoms.sum()} bins hold more than 0.1% of the mass")
for c, w in zip(bins.centers[atoms], npmle.rho[atoms]):
    print(f"  z = {c:5.2f}: {w:.3f}")
```

```{code-cell} python
:tags: [remove-cell]

assert npmle.kkt_gap < 1e-6 and atoms.sum() <= 10
```

The NPMLE is discrete: it puts all its mass on a few grid points, as the theory of the
Kiefer–Wolfowitz estimator says it must. It finds the mass at zero, but represents the
Gaussian component by a few atoms. It gives no measure of its own uncertainty. The
default `method="auto"` samples the full posterior of $\rho$:

```{code-cell} python
fit = binference.infer(lk, n_samples=500, rng=1)
print(fit.info["auto_method"], "-", fit.info["auto_reason"])
```

```{code-cell} python
---
mystnb:
  figure:
    caption: The posterior of the population fractions (median, 68% and 95% bands), the
      NPMLE's atoms (vertical lines) and the truth (dashed). The truth's spike at zero,
      0.6, runs off the top.
  image:
    alt: Step plot of the fraction per bin against the effect from -2 to 5. The posterior
      spreads the null mass over the bins around zero, with wide bands, and has a broad
      hump near 2.5. The NPMLE puts a tall line at zero and shorter ones near 1.9 and
      3.3.
---
fig, ax = plt.subplots(figsize=(6, 4.2))
fit.plot(ax=ax, which="fractions", truth=rho_true, label="median")
ax.vlines(bins.centers[atoms], 0, npmle.rho[atoms], color="C1", lw=2.5, label="NPMLE")
ax.set(xlabel="effect $z$", ylim=(0, 0.35))
ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1));
```

The posterior bands are wide. The standard errors are 5 to 15 bins wide, so the data
cannot tell a point mass at zero from a narrow bump around it. The posterior spreads the
null mass over neighbouring bins. Sums over wider ranges are better determined than
single bins:

```{code-cell} python
near, mass = {}, {}
for cut in (0.5, 1.25):
    near[cut] = b = np.abs(bins.centers) < cut
    mass[cut] = m = fit.functional(lambda rho: rho[b].sum())
    print(f"mass within {cut} of zero: posterior {m.mean():.3f} ± {m.std():.3f}, "
          f"NPMLE {npmle.rho[b].sum():.3f}, truth {rho_true[b].sum():.3f}")
ratio = fit.prior_information_ratio()
print(f"prior information ratio: {ratio:.2f}")
```

```{code-cell} python
:tags: [remove-cell]

assert abs(mass[1.25].mean() - rho_true[near[1.25]].sum()) < 3 * mass[1.25].std()
assert rho_true[near[0.5]].sum() - mass[0.5].mean() > 3 * mass[0.5].std()
assert abs(npmle.rho[near[0.5]].sum() - rho_true[near[0.5]].sum()) < 0.05
assert ratio > 1
```

Within 1.25 of zero, the posterior agrees with the truth. Within 0.5 it falls well short,
while the NPMLE does not. The likelihood changes little when mass moves from the spike
into the empty bins beside it. A posterior spread along that direction has its mean
inside the simplex, away from the spike. The prior information ratio
({meth}`~binference.PopulationResult.prior_information_ratio`) is above one. It says
that the flat Dirichlet on 70 bins weighs more than the data in this fine structure. Ask
about the null fraction as a sum over a range the data can resolve, or use a prior that
expects a spike.

## Shrinkage estimates for each unit

Each unit's posterior under the estimated population is `assignment_probabilities()`:
its probability of lying in each bin. For the NPMLE this is the usual plug-in empirical
Bayes posterior. For the full fit it is averaged over the posterior draws of $\rho$, so
the uncertainty of the population carries into each unit. Averaging the bin centres over
it approximates the posterior mean. `sample_values()` draws the units' true values
within the bins from their own likelihoods, which removes that approximation.

```{code-cell} python
eb_npmle = npmle.assignment_probabilities() @ bins.centers
draws = fit.sample_values(rng=2)                  # (500 draws, units)
eb_full = draws.mean(axis=0)
eb_centres = fit.assignment_probabilities() @ bins.centers
print("posterior mean from bin centres vs from sample_values, largest difference:",
      f"{np.abs(eb_centres - eb_full).max():.3f}")  # includes the draws' Monte Carlo error
```

For comparison, the oracle Bayes rule uses the true distribution of the effects. It
integrates over $z$ numerically:

```{code-cell} python
grid = np.linspace(-6, 9, 3001)
dz = grid[1] - grid[0]
slab = 0.4 * stats.norm.pdf(grid, 2.5, 0.7)
like = stats.norm.pdf(d[:, None], grid, sigma[:, None])
oracle = (like @ (slab * grid)) * dz / ((like @ slab) * dz + 0.6 * stats.norm.pdf(d, 0, sigma))

mse = {name: np.mean((est - z) ** 2) for name, est in
       [("raw estimate d", d), ("EB, NPMLE", eb_npmle),
        ("EB, full posterior", eb_full), ("oracle Bayes", oracle)]}
for name, v in mse.items():
    print(f"{name:19s} mean squared error {v:.3f}")
```

```{code-cell} python
:tags: [remove-cell]

eb = max(mse["EB, NPMLE"], mse["EB, full posterior"])
assert eb < 0.5 * mse["raw estimate d"] and eb < 1.1 * mse["oracle Bayes"]
```

Both empirical Bayes estimates more than halve the error of the raw estimates. They come
close to the oracle that knows the true distribution. The plot shows the shrinkage.
Estimates near zero are pulled to zero, and those near the non-null group toward 2.5.
Noisier units, the darker points, move further.

```{code-cell} python
---
mystnb:
  figure:
    caption: Each unit's posterior mean against its raw estimate, shaded by its standard
      error. The dashed line marks no shrinkage.
  image:
    alt: Scatter of 1500 points below and beside the dashed diagonal. Estimates below
      about 1 are pulled toward zero, and large estimates are pulled down toward 2.5 to
      3. Darker points, with larger errors, lie furthest from the diagonal.
---
ramp = LinearSegmentedColormap.from_list("ramp", ["#F7C3B5", "#EC5F3E", "#7A2412"])
fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(d, eb_full, c=sigma, s=8, cmap=ramp, lw=0)
ax.plot([-3, 7], [-3, 7], color="0.45", lw=1, ls="--")
ax.set(xlabel="raw estimate $d_i$", ylabel="posterior mean of $z_i$", xlim=(-3, 7),
       ylim=(-1, 5))
fig.colorbar(sc, ax=ax, label=r"standard error $\sigma_i$");
```

The draws also give each unit an interval. Their coverage of the true effects checks the
whole fit:

```{code-cell} python
lo, hi = np.quantile(draws, [0.05, 0.95], axis=0)
covered = (z >= lo) & (z <= hi)
print(f"90% intervals cover {covered.mean():.3f} of the true effects; "
      f"mean width {np.mean(hi - lo):.2f}, against {2 * 1.645 * sigma.mean():.2f} for d ± 1.645 σ")
```

```{code-cell} python
:tags: [remove-cell]

assert covered.mean() > 0.9 - 3 * np.sqrt(0.09 / n) and np.mean(hi - lo) < 2 * 1.645 * sigma.mean()
```

## What to take away

- Deconvolution with known standard errors is binference's model, with
  `Likelihoods.from_gaussian` on a grid of bins.
- `method="map"` with the default flat Dirichlet prior is the grid NPMLE. The full
  posterior of $\rho$ shows how much of it the data determine.
- `assignment_probabilities()` and `sample_values()` give each unit's empirical Bayes
  posterior, with the population's uncertainty included.
- A smooth $\rho$, as in Efron's g-modelling, corresponds to a smoothness prior such as
  `priors.LogisticGP()` ({doc}`priors-and-smoothness`). The
  [empirical Bayes background page](../fields/empirical-bayes) places this tutorial among
  the packages of that field.
