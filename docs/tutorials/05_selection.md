---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Selection effects and rates

Catalogues are rarely complete. This tutorial treats the two common cases,
using the vocabulary of the [user guide](../user-guide/selection):

1. **Occurrence rates with a completeness that depends on the true value.**
   This is `selection_mode="on_truth"` with Poisson rates and an exposure.
2. **A cut on the measured value**, as in a flux-limited survey. This is
   `selection_mode="on_measurement"`.

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import binference

plt.rcParams["figure.dpi"] = 110
rng = np.random.default_rng(2026)
```

## 1. Occurrence rates of planets

We simulate a transit survey of 5000 stars. Planet radii are binned in
$x=\log_{10}(R/R_\oplus)$, with ten bins from 0.5 to 16 Earth radii. The
occurrence rate $\lambda_k$ is the expected number of planets per star in
bin $k$. The survey detects a planet with probability $S_k$, which falls
steeply for small planets. Measured radii scatter by 0.08 dex.

```{code-cell} python
bins = binference.Bins.linear(np.log10(0.5), np.log10(16), 10)
x = bins.centers
n_stars = 5_000
rate_true = 0.25 * np.exp(-0.5 * ((x - 0.3) / 0.35) ** 2) + 0.02      # planets per star per bin
completeness = 1.0 / (1.0 + np.exp(-(x - 0.1) / 0.12))               # detection probability

n_planets = rng.poisson(n_stars * rate_true)                          # all planets, per bin
edges = bins.edges[0]
true_x = np.concatenate([rng.uniform(edges[k], edges[k + 1], n) for k, n in enumerate(n_planets)])
true_bin = bins.index(true_x)
detected = rng.random(true_x.size) < completeness[true_bin]
sigma = 0.08
obs_x = true_x[detected] + sigma * rng.standard_normal(detected.sum())
print(f"{n_planets.sum()} planets, {detected.sum()} detected")
```

Detection depends on the true radius, not on the measurement, so this is
`selection_mode="on_truth"`. To infer rates per star, the exposure is the number
of stars times the completeness: the expected number of detections in bin
$k$ is $N_\star S_k\lambda_k$. These values exceed one, so the selection is
declared an exposure.

```{code-cell} python
lk = binference.Likelihoods.from_gaussian(obs_x, sigma, bins)
fit = binference.infer(
    lk, binference.priors.GammaRates(shape=1.0, rate=1e-3),
    selection=binference.Selection(n_stars * completeness, "on_truth", scale="exposure"),
    rng=1,
)
idx = bins.index(obs_x)                                               # -1 outside the bins
naive = np.bincount(idx[idx >= 0], minlength=bins.K) / n_stars        # detected / stars
```

```{code-cell} python
radius = 10 ** x
lo, hi = fit.interval(0.68, which="rates")
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.errorbar(radius, fit.rates.mean(axis=0), yerr=[fit.rates.mean(axis=0) - lo, hi - fit.rates.mean(axis=0)],
            fmt="o", color="#2a78d6", ms=5, capsize=2, label="binference (68% intervals)")
ax.plot(radius, rate_true, "k--", lw=1.5, label="truth")
ax.plot(radius, naive, "s", color="#eb6834", ms=5, label="detected / stars")
ax.set(xscale="log", yscale="log", xlabel="planet radius [Earth radii]",
       ylabel="planets per star per bin")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False);
```

Naive counts miss most small planets. binference's rates recover the truth, with
error bars that widen where few planets were detected. Integrated quantities
come from the draws:

```{code-cell} python
small = x < np.log10(2.0)
per_star = fit.rates[:, small].sum(axis=1)
print(f"planets smaller than 2 Earth radii per star: {per_star.mean():.3f} ± {per_star.std():.3f}"
      f"   (truth {rate_true[small].sum():.3f})")
```

## 2. A cut on the measured value

A flux-limited redshift survey keeps an object only if its *measured*
redshift is below $z_{\rm lim}=1.6$. Objects whose true redshift is just
above the limit can scatter in, and objects just below can scatter out.

```{code-cell} python
zbins = binference.Bins.linear(0.0, 2.5, 25)
rho_true = binference.simulate.population(zbins, lambda z: z**2 * np.exp(-z / 0.4))
labels = rng.choice(zbins.K, size=5_000, p=rho_true)
ze = zbins.edges[0]
z_true = rng.uniform(ze[labels], ze[labels + 1])
sig_z = 0.12
z_obs = z_true + sig_z * rng.standard_normal(z_true.size)
keep = z_obs < 1.6
lk_z = binference.Likelihoods.from_gaussian(z_obs[keep], sig_z, zbins)
print(f"kept {keep.sum()} of {keep.size}")
```

Here the selection acts on the measurement, so `selection_mode="on_measurement"`.
$S_k$ is the probability that an object in bin $k$ passes the cut, averaged
over the bin. These are already bin averages, so they go in as a `Selection`
object (a raw array would be checked for steep changes between bins):

```{code-cell} python
fine = np.linspace(0, 2.5, 2501)
pass_prob = stats.norm.cdf((1.6 - fine) / sig_z)
S = np.array([pass_prob[(fine >= a) & (fine < b)].mean() for a, b in zip(ze[:-1], ze[1:])])

gp = binference.priors.LogisticGP()
kw = dict(rng=1, n_samples=500, n_warmup=500)
fits = {
    "selection on the measurement": binference.infer(
        lk_z, gp, selection=binference.Selection(S, "on_measurement"), **kw
    ),
    "no selection": binference.infer(lk_z, gp, **kw),
}
```

We use a Gaussian-process prior here. Beyond the cut the data say almost
nothing, and a smoothness prior is what lets the population be extrapolated
from the shape below the cut. With a Dirichlet prior those bins would be
nearly unconstrained.

```{code-cell} python
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, constrained_layout=True)
for ax, (name, res) in zip(axes, fits.items()):
    res.plot(ax=ax, truth=rho_true)
    ax.set_title(name, loc="left", fontsize=10)
    ax.axvline(1.6, color="0.5", lw=1, ls=":")
    ax.set_xlabel("z")
axes[1].get_legend().remove()
```

Ignoring the cut truncates the population sharply at the limit. With the
selection function, the population just beyond $z_{\rm lim}$ is recovered
from the objects that scattered across the cut, and the smoothness prior
carries it further. The widening bands show where the data run out.

```{note}
Passing `selection_mode="on_truth"` here would be wrong. It would multiply each
object's likelihood by $S_k$, as if objects were selected on their true
redshift, and count the selection twice. Always ask what the selection acts
on.
```
