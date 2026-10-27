---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Selection effects and rates

Catalogues are rarely complete. The {term}`selection` $S_k$ is the probability that an
object in bin $k$ enters the catalogue. This tutorial treats the two common cases, in the
terms of the [selection guide](../guides/selection):

1. **Occurrence rates, with a completeness that depends on the true value.** The
   {term}`selection mode` is `"on_truth"`, with Poisson {term}`rates` and an
   {term}`exposure`.
2. **A cut on the measured value**, as in a flux-limited survey. The selection mode is
   `"on_measurement"`.

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
from scipy import stats

import binference

rng = np.random.default_rng(2026)
```

## 1. Occurrence rates of planets

A simulated transit survey watches 5000 stars. Planet radii $R$ are binned in
$\log_{10}(R/R_\oplus)$, with ten bins from 0.5 to 16 Earth radii. The occurrence rate
$\lambda_k$ is the expected number of planets per star in bin $k$. The survey detects a
planet with probability $S_k$, which falls steeply for small planets. Measured radii
scatter by 0.08 dex.

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

Detection depends on the true radius, not on the measurement, so the selection acts on
the truth. To infer rates per star, the exposure is the number of stars $N_\star$ times
the completeness. The expected number of detections in bin $k$ is then
$N_\star S_k\lambda_k$. These values exceed one, so the selection is declared an
exposure.

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
---
mystnb:
  figure:
    caption: Occurrence rates per star in each radius bin, inferred by binference (circles
      with 68% intervals), detected planets per star (squares) and the truth (dashed).
  image:
    alt: Log-log plot of planets per star against radius from 0.5 to 16 Earth radii. The
      dashed truth peaks near 2 Earth radii. The circles follow it, with longer intervals
      for small planets. The squares fall below the truth for planets smaller than about
      3 Earth radii, by a factor of ten in the smallest bin.
---
OBJECT = "#EC5F3E"   # the colour of single objects in these docs
radius = 10 ** x
mean = fit.rates.mean(axis=0)
lo, hi = fit.interval(0.68, which="rates")
fig, ax = plt.subplots(figsize=(6, 4.8))
ax.errorbar(radius, mean, yerr=[mean - lo, hi - mean], fmt="o", color="C0", ms=7,
            capsize=3, label="binference")
ax.plot(radius, rate_true, "k--", lw=1.5, label="truth")
ax.plot(radius, naive, "s", color=OBJECT, ms=7, label="detected / stars")
ax.set(xscale="log", yscale="log", xlabel=r"planet radius [$R_\oplus$]",
       ylabel="planets per star")
ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1));
```

Naive counts miss most small planets. binference's rates recover the truth. Their error
bars widen where few planets were detected. Integrated quantities come from the draws:

```{code-cell} python
small = x < np.log10(2.0)
per_star = fit.rates[:, small].sum(axis=1)
print(f"planets smaller than 2 Earth radii per star: {per_star.mean():.3f} ± {per_star.std():.3f}"
      f"   (truth {rate_true[small].sum():.3f})")
```

## 2. A cut on the measured value

A flux-limited redshift survey keeps an object only if its measured redshift is below
$z_{\rm lim}=1.6$. Objects whose true redshift is just above the limit can scatter in,
and objects just below it can scatter out.

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

Here the selection acts on the measurement. $S_k$ is the probability that an object in
bin $k$ passes the cut, averaged over the bin. These values are already bin averages, so
they go in as a `Selection` object. A raw array would be checked for steep changes
between bins.

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

Both fits use a Gaussian-process prior. Beyond the cut the data say almost nothing. A
{term}`smoothness prior` extends the shape below the cut into those bins; with a
Dirichlet prior they would be nearly unconstrained.

```{code-cell} python
---
mystnb:
  figure:
    caption: The redshift population (median, 68% and 95% bands) with the selection on
      the measurement (top) and without it (bottom). The dotted line marks the cut at
      1.6; the truth is dashed.
  image:
    alt: Two stacked step plots of density against redshift from 0 to 2.5. In the top
      one the bands follow the dashed truth past the dotted cut, and widen greatly toward
      2.5. In the bottom one the population drops to zero at the cut, while the truth
      continues.
---
fig, axes = plt.subplots(2, 1, figsize=(6, 6.4), sharex=True, sharey=True)
for ax, (name, res) in zip(axes, fits.items()):
    res.plot(ax=ax, truth=rho_true, label="median")
    ax.set_title(name, loc="left")
    ax.axvline(1.6, color="0.45", lw=1, ls=":")
axes[0].get_legend().remove()
axes[1].legend(loc="upper right")
axes[0].set_xlabel("")
axes[1].set_xlabel("z");
```

Ignoring the cut truncates the population sharply at the limit. With the selection,
the population just beyond $z_{\rm lim}$ is recovered from the objects that scattered
across the cut. The smoothness prior carries it further, and the widening bands show
where the data run out.

```{note}
Passing `"on_truth"` here would be wrong. It would multiply each object's likelihood
by $S_k$, as if objects were selected on their true redshift, and so count the selection
twice. Always ask what the selection acts on.
```
