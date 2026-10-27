---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Quick start: a redshift distribution

This tutorial infers the redshift distribution of 2000 galaxies. Each galaxy has a
photometric redshift with a Gaussian error of its own. The tutorial compares the result
with {term}`stacking` the galaxies' likelihoods. It then derives quantities with error
bars, and looks at what the population says about single galaxies.

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

import binference

OBJECT = "#EC5F3E"   # the colour of single objects in these docs
```

## A mock catalogue

{func}`binference.simulate.mock_1d` draws true redshifts $z$ from a population with two
peaks. It adds Gaussian errors with standard deviations between 0.05 and 0.2. The 30 bins
on $[0, 3]$ are 0.1 wide, so the errors span half a bin to two bins. The
{term}`mock catalogue` holds the likelihoods, the truth and the measurements.

```{code-cell} python
mock = binference.simulate.mock_1d(2000, rng=42)
lk, bins = mock.likelihoods, mock.bins
print(lk)
print("measured z of the first galaxies:", mock.measurements["mean"][:4].round(2))
print("their errors:                   ", mock.measurements["sigma"][:4].round(2))
```

Each galaxy $i$ contributes one row of {term}`bin-averaged likelihoods <bin-averaged
likelihood>` $L_{ik}$, one value per bin $k$. Five of the rows look like this:

```{code-cell} python
---
mystnb:
  figure:
    caption: Bin-averaged likelihoods of five galaxies, each scaled to a maximum of one.
  image:
    alt: Five step curves over redshift 0 to 3, each a narrow or wide bump that peaks
      at one. The narrow bumps cover one or two bins and the wide ones five or six.
---
fig, ax = plt.subplots()
L = lk.toarray()
for i in range(5):
    ax.step(bins.edges[0], np.append(L[i], L[i, -1]), where="post", color=OBJECT, lw=1.2)
ax.set(xlabel="z", ylabel="likelihood", xlim=(0, 3));
```

## Stacking versus the hierarchical posterior

Stacking adds up the galaxies' likelihoods. It is the first step of an iteration whose
fixed point is the most likely population, and it gives no error bars. binference
samples the posterior of the {term}`population fractions` $\rho_k$ instead:

```{code-cell} python
stack = lk.stack()
result = binference.infer(lk, rng=1)      # method="auto" picks an exact engine here
print(result.info["auto_reason"])
result
```

```{code-cell} python
---
mystnb:
  figure:
    caption: The population posterior (median and 68% and 95% bands) under a flat
      Dirichlet prior, with the truth (dashed) and the stack (dotted).
  image:
    alt: Step plot of density against redshift. The posterior median is jagged and its
      bands are wide; it follows the dashed truth, which has a broad peak near 0.6 and
      a narrower one near 1.6. The dotted stack is smoother and broader than the truth.
---
fig, ax = plt.subplots(figsize=(6, 4.2))
result.plot(ax=ax, truth=mock.truth.rho, stack=stack, sample_color=OBJECT, label="median")
ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1))    # above the plot
ax.set_xlabel("z");
```

The bands are wide and the median is jagged. The sampler is not at fault. The errors are
as wide as the bins, so the data barely tell neighbouring bins apart. The flat
{term}`Dirichlet prior` lets those bins trade mass freely, so they are strongly
anti-correlated:

```{code-cell} python
corr = result.corr()
print("correlation of bin 10 with bins 9, 10, 11:", corr[10, 9:12].round(2))
```

```{code-cell} python
:tags: [remove-cell]

assert corr[10, 9] < -0.3 and corr[10, 11] < -0.3
```

The summary table gives the numbers per bin and the convergence diagnostics:

```{code-cell} python
result.summary()
```

## Derived quantities

Sums and averages over bins are much better determined than single bins, because the
anti-correlations cancel. Compute a derived quantity on each posterior draw:

```{code-cell} python
centers = bins.centers
zbar = result.functional(lambda rho: rho @ centers)
f_high = result.functional(lambda rho: rho[centers > 1.5].sum())

print(f"mean redshift:   {zbar.mean():.3f} ± {zbar.std():.3f}   (truth {mock.truth.rho @ centers:.3f})")
print(f"fraction z>1.5:  {f_high.mean():.3f} ± {f_high.std():.3f}   (truth {mock.truth.rho[centers > 1.5].sum():.3f})")
print(f"stacked mean z:  {stack @ centers:.3f}  (no error bar)")
```

Adding the per-bin variances ignores the correlations, and gives a much larger error bar
on the mean redshift:

```{code-cell} python
naive = np.sqrt(np.sum((result.std * centers) ** 2))
print(f"error ignoring correlations: {naive:.3f}   vs   correct: {zbar.std():.3f}")
```

```{code-cell} python
:tags: [remove-cell]

assert naive > 3 * zbar.std()
```

## A smoothness prior

If you expect the population to be smooth over a few bins, say so with a
{term}`smoothness prior`. A Gaussian-process prior on $\log\rho$ shares information
between neighbouring bins. binference infers its amplitude and length scale, its
{term}`hyperparameters <hyperparameter>`, from the data:

```{code-cell} python
smooth = binference.infer(lk, binference.priors.LogisticGP(), rng=1, n_samples=500)  # shorter run
hyper = smooth.info["hyperparameters"]
print({k: np.round(np.quantile(v, [0.16, 0.5, 0.84]), 2) for k, v in hyper.items()})
```

```{code-cell} python
---
mystnb:
  figure:
    caption: The population posterior under a flat Dirichlet prior (top) and a
      Gaussian-process prior (bottom), with the truth dashed.
  image:
    alt: Two stacked step plots of density against redshift. The top one, under the
      Dirichlet prior, has wide jagged bands. The bottom one, under the Gaussian-process
      prior, has narrow smooth bands that follow the dashed truth.
---
fig, axes = plt.subplots(2, 1, figsize=(6, 6.4), sharex=True, sharey=True)
result.plot(ax=axes[0], truth=mock.truth.rho)
smooth.plot(ax=axes[1], truth=mock.truth.rho, label="median")
axes[0].set_title("Dirichlet prior", loc="left")
axes[1].set_title("Gaussian-process prior", loc="left")
axes[0].get_legend().remove()
axes[1].legend(loc="upper right")
axes[0].set_xlabel("")
axes[1].set_xlabel("z");
```

```{code-cell} python
:tags: [remove-cell]

assert np.median(smooth.std / result.std) < 0.5
```

The smooth posterior is far tighter per bin and still covers the truth. Which prior is
better is a scientific question, but the data can weigh in. Fit each prior to 1500
galaxies, with shorter NUTS runs than the default. Then score the other 500 with the
{term}`log predictive density`:

```{code-cell} python
train, test = lk[:1500], lk[1500:]
lp = {}
for name, prior in [("Dirichlet", None), ("GP", binference.priors.LogisticGP())]:
    fit = binference.infer(train, prior, method="nuts", rng=2, n_samples=300, n_warmup=300)
    lp[name] = fit.log_predictive(test).sum()
print({k: round(v, 1) for k, v in lp.items()})
```

```{code-cell} python
:tags: [remove-cell]

assert lp["GP"] > lp["Dirichlet"]
```

Higher is better: the smoothness prior predicts the held-out galaxies better.

## Single galaxies

The population also sharpens each galaxy's redshift. Its {term}`hierarchical posterior`
$P(k_i=k\mid D)$ combines its own likelihood with the inferred population. It is
narrower than the likelihood wherever the population changes across the likelihood's
width. `assignment_probabilities()` returns these posteriors for every galaxy. Here they
pick the galaxy whose redshift the population narrows most, measured by the spread of its
posterior over the bins:

```{code-cell} python
---
mystnb:
  figure:
    caption: One galaxy's own likelihood (dashed) and its hierarchical posterior
      (solid). The dotted line marks its true redshift.
  image:
    alt: Two step curves over redshift. The dashed coral likelihood is broad. The solid
      indigo posterior is taller and narrower, and the dotted vertical line of the true
      redshift falls inside it.
---
from binference.plotting import plot_individual

def spread(p):                       # standard deviation of each row over the bin centres
    return np.sqrt(p @ centers**2 - (p @ centers) ** 2)

own = L * bins.volumes
own /= own.sum(axis=1, keepdims=True)
post = smooth.assignment_probabilities()
i = int(np.argmax(spread(own) / spread(post)))
print(f"galaxy {i}: spread {spread(own)[i]:.2f} alone, {spread(post)[i]:.2f} in the population")

fig, ax = plt.subplots(figsize=(6, 4.2))
plot_individual(smooth, i, ax=ax, sample_color=OBJECT)
ax.axvline(mock.truth.values[i], color="k", lw=1, ls=":", label="true z")
ax.set_xlabel("z")
ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1));
```

```{code-cell} python
:tags: [remove-cell]

assert spread(post)[i] < 0.9 * spread(own)[i]
```

The effect is modest here, because the population is smooth on the scale of the
galaxy's error. It grows where the population has sharp features, as in
{doc}`empirical-bayes`.

`result.sample_values()` gives joint draws of the galaxies' true redshifts.

## The population versus this sample

$\rho$ describes the population that the 2000 galaxies were drawn from. The redshift
histogram of these galaxies, the {term}`realised sample`, is a different quantity with
its own uncertainty:

```{code-cell} python
counts = smooth.realized_counts(n_draws=500, rng=0)
print("sd of realised counts per bin:", counts.std(axis=0)[8:12].round(1))
print("sd of population counts per bin:", (smooth.samples * lk.W).std(axis=0)[8:12].round(1))
```

```{code-cell} python
:tags: [remove-cell]

assert np.all(counts.std(axis=0)[8:12] > (smooth.samples * lk.W).std(axis=0)[8:12])
```

Here the histogram is the less certain of the two. The smoothness prior pins the
population down. But with errors as wide as the bins, it cannot say which bin each galaxy
fell in. With precise measurements the order reverses. Each galaxy's bin is then nearly
known, while $\rho$ keeps the shot noise of drawing 2000 galaxies.

## Next steps

- {doc}`priors-and-smoothness` compares more priors on a sparser catalogue.
- {doc}`posteriors-on-a-grid` and {doc}`posterior-samples` start from posteriors
  instead of likelihoods.
- {doc}`choosing-an-engine` compares the engines.
- The [guides](../guides/index) explain every option.
