---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Quick start: a redshift distribution from photo-z likelihoods

This tutorial infers a redshift distribution $n(z)$ from 2000 galaxies whose
photometric redshifts have heteroscedastic Gaussian errors. It compares the
result with the common shortcut of stacking the per-galaxy PDFs, derives
quantities with honest error bars, and looks at what the population says
about individual galaxies.

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np

import binference

plt.rcParams["figure.dpi"] = 110
```

## A mock catalogue

{func}`binference.simulate.mock_1d` draws true redshifts from a bimodal population
and adds Gaussian errors with standard deviations between 0.05 and 0.2, that
is, between half and twice the width of the 30 bins on $[0,3]$. It returns the
likelihoods, the truth and the measurements.

```{code-cell} python
mock = binference.simulate.mock_1d(2000, rng=42)
lk, bins = mock.likelihoods, mock.bins
print(lk)
print("measured z of the first galaxies:", mock.measurements["mean"][:4].round(2))
print("their errors:                   ", mock.measurements["sigma"][:4].round(2))
```

Each galaxy contributes one row of bin-averaged likelihoods $L_{ik}$. Here are
a few of them:

```{code-cell} python
fig, ax = plt.subplots(figsize=(7, 3))
L = lk.toarray()
for i in range(5):
    ax.step(bins.edges[0], np.append(L[i], L[i, -1]), where="post", lw=1.5)
ax.set(xlabel="z", ylabel="likelihood (row max = 1)", xlim=(0, 3))
ax.spines[["top", "right"]].set_visible(False)
```

## Stacking versus the hierarchical posterior

The usual shortcut adds up the per-galaxy PDFs. That is one step of an
iteration whose fixed point is the maximum-likelihood population, and it
gives no error bars. binference samples the full posterior instead:

```{code-cell} python
stack = lk.stack()
result = binference.infer(lk, rng=1)      # method="auto" picks an exact engine here
print(result.info["auto_reason"])
result
```

```{code-cell} python
ax = result.plot(truth=mock.truth.rho, stack=stack)
ax.set_xlabel("z");
```

The posterior bands are wide and the median is jagged. This is not noise in
the sampler: neighbouring bins really are poorly separated by data whose
errors are as large as the bins, and the flat Dirichlet prior lets them
trade mass freely. The bins are strongly anti-correlated:

```{code-cell} python
corr = result.corr()
print("correlation of bin 10 with bins 9, 10, 11:", corr[10, 9:12].round(2))
```

```{code-cell} python
:tags: [remove-cell]

assert corr[10, 9] < -0.3 and corr[10, 11] < -0.3
```

The summary table gives per-bin numbers and convergence diagnostics:

```{code-cell} python
result.summary()
```

## Derived quantities

Because neighbouring bins are anti-correlated, sums and averages over bins
are much better constrained than single bins. Compute derived quantities on
each posterior draw:

```{code-cell} python
centers = bins.centers
zbar = result.functional(lambda rho: rho @ centers)
f_high = result.functional(lambda rho: rho[centers > 1.5].sum())

print(f"mean redshift:   {zbar.mean():.3f} ± {zbar.std():.3f}   (truth {mock.truth.rho @ centers:.3f})")
print(f"fraction z>1.5:  {f_high.mean():.3f} ± {f_high.std():.3f}   (truth {mock.truth.rho[centers > 1.5].sum():.3f})")
print(f"stacked mean z:  {stack @ centers:.3f}  (no error bar)")
```

Summing per-bin variances would give a much larger, wrong error bar on the
mean redshift:

```{code-cell} python
naive = np.sqrt(np.sum((result.std * centers) ** 2))
print(f"error ignoring correlations: {naive:.3f}   vs   correct: {zbar.std():.3f}")
```

```{code-cell} python
:tags: [remove-cell]

assert naive > 3 * zbar.std()
```

## A smoothness prior

If you expect $n(z)$ to be smooth on scales of a few bins, say so. A
Gaussian-process prior on $\log n(z)$, with its amplitude and length scale
inferred from the data, shares information between neighbouring bins:

```{code-cell} python
smooth = binference.infer(lk, binference.priors.LogisticGP(), rng=1)
hyper = smooth.info["hyperparameters"]
print({k: np.round(np.quantile(v, [0.16, 0.5, 0.84]), 2) for k, v in hyper.items()})
```

```{code-cell} python
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True, constrained_layout=True)
result.plot(ax=axes[0], truth=mock.truth.rho)
smooth.plot(ax=axes[1], truth=mock.truth.rho)
axes[0].set_title("Dirichlet prior", loc="left")
axes[1].set_title("Gaussian-process prior", loc="left")
axes[1].get_legend().remove()
for ax in axes:
    ax.set_xlabel("z")
```

```{code-cell} python
:tags: [remove-cell]

assert np.median(smooth.std / result.std) < 0.5
```

The smooth posterior is far tighter per bin and still covers the truth.
Which prior is better is a scientific question, but the data can weigh in.
Compare how well each fit predicts galaxies it has not seen:

```{code-cell} python
train, test = lk[:1500], lk[1500:]
lp = {}
for name, prior in [("Dirichlet", None), ("GP", binference.priors.LogisticGP())]:
    fit = binference.infer(train, prior, rng=2)
    lp[name] = fit.log_predictive(test).sum()
print({k: round(v, 1) for k, v in lp.items()})
```

```{code-cell} python
:tags: [remove-cell]

assert lp["GP"] > lp["Dirichlet"]
```

Higher is better. The smoothness prior predicts the held-out galaxies better.

## Individual galaxies

The population also sharpens each galaxy's redshift. Galaxy $i$'s posterior
under the inferred population, $P(z_i\in k\mid D)$, is narrower than its own
likelihood wherever the population is informative:

```{code-cell} python
from binference.plotting import plot_individual

i = int(np.argmax(mock.measurements["sigma"]))       # the galaxy with the largest error
ax = plot_individual(smooth, i)
ax.axvline(mock.truth.values[i], color="k", lw=1, ls=":", label="true z")
ax.legend(frameon=False);
```

`result.assignment_probabilities()` gives these posteriors for every
galaxy, and `result.sample_values()` gives joint draws of their true
redshifts.

## Population versus this sample

$\rho$ describes the population the 2000 galaxies were drawn from. The
redshift histogram of *these* galaxies, which a lensing kernel would need, is
a different quantity with its own uncertainty:

```{code-cell} python
counts = smooth.realized_counts(n_draws=500, rng=0)
print("sd of realised counts per bin:", counts.std(axis=0)[8:12].round(1))
print("sd of population counts per bin:", (smooth.samples * lk.W).std(axis=0)[8:12].round(1))
```

```{code-cell} python
:tags: [remove-cell]

assert np.all(counts.std(axis=0)[8:12] > (smooth.samples * lk.W).std(axis=0)[8:12])
```

Here the histogram is the less certain of the two. The smoothness prior pins
the population down, but with errors as wide as the bins it cannot say which
bin each galaxy fell in. With precise measurements the order reverses: each
galaxy's bin is then nearly known, while $\rho$ keeps the shot noise of
drawing 2000 galaxies from it.

## Next steps

- {doc}`02_inputs` shows how to feed in posteriors computed under an interim
  prior, posterior samples, and unequal bins.
- {doc}`03_engines` compares the engines.
- The [user guide](../user-guide/index) explains every option.
