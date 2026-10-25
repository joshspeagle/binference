---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Inputs and conventions: posteriors, samples and unequal bins

Most surveys do not publish likelihoods. They publish **posteriors**,
computed under some prior, or **samples** from per-object fits. This
tutorial shows how to use those correctly, and what happens otherwise. It
also shows why unequal bins need care.

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np

import binference

plt.rcParams["figure.dpi"] = 110
bins = binference.Bins.linear(0.0, 3.0, 20)
```

## Posteriors under an interim prior

`mock_1d(interim_prior="gaussian")` also returns what a photo-z code would
report: per-object posteriors computed under a broad Gaussian prior centred
at $z=1.8$, which is quite different from the true population.

```{code-cell} python
mock = binference.simulate.mock_1d(2000, bins, noise=(0.2, 0.4), interim_prior="gaussian", rng=3)
P, interim = mock.posteriors, mock.interim_prior
print(P.shape, "rows sum to", P.sum(axis=1)[:3].round(6))
```

The correct way divides the interim prior out. The tempting way feeds the
posteriors in as if they were likelihoods:

```{code-cell} python
kw = dict(rng=1, n_samples=500, n_warmup=500)   # shorter runs than the default, for speed
right = binference.infer(binference.Likelihoods.from_posteriors(P, bins, interim), **kw)
wrong = binference.infer(binference.Likelihoods(P, bins), **kw)
```

```{code-cell} python
fig, ax = plt.subplots(figsize=(7.5, 4))
right.plot(ax=ax, truth=mock.truth.rho, label="posteriors ÷ interim prior")
wrong.plot(ax=ax, color="#eb6834", label="posteriors used as likelihoods")
ip = interim / bins.volumes
ax.step(bins.edges[0], np.append(ip, ip[-1]), where="post", color="0.5", lw=1, label="interim prior")
ax.legend(frameon=False, fontsize=8)
ax.set_xlabel("z");
```

Using posteriors directly double-counts the interim prior, and the
population is dragged toward where that prior is large. The error bars do
not warn you: the wrong answer is just as confident.

```{code-cell} python
z = (wrong.mean - mock.truth.rho) / wrong.std
print("largest bias of the wrong fit:", np.abs(z).max().round(1), "posterior sd")
z = (right.mean - mock.truth.rho) / right.std
print("largest deviation of the right fit:", np.abs(z).max().round(1), "posterior sd")
```

## Posterior samples

Per-object MCMC or nested-sampling runs give samples rather than binned
PDFs. {meth}`~binference.Likelihoods.from_samples` histograms them onto the bins
and divides out the prior they were drawn under:

```{code-cell} python
mock_s = binference.simulate.mock_1d(2000, bins, noise=(0.2, 0.4), interim_prior="gaussian",
                                 n_samples=100, rng=3)
lk_s = binference.Likelihoods.from_samples(mock_s.samples, bins, mock_s.interim_prior)
print(lk_s, "median effective samples per object:", np.median(lk_s.meta["n_eff"]))
```

Histogrammed likelihoods are noisy: with 100 samples spread over a few bins,
each entry has Poisson noise. The estimate stays consistent, but the noise
adds uncertainty that the model does not know about. Compare with a fit to
the exact likelihoods of the same objects:

```{code-cell} python
exact = binference.infer(mock_s.likelihoods, **kw)
from_s = binference.infer(lk_s, **kw)
shift = np.abs(exact.mean - from_s.mean) / exact.std
print("largest shift between the two fits:", shift.max().round(2), "posterior sd")
```

Use more samples per object, or coarser bins, when the median effective
sample size is small. binference warns below 20.

## Unequal bins: masses versus densities

The model needs bin-*averaged* likelihoods. A likelihood or posterior
*integrated* over each bin (a CDF difference, a histogram) is a **mass** and
must be divided by the bin width. With equal bins this cannot matter. With
unequal bins it does:

```{code-cell} python
from scipy import stats

ubins = binference.Bins([0.0, 0.25, 0.5, 1.0, 2.0, 3.0])
rho_true = np.array([0.15, 0.25, 0.3, 0.2, 0.1])
umock = binference.simulate.mock_1d(5000, ubins, density=rho_true, noise=0.15, rng=4)
mu, sig = umock.measurements["mean"], umock.measurements["sigma"]

# posterior masses under a flat prior: CDF differences across each bin
edges = ubins.edges[0]
cdf = stats.norm.cdf((edges[None, :] - mu[:, None]) / sig[:, None])
masses = np.diff(cdf, axis=1)

as_masses = binference.infer(binference.Likelihoods.from_posteriors(masses, ubins, "flat"),
                         method="map").rho
as_densities = binference.infer(binference.Likelihoods(masses, ubins), method="map").rho
print("truth          ", rho_true)
print("masses (right) ", as_masses.round(3))
print("densities      ", as_densities.round(3))
print("bin widths     ", ubins.volumes)
```

Treating masses as densities distorts the population: here it moves mass
from a narrow bin into its neighbours, by far more than the statistical
error. `from_posteriors` takes
`representation="mass"` by default; pass `representation="density"` for PDFs
evaluated at bin centres. binference warns when the rows do not look like the
representation you declared.

## Other inputs

The same idea covers every adapter: say what your numbers are, and binference
converts them to likelihoods.

| your data | constructor |
|---|---|
| likelihood values or log-likelihoods on bins | `Likelihoods(L, bins)`, `Likelihoods.from_log` |
| Gaussian measurement and error | `Likelihoods.from_gaussian(mean, sigma, bins)` |
| any likelihood function or SciPy distribution | `from_function`, `from_distributions` |
| posteriors on bins plus the interim prior | `from_posteriors(P, bins, interim_prior)` |
| posterior samples plus the interim prior | `from_samples(samples, bins, interim_prior)` |
| counts and a response matrix (unfolding) | `from_response(counts, R, bins=bins)` |
| classifier probabilities and training prevalence | `from_classifier(probs, train_prior)` |

The last two are covered in {doc}`06_categories_unfolding`.
