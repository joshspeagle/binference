---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Gaussian measurements: a redshift distribution

This tutorial infers the redshift distribution of 2000 galaxies. Each galaxy has a
photometric redshift with a Gaussian error of its own. The tutorial compares the result
with {term}`stacking` the galaxies' likelihoods. It then derives quantities with error
bars, and shows what the population says about single galaxies.

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

Stacking adds up the galaxies' likelihoods and gives no error bars. It is the first step
of EM, which converges to the most likely population
([theory](../how-it-works/point-estimates.md#em-is-iterated-stacking)).
binference samples the posterior of the {term}`population fractions` $\rho_k$ instead,
here with {term}`NUTS`:

```{code-cell} python
stack = lk.stack()
result = binference.infer(lk, rng=1, n_samples=500)   # method="auto" picks the engine
print(result.info["auto_method"], "-", result.info["auto_reason"])
result
```

```{code-cell} python
:tags: [remove-cell]

assert result.info["auto_method"] == "nuts"
```

```{code-cell} python
---
mystnb:
  figure:
    caption: The population posterior under a flat Dirichlet prior, with the truth
      (dashed) and the stack (dotted). The bands hold 68% and 95% of the posterior.
  image:
    alt: Step plot of density against redshift. The posterior median is jagged and its
      bands are wide. It follows the dashed truth, which has a broad peak near 0.6 and a
      narrower one near 1.6. The dotted stack is smoother and broader than the truth.
---
fig, ax = plt.subplots(figsize=(6, 4.2))
result.plot(ax=ax, truth=mock.truth.rho, stack=stack, sample_color=OBJECT, label="median")
ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1))    # above the plot
ax.set_xlabel("z");
```

The bands are wide and the median is jagged. The errors are as wide as the bins, so the
data barely tell neighbouring bins apart. Under the flat {term}`Dirichlet prior`, mass
moves freely between those bins, so they are strongly anti-correlated:

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

If you expect the population to be smooth over a few bins, a {term}`smoothness prior`
narrows the bands; {doc}`priors-and-smoothness` compares the priors.

## Single galaxies

The population also narrows each galaxy's redshift. Its {term}`hierarchical posterior`
$P(k_i=k\mid D)$ combines its own likelihood with the inferred population. It is
narrower than the likelihood wherever the population changes across the likelihood's
width. `assignment_probabilities()` returns these posteriors for every galaxy. The
measure of narrowing here is the ratio of the spreads of the two over the bins:

```{code-cell} python
def spread(p):                       # standard deviation of each row over the bin centres
    return np.sqrt(p @ centers**2 - (p @ centers) ** 2)

own = L * bins.volumes
own /= own.sum(axis=1, keepdims=True)
post = result.assignment_probabilities()
ratio = spread(own) / spread(post)
broad = mock.measurements["sigma"] > 0.15           # galaxies with large errors
print(f"median narrowing, galaxies with errors above 0.15: {np.median(ratio[broad]):.2f}")
```

The effect is modest: the median ratio is 1.06, so the spread shrinks by about 6%. Here
is a typical broad galaxy. It is the one closest to the median narrowing whose true
redshift lies inside the central 68% of its posterior:

```{code-cell} python
---
mystnb:
  figure:
    caption: A typical galaxy's own likelihood (dashed) and its hierarchical posterior
      (solid). The dotted line marks its true redshift.
  image:
    alt: Two step curves over redshift that nearly coincide. The dashed coral likelihood
      is broad and smooth. The solid indigo posterior is slightly narrower and follows
      the steps of the population. The dotted line of the true redshift lies near the
      middle of both.
---
from binference.plotting import plot_individual

cdf = np.cumsum(post, axis=1)
k_true = bins.index(mock.truth.values)

def inside_68(j):                    # the true bin overlaps the central 68% of the posterior
    below = cdf[j, k_true[j] - 1] if k_true[j] > 0 else 0.0
    return below < 0.84 and cdf[j, k_true[j]] > 0.16

candidates = np.flatnonzero(broad)[np.argsort(np.abs(ratio[broad] - np.median(ratio[broad])))]
i = int(next(j for j in candidates if inside_68(j)))
print(f"galaxy {i}: narrowing {ratio[i]:.2f}")

fig, ax = plt.subplots(figsize=(6, 4.2))
plot_individual(result, i, ax=ax, sample_color=OBJECT)
ax.axvline(mock.truth.values[i], color="k", lw=1, ls=":", label="true z")
ax.set_xlabel("z")
ax.legend(loc="upper right");
```

```{code-cell} python
:tags: [remove-cell]

assert 1.0 < np.median(ratio[broad]) < 1.2 and inside_68(i)
```

The effect grows where the population has sharp features, as in {doc}`empirical-bayes`.
`result.sample_values()` gives joint draws of the galaxies' true redshifts.

$\rho$ describes the population that the galaxies were drawn from; the redshift histogram
of these 2000 galaxies, the {term}`realised sample`, is a different quantity
([population versus realised sample](../guides/results.md#population-versus-realised-sample)).

## Next steps

- {doc}`priors-and-smoothness` compares priors on a sparser catalogue.
- {doc}`posteriors-on-a-grid` and {doc}`posterior-samples` start from posteriors
  instead of likelihoods.
- {doc}`choosing-an-engine` compares the engines.
- The [guides](../guides/index) explain every option.
