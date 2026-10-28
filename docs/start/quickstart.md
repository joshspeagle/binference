---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Quick start

This page takes you from two arrays, a measured value and its error for each object,
to a population with its uncertainty. It needs binference with the `plot` extra
({doc}`install`). If your data are posteriors, samples, counts or classifier outputs,
{doc}`is-it-right` names the constructor to use instead.

## Your data

Here a simulation stands in for your data: 2000 galaxies with photometric redshifts
`z_obs`: distances estimated from each galaxy's colours. Their Gaussian errors `z_err`
lie between 0.05 and 0.2. Choose bins that cover the range of the true values: here 15
bins of width 0.2 on $[0, 3]$. Replace the simulation with your own arrays.

```{code-cell} python
import binference

bins = binference.Bins.linear(0.0, 3.0, 15)
mock = binference.simulate.mock_1d(2000, bins, rng=42)
z_obs, z_err = mock.measurements["mean"], mock.measurements["sigma"]
```

## Likelihoods

`from_gaussian` turns each measurement into a {term}`bin-averaged likelihood`: the
probability of the measurement if the true value lay in each bin.

```{code-cell} python
lk = binference.Likelihoods.from_gaussian(z_obs, z_err, bins)
lk
```

## Infer the population

One call infers the {term}`population fractions` $\rho$. The default engine,
`method="auto"`, chooses an exact sampler and records why.

```{code-cell} python
result = binference.infer(lk, rng=1)
result.info["auto_reason"]
```

`infer` checks convergence after the fit and warns if the chains disagree or give too
few effective draws. No warning appeared here.

## Read the summary

```{code-cell} python
result.summary()
```

The title names the {term}`engine`, the number of {term}`draws <draw>` and the kind of
input. The table has one row per bin:

| Column | Meaning |
|---|---|
| `bin` | the bin's edges |
| `mean`, `sd` | the posterior mean and standard deviation of the bin's fraction $\rho_k$ |
| `median` | the posterior median of $\rho_k$ |
| `q16`, `q84` | the ends of the central 68% {term}`credible interval`; `result.summary(level=0.9)` gives 90% |
| `ess_bulk` | the {term}`effective sample size`: how many independent draws the correlated draws are worth |
| `rhat` | the split {term}`R-hat`: close to one when the chains agree |
| `support` | the {term}`data support`: how many objects' worth of likelihood falls in the bin |

The {doc}`../reference/defaults` page gives the R-hat and ESS limits that `check()`
applies. In the last bin, the prior sets the fraction: see {term}`data support`.

## Plot it

```{code-cell} python
---
mystnb:
  image:
    alt: >-
      The inferred population density against redshift from 0 to 3: an indigo step
      curve, the posterior median, inside nested indigo bands for the 68% and 95%
      intervals. It rises to a peak near z = 0.5, stays high to z = 2 and falls to
      nearly zero above z = 2.4. The dashed true density lies inside the bands.
  figure:
    caption: >-
      The population inferred from 2000 simulated galaxies: the posterior median
      density with its 68% and 95% bands, and the true density (dashed).
---
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
fig.set_figheight(4.4)
result.plot(ax=ax, truth=mock.truth.rho)
ax.get_legend().set(loc="lower left", bbox_to_anchor=(0, 1))  # above the plot
ax.set_xlabel("z");
```

Your own data have no `truth`; leave it out.

## Use the draws

Every quantity computed from the population has an uncertainty. `functional`
evaluates a function on each posterior draw, here the mean redshift:

```{code-cell} python
zbar = result.functional(lambda rho: rho @ bins.centers)
print(f"mean redshift: {zbar.mean():.3f} ± {zbar.std():.3f}")
```

Each galaxy's posterior under the population, its {term}`hierarchical posterior`, is
narrower than its own likelihood wherever the population is informative. Here is the
first galaxy's, rounded:

```{code-cell} python
result.assignment_probabilities()[0].round(2)
```

## Next

- {doc}`../tutorials/gaussian-quickstart` goes further with a similar mock: stacking, correlated bins,
  smoothness priors and the {term}`realised sample`.
- The {doc}`results guide <../guides/results>` lists everything a result holds.
- The {doc}`engines guide <../guides/engines>` explains how `method="auto"` chooses.
