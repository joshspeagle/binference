# Posteriors under an interim prior

Use this page if your code reports a **posterior** for each object. Most
photometric-redshift codes, classifiers and per-object fits do. A posterior already
includes a prior, the {term}`interim prior` $\pi(z)$. binference divides it out to
recover the likelihoods, so you must say what it was.

```{warning}
Never pass posteriors as likelihoods. The fitted population is then biased toward the
interim prior, often by many standard deviations
([why](../../how-it-works/theory.md#likelihoods-versus-posteriors)).
```

## A table of posteriors

Pass the posterior masses per bin $P_{ik}$ and the interim prior to `from_table`:

```python
import numpy as np
import binference

rng = np.random.default_rng(1)
bins = binference.Bins.linear(0.0, 3.0, 30)
z_obs = rng.uniform(0.2, 2.8, size=500)
sigma = rng.uniform(0.05, 0.3, size=500)
L = np.exp(-0.5 * ((bins.centers[None, :] - z_obs[:, None]) / sigma[:, None]) ** 2)

interim = binference.simulate.population(bins, lambda z: np.exp(-0.5 * ((z - 1.2) / 0.8) ** 2))
P = L * interim                          # posteriors under the interim prior
P /= P.sum(axis=1, keepdims=True)

lk = binference.Likelihoods.from_table(P, bins, interim=interim)
print(np.allclose(lk.toarray(), binference.Likelihoods(L, bins).toarray()))
```

A posterior table defaults to masses per bin, `representation="mass"`. binference warns
if the rows do not sum to one under the representation you declared
([warnings](../../reference/warnings.md#inputwarning)).

## Describe the interim prior

The interim prior is a {class}`~binference.Density`: a fixed density over the bin
coordinate. It means the same in every constructor.

```python
D = binference.Density
D.flat()                                   # flat in the measure coordinate
D.flat("z")                                # flat in z
D.from_masses(interim)                     # probability mass per bin
D.from_densities(interim / bins.volumes)   # density per unit z in each bin
D.from_function(lambda z: np.exp(-z))      # a function, integrated over each bin
```

`D.from_distribution(dist)` takes a frozen `scipy.stats` distribution. In place of a
`Density` you can pass `"flat"`, a function, a distribution or an array.

An array alone is accepted only where {term}`masses <mass>` and densities agree: on
categories and on bins of equal volume. On unequal bins, add
`interim_representation="mass"` or `"density"`.

On `Bins.log`, `D.flat()` is flat in $\log z$: equal mass per decade. If your code used
a prior flat in $z$, pass `D.flat("z")`.

## Classifier probabilities

A classifier's class probabilities are posteriors. Their interim prior is the class
mix the classifier learned from its training set. Pass it as `interim=` on
{meth}`Bins.categories <binference.Bins.categories>`. The result is right only if the
probabilities are calibrated. {doc}`binned` gives a route that does not need
calibration.

## Stacking is not the answer

`lk.stack()` is the familiar "stacked PDF": the mean of the per-object posteriors under
the interim prior. For a posterior table it gives back the mean of your input rows. It
is one EM step from the interim prior, and it is biased whenever errors are comparable
to the features of the population ({term}`stacking`). Use it only for comparison.
The {term}`KKT gap` says how far a population is from the maximum-likelihood one; it is
zero at the optimum:

```python
model = binference.PopulationModel(lk)
fit = binference.infer(lk, method="map")
print(model.kkt_gap(lk.stack()), model.kkt_gap(fit.rho))
```

The {doc}`inputs tutorial </tutorials/02_inputs>` shows the bias of a fit that treats
posteriors as likelihoods.
