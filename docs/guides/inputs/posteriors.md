# Posteriors under an interim prior

Use this page if your code reports a **posterior** for each object. Most
photometric-redshift codes, classifiers and per-object fits do. A posterior already
includes a prior, the {term}`interim prior` $\pi(z)$. binference divides it out to
recover the likelihoods, so you must say what it was.

```{warning}
Never pass posteriors as likelihoods. The fitted population is then biased toward the
interim prior, often by many standard deviations
([why](../../how-it-works/model.md#likelihoods-versus-posteriors)).
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

## Photometric-redshift posteriors from qp

A [`qp` histogram ensemble](https://qp.readthedocs.io/en/stable/user_guide/parameterizations/hist.html)
holds one redshift PDF per object. Evaluate its CDF at the bin edges and difference
adjacent values to obtain each bin's posterior mass. This accounts for unequal bin
widths. Choose edges that cover the posterior support, and compute the interim prior
masses on those same edges using the prior from the photo-$z$ fit.

Install the optional package with `pip install qp-prob`. This example uses two
histograms with densities per unit redshift and checks the conversion with two
choices of interim prior.

```python skip
import binference
import numpy as np
import qp

edges = np.array([0.0, 1.0, 3.0])
bins = binference.Bins(edges)
ensemble = qp.hist.create_ensemble(
    bins=edges, pdfs=np.array([[0.50, 0.25], [0.20, 0.40]])
)
posterior_mass = np.diff(ensemble.cdf(edges), axis=-1)
np.testing.assert_allclose(posterior_mass, [[0.50, 0.50], [0.20, 0.80]], rtol=1e-14)

for prior_mass, expected_ratio in (([0.5, 0.5], [1.0, 0.25]), ([0.25, 0.75], [3.0, 0.75])):
    qp_likelihoods = binference.Likelihoods.from_table(
        posterior_mass,
        bins,
        interim=prior_mass,
        representation="mass",
        interim_representation="mass",
    )
    rows = qp_likelihoods.toarray()
    np.testing.assert_allclose(rows[:, 0] / rows[:, 1], expected_ratio, rtol=1e-14)
```

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

The {doc}`inputs tutorial </tutorials/posteriors-on-a-grid>` shows the bias of a fit that treats
posteriors as likelihoods.
