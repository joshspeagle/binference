# Choose a prior

The prior on the population matters most when the data say least: many bins, few
objects, or measurement errors as wide as the bins. Start with the default. Change it
when the checks at the end of this page say the prior, not the data, sets the answer.
What each prior means is in the [theory](../how-it-works/theory.md#priors).

| Your situation | Prior to try |
|---|---|
| no strong reason to choose | `Dirichlet(1.0)`, the default |
| about as many bins as objects | `Dirichlet.reference()` |
| a jagged fit, with errors as wide as the bins | `LogisticGP()` or `LogisticRandomWalk(order=2)` |
| a reference sample of the same population | `Dirichlet.from_counts(counts, strength)` |
| unequal bins, with a Dirichlet | `Dirichlet.from_base_measure(...)` |
| expected numbers of objects, with prior information on them | `GammaRates(shape, rate)` |

binference's priors fall into three families. The {term}`Dirichlet prior` family is
{term}`conjugate <conjugate prior>`, so every engine takes it. The
{term}`logistic priors <logistic prior>` add smoothness across neighbouring bins; Gibbs
and VB do not accept them. The {term}`Gamma rates prior` models expected counts rather than
fractions.

```python
import numpy as np
import binference
from binference import priors

mock = binference.simulate.mock_1d(1000, binference.Bins.linear(0, 3, 30), noise=(0.1, 0.3), rng=4)
lk = mock.likelihoods
```

Pass a prior as the second argument of {func}`~binference.infer`, or as `prior=`
to {class}`~binference.PopulationModel` and {class}`~binference.PopulationEstimator`.

## The Dirichlet family

```python
flat = priors.Dirichlet(1.0)                         # the default
sparse = priors.Dirichlet.reference()                # concentration 1/K per bin
by_volume = priors.Dirichlet.from_base_measure(5.0)  # concentration in proportion to bin width
ref_counts = np.bincount(mock.truth.labels[:50], minlength=30)
informed = priors.Dirichlet.from_counts(ref_counts, strength=0.5)

for p in (flat, sparse, informed):
    print(p, binference.infer(lk, p, method="map").rho[:3].round(4))
```

- **`Dirichlet(1.0)`**: use it unless you have a reason not to.
- **`Dirichlet.reference()`**: use it when the number of bins $K$ is comparable to the
  number of objects $N$. There the default biases the fit toward equal fractions.
- **`Dirichlet.from_base_measure(a)`**: use it on unequal bins, so the prior means the
  same density however you bin.
- **`Dirichlet.from_counts(counts, strength)`**: use it when you have the counts
  $n^{\rm ref}_k$ of a reference sample. `strength` is $\kappa$: 1 trusts the sample
  fully, and smaller values trust it less.

`GeneralizedDirichlet`, `DirichletTree` and `PolyaTree` stay conjugate, with more
freedom in how bins co-vary:

```python
gd = priors.GeneralizedDirichlet(a=2.0, b=6.0)        # stick-breaking Betas
tree = priors.DirichletTree([[0, 1, 2], [3, [4, 5]]], alpha=2.0)
polya = priors.PolyaTree(c=2.0)                       # dyadic, multi-scale
print(binference.infer(lk, polya, method="map").rho[:3].round(4))
```

- **`DirichletTree`**: use it when groups of bins belong together, such as subclasses
  of a class. Pass the tree as nested lists of bin indices.
- **`PolyaTree`**: use it on an ordered grid for a conjugate prior with structure at
  several scales.

## Smoothness: the logistic family

If the Dirichlet fit is jagged and the errors are as wide as the bins, use a
{term}`smoothness prior`. It infers how smooth the population is:

```python
rw = priors.LogisticRandomWalk(order=2)
gp = priors.LogisticGP(kernel="matern32")
print(binference.infer(lk, gp, method="map").rho[:3].round(4))
```

| prior | use it when | hyperparameters (inferred by default) |
|---|---|---|
| `LogisticRandomWalk(order=1)` | neighbouring bins should be close | `scale` |
| `LogisticRandomWalk(order=2)` | the population follows smooth trends | `scale` |
| `LogisticGP(kernel=...)` | you want a stationary Gaussian process: `"matern12"`, `"matern32"`, `"matern52"`, `"rbf"` | `amplitude`, `length_scale` (one per axis) |
| `LogisticNormal(mean, cov)` | you have a covariance, or want independent bins | `scale` |

- **Grids.** All four work on 1-D bins and on grids. Only `LogisticNormal` accepts
  masked grids and categories.
- **Rough populations.** If a `LogisticGP` fit looks too smooth, raise `n_basis`
  ({doc}`defaults <../reference/defaults>`).
- **Hyperpriors.** Give a {term}`hyperparameter` a {term}`hyperprior`, or a number to
  fix it:
  ```python
  gp_fixed = priors.LogisticGP(amplitude=1.5, length_scale=priors.LogNormal(np.log(0.5), 0.5))
  rw_tight = priors.LogisticRandomWalk(order=2, scale=priors.HalfNormal(0.3))
  ```
- **A reference population.** With `reference` its fraction in each bin,
  `mean=np.log(reference)` centres the prior on it.

## Rates instead of fractions

`GammaRates(shape, rate)` puts independent Gamma priors on the **expected
number** of objects per bin, $\lambda_k$. The data then form a Poisson
process. Every fit of fractions already gives the total and the rates, under a
scale-free prior on the total ({doc}`selection`). Use `GammaRates` when the rates need
their own prior, for example bin-specific shapes or an informative total. Combine it
with an exposure or a selection function. Occurrence rates per star, or counts per unit
time or area, are estimated this way.

```python
est = binference.infer(lk, priors.GammaRates(shape=1.0, rate=1e-3), method="map")
print(est.rates.sum().round(1), est.rho.sum())
```

Sampled results hold both `rates`, the draws of $\lambda$, and `samples`, the fractions
they imply. NUTS and Gibbs both sample rates.

## Check a prior

Draw from the prior to see which populations it allows before any data:

```python
model = binference.PopulationModel(lk, gp)
draws = model.sample_prior(5, rng=0)       # five populations from the prior
print(draws.shape, np.allclose(draws.sum(axis=1), 1.0))
```

After a fit, two numbers say how much the prior sets the answer. A bin with little
{term}`data support` takes its fraction from the prior ({doc}`bins`).
`result.prior_information_ratio()` compares the prior's information with the data's.
It is small when the data dominate, and near one or above when the prior does.

```python
fit = binference.infer(lk, rw, method="map")
print(round(fit.prior_information_ratio(), 3))
```

To choose between priors, fit each and compare the
{term}`log predictive density <log predictive density>` of held-out objects
({doc}`results`). {func}`binference.simulate.prior_predictive` simulates whole
catalogues from the prior, which is what simulation-based calibration needs
({doc}`checking`).
