# Inputs and conventions

pastry's model needs one thing from your data: for every object $i$ and bin
$k$, the **bin-averaged likelihood** $L_{ik}$, the probability density of the
object's data if its true value lies in bin $k$
([theory](../theory.md#objects-bins-and-likelihoods)). The
{class}`~pastry.Likelihoods` class holds these values together with their
{class}`~pastry.Bins`, optional weights and an optional selection function. Its
constructors convert the usual data products into likelihoods.

```python
import numpy as np
import pastry

rng = np.random.default_rng(1)
```

## Bins

```python
bins = pastry.Bins.linear(0.0, 3.0, 30)          # 30 equal bins on [0, 3]
logbins = pastry.Bins.log(1e-2, 1e2, 20)          # equal in log
irregular = pastry.Bins([0.0, 0.2, 0.5, 1.0, 3.0])  # explicit edges
grid = pastry.Bins.grid(np.linspace(-0.5, 2.0, 26), np.linspace(-5.0, 15.0, 41))  # 2-D
classes = pastry.Bins.categories(["star", "galaxy", "quasar"])
print(bins, grid.shape, grid.K, classes.labels)
```

Bins know their `edges`, `centers`, `volumes` (widths in 1-D) and how to
`index` values. N-D grids are flattened in C order: use `unflatten` and
`flatten` to move between the `(K,)` vector and the grid. Categories have unit
volumes and no geometry.

## Likelihood matrices

If you already have $L_{ik}$, for example a likelihood evaluated at bin
centres, pass it directly:

```python
z_true = rng.uniform(0.2, 2.8, size=500)
sigma = rng.uniform(0.05, 0.3, size=500)
z_obs = z_true + sigma * rng.standard_normal(500)

L = np.exp(-0.5 * ((bins.centers[None, :] - z_obs[:, None]) / sigma[:, None]) ** 2)
lk = pastry.Likelihoods(L, bins)
print(lk)
```

- **Row scaling is irrelevant.** Each row can carry any positive constant.
  pastry rescales rows internally and keeps the scales separately
  (`lk.log_scale`) so absolute log-likelihoods remain available.
- **Log space.** Pass `log=True` (or use `Likelihoods.from_log`) for
  log-likelihoods; `-inf` marks impossible bins.
- **Representation.** `representation="density"` (the default) means each
  value is the likelihood *at* (or averaged over) the bin.
  `"mass"` means the likelihood *integrated* over the bin, such as CDF
  differences; it is divided by the bin volume. This only matters for unequal
  bins, where it matters a lot (see {doc}`pitfalls`).
- **Sparse matrices.** SciPy sparse matrices are kept sparse.
  `lk.to_sparse(rtol)` drops entries below `rtol` times the row maximum.
- **Weights.** `weights=` gives each object a multiplicity. `lk.compress()`
  merges identical rows into weighted rows: inference is unchanged, and it
  can be much faster for discrete data such as binned counts.
- **Objects with no support.** A row of zeros means the object cannot be in
  any bin. That is an error, unless you pass `drop_empty=True` to drop such
  objects with a warning.

## Gaussian measurements

For a measurement $\hat z_i\pm\sigma_i$ the bin-averaged likelihood is exact,
computed with stable log-CDF differences that work far into the tails:

```python
lk = pastry.Likelihoods.from_gaussian(z_obs, sigma, bins)
print(lk.N, lk.K)
```

## Posteriors computed under an interim prior

Most photometric-redshift codes, classifiers and per-object fits report
**posteriors**, which already include a prior. Divide that **interim prior**
out to recover likelihoods. pastry requires you to say what it was:

```python
interim = pastry.simulate.population(bins, lambda z: np.exp(-0.5 * ((z - 1.2) / 0.8) ** 2))
P = L * interim                      # posteriors under the interim prior (rows unnormalised)
P /= P.sum(axis=1, keepdims=True)

lk_post = pastry.Likelihoods.from_posteriors(P, bins, interim_prior=interim)
np.testing.assert_allclose(lk_post.toarray(), pastry.Likelihoods(L, bins).toarray(), atol=1e-12)
```

`interim_prior` can be:
- `"flat"`, a uniform density in the bin coordinate;
- an array of bin masses (or densities, with `interim_representation="density"`);
- a callable density, integrated over each bin.

Posteriors themselves can be masses per bin (the default) or densities at
bin centres (`representation="density"`). pastry warns if the rows do not
look like the representation you declared.

## Posterior samples

Samples from per-object fits (MCMC chains, nested-sampling output) are
histogrammed onto the bins and divided by the interim prior. Ragged lists
(different numbers of samples per object) and importance weights are
supported:

```python
samples = [z + s * rng.standard_normal(200) for z, s in zip(z_obs[:100], sigma[:100])]
lk_samp = pastry.Likelihoods.from_samples(samples, bins, interim_prior="flat")
print(lk_samp.is_sparse, np.median(lk_samp.meta["n_eff"]))
```

The per-object effective sample size and the fraction of samples outside the
bins are stored in `lk_samp.meta`. pastry warns if most objects have fewer
effective samples than `min_samples` (default 20); histogrammed likelihoods
are then noisy.

## Likelihood functions and distributions

For an arbitrary likelihood, give a function of the bin coordinate; it is
averaged over each bin by Gauss–Legendre quadrature. SciPy frozen
distributions work too:

```python
from scipy import stats

def loglike(z):  # (n_points, ndim) or (n_points,) -> (N, n_points)
    z = np.asarray(z).reshape(-1)
    return stats.t.logpdf(z[None, :], df=3, loc=z_obs[:, None], scale=sigma[:, None])

lk_t = pastry.Likelihoods.from_function(loglike, bins, log=True)
lk_d = pastry.Likelihoods.from_distributions(
    [stats.norm(m, s) for m, s in zip(z_obs[:10], sigma[:10])], bins
)
print(lk_t.N, lk_d.N)
```

## Response matrices and classifiers

Two adapters cover the other common data products.

- **Unfolding.** `from_response(counts, response, bins=bins)` takes observed counts
  in reconstructed bins and a response matrix $R_{bk}=p(\text{reco }b\mid\text{true }k)$.
  It builds one weighted likelihood row per reconstructed bin, and attaches
  the efficiency $\sum_bR_{bk}$ as a selection function with mode
  `"observed"`.
- **Quantification.** `from_classifier(probs, train_prior)` turns classifier
  probabilities $p(k\mid x_i)$ into likelihoods by dividing out the training
  class prevalence.

```python
mu = pastry.simulate.mock_unfolding(3000, rng=2)
lk_unfold = pastry.Likelihoods.from_response(
    mu.measurements["reco_counts"], mu.truth.extra["response"], bins=mu.bins
)
mc = pastry.simulate.mock_categorical(1000, 4, rng=3)
lk_class = pastry.Likelihoods.from_classifier(mc.posteriors, mc.interim_prior)
print(lk_unfold.selection.mode, lk_class.bins.labels)
```

## Working with `Likelihoods`

```python
lk = pastry.Likelihoods.from_gaussian(z_obs, sigma, bins)
sub = lk[:100]                       # subsets keep bins, weights and selection
dense = lk.toarray()                 # normalised values (each row's max is 1)
stack = lk.stack()                   # naive stacked estimate, for comparison only
print(sub.N, dense.shape, stack.sum().round(6))
```

`lk.stack()` is the familiar "stacked PDF". It is exactly one EM step from
the interim prior, and it is biased whenever measurement errors are
comparable to the features of the population. It is there for comparison,
not for inference.
