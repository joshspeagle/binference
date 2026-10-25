# Inputs and conventions

binference's model needs one thing from your data: for every object $i$ and bin
$k$, the **bin-averaged likelihood** $L_{ik}$, the probability density of the
object's data if its true value lies in bin $k$
([theory](../theory.md#objects-bins-and-likelihoods)). The
{class}`~binference.Likelihoods` class holds these values together with their
{class}`~binference.Bins`, optional weights and an optional selection function. Its
constructors convert the usual data products into likelihoods.

```python
import numpy as np
import binference

rng = np.random.default_rng(1)
```

## Bins

```python
bins = binference.Bins.linear(0.0, 3.0, 30)          # 30 equal bins on [0, 3]
logbins = binference.Bins.log(1e-2, 1e2, 20)          # equal in log
irregular = binference.Bins([0.0, 0.2, 0.5, 1.0, 3.0])  # explicit edges
grid = binference.Bins.grid(np.linspace(-0.5, 2.0, 26), np.linspace(-5.0, 15.0, 41))  # 2-D
classes = binference.Bins.categories(["star", "galaxy", "quasar"])
print(bins, grid.shape, grid.K, classes.labels)
```

Bins know their `edges`, `centers`, `volumes` (widths in 1-D) and how to
`index` values. N-D grids are flattened in C order: use `unflatten` and
`flatten` to move between the `(K,)` vector and the grid. Categories have unit
volumes and no geometry.

### Measures: flat in which coordinate?

Within a bin the population is taken to be flat in some coordinate
$u = T(z)$, the bin axis's *measure*. `Bins.linear` and explicit edges use
$u = z$; `Bins.log` uses $u = \log z$, so a log bin's centre is the geometric
mean of its edges. Pass `measure=` to choose: `"linear"`, `"log"`, or a
`binference.Measure` built from a transform and its inverse or from a positive
density $g(z)$ (the map is then its cumulative integral, computed on a fine grid).
Hand-built edges that are geometrically spaced raise an `InputWarning` unless a
measure is given.

```python
sqrt_bins = binference.Bins([0.0, 1.0, 4.0, 9.0],
                            measure=binference.Measure.transform(np.sqrt, np.square))
custom = binference.Bins(np.linspace(1.0, 10.0, 6),
                         measure=binference.Measure.density(lambda z: 1 / z, 1.0, 10.0))
print(sqrt_bins.centers, sqrt_bins.u_edges[0])
```

### Masked grids

A `mask` removes cells from the support, for example the unphysical corner of
a colour–magnitude diagram. Masked cells take no part in the model: `K` counts
the active cells, `index` returns `-1` for values in a masked cell, and
`unflatten` fills masked cells with NaN.

```python
mask = np.ones((4, 5), dtype=bool)
mask[0, 0] = False
masked = binference.Bins.grid(np.linspace(0, 4, 5), np.linspace(0, 5, 6), mask=mask)
print(masked.K, masked.shape, masked.index([[0.5, 0.5], [0.5, 1.5]]))
```

Smoothness priors (`LogisticRandomWalk`, `LogisticGP`) do not yet accept
masked grids.

### Within-bin shape and internal resolution: `Basis`

By default the population is flat (in $u$) within each bin, which biases the
fractions when it changes over a bin: for a population falling as $e^{-1.5z}$
on $[0, 3]$ in 10 bins, with Gaussian errors $\sigma = 0.2$, the error
reaches 7% of a bin's mass even with unlimited data. A `binference.Basis` separates the bins you report in from the
components the model fits. The population is a mixture
$f(z) = \sum_k w_k \phi_k(z)$ of components of unit mass on the bins refined
`resolution` times per axis; `order=0` makes each component flat on its cell,
`order=1` a linear B-spline (a hat centred on its cell, so the density is
continuous). Fractions are reported in the bins,
$\rho_j = \sum_k A_{jk} w_k$, with $A_{jk}$ the mass of component $k$ in bin
$j$ (`basis.matrix`).

Pass the basis wherever bins go. Constructors that know the likelihood within
bins (`from_gaussian`, `from_density`, `from_samples`) integrate it against
each component on `oversample` quadrature cells per component (8 for hats);
per-bin inputs (arrays, `from_table`, `from_log`, and `lk.on_basis(basis)` for
anything else, such as `from_binned`) take the likelihood as flat within each
bin. A selection function given as a callable is averaged with each component
as the weight; arrays per bin are mapped with $A$.

```python
bins = binference.Bins.linear(0.0, 3.0, 10)
basis = binference.Basis(bins, order=1, resolution=4)
mock = binference.simulate.mock_1d(500, bins, rng=1)
lk = binference.Likelihoods.from_gaussian(
    mock.measurements["mean"], mock.measurements["sigma"], basis
)
res = binference.infer(lk, binference.priors.LogisticRandomWalk(order=2), method="map")
print(lk.K, basis.C, res.rho.shape)  # 40 components, reported in 10 bins
```

The priors see the components (`lk.bins` is `basis.components`), so a finer
resolution needs a smoothing prior: components narrower than the measurement
errors are not identified one by one, and a flat Dirichlet then rings from
component to component. In the noise-free limit of the example above, with
`LogisticRandomWalk(order=2)`, the largest error of a bin's fraction was 7.4%
with plain bins, 3.3% with hats, 0.28% with flat components at
`resolution=4` and 0.11% with hats at `resolution=4`
(`tests/test_basis.py`). Every result reports in the bins; per-object
outputs share each component among the bins by $A$, which is exact for flat
components and ignores the likelihood's variation within a hat. Gibbs label
draws (`store_labels`) index the components.

## Likelihood matrices

If you already have $L_{ik}$, pass it directly. A likelihood evaluated at bin
centres, as below, is only an approximation to the bin average $L_{ik}$; for narrow
likelihoods it can be far off, and `from_gaussian` or `from_density` compute the
bin averages properly:

```python
z_true = rng.uniform(0.2, 2.8, size=500)
sigma = rng.uniform(0.05, 0.3, size=500)
z_obs = z_true + sigma * rng.standard_normal(500)

L = np.exp(-0.5 * ((bins.centers[None, :] - z_obs[:, None]) / sigma[:, None]) ** 2)
lk = binference.Likelihoods(L, bins)
print(lk)
```

- **Row scaling is irrelevant.** Each row can carry any positive constant.
  binference rescales rows internally and keeps the scales separately
  (`lk.log_scale`) so absolute log-likelihoods remain available.
- **Log space.** Pass `log=True` (or use `Likelihoods.from_log`) for
  log-likelihoods; `-inf` marks impossible bins.
- **Representation.** `representation="density"` (the default) means each
  value is the likelihood *at* (or averaged over) the bin.
  `"mass"` means the likelihood *integrated* over the bin, such as CDF
  differences; it is divided by the bin volume. This only matters for unequal
  bins, where it matters a lot (see {doc}`pitfalls`).
- **Sparse matrices.** SciPy sparse matrices are kept sparse.
  `lk.to_sparse(rtol)` drops entries below `rtol` times the row maximum
  ($0\le$ `rtol` $\le 1$).
- **Weights.** `weights=` gives each object an integer multiplicity. `lk.compress()`
  merges identical rows into weighted rows: inference is unchanged, and it
  can be much faster for discrete data such as binned counts. Non-integer weights
  need `fractional_weights=True` and give a weighted pseudo-likelihood, whose
  posterior is not a proper posterior; binference warns.
- **Stored arrays are read-only.** `lk.values`, `lk.log_scale` and `lk.weights` cannot
  be changed in place; use `with_weights`, `subset` or a new constructor call.
- **Objects with no support.** A row of zeros means the object cannot be in
  any bin. That is an error, unless you pass `drop_empty=True` to drop such
  objects with a warning.

## Gaussian measurements

For a measurement $\hat z_i\pm\sigma_i$ the bin-averaged likelihood is exact on
linear bins, computed with stable log-CDF differences that work far into the tails.
On other measures it is averaged in the measure coordinate by adaptive quadrature
(see below). Each object's fraction of likelihood mass outside the bins is kept in
`lk.meta["outside_fraction"]`, and binference warns when more than 10% of the
objects have more than 90% of their mass outside:

```python
lk = binference.Likelihoods.from_gaussian(z_obs, sigma, bins)
print(lk.N, lk.K)
```

On N-D grids, measurement errors are independent per axis by default (`sigma` of shape
`(N, ndim)`). For correlated errors pass each object's covariance matrix instead,
`from_gaussian(mean, bins=grid, cov=cov)` with `cov` of shape `(ndim, ndim)` or
`(N, ndim, ndim)`; the bin averages are then computed by adaptive cubature to `tol`.

## Posteriors computed under an interim prior

Most photometric-redshift codes, classifiers and per-object fits report
**posteriors**, which already include a prior. Divide that **interim prior**
out to recover likelihoods. binference requires you to say what it was:

```python
interim = binference.simulate.population(bins, lambda z: np.exp(-0.5 * ((z - 1.2) / 0.8) ** 2))
P = L * interim                      # posteriors under the interim prior (rows unnormalised)
P /= P.sum(axis=1, keepdims=True)

lk_post = binference.Likelihoods.from_table(P, bins, interim=interim)
np.testing.assert_allclose(lk_post.toarray(), binference.Likelihoods(L, bins).toarray(), atol=1e-12)
```

The interim prior is a `binference.Density`, a fixed density over the bin
coordinate with the same meaning in every constructor:

```python
D = binference.Density
D.flat()                             # flat in the measure coordinate (equal mass per decade on log bins)
D.flat("z")                          # flat in z
D.from_masses(interim)               # probability mass per bin
D.from_densities(interim / bins.volumes)   # density per unit z in each bin
D.from_function(lambda z: np.exp(-z))      # a callable density, integrated over each bin
```

`D.from_distribution(dist)` takes a frozen `scipy.stats` distribution. In place of a
`Density` you can pass `"flat"`, a callable, a distribution or an array. An array is
accepted on its own only where masses and densities agree: on categories and on bins of
equal volume. On unequal bins, pass `interim_representation="mass"` or `"density"`.

`from_table` holds either kind of table, so it asks which: pass `kind="likelihood"`
for likelihood values, or `interim=` for posteriors. A posterior table defaults to
masses per bin (`representation="mass"`), a likelihood table to values at or averaged
over each bin (`"density"`). binference warns if the rows do not look like the
representation you declared.

## Posterior samples

Samples from per-object fits (MCMC chains, nested-sampling output) are
histogrammed onto the bins and divided by the interim prior. Ragged lists
(different numbers of samples per object) and importance weights are
supported:

```python
samples = [z + s * rng.standard_normal(200) for z, s in zip(z_obs[:100], sigma[:100])]
lk_samp = binference.Likelihoods.from_samples(samples, bins, interim="flat")
print(lk_samp.is_sparse, np.median(lk_samp.meta["n_eff"]))
```

One estimator covers every interim prior: each sample is weighted by $T'(z)/\pi(z)$,
which stays exact when the interim prior varies within a bin. An interim prior known only
per bin (`"flat"`, masses or densities) is read as flat in the measure coordinate within
each bin, so the estimate becomes each object's counts divided by the interim masses. Samples drawn from a likelihood are samples under
`binference.Density.flat("z")`.

The effective number of samples inside the bins and the fraction outside are stored in
`lk_samp.meta`; binference warns when more than 10% of the objects have fewer effective
samples than `min_samples` (default 20). Bins that no sample reached have zero likelihood;
`floor=a` adds `a` pseudo-samples per object (whatever the scale of `sample_weights`), spread over the bins in proportion to the
interim prior. At a fitted population, `lk_samp.sample_variance(rho)` gives each object's
relative Monte Carlo variance $V_i(\rho)$ and effective sample size; `result.check()`
warns when $\sum_i V_i > 1$, which is roughly the bias that sample noise puts into the
total log-likelihood.

## Likelihood functions and distributions

For an arbitrary likelihood or posterior, give a vectorised function of the bin
coordinate, or a list of SciPy frozen distributions, to `from_density`. As with
`from_table`, say what it is with `kind="likelihood"` or `interim=`:

```python
from scipy import stats

def loglike(z):  # (n_points, ndim) or (n_points,) -> (N, n_points)
    z = np.asarray(z).reshape(-1)
    return stats.t.logpdf(z[None, :], df=3, loc=z_obs[:, None], scale=sigma[:, None])

lk_t = binference.Likelihoods.from_density(loglike, bins, kind="likelihood", log=True)
lk_d = binference.Likelihoods.from_density(
    [stats.norm(m, s) for m, s in zip(z_obs[:10], sigma[:10])], bins, kind="likelihood"
)
print(lk_t.N, lk_d.N)
```

Likelihoods are averaged over each bin uniformly in the measure coordinate, which is
the shape the model gives the population within a bin; posteriors are integrated in
$z$ and divided by the interim prior's bin masses. On 1-D bins, distributions with a
CDF are integrated exactly where that gives the answer (posteriors, and likelihoods
on linear bins), in log space so far tails keep their precision. Everything else uses
adaptive quadrature: each bin is split in half along every axis until the difference
between 5- and 3-point Gauss–Legendre estimates is below `tol` (default $10^{-3}$)
relative to the object's largest bin average, at most `max_depth=12` times. Bins that
do not converge raise an `InputWarning`; `n_sub=` fixes the number of points per bin
and axis instead.

## Binned measurements and classifiers

A binned measurement records for each object only the bin $j$ of a measured quantity: a
reconstructed energy, a classifier score, a predicted label. Its
{class}`~binference.Response` is the matrix
$R_{jk}=p(\text{measured in }j,\text{ observed}\mid\text{true bin }k)$, whose column
sums are the efficiencies $\epsilon_k\le1$. `Likelihoods.from_binned(response,
counts=...)` (or `values=...`, the measured value of each object) gives every object
measured in bin $j$ the likelihood row $R_{j\cdot}$, on the response's truth bins, and
attaches the efficiency as a selection function with mode `"on_measurement"`.

A response comes from one of three places:

- **A known matrix**: `Response(R, truth_bins, measured_bins)`. A *migration* matrix,
  $p(\text{measured }j\mid\text{true }k,\text{ observed})$ with columns that sum to
  one, takes the efficiency separately: `Response(migration, truth_bins,
  efficiency=eps)`. Columns that sum to more than one are rejected.
- **Tallied events**: `Response.from_counts(counts, generated, truth_bins,
  measured_bins)`, or from the events themselves, `Response.from_simulation(true_values,
  measured_values, truth_bins, measured_bins)`, where a measured value that is NaN, or
  outside the measured bins, marks an event that was not observed.
- **A labelled validation set**, for classifiers: `Response.from_labelled(scores,
  labels)`. Real-valued scores (for example the probability of one class) get 10 score
  bins holding equal numbers of validation objects, the outer two open-ended; predicted
  labels give the confusion matrix.

Responses built from events keep the tallies (`response.counts`, `response.generated`).
`open_ended=True` lets the outer two of 1-D measured bins take every value below or above
them; `from_labelled` sets it for its score bins.

**Classifiers** reach the population by one of two routes, chosen by what you have
(DECISIONS #29); `result.summary()` names the inputs in its title (`posteriors` for
route A, `binned, response from labelled data` for route C).

- **Route A**: posteriors and the training prevalence. `from_table(probs,
  Bins.categories(classes), interim=train_prior)` divides the prevalence out. It is
  exact only if the posteriors are calibrated and `train_prior` is the prevalence the
  classifier effectively learned.
- **Route C**: scores and a labelled validation set. `from_binned(Response.from_labelled(
  val_scores, val_labels), values=scores)` needs neither calibration nor the training
  prior, only label shift: the score distribution within each class is the same in the
  validation and the target set. Prefer it when a validation set is available.

When there are more measured bins than truth bins, `result.check()` compares the
measured counts with those each posterior draw predicts through the response (a
posterior predictive p-value that includes the response's own sampling noise). It warns
when $p<10^{-3}$, which for a classifier means label shift is violated.

```python
mu = binference.simulate.mock_unfolding(3000, rng=2)
lk_unfold = binference.Likelihoods.from_binned(mu.response, counts=mu.measurements["reco_counts"])
mc = binference.simulate.mock_categorical(1000, 4, rng=3)
lk_a = binference.Likelihoods.from_table(mc.posteriors, mc.bins, interim=mc.interim)
lk_c = binference.Likelihoods.from_binned(mc.response, values=mc.measurements["feature"])
print(lk_unfold.selection.mode, mc.response, lk_c.bins.labels)
```

## Working with `Likelihoods`

```python
lk = binference.Likelihoods.from_gaussian(z_obs, sigma, bins)
sub = lk[:100]                       # subsets keep bins, weights and selection
dense = lk.toarray()                 # normalised values (each row's max is 1)
stack = lk.stack()                   # naive stacked estimate, for comparison only
print(sub.N, dense.shape, stack.sum().round(6))
```

`lk.stack()` is the familiar "stacked PDF": the mean of the per-object posteriors
under the interim prior the likelihoods were built with (flat if there was none),
so for posteriors given to `from_table` it reproduces the mean of the input posteriors. It is
exactly one EM step from the interim prior, and it is biased whenever measurement
errors are comparable to the features of the population. It is there for
comparison, not for inference. The KKT gap measures how far it is from the
maximum-likelihood population (zero at the optimum):

```python
model = binference.PopulationModel(lk)
print(model.kkt_gap(lk.stack()), model.kkt_gap(binference.infer(lk, method="map").rho))
```
