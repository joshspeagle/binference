# Priors

The prior on the population matters most when the data are least
informative: many bins, few objects, or measurement errors as large as the
bins. binference's priors fall into three families. The **Dirichlet family** is
conjugate and works with every engine. The **logistic family** adds
smoothness across neighbouring bins. **Gamma rates** model expected counts
rather than fractions.

```python
import numpy as np
import binference
from binference import priors

mock = binference.simulate.mock_1d(1000, binference.Bins.linear(0, 3, 30), noise=(0.1, 0.3), rng=4)
lk = mock.likelihoods
```

Pass a prior as the second argument of {func}`~binference.infer`, or as `prior=`
to {class}`~binference.PopulationModel` and {class}`~binference.PopulationEstimator`.
The default is `Dirichlet(1.0)`.

## The Dirichlet family

```python
flat = priors.Dirichlet(1.0)                   # uniform on the simplex (default)
sparse = priors.Dirichlet.reference()          # alpha = 1/K: K pseudo-counts become 1
by_volume = priors.Dirichlet.from_base_measure(5.0)  # alpha proportional to bin width (in u)
ref_counts = np.bincount(mock.truth.labels[:50], minlength=30)
informed = priors.Dirichlet.from_counts(ref_counts, strength=0.5)  # alpha = 1 + 0.5 m

for p in (flat, sparse, informed):
    print(p, binference.infer(lk, p, method="map").rho[:3].round(4))
```

- **`alpha = 1`** is uniform over all possible populations, but it adds one
  pseudo-count per bin. With $K$ comparable to $N$ that flattens the
  estimate; `Dirichlet.reference()` ($\alpha=1/K$) adds one pseudo-count in
  total.
- **Unequal bins.** A symmetric $\alpha$ is not consistent under merging
  bins. `from_base_measure` makes $\alpha_k$ proportional to the bin masses of
  a base measure, a `binference.Density` (by default flat in the measure
  coordinate: bin widths on linear bins, widths in $\log z$ on log bins), so the
  prior describes the same density however you bin it.
- **Reference samples.** `from_counts(m, strength)` encodes a reference
  sample with counts $m$ as a power prior. `strength=1` trusts it fully;
  smaller values down-weight a sample that may not be representative.

`GeneralizedDirichlet`, `DirichletTree` and `PolyaTree` relax the
Dirichlet's rigid covariance while staying conjugate:

```python
gd = priors.GeneralizedDirichlet(a=2.0, b=6.0)        # stick-breaking Betas
tree = priors.DirichletTree([[0, 1, 2], [3, [4, 5]]], alpha=2.0)
polya = priors.PolyaTree(c=2.0)                       # dyadic, multi-scale
print(binference.infer(lk, polya, method="map").rho[:3].round(4))
```

- A **Dirichlet tree** puts a Dirichlet on every internal node, so bins
  under the same node co-vary. Pass the tree as nested lists of bin indices.
- A **Pólya tree** is the balanced binary tree over an ordered grid. It is
  centred on a base measure, and its fluctuations shrink at finer scales. It
  is a conjugate, multi-scale alternative to the smoothness priors below.

## Smoothness: the logistic family

With broad likelihoods, neighbouring bins trade mass and a Dirichlet
posterior is jagged. That is honest, since the data really cannot separate
them, but it is rarely what the population looks like. Logistic priors put a
Gaussian prior on the log-density and infer how smooth it is. By default they
are centred on a density flat in the measure coordinate (log-fractions equal
to the log bin widths in $u$: in $z$ on linear bins, in $\log z$ on log bins),
so the penalty acts on the density whatever the binning, `Bins.log`
included:

```python
rw = priors.LogisticRandomWalk(order=2)        # penalise curvature of log rho
gp = priors.LogisticGP(kernel="matern32")       # Hilbert-space Gaussian process
print(binference.infer(lk, gp, method="map").rho[:3].round(4))
```

| prior | smoothness | hyperparameters (inferred by default) |
|---|---|---|
| `LogisticRandomWalk(order=1)` | penalises jumps between neighbours | `scale` of the differences |
| `LogisticRandomWalk(order=2)` | penalises curvature; linear trends (and, on 2-D grids, the surface $xy$) are free | `scale` |
| `LogisticGP(kernel=...)` | stationary GP: `"matern12"`, `"matern32"`, `"matern52"`, `"rbf"` | `amplitude`, `length_scale` (one per axis) |
| `LogisticNormal(mean, cov)` | any fixed covariance, or independent bins | `scale` |

- **Grids.** All of them work on 1-D bins and N-D grids. The random walk
  then uses the sum of per-axis difference penalties, and the GP takes one
  length scale per axis.
- **GP length scales** are in the measure coordinate ($\log z$ on `Bins.log`).
  The default prior keeps them above what the basis of `n_basis` sines per
  axis (default: one per bin) can represent; for rougher fields raise
  `n_basis`, which lowers that bound too.
- **Hyperparameters.** Pass a number to fix one, or a hyperprior:
  ```python
  gp_fixed = priors.LogisticGP(amplitude=1.5, length_scale=priors.LogNormal(np.log(0.5), 0.5))
  rw_tight = priors.LogisticRandomWalk(order=2, scale=priors.HalfNormal(0.3))
  ```
- **Reference means.** `mean=np.log(reference)`, with `reference` the
  fraction of a reference population in each bin, centres the prior on that
  distribution; the amplitude or scale then says how far the population may
  depart from it.
- **Engines.** Logistic priors are not conjugate. Use NUTS (which the default
  `method="auto"` picks for them unless the Laplace check passes), MAP or Laplace; Gibbs and VB need conjugate priors.

```{tip}
Which prior should I use? Start with the default. If the posterior is
jagged, your measurement errors are comparable to the bins. Try
`LogisticGP()` or `LogisticRandomWalk(order=2)`, and compare the fits on
held-out objects with `result.log_predictive` (see {doc}`results`).
```

## Rates instead of fractions

`GammaRates(shape, rate)` puts independent Gamma priors on the **expected
number** of objects per bin, $\lambda_k$. The data then form a Poisson
process. Every fit of fractions already gives the total and the rates under a
scale-free prior on the total (see {doc}`selection`); use `GammaRates` when
the rates need their own prior, for example bin-specific shapes or an
informative total, and combine them with an exposure or selection function.

```python
est = binference.infer(lk, priors.GammaRates(shape=1.0, rate=1e-3), method="map")
print(est.rates.sum().round(1), est.rho.sum())
```

Sampled results expose both `rates` (draws of $\lambda$) and `samples` (the
implied fractions); NUTS and Gibbs both support rates.

## Checking a prior

Draw from the prior to see what it believes before seeing data:

```python
model = binference.PopulationModel(lk, gp)
draws = model.sample_prior(5, rng=0)       # five populations from the prior
print(draws.shape, np.allclose(draws.sum(axis=1), 1.0))
```

{func}`binference.simulate.prior_predictive` goes one step further: it simulates
whole catalogues from the prior, which is what simulation-based calibration
needs (see {doc}`validation`).
