# Gaussian measurements

Use this page if each object has a measurement $\hat z_i$ with a Gaussian error
$\sigma_i$: a photometric redshift with its error bar, a parallax, a test score.

```python
import numpy as np
import binference

rng = np.random.default_rng(1)
z_true = rng.uniform(0.2, 2.8, size=500)
sigma = rng.uniform(0.05, 0.3, size=500)
z_obs = z_true + sigma * rng.standard_normal(500)

bins = binference.Bins.linear(0.0, 3.0, 30)
lk = binference.Likelihoods.from_gaussian(z_obs, sigma, bins)
print(lk.N, lk.K)
```

On linear bins the {term}`bin-averaged likelihood` is exact. binference computes it from
differences of the log normal CDF, which stay accurate far into the tails. On other
{term}`measures <measure>`, such as `Bins.log`, it averages over each bin by adaptive
quadrature to `tol`.

## Measurements outside the bins

Part of a likelihood can fall outside the bins. Each object's share outside is kept in
`lk.meta["outside_fraction"]`:

```python
print(lk.meta["outside_fraction"].max().round(3))
```

binference warns when many objects lie mostly outside the bins; the
{doc}`defaults <../../reference/defaults>` page gives the shares. If the warning
appears, widen the bins. Pass `warn_outside=False` if the loss is intended.

## Two or more dimensions

On a grid, errors are independent per axis by default: pass `sigma` of shape
`(N, ndim)`. For correlated errors pass each object's covariance matrix as `cov`, of
shape `(ndim, ndim)` or `(N, ndim, ndim)`. The bin averages are then computed by
adaptive cubature to `tol`.

```python
grid = binference.Bins.grid(np.linspace(0, 3, 13), np.linspace(0, 3, 13))
mean = rng.uniform(0.5, 2.5, size=(200, 2))
cov = np.array([[0.04, 0.02], [0.02, 0.09]])
lk2 = binference.Likelihoods.from_gaussian(mean, bins=grid, cov=cov)
print(lk2.N, lk2.K)
```

The {doc}`quick start </tutorials/01_quickstart>` fits a population from Gaussian
measurements end to end.
