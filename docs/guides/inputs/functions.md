# Likelihood functions and distributions

Use this page if you can evaluate each object's likelihood $\ell_i(z)=p(d_i\mid z)$ at
any true value $z$. Examples are a Student-$t$ error, a skewed error from a fit, or a
SciPy distribution per object. {meth}`~binference.Likelihoods.from_density` averages
each function over the bins.

## Say what the function is

A function can be a likelihood or a posterior, and binference cannot tell them apart.
Pass `kind="likelihood"` for a likelihood. For a posterior, pass the prior it was
computed under as `interim=` ({doc}`posteriors`).

## A vectorised function

The function takes the points, shape `(M,)` in 1-D or `(M, ndim)` on a grid. It returns
an `(N, M)` array: one row per object. With `log=True` it returns logarithms.

```python
import numpy as np
from scipy import stats

import binference

rng = np.random.default_rng(1)
z_obs = rng.uniform(0.2, 2.8, size=300)
sigma = rng.uniform(0.05, 0.3, size=300)
bins = binference.Bins.linear(0.0, 3.0, 30)


def loglike(z):
    z = np.asarray(z).reshape(-1)
    return stats.t.logpdf(z[None, :], df=3, loc=z_obs[:, None], scale=sigma[:, None])


lk_t = binference.Likelihoods.from_density(loglike, bins, kind="likelihood", log=True)
print(lk_t.N, lk_t.K)
```

Define the function at module level, not as a `lambda`. A lambda cannot be pickled, so
saved or copied likelihoods lose their shape within bins.

## A list of distributions

Pass one frozen SciPy distribution per object, or any object with `logpdf`, `pdf`,
`logcdf` or `cdf`:

```python
dists = [stats.norm(m, s) for m, s in zip(z_obs[:10], sigma[:10])]
lk_d = binference.Likelihoods.from_density(dists, bins, kind="likelihood")
print(lk_d.N)
```

## How the bin averages are computed

- **Likelihoods** are averaged over each bin, uniformly in the bin's
  {term}`measure` coordinate. That is the shape the model gives the population within
  a bin.
- **Posteriors** are integrated over each bin in $z$, then divided by the
  {term}`interim prior`'s mass in the bin.
- **On 1-D bins**, a distribution with a CDF is integrated exactly where that gives
  the answer: for posteriors, and for likelihoods on linear bins. The integral is taken
  in log space, so far tails keep their precision; the upper tail needs `sf` or
  `logsf`, which SciPy's distributions have.
- **Everything else** uses adaptive quadrature. Each bin is halved along every axis
  until the error estimate is below `tol`, relative to the object's largest bin
  average (its largest bin mass for posteriors). At most `max_depth` halvings are made.

If some bins do not reach `tol`, binference raises an `InputWarning`
([warnings](../../reference/warnings.md#inputwarning)). If that happens, raise
`max_depth` or `tol`, or pass `n_sub=` for a fixed number of points per bin and axis.
