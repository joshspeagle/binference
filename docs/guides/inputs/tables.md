# Likelihood tables

Use this page if you already have a number per object and bin. Examples are a
likelihood evaluated on the bins, a table from another code, or genotype likelihoods
per stock. Such a table is a {term}`likelihood matrix` $L_{ik}$, with one row per object and one column
per bin.

```python
import numpy as np
import binference

rng = np.random.default_rng(1)
z_obs = rng.uniform(0.2, 2.8, size=500)
sigma = rng.uniform(0.05, 0.3, size=500)
bins = binference.Bins.linear(0.0, 3.0, 30)

L = np.exp(-0.5 * ((bins.centers[None, :] - z_obs[:, None]) / sigma[:, None]) ** 2)
lk = binference.Likelihoods(L, bins)
print(lk)
```

`Likelihoods.from_table(L, bins, kind="likelihood")` does the same. It makes you say
that the table holds likelihoods, not posteriors ({doc}`posteriors`).

```{warning}
A likelihood evaluated at bin centres, as above, only approximates the bin average.
For likelihoods narrower than a bin it can be far off. If you know the likelihood's
form, use {doc}`gaussian` or {doc}`functions` instead.
```

## Conventions

- **Row scaling does not matter.** Each row can carry any positive constant.
  binference rescales the rows and keeps the scales in `lk.log_scale`, so absolute
  log-likelihoods stay available.
- **Log values.** Pass `log=True`, or use `Likelihoods.from_log`, for
  log-likelihoods. `-inf` marks an impossible bin.
- **Masses or densities.** The default, `representation="density"`, reads each value
  as the likelihood *at* the bin or *averaged over* it. `representation="mass"` reads
  it as the likelihood *integrated over* the bin, such as a CDF difference. binference
  divides a {term}`mass` by the {term}`bin volume`. On equal bins the choice makes no
  difference; on unequal bins it matters a lot
  ([masses versus densities](../../how-it-works/model.md#masses-versus-densities)).
- **Objects with no support.** A row of zeros means the object cannot lie in any bin.
  That raises an error. Pass `drop_empty=True` to drop such objects with a warning.

```python
loglik = -0.5 * ((bins.centers[None, :] - z_obs[:, None]) / sigma[:, None]) ** 2
lk_log = binference.Likelihoods.from_log(loglik, bins)
print(np.allclose(lk_log.toarray(), lk.toarray()))
```

`lk.toarray()` returns the values with each row's largest value set to one. The
{doc}`mixed-stocks tutorial </tutorials/10_mixed_stocks>` builds a table of
log-likelihoods over categories from genotypes.
