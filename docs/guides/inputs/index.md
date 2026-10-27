# Prepare your inputs

binference needs one thing from your data. For each object $i$ and bin $k$ it needs
the {term}`bin-averaged likelihood` $L_{ik}$: how probable the object's data are if its
true value lies in bin $k$. A {class}`~binference.Likelihoods` object holds these values
with their {class}`~binference.Bins`. Its constructors turn the usual data products
into likelihoods.

## Find your input

| What you have for each object | Constructor | Page |
|---|---|---|
| a measurement and its Gaussian error | `Likelihoods.from_gaussian` | {doc}`gaussian` |
| a likelihood function, or a SciPy distribution | `Likelihoods.from_density` | {doc}`functions` |
| likelihood values per bin | `Likelihoods(L, bins)`, `from_table`, `from_log` | {doc}`tables` |
| a posterior per bin, computed under some prior | `Likelihoods.from_table(..., interim=)` | {doc}`posteriors` |
| samples from a posterior or a likelihood | `Likelihoods.from_samples` | {doc}`samples` |
| a measured bin, or a classifier's score or label | `Likelihoods.from_binned` | {doc}`binned` |

If your code reports *posteriors*, they already contain a prior. Say which prior, or the
population is pulled toward it ({doc}`posteriors`).

## Bins

Every constructor takes the bins of the true value. Build them first:

```python
import numpy as np
import binference

bins = binference.Bins.linear(0.0, 3.0, 30)              # 30 equal bins on [0, 3]
logbins = binference.Bins.log(1e-2, 1e2, 20)             # equal in log z
irregular = binference.Bins([0.0, 0.2, 0.5, 1.0, 3.0])   # explicit edges
grid = binference.Bins.grid(np.linspace(-0.5, 2.0, 26), np.linspace(-5.0, 15.0, 41))
classes = binference.Bins.categories(["star", "galaxy", "quasar"])
print(bins.K, grid.shape, grid.K, classes.labels)
```

Bins know their `edges`, `centers` and `volumes` (widths in 1-D), and `index` returns the
bin of each value. A 2-D grid is flattened in C order: `unflatten` and `flatten` move
between the vector of $K$ values and the grid. Categories have unit volumes and no
geometry. {doc}`../bins` says how many bins to use and where to put them.

```{toctree}
:maxdepth: 1

gaussian
functions
tables
posteriors
samples
binned
advanced
```
