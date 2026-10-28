# Advanced input options

This page covers options that most analyses do not need. They are the coordinate in
which the population is flat within a bin, masked grids, a population with shape within
bins, sparse storage and weights.

```python
import numpy as np
import binference
```

## Measures: flat in which coordinate?

Within a bin, the model takes the population to be flat in some coordinate $u=T(z)$:
the bin axis's {term}`measure`. `Bins.linear` and explicit edges use $u=z$. `Bins.log`
uses $u=\log z$, so a log bin's centre is the geometric mean of its edges.

Pass `measure=` to choose: `"linear"`, `"log"`, or a {class}`~binference.Measure`. A
`Measure` is built from a transform and its inverse, or from a positive density
$g(z)$; the transform is then the cumulative integral of $g$.

```python
sqrt_bins = binference.Bins([0.0, 1.0, 4.0, 9.0],
                            measure=binference.Measure.transform(np.sqrt, np.square))
custom = binference.Bins(np.linspace(1.0, 10.0, 6),
                         measure=binference.Measure.density(lambda z: 1 / z, 1.0, 10.0))
print(sqrt_bins.centers, sqrt_bins.u_edges[0])
```

If hand-built edges are spaced geometrically and no measure is given, binference warns
([warnings](../../reference/warnings.md#inputwarning)).

## Masked grids

A `mask` removes cells from a grid, for example the unphysical corner of a
colour–magnitude diagram. Masked cells take no part in the model. `K` counts the active
cells, `index` returns `-1` for values in a masked cell, and `unflatten` fills masked
cells with NaN.

```python
mask = np.ones((4, 5), dtype=bool)
mask[0, 0] = False
masked = binference.Bins.grid(np.linspace(0, 4, 5), np.linspace(0, 5, 6), mask=mask)
print(masked.K, masked.shape, masked.index([[0.5, 0.5], [0.5, 1.5]]))
```

The random-walk and Gaussian-process priors do not accept masked grids
({doc}`../priors`).

## Shape within bins: `Basis`

By default the population is flat (in $u$) within each bin. A {term}`basis` lets it
vary. A {class}`~binference.Basis` separates the bins you report in from the components
the model fits:

- **Components.** The population is a mixture $\sum_k\beta_k\psi_k(z)$ of components
  $\psi_k$ of unit mass, on the bins refined `resolution` times per axis.
- **Order.** `order=0` makes each component flat on its cell. `order=1` makes it a
  linear B-spline: a hat centred on its cell, so the density is continuous. Hats need
  an unmasked grid.
- **Reporting.** Fractions are reported in the bins, $\rho_j=\sum_kA_{jk}\beta_k$, with
  $A_{jk}$ the mass of component $k$ in bin $j$ (`basis.matrix`).

Pass the basis wherever bins go:

```python
bins = binference.Bins.linear(0.0, 3.0, 10)
basis = binference.Basis(bins, order=1, resolution=4)
mock = binference.simulate.mock_1d(500, bins, rng=1)
lk = binference.Likelihoods.from_gaussian(
    mock.measurements["mean"], mock.measurements["sigma"], basis
)
res = binference.infer(lk, binference.priors.LogisticRandomWalk(order=2), method="map")
print(lk.K, basis.C, res.rho.shape)   # 40 components, reported in 10 bins
```

How each input meets the components:

- **Inputs known within bins** (`from_gaussian`, `from_density`, `from_samples`) are
  integrated against each component on `oversample` quadrature cells per component and
  axis ({doc}`defaults <../../reference/defaults>`).
- **Inputs per bin** (arrays, `from_table`, `from_log`) are taken as flat within each
  bin. For any other input, such as `from_binned`, call `lk.on_basis(basis)`.
- **A selection** given as a function is averaged with each component as the weight.
  An array per bin is mapped with $A$.

The priors see the components: `lk.bins` is `basis.components`. A resolution above one
therefore needs a {term}`smoothness prior`. Components narrower than the measurement
errors are not identified one by one. With a flat Dirichlet, the fit then oscillates
from one component to the next.

Every result reports in the bins. Per-object outputs share each component among the
bins by $A$. That is exact for flat components and ignores the likelihood's variation
within a hat. Gibbs label draws (`store_labels`) index the components. {doc}`../bins`
says when a basis is worth it.

## Sparse likelihoods

Narrow likelihoods on many bins are mostly zeros. SciPy sparse matrices are kept
sparse. `lk.to_sparse(rtol)` drops entries below `rtol` times each row's largest value,
with $0\le$ `rtol` $\le1$. `from_samples` stores its result sparsely when few entries
are non-zero ({doc}`defaults <../../reference/defaults>`). {doc}`../performance` says
when sparse storage is faster.

```python
sparse = lk.to_sparse(1e-8)
print(sparse.is_sparse)
```

## Weights

`weights=` gives each object an integer multiplicity $w_i$. `lk.compress()` merges
identical rows into weighted rows. Inference on the population is unchanged, and it can
be much faster for discrete data such as binned counts.

Non-integer weights need `fractional_weights=True`. They define a weighted
pseudo-likelihood, whose posterior is not a proper posterior, and binference warns.

```python
counts = np.repeat(np.eye(4), 50, axis=0)       # 200 objects, 4 distinct rows
lk_c = binference.Likelihoods(counts + 0.1, binference.Bins.linear(0, 1, 4))
print(lk_c.N, lk_c.compress().N, lk_c.compress().weights)
```

## Working with `Likelihoods`

```python
sub = lk[:100]                       # a subset keeps bins, weights and selection
dense = lk.toarray()                 # values with each row's largest set to one
print(sub.N, dense.shape)
```

The stored arrays `lk.values`, `lk.log_scale` and `lk.weights` are read-only. To change
them, use `with_weights`, `with_selection`, `subset` or a new constructor call.
