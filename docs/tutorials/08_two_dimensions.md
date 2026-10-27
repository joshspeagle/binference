---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# A two-dimensional population

Nothing in binference is specific to one dimension. Bins can be a grid in any
number of dimensions, such as a colour–magnitude diagram, a mass–ratio plane,
or redshift and luminosity. This tutorial infers a 2-D population with a
curved ridge, like a stellar sequence, from noisy measurements in both
coordinates.

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np

import binference

grid = binference.Bins.grid(np.linspace(0, 1, 13), np.linspace(0, 1, 13))   # 12 x 12 cells
mock = binference.simulate.mock_2d(3000, grid, density="banana", noise=(0.02, 0.06), rng=8)
lk = mock.likelihoods
print(grid, grid.shape, lk)
```

Each object's likelihood is a 2-D Gaussian over the grid, with errors that
vary from object to object and between the two axes. The flattened likelihood
matrix has one column per cell, in C order: use `grid.unflatten` and
`grid.flatten` to convert.

```{code-cell} python
obs = mock.measurements["mean"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
ex, ey = grid.edges
axes[0].pcolormesh(ex, ey, grid.unflatten(mock.truth.rho / grid.volumes).T, cmap="Blues")
axes[0].set_title("true population density", loc="left")
axes[1].scatter(obs[:, 0], obs[:, 1], s=2, color="0.3", alpha=0.5)
axes[1].set(xlim=(0, 1), ylim=(0, 1), title="measured values")
for ax in axes:
    ax.set(xlabel="x", ylabel="y")
```

## A smooth population on the grid

A Gaussian-process prior with one length scale per axis lets the data decide
how smooth the population is in each direction. This is the largest fit in
the tutorials, so `n_jobs=-1` runs the four chains in parallel processes.
They use the same random streams as a serial run.

```{code-cell} python
result = binference.infer(lk, binference.priors.LogisticGP(n_basis=10), rng=1, n_samples=500,
                      n_warmup=500, n_jobs=-1)
print(result)
print({k: np.quantile(v, [0.16, 0.5, 0.84]).round(2)
       for k, v in result.info["hyperparameters"].items()})
```

Ten sines per axis (`n_basis=10`) are enough for a 12 × 12 grid. The default,
one sine per bin, gives 144 modes, and NUTS then moved slowly through their
energy (E-BFMI below 0.3 in one chain;
[what it means](../guides/troubleshooting.md#not-looking-at-the-diagnostics)).

```{code-cell} python
axes = result.plot()
for ax in axes:
    ax.set(xlabel="x", ylabel="y")
```

The left panel is the posterior median density and the right panel the
relative uncertainty per cell. It is small along the ridge and large where
there are few objects.

## Derived quantities across the grid

Every derived quantity is computed per draw, so correlations between cells
are propagated. For example, here is the marginal distribution along $x$,
and the mean $y$ at each $x$ (the shape of the ridge):

```{code-cell} python
draws = result.samples.reshape(-1, *grid.shape)        # (n_draws, n_x, n_y)
marg_x = draws.sum(axis=2)                              # fraction per x column
yc = grid.centers.reshape(*grid.shape, 2)[0, :, 1]      # y at the cell centres
mean_y = (draws * yc).sum(axis=2) / marg_x              # mean y in each x column

true = grid.unflatten(mock.truth.rho)
xc = 0.5 * (ex[1:] + ex[:-1])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
for ax, d, t, lab in [(axes[0], marg_x, true.sum(axis=1), "fraction in each x column"),
                      (axes[1], mean_y, (true * yc).sum(axis=1) / true.sum(axis=1), "mean y at given x")]:
    lo, med, hi = np.quantile(d, [0.16, 0.5, 0.84], axis=0)
    ax.fill_between(xc, lo, hi, color="#2a78d6", alpha=0.3, lw=0, label="68% interval")
    ax.plot(xc, med, color="#2a78d6", lw=1.5, label="posterior median")
    ax.plot(xc, t, "k--", lw=1.5, label="truth")
    ax.set(xlabel="x", ylabel=lab)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False);
```

## Scaling up

The cost of each likelihood evaluation grows with the number of objects
times the number of cells (the number of stored entries for sparse
likelihoods). For larger grids:

- Store likelihoods sparsely (`lk.to_sparse(rtol)`): objects with small
  errors cover only a handful of cells.
- Run chains in parallel with `n_jobs=-1`.
- Get a quick estimate with `method="map"` before sampling.

The {doc}`performance guide <../guides/performance>` has more.
