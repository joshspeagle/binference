---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# A two-dimensional population

Nothing in binference is specific to one dimension. Bins can be a grid in any number of
dimensions, such as a colour–magnitude diagram or a plane of mass and mass ratio. This
tutorial infers a two-dimensional population with a curved ridge, like a stellar
sequence, from measurements that are noisy in both coordinates.

```{code-cell} python
:tags: [remove-cell]

# figures 6 in wide with 17 pt text keep text above 12 px on a 390 px phone screen
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.figsize": (6, 3.4), "font.size": 17,
                     "figure.constrained_layout.use": True})
```

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import binference

OBJECT = "#EC5F3E"   # the colour of single objects in these docs

grid = binference.Bins.grid(np.linspace(0, 1, 13), np.linspace(0, 1, 13))   # 12 x 12 cells
mock = binference.simulate.mock_2d(3000, grid, density="banana", noise=(0.02, 0.06), rng=8)
lk = mock.likelihoods
print(grid, grid.shape, lk)
```

Each object's likelihood is a two-dimensional Gaussian over the grid. Its errors vary
from object to object and between the two axes. The likelihood matrix has one column per
cell, in C order; `grid.unflatten` and `grid.flatten` convert between the two layouts.

```{code-cell} python
---
mystnb:
  figure:
    caption: The true population density on the 12 by 12 grid (top) and the measured
      values of the 3000 objects (bottom).
  image:
    alt: Top, a shaded grid with a curved ridge rising from the lower left to the upper
      right. Bottom, a scatter of points that traces the same ridge, blurred by the
      measurement errors.
---
obs = mock.measurements["mean"]
ex, ey = grid.edges
ramp = LinearSegmentedColormap.from_list("ramp", ["#F1F2FA", "C0"])
fig, axes = plt.subplots(2, 1, figsize=(6, 9))
im = axes[0].pcolormesh(ex, ey, grid.unflatten(mock.truth.rho / grid.volumes).T, cmap=ramp)
fig.colorbar(im, ax=axes[0], label="density")
axes[0].set_title("true population", loc="left")
axes[1].scatter(obs[:, 0], obs[:, 1], s=3, color=OBJECT, alpha=0.4, lw=0)
axes[1].set(xlim=(0, 1), ylim=(0, 1))
axes[1].set_title("measured values", loc="left")
for ax in axes:
    ax.set(xlabel="x", ylabel="y");
```

## A smooth population on the grid

A Gaussian-process prior with one length scale per axis lets the data decide how smooth
the population is in each direction. This is the largest fit in the tutorials, so
`n_jobs=-1` runs the four chains in parallel processes. They use the same random streams
as a serial run.

```{code-cell} python
result = binference.infer(lk, binference.priors.LogisticGP(n_basis=10), rng=1, n_samples=500,
                          n_warmup=500, n_jobs=-1)
print(result)
print({k: np.quantile(v, [0.16, 0.5, 0.84]).round(2)
       for k, v in result.info["hyperparameters"].items()})
```

Ten sines per axis (`n_basis=10`) are enough for a 12 × 12 grid. The default, one sine
per bin, gives 144 modes, which makes each NUTS step slower.

```{code-cell} python
---
mystnb:
  figure:
    caption: The posterior median density (top) and the relative uncertainty of each
      cell, its standard deviation over its median (bottom).
  image:
    alt: Top, a shaded grid with the same curved ridge as the true population. Bottom,
      a grey grid that is pale along the ridge, where the uncertainty is small, and
      darker away from it. It is darkest below the ridge near x of 0.5, where the median
      is close to zero.
---
fig, axes = plt.subplots(2, 1, figsize=(6, 9))
result.plot(ax=axes)
for ax in axes:
    ax.set(xlabel="x", ylabel="y");
```

The relative uncertainty is small along the ridge and large where there are few objects.

## Derived quantities across the grid

Every derived quantity is computed per draw, so the correlations between cells carry
through. Here are the marginal distribution along $x$, and the mean $y$ at each $x$,
which traces the ridge:

```{code-cell} python
---
mystnb:
  figure:
    caption: The fraction of the population in each column of x (top) and the mean y in
      each column (bottom), with 68% bands. The truth is dashed.
  image:
    alt: Two stacked plots against x. Top, a hump that peaks near x of 0.4, with a
      narrow band around the dashed truth. Bottom, steps that rise from about 0.25 to
      0.75, with a narrow band, on top of the dashed truth.
---
draws = result.samples.reshape(-1, *grid.shape)        # (n_draws, n_x, n_y)
marg_x = draws.sum(axis=2)                              # fraction per x column
yc = grid.centers.reshape(*grid.shape, 2)[0, :, 1]      # y at the cell centres
mean_y = (draws * yc).sum(axis=2) / marg_x              # mean y in each x column

true = grid.unflatten(mock.truth.rho)
fig, axes = plt.subplots(2, 1, figsize=(6, 6.4), sharex=True)
for ax, d, t, lab in [(axes[0], marg_x, true.sum(axis=1), "fraction"),
                      (axes[1], mean_y, (true * yc).sum(axis=1) / true.sum(axis=1), "mean y")]:
    lo, med, hi = np.quantile(d, [0.16, 0.5, 0.84], axis=0)
    ax.fill_between(ex, np.append(lo, lo[-1]), np.append(hi, hi[-1]), step="post",
                    color="C0", alpha=0.3, lw=0, label="68% interval")
    ax.step(ex, np.append(med, med[-1]), where="post", color="C0", label="median")
    ax.step(ex, np.append(t, t[-1]), where="post", color="k", ls="--", label="truth")
    ax.set(ylabel=lab)
axes[1].legend(loc="lower right")
axes[1].set_xlabel("x");
```

## Scaling up

Each likelihood evaluation costs the number of objects times the number of cells, or the
number of stored entries for sparse likelihoods. For larger grids:

- Store the likelihoods sparsely with `lk.to_sparse(rtol)`: objects with small errors
  cover only a few cells.
- Run the chains in parallel with `n_jobs=-1`.
- Get a quick estimate with `method="map"` before sampling.

The {doc}`performance guide <../guides/performance>` has more.
