---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Choosing an engine

binference offers several {term}`engines <engine>` that compute the same posterior. This
tutorial runs all of them on one problem, and compares their accuracy, speed and
diagnostics. {term}`NUTS`, which is exact for any prior, is the reference.

```{code-cell} python
:tags: [remove-cell]

# figures 6 in wide with 17 pt text keep text above 12 px on a 390 px phone screen
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.figsize": (6, 3.4), "font.size": 17,
                     "figure.constrained_layout.use": True})
```

```{code-cell} python
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

import binference

bins = binference.Bins.linear(0, 3, 20)
mock = binference.simulate.mock_1d(1500, bins, noise=(0.1, 0.3), rng=11)
lk = mock.likelihoods
```

## Running every engine

```{code-cell} python
fits, times = {}, {}
for method in ["nuts", "gibbs", "laplace", "vb", "bootstrap"]:
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # we look at the diagnostics ourselves below
        fits[method] = binference.infer(lk, method=method, rng=1, n_samples=500, check=False)
    times[method] = time.perf_counter() - t0
t0 = time.perf_counter()
est = binference.infer(lk, method="map")
times["map"] = time.perf_counter() - t0
print({k: f"{v:.1f} s" for k, v in times.items()})
```

## Accuracy against the reference

The engines agree on the means and differ most in the standard deviations.

```{code-cell} python
---
mystnb:
  figure:
    caption: Each engine's posterior mean minus the NUTS mean (top), and its posterior
      standard deviation over the NUTS one (bottom), per bin. The MAP estimate has no
      standard deviation.
  image:
    alt: Two stacked scatter plots against z. Top, the differences from the NUTS mean
      lie close to zero for Gibbs, Laplace and VB. They scatter by up to 0.07 for MAP and
      0.04 for the bootstrap. Bottom, on a log scale, Gibbs and Laplace lie near one. The
      bootstrap lies near one except in the sparse bins above z of 2.4. VB lies between
      0.1 and 0.5 in most bins.
---
ref = fits["nuts"]
fig, axes = plt.subplots(2, 1, figsize=(6, 7.2), sharex=True)
zc = bins.centers
styles = {"gibbs": "o", "laplace": "s", "vb": "^", "bootstrap": "D"}
for m, mk in styles.items():
    axes[0].plot(zc, fits[m].mean - ref.mean, mk, ms=6, label=m)
    axes[1].plot(zc, fits[m].std / ref.std, mk, ms=6, label=m)
axes[0].plot(zc, est.rho - ref.mean, "x", color="k", ms=7, mew=2, label="map")
axes[0].axhline(0, color="0.6", lw=1)
axes[1].axhline(1, color="0.6", lw=1)
axes[0].set(ylabel="mean − NUTS")
axes[1].set(xlabel="z", ylabel="sd / NUTS sd", yscale="log")
axes[1].set_yticks([0.1, 0.2, 0.5, 1, 2], ["0.1", "0.2", "0.5", "1", "2"])
axes[1].minorticks_off()
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3);
```

- **{term}`Gibbs`** targets the same exact posterior. Its differences from NUTS are
  Monte Carlo noise.
- **{term}`Mean-field VB <mean-field VB>`** is far too narrow. It ignores the coupling
  between the objects' bins and the population, which is strong when likelihoods are
  broad. binference always flags its results as approximate.
- **{term}`Laplace`** is approximate, and its {term}`Pareto k-hat` $\hat k$ says how far
  to trust it.
- **The {term}`bootstrap`** describes the spread of the MAP estimate over resampled
  catalogues. That is a different quantity from the posterior uncertainty.
- **{term}`MAP`** gives a single point. With a flat prior it can sit exactly on zero in
  some bins.

```{code-cell} python
print("Laplace k-hat:", round(fits["laplace"].info["k_hat"], 2),
      "approximate:", fits["laplace"].approximate)
print("VB approximate:", fits["vb"].approximate)
print("MAP optimality gap:", f"{est.kkt_gap:.1e}")
```

```{code-cell} python
:tags: [remove-cell]

assert np.median(fits["vb"].std / ref.std) < 0.5
assert np.abs((fits["gibbs"].mean - ref.mean) / ref.std).max() < 0.5
```

## Efficiency of the exact samplers

NUTS and Gibbs are both exact, so compare them by {term}`effective samples <effective
sample size>` per second:

```{code-cell} python
for m in ["nuts", "gibbs"]:
    ess = fits[m].diagnostics["ess_bulk"]
    print(f"{m:6s} min bulk ESS {ess.min():7.0f}  ->  {ess.min() / times[m]:7.1f} per second")
model = binference.PopulationModel(lk)
print("fraction of missing information:", round(model.missing_information(est.rho), 3))
```

```{code-cell} python
:tags: [remove-cell]

assert model.missing_information(est.rho) > 0.9
```

The fraction of {term}`missing information` printed above is close to one, so Gibbs
mixes slowly here ([Gibbs sampling](../guides/engines.md#gibbs-sampling)).

With narrow likelihoods every object's bin is nearly known, and Gibbs is the faster
sampler:

```{code-cell} python
narrow = binference.simulate.mock_1d(1500, bins, noise=(0.01, 0.03), rng=11)
per_second = {}
for m in ["nuts", "gibbs"]:
    t0 = time.perf_counter()
    r = binference.infer(narrow.likelihoods, method=m, rng=1, n_samples=500, check=False)
    dt = time.perf_counter() - t0
    per_second[m] = r.diagnostics["ess_bulk"].min() / dt
    print(f"{m:6s} {dt:5.1f} s, min ESS per second: {per_second[m]:8.1f}")
```

```{code-cell} python
:tags: [remove-cell]

assert per_second["gibbs"] > per_second["nuts"]
```

## Diagnostics

Every sampled result carries diagnostics, and `binference.infer` warns when they look
bad. Here is what a problem looks like. A deliberately short Gibbs run on the broad
problem triggers a warning:

```{code-cell} python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    short = binference.infer(lk, method="gibbs", rng=3, n_warmup=20, n_samples=100)
print(caught[0].message if caught else "no warnings")
```

```{code-cell} python
---
mystnb:
  figure:
    caption: The fraction in bin 8 over 100 iterations of each of the four chains of the
      short Gibbs run.
  image:
    alt: Four wiggly lines over 100 iterations. Each drifts slowly and they sit at
      different levels, so they do not overlap well.
---
fig = plt.figure(figsize=(6, 3.6), layout="tight")
short.plot_trace(idx=[8], fig=fig);
```

```{code-cell} python
:tags: [remove-cell]

assert caught and short.diagnostics["rhat"].max() > 1.1
```

The four {term}`chains <chain>` have not mixed. Run longer, or use NUTS.

The [engines guide](../guides/engines.md#choosing-an-engine) says which engine to use
when, and what `method="auto"` chooses.
