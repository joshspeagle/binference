---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Choosing an engine

binference offers several ways to compute the same posterior. This tutorial runs
all of them on one problem and compares accuracy, speed and diagnostics. NUTS,
exact for any prior, serves as the reference.

```{code-cell} python
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

import binference

mock = binference.simulate.mock_1d(1500, binference.Bins.linear(0, 3, 20), noise=(0.1, 0.3), rng=11)
lk = mock.likelihoods
```

## Running every engine

```{code-cell} python
fits, times = {}, {}
for method in ["nuts", "gibbs", "laplace", "vb", "bootstrap"]:
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # we look at the diagnostics ourselves below
        fits[method] = binference.infer(lk, method=method, rng=1, check=False)
    times[method] = time.perf_counter() - t0
t0 = time.perf_counter()
est = binference.infer(lk, method="map")
times["map"] = time.perf_counter() - t0
print({k: f"{v:.1f} s" for k, v in times.items()})
```

## Accuracy against the reference

The fractions' standard deviations are where the engines differ. The means
mostly agree.

```{code-cell} python
ref = fits["nuts"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
x = mock.bins.centers
styles = {"gibbs": "o", "laplace": "s", "vb": "^", "bootstrap": "D"}
for m, mk in styles.items():
    axes[0].plot(x, fits[m].mean - ref.mean, mk, ms=4, label=m)
    axes[1].plot(x, fits[m].std / ref.std, mk, ms=4, label=m)
axes[0].plot(x, est.rho - ref.mean, "x", color="k", ms=5, label="map")
axes[0].axhline(0, color="0.6", lw=1)
axes[1].axhline(1, color="0.6", lw=1)
axes[0].set(xlabel="z", ylabel="mean − NUTS mean")
axes[1].set(xlabel="z", ylabel="sd / NUTS sd", yscale="log")
axes[1].legend(frameon=False, ncol=2, fontsize=8)
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
```

- **Gibbs** targets the same exact posterior. Its differences from NUTS are
  Monte Carlo noise.
- **Mean-field VB** is far too narrow. It ignores the coupling between the
  galaxies' bins and the population, which is strong when likelihoods are
  broad. binference always flags its results as approximate.
- **Laplace** is approximate, and its Pareto $\hat k$ says how far to trust
  it.
- **Bootstrap** describes the spread of the MAP estimate over resampled
  catalogues, which is a different quantity from posterior uncertainty.
- **MAP** gives a single point, and with a flat prior it can sit exactly on
  zero in some bins.

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

Both NUTS and Gibbs are exact, so compare them by **effective** samples per
second:

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

Gibbs alternates between each galaxy's bin and the population. When a
galaxy's likelihood spans several bins, its bin is poorly determined, so the
population moves slowly from one sweep to the next. The
**fraction of missing information** $f$ (near one here) measures this, and the
Gibbs autocorrelation time grows like $(1+f)/(1-f)$. NUTS moves the population
directly and does not suffer from it.

With **narrow** likelihoods the picture flips: every galaxy's bin is nearly
known, Gibbs mixes almost instantly, and a sweep costs less than a NUTS
trajectory.

```{code-cell} python
narrow = binference.simulate.mock_1d(1500, binference.Bins.linear(0, 3, 20), noise=(0.01, 0.03), rng=11)
for m in ["nuts", "gibbs"]:
    t0 = time.perf_counter()
    r = binference.infer(narrow.likelihoods, method=m, rng=1, check=False)
    dt = time.perf_counter() - t0
    print(f"{m:6s} {dt:5.1f} s, min ESS per second: {r.diagnostics['ess_bulk'].min() / dt:8.1f}")
```

## Diagnostics

Every sampled result carries diagnostics, and `binference.infer` warns
automatically when they look bad. Here is what a problem looks like. A
deliberately short Gibbs run on the broad problem triggers a warning:

```{code-cell} python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    short = binference.infer(lk, method="gibbs", rng=3, n_warmup=20, n_samples=100)
print(caught[0].message if caught else "no warnings")
```

```{code-cell} python
axes = short.plot_trace(idx=[8])
```

```{code-cell} python
:tags: [remove-cell]

assert caught and short.diagnostics["rhat"].max() > 1.1
```

The four chains have not mixed. Run longer, or use NUTS.

## Summary

| engine | exact | speed | use it for |
|---|---|---|---|
| NUTS | yes | moderate | any prior; what `method="auto"` falls back to |
| Gibbs | yes | fast per draw | conjugate priors with narrow likelihoods; joint labels |
| MAP / EM | point estimate | very fast | quick looks, huge problems |
| Laplace | approximately | fast | large problems with few bins; check $\hat k$ |
| VB | no | very fast | never for error bars |
| bootstrap | different question | slow | frequentist comparisons |
