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

## Running the engines

```{code-cell} python
fits, times = {}, {}
for method in ["nuts", "laplace", "vb", "bootstrap"]:
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

```{code-cell} python
:tags: [remove-cell]

assert not fits["nuts"].check()              # the reference has converged
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
      lie close to zero for Laplace and VB. They scatter by up to 0.07 for MAP and 0.04
      for the bootstrap. Bottom, on a log scale, Laplace lies near one. The bootstrap
      lies near one except in the sparse bins above z of 2.4. VB lies between 0.1 and
      0.5 in most bins.
---
ref = fits["nuts"]
fig, axes = plt.subplots(2, 1, figsize=(6, 7.2), sharex=True)
zc = bins.centers
styles = {"laplace": "s", "vb": "^", "bootstrap": "D"}
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
```

## Gibbs on this problem

{term}`Gibbs` is exact too, but on this problem it mixes slowly. Every sampled result
carries diagnostics, and `binference.infer` warns when they look bad. A Gibbs run with
the default settings triggers a warning:

```{code-cell} python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    gibbs = binference.infer(lk, method="gibbs", rng=1)
print(caught[0].message if caught else "no warnings")
model = binference.PopulationModel(lk)
print("fraction of missing information:", round(model.missing_information(est.rho), 3))
```

```{code-cell} python
:tags: [remove-cell]

assert caught and gibbs.diagnostics["rhat"].max() > 1.05
assert model.missing_information(est.rho) > 0.9
```

The fraction of {term}`missing information` printed above is close to one, which is why
Gibbs mixes slowly here ([Gibbs sampling](../guides/engines.md#gibbs-sampling)). The
trace of the bin with the largest R-hat shows it:

```{code-cell} python
---
mystnb:
  figure:
    caption: The fraction in the bin with the largest R-hat, over the iterations of each
      of the four chains of the Gibbs run.
  image:
    alt: Four wiggly lines over 1000 iterations. Each wanders slowly over a wide range,
      and the chains spend long stretches at different levels.
---
worst = int(np.argmax(gibbs.diagnostics["rhat"]))
fig = plt.figure(figsize=(6, 3.6), layout="tight")
gibbs.plot_trace(idx=[worst], fig=fig);
```

The four {term}`chains <chain>` have not mixed. Run much longer, or use NUTS.

With narrow likelihoods every object's bin is nearly known, and Gibbs is the faster
sampler. Both runs below pass their diagnostics; the comparison is in
{term}`effective samples <effective sample size>` per second:

```{code-cell} python
narrow = binference.simulate.mock_1d(1500, bins, noise=(0.01, 0.03), rng=11)
per_second, narrow_fits = {}, {}
for m in ["nuts", "gibbs"]:
    t0 = time.perf_counter()
    narrow_fits[m] = r = binference.infer(narrow.likelihoods, method=m, rng=1, n_samples=500,
                                          check=False)
    dt = time.perf_counter() - t0
    per_second[m] = r.diagnostics["ess_bulk"].min() / dt
    print(f"{m:6s} {dt:5.1f} s, min ESS per second: {per_second[m]:8.1f}")
```

```{code-cell} python
:tags: [remove-cell]

assert not narrow_fits["nuts"].check() and not narrow_fits["gibbs"].check()
assert per_second["gibbs"] > per_second["nuts"]
```

The [engines guide](../guides/engines.md#choosing-an-engine) says which engine to use
when, and what `method="auto"` chooses.
