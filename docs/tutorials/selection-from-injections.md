---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Selection measured by injections

Gravitational-wave detectors see heavy binary black holes to much larger distances than
light ones. A catalogue of detections therefore over-represents heavy systems. The
correction needs the detection probability $S(m)$ at each mass $m$, which is not known in
closed form. It is measured with {term}`injections`: simulated signals of known mass are
added to the detector data, and the pipeline records which it recovers. Exoplanet surveys
measure their completeness the same way.

{doc}`selection-effects` uses a selection that is known exactly. Here a finite set of
injections measures it, so it is uncertain. binference infers it together with the
population ({meth}`~binference.Selection.from_injections`). The tutorial compares three
fits: one that ignores the selection, one that plugs in the injection estimate, and one
that infers both together.

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
from scipy import stats

import binference

rng = np.random.default_rng(150914)
```

## A simulated catalogue

The primary masses of the mergers follow a power law $p(m)\propto m^{-2.3}$ between 5
and 85 solar masses (90% of mergers), plus a Gaussian peak at 34 solar masses with width
4 (10%). A merger is detected with probability $S(m) = 0.8/[1 + (m/20)^{-3}]$.
Detection is a coin flip independent of the mass measurement, so the selection acts on
the true mass (`"on_truth"`). Each detected merger has a measured $\ln m$ with a Gaussian
error between 0.05 and 0.2. Gaussian likelihoods keep this tutorial about the selection;
{doc}`posterior-samples` starts from samples instead.

```{code-cell} python
m_lo, m_hi, slope = 5.0, 85.0, -2.3

def draw_masses(n):
    a = slope + 1
    power_law = (m_lo**a + rng.random(n) * (m_hi**a - m_lo**a)) ** (1 / a)
    peak = np.clip(rng.normal(34.0, 4.0, n), m_lo, m_hi)
    return np.where(rng.random(n) < 0.1, peak, power_law)

def S(m):
    return 0.8 / (1 + (np.asarray(m) / 20.0) ** -3)

n_mergers = 6000
m_true = draw_masses(n_mergers)
detected = rng.random(n_mergers) < S(m_true)
sigma = rng.uniform(0.05, 0.2, detected.sum())
x_obs = np.log(m_true[detected]) + sigma * rng.standard_normal(detected.sum())
print(f"{n_mergers} mergers, {detected.sum()} detected")
```

The bins are 16 equal intervals in $\ln m$. The true population fractions follow from
the distribution functions of the two components:

```{code-cell} python
bins = binference.Bins.linear(np.log(m_lo), np.log(m_hi), 16)
edges = np.exp(bins.edges[0])
a = slope + 1
pl_cdf = (edges**a - m_lo**a) / (m_hi**a - m_lo**a)
rho_true = 0.9 * np.diff(pl_cdf) + 0.1 * np.diff(stats.norm.cdf(edges, 34.0, 4.0))
lk = binference.Likelihoods.from_gaussian(x_obs, sigma, bins)
```

## The injection campaign

The campaign injects 3000 signals with masses drawn uniformly in $\ln m$. Each is
recovered with probability $S(m)$. In the lightest bins only a few injections are
recovered, so the detection probability there is known only roughly.
`Selection.from_injections` tallies the injected and recovered signals per bin,
$n^{\rm inj}_k$ and $n^{\rm rec}_k$. It gives each bin's detection probability $S_k$ a
Beta posterior
([selection guide](../guides/selection.md#selection-measured-by-injections)).
`sel.injections` holds the tallies.

```{code-cell} python
n_inj = 3000
m_inj = np.exp(rng.uniform(np.log(m_lo), np.log(m_hi), n_inj))
found = rng.random(n_inj) < S(m_inj)
sel = binference.Selection.from_injections(np.log(m_inj), found, bins, "on_truth")
injected, recovered, (a_prior, b_prior), _ = sel.injections
print("recovered / injected, lightest five bins:",
      [f"{r}/{n}" for r, n in zip(recovered[:5], injected[:5])])
```

## Four fits

1. **No selection.** The catalogue is treated as a fair sample.
2. **Plug-in.** The injection estimate `sel.at_mean()` is used as if it were exact.
3. **Joint.** The detection probabilities are inferred together with the population,
   from the injections and the catalogue.
4. **Exact.** For reference, the true $S(m)$ averaged over each bin
   ({meth}`~binference.Selection.from_function`). A real analysis does not have it.

All four use NUTS, so that they are comparable. In the joint model the population and
the detection probability trade off against each other in the lightest bins, where there
are few detections.

```{code-cell} python
exact = binference.Selection.from_function(lambda x: S(np.exp(x)), bins, "on_truth")
kw = dict(method="nuts", n_warmup=500, rng=1)
fits = {
    "no selection": binference.infer(lk, **kw),
    "plug-in injections": binference.infer(lk, selection=sel.at_mean(), **kw),
    "joint": binference.infer(lk, selection=sel, **kw),
    "exact selection": binference.infer(lk, selection=exact, **kw),
}
d = fits["joint"].diagnostics
print(f"joint fit: max R-hat {d['rhat'].max():.3f}, min bulk ESS {d['ess_bulk'].min():.0f}, "
      f"selection max R-hat {d['rhat_selection'].max():.3f}, divergent {d['n_divergent']}")
```

```{code-cell} python
:tags: [remove-cell]

assert max(d["rhat"].max(), d["rhat_selection"].max()) < 1.01 and d["n_divergent"] == 0
```

```{code-cell} python
---
mystnb:
  figure:
    caption: Population fractions of the primary mass (median, 68% and 95% bands) from
      fits that ignore the selection, plug in the injection estimate, and infer it
      jointly. The truth is dashed.
  image:
    alt: Three stacked step plots of the fraction per bin against ln m from 1.6 to 4.4.
      In the top one, without selection, the light bins fall far below the dashed truth
      and the heavy ones rise above it. In the other two the bands follow the truth; the
      joint fit's bands are wider in the lightest bins.
---
fig, axes = plt.subplots(3, 1, figsize=(6, 9), sharex=True, sharey=True)
for ax, name in zip(axes, ["no selection", "plug-in injections", "joint"]):
    fits[name].plot(ax=ax, which="fractions", truth=rho_true, label="median")
    ax.set_title(name, loc="left")
    ax.set_xlabel("")
    ax.get_legend().remove()
axes[1].legend(loc="upper right")
axes[-1].set_xlabel(r"$\ln(m / M_\odot)$");
```

```{code-cell} python
:tags: [remove-cell]

assert (fits["joint"].std[:3] > fits["plug-in injections"].std[:3]).all()
```

Ignoring the selection gives the population of detections, which is heavily weighted
toward high masses. Both corrected fits recover the shape. They differ in the lightest
bins, where few injections were recovered: the joint fit's bands are wider there.

## Derived quantities

Two derived quantities depend most on the lightest bins. One is the fraction of mergers
in the four lightest bins, below 10.2 solar masses. The other is the {term}`total`
number of mergers $\Lambda$, including those never detected (`result.total`):

```{code-cell} python
light = np.arange(bins.K) < 4                  # upper edge at 10.15 Msun
frac = {name: res.functional(lambda rho: rho[light].sum()) for name, res in fits.items()}
print(f"{'':20s} fraction below 10.2 Msun total mergers")
for name, res in fits.items():
    f = frac[name]
    print(f"{name:20s} {f.mean():.3f} ± {f.std():.3f}            "
          f"{res.total.mean():6.0f} ± {res.total.std():4.0f}")
print(f"{'truth':20s} {rho_true[light].sum():.3f}                    {n_mergers:6d}")
```

```{code-cell} python
:tags: [remove-cell]

tot = {k: r.total for k, r in fits.items()}
assert abs(tot["no selection"].mean() - detected.sum()) < 3 * tot["no selection"].std()
assert tot["joint"].std() > tot["plug-in injections"].std()
assert frac["joint"].std() > frac["plug-in injections"].std()
assert all(n_mergers - tot[k].mean() > 2 * tot[k].std() for k in ("plug-in injections", "joint"))
assert abs(tot["exact selection"].mean() - n_mergers) < 3 * tot["exact selection"].std()
```

Without a selection, the total is just the number of detections. The joint fit's error
bars are larger than the plug-in fit's. They include the uncertainty of the detection
probability from a finite number of injections, which the plug-in fit treats as exact.
Both fall below the truth, because this injection set recovered more light signals than
$S(m)$ predicts. The exact-selection fit does not share that error:

```{code-cell} python
expected = injected * exact.values
print("lightest five bins, recovered:", recovered[:5].tolist())
print("expected from S(m):            ", expected[:5].round(1).tolist())
```

```{code-cell} python
:tags: [remove-cell]

assert recovered[:5].sum() > expected[:5].sum()
```

A total this poorly determined also depends on the prior, which by default pulls it
down ([selection guide](../guides/selection.md#rates-and-exposure)). Before you quote a
total from such data, check its calibration on simulations
({doc}`validating-an-analysis`).

The joint fit also returns posterior draws of the detection probability,
`info["selection_draws"]`. The catalogue informs only the ratios between bins, so the
level comes from the injections:

```{code-cell} python
---
mystnb:
  figure:
    caption: The detection probability per bin with 90% intervals, from the injections
      alone (squares) and from the joint fit (circles), and the true S(m) (dashed).
  image:
    alt: Plot of detection probability on a log scale against ln m. The dashed truth
      rises from about 0.01 at the lightest masses to 0.8 at the heaviest. Both sets of
      intervals follow it; in the lightest bins they are long, and the joint fit's
      circles lie close to the injections' squares.
---
S_draws = fits["joint"].info["selection_draws"].reshape(-1, bins.K)
S_lo, S_hi = np.quantile(S_draws, [0.05, 0.95], axis=0)
beta = stats.beta(a_prior + recovered, b_prior + injected - recovered)
fine = np.linspace(np.log(m_lo), np.log(m_hi), 400)

fig, ax = plt.subplots(figsize=(6, 4.8))
ax.plot(fine, S(np.exp(fine)), "k--", lw=1.2, label="truth")
for dx, mid, lo, hi, color, marker, label in [
    (-0.02, beta.mean(), beta.ppf(0.05), beta.ppf(0.95), "C1", "s", "injections alone"),
    (0.02, S_draws.mean(axis=0), S_lo, S_hi, "C0", "o", "joint fit"),
]:
    ax.errorbar(bins.centers + dx, mid, yerr=[mid - lo, hi - mid], fmt=marker, color=color,
                ms=6, capsize=2, label=label)
ax.set(xlabel=r"$\ln(m / M_\odot)$", ylabel="detection probability", yscale="log")
ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1));
```

## What to take away

- `Selection.from_injections` passed to `infer` infers the selection together with the
  population. `sel.at_mean()` is a plug-in that ignores the selection's uncertainty,
  which is large where few injections are recovered.
- Ask what the selection acts on. Here detection is a coin flip that depends only on
  the true mass, so `"on_truth"` is exact. In real detectors, detection and measurement
  share the same noise, and the selection acts on the measurement
  ({doc}`selection-effects`). The
  [gravitational-wave background page](../fields/gravitational-waves) says what that
  means for a selection measured by injections.
