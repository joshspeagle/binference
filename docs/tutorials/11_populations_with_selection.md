---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# A mass distribution with a selection measured by injections

Gravitational-wave detectors see heavy binary black holes to much larger
distances than light ones, so a catalogue of detections over-represents
heavy systems. Population analyses correct for this with the detection
probability $p_{\rm det}(m)$, which is not known in closed form: it is
estimated by injecting simulated signals of known mass into the detector
data and recording which are recovered. Exoplanet surveys measure their
completeness the same way.

{doc}`05_selection` uses a completeness that is known exactly. Here the
selection is measured by a finite set of injections, so it is uncertain,
and binference infers it jointly with the population
({meth}`~binference.Selection.from_injections`). The tutorial compares
ignoring the selection, plugging in the injection estimate, and inferring
both together.

```{code-cell} python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import binference

plt.rcParams["figure.dpi"] = 110
rng = np.random.default_rng(150914)
```

## A simulated catalogue

Primary masses of the mergers follow a power law $p(m)\propto m^{-2.3}$
between 5 and 85 solar masses (90% of systems) plus a Gaussian peak at
34 solar masses with width 4 (10%). A merger is detected with probability
$p_{\rm det}(m) = 0.8/[1 + (m/20)^{-3}]$, and detection is a coin flip
independent of the mass measurement, so the selection acts on the true
mass (`"on_truth"`). Each detected merger has a measured $\ln m$ with a
Gaussian error between 0.05 and 0.2.

```{code-cell} python
m_lo, m_hi, slope = 5.0, 85.0, -2.3

def draw_masses(n):
    a = slope + 1
    power_law = (m_lo**a + rng.random(n) * (m_hi**a - m_lo**a)) ** (1 / a)
    peak = np.clip(rng.normal(34.0, 4.0, n), m_lo, m_hi)
    return np.where(rng.random(n) < 0.1, peak, power_law)

def p_det(m):
    return 0.8 / (1 + (np.asarray(m) / 20.0) ** -3)

n_mergers = 6000
m_true = draw_masses(n_mergers)
detected = rng.random(n_mergers) < p_det(m_true)
sigma = rng.uniform(0.05, 0.2, detected.sum())
x_obs = np.log(m_true[detected]) + sigma * rng.standard_normal(detected.sum())
print(f"{n_mergers} mergers, {detected.sum()} detected")
```

The bins are 16 equal intervals in $\ln m$. The true population in them
follows from the two components' distribution functions:

```{code-cell} python
bins = binference.Bins.linear(np.log(m_lo), np.log(m_hi), 16)
edges = np.exp(bins.edges[0])
a = slope + 1
pl_cdf = (edges**a - m_lo**a) / (m_hi**a - m_lo**a)
rho_true = 0.9 * np.diff(pl_cdf) + 0.1 * np.diff(stats.norm.cdf(edges, 34.0, 4.0))
lk = binference.Likelihoods.from_gaussian(x_obs, sigma, bins)
```

## The injection campaign

3000 signals are injected with masses drawn uniformly in $\ln m$, and each
is recovered with probability $p_{\rm det}$. In the lightest bins only a
few injections are recovered, so the detection probability there is known
only roughly. `Selection.from_injections` tallies them per bin and gives
each bin's detection probability a Beta posterior (see the
[user guide](../user-guide/selection)); `sel.injections` holds the tallies.

```{code-cell} python
n_inj = 3000
m_inj = np.exp(rng.uniform(np.log(m_lo), np.log(m_hi), n_inj))
found = rng.random(n_inj) < p_det(m_inj)
sel = binference.Selection.from_injections(np.log(m_inj), found, bins, "on_truth")
injected, recovered, (a_prior, b_prior), _ = sel.injections
print("recovered / injected, lightest five bins:",
      [f"{r}/{n}" for r, n in zip(recovered[:5], injected[:5])])
```

## Four fits

1. **No selection.** The catalogue is treated as a fair sample.
2. **Plug-in.** The injection estimate `sel.at_mean()` is used as if it
   were exact.
3. **Joint.** The detection probabilities are inferred together with the
   population, from the injections and the catalogue.
4. **Exact.** For reference, the true $p_{\rm det}$ averaged over each bin
   ({meth}`~binference.Selection.from_function`), which a real analysis
   does not have.

All four use NUTS so that they are comparable. In the joint model the
population and the detection probability trade off against each other in
the lightest bins, where there are few detections.

```{code-cell} python
exact = binference.Selection.from_function(lambda x: p_det(np.exp(x)), bins, "on_truth")
kw = dict(method="nuts", n_samples=500, n_warmup=500, rng=1)
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
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, constrained_layout=True)
for ax, name in zip(axes, ["no selection", "plug-in injections", "joint"]):
    fits[name].plot(ax=ax, which="fractions", truth=rho_true)
    ax.set_title(name, loc="left", fontsize=10)
    ax.set_xlabel(r"$\ln(m / M_\odot)$")
for ax in axes[1:]:
    ax.get_legend().remove()
```

```{code-cell} python
:tags: [remove-cell]

assert (fits["joint"].std[:3] > fits["plug-in injections"].std[:3]).all()
```

Ignoring the selection gives the population of *detections*, heavily
weighted toward high masses. Both corrected fits recover the shape. They
differ in the lightest bins, where few injections were recovered: the
joint fit's bands are wider there.

## Derived quantities

The fraction of mergers lighter than 10 solar masses, and the total number
of mergers including those never detected (`result.total`, see the [user
guide](../user-guide/selection)), depend most on the lightest bins:

```{code-cell} python
light = bins.centers < np.log(10.0)
frac = {name: res.functional(lambda rho: rho[light].sum()) for name, res in fits.items()}
print(f"{'':20s} fraction below 10 Msun   total mergers")
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
assert all(tot[k].mean() < n_mergers for k in ("plug-in injections", "joint", "exact selection"))
```

Without a selection, `total` is just the number of detections. The joint
fit's error bars are larger than the plug-in fit's, because they include
the uncertainty of the detection probability from a finite number of
injections; the plug-in fit treats that probability as exact. All three
corrected totals fall below the truth, for two reasons. This injection set
recovered more light signals than $p_{\rm det}$ predicts, which lowers the
plug-in and joint totals only:

```{code-cell} python
expected = injected * exact.values
print("lightest five bins, recovered:", recovered[:5].tolist())
print("expected from p_det:          ", expected[:5].round(1).tolist())
```

```{code-cell} python
:tags: [remove-cell]

assert recovered[:5].sum() > expected[:5].sum()
```

And the default prior pulls a poorly determined total down, which lowers
all three ([user guide](../user-guide/selection)); here the lightest bins
hold few detections. Before quoting a total from such data, check its
calibration on simulations ({doc}`07_validation`).

The joint fit also returns posterior draws of the detection probability
(`info["selection_draws"]`). The catalogue informs only the ratios between
bins, so its level comes from the injections:

```{code-cell} python
S_draws = fits["joint"].info["selection_draws"].reshape(-1, bins.K)
S_lo, S_hi = np.quantile(S_draws, [0.05, 0.95], axis=0)
beta = stats.beta(a_prior + recovered, b_prior + injected - recovered)
fine = np.linspace(np.log(m_lo), np.log(m_hi), 400)

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(fine, p_det(np.exp(fine)), "k--", lw=1.2, label=r"true $p_{\rm det}$")
for dx, mid, lo, hi, color, label in [
    (-0.02, beta.mean(), beta.ppf(0.05), beta.ppf(0.95), "#eb6834", "injections alone (90%)"),
    (0.02, S_draws.mean(axis=0), S_lo, S_hi, "#1baf7a", "joint fit (90%)"),
]:
    ax.errorbar(bins.centers + dx, mid, yerr=[mid - lo, hi - mid], fmt="o", color=color,
                ms=4, capsize=2, label=label)
ax.set(xlabel=r"$\ln(m / M_\odot)$", ylabel="detection probability", yscale="log")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8);
```

## What to take away

- `Selection.from_injections` passed to `infer` infers the selection jointly
  with the population; `sel.at_mean()` is a plug-in that ignores the
  selection's uncertainty, which is large where few injections are
  recovered.
- Ask what the selection acts on. Here detection is a coin flip that depends
  only on the true mass, so `"on_truth"` is exact. In real detectors
  detection and measurement share the same noise, and the selection acts on
  the measurement ({doc}`05_selection`; the
  [gravitational-wave background page](../background/gravitational-waves)
  says what that means for injection-based selection).
