---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Check a result

Check a result in two stages. First, the diagnostics say whether the engine explored
the posterior. Second, simulations say whether the posterior is *right*: whether your
likelihoods, prior and engine together give calibrated answers. binference ships the
tools for both, the same ones its own test suite uses.

```python
import numpy as np
import binference
from binference import priors
from binference.validation import coverage, exact_posterior, sbc

mock = binference.simulate.mock_1d(800, binference.Bins.linear(0, 3, 15), rng=6)
result = binference.infer(mock.likelihoods, method="nuts", rng=1)
```

## Diagnostics

`binference.infer` checks every NUTS and Gibbs result and warns about any problem. Call
`result.check()` yourself for other engines, or to change the limits:

```python
d = result.diagnostics                     # rhat, ess_bulk, ess_tail, n_divergent, ...
print(d["rhat"].max().round(3), d["ess_bulk"].min().round(0), d.get("n_divergent"))
problems = result.check()                  # warns, and returns the messages
print(problems)
```

`check()` raises a `ConvergenceWarning` for these problems. The limits are on the
{doc}`defaults <../reference/defaults>` page.

- **{term}`R-hat`** above its limit: the chains disagree. binference splits each chain
  in half, so R-hat works for a single chain too.
- **Bulk or tail {term}`ESS`** below its limit per chain: too few effective draws.
  This applies to any Markov chain, including draws wrapped with
  `PopulationResult.from_samples`.
- **NUTS health**: {term}`divergent transitions <divergent transition>`, saturated
  {term}`tree depth`, and low {term}`E-BFMI`.
- **Importance weights** whose {term}`Pareto k-hat` is too large, and an optimiser that
  did not converge.

If the chains disagree or the effective sample sizes are low, run longer. If Gibbs
mixes slowly, switch to NUTS. A few isolated divergences among thousands of draws are
usually harmless if R-hat and the effective sample sizes pass. Divergences that persist
with a higher `target_accept` mark a region the sampler cannot explore, such as a
[funnel](troubleshooting.md#a-smoothness-scale-near-zero). Change the prior or its
hyperprior rather than ignore them.

A low E-BFMI means that NUTS moves slowly through the energy distribution, so it may
under-explore the tails. A longer warm-up or another prior may help; a higher
`target_accept` does not. Otherwise compare the fit with a longer run before you rely
on its tails.

Problems with the inputs raise an `InputWarning` instead. Three are checked:

- the Monte Carlo noise of likelihoods built from samples ({doc}`inputs/samples`);
- binned counts that the response does not describe ({doc}`inputs/binned`);
- with `check(support_min=...)`, bins whose {term}`data support` is low ({doc}`bins`).

Where it can, a warning ends with a hint. After Gibbs, the hint gives the
{term}`missing information` and the autocorrelation time it implies.
`check(strict=True)` raises a `binference.DiagnosticsError` instead of warning. Every
message, its cause and its fix are on the {doc}`warnings page
<../reference/warnings>`. A result from an approximate engine has
`result.approximate` set, and its summary says so.

A point estimate has its own check. `est.check()` warns when the optimiser did not
converge, or when the {term}`KKT gap` per object exceeds `kkt_max`:

```python
est = binference.infer(mock.likelihoods, method="map")
print(est.check(), est.kkt_gap)
```

## What good and bad look like

```{code-cell} python
---
tags: [remove-input]
mystnb:
  image:
    alt: >-
      Top: trace plots of one bin's fraction for four Gibbs chains. With narrow
      likelihoods the chains overlap as fast noise around one value, with R-hat near
      one and a large ESS. With broad likelihoods they wander slowly and apart, with a
      large R-hat and a tiny ESS. Bottom: four SBC rank histograms of 200 simulations
      each, with a grey band for uniform ranks. The calibrated pipeline stays inside
      the band. Posteriors too narrow give a U shape, too wide a hump, and biased ones
      a slope from low ranks down to high ones.
  figure:
    caption: >-
      Top: four Gibbs chains for one bin, 1000 objects in ten bins. With narrow
      likelihoods successive draws are nearly independent. With broad ones the
      missing information is near one, and the chains have not mixed. Bottom: SBC
      rank histograms of one bin, from 200 simulations of 100 objects in three bins,
      fitted by Laplace. The pipelines count each object three times (too narrow), a
      quarter (too wide), or treat posteriors under a prior favouring the bin as
      likelihoods (biased). Each bar of a uniform histogram falls inside the grey band 99% of the time.
---
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import binference
from binference import priors
from binference.validation import sbc

warnings.simplefilter("ignore", binference.BinferenceWarning)
ink = plt.rcParams["text.color"]

ten = binference.Bins.linear(0.0, 3.0, 10)
traces = {}
for name, noise in [("narrow likelihoods", 0.05), ("broad likelihoods", 0.6)]:
    lk_t = binference.simulate.mock_1d(1000, ten, noise=noise, rng=4).likelihoods
    traces[name] = binference.infer(lk_t, method="gibbs", n_samples=500, rng=1, check=False)

dirichlet = priors.Dirichlet(1.0)
simulate_3 = binference.simulate.prior_predictive(
    dirichlet, binference.Bins.linear(0, 1, 3), n_objects=100, noise=0.1
)


def pipeline(scale=1.0, interim=None):
    def fit(lk, rng):
        if interim is not None:  # posteriors passed as likelihoods
            lk = binference.Likelihoods(lk.toarray() * interim, lk.bins)
        if scale != 1.0:  # each object counted `scale` times
            lk = lk.with_weights(lk.weights * scale, fractional_weights=True)
        return binference.infer(lk, dirichlet, method="laplace", n_samples=200, rng=rng,
                                check=False)
    return fit


pipelines = {
    "calibrated": pipeline(),
    "too narrow": pipeline(scale=3.0),
    "too wide": pipeline(scale=0.25),
    "biased": pipeline(interim=np.array([2.0, 1.0, 1.0])),
}
ranks = {name: sbc(simulate_3, fit, n_simulations=200, n_draws=19, log_likelihood=False,
                   rng=0) for name, fit in pipelines.items()}

with plt.rc_context({"font.size": 12}):
    fig = plt.figure(figsize=(5.0, 8.6), constrained_layout=True)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.25])
    axes = top.subplots(2, 1, sharex=True)
    for a, (name, r) in zip(axes, traces.items()):
        for c in range(r.chains.shape[0]):
            a.plot(r.chains[c, :, 4], lw=0.8, alpha=0.8, color=f"C{c}")
        d = r.diagnostics
        a.set_title(f"{name}: $\\hat R$ = {d['rhat'].max():.2f}, "
                    f"ESS = {d['ess_bulk'].min():.0f}", loc="left", fontsize=12)
        a.set_ylabel("fraction, bin 5")
    axes[1].set_xlabel("draw")
    grid = bottom.subplots(2, 2, sharex=True, sharey=True)
    for a, (name, s) in zip(grid.flat, ranks.items()):
        n_sim = s.ranks.shape[0]
        lo, hi = stats.binom.interval(0.99, n_sim, s.expected(10)[0] / n_sim)
        a.axhspan(lo, hi, color=ink, alpha=0.1, lw=0)
        a.stairs(s.histogram(10)[0], np.arange(11), color="C0", fill=True, alpha=0.7)
        a.set_title(name, loc="left", fontsize=12)
        a.set_xticks([0, 10], ["low", "high"])
    for a in grid[1]:
        a.set_xlabel("rank of the truth")
    for a in grid[:, 0]:
        a.set_ylabel("simulations")
    plt.show()
```

**Traces.** Good chains overlap and look like noise around one level. Bad chains
wander slowly, or sit at different levels. If they do, run longer or switch engine
({doc}`engines`).

**Rank histograms.** {term}`SBC` ranks the truth among the posterior draws. If the
pipeline is calibrated, the ranks are uniform:

- a **U shape** means the intervals are too narrow;
- a **hump** means they are too wide;
- a **slope** means the posterior is biased.

## Simulation-based calibration

Simulation-based calibration (SBC; {cite:alp}`talts2018`) fits many catalogues
simulated from the prior, and ranks the truth among each fit's draws. How it works is
in the [theory](../how-it-works/validation.md). Read the rank
histograms as in the figure above.

```python
prior = priors.Dirichlet(1.0)
simulate = binference.simulate.prior_predictive(prior, binference.Bins.linear(0, 1, 4),
                                            n_objects=40, noise=0.2)

def fit(lk, rng):
    return binference.infer(lk, prior, method="gibbs", rng=rng, n_chains=4,
                            n_warmup=100, n_samples=600, check=False)

result = sbc(simulate, fit, n_simulations=40, n_draws=19, rng=0)
print(result.names, result.pvalues().round(2), result.passed())
fig = result.plot()
```

- **`simulate(rng)`** returns `(likelihoods, true_rho)`.
  `simulate.prior_predictive` builds one from a prior, optionally with a known
  selection in either mode (`selection=`, `selection_mode=`), and you can write
  your own to include your real selection or noise model. For rate models,
  return the true rates (as `prior_predictive` does, drawing the number of
  objects from a Poisson): ranks and coverage are then computed on the rates.
- **`fit(likelihoods, rng)`** returns a result. Use the settings you will use
  in production.
- **`log_likelihood`.** Besides each bin, SBC ranks the joint log-likelihood of the true
  population. Pass `log_likelihood=False` to leave it out.
- **Tests.** `result.pvalues()` and `result.passed()` test each quantity's ranks with
  a chi-square on their histogram. `method="ecdf"` uses the ECDF test of
  {cite:t}`sailynoja2022` instead. To require both, call
  `result.passed(method="chisquare")` and `result.passed(method="ecdf")`.
- **Size.** A real check needs a few hundred simulations and roughly
  independent draws: keep `n_draws` below the effective sample size.

binference's own slow tests run SBC on its engines and priors
([theory](../how-it-works/validation.md)).

## Coverage

{func}`~binference.validation.coverage` measures how often central credible
intervals contain the truth, for each bin and for any functional. It answers the
frequentist question: do my 68% error bars contain the truth 68% of the time?

```python
centers = binference.Bins.linear(0, 1, 4).centers
cov = coverage(simulate, fit, n_simulations=40, levels=(0.68, 0.95),
               functionals={"mean": lambda rho: rho @ centers}, rng=1)
print(cov.levels, cov.average().round(2), cov.stderr().ravel().round(2))
```

To see how the intervals behave for one particular truth, simulate from a fixed
population instead of from the prior
([why](../how-it-works/validation.md)).

## Exact posteriors for small problems

With two or three bins, {func}`~binference.validation.exact_posterior` gives the
exact posterior. Compare any engine against it:

```python
bins = binference.Bins([0.0, 1.0, 2.0, 3.0])
mock = binference.simulate.mock_1d(200, bins, density=np.array([0.5, 0.3, 0.2]), noise=(0.3, 0.8), rng=8)
exact = exact_posterior(mock.likelihoods, priors.Dirichlet(1.0))
res = binference.infer(mock.likelihoods, rng=1, n_samples=1000, n_warmup=500)
print(exact.mean.round(4), res.mean.round(4))
print(exact.std.round(4), res.std.round(4))
```

## Mock catalogues

{mod}`binference.simulate` makes catalogues with known truth for every input type, and
`prior_predictive` makes the simulators that SBC and coverage need.
