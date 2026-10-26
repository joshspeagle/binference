---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Validating an analysis

Convergence diagnostics say whether a sampler explored the posterior, not
whether the posterior is calibrated. This tutorial uses binference's validation
tools to check a whole pipeline: an exact reference for a small problem,
simulation-based calibration (SBC), and coverage. It also shows what a
failure looks like.

```{code-cell} python
import warnings

import matplotlib.pyplot as plt
import numpy as np

import binference
from binference import priors
from binference.validation import coverage, exact_posterior, sbc

plt.rcParams["figure.dpi"] = 110
```

## Exact posteriors

With three bins the posterior can be integrated on a grid. Here it is
compared with NUTS and Gibbs on data with selection effects:

```{code-cell} python
bins = binference.Bins([0.0, 1.0, 2.0, 3.0])
S = np.array([0.9, 0.5, 0.25])
mock = binference.simulate.mock_1d(250, bins, density=np.array([0.4, 0.35, 0.25]),
                               noise=(0.3, 0.8), selection=S, rng=777)
exact = exact_posterior(mock.likelihoods, priors.Dirichlet(1.0))
nuts = binference.infer(mock.likelihoods, method="nuts", rng=1)
gibbs = binference.infer(mock.likelihoods, method="gibbs", rng=1, n_samples=4000)
print("exact mean", exact.mean.round(4), " sd", exact.std.round(4))
print("NUTS  mean", nuts.mean.round(4), " sd", nuts.std.round(4))
print("Gibbs mean", gibbs.mean.round(4), " sd", gibbs.std.round(4))
```

binference's test suite makes these comparisons for every engine, prior family
and selection mode.

## Simulation-based calibration

SBC (Talts et al. 2018) checks the whole loop: prior, simulator and
inference. Draw a population from the prior, simulate a catalogue, fit it,
and record where the true value falls among the posterior draws. Over many
repetitions these ranks are uniform if, and only if, the pipeline is
calibrated.

```{code-cell} python
prior = priors.Dirichlet(1.0)
grid = binference.Bins.linear(0.0, 1.0, 4)
simulate = binference.simulate.prior_predictive(prior, grid, n_objects=50, noise=0.25)

def fit_gibbs(lk, rng):
    return binference.infer(lk, prior, method="gibbs", rng=rng, n_chains=1,
                        n_warmup=200, n_samples=1000, check=False)

result = sbc(simulate, fit_gibbs, n_simulations=150, n_draws=49, rng=0)
print("uniformity p-values:", {n: round(float(v), 3) for n, v in zip(result.names, result.pvalues(10))})
fig = result.plot()
```

The rank histograms sit inside the band expected for uniform ranks. The last
panel ranks the joint log-likelihood of the true population, which is
sensitive to errors in the correlations between bins that per-bin ranks can
miss.

### What a failure looks like

Mean-field variational Bayes is over-confident, so the true value too often
falls outside the bulk of its draws. The rank histograms become U-shaped:

```{code-cell} python
def fit_vb(lk, rng):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", binference.ApproximationWarning)
        return binference.infer(lk, prior, method="vb", rng=rng, n_samples=1000)

bad = sbc(simulate, fit_vb, n_simulations=150, n_draws=49, rng=0)
print("uniformity p-values:", {n: round(float(v), 4) for n, v in zip(bad.names, bad.pvalues(10))})
fig = bad.plot()
```

## Coverage

Coverage asks the frequentist question directly: how often do central
credible intervals contain the truth? With data simulated from the prior,
Bayesian intervals have nominal coverage on average. Coverage can be checked
for any functional, here the population mean:

```{code-cell} python
centers = grid.centers
cov = coverage(simulate, fit_gibbs, n_simulations=150, levels=(0.5, 0.9),
               functionals={"mean": lambda rho: rho @ centers}, rng=1)
print("levels:", cov.levels)
print("coverage per quantity (rows = levels):")
print(np.round(cov.rates(), 2))
print("standard error:", cov.stderr().ravel().round(3))
```

## Validating your own pipeline

Write a simulator that reproduces your real catalogue: the same number of
objects, measurement errors and selection function. Then call `sbc` or
`coverage` with the exact settings you use in production, as in
`prior_predictive`. A few hundred simulations of a scaled-down version of the
problem usually expose any problem with the likelihoods, the selection model
or the sampler settings.
