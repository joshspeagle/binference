# Diagnostics and validation

Convergence diagnostics tell you whether a sampler explored the posterior.
They cannot tell you whether the posterior is *right*: whether your
likelihoods, prior and engine together give calibrated answers. binference ships
the tools to check that, the same ones its own test suite uses.

```python
import numpy as np
import binference
from binference import priors
from binference.validation import coverage, exact_posterior, sbc
```

## Exact posteriors for small problems

With two or three bins, {func}`~binference.validation.exact_posterior` integrates
the posterior on a fine grid over the simplex. Any engine can then be compared
against it:

```python
bins = binference.Bins([0.0, 1.0, 2.0, 3.0])
mock = binference.simulate.mock_1d(200, bins, density=np.array([0.5, 0.3, 0.2]), noise=(0.3, 0.8), rng=8)
exact = exact_posterior(mock.likelihoods, priors.Dirichlet(1.0))
res = binference.infer(mock.likelihoods, rng=1, n_samples=1000, n_warmup=500)
print(exact.mean.round(4), res.mean.round(4))
print(exact.std.round(4), res.std.round(4))
```

## Simulation-based calibration

Simulation-based calibration (SBC; Talts et al. 2018) repeats three steps
many times:
1. draw a population from the prior;
2. simulate a catalogue from that population;
3. fit it, and record the **rank** of the true value among the posterior draws.

If the whole pipeline is calibrated, the ranks are uniform. Deviations have
recognisable shapes: a U-shape means intervals are too narrow, a hump means
they are too wide, and a slope means bias.

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
- **`loglike`.** Besides each bin, SBC ranks the joint log-likelihood of the
  true population. This is sensitive to errors in the correlations between
  bins.
- **Tests.** `result.pvalues()` and `result.passed()` test each quantity's
  ranks with a chi-square on their histogram; `method="ecdf"` uses the ECDF
  test with simultaneous bands of Säilynoja, Bürkner & Vehtari (2022)
  instead, which is more sensitive to bias. Neither test dominates the other:
  to require both, call `result.passed(method="chisquare")` and
  `result.passed(method="ecdf")`.
- **Size.** A real check needs a few hundred simulations and roughly
  independent draws: keep `n_draws` below the effective sample size.

binference's slow test suite runs SBC for NUTS and Gibbs with the Dirichlet,
generalised-Dirichlet and random-walk priors (the second-order walk with its
scale inferred), for Gibbs with a known selection in both modes, with Gamma
rates and an exposure, and with the Pólya tree, and for the Laplace
approximation. It also shows that mean-field
VB *fails* SBC, as expected, and that SBC rejects a Gibbs pipeline whose
posteriors are 18% too narrow (half the objects counted twice).

## Coverage

{func}`~binference.validation.coverage` measures how often central credible
intervals contain the truth, for each bin and for any functional. It answers
the frequentist question most readers care about: do my 68% error bars
contain the truth 68% of the time?

```python
centers = binference.Bins.linear(0, 1, 4).centers
cov = coverage(simulate, fit, n_simulations=40, levels=(0.68, 0.95),
               functionals={"mean": lambda rho: rho @ centers}, rng=1)
print(cov.levels, cov.average().round(2), cov.stderr().ravel().round(2))
```

With data simulated from the prior, Bayesian intervals have exactly their
nominal coverage *on average over the prior*. Simulate from a fixed
population instead to see how they behave for a particular truth.

## Mock catalogues

{mod}`binference.simulate` generates catalogues with known truth. Iterating over
a mock gives `(likelihoods, truth)`.

| simulator | what it makes |
|---|---|
| `mock_1d` | 1-D populations: named shapes, callables or bin fractions; Gaussian errors; interim-prior posteriors (`mock.posteriors`); posterior samples; selection on the truth (`selection=`), on the measurement (`measured_range=`) or both; outliers |
| `mock_2d` | 2-D grids, e.g. a colour–magnitude-like ridge |
| `mock_categorical` | classifier outputs with a training prior and a labelled validation set, for quantification |
| `mock_unfolding` | counts in reconstructed bins with a `Response` and efficiency |
| `prior_predictive` | a simulator drawing populations from a prior, for SBC and coverage |

```python
mock = binference.simulate.mock_1d(500, interim="gaussian", n_samples=100, rng=2)
truth = mock.truth
print(truth.rho.shape, truth.counts.sum(), mock.posteriors.shape, mock.samples.shape)
```

```{note}
`mock_1d` draws true values from a continuous density by default. The
histogram model assumes the population is flat within each bin, so with
coarse bins and many objects the fit shows a small *binning bias*
({doc}`pitfalls`). Pass bin fractions as `density` (e.g.
`density=binference.simulate.population(bins, "bimodal")`) to draw values uniformly
within bins, for which the model is exactly right.
```
