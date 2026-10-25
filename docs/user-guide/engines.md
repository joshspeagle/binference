# Inference engines

{func}`binference.infer` builds the model and runs an engine in one call. The
default, NUTS, samples the exact posterior and works with every prior. The
other engines trade generality or accuracy for speed, or answer a different
question.

```python
import numpy as np
import binference

mock = binference.simulate.mock_1d(600, binference.Bins.linear(0, 3, 12), rng=5)
lk = mock.likelihoods
```

## Choosing an engine

| `method=` | what you get | exact? | use it when |
|---|---|---|---|
| `"nuts"` (default) | posterior draws | yes | always a good choice; mixes well even with broad likelihoods |
| `"gibbs"` | draws plus joint latent labels | yes | conjugate priors and narrow likelihoods; you want joint labels |
| `"map"` / `"em"` | a point estimate with a certificate | n/a | quick looks, initialisation, very large problems |
| `"laplace"` | importance-corrected Gaussian draws | approximately | very large $N$ with few bins; check the Pareto $\hat k$ |
| `"vb"` | mean-field variational draws | no | speed when error bars do not matter (they are too narrow) |
| `"bootstrap"` | sampling distribution of the MAP | no (frequentist) | comparison with frequentist intervals |
| `"auto"` | NUTS, or Laplace for very large inputs | | scripts that must scale |

```python
res = binference.infer(lk, rng=1, n_samples=500, n_warmup=500)         # NUTS
gib = binference.infer(lk, method="gibbs", rng=1, n_samples=2000, check=False)
est = binference.infer(lk, method="map")
print(res.mean[:4].round(3), gib.mean[:4].round(3), est.rho[:4].round(3))
```

Extra keyword arguments go to the engine, e.g. `n_chains`, `n_warmup`,
`target_accept` or `n_jobs` for NUTS. `n_samples` is the number of draws
**per chain** for NUTS and Gibbs, and the total for the other samplers.
`rng` accepts a seed or a `numpy.random.Generator`, and the same seed gives
the same draws.

## NUTS

The No-U-Turn Sampler (Hoffman & Gelman 2014) runs on unconstrained
coordinates: isometric log-ratios for simplex priors, whitened fields with
log-hyperparameters for smoothness priors, and log-rates for rates. By
default it runs four chains, each with 1000 warm-up and 1000 sampling
iterations, starting near the posterior mode.

- **Warm-up** adapts the step size and the mass matrix. For smoothness priors
  it also re-parametrises the prior and starts a dense mass matrix from the
  local curvature ([theory](../theory.md#smoothness-the-logistic-family)).
- **Diagnostics** are computed automatically: $\hat R$, effective sample
  sizes, divergences, tree depth and E-BFMI. `binference.infer(check=True)`, the
  default, warns about any problem.
- **Parallel chains.** `n_jobs=-1` runs the chains in separate processes.
  They use the same random streams as a serial run, so the draws agree up to
  floating-point rounding. In scripts on macOS and Windows, put the call
  under `if __name__ == "__main__":`.

Use the engine class directly to resume a run, stream draws, or inspect
adaptation:

```python
from binference.samplers import NUTS

model = binference.PopulationModel(lk)
engine = NUTS(model, n_chains=2, n_warmup=400, rng=2)
first = engine.run(200)                    # warm-up + 200 draws per chain
more = engine.run(200)                     # 200 more, no new warm-up: 400 in total
for positions in engine.sample(5):         # a generator over iterations
    pass
print(more.chains.shape, engine.result.chains.shape, more.info["step_size"].round(3))
```

## Gibbs sampling

The data-augmentation Gibbs sampler alternates between drawing every
object's bin given $\rho$ and drawing $\rho$ given the counts, as in
Leistedt et al. (2016). It needs a conjugate prior (the Dirichlet family or
`GammaRates`) and integer weights. Selection is handled exactly, by also
imputing the missed objects.

Gibbs sampling is exact, but its autocorrelation grows with the **fraction
of missing information**: when likelihoods are broader than the bins,
successive draws are strongly correlated. Check the effective sample size,
or ask the model directly:

```python
print("missing information:", round(model.missing_information(est.rho), 2))
print("Gibbs bulk ESS:", gib.diagnostics["ess_bulk"].min().round(0),
      "from", gib.n_draws, "draws")
```

A missing-information fraction near one means Gibbs will mix slowly: its
autocorrelation time is roughly $(1+f)/(1-f)$ iterations. Use NUTS. When a
Gibbs run fails its checks, {meth}`~binference.PopulationResult.check` reports
this number in its hint. `store_labels=True` keeps the latent labels for joint inference on
individual objects, although {meth}`~binference.PopulationResult.sample_assignments`
gives exact joint draws from any engine's output.

## Point estimates

`method="map"` returns a {class}`~binference.PointEstimate`.

- **Dirichlet priors with concentrations of at least one** (including the
  default flat prior, which gives the maximum-likelihood estimate) use
  mix-SQP (Kim et al. 2020), a second-order method for mixture proportions.
  It converges in a handful of iterations even when likelihoods are much
  broader than the bins, where EM needs thousands. With a selection function
  it applies to the flat prior only: it solves for the fractions of the
  *observed* population, $\phi_k\propto S_k\rho_k$, in which the likelihood
  is concave, and maps the mode back. `GammaRates` with shapes of at least
  one use it too, once the total rate is profiled out.
- **Other conjugate priors** (small concentrations, trees, `GammaRates` with
  shapes below one, informative priors with selection) use SQUAREM-accelerated EM on the
  simplex. EM is "iterated stacking", and its fixed point is the
  maximum-likelihood or MAP population. If it stops before converging, the
  warning reports the fraction of missing information, which sets its speed.
- **Priors whose density is unbounded at the edge of the simplex**
  (concentrations below one, Pólya trees) and all non-conjugate priors use
  L-BFGS in unconstrained coordinates, which keeps the mode in the interior.
- **The certificate** is the KKT optimality gap, which should be about zero.
  `MAP(tol=1e-8)` bounds it for EM, bounds the same gap of the problem mix-SQP
  solves (the observed fractions under selection, the composition for rates),
  and bounds the largest gradient over the total weight for L-BFGS. A run that
  can no longer improve its objective in floating point is accepted below
  `1e3 * tol`. `est.log_posterior` is
  always the density of the fractions or rates, `model.log_posterior(est.value)`;
  L-BFGS also stores the density in its own coordinates as
  `est.info["log_density_unconstrained"]`.

```python
print(est, est.kkt_gap < 1e-6, est.converged)
lap = est.laplace(n_samples=2000, rng=0)   # a quick Gaussian approximation around the mode
print(lap.approximate, round(lap.info["k_hat"], 2))
```

## Approximations

- **`"laplace"`** fits a Gaussian at the unconstrained mode and corrects it by
  Pareto-smoothed importance sampling. If the diagnostic $\hat k$ reaches
  $\min(1-1/\log_{10}S,\,0.7)$ for $S$ proposals (0.7 from about 2,000), the
  correction is unreliable and the result is flagged `approximate`. This is
  common with ten or more bins, because nearly empty bins have one-sided
  tails.
- **`"vb"`** (mean-field variational Bayes) is very fast but treats each
  object's bin as independent of $\rho$, so its intervals are too narrow,
  often by a factor of several when likelihoods are broad. It always warns
  and is flagged `approximate`.
- **`"bootstrap"`** resamples objects (`kind="nonparametric"`) or reweights
  them (`kind="bayesian"`) and refits the MAP. Integer weights count
  objects, so a compressed row of weight 5 stands for five objects that are
  resampled one by one. For rate models the number of objects varies between
  replicates (Poisson resampling), so the total rate does too. The result is
  the sampling distribution of an estimator, not a posterior.

## Scikit-learn style

{class}`~binference.PopulationEstimator` wraps the same machinery with the
scikit-learn conventions: `fit`, `predict_proba`, `score`, `sample`,
`get_params` and `set_params`. scikit-learn itself is not required.

```python
est = binference.PopulationEstimator(method="map").fit(lk[:400])
print(round(est.score(lk[400:]), 3), est.predict_proba(lk[400:403]).shape)
```

`score` is the mean log posterior-predictive density per held-out object. Use
it to compare priors or binnings by cross-validation.
