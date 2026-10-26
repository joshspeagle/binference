# Inference engines

{func}`binference.infer` builds the model and runs an engine in one call. The
default, `method="auto"`, picks the engine from rules measured on a benchmark
matrix (below). NUTS samples the exact posterior and works with every prior;
the other engines trade generality or accuracy for speed, or answer a
different question.

```python
import numpy as np
import binference

mock = binference.simulate.mock_1d(600, binference.Bins.linear(0, 3, 12), rng=5)
lk = mock.likelihoods
```

## Choosing an engine

| `method=` | what you get | exact? | use it when |
|---|---|---|---|
| `"auto"` (default) | Gibbs, Laplace or NUTS | yes, or flagged | you have no reason to choose (below) |
| `"nuts"` | posterior draws | yes | always a good choice; mixes well even with broad likelihoods |
| `"gibbs"` | draws plus joint latent labels | yes | conjugate priors and narrow likelihoods; you want joint labels |
| `"map"` / `"em"` | a point estimate with a certificate (`"em"`: by EM itself rather than mix-SQP) | n/a | quick looks, initialisation, very large problems |
| `"laplace"` | importance-corrected Gaussian draws | approximately | very large $N$ with few bins; check the Pareto $\hat k$ |
| `"vb"` | mean-field variational draws | no | speed when error bars do not matter (they are too narrow) |
| `"bootstrap"` | sampling distribution of the MAP | no (frequentist) | comparison with frequentist intervals |

```python
res = binference.infer(lk, method="nuts", rng=1, n_samples=500, n_warmup=500)
gib = binference.infer(lk, method="gibbs", rng=1, n_samples=2000, check=False)
est = binference.infer(lk, method="map")
print(res.mean[:4].round(3), gib.mean[:4].round(3), est.rho[:4].round(3))
```

Extra keyword arguments go to the engine, e.g. `n_chains`, `n_warmup`,
`target_accept` or `n_jobs` for NUTS. `n_samples` is the number of draws
**per chain** for NUTS and Gibbs, and the total for the other samplers.
`rng` accepts a seed or a `numpy.random.Generator`, and the same seed gives
the same draws.

## How `method="auto"` chooses

`method="auto"` takes the first of these that applies, and records its choice
and the reason in `result.info["auto_method"]` and `result.info["auto_reason"]`:

1. **Gibbs** for a selection or response inferred jointly with a conjugate
   prior (the only exact engine for these with a conjugate prior).
2. **Gibbs** for a conjugate prior (the Dirichlet family or `GammaRates`) with
   integer weights when the fraction of missing information $f$ at the MAP is
   at most `gibbs_max_missing` (0.98). It runs enough draws per chain for a
   smallest bulk ESS of about `target_ess` (400), from the autocorrelation time
   $(1+f)/(1-f)$, and at least 1000.
3. **Laplace**, importance-corrected, kept when its Pareto $\hat k$ check
   passes and its importance ESS is at least `target_ess`.
4. **NUTS** otherwise. Above `nnz_limit` ($2\times10^7$) stored likelihood
   entries it keeps the Laplace draws instead, flagged approximate.

The three thresholds are options of the engine,
{class}`binference.inference.Auto`, and go to `infer` like any other, e.g.
`binference.infer(lk, target_ess=1000)`. Engine options rule out the engines
that do not take them: `target_accept` or `n_jobs` leave only NUTS, `n_warmup`
leaves Gibbs or NUTS, and options that no candidate takes raise an error. Pass
`method=` to choose yourself. {class}`~binference.PopulationEstimator` uses
`method="auto"` too.

The rules come from `benchmarks/matrix.py`: 18 mock catalogues, each varying
one factor around a baseline (1-D, 20 bins, 1000 objects, Gaussian errors one
bin wide): error widths of 0.3 and 3 bins, 100 and 10,000 objects, 10 and 50
bins, sparse and spiky populations, selection on the truth and on the
measurement, posteriors with a correct and a wrong interim prior, 20 and 200
samples per object, an 8×8 grid, classifier outputs, and a photo-z-like set
(30 bins, 5000 objects, aliased and outlying likelihoods). Each was fitted with
the priors that apply (flat and $\alpha=0.1$ Dirichlet, random walk, GP, Pólya
tree), 43 combinations in all, and each engine at its defaults was compared with
a reference NUTS run of 20,000 draws. Measured on one thread per fit, four fits
at a time (results in `benchmarks/results/`):

| engine | largest mean error (reference s.d.) | s.d. ratio | smallest bulk ESS per second |
|---|---|---|---|
| NUTS | 0.08 | 0.69–1.19 | 0.1–339, median 80 |
| Gibbs | 0.94 | 0.24–1.75 | 0.6–791, median 14 |
| Laplace, check passed (4 of 43) | 0.05 | 0.96–1.04 | 1751–6607 |
| Laplace, check failed (39 of 43) | 8.1 | 0.05–7.3 | |
| VB | 2.6 | 0.00–1.05 | |

- **NUTS** was within 0.08 s.d. of the reference in every mean. Its s.d. ratios
  outside 0.9–1.1 are near-empty bins under the random-walk prior, whose
  posteriors are heavy-tailed. It is slow with Dirichlet concentrations below
  one (0.1: 2–5 ESS per second, 859 s on the photo-z set).
- **Gibbs** beat NUTS in all 5 combinations where the missing information $f$
  was at most 0.976 (103–791 ESS per second against NUTS's 2–336) and in 3 of
  the 21 where it was 0.994 or more (0.6–67 against 0.1–339). The threshold 0.98 lies in that gap; the
  run length then makes up for the autocorrelation.
- **Laplace** passed its check only with the Pólya tree, and there agreed with
  the reference to 0.05 s.d. in 0.6–1.1 s. Where the check failed its errors
  reached 8 s.d.; the importance ESS requirement also rejects the two fits whose
  $\hat k$ passed with fewer than 400 effective draws (one had quantiles 0.9
  s.d. off).

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
- **`"bootstrap"`** resamples objects (`scheme="nonparametric"`) or reweights
  them (`scheme="bayesian"`) and refits the MAP. Integer weights count
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
