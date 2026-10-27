# Choose an engine

{func}`binference.infer` builds the model and runs an {term}`engine` in one call. The
default, `method="auto"`, picks the engine for you, and most analyses should keep it.
This page says when to override it, and how to pass options to an engine. What each
engine computes, and when it is exact, is in the
[theory](../how-it-works/theory.md#the-engines).

```python
import numpy as np
import binference

mock = binference.simulate.mock_1d(600, binference.Bins.linear(0, 3, 12), rng=5)
lk = mock.likelihoods
```

## The engines

| `method=` | use it when |
|---|---|
| `"auto"` (default) | you have no reason to choose |
| `"nuts"` | always a good choice; it mixes well even with broad likelihoods |
| `"gibbs"` | the prior is conjugate and the likelihoods narrow, or you want joint labels |
| `"map"` / `"em"` | a quick look, a starting point, or a very large problem |
| `"laplace"` | very many objects in few bins; check its Pareto $\hat k$ |
| `"vb"` | speed matters and error bars do not (they are too narrow) |
| `"bootstrap"` | you want frequentist intervals to compare with |

```python
res = binference.infer(lk, method="nuts", rng=1, n_samples=500, n_warmup=500)
gib = binference.infer(lk, method="gibbs", rng=1, n_samples=2000, check=False)
est = binference.infer(lk, method="map")
print(res.mean[:4].round(3), gib.mean[:4].round(3), est.rho[:4].round(3))
```

Extra keyword arguments go to the engine, e.g. `n_chains`, `n_warmup`,
`target_accept` or `n_jobs` for NUTS. For NUTS, Gibbs and `"auto"`, `n_samples` is the
number of draws **per chain**; auto's Laplace draws as many as four chains would. For
the other samplers it is the total. `rng` accepts a seed or a
`numpy.random.Generator`, and the same seed gives the same draws.

## What auto does

{term}`Auto <auto>` runs Gibbs where Gibbs mixes well, then importance-corrected
Laplace where its check passes, and NUTS otherwise. {class}`binference.inference.Auto`
states the exact rules. They come from a benchmark of every engine against long
reference runs ({doc}`theory <../how-it-works/theory>`).

- **What it chose.** The choice and its reason are in `result.info["auto_method"]`
  and `result.info["auto_reason"]`. Where auto computed the MAP to choose, the result
  keeps it as `result.best`.
- **Its thresholds** (`target_ess`, `gibbs_max_missing`, `max_gibbs_draws`,
  `nnz_limit`) go to `infer` like any other option, e.g.
  `binference.infer(lk, target_ess=1000)`.
- **Engine options rule out engines** that do not take them. `target_accept` or
  `n_jobs` leave only NUTS, and `n_warmup` leaves Gibbs or NUTS. Options that no
  candidate takes raise an error.
- **An explicit `n_samples`** fixes the length of a Gibbs run, so check the result;
  `infer` does by default.

{class}`~binference.PopulationEstimator` uses `method="auto"` too.

## When to override auto

Pass `method=` to choose yourself:

- **You need a quick look, or the problem is huge.** Use `method="map"`: a point
  estimate in seconds. `method="laplace"` adds approximate error bars.
- **Auto keeps Laplace on a very large input and flags it approximate.** If you can
  afford it, run `method="nuts"` on the full data or on a subset to check it.
- **NUTS reports divergences.** Raise `target_accept`, or change the prior
  ({doc}`troubleshooting`).
- **You want joint labels for every object.** Use `method="gibbs"` with
  `store_labels=True`, or `result.sample_assignments` from any engine.
- **You want frequentist intervals.** Use `method="bootstrap"`; its draws are not a
  posterior.

## NUTS

{term}`NUTS` {cite:p}`hoffman2014` runs four chains by default, each with 1000
{term}`warm-up` and 1000 sampling iterations. `binference.infer` checks its
diagnostics and warns about any problem ({doc}`checking`).

- **Parallel chains.** `n_jobs=-1` runs the chains in separate processes. They use
  the same random streams as a serial run, so the draws agree up to floating-point
  rounding. In scripts on macOS and Windows, put the call under
  `if __name__ == "__main__":`.
- **Divergences.** Raise `target_accept`, for example to 0.95.

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

## Gibbs

{term}`Gibbs` needs a {term}`conjugate prior` (the Dirichlet family or `GammaRates`) and
integer weights. It mixes slowly when the {term}`missing information` is near one, which
happens when likelihoods are broader than the bins. Check the effective sample size, or
ask the model:

```python
print("missing information:", round(model.missing_information(est.rho), 2))
print("Gibbs bulk ESS:", gib.diagnostics["ess_bulk"].min().round(0),
      "from", gib.n_draws, "draws")
```

If the missing information is near one, use NUTS. When a Gibbs run fails its checks,
{meth}`~binference.PopulationResult.check` reports this number in its hint.
`store_labels=True` keeps every object's bin in every draw.
{meth}`~binference.PopulationResult.sample_assignments` gives such joint draws from any
engine's output.

## Point estimates

`method="map"` returns a {class}`~binference.PointEstimate`. Three solvers share the
work: {term}`mix-SQP`, EM and {term}`L-BFGS`. {class}`~binference.samplers.MAP` gives the
routing, and the [theory](../how-it-works/theory.md#point-estimates-em-is-iterated-stacking)
explains it.

- **The choice** is recorded in `est.info["algorithm"]` and `est.info["space"]`.
- **The certificate** is the {term}`KKT gap`, `est.kkt_gap`, which should be about
  zero. `MAP(tol=...)` bounds it ({doc}`defaults <../reference/defaults>`). An L-BFGS
  estimate has `est.kkt_gap` NaN and keeps its largest absolute gradient in
  `est.info["grad_norm"]`.
- **`est.log_posterior`** is always the density of the fractions or rates,
  `model.log_posterior(est.value)`. L-BFGS also stores the density in its own
  coordinates as `est.info["log_density_unconstrained"]`.

```python
print(est, est.kkt_gap < 1e-6, est.converged)
lap = est.laplace(n_samples=2000, rng=0)   # a quick Gaussian approximation around the mode
print(lap.approximate, round(lap.info["k_hat"], 2))
```

## Approximations

- **`"laplace"`** is {term}`approximate` when its {term}`Pareto k-hat` check fails, which
  is common with ten or more bins. Auto keeps it only when the check passes.
- **`"vb"`** always warns, and its intervals are too narrow, often several times so
  when likelihoods are broad.
- **`"bootstrap"`** resamples objects (`scheme="nonparametric"`) or reweights them
  (`scheme="bayesian"`), and refits the MAP. A compressed row of weight 5 is resampled
  as five objects. Its draws describe an estimator, not a posterior.

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
