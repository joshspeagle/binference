# Choose an engine

{func}`binference.infer` builds the model and runs an {term}`engine` in one call. The
default, `method="auto"`, picks the engine for you, and most analyses should keep it.
This page says when to override it, and how to pass options to an engine. What each
engine computes, and when it is exact, is on the
{doc}`engines concept page <../concepts/engines>`.

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
| `"laplace"` | very many objects in few bins; check its {term}`Pareto k-hat` |
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
number of draws **per chain**. If auto chooses Laplace, it draws four times
`n_samples`. For the other samplers it is the total. `rng` accepts a seed or a
`numpy.random.Generator`, and the same seed gives the same draws.

## What auto does

{term}`Auto <auto>` runs Gibbs where Gibbs mixes well, then importance-corrected
Laplace where its check passes, and NUTS otherwise. The rules, and the benchmark of
every engine against long reference runs that set them, are in the
{doc}`how auto chooses <../how-it-works/auto>`. {class}`binference.inference.Auto` lists its
options.

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
- **Auto keeps Laplace and flags it approximate.** It does so for a very large input,
  or when no other engine takes the options given. If you can
  afford it, run `method="nuts"` on the full data or on a subset to check it.
- **NUTS reports divergences.** Raise `target_accept`, or change the prior
  ([a smoothness scale near zero](troubleshooting.md#a-smoothness-scale-near-zero)).
- **You want joint labels for every object.** Use `method="gibbs"` with
  `store_labels=True`, or `result.sample_assignments` from any engine.
- **You want frequentist intervals.** Use `method="bootstrap"`; its draws are not a
  posterior.

## NUTS

{term}`NUTS` {cite:p}`hoffman2014` takes the numbers of chains, {term}`warm-up` and
draws in its signature ({class}`~binference.samplers.NUTS`). `binference.infer` checks
its diagnostics and warns about any problem ({doc}`checking`). `n_jobs=-1` runs the
chains in parallel ({doc}`performance`). If NUTS reports divergences, see
[diagnostics](checking.md#diagnostics).

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
routing, and [point estimates](../how-it-works/point-estimates.md)
explains it.

- **The choice** is recorded in `est.info["algorithm"]` and `est.info["space"]`.
- **Which mode** depends on that space. On the simplex (EM, mix-SQP) the mode of a
  $\mathrm{Dir}(\alpha)$ prior with counts $n_k$ is proportional to $n_k+\alpha_k-1$; in the
  unconstrained coordinates (L-BFGS) it is proportional to $n_k+\alpha_k$, because the
  density of those coordinates includes their Jacobian
  ([two modes](../how-it-works/point-estimates.md#two-coordinate-systems-two-modes)).
- **The convergence check** is the {term}`KKT gap`, `est.kkt_gap`, which should be
  about zero. `MAP(tol=...)` bounds it. For L-BFGS, which reports a gradient instead,
  see {class}`~binference.samplers.MAP`.
- **`est.log_posterior`** is always the density of the fractions or rates,
  `model.log_posterior(est.value)`. L-BFGS also stores the density in its own
  coordinates as `est.info["log_density_unconstrained"]`.

```python
print(est, est.kkt_gap < 1e-6, est.converged)
lap = est.laplace(n_samples=2000, rng=0)   # a quick Gaussian approximation around the mode
print(lap.approximate, round(lap.info["k_hat"], 2))
```

## Approximations

- **`"laplace"`** is {term}`approximate` when its Pareto k-hat check fails. Auto
  keeps it when the check passes. It also keeps it, flagged approximate, for a very
  large input or when no other engine takes the options given.
- **`"vb"`** always warns, and its intervals are too narrow, most of all when
  likelihoods are broad.
- **`"bootstrap"`** resamples objects (`scheme="nonparametric"`) or reweights them
  (`scheme="bayesian"`), and refits the MAP. A compressed row of weight 5 is resampled
  as five objects. Its draws describe an estimator, not a posterior.

## Use the scikit-learn interface

{class}`~binference.PopulationEstimator` wraps the same machinery with the
scikit-learn conventions: `fit`, `predict_proba`, `score`, `sample`,
`get_params` and `set_params`. scikit-learn itself is not required.

```python
est = binference.PopulationEstimator(method="map").fit(lk[:400])
print(round(est.score(lk[400:]), 3), est.predict_proba(lk[400:403]).shape)
```

`score` is the mean {term}`log predictive density` per held-out object. Use it to
compare priors or binnings by cross-validation.
