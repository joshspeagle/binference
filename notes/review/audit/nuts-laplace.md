# Audit: NUTS core, NUTS engine, Laplace

Slice: `src/pastry/samplers/_nuts_core.py`, `src/pastry/samplers/nuts.py`, `src/pastry/samplers/laplace.py` (branch `claude/pastry-v0.1`, HEAD 5c56942).
Scripts and outputs: `/tmp/claude-0/-home-user-pastry/63ffb1b4-5826-5f0b-ade5-c8209761737e/scratchpad/audit/nuts-laplace/a01..a12_*.py`.
The full fast test suite passes (268 passed, 15 deselected).

## Summary

- The NUTS transition matches Stan's `base_nuts` line by line. I fetched the current `develop` source to check this. It covers multinomial sampling, biased progressive sampling at the top level, all three U-turn checks in both directions, the divergence rule, acceptance statistics, dual averaging and the regularised metric. The moments are calibrated: across 60 dimension×seed combinations the z-scores of the mean have variance 0.99 (dense) and 1.03 (diagonal). NUTS also agrees with Gibbs on hard Dirichlet(0.5) posteriors.
- Draws are identical with and without reparametrisation, within Monte Carlo error. Parallel runs are bit-identical to serial runs, including after resuming. Review finding 7 (divergences in weak-data fits) no longer reproduces under the new gate: 0 divergences in 9 small cases.
- **Main issue (N-1, N-2):** NUTS returns `result.x`, `result.log_prob`, `result.model` and `info["inv_metric"]` in its *internal* partly centred coordinates, not in the user model's coordinates. The geometry step also silently assumes the input prior is non-centred. If you feed `result.model` back in, the chains jump to a different point at iteration 25 (Δlog ρ up to 5.7).
- **Minor bugs:**
  - the worker thread limit *raises* BLAS threads above a limit the user set (N-3);
  - Laplace returns fewer draws than asked for (N-4);
  - `max_depth=0` is accepted and freezes the chain (N-5);
  - genuine `TypeError`/`AttributeError`s from the model are masked as "cannot run in parallel" (N-7).
- **Quality:**
  - each dense momentum draw does a full LU solve, about 12× slower than a triangular solve at d = 200 (N-9);
  - the "diagonal-only" Hessian branch saves nothing (N-10);
  - Laplace duplicates the finite-difference Hessian and eigen-clipping code (N-11);
  - there is some dead or duplicated code in the tree builder (N-12).
- Counts: bug-minor 5 (N-2, N-3, N-4, N-5, N-7), consistency 2 (N-1, N-14), statistical 1, numerical 1, efficiency 3, simplify 2, docs 1; plus 1 out-of-slice note.

---

### N-1. NUTS results carry x, log_prob, model and metric in the internal reparametrised coordinates
**Category:** consistency (bug-minor)
**Location:** `nuts.py:384` (`self.model = model`), `nuts.py:526-554` (`result`)
**Claim:**
- For logistic priors, `_adapt_geometry` replaces `self.model` with `base._with_bound_prior(prior0.with_centering(lam))`, and `result` hands this model to `PopulationResult` along with the raw chain positions.
- The fractions are correct: they are transformed with the centred model. However, `result.x` and `result.log_prob` are in the partly centred coordinates. `log_prob` includes the Jacobian of that parametrisation.
- `result.model` is not the user's model. `result.model.prior.lam ≠ 1`.
- `info["inv_metric"]` is also in those coordinates.
- Every other engine (MAP, Laplace) reports `x` in the model's unconstrained coordinates. Spec §8 defines `log_prob(x)` on those coordinates for interoperability.
**Evidence:** `a04_reparam_equiv.py`:
```
result.model is user model: False
user model log_prob(result.x) = -2998.88   result.log_prob = -487.70   result.model.log_prob(x) = -487.70
m.transform(result.x) == result.samples[0]? False
after recentre to base: m.transform ok? True
```
**Suggested fix:**
- In `result`, map positions back with `self.model.prior.recentre(x, ones)`. This is cheap and vectorisable per draw, because `recentre` depends only on the hyperparameter coordinates.
- Pass `self._base_model` to `PopulationResult`.
- Recompute or adjust `log_prob` to base coordinates: add the log-Jacobian Σ_j (λ_j − 1)·log s_j(θ), or simply call `base.log_prob` on the mapped x.
- Store `lam` in `info`, and either document `inv_metric` as being in the internal coordinates or drop it.
**Confidence:** confirmed by running.

### N-2. `_adapt_geometry` assumes the user's prior is fully non-centred (λ = 1)
**Category:** bug-minor
**Location:** `nuts.py:52-54`, `nuts.py:361-383`
**Claim:**
- `pts = [cur.recentre(ch.x, ones)]` maps the chains to λ = 1 coordinates. The code then evaluates `fn0 = base.log_prob_and_grad` at those points.
- It also calls `prior0.recentre(x, lam)` and `_conditional_log_scale_sd(base, prior0, pts)`. Each of these treats `x` as if it were in `prior0`'s own coordinates.
- That holds only when `prior0.lam == 1`. A model built from `with_centering`, in particular the `result.model` that N-1 returns, breaks it. The Hessian, λ and the gate are then computed at the wrong points, and the chains are restarted at a different state.
- There is no bias: warm-up continues afterwards. But the geometry decisions are garbage.
**Evidence:** `a05_centred_base.py`. These are the fractions before and after `_adapt_geometry`, which should be the same point in two parametrisations:
```
base (lam=1)   same fractions after reset: [True, True]   max |dlog rho|: 0.0
centred input  same fractions after reset: [False, False] max |dlog rho|: 5.694
```
**Suggested fix:** At the top of `_adapt_geometry`, normalise the base model: `prior0 = base.prior.with_centering(ones)` and `base = base._with_bound_prior(prior0)`. Then use `prior0` and `base` throughout. Alternatively, do this in `__init__` and keep `self._base_model` non-centred. Fixing N-1 removes the most likely route into this, but the assumption should still be enforced.
**Confidence:** confirmed by running.

### N-3. The worker thread limit overrides a stricter limit set by the user; the core count ignores CPU affinity
**Category:** bug-minor
**Location:** `nuts.py:170-178`, `nuts.py:414`, `nuts.py:275-276`
**Claim:**
- `_limit_threads(n)` calls `threadpool_limits(limits=n)` with `n = cpu_count // n_workers`. This *raises* the BLAS pool of each forked worker to `n` threads even when the user deliberately set `OMP_NUM_THREADS=1` or used `threadpool_limits(1)`. This is common on clusters and in shared jobs.
- `os.cpu_count()` returns all cores of the node, not the cores the process may use (`os.sched_getaffinity` / `os.process_cpu_count()` in 3.13). Under SLURM or cgroups this oversubscribes by the node/allocation ratio. `n_jobs=-1` has the same problem.
- D14's rationale ("environment variables (ignored by forked workers)") is not accurate. The forked worker does inherit the parent's limit; it is pastry that overrides it.
**Evidence:** `a06_thread_limits.py`, run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`:
```
parent: [('openblas', 1), ('openblas', 1)]
worker (n_threads=2): [('openblas', 2), ('openblas', 2)]
```
**Suggested fix:**
- In the parent, compute `cur = min(lib["num_threads"] for lib in threadpool_info())`, with a default if empty.
- Pass `max(1, min(cur, n_cpus // n_workers))`, where `n_cpus = len(os.sched_getaffinity(0))` if available, else `os.cpu_count()`.
- Use the same `n_cpus` for `n_jobs=-1`.
**Confidence:** confirmed by running (the thread override); the affinity part is from reasoning.

### N-4. Laplace silently returns fewer draws than requested
**Category:** bug-minor
**Location:** `laplace.py:128`, `laplace.py:157`
**Claim:** With `importance=False` and `n_proposals < n_samples`, `X = X[:n_samples]` returns only `n_proposals` draws, without a warning.
**Evidence:** `a08_laplace.py`: `Laplace(m, importance=False, n_proposals=100).run(4000)` gives `n_draws = 100`.
**Suggested fix:** Without importance sampling, ignore `n_proposals` and set `M = n_samples`. Alternatively, validate `n_proposals >= n_samples` there and require `n_proposals >= 5` for PSIS.
**Confidence:** confirmed by running.

### N-5. `max_depth` is not validated; `max_depth=0` freezes the chain
**Category:** bug-minor
**Location:** `nuts.py:290`, `_nuts_core.py:188-215`
**Claim:**
- With `max_depth=0`, the tree loop never runs, so every transition returns the initial point with acceptance 0/1 = 0.
- Dual averaging then drives ε towards 0.
- Negative values behave the same way.
**Evidence:** `a12_edges.py`: `max_depth=0: unique draws 1, step size [1.47e-05]`.
**Suggested fix:** Raise `ValueError` unless `max_depth >= 1`. Also validate `n_warmup >= 0` and `jitter >= 0`.
**Confidence:** confirmed by running.

### N-6. The Laplace k̂ threshold ignores the number of proposals
**Category:** statistical
**Location:** `laplace.py:148`
**Claim:**
- Vehtari et al. (2024, "Pareto smoothed importance sampling", §3) give the reliability threshold as k̂ < min(1 − 1/log10 S, 0.7). This rule is used by `loo`/`posterior` and by recent ArviZ.
- For S < about 2150 proposals the threshold is below 0.7. Examples:
  - `run(1000)` (S = 2000): 0.697;
  - `run(200)` (S = 400): 0.62;
  - `run(100)` (S = 200): 0.57.
- Small runs can therefore be reported `approximate=False` when the correction is unreliable.
**Evidence:** reasoning from the published threshold. At the default `run(4000)`, S = 8000, so the threshold is 0.7 and the default is unaffected.
**Suggested fix:**
- Use `thr = min(1 - 1/np.log10(M), 0.7)`.
- Report `thr` in `info` and in the warning text.
- Note it in D10.
**Confidence:** plausible (statistical convention, not a defect at the defaults).

### N-7. Parallel fallback catches `TypeError`/`AttributeError` raised *inside* the workers
**Category:** bug-minor
**Location:** `nuts.py:411-434`
**Claim:**
- The `try` block covers both the trial `pickle.dumps` and `list(ex.map(...))`. A genuine `TypeError` or `AttributeError` raised by the model or user code during sampling in a worker is re-raised by `ex.map`.
- The engine then reports it as "Could not run chains in parallel", sets `n_jobs = 1` for good and reruns everything serially. Only then does the real error surface, after a full extra run.
- `OSError` from user code is treated the same way.
**Evidence:** reasoning from the code. `concurrent.futures` re-raises worker exceptions with their original type.
**Suggested fix:**
- Restrict the pickling-related catch (`PicklingError`, `AttributeError`, `TypeError`) to the `pickle.dumps` probe.
- Around the pool, catch only `BrokenProcessPool`, `OSError` and `NotImplementedError` raised while creating the pool. Better still, wrap the worker body and tag its exceptions.
**Confidence:** plausible (not run).

### N-8. `math.log(rng.random())` can raise
**Category:** numerical
**Location:** `_nuts_core.py:156`, `_nuts_core.py:201`
**Claim:** `Generator.random()` returns values in [0, 1) and can return exactly 0.0, with probability 2⁻⁵³ per call. `math.log(0.0)` raises `ValueError` instead of returning −inf. The risk is negligible in practice but it is a latent crash.
**Suggested fix:** Compare on the probability scale. For example, `rng.random() < math.exp(min(0.0, t2.log_w - log_w))`, as Stan does, which also skips the exponential when the difference is positive. Or use `math.log1p(-rng.random())`.
**Confidence:** confirmed by reasoning (the probability is too small to reproduce).

### N-9. Dense momentum draw uses a general LU solve every transition
**Category:** efficiency
**Location:** `_nuts_core.py:44-49`
**Claim:** `np.linalg.solve(self.chol.T, z)` refactorises a triangular matrix with LU on every draw. At d = 200, the size where "auto" still picks dense, this costs about as much as one gradient evaluation per transition.
**Evidence:** `a10_hotpath.py`, single-threaded:
```
d=200: draw (np.linalg.solve) 595 us, solve_triangular 49 us, mv 9 us
LogisticGP K=40 N=300 dim=42: log_prob_and_grad 311 us; K=190 N=2000 dim=50: 770 us
```
**Suggested fix:** In `Metric.__init__`, precompute `self._chol_inv_T = scipy.linalg.solve_triangular(chol, I, lower=True).T`, which is upper-triangular. The draw then becomes `self._chol_inv_T @ z`, a single matrix-vector product. Alternatively, use `solve_triangular(chol, z, trans="T", lower=True)`.

A smaller item: each leapfrog does two metric products, `mv(p_half)` in `leapfrog` and `mv(p1)` in `_leaf`. Keeping `inv_m @ g` alongside the velocity would make it one, but this is negligible next to the gradient.
**Confidence:** confirmed by running (timings).

### N-10. The "diagonal only" curvature branch for d > 500 costs and allocates as much as the full Hessian
**Category:** efficiency / simplify
**Location:** `nuts.py:368-374`, `_nuts_core.py:265-281`
**Claim:**
- `fd_hessian(fn0, x, cols=range(n_act))` builds a `(d, n_act)` matrix. The hyperparameters are 1–3 coordinates, so `n_act ≈ d`. Only its diagonal is used.
- So the d > 500 branch does the same O(d) gradient pairs and O(d²) memory as the d ≤ 500 branch. About 200 MB at d = 5000.
- The only difference between the branches is averaging and symmetrisation, which do not change the diagonal.
- The d ≤ 500 branch computes the full Hessian although only its first `n_act` diagonal entries are used.
**Suggested fix:**
- Add an `fd_hessian_diag(fn, x, idx)` that keeps only `(g(x+e_j) − g(x−e_j))_j / 2h` and returns a vector.
- Use it in one code path for all d.
- Keep the full Hessian only for the metric (d ≤ 500).
**Confidence:** confirmed by reading (no run needed).

### N-11. Laplace re-implements the finite-difference Hessian and eigen-clipping, and decomposes twice
**Category:** simplify
**Location:** `laplace.py:99-118`, `laplace.py:125-127`
**Claim:**
- The Hessian loop is `_nuts_core.fd_hessian`, line for line, apart from its default step.
- The eigenvalue clipping duplicates `covariance_from_hessian`, with a different floor: 1e-8 here against 1e-6 there.
- `run()` then eigendecomposes `cov` again, although `mode_and_covariance` already has `(lam, U)` of −H. So `sd_mat = U / sqrt(lam)`.
**Suggested fix:**
- Call `fd_hessian(m.log_prob_and_grad, x0, h=self.step)`.
- Share one helper that returns `(cov, U, lam, clipped)`.
- Have `mode_and_covariance` also return, or cache, the factor.
- Move `fd_hessian` and `covariance_from_hessian` to a neutral module (e.g. `_utils` or a `samplers/_curvature.py`), since they are not NUTS-specific.
**Confidence:** confirmed by reading.

### N-12. Small duplications and dead code in the tree builder and engine
**Category:** simplify / efficiency
**Location:** `_nuts_core.py:98-110,155-165,201-209`, `nuts.py:323`
**Claim:**
- `_merge_ok` computes `t1.rho + t2.rho`, and the caller computes it again on the next line (`_nuts_core.py:160`, `:205`). That is one extra vector allocation per merge.
- The merge block (multinomial choice, U-turn check, end update) appears twice, in `_build` and `nuts_step`. The two copies differ only in the biased (top-level) versus unbiased (subtree) acceptance. A helper `_merge(t1, t2, v, rng, biased)` would remove 15 duplicated lines.
- In `nuts.py:323`, `mode.x if mode.x is not None else ...` is dead: `MAP(space="unconstrained")` always sets `x`.
**Suggested fix:**
- Compute `rho = t1.rho + t2.rho` once and pass it to `_merge_ok`.
- Factor out the merge.
- Drop the dead fallback.
**Confidence:** confirmed by reading.

### N-13. Parallel overhead: model pickled n_chains + 1 times per call; pool rebuilt per phase; no progress bar
**Category:** efficiency
**Location:** `nuts.py:396-444`, `nuts.py:446-475`
**Claim:**
- Each `_parallel` call pickles the model once for the probe and once per chain. It also starts a fresh `ProcessPoolExecutor`.
- Logistic priors make at least three calls per `run()`: warm-up before the switch, warm-up after it, and sampling. Each extra `run()` adds one more.
- For large likelihood matrices this dominates short runs. In `a07_parallel.py`, parallel was slower than serial on small problems: 2.4 s against 2.0 s, and 5.0 s against 4.0 s.
- `progress=True` is silently ignored with `n_jobs > 1`.
**Suggested fix:**
- Send the model once per worker, through the pool `initializer` stored in a module global.
- Send only the chain states per call.
- Keep one executor alive for the warm-up phases.
- Document that there is no progress bar in parallel mode.
**Confidence:** confirmed by running (timings); the cause is from reasoning.

### N-14. First adaptation window differs from Stan for 151 ≤ n_warmup ≤ 199
**Category:** consistency
**Location:** `_nuts_core.py:252-261`
**Claim:**
- Stan's `restart()` never extends the first window.
- Pastry applies the "extend if the next window would not fit" rule to the first window as well. For n_warmup = 151, it uses one window `[75,101)`. Stan uses `[75,100)` and then a one-draw window `[100,101)`.
- This is arguably better than Stan, and harmless. All other lengths from 0 to 3000 match a port of Stan's `windowed_adaptation`.
**Evidence:** `a03_windows_vs_stan.py`: 49 of 3001 lengths differ, all of them in 151–199.
**Suggested fix:** None needed. Note the deviation in the docstring ("Stan-style").
**Confidence:** confirmed by running.

### N-15. Docstrings that do not match behaviour
**Category:** docs
**Location:** `nuts.py:227-231`, `nuts.py:243-252`, `nuts.py:129-141`, `_nuts_core.py:218`
**Claim:**
- `init` of shape `(dim,)` also gets `jitter` added, while `(n_chains, dim)` does not. Neither case is checked for a finite log density. None of this is documented.
- The Notes say "after 25 warm-up iterations". The code does it after `min(25, init_buffer)` iterations. It is skipped entirely when `n_warmup < 20`: there is no reparametrisation and no metric adaptation (confirmed in `a12_edges.py`: `n_warmup=15` leaves the model un-reparametrised and the metric the identity).
- "Later adaptation windows refine the metric … shrinking toward that curvature-based start." The shrinkage target is actually the *previous window's* metric (`prev`), so the shrinkage is recursive. At d = 200, the final metric puts weight 0.71 on the last window, 0.14 on the one before, and only 0.07 on the Hessian.
- `find_reasonable_eps` follows Hoffman & Gelman (start at 1.0, threshold 0.5, one momentum draw). It does not follow Stan's `init_stepsize` (continue from the current ε, threshold 0.8, fresh momentum each trial). This is fine, but the module calls the adaptation "Stan-style".
**Suggested fix:** Amend the docstrings. Optionally, warn when the user-supplied `init` has a non-finite log density.
**Confidence:** confirmed by reading and running.

### Out-of-slice note (validation.py): the grid reference is biased for Dirichlet α < 1
- `exact_posterior` uses a midpoint grid. With the integrable singularity ρ^(α−1), this underweights the corners.
- Example: Dirichlet(0.5), K = 3, N = 20. The mean of ρ₃ is:
  - 0.0359 at `n_grid=600`;
  - 0.0337 at `n_grid=2000`;
  - 0.0313 ± 0.0002 from Gibbs (20k×4);
  - 0.0308 ± 0.0004 from NUTS.
- Run naively, NUTS looks wrong at z = −8.8 (`a08_laplace.py`, `a09_alpha_half.py`). It is not: NUTS and Gibbs agree.
- Anyone validating against `exact_posterior` with α < 1 should be warned. Possible fixes are a change of variables, such as a grid in ρ^α, or rejecting α < 1.

---

## Checked and found correct

- **Leaf (Stan comparison):**
  - Divergence when the energy error exceeds 1000, or when H is non-finite (Stan maps NaN to +∞).
  - Log weight H0 − h; acceptance statistic min(1, e^{ΔH}) averaged over all leapfrogs, including rejected subtrees.
  - `n_leapfrog`; tree depth incremented only for valid subtrees.
  - Energy of the selected state; ρ initialised to p0.
- **Sampling within the tree:**
  - Unbiased multinomial selection inside subtrees.
  - Biased progressive selection at the top level, compared against the old log weight.
- **U-turn checks:** all three checks inside `build_tree` and at the top level, in both directions. These are checked against the current Stan source; the top level uses `p_fwd_fwd` and `p_bck_bck` for the old tree's ends, as pastry does. They use velocities M⁻¹p.
- **Dense metric:** momentum p = C^{-T}z with M⁻¹ = CCᵀ.
- **Dual averaging:** t0 = 10, γ = 0.05, κ = 0.75, μ = log(10ε). It restarts after every metric update and after the geometry reset, and the final ε is exp(x̄).
- **Metric estimation:** sample variance/covariance with ddof = 1. In un-informed mode it uses Stan's (n/(n+5))·Σ + 10⁻³·(5/(n+5))·I regularisation, both diagonal and dense.
- **Warm-up bookkeeping:**
  - Window bookkeeping is in absolute iterations and is not disturbed by the reset.
  - The reset happens before the first window opens, so no draws from the old geometry enter the estimate.
- **Moments on known targets** (`a01`, `a02`):
  - anisotropic 20-d Gaussian (sd 0.01–10);
  - correlated 10-d Gaussian with dense and diagonal metrics;
  - 1-d and 100-d standard normals;
  - aggregated z² ≈ 1, with no divergences.
- **Exact K = 3 Dirichlet(1) posterior:** NUTS z within ±2.3 and SD ratios 1.01–1.02 (`a08`). For Dirichlet(0.5), NUTS agrees with Gibbs (`a09`).
- **Reparametrisation:**
  - `recentre` and `transform` are consistent: the fractions are the same point (`a05`, base case).
  - Hyperparameter coordinates u = log θ are untouched by the centring, so `info["hyperparameters"]` is correct.
  - Partial and none: fraction means agree (χ² ≈ 19.6 on 12 correlated components) and the log-scale means agree (z = −1.25) (`a04`).
  - λ = 1/max(−H_jj, 1) in non-centred coordinates equals the Gaussian-optimal σ²/(σ² + s²).
  - The gate: `dlogs[free]` and the hyperparameter Hessian block are indexed consistently. Fixed-scale modes are excluded from the median.
- **Review finding 7:** the weak-data divergences no longer reproduce. There were 0 divergences in 9 small cases (LogisticNormal K = 3, N = 40; RW2 K = 6, N = 30; RW2 K = 10, N = 100), with 0–2 in the remaining runs for both settings (`a11`).
- **Resuming and parallel:** `run()` twice is bit-identical between serial and parallel. Parallel equals serial exactly, warm-up divergences included. The chains' RNG state travels with the pickled chain (`a07`). The fallback on an unpicklable model works (regression test).
- **E-BFMI:** Σ(ΔE)²/Σ(E−Ē)² per chain. It differs from ArviZ's mean/var form only by a factor of n/(n−1).
- **Laplace:**
  - MAP in unconstrained coordinates, so the Jacobian is included.
  - Central-difference Hessian, symmetrised.
  - Multivariate-t proposal Z/√(χ²_ν/ν) with log q = −(ν+d)/2·log(1 + zᵀz/ν), up to a constant.
  - Gaussian log q.
  - The PSIS weights are normalised, so ESS = 1/Σw².
  - Systematic resampling.
  - Against the exact K = 3 posterior, the means and SDs are within about 0.002 with k̂ = 0.27–0.54. The uncorrected version is visibly biased and is flagged, as intended (`a08`).
