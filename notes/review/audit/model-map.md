# Audit: model and MAP slice

Slice: `src/pastry/model.py`, `_kernels.py`, `_simplex.py`, `samplers/base.py`,
`samplers/optimize.py`. Scripts and outputs are in
`/tmp/claude-0/-home-user-pastry/63ffb1b4-5826-5f0b-ade5-c8209761737e/scratchpad/audit/model-map/`
(prefix `$A/` below). Line numbers refer to branch `claude/pastry-v0.1` at `eba1285`.

## Summary

The core math holds up. The log posterior (both selection modes, rates), its gradients,
the ILR Jacobian, the EM updates and imputation, the φ ∝ Sρ mapping in mix-SQP, and every
MAP path all agree with finite differences and brute-force optimisation to 1e-8 or better.
The problems are around the edges:

- **One major bug (M-1).** The bootstrap path `run(weights=...)` breaks EM when a zero-weight
  row is the only support of a bin. With the default `GammaRates`, 79 of 200 bootstrap
  replicates came back as all-zero rates (NaN fractions) on a simple example, with no warning.
- **`missing_information` is wrong with selection (M-2).** It gives 0.50 where the true EM
  rate is 0.85, and it ignores the prior. A corrected formula matches the numerical EM
  Jacobian to 4 decimals.
- **`kkt_gap` problems.** It returns 0.0 (a "certificate") at points where the likelihood is
  −inf (M-3). For rates it depends on the units of the exposure (M-4).
- **Efficiency.** On large dense problems mix-SQP MAP spends about 90% of its time outside
  the solver, in an SVD and a 50-step EM warm start (M-5).
- **Smaller items.** The rest are consistency problems (three meanings of
  `PointEstimate.log_posterior`, `tol`), a wrong scale in a start value, and simplifications.
  The main simplification is a single "model with these weights" helper, which would also
  fix M-1.

Counts: bug-major 1, bug-minor 4 (one out of slice), statistical 1, numerical 2 (M-6 is
also a design point), efficiency 1, consistency 2, simplify 3, docs 0 (docstring mismatches
are filed under consistency/simplify): 14 findings.

---

### M-1. Zero-weight rows make EM reset to uniform or collapse rates to zero (bootstrap path)

**Category** bug-major
**Location** `_kernels.py:71` (impossibility check over all rows); `samplers/base.py:42-56`
(`em_update`); `samplers/optimize.py:278-293` (`_objective`); reached through
`bootstrap.py:88` → `MAP.run(weights=wb)`.

**Claim.** `loglike_and_counts` returns `(-inf, zeros)` if *any* row has s_i ≤ 1e-250,
including rows whose (bootstrap) weight is 0. `em_update` does not check the returned
log-likelihood and feeds the zero counts into `map_from_counts`:
- for `Dirichlet(1)`, the result is the uniform vector;
- for `GammaRates(shape=1)`, it is all zeros, an absorbing state.

The trigger is a bin whose only support comes from rows with zero weight. One EM step sends
ρ_k to exactly 0 (n̂_k = ρ_k·Σ w_i L_ik/s_i = 0). The zero-weight row then has s_i = 0, and the
next step resets. The bootstrap gives each object weight 0 with probability about 37%, so this
happens whenever narrow likelihoods leave a bin supported by one object.

The same reset happens without bootstrap weights for `space="simplex"` with α < 1 and a small
non-integer weight (`map_from_counts` clips n_k + α − 1 < 0 to 0).

L-BFGS is immune because it drops zero-weight rows (`optimize.py:371-374`). mix-SQP is also
immune, since it backtracks so ρ_k never reaches 0 exactly, although it needs about 5× more
iterations. The EM paths are not immune. EM is the default for `GammaRates(shape≥1)`, for
Dirichlet with α ≥ 1 and α ≠ 1 under selection, and for bounded tree priors.

**Evidence**
- `$A/m01_zero_weight_rows.py`: 41 objects, bin 4 supported only by the last object, which
  gets weight 0. `Dirichlet(1.0) em` returns `[0.2 0.2 0.2 0.2 0.2]` with `conv=True, n_iter=1`.
  The reference with the row dropped is `[0.25 0.25 0.25 0.25 0]`.
- `$A/m01c_rates_count.py`: `Bootstrap(PopulationModel(lk, GammaRates()), rng=3).run(200)`
  gives `replicates with all-zero rates: 79 of 200`, `NaN fractions: 79`, and no warning
  (Bootstrap suppresses `ConvergenceWarning`).
- `$A/m08_edge_cases.py` (last line): `Dirichlet(0.5)`, `space="simplex"`, one object of
  weight 0.3 alone in bin 3 returns `[0.25 0.25 0.25 0.25]`, `conv True`.

**Suggested fix.** Drop zero-weight rows once, in a single place (see M-10:
`model.with_weights(w)` returning a shallow copy with `_L`, `_w` and `W` restricted to
`w > 0`), and use it in all three algorithms. Also make `em_update` (or `MAP._run_em`) raise
or warn when the log-likelihood is −inf instead of using zero counts. The silent reset to the
prior mode is the dangerous part. A regression test: rate-model bootstrap on the m01c data
should give no all-zero replicate.

**Confidence** confirmed by running.

### M-2. `missing_information` is wrong with a selection function, and ignores the prior

**Category** statistical
**Location** `model.py:282-317`

**Claim.** The function computes λ_max(I_c⁻¹ I_m) with I_m = diag(n̂) − Σ w r rᵀ (from L̃)
and I_c = W(diag ρ − ρρᵀ). That is the EM rate only without selection and without a prior.

With selection, EM (and Gibbs) impute the missed objects, so the complete-data information
is (W/(Ŝ·ρ))(diag ρ − ρρᵀ), with Ŝ = `_detect`. The observed information is
W(diag φ − φφᵀ) − I_m, where φ = Sρ/(S·ρ), not I_c − I_m. Using I_c − I_m, the generalised
eigenvalue can exceed 1, and the result is then clipped to exactly 1.0.

The prior adds (A − K)(diag ρ − ρρᵀ) to both matrices for the MAP-EM that `_run_em` uses.
The value feeds the EM non-convergence message (`optimize.py:324`) and the Gibbs
autocorrelation hint in `results.check` (`results.py:602`). With selection, both messages
are wrong.

**Evidence.** `$A/m05_missing_info_rate.py` compares against the spectral radius of a
numerical Jacobian of the package's own `em_update` at the fixed point (Dempster, Laird &
Rubin):

```
sigma=0.05 fractions, sel true      missing_information=0.4995  EM Jacobian radius=0.8466
sigma=0.05 fractions, sel observed  missing_information=0.5099  EM Jacobian radius=0.8458
sigma=0.05 fractions, Dirichlet(50) missing_information=0.3445  EM Jacobian radius=0.3137
sigma=0.12 fractions, sel true      missing_information=1.0000  EM Jacobian radius=0.9802
sigma=0.12 fractions, Dirichlet(50) missing_information=0.9667  EM Jacobian radius=0.8804
(no selection, rates, rates+selection: exact agreement)
```

`$A/m05b_corrected_formula.py` implements f = 1 − λ_min(I_obs, I_c) with the matrices above.
It matches the Jacobian radius to 4 decimals in all 8 cases (for example 0.8466/0.8466 and
0.9968/0.9968).

**Suggested fix.** In η coordinates (projected with V for fractions):
- I_obs = W(diag φ − φφᵀ) − I_m + (A − K)(diag ρ − ρρᵀ), where the last term applies for
  Dirichlet MAP and the selection part only with selection;
- I_c = (W/(Ŝ·ρ) + A − K)(diag ρ − ρρᵀ);
- return 1 − λ_min(I_obs, I_c).

This reduces to the current formula when there is no selection and α = 1. Rates need no
change (Poisson thinning, no imputation). Decide whether the prior term belongs in the
diagnostic. For Gibbs the relevant quantity has no prior term. A `include_prior` flag, as in
`kkt_gap`, would cover both uses.

**Confidence** confirmed by running.

### M-3. `kkt_gap` reports 0.0 (optimal) at a point where an object has zero evidence

**Category** bug-minor
**Location** `model.py:266-280`

**Claim.** If s_i = 0 for some object, w/s gives inf, the matvec gives NaN, and
`max(0.0, np.max(G - mu))` returns 0.0, because Python's `max` keeps the first argument when
it is compared with NaN. The certificate therefore says "optimal" exactly where the
log-likelihood is −inf. Users call `pastry.diagnostics.kkt_gap` on arbitrary values, for
example values from other codes.

**Evidence.** `$A/m08_edge_cases.py` prints
`kkt_gap at [0, .5, .5] (object 0 impossible): 0.0` with RuntimeWarnings, and
`log_likelihood there: -inf`.

**Suggested fix.** Return `np.inf` when `s.min() <= 0` (reuse the `loglike_and_counts`
guard), and use `np.nanmax`/explicit NaN checks rather than Python `max`.

**Confidence** confirmed by running.

### M-4. `kkt_gap` for rates depends on the units of the exposure

**Category** numerical
**Location** `model.py:276-278`

**Claim.** For rates, G_k = Σ w L_ik/s_i − S_k + ∂log p has units 1/λ. Dividing by W does not
make it dimensionless. Rescaling the exposure by c rescales λ̂ by 1/c and the gap by c, so the
same relative error can read 1e-8 or 1e-2. The fractions branch is unit-free because
v·G ~ W.

**Evidence.** `$A/m04_kkt_rates_units.py`, same data, 1% perturbation of the MAP:

```
exposure unit 0.001: at 1.01*MAP 1.76e-08
exposure unit 1:     at 1.01*MAP 1.76e-05
exposure unit 1000:  at 1.01*MAP 1.76e-02
fractions (any unit): 7.07e-03
```

`$A/m09_em_false_convergence.py` shows the practical effect: a non-converged plain EM
(log-likelihood deficit 2e-2) and a converged SQUAREM run both report a rates KKT gap of
1.1e-5.

**Suggested fix.** Use a relative violation, for example (S_k + b_k)⁻¹·G_k, which equals
n̂_k/(λ_k S_k) − 1 without a prior and is the analogue of the fractions g_k − 1. Another option
is λ_k|G_k|/W on the support plus max(G_k, 0)/(S_k + b_k) off it.

**Confidence** confirmed by running.

### M-5. mix-SQP MAP spends about 90% of its time in a needless SVD and the 50-step EM start

**Category** efficiency
**Location** `optimize.py:47-52` (SVD for dense N > 10K); `optimize.py:247` →
`base.py:309-326` (`interior_start`, 50 EM steps)

**Claim.**
- **The SVD.** The low-rank Hessian costs a full SVD of the N×K matrix. That costs
  O(NK²), as much as several exact Hessians Lᵀ diag(d) L, while mix-SQP typically converges in
  3–6 iterations, and when L has full numerical rank (narrow PDFs) the SVD saves nothing.
- **The warm start.** mix-SQP does not need `interior_start`. It converges from uniform in a
  handful of iterations (Kim et al. start from uniform), but `interior_start` performs 50 full
  EM passes (100 matvecs) first.

**Evidence.** `$A/m07b_profile.py` (N = 2e5, K = 40, dense) takes 10.2 s in total:
`svd` 4.45 s, `interior_start` 4.25 s, mix-SQP iterations about 1 s.
`$A/m07_mixsqp_lowrank_cost.py`:

```
K=40  sigma=0.01: rank 40/40;  SVD 1.4s; full MAP 2.9s (3 it); one direct Hessian 0.12s
K=200 sigma=0.2 : rank 17/200; SVD 47s;  full MAP 134s (5 it); one direct Hessian 10.5s
```

These timings were taken with another job running; the ratios are what matter.

**Suggested fix.**
- Start mix-SQP from uniform, or from `self.init`. Keep `interior_start` for samplers.
- Drop the SVD, or replace it with a cheap rank test on the K×K Gram matrix (`eigh(LᵀL)`). Use
  the low-rank form only when r ≤ K/2, and compute it lazily after the first iteration if
  the solver has not already converged.

Expected speed-up is about 5–10× for large dense inputs.

**Confidence** confirmed by running (profile).

### M-6. EM/SQUAREM stopping rule is not tied to optimality; rates always use EM

**Category** numerical / design
**Location** `optimize.py:315-320`; `_mixsqp_applies` (`optimize.py:226-242`)

**Claim.** `_run_em` stops when the relative objective change is ≤ `tol` and
max|Δv|/max|v| ≤ 1e-7. When the missing information is near 1, each EM step moves only a
fraction (1 − f) of the remaining distance, so both criteria can be met far from the mode.
SQUAREM reduces this but does not remove it. The KKT gap is computed afterwards but not used
to decide convergence. Rates with `GammaRates` always use this path, although for shape = 1
the problem reduces exactly to mix-SQP.

**Evidence.** `$A/m09_em_false_convergence.py`. Rates `GammaRates(1, 1e-12)` (default
algorithm SQUAREM) are compared against the exact answer W·ρ̂ (mix-SQP fractions MLE, KKT
7e-10):

```
N=2000 K=40 sig=0.1 squarem=True : n_iter=2327  converged=True  kkt=1.1e-05 loglik deficit=5.7e-06 max|lam-lam_ref|/N=1.4e-04
N=2000 K=40 sig=0.1 squarem=False: n_iter=20000 converged=False kkt=1.1e-05 loglik deficit=2.1e-02 max|lam-lam_ref|/N=5.3e-02
(reference: mix-SQP fractions MLE x N, kkt 7e-10, 4 iterations)
```

So SQUAREM declares convergence after 2327 iterations, with a log-likelihood deficit of 6e-6
and per-bin errors of 1.4e-4·N. This is small statistically, but mix-SQP reaches 7e-10 in 4
iterations. Plain EM (`accelerate=False`) is far off after 20000 iterations, and it does
report non-convergence. Both runs show the same rates KKT gap (M-4). The larger N = 20000
cases were stopped because they took too long.

**Suggested fix.**
1. Make the stopping rule (or at least `converged`) require `kkt_gap ≤ tol_kkt`, with M-4
   fixed for rates.
2. For `GammaRates` with all shapes = 1 (and any b), use mix-SQP. The rate likelihood
   factorises as λ = Λρ. Profiling out Λ = W/((S + b)·ρ) leaves
   Σ w log(L̃·ρ) − W log((S + b)·ρ), which is exactly the selection problem mix-SQP already
   solves with φ ∝ (S + b)ρ.

**Confidence** confirmed by running (part 1); the reduction in part 2 is derived, not
implemented.

### M-7. `interior_start` for rates with selection is off by a factor 1/(S·φ)

**Category** bug-minor
**Location** `samplers/base.py:327-331`

**Claim.** In the loop, `rho` is the composition of the *observed* objects: for rates the
loop does not impute missed objects, and its fixed point satisfies n̂ ∝ S_kλ_k. The MLE-like
start is therefore λ_k = W·ρ_k/S_k. The code divides additionally by S·ρ, which inflates the
start by 1/(S·φ). This is only a starting point, but it is the start for Gibbs, NUTS and EM
on rate models.

**Evidence.** `$A/m06_interior_start_rates.py`: start total 14664 against MLE total 8886
(true mode), and 16886 against 10063 (observed mode). Without selection it is exact (2000
against 2000).

**Suggested fix.** `return np.maximum(model.W * rho / np.maximum(S, 1e-12), 1e-12)`.

**Confidence** confirmed by running.

### M-8. `PointEstimate.log_posterior` has three different meanings

**Category** consistency
**Location** `optimize.py:269, 335` (`_objective`: relative log-likelihood, no `_ll_const`,
simplex density) vs `optimize.py:408` (L-BFGS: `log_prob_and_grad`, which includes
`_ll_const` and the ILR Jacobian, i.e. a density in x).

Related: with `weights`, EM and mix-SQP report `log_likelihood = kkt_gap = nan`, while L-BFGS
reports the weighted model's log-likelihood.

**Claim.** The value cannot be compared across algorithms, nor with
`model.log_posterior(value)`.

**Evidence.** `$A/m11_logpost_definitions.py` (rows scaled by 7, `Dirichlet(2)`):

```
mixsqp / em : log_posterior=-204.6480   model.log_posterior(value)=142.5548
unconstrained: log_posterior=133.5366   model.log_posterior(value)=142.4407
model._ll_const = 347.20
```

**Suggested fix.** Report `model.log_posterior(value)` (absolute, simplex density) everywhere.
For L-BFGS also store the x-space value under `info["log_density_x"]`. Use the same
nan-or-value policy for the weighted path in all algorithms.

**Confidence** confirmed by running.

### M-9. `tol` means something different in each algorithm

**Category** consistency
**Location** `optimize.py:150-151` (docstring "Relative tolerance on the objective"); 249
(mix-SQP: KKT tolerance = 100·tol); 315-318 (EM: objective tol plus a hard-coded 1e-7 on
Δv); 390 (L-BFGS ignores `tol`).

**Claim.** A user who loosens `tol` to save time gets no effect from L-BFGS and only a
partial effect from EM. The docstring describes only the EM objective test.

**Suggested fix.** Define `tol` as a tolerance on the (relative) KKT gap or gradient norm for
every algorithm, document it, and drop the hidden 1e-7 (or tie it to `tol`).

**Confidence** confirmed by reading.

### M-10. Weight overrides are implemented three times (and differently)

**Category** simplify
**Location** `base.py:42-50` (`em_update` recomputes counts with the new weights);
`optimize.py:280-289` (`_objective` re-implements `_ll_and_deta` for weights);
`optimize.py:371-374` (L-BFGS rebuilds a full `PopulationModel` from a likelihood subset).

**Claim.** Three code paths for one idea, and only the third drops zero-weight rows, which is
the root of M-1. The second duplicates the selection and rates normaliser logic of
`_ll_and_deta`.

**Suggested fix.** Add `PopulationModel.with_weights(w)` returning
`copy.copy(self)` with `_L`, `_w`, `W` and `_ll_const` restricted to `w > 0`, alongside
`_with_bound_prior`. `MAP.run` then does `m = self.model if weights is None else
self.model.with_weights(weights)`. `em_update` loses its `weights` argument, and `_objective`
becomes `m.log_posterior` (see M-8).

**Confidence** confirmed by reading.

### M-11. Selection imputation (expected missed counts) is written out three times

**Category** simplify
**Location** `samplers/base.py:53-56`, `base.py:319-321`, `samplers/vb.py:66-68`
(the Gibbs draw `gibbs.py:129-135` is the stochastic version).

**Claim.** `nhat + W * rho * (1 - S) / (S @ rho)` with `S = model._detect` appears three
times, and every copy has to remember to use `_detect` rather than `selection`.

**Suggested fix.** Add `PopulationModel.complete_counts(value, nhat, W=None)` (or
`expected_missed(value, W)`) and call it from EM, `interior_start` and VB.

**Confidence** confirmed by reading.

### M-12. Dead code and small cleanups

**Category** simplify
**Location** various

- `optimize.py:242`: `np.all(m._detect > 0)` is always true, because `Selection` rejects
  values ≤ 0 (`selection.py:62`).
- `_mixsqp_applies` duck-types with `getattr(prior, "alpha")` / `hasattr(prior, "tree")`
  (`optimize.py:229-235`). `isinstance(prior, _BoundDirichlet)` would say what is meant.
- `_kernels.is_sparse` is never used (the kernels call `sp.issparse`).
- `_simplex.ilr`, `ilr_inv` and `clr` are used only by tests.
- `base.py:316`: `nhat = None` is dead.
- Function-local imports: `helmert_basis` in `missing_information` (`model.py:310`), `kern`
  in `em_update` and `_objective`. There is no import cycle that requires them.
- `model.py:105-109` duplicates `effective_likelihoods` (`model.py:137-142`); `__init__`
  could call it.
- `missing_information` names its subset index `sel`, which in the rest of the class means a
  selection.
- `interior_start` docstring says "a few" EM steps; it is 50 full passes (see M-5).

**Suggested fix** as listed.
**Confidence** confirmed by reading / grep.

### M-13. `optimize.nnls` failure is not caught

**Category** bug-minor
**Location** `optimize.py:99`

**Claim.** With scipy ≥ 1.12, `nnls` raises `RuntimeError("Maximum number of iterations
reached.")` after `maxiter=50K` (verified in the installed scipy 1.18.1 source). An
ill-conditioned Hessian would then crash MAP, and Bootstrap with it, rather than fall back.
This was not triggered in 460 random problems.

**Suggested fix.** Catch `RuntimeError`, then either take a projected-gradient step or fall
back to EM, and mark `converged=False`.

**Confidence** speculative (code path verified; no failing input found).

### M-14. (Out of slice, cross-reference) `LogisticGP()` crashes for K = 2

**Category** bug-minor (belongs to `priors.md`)
**Location** `priors/logistic.py:517`

**Claim.** With 2 equal bins, 2·min(width) = range, so `InverseGamma.from_quantiles(lower,
upper)` raises "Need 0 < lower < upper." Found while testing K = 2 in
`$A/m08_edge_cases.py`.

**Confidence** confirmed by running.

---

## Checked and found correct

- **Log posterior, independently recomputed.** I recomputed it for no selection, `"observed"`
  and `"true"` modes, and rates with exposure. Dense and sparse; non-integer weights;
  Dirichlet (scalar, vector, α < 1), `DirichletTree`, `GammaRates` and `LogisticGP`. It
  matches `log_likelihood(absolute=False)` to 1e-12 (`$A/m02_gradients.py`).
- **Gradients.** `log_prob_and_grad` gradients match central finite differences in every
  combination above, with worst relative error 7e-9 (`m02`).
- **ILR Jacobian.** The numerical log|det| of ∂ρ₁..K−1/∂x equals ½ log K + Σ log ρ_k to 6e-11.
  The Dirichlet x-space density α·log ρ + const is exact (`m02`).
- **mix-SQP.** Default MAP (mix-SQP, including selection via φ ∝ Sρ) was compared with
  multi-start SLSQP on 60 random problems: K ∈ {2, 3, 5, 12, 30}, N from 5 to 400, dense and
  sparse, integer, non-integer and real weights, both selection modes, α ≥ 1. The objective
  was never worse than brute force by more than round-off (`m03`).
- **mix-SQP KKT.** In 400 further random problems: no non-convergence, and the KKT gap was at
  most 6e-8 (`m03b`). The relaxed-objective constant c = (W + A)/W does force Σx = 1 at the
  optimum, and the NNLS form of the QP is correct.
- **EM-path MAPs.** Dirichlet(α > 1) with selection (both modes), `GammaRates` with selection
  (both modes) and a bounded `DirichletTree` all match brute-force BFGS to 1e-8 relative, with
  objective differences ≤ 1e-10 (`m10`). The EM M-steps, the rates M-step
  (a + n − 1)/(b + S) and the missed-object imputation W ρ_k(1 − Ŝ_k)/(Ŝ·ρ) are correct.
  Using `_detect` (max = 1) leaves the fixed point unchanged and gives the smallest
  augmentation, hence the fastest EM.
- **KKT gap, fractions.** The multiplier μ = v·G and the one-sided max(G − μ) together check
  both complementarity and dual feasibility. Values are about 0 at verified optima with prior
  and selection terms.
- **`missing_information` without selection.** Without selection, for fractions and for rates
  (with or without exposure), it equals the EM Jacobian radius exactly (`m05`).
- **SQUAREM.** The SqS3 step length, backtracking toward −1, the stabilising EM step and the
  monotonicity check follow Varadhan & Roland (2008).
- **L-BFGS paths.** With `weights`, the L-BFGS path drops zero rows and rebuilds selection
  correctly.
- **K = 2.** Works in every MAP path except the M-14 prior crash (`m08`).
- **`_kernels`.** The `responsibilities` zero-row guard and the CSR label sampler
  normalisation are correct, and non-integer weights are routed away from `sample_counts` by
  its callers.
- **`L.T @ y`.** It is as fast as `y @ L` for C-ordered dense L, so there is nothing to gain
  in `rmatvec` (`m12`).
