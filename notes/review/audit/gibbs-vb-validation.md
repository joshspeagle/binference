# Audit: Gibbs, VB, bootstrap, validation, simulate

Slice: `samplers/gibbs.py`, `samplers/vb.py`, `samplers/bootstrap.py`, `validation.py`,
`simulate.py`, plus `_kernels.sample_counts`/`sample_labels`, which only Gibbs uses.
Scripts and outputs are in
`/tmp/claude-0/-home-user-pastry/63ffb1b4-5826-5f0b-ade5-c8209761737e/scratchpad/audit/gibbs-vb-validation/`
(`aNN_*.py`, `aNN_out.txt`). Source files were not modified.

## Summary

- **The samplers are statistically correct.** Gibbs matches the exact posterior for K = 3 in
  every configuration tested: plain, sparse, selection in both modes, multiplicity weights, and
  weights with selection (|z| ≲ 2, SD ratio within 1.5%). It also passes a small SBC run. With
  α = 0.5 Gibbs appeared to disagree with `exact_posterior` (z ≈ −7). An independent accurate
  quadrature shows the reference is wrong, not Gibbs (G-3).
- **SBC p-values have a calibration bug (G-1).** Rank histograms use bins that hold unequal
  numbers of integer ranks, but the χ² test assumes equal bins. With the tutorial's settings
  (`n_draws=49`, 150 simulations) a perfectly calibrated sampler is rejected at α = 0.01 9.3% of
  the time. With 400 simulations the rate is 39%.
- **Gibbs label step with multiplicity weights and sparse likelihoods (G-2).** It densifies every
  block: one iteration takes 13 s and 1.5 GB, against 0.07 s and 8 MB with unit weights.
- **`exact_posterior` is inaccurate at the simplex boundary (G-3).** Its lattice treats the third
  coordinate differently from the other two, drops a half-cell strip, and uses a midpoint rule on
  ρ^(α−1) singularities. For α < 1 the error in the mean reaches 0.1–0.6 posterior SD, and
  boundary quantiles are off by factors of 3 to 10⁶. Its memory also grows as N × 20000 doubles
  (G-5).
- **Smaller issues.** Mocks accept a detection probability above 1 and then generate data from a
  different model than the one attached (G-4). Gibbs has some gaps in input validation (G-6).
  There is duplicated selection logic and other cleanups (G-7 to G-12).
- **VB and bootstrap.** VB behaves as documented: means are right and widths are 0.3–0.75 of the
  true widths. The bootstrap weights (nonparametric, Bayesian, rates) are correct after the earlier
  fixes.

---

### G-1. SBC rank histograms use unequal-width bins, so the χ² uniformity test is miscalibrated

- **Category:** statistical
- **Location:** `src/pastry/validation.py:132-150` (`SBCResult.histogram`, `_default_bins`,
  `pvalues`, `passed`). The same issue affects the 99% band in `src/pastry/plotting.py:335`.
- **Claim:** Ranks take the n_draws + 1 integer values 0..n_draws. The bin edges are
  `linspace(0, n_draws+1, n_bins+1)`. When n_bins does not divide n_draws + 1, bins hold different
  numbers of integer ranks (for example 2 or 3). Their expected counts under uniformity therefore
  differ by up to 50%. `stats.chisquare(h)` assumes equal expected counts, so calibrated samplers
  are rejected too often. The default `n_bins = min(20, n_sims//5, n_draws+1)` produces such
  combinations routinely. The tutorial (`docs/tutorials/07_validation.md:69,90`: `n_draws=49`,
  `n_simulations=150`) and the user guide (`n_draws=49`) are both affected.
- **Evidence:** `a03_sbc_bins.py` draws exactly uniform ranks and runs `pvalues()` 2000 times
  (`a03_out.txt`):

  | n_draws | n_sims | bins | ranks per bin | P(p ≤ 0.01), nominal 0.01 |
  |---|---|---|---|---|
  | 99 | 100 | 20 | 5/5 | 0.011 |
  | 99 | 90 | 18 | 5/6 | 0.021 |
  | 100 | 100 | 20 | 5/6 | 0.015 |
  | 49 | 150 (tutorial) | 20 | 2/3 | **0.093** |
  | 49 | 400 | 20 | 2/3 | **0.391** |

  The shipped slow tests use `n_draws=99` with `n_bins=10`, which divides evenly, so they are not
  affected.
- **Suggested fix:** Either:
  - pass `f_exp` proportional to the number of integer ranks in each bin, for example
    `np.histogram(np.arange(n_draws+1), edges)[0] * n_sims / (n_draws+1)`; or
  - restrict `_default_bins` to divisors of `n_draws+1` and raise on a non-dividing `n_bins`.

  Use the same expected counts for the band in `plot_sbc`. An ECDF-difference test (Säilynoja et
  al. 2022) would avoid binning altogether.
- **Confidence:** confirmed by running.

### G-2. Gibbs with multiplicity weights densifies sparse likelihoods: 200× slower and about 1.5 GB per iteration

- **Category:** efficiency
- **Location:** `src/pastry/_kernels.py:167-176` (`sample_counts`, non-unit path), called every
  iteration from `src/pastry/samplers/gibbs.py:125`.
- **Claim:** For integer weights other than 1, each chunk goes through
  `responsibilities(L[sl], rho)`, which calls `.toarray()`. That builds a dense
  `(min(N, 65536), K)` matrix, followed by an `np.clip` copy and `R / R.sum()`. Then
  `rng.multinomial` runs over all K cells of every row. Sparse storage therefore does nothing
  exactly where it is used for large-K, compressed catalogues.
- **Evidence:** `a11_gibbs_sparse_weights_memory.py` (`a11_out.txt`), one `sample_counts` call with
  N = 20000, K = 2500 and 10 nonzeros per row:
  - unit weights: 0.07 s, +8 MB;
  - weights = 2: 13.2 s, +1520 MB.
- **Suggested fix:** For sparse L, avoid the dense path. Either:
  - draw Σw labels with the existing CSR inverse-CDF sampler on rows repeated by multiplicity, in
    chunks; or
  - draw multinomials over each row's nonzeros, grouped by nonzero count.

  For dense L, a smaller chunk would at least bound the memory used.
- **Confidence:** confirmed by running.

### G-3. `exact_posterior` quadrature is biased near the simplex boundary, badly so for α < 1

- **Category:** numerical
- **Location:** `src/pastry/validation.py:79-87` (grid) and `:53-58` (`quantile`)
- **Claim:** For K = 3 the grid uses midpoints (a, b) = ((i+½)/n, (j+½)/n) with a + b < 1. This
  has three problems:
  1. ρ₃ = 1 − a − b sits on multiples of 1/n (smallest value 1/n), while ρ₁ and ρ₂ sit on
     half-integers (smallest value ½n). The rule is not symmetric between components.
  2. The half-cells along a + b = 1 are dropped. That is a strip of area 1/(2n), all at ρ₃ ≈ 0.
  3. A midpoint rule on the integrable singularities ρ_k^(α−1), for α < 1, misplaces boundary
     mass. The same happens for K = 2.

  Marginals near the boundary are therefore wrong: means are biased, SDs too small or too large,
  and lower quantiles off by orders of magnitude.
- **Evidence:**
  - `a02_exact_quadrature.py` (`a02_out.txt`) uses data with a known Dir(α + counts) posterior
    and `n_grid=400`:
    - K = 3, α = 0.5, counts (20, 15, 0): the error in the ρ₃ mean is +0.23 posterior SD. The
      5% quantile is 2.5e-3 against the true 5.5e-5.
    - K = 3, α = 0.2: the ρ₃ SD ratio is 1.33 and the ρ₃ mean error is +0.63 SD.
    - K = 2, α = 0.2: the SD ratio is 0.95 and the 5% quantile is 2.0e-3 against the true 7.7e-6.
    - Even with α = 1 the ρ₃ mean is off by 0.05 SD, while ρ₁ is off by only 0.009 SD, which
      shows the asymmetry.
  - `a01_gibbs_vs_exact.py`: Gibbs against `exact_posterior` with α = 0.5 gives z = −6.7 (dense)
    and −7.3 (sparse) for ρ₃.
  - `a08_accurate_quadrature.py` builds a stick-breaking, cosine-substitution Gauss–Legendre
    reference. It recovers Dir(α) means to 0.003 SD even at α = 0.2. On the a01 case its ρ₃ mean
    is 0.04010. Long Gibbs runs give 0.04021 ± 0.00039, and `exact_posterior` gives 0.04593
    (an error of 0.12 SD).

  The current repo tests use α ≥ 1 and loose tolerances, so they pass. The function is
  nonetheless documented as the reference for testing engines.
- **Suggested fix:** Use a boundary-aware rule. The a08 construction is about 15 lines: stick
  breaking ρ = (x, (1−x)y, (1−x)(1−y)), with x = (1 − cos πs)/2 and Gauss–Legendre in s and t,
  carrying Jacobian weights. Return per-point weights that include the Jacobian. A cheaper option
  is a symmetric centroid rule on the triangulated simplex, but it does not fix the α < 1
  singularity. Document the accuracy either way.
- **Confidence:** confirmed by running.

### G-4. Mocks accept detection probabilities above 1 and then simulate a different model

- **Category:** bug-minor
- **Location:** `src/pastry/simulate.py:175-179` (`_select`), and `mock_1d`/`mock_2d`
  (`selection=`)
- **Claim:** `_select` thins with `rng.random() < S[label]`, so the mock really uses
  min(S, 1). The raw S is attached to the likelihoods, and the model uses its ratios, which the
  `_detect` rescaling allows. For any S above 1 the data and the attached model disagree, and
  inference is silently biased.
- **Evidence:** `a10_misc.py` (`a10_out.txt`): with `selection=[2, 1]` and true ρ = (0.5, 0.5),
  49.7% of the observed objects are in bin 0, where the model expects 66.7%. The MAP returns
  ρ = (0.329, 0.671).
- **Suggested fix:** Require 0 ≤ S ≤ 1 in the mocks and raise a `ValueError` otherwise. Using
  S / max(S) would also work, but then the rescaling must be documented.
- **Confidence:** confirmed by running.

### G-5. `exact_posterior` memory scales as N × 20000 doubles per chunk

- **Category:** efficiency
- **Location:** `src/pastry/validation.py:65, 91-95`
- **Claim:** `S = L @ P.T` and `np.log(S)` build two `(N, chunk)` dense arrays with
  `chunk=20000` grid points, whatever the size of N. Typical mocks have N = 2000.
- **Evidence:** `a06_exact_memory.py` (`a06_out.txt`), with `n_grid=200` (one chunk): peak RSS
  grows by 306 MB at N = 1000 and by 1521 MB at N = 5000.
- **Suggested fix:** Use `chunk = max(1, int(budget // N))` with a budget of about 2e6 to 1e7
  elements. Alternatively, sum `w_i log(L_i·P)` over object blocks.
- **Confidence:** confirmed by running.

### G-6. Gibbs input handling: `chunk=None` crashes, label indices fail late, `init` is shared by all chains

- **Category:** bug-minor
- **Location:** `src/pastry/samplers/gibbs.py:90, 98, 111-113`
- **Claim:**
  - `int(chunk)` rejects `None`, although every kernel accepts `chunk=None` (no chunking).
  - `store_labels` indices are not range-checked. An out-of-range index raises `IndexError` only
    after the whole burn-in. `np.unique` also silently reorders and deduplicates the indices; the
    result's `label_index` records this, but the docstring does not say so.
  - A user `init` is copied identically into every chain. That removes the overdispersion R-hat
    relies on, and nothing is validated: shape, positivity, or normalisation for fractions.
- **Evidence:** `a10_misc.py`: `Gibbs(m, chunk=None)` raises `TypeError: int() argument must be
  ... not 'NoneType'`. `store_labels=[0, 5000]` with N = 200 raises `IndexError` at sampling time.
- **Suggested fix:**
  - `self.chunk = None if chunk is None else int(chunk)`.
  - Check `0 <= idx < N` in `__init__`.
  - Validate `init` with shape (K,), finite and positive. Either document that all chains start
    from it, or jitter each chain's start, for example one Gibbs step from the init per chain.
- **Confidence:** confirmed by running.

### G-7. The "expected missed objects" formula is duplicated in four places

- **Category:** simplify
- **Location:**
  - `src/pastry/samplers/vb.py:65-68`
  - `src/pastry/samplers/base.py:53-56` (`em_update`)
  - `src/pastry/samplers/base.py:70-72` (`interior_start`)
  - the Gibbs equivalent, `src/pastry/samplers/gibbs.py:128-135`, which uses the same q = ρ(1−S)
    and p_det = S·ρ
- **Claim:** `W * rho * (1 - S) / (S @ rho)` with `S = model._detect` is written out three times,
  and Gibbs repeats the ingredients a fourth time. Every copy has to remember `_detect`, as opposed
  to `selection`, and the renormalisation of ρ.
- **Suggested fix:** Add one helper on `PopulationModel`, for example
  `_missed_counts(rho, W=None)` returning the expected missed counts, and possibly
  `_missed_draw(rho, rng)` for Gibbs. Call them from all four places.
- **Confidence:** confirmed by reading.

### G-8. Bootstrap: its own integer test, and replicate convergence failures are discarded

- **Category:** consistency
- **Location:** `src/pastry/samplers/bootstrap.py:75, 77-79, 86-88`
- **Claim:**
  - Integer detection uses `abs(w - round(w)) < 1e-9`. Gibbs uses `_utils.is_integer_array`,
    with tolerance 1e-12. Weights inside that gap are treated as multiplicities by one engine and
    rejected by the other.
  - `ConvergenceWarning` is suppressed for both the base fit and every replicate, and
    `est.converged` is dropped. Non-converged replicates are therefore included silently.
- **Suggested fix:** Use `is_integer_array`. Count replicates with `est.converged == False`,
  store the count in `info["n_not_converged"]`, and warn once if it is non-zero. Renaming the
  constructor argument `kind` would also help, since `result.kind` means fractions or rates; for
  example `scheme="nonparametric"|"bayesian"`.
- **Confidence:** confirmed by reading.

### G-9. SBC collects `min_ess` but does not use or document it

- **Category:** docs
- **Location:** `src/pastry/validation.py:114-130, 196-197, 220-222`
- **Claim:**
  - The `sbc` docstring says the thinned draws "should be roughly independent, i.e. fewer than
    the ESS", and the minimum bulk ESS is stored on the result. Nothing checks it, and the
    `SBCResult` docstring does not list `min_ess`.
  - Autocorrelated draws make the rank histograms ∩-shaped, which looks like over-dispersion.
- **Suggested fix:**
  - Document `min_ess`.
  - Warn once, when any simulation has `min_ess < n_draws`, naming the fraction of simulations
    affected.
- **Confidence:** confirmed by reading.

### G-10. Inconsistent conventions in mocks and validation outputs

- **Category:** consistency
- **Location:**
  - `src/pastry/simulate.py:395-445` (`mock_unfolding`) compared with `mock_1d` and `mock_2d`
  - `src/pastry/validation.py:211, 282`
- **Claim:**
  - `mock_1d` and `mock_2d` take the number of *observed* objects (`n_objects`, after selection).
    `mock_unfolding` takes the number of *true* events before detection (`n_events`), so the
    observed count is random. The docstring does not state this.
  - `sbc` and `coverage` label their quantities `rho[k]` even when they rank rates (λ) for rate
    models.
- **Suggested fix:**
  - State the convention in the `mock_unfolding` docstring ("events before detection;
    `truth.counts.sum()` are observed").
  - Name the quantities `lambda[k]` for rate results.
- **Confidence:** confirmed by reading.

### G-11. Simplifications in `simulate.py`

- **Category:** simplify
- **Location:** `src/pastry/simulate.py:254-275, 301-318, 248-253 vs 345-350`
- **Claim:**
  - The interim-prior resolution in `mock_1d` uses four branches and a `pi_dens`/`Pi` pair. It can
    be a single helper returning (density or None, masses), reusing `_as_density`. The `"gaussian"`
    special case is the only difference.
  - `_posterior_samples_1d` loops over objects in Python. It could be vectorised in chunks: a
    cumulative sum over the grid for a block of objects, then a row-wise `searchsorted`.
  - `mock_1d` and `mock_2d` repeat the same likelihood, counts, `Truth` and `Mock` tail.
- **Suggested fix:** As described. These are not urgent.
- **Confidence:** confirmed by reading.

### G-12. VB has no ELBO; convergence is judged on the step in the mean

- **Category:** design
- **Location:** `src/pastry/samplers/vb.py:52-77`
- **Claim:** No ELBO is computed, so there is no monotonicity check or evidence estimate.
  Convergence uses the relative change of E_q[ρ] per iteration, which measures the step, not the
  distance to the fixed point. With selection, the plug-in update for missed objects is not
  coordinate ascent on any ELBO, so monotonicity cannot be assumed. In practice it behaves well:
  stopping at 1e-9 is within 3e-7 relative, or 4e-6 posterior SD, of the 1e-14 answer
  (`a10_misc.py`). Every test case converged in under 4000 iterations, with noise up to 6 bin
  widths (`a05_vb_convergence.py`).
- **Suggested fix:** Optional. Compute the ELBO without selection, as a diagnostic stored in
  `info`. Alternatively, document that convergence is judged on the mean. No change is needed for
  correctness.
- **Confidence:** confirmed by running (behaviour). The design point is a judgement call.

---

## Checked and found correct

- **Gibbs conditionals.**
  - z ∝ L̃ρ, with L̃ including S in `"true"` mode.
  - ρ | n ~ Dir(α + n), or the tree slot update.
  - λ | n ~ Gamma(a + n, b + S).
  - Missed objects: M | ρ ~ NegBin(W, S·ρ), with numpy's failures-before-W-successes convention,
    and their bins ~ Mult(ρ ∘ (1−S)). This reproduces (S·ρ)^(−W) in both modes, and rescaling to
    `_detect` changes only a constant.
  - Labels are stored jointly with the ρ draw that conditions on them.
  - The empirical check (`a01_out.txt`) covers plain, sparse, `"true"` and `"observed"` selection,
    multiplicity weights, and weights with selection, all for K = 3. Against an accurate reference
    (a08), |z| ≲ 2.7 and the SD ratio is 0.98–1.05.
- **Label samplers.**
  - Dense inverse-CDF: u ∈ (0, total], so a zero-probability bin cannot be chosen.
  - CSR: per-row normalisation keeps rounding at about 1e-11 per chunk, and the clip stays within
    the row.
  - The multinomial path with per-row m_i is correct. Its efficiency is covered in G-2.
- **Exact zeros.** Gibbs cannot get stuck on exact zeros: `_dirichlet_draw` and `_gamma_draw`
  draw in log space and floor at 1e-300.
- **Gibbs bookkeeping.** Thinning, multiple `run()` calls, partial-generator `finally` handling
  and the label/draw alignment are correct.
- **Small SBC run.** `a09_sbc_small.py` (Dirichlet(1), K = 4, N = 50, 200 simulations): Gibbs
  passes, with all p > 0.02; VB fails as documented, with p = 0.
- **VB updates.**
  - `vb_expected_log`/`vb_update` for the Dirichlet and the rates.
  - The scale-invariant use of exp(E log ρ) in `loglike_and_counts`.
  - The plug-in missed counts W·ρ(1−S)/(S·ρ).
  - Means agree with the exact posterior to within 0.5 SD, with widths underestimated as
    documented.
- **Bootstrap weights.**
  - Nonparametric: Mult(W, w/W) for multiplicities; Poisson(w) for rates.
  - Non-integer weights: w·Mult(N, 1/N) or w·Poisson(1).
  - Bayesian: Gamma(w, 1), renormalised to W for fractions only.
  - The weighted MAP objective uses the replicate's own Σw in the selection term.
  - Replicates start at 0.5·base + 0.5·mean, which is interior.
- **`ExactPosterior` and `exact_posterior` (apart from G-3 and G-5).** Normalisation through
  logsumexp, mean, covariance, the midpoint-CDF `quantile`, the selection term with
  `model.selection` and L̃, and the K = 2 grid are correct.
- **SBC and coverage.**
  - Rank = #{draws < truth} ∈ 0..n_draws, for fractions or rates.
  - Even thinning gives distinct indices.
  - The joint log-likelihood rank uses `absolute=False` consistently.
  - The Bonferroni rule in `passed` is correct.
  - Coverage uses central `np.quantile` intervals, the nominal binomial standard error, and
    functionals on the model's scale.
- **Mocks.**
  - `truth.rho` (Gauss–Legendre bin masses) matches the label distribution actually sampled by
    `_draw_values`. Monte Carlo with 4×10⁶ draws on log-spaced bins gives max |z| = 2.4 over 30
    bins (`a04b_out.txt`). Note: `a04_mock_truth.py` assigns fine cells by their centre and is not
    the right comparison; `a04b` is.
  - Selection thinning on the true bin, and the `n_drawn` accounting, are correct.
  - Interim posteriors are L·Π normalised. Posterior samples are drawn by inverse CDF on a fine
    grid under the interim density.
  - In `mock_categorical`, L ∝ post/π recovers the Gaussian class likelihoods.
  - In `mock_unfolding`, the response R[j,k] = P(reco j, detected | true bin k), uniform within
    bin k, is consistent with the simulation. The uniform-within-bin assumption is documented.
  - The binning-bias notes are accurate.
- **`prior_predictive`.** It draws ρ from the bound prior and then N objects exactly under the
  histogram model, with values uniform within bins. It therefore samples the joint prior
  predictive of the model being fitted.
