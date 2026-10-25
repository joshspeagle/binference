# Audit: tests, docs and packaging (cross-cutting slice)

Auditor: independent background audit, 2026-10-22, branch `claude/pastry-v0.1` at `eba1285`.
Scripts and raw outputs: `/tmp/claude-0/-home-user-pastry/63ffb1b4-5826-5f0b-ade5-c8209761737e/scratchpad/audit/tests-docs/`.
Nothing under `src/`, `tests/` or `docs/` was modified.

## Summary

- **The slow statistical suite passes but has little power.** At the shipped sizes, SBC detects
  posteriors that are 30% too narrow 80% of the time, 20% too narrow 16% of the time, and 10%
  too narrow never. Coverage and recovery tests have similar blind spots. A realistic
  multiplicity bug (half the objects counted twice) was run through the shipped SBC
  configuration. With the shipped seed it *passes*: the bug is missed (T-1). The SBC suite also never covers inferred hyperparameters,
  the GP, selection, rates, Pólya/Dirichlet trees or Laplace (T-2).
- **Several fast tests do not test what they claim.** Examples: the bootstrap "interior start"
  regression test, VB "mean in the right place", mock selection thinning, and `sample_values`
  in-bin placement (T-3, T-11).
- **The coverage run** (90% line and branch) shows untested documented behaviour. This covers
  the divergence, k̂ and tree-depth hints of `check()`, VB with selection or rates, weighted
  L-BFGS (Bootstrap with non-conjugate priors), user `init` for NUTS, and 2-D smoothness priors
  (T-9, T-10).
- **One real bug surfaced from a docs snippet.** mix-SQP raises a `RuntimeWarning` (an error
  under the suite's `filterwarnings = error`) for Dirichlet priors that mix α = 1 with α > 1,
  e.g. `Dirichlet.from_counts` with empty bins (T-4).
- **Docs vs code.**
  - E-BFMI is documented as checked but is not (T-5).
  - `predict_proba` is documented as chunked but is not (T-6).
  - `engines.md` routes small-concentration priors to EM; the code routes them to L-BFGS (T-7).
  - Several smaller inaccuracies: install extras, a theory caveat, timings and the test plan
    are stale.
- **Docs code is not exercised in CI.** All user-guide snippets and the README do run today: I
  executed them and the README output reproduces exactly. Nothing in CI keeps them that way
  (T-8).
- **Packaging and CI are basically sound.**
  - The minimum-dependency job is viable: the full fast suite passes on Python 3.11 with
    numpy 1.25.2, scipy 1.11.4, matplotlib 3.7.5 and threadpoolctl 3.1.0.
  - Gaps: unpinned ruff, no job timeouts, no wheel smoke test, README relative links break on
    PyPI (T-14, T-15).

Counts by category (20 findings): test-gap 4 (T-2, T-4, T-9, T-10); weak-test 3 (T-1, T-3, T-11); docs-wrong 5 (T-5, T-6, T-7, T-16, T-17); consistency 3 (T-12, T-18, T-19); ci 2 (T-8, T-14); packaging 1 (T-15); simplify 2 (T-13, T-20).

---

### T-1. The slow statistical tests cannot detect moderate miscalibration

- **Category:** weak-test
- **Location:**
  - `tests/test_slow.py:36-57` (SBC);
  - `tests/test_slow.py:69-83` (coverage);
  - `tests/test_slow.py:164-218` (recovery).
- **Claim:**
  - The slow suite is the package's evidence that NUTS and Gibbs are calibrated (README,
    `validation.md`, test plan §2.11).
  - "Pass criterion: χ² uniformity test … p > 1e-3 after Bonferroni."
- **Evidence:**
  - **Method.** `sbc_power.py` simulates the exact pass criteria on a Gaussian toy: the engine
    reports N(m + b·σ, (cσ)²) where the truth is N(m, σ²).
  - **SBC.** Settings: 200 simulations, 99 draws, 10 rank bins, per-quantity
    p < 1e-3/5. Probability that one quantity rejects:

    | width factor c | 1.0 | 0.9 | 0.8 | 0.7 | 1.2 | 1.4 |
    |---|---|---|---|---|---|---|
    | reject | 0.00 | 0.00 | 0.16 | 0.80 | 0.02 | 0.59 |

    | bias b (σ) | 0.1 | 0.2 | 0.3 | 0.5 |
    |---|---|---|---|---|
    | reject | 0.01 | 0.03 | 0.25 | 0.97 |

    The four fractions sum to one, so they are strongly correlated and add little beyond a
    single quantity. The joint log-likelihood rank adds the most (see the realistic bug below).
  - **Coverage.** Settings: 300 simulations, |rate − nominal| < 4·SE per quantity. The 68%
    tolerance is ±0.108, so any 68% coverage between 0.57 and 0.79 passes.

    | width factor c | reject at 68% | reject at 95% |
    |---|---|---|
    | 0.9 | 0.02 | 0.06 |
    | 0.8 | 0.47 | 0.79 |
    | 1.2 | 0.19 | 0.00 |

  - **Recovery.** Settings: 240 z-scores, `0.75 < sd < 1.25`. Rejection is 0.01 at c = 0.9,
    0.08 at 0.85 and 0.48 at 0.8.
  - **Realistic bug.** I ran a multiplicity bug through the exact `gibbs-dirichlet` SBC
    configuration: half the objects get weight 2, so posteriors are about 18% too narrow. The
    script is `sbc_negative_control.py`. Results over 200 simulations each:

    | rng | min p | threshold | outcome |
    |---|---|---|---|
    | 2026 (the shipped seed) | 6e-4, on `loglike` | 2e-4 | **passes: the bug is missed** |
    | 1 | below 1e-4 | 2e-4 | fails |
    | 2 | below 1e-4 | 2e-4 | fails |

    At the shipped seed the per-bin p-values are 0.055–0.31; only `loglike` comes near the threshold. This is a coin flip that the shipped seed happens to lose.
  - **Existing negative controls.** There are two:
    - VB `xfail`, whose under-dispersion is a factor of 2 or more;
    - Geweke with the missed objects removed, at |z| > 8.

    Both are gross errors. Nothing shows the suite can see a 10–20% error, the size that
    quietly corrupts error bars.
- **Suggested fix:**
  - **More simulations.** Gibbs at K = 4, N = 40 costs well under a second per fit, so 1000+
    simulations are affordable for the conjugate engines.
  - **Better statistics.** Pool the rank test across quantities, or use the ECDF-difference
    test with simultaneous bands (Säilynoja, Bürkner & Vehtari 2022); both beat 10-bin χ².
  - **A calibrated negative control.** Add a test that the same SBC configuration *rejects* a
    pipeline with a 15–20% width error, e.g. the duplicated-weights fit above or a tempered
    likelihood. This documents the suite's power.
  - **Pooled recovery tests.** For recovery, test the pooled z-variance with its chi-square
    standard error instead of a ±25% window.
- **Confidence:** confirmed. The power numbers are simulated for the exact criteria; the
  pastry negative control was run.

### T-2. SBC and coverage miss the paths most likely to be wrong

- **Category:** test-gap
- **Location:** `tests/test_slow.py:36-57`; `src/pastry/simulate.py:451-476`; test plan §2.11.
- **Claim:** The test plan promises SBC for "NUTS and Gibbs (and Laplace …)" with Dirichlet, GD
  and `LogisticRandomWalk` priors. CONTRIBUTING says new samplers need SBC or Geweke tests.
- **Evidence:**
  - **Fixed hyperparameters.** The only smoothness-prior SBC is
    `LogisticRandomWalk(order=1, scale=0.8)`, with the scale *fixed*. Hyperparameter
    inference, which drives partial centring, the dense metric start and the field-conditional
    spread rule (`_CENTRING_MAX_SD = 0.28`), is never SBC'd in the suite.
  - **Missing configurations.** No SBC or coverage covers:
    - `LogisticGP`;
    - NUTS or Gibbs with selection, in either mode;
    - rate models;
    - `PolyaTree` or `DirichletTree`;
    - Laplace with importance resampling (promised in the plan).
  - **Not added after the previous review.** The earlier review ran several of these offline
    (`r19_sbc_hyper.py` and others in the review directory) but did not add them to the suite.
  - **Simulator limits.** `prior_predictive` rejects rate priors and has no `selection`
    argument, so the package's own tools cannot express these checks without custom
    simulators.
  - **Gibbs Geweke test.** It covers only `Dirichlet` with categorical emissions. Gibbs
    conditionals for GD, trees and `GammaRates` are checked only against NUTS on one data set
    each (fast suite, `sd_rtol=0.2`).
- **Suggested fix:**
  - Add `selection=`/`selection_mode=` and rate support to `prior_predictive`.
  - Add slow SBC cases:
    - `LogisticRandomWalk(order=2)` with an inferred scale;
    - `LogisticGP()` on about 8 bins;
    - Gibbs with selection, in both modes;
    - `GammaRates` with exposure;
    - `PolyaTree`;
    - Laplace (strict `xfail` if it is expected to fail at K = 4).
  - Assert `res.min_ess >= n_draws` so that autocorrelated thinning cannot masquerade as
    miscalibration.
- **Confidence:** confirmed.

### T-3. Bootstrap "interior start" regression test asserts nothing about the engine

- **Category:** weak-test
- **Location:** `tests/test_review_regressions.py:46-54`
- **Claim:** The docstring says "A sparse MLE must not trap replicates: each refit is certified
  optimal."
- **Evidence:**
  - The test computes its own `start = 0.5*MAP(m).run().value + 0.5/20` and asserts `start > 0`.
  - It also checks the output shape.
  - Neither touches what `Bootstrap` does. Reverting `bootstrap.py:88` to `init=base.value`
    would still pass.
  - The fix itself works. `bootstrap_kkt.py` spies on the replicate weights and computes each
    replicate's KKT gap on its weighted model. Over 60 replicates the median gap is 1.4e-9 and
    the maximum 7.5e-7, and none exceeds 1e-6. The review measured up to 9.5e-2 before the fix.
- **Suggested fix:** Assert the per-replicate KKT gap, as in `bootstrap_kkt.py`: patch
  `_replicate_weights` to record the weights, or expose the weights or gaps in `info`.
  Replicate ConvergenceWarnings are silently suppressed (`bootstrap.py:87`), so also record
  `n_unconverged` in `info` and test that it is 0.
- **Confidence:** confirmed.

### T-4. mix-SQP warns (an error under the test config) for Dirichlet priors mixing α = 1 and α > 1

- **Category:** test-gap (a code bug found through the docs)
- **Location:**
  - `src/pastry/samplers/optimize.py:90`;
  - `docs/user-guide/priors.md` (the `from_counts` snippet);
  - `tests/test_engines.py:332-356`.
- **Claim:** `Dirichlet.from_counts(m, strength)` is a documented, mix-SQP-eligible prior.
- **Evidence:**
  - **Cause.** `np.where(pos, a / (W * np.maximum(x, 1e-300) ** 2), 0.0)` evaluates the
    division for every k. Where `a_k = 0` and `x_k ≈ 0`, `1e-300**2` underflows to 0 and
    0/0 is evaluated.
  - **What users see.** Running `priors.md` prints
    `RuntimeWarning: invalid value encountered in divide`.
  - **Under warnings-as-errors.** `mixsqp_warning.py` turns warnings into errors (as
    `pyproject.toml` does for tests), and there `infer(..., method="map")` raises.
  - **Not tested.** The mix-SQP tests use only uniform α (1.0 or 2.5), so no test has some
    `a_k = 0` and some `a_k > 0`.
- **Suggested fix:** Compute only on `pos`, e.g. `H[pos, pos] += a[pos] / (W * x[pos]**2)`.
  Add a parametrised case `Dirichlet([1, 1, 3, 1, 2, …])` (or `from_counts` with zeros) to
  `test_mixsqp_matches_em_and_is_certified`.
- **Confidence:** confirmed.

### T-5. `check()` ignores E-BFMI (and the spec's KKT threshold), contrary to the docs

- **Category:** docs-wrong
- **Location:**
  - `docs/theory.md:406-410`;
  - `docs/user-guide/engines.md:52-54`;
  - `src/pastry/results.py:583-636`;
  - `docs/design/spec.md:463-469`.
- **Claim:**
  - Theory: "NUTS health. Divergences, saturated tree depth and E-BFMI. `result.check()`
    turns these into warnings."
  - engines.md: "… tree depth and E-BFMI. `pastry.infer(check=True)` … warns about any
    problem."
  - The spec's default thresholds include "KKT gap > 1e-6 relative".
- **Evidence:**
  - `diagnostics` copies `ebfmi` from `info`, but `check()` never reads it.
  - Nothing checks the KKT gap: `PointEstimate` has no `check()`, and MAP only warns on its
    own convergence flag.
  - No test sets a low E-BFMI or large KKT gap.
- **Suggested fix:** Add an E-BFMI < 0.3 message to `check()`, with a test, or drop E-BFMI from
  the two pages. Either implement the KKT threshold or remove it from spec §7.3.
- **Confidence:** confirmed.

### T-6. `predict_proba` and `log_predictive` are not chunked; performance.md says they are

- **Category:** docs-wrong
- **Location:**
  - `docs/user-guide/performance.md:69-71`;
  - `src/pastry/results.py:269-289, 291-314`.
- **Claim:** "The per-object methods (`assignment_probabilities`, `predict_proba`) work in
  chunks, so their memory use does not grow with N."
- **Evidence:**
  - **`predict_proba`.** It calls `L.toarray()` on the new objects' likelihoods, which
    densifies sparse input. It then allocates `acc = np.zeros_like(L)`. Memory is O(N·K),
    dense.
  - **`log_predictive`.** It builds an `(N_new, n_draws)` dense matrix. With the default 500
    draws that is 4 GB at N = 10⁶.
  - **`PopulationEstimator.score`.** It uses `log_predictive`.
  - Only `assignment_probabilities` uses `chunk_slices`.
- **Suggested fix:** Chunk both methods as `assignment_probabilities` does, or correct the
  sentence.
- **Confidence:** confirmed (code reading).

### T-7. engines.md contradicts itself and the code on which priors use EM

- **Category:** docs-wrong
- **Location:**
  - `docs/user-guide/engines.md:113-121`;
  - `src/pastry/samplers/optimize.py:191-212`.
- **Claim:**
  - One bullet: "Other conjugate priors (small concentrations, trees, `GammaRates`,
    informative priors with selection) use SQUAREM-accelerated EM."
  - The next bullet: "Priors whose density is unbounded … (concentrations below one, Pólya
    trees) … use L-BFGS."
- **Evidence:**
  - `_resolve_space` sends any prior with `unbounded_density()` to L-BFGS:
    - Dirichlet with α < 1;
    - Pólya trees;
    - GammaRates with a < 1.
  - `test_map_unconstrained_and_logistic` asserts `auto.info["space"] == "unconstrained"` for
    `Dirichlet(0.5)`.
  - So "small concentrations" and, for Pólya, "trees" belong in the L-BFGS bullet only.
- **Suggested fix:** Change the EM bullet to "generalised Dirichlet and Dirichlet trees with
  concentrations ≥ 1, `GammaRates` with shape ≥ 1, and Dirichlet priors other than the flat one
  under selection".
- **Confidence:** confirmed.

### T-8. No CI job executes the README or user-guide code; other promised docs and packaging checks are missing

- **Category:** ci
- **Location:**
  - `.github/workflows/ci.yml`;
  - `docs/_scripts/run_snippets.py`;
  - test plan §2.12.
- **Claim:** The test plan says:
  - "README example: a doctest-style test extracts the snippet and runs it";
  - "a smoke test installs the wheel in a clean virtualenv and imports it";
  - "numpydoc validation is run as a lint step".
- **Evidence:**
  - **Snippets.** User-guide pages use plain ```` ```python ```` fences, which myst-nb does not
    execute. `run_snippets.py` exists but nothing calls it.
  - **They do run today.** I ran `install.md` and all user-guide pages: every page was OK, in
    about 25 s in total. The README quick-start output reproduces digit for digit. They will
    drift silently from here.
  - **Build job.** It only runs `python -m build` and `twine check`; it never installs the
    wheel.
  - **numpydoc.** `docs/conf.py` sets `numpydoc_validation_checks = set()`, so validation is
    off.
- **Suggested fix:**
  - Add `python docs/_scripts/run_snippets.py docs/install.md docs/user-guide/*.md README.md`
    to the docs job (about 30 s). The script handles ```` ```python ```` fences, so README works
    as is.
  - Add `pip install dist/*.whl && python -c "import pastry"` from a temporary directory to the
    build job.
  - Turn on a small numpydoc check set (e.g. `{"PR01", "PR02", "RT01"}`) or drop the promise.
- **Confidence:** confirmed.

### T-9. Important code paths with no test (from the coverage run)

- **Category:** test-gap
- **Location:** `pytest -q --cov=pastry` gives 90% line+branch; full report in `cov.txt`.
- **Claim:** "Every public behaviour is covered by a fast, deterministic test" (test plan §1).
- **Evidence:** The misses that matter:
  - **`results.py:609-624`.** The `check()` messages and hints for divergences, Pareto k̂ and
    max-depth saturation, including the "small concentrations → use Gibbs" hint. The user
    guide and tutorials describe all of these. Only the R-hat path is tested.
  - **`samplers/vb.py:66-68`, `priors/rates.py:136-149`.** VB with selection, and VB for rate
    models: entirely untested. `vb_untested.py` shows the means are right (selection: VB
    0.4436/0.2926/0.2638 against exact 0.4443/0.2928/0.2629; rates within 1% of NUTS). A
    regression would go unseen.
  - **`samplers/optimize.py:370-374`.** The weighted L-BFGS path. Bootstrap uses it with
    non-conjugate priors or α < 1, and Bootstrap is never run with a logistic prior.
  - **`samplers/nuts.py:311-317`.** User-supplied `init`, both shapes; only the error is
    tested.
  - **`samplers/laplace.py:110-116`.** Clipping of a non-positive-definite Hessian, and its
    `ApproximationWarning`.
  - **`likelihoods.py:1063-1068`.** `sample_weights` for ragged sample lists. It is documented
    in inputs.md ("Ragged lists … and importance weights are supported").
  - **`likelihoods.py:94-98`.** Callable interim prior on categorical bins.
  - **`simulate.py:259-266`.** `mock_1d(interim_prior=callable | name | array)`.
  - **`selection.py`.** Length mismatch, mode conflict with a `Selection` object, and mode
    given without a selection.
- **Suggested fix:**
  - Add small fast tests for each:
    - synthetic results with `info={"n_divergent": 5}`, `{"k_hat": 0.9}` or
      `{"max_depth_fraction": 0.2}`, asserting the hint text;
    - VB against `exact_posterior` with selection (K = 3);
    - VB rates against Gibbs rates;
    - `Bootstrap(PopulationModel(lk, LogisticRandomWalk()))`, checking its shape;
    - `NUTS(init=…)`.
  - Consider `--cov-fail-under=88` in CI to stop regressions.
- **Confidence:** confirmed.

### T-10. Smoothness priors on N-D grids have no correctness test

- **Category:** test-gap
- **Location:**
  - `tests/test_priors.py:252-299, 382-388`;
  - `tests/test_model.py:52-80`.
- **Claim:**
  - "All of them work on 1-D bins and N-D grids. The random walk then uses the sum of per-axis
    difference penalties, and the GP takes one length scale per axis" (`priors.md`).
  - Test plan §2.4 also promises "LogisticNormal: sample moments of η match the given mean and
    covariance".
- **Evidence:**
  - **Gradient checks.** All finite-difference gradient checks for logistic priors use 1-D
    bins: 6 bins in `test_logistic_gradients`, `lk6` in `test_gradients_match_finite_differences`.
  - **2-D coverage.** `test_random_walk_2d_structure` only counts null modes. Nothing checks
    `_KronBasis` for D = 2, the D-dimensional spectral density, or ARD
    `length_scale_0/length_scale_1` (log-prior or gradient). Only tutorial 08 exercises the 2-D
    GP, and it asserts nothing.
  - **`LogisticNormal`.** `test_logistic_sampling_and_init` checks only shapes and sums.
- **Suggested fix:**
  - Parametrise `test_logistic_gradients` and `test_gradients_match_finite_differences` over a
    `Bins.grid(5, 4)` for `LogisticRandomWalk(order=1, 2)` and `LogisticGP()`.
  - Compare the 2-D GP covariance against the exact product kernel, as
    `test_gp_matches_exact_kernel` does in 1-D.
  - Add an η-moment test for `LogisticNormal(mean, cov)`.
- **Confidence:** confirmed.

### T-11. Assertions that are too loose or do not match their comments

- **Category:** weak-test
- **Location:** as listed.
- **Claim:** The tests document the behaviours named in their comments and docstrings.
- **Evidence:**
  - **`test_slow.py:232-234`.** The comment says "but its mean is in the right place on
    well-constrained bins". The assertion is `model.missing_information(n.mean) > 0.9`, which
    says nothing about VB's mean.
  - **`test_simulate_validation.py:51-55`.** `test_mock_selection_and_outliers` asserts only
    `n_drawn > 400` and the mode. It checks neither the thinning rate per bin (plan: "Selection
    mocks thin as expected") nor the outlier fraction.
  - **`test_results.py:89-92`.** `sample_values` is checked against the *outer* grid edges,
    not the drawn bin. A value placed in the wrong bin passes.
  - **`test_diagnostics.py:18-30`.** AR(1) ESS and τ use `rel=0.3` and `rel=0.25` on 80,000
    draws; the plan says "within 10%". At φ = 0 and 0.5 the estimator is accurate to a few
    percent, so a 20–25% ESS bias (e.g. in Geyer truncation) would pass. There is no reference
    value (ArviZ or posterior::ess_bulk) for any fixed array.
  - **`test_engines.py:468-478`.** `test_bootstrap` accepts any sd between 0.3× and 2.5× the
    posterior sd. Finding 1 of the previous review (10–30× too wide) would have passed this
    test only marginally.
  - **`test_engines.py:173-185`.** `test_nuts_logistic_prior_runs` checks shapes only.
  - **`test_slow.py:150-158`.** `test_gibbs_and_nuts_agree_on_larger_problem` assumes NUTS ESS is
    half the draws instead of using `diag.mcse`. The `std` agreement is `rtol=0.25`.
  - **`test_priors.py:418`.** `assert special.gammaln(1.0) == 0.0  # keep scipy.special import
    used` is dead code; drop the import instead.
- **Suggested fix:** Tighten each to the stated behaviour:
  - VB: `np.abs(v.mean - n.mean) < 3 * n.std` on bins with small missing information.
  - Mock selection: per-bin selected fraction within binomial error of S.
  - `sample_values`: `bins.index(vals) == labels`; this needs a shared `rng`, or
    `sample_values` could return labels too.
  - Diagnostics: 10% tolerances for φ ≤ 0.5, plus one golden-value test against ArviZ numbers
    stored in the test.
  - Bootstrap: compare against a large-N posterior sd at about 30%.
- **Confidence:** confirmed.

### T-12. The test plan and spec are stale against the suite and the code

- **Category:** consistency
- **Location:**
  - `docs/design/test-plan.md` (§1, §2.5-2.12);
  - `docs/design/spec.md:251, 440, 469`.
- **Claim:** The test plan is the published mapping from requirements to tests (rendered in the
  docs under "Design notes").
- **Evidence:**
  - **Missing files.** The plan names files that do not exist: `test_engine_nuts.py`,
    `test_engine_gibbs.py`, `test_engine_map.py`, `test_engine_laplace.py`,
    `test_engine_vb.py`, `test_engine_bootstrap.py`, `test_infer_api.py`, `test_estimator.py`,
    `test_simulate.py` and `test_validation.py`. The tests live in `test_engines.py`,
    `test_api.py` and `test_simulate_validation.py`.
  - **Promised tests not implemented:**
    - NUTS adaptation on a pure Gaussian target (step size, metric → true variances);
    - funnel divergence reporting;
    - "analytic Hessian matches FD" (Laplace uses an FD Hessian);
    - `progress=True` prints nothing without tqdm;
    - SBC for Laplace;
    - cross-engine at K = 30, N = 2×10⁴ (the suite uses K = 15, N = 5000);
    - recovery "with outliers";
    - README test, wheel smoke test and numpydoc lint (T-8).
  - **Stale warning flag.** §1 says CI uses `-W error::pastry.PastryWarning`; it uses the
    pyproject `filterwarnings = error`.
  - **Seed sweeping.** §1 says statistical tests are seed-robust and checked with
    `PASTRY_TEST_SEEDS`. Only about 34 of 181 test functions take the `seed` fixture. Every
    engine-vs-exact comparison hard-codes its seeds, so the sweep never touches them.

    I swept those by hand (`seed_sweep.py`): NUTS, Gibbs and Laplace against the K = 3 exact
    posterior pass for all 12 seeds tried. The tolerances are sound, but the documented
    mechanism does not cover them.
  - **Stale spec.**
    - `spec.md:251` gives `from_posteriors(…, interim_representation="mass")`; the code
      defaults to `None`, which follows `representation`, since review finding 10.
    - `spec.md:440` says `sample_values` is "uniform by default"; it is exact for Gaussian
      likelihoods.
    - `spec.md:469` lists the KKT threshold (T-5).
- **Suggested fix:**
  - Update the plan's file names and mark unimplemented items as deferred, or implement the
    cheap ones (Gaussian-target NUTS, funnel).
  - Make the engine tests take `seed` so that `PASTRY_TEST_SEEDS` reaches them.
  - Refresh the spec rows.
- **Confidence:** confirmed.

### T-13. Test-suite timing claims disagree; three tests take 40% of the fast suite

- **Category:** simplify
- **Location:**
  - `CONTRIBUTING.md:20-21`;
  - `CLAUDE.md`;
  - test plan §1;
  - `tests/test_engines.py:381-393`;
  - `tests/test_review_regressions.py:220-228`.
- **Claim:**
  - CONTRIBUTING: fast suite "~40 s", slow suite "~10 min".
  - CLAUDE.md: about 80 s and about 7 min.
  - Test plan: "target is under 2 minutes on one core".
- **Evidence:**
  - **Measured on this shared, loaded 4-core box:**
    - fast suite 160 s with coverage;
    - 164 s with `--durations`;
    - 143 s on Python 3.11 with the minimum dependencies, 1 BLAS thread.
  - **Slowest tests:**

    | test | time |
    |---|---|
    | `test_mixsqp_with_selection_solves_for_observed_fractions[true]` | 30.0 s |
    | `test_mixsqp_with_selection_solves_for_observed_fractions[observed]` | 18.8 s |
    | `test_weak_data_smoothness_fits_stay_non_centred` | 18.7 s |
    | everything else | under 7 s each |

  - **Why the mix-SQP tests are slow.** The cost is the EM reference with `max_iter=200000,
    tol=1e-14` on N = 3000, K = 20, which crawls.
  - **Why the centring test is slow.** It runs default NUTS (4 chains × 1000 warm-up), but the
    centring decision is made at warm-up iteration 25.
  - Slow-suite timing: see the end of this report.
- **Suggested fix:**
  - **mix-SQP tests.** Replace the EM reference with a certificate: the KKT gap plus "one EM
    step leaves `sqp.rho` unchanged to 1e-8". Or use EM warm-started from `sqp.rho`. Either
    keeps the check and saves about 45 s.
  - **Centring test.** Use `n_warmup=60, n_chains=2` and `run(1)`, as
    `test_data_rich_gp_fits_are_partially_centred` does.
  - **Timings.** State one measured figure in one place (CONTRIBUTING) and point to it.
- **Confidence:** confirmed for the durations. Absolute times are inflated by load from
  parallel auditors (load average about 8 on 4 cores).

### T-14. CI details: unpinned ruff, no timeouts, matrix claims, slow suite never on PRs

- **Category:** ci
- **Location:** `.github/workflows/ci.yml`; `.pre-commit-config.yaml`; `CONTRIBUTING.md:80-82`.
- **Claim:** CONTRIBUTING: "CI runs … the fast suite on Linux, macOS and Windows (Python
  3.11–3.14)".
- **Evidence:**
  - **ruff.** The lint job installs `ruff>=0.6` (latest), while pre-commit pins `v0.17.1`. A
    new ruff release can change `ruff format --check` output and fail every PR, and the
    `needs: lint` dependency then also blocks all tests.
  - **Timeouts.** The test jobs have no `timeout-minutes`. A hung process pool (parallel
    NUTS on macOS or Windows `spawn`) would burn the 6-hour default.
  - **Matrix.** macOS and Windows run only 3.13.
  - **Minimum dependencies.** The job pins numpy, scipy and matplotlib but not
    `threadpoolctl==3.1.*`. I checked: 3.1.0 works with the suite.
  - **Slow suite.** It runs only on `schedule` (default branch) and `workflow_dispatch`. PRs
    that change samplers or priors never run it, and pushes to `claude/pastry-v0.1` trigger
    CI only through a PR.
  - **API compatibility checked (fine).** The minimum-dependency job is viable: on Python 3.11
    with numpy 1.25.2, scipy 1.11.4, matplotlib 3.7.5 and threadpoolctl 3.1.0, the fast suite
    gives "267 passed, 1 skipped" (`mindeps.txt`). APIs I checked as present at the minimum
    versions:
    - `sp.diags_array`;
    - `Generator.spawn`;
    - `optimize.nnls(maxiter=)`;
    - sparse-array row indexing.
- **Suggested fix:**
  - Pin ruff in CI to the pre-commit version, or run `pre-commit run --all-files`.
  - Add `timeout-minutes: 30` to the test jobs.
  - Add `threadpoolctl==3.1.*` to the minimum-dependency pins.
  - Run the slow suite on PRs with a label (`if: contains(github.event.pull_request.labels.*.name, 'slow')`).
  - Correct the CONTRIBUTING sentence.
- **Confidence:** confirmed.

### T-15. README links and image break on PyPI; no docs URL; version duplicated by hand

- **Category:** packaging
- **Location:**
  - `README.md:12, 148, 161, 174, 181`;
  - `pyproject.toml` (`[project.urls]`);
  - `CITATION.cff:16`.
- **Claim:** README is the PyPI long description (`readme = "README.md"`).
- **Evidence:**
  - **Relative paths.** PyPI does not resolve them, so these break on the project page and
    `twine check` does not catch it:
    - `docs/_static/quickstart.png`;
    - `docs/`;
    - `reports/`;
    - `CITATION.cff`;
    - `LICENSE`.
  - **Docs URL.** `[project.urls]` has no Documentation entry.
  - **Version.** It appears in `_version.py`, `CITATION.cff` (`0.1.0.dev0`) and README "Status"
    (`0.1.0.dev0`), with no check that they agree.
- **Suggested fix:**
  - Use absolute `https://raw.githubusercontent.com/joshspeagle/pastry/main/...` and
    `https://github.com/...` links in README.
  - Add `Documentation =` once the docs are hosted.
  - Add a tiny test that `CITATION.cff`'s `version` equals `pastry.__version__`, or drop the
    version from README prose.
- **Confidence:** confirmed (PyPI behaviour is standard).

### T-16. install.md describes the extras incorrectly

- **Category:** docs-wrong
- **Location:** `docs/install.md:27, 29`
- **Claim:**
  - "`test` | pytest and coverage".
  - "`dev` | all of the above plus ruff and pre-commit".
- **Evidence:**
  - `dev = ["pastry-inference[test,docs,plot]", "ruff>=0.6", "pre-commit>=3"]` does not
    include `xarray`, so `to_datatree` tests are skipped in a dev install.
  - `test` also pulls in matplotlib.
- **Suggested fix:** Add `xarray` to `dev`, which matches "all of the above" and lets dev
  installs run every test, or say "test, docs and plot".
- **Confidence:** confirmed.

### T-17. Theory: the Gamma-rates/Dirichlet equivalence needs a common rate; S ∈ (0, 1] vs S > 1

- **Category:** docs-wrong (math, minor)
- **Location:** `docs/theory.md:245-246, 195`
- **Claim:**
  - "independent Gamma rates imply a Dirichlet on ρ only without selection".
  - "Let S_k ∈ (0, 1] be the probability …".
- **Evidence:**
  - **Gamma rates.** Independent Gamma(a_k, b_k) rates give Dir(a) fractions only when every
    b_k equals a common b. With distinct b_k the fractions are not Dirichlet even without
    selection. `GammaRates(shape, rate)` accepts per-bin rates.
  - **Selection values.** `selection.md` and the code accept S_k > 1 for fractions. Only ratios
    matter, and the augmentation normalises them. The theory page states S as a probability
    with no note.
- **Suggested fix:**
  - "… imply a Dirichlet(a) on ρ only for a common rate b and without selection".
  - Add one sentence: for fractions S is needed only up to a constant.
- **Confidence:** confirmed.

### T-18. `which=` vocabulary differs between plotting and results; PointEstimate plots fractions for `which="rate"`

- **Category:** consistency
- **Location:**
  - `src/pastry/plotting.py:35-49`;
  - `src/pastry/results.py:440-449`;
  - `docs/user-guide/results.md` ("`which="fraction"` for fractions").
- **Claim:** One `which=` argument across `plot`, `summary`, `quantile` and `interval`.
- **Evidence:**
  - `plot` accepts `"density"|"fraction"|"rate"`; `summary`, `quantile` and `interval` accept
    `"fractions"|"rates"|"density"`. `result.plot(which="fractions")` raises.
  - For a `PointEstimate`, `_draws` returns `rho` for any `which` other than `"density"`, so
    `which="rate"` silently plots fractions.
  - No test covers either.
- **Suggested fix:** Accept both spellings everywhere, normalising in one helper. Make
  `PointEstimate` honour `"rate"`. Add a test.
- **Confidence:** confirmed (code reading).

### T-19. Small docs inconsistencies and unasserted prose numbers in tutorials

- **Category:** consistency
- **Location:**
  - `docs/tutorials/03_engines.md:105`;
  - `docs/tutorials/06_categories_unfolding.md:116`;
  - `docs/user-guide/results.md:131`;
  - `src/pastry/inference.py:68, 114-123`.
- **Evidence:**
  - **Gibbs autocorrelation time.** 03 says it grows "like 1/(1−f)". engines.md, `check()` and
    `model.missing_information` use (1+f)/(1−f). Same scaling, different formula.
  - **Prose numbers.** Tutorial prose states numbers that only the executed output determines,
    e.g. "One transition out of 4000 diverged" (06:116) and "near one here" (03).
    - The cached build (13:31, after the last sampler change) agrees today.
    - No hidden assert pins them, so a sampler change can make the prose false while the
      docs build stays green.
  - **`pastry.plotting`.** `import pastry` does not import it, so the reference
    `pastry.plotting.plot_individual` fails as attribute access unless something imported the
    submodule first. `results.md:131` links it.
  - **`infer` details:**
    - the docstring's method list omits `"em"`;
    - the error for `method="em", n_samples=…` says "method='map'";
    - `method="AUTO"` works but does not record `info["auto_method"]`, because line 122
      compares the raw `method`, not `key`.
- **Suggested fix:**
  - Use (1+f)/(1−f) everywhere.
  - Add `assert` cells tagged `remove-cell` beside numbers quoted in prose.
  - Import `plotting` lazily via a module `__getattr__`, or add `PopulationResult.plot_individual`.
  - Fix the `infer` nits.
- **Confidence:** confirmed.

### T-20. Dead configuration and unchecked diagnostics fields

- **Category:** simplify
- **Location:**
  - `pyproject.toml:[tool.ruff.lint] ignore`;
  - `src/pastry/validation.py:130, 220-221`.
- **Evidence:**
  - `ignore` lists `N803` and `N806`, but `N` (pep8-naming) is not selected.
  - `SBCResult.min_ess` is computed but never used, neither by `passed()` nor by the tests.
- **Suggested fix:**
  - Drop the two ignores.
  - Have `passed()` (or the slow tests) warn or fail when `min_ess < n_draws`, since ranks
    from autocorrelated draws are not uniform even for a correct sampler.
- **Confidence:** confirmed.

## Not verified: performance numbers in the docs

The docs make these timing claims:
- `performance.md`: NUTS on 2000 × 30 takes "about 30 s on one core"; kernel timings of 1.2,
  1.6, 9 and 10 ms; MAP on 10⁵ × 100 takes "about a second".
- `README.md`: "seconds for 10⁵ objects".

I measured them on a box shared with other auditors (load average about 8 on 4 cores):

| claim | documented | measured |
|---|---|---|
| NUTS, 2000 × 30 | about 30 s | 54 s |
| kernels | 1.2, 1.6, 9, 10 ms | 1.3, 3.3, 14, 25 ms |
| MAP, narrow likelihoods | about a second | 6.5 s |
| MAP, broad likelihoods | a few seconds | 4.7 s |

These are consistent with the claims under a 2–3× load penalty, so I cannot call any of them
wrong. One thing is odd: the docs say broad likelihoods are slower for MAP ("a few seconds when
likelihoods are broad"), but the narrow case was slower here. Worth re-timing on an idle
machine.

## Checked and found correct

- **README.** The quick-start reproduces exactly: the first five summary rows match to every
  printed digit (seed 42/1, 4000 draws).
- **Docs snippets run.**
  - Every block in `install.md` and the nine user-guide pages runs without error under
    `run_snippets.py`. The only stray output is the warning in T-4.
  - Parameter names used in the docs exist with the documented defaults:
    - `representation` defaults: `"density"` for `Likelihoods`, `"mass"` for
      `from_posteriors`;
    - `interim_representation` follows `representation`;
    - `min_samples=20`;
    - the `n_jobs`, `target_accept` and `store_labels` options;
    - `Dirichlet.reference` and `from_counts` (α = 1 + λm);
    - `from_base_measure`;
    - `GammaRates(shape, rate)`;
    - the GP kernels `matern12/32/52` and `rbf`;
    - `boundary_factor`;
    - the length-scale prior: 1% below two bin widths, 1% above the range;
    - NUTS defaults: 4 chains, 1000 warm-up, `target_accept` 0.8 or 0.9, dense metric up to
      200 dimensions, curvature start up to 500;
    - Pareto k̂ threshold 0.7;
    - `AUTO_NNZ_LIMIT = 2e7`.
- **theory.md maths re-derived and correct:**
  - the marginal likelihood L_i·ρ;
  - the augmented form;
  - the responsibilities;
  - concavity;
  - EM as iterated stacking, and stacking as one EM step from π;
  - the KKT conditions;
  - the interim-prior identity Σ w log(P_i·ρ) = ℓ(π⊙ρ/π·ρ) + W log(π·ρ) + const, and the
    collapse onto argmax Π;
  - the observed-fraction change of variables for mix-SQP, in both modes;
  - the selection likelihood (eq-S) with the 1/Λ prior;
  - Λ | ρ ~ Gamma(W, S·ρ) and M | ρ ~ NegBin(W, S·ρ), which matches numpy's parametrisation;
  - the expected missed counts Wρ_k(1−S_k)/(S·ρ);
  - the Poisson-process rates likelihood and its Gamma conditional;
  - the ILR Jacobian (Σ log ρ + ½ log K) and the "α in place of α−1" Laplace centre;
  - the logit gradients n̂ − Wρ, n̂ − Wρ⊙S/(S·ρ) and n̂ − S⊙λ;
  - the partial non-centring algebra (variance preserved; λ = 1/(1+s²F) = 1/(−H_jj));
  - the GP sine basis and frequency πj/2L, which matches `logistic.py:504-510`.
- **References.** Bibliographic details in theory.md, citing.md and CITATION.cff check out:
  - journal, volume and page for all the papers;
  - Kim et al. 2020 JCGS 29:261;
  - Vehtari et al. 2024 JMLR 25(72);
  - Stanley, Patil & Kuusela 2022 JINST 17 P10013.
- **Engine-vs-exact tests are seed-robust.** All 12 extra seeds pass for NUTS, Gibbs and
  Laplace against the K = 3 quadrature with the shipped tolerances.
- **Geweke test.** It has a real negative control (the broken kernel gives |z| > 8), and
  batch-means standard errors are the right choice for the autocorrelated chain.
- **The previous review's fixes hold in practice.**
  - Bootstrap replicates are now certified optima (T-3 is about the test, not the code).
  - VB selection and rate paths give correct means (T-9).
- **Minimum dependencies.** Python 3.11, numpy 1.25, scipy 1.11, matplotlib 3.7 and
  threadpoolctl 3.1 are all viable; no newer-than-minimum APIs are used.
- **Packaging metadata.**
  - `pyproject.toml`: hatchling ≥ 1.27 for PEP 639 `license`/`license-files`; the sdist
    includes tests and CITATION; `xarray>=2024.10` matches `DataTree`'s arrival.
  - `.gitignore` excludes `docs/_build`, `jupyter_execute` and `.jupyter_cache`.
  - No tracked file exceeds the 1000 kB pre-commit limit.
- **Benchmarks.** `benchmarks/bench_kernels.py` runs and covers the operations listed in test
  plan §2.13.

## Slow suite (run once)

`pytest -m slow` (OMP_NUM_THREADS=1, loaded machine): **14 passed, 1 xfailed** (the VB
SBC, as intended) in **645 s (10.8 min)**.

Slowest:

| test | time |
|---|---|
| `test_sbc[nuts-dirichlet]` | 207 s |
| `test_sbc[nuts-randomwalk]` | 168 s |
| `test_vb_underestimates_widths_vs_nuts` | 54 s |
| `test_sbc[gibbs-gdirichlet]` | 44 s |
| `test_gibbs_and_nuts_agree_on_larger_problem` | 36 s |
| `test_coverage_gibbs` | 36 s |
| `test_sbc[gibbs-dirichlet]` | 24 s |

Simplification opportunities:
- **`test_vb_underestimates_widths_vs_nuts` (54 s).** It mostly runs NUTS on 3000 × 15 to show
  that VB is too narrow. The fast suite already shows this against the exact posterior
  (`test_vb_exact_for_onehot_and_overconfident_otherwise`), and the VB SBC `xfail` shows it
  again. Drop the test, or reuse the NUTS fit from `test_gibbs_and_nuts_agree_on_larger_problem`
  through a module fixture.
- **The NUTS SBC cases (375 s together).** They dominate the run. Spending that time on more
  Gibbs simulations, plus the missing configurations of T-2, would buy far more power per
  minute than the current split.
