# binference — test plan (v0.1)

This plan turns the requirements in [`spec.md`](spec.md) into tests. It has three goals:

1. Every public behaviour is covered by a fast, deterministic test.
2. Every inference engine is shown to target the right posterior, using exact references, invariances, cross-engine agreement and simulation-based calibration.
3. The docs and tutorials stay executable.

The plan was written before the code. It was brought up to date with the suite on 2026-10-26 (issue #79, audit T-12): file names are those in `tests/`, and items that were planned but have no test are marked **Not implemented**.

## 1. Principles

- **Deterministic.** Every random test uses a fixed seed passed through `rng`. Tests that rely on statistical tolerances use a seed that passes, plus tolerances wide enough that the false-failure probability is small for any seed. `BINFERENCE_TEST_SEEDS=0,1,2 pytest` loops over seeds in the tests that take the `seed` or `rng` fixture (`tests/conftest.py`), in the same spirit as dynesty. Most engine-against-exact tests fix their own seeds, so the sweep does not reach them; their tolerances were checked by hand over 12 seeds (audit T-12).
- **Tolerances scale with Monte Carlo error.** Compare against references with

  $$|\hat\theta - \theta_\text{ref}| \le z\cdot \mathrm{MCSE}(\hat\theta) + \epsilon_\text{abs}, \qquad z = 4 \text{ to } 5.$$

  Standard deviations use a relative tolerance derived from the effective sample size (ESS); `tests/conftest.py` has `sd_mcse` for it.
- **Two tiers.**
  - Default `pytest` runs fast tests: unit, exactness on small problems, and API tests (`pyproject.toml` adds `-m "not slow"`). They take about 90 s on one core.
  - `pytest -m slow` runs the statistical validation: SBC, coverage, larger cross-engine comparisons, the Geweke tests and the benchmark-matrix references (§2.11). It takes about 20 minutes on one thread. CI runs it weekly, by hand, or on a pull request labelled `slow`.
- **Warnings are errors.** `pyproject.toml` sets `filterwarnings = ["error", ...]`, so every warning fails a test unless the test expects it with `pytest.warns` or filters it.
- **References.** Module docstrings cite the spec sections, requirement IDs (`R-…`), decisions (`DECISIONS #n` in `notes/review/DECISIONS.md`) and audit findings their tests cover. Test names are sentences of behaviour.

## 2. Test inventory

§2.1–2.13 follow the plan's original structure; §2.14 lists the files added with later decisions.

### 2.1 `tests/test_bins.py` (R-BINS-*)

**Constructors and geometry**
- `Bins(edges)`: edges are validated (monotone, finite, ≥ 2 values). Check `K`, `centers`, `volumes` and `ndim == 1`.
- `Bins.linear`, `Bins.log` and `Bins.from_centers`: spacing and endpoint values.
- `Bins.grid` (2-D and 3-D):
  - C-order flattening;
  - `shape`;
  - `volumes` equal the outer product of the widths;
  - `unflatten` and `flatten` round-trip;
  - `centers` has shape `(K, ndim)`.
- `Bins.categories(["a", "b", "c"])` and `Bins.categories(4)`: `ordered` is False, volumes are 1, labels are set.

**Lookup and structure**
- `index` / `contains`: values on edges are assigned left-closed, the last edge is right-closed, and out-of-range values map to −1. N-D works.
- `neighbors()`: correct adjacency for 1-D chains and 2-D 4-neighbour grids. Categories raise.
- `coarsen(factor)`: the map between the bins is right, volumes add up, and a factor that does not divide the axis raises.
- Equality and repr; immutability (attribute assignment raises); pickling.

Measures and masks are in `tests/test_measures_masks.py` (§2.14).

### 2.2 `tests/test_likelihoods.py` (R-INPUT-*)

**Constructor**
- Basic construction from a dense array, a sparse CSR matrix and a list of lists.
- `log_scale` is recorded, the row max is 1, and `values * exp(log_scale)` reproduces the input.
- `from_log`: agrees with exponentiating directly, and stays stable when log values are around −1e4.

**Validation**
- Negative values, NaN and inf raise.
- All-zero rows raise. With `drop_empty=True` they are dropped and an `InputWarning` is emitted.
- A weights length mismatch raises, and non-positive weights raise. (More validation and storage tests are in `tests/test_hygiene.py`.)

**Representations**
- "mass" divided by the volumes equals "density" for a known Gaussian on unequal bins.
- **Unequal-bin regression test:** a known population with unequal bins is recovered by the MAP when the right representation is used. The wrong representation gives a bias above a threshold, which documents why the argument exists.

**`from_table`** (which replaced `from_posteriors`, DECISIONS #32)
- A table must say whether it holds likelihoods (`kind="likelihood"`) or posteriors (`interim=`); omitting both raises (R-INPUT-1).
- `"flat"` is accepted.
- **Interim-prior removal:** simulate likelihoods $\ell$, form posteriors $p \propto \ell\pi$, then convert back with $\pi$. The result is proportional to $\ell$ row by row.
- A zero interim prior where a posterior is positive raises.
- Representation sanity warnings fire on unequal bins.

**`from_samples`** (more in `tests/test_samples.py`)
- A histogram of known samples gives the expected masses divided by the interim masses.
- Out-of-range samples and per-object effective sample counts are reported in `meta`.
- Flat and non-flat interim priors; N-D samples.

**Other constructors**
- `from_gaussian`:
  - 1-D values match the analytic CDF differences divided by the volume;
  - far tails stay finite;
  - N-D with diagonal σ equals the product of 1-D terms.
- `from_density` (which replaced `from_distributions` and `from_function`): agrees with `from_gaussian` for SciPy `norm` objects, and its quadrature converges to `from_gaussian`. **Not implemented:** a skewed distribution such as `skewnorm`.
- `from_binned`: rows equal the response rows, weights equal the counts, and zero-count rows are dropped. The efficiency comes back as an `"on_measurement"` selection. `Response` and the classifier routes are tested in `tests/test_response.py`.
- Route A (`from_table` on categories with the training prior): $p/\pi_\text{train}$ is proportional to the row values.

**Methods**
- `compress()`: duplicate rows merge into summed weights, and the posterior is identical.
- `subset` and `lk[idx]` keep the weights.
- `stack()` is one EM step from the interim prior.
- A selection array without a mode raises.

### 2.3 `tests/test_simplex.py` (internal helpers)

- The Helmert basis is orthonormal with columns summing to zero.
- Round trips:
  - ILR to softmax and back is the identity (1e-12);
  - softmax is stable for inputs around ±1e3.
- The log-Jacobian of the ILR map matches a finite-difference Jacobian determinant to a constant, for K = 2..6.

### 2.4 `tests/test_priors.py` (§5)

More prior tests, from the audit, are in `tests/test_prior_fixes.py` (§2.14).

**Every prior, parametrised over the families**
- The density integrates to 1 on the 2-simplex (K = 3 quadrature), and sample moments match.
- The unconstrained log-density and its gradient agree with finite differences at random points.
- `prior_transform(u)`: uniform `u` gives draws whose moments match `sample`.

**Conjugate priors**
- `posterior_sample(counts)` and `map_from_counts` equal the brute-force posterior and its mode.
- The VB expected logs are correct.

**Family-specific checks**
- Generalised Dirichlet:
  - `GeneralizedDirichlet(a, b)` with b_k = Σ_{j>k} a_j reduces to `Dirichlet(a)` in its density;
  - it can produce positive correlations (Monte Carlo check).
- `DirichletTree`: a tree that satisfies the consistency condition collapses to a Dirichlet; invalid trees raise.
- `PolyaTree`: the prior mean equals the base measure, and the variance decreases as `c` grows.
- `Dirichlet.reference`, `from_counts` and `from_base_measure` give the documented α values; draws with small concentrations have no exact zeros.

**Logistic family**
- `LogisticNormal`: sample moments of η match the given mean and covariance.
- `LogisticRandomWalk`:
  - prior draws have the expected roughness;
  - the 2-D neighbour structure is correct;
  - categorical bins raise.
- `LogisticGP` (Hilbert-space approximation): the covariance matches the exact kernel, in 1-D and for 2-D ARD kernels; the RBF spectral density is correct; the default length-scale prior tracks the grid and the basis.
- Gradients with respect to the field and the hyperparameters match finite differences, centred and non-centred; partial centring is a reparametrisation; scale underflow is survived.
- Hyperpriors (`HalfNormal`, `LogNormal`, `InverseGamma`, `Gamma`, `Fixed`): log-density and gradient checks.

**Rates**
- `GammaRates`: moments and conjugate update.

### 2.5 `tests/test_model.py` (§3, §6.1)

More model tests, from the audit, are in `tests/test_model_fixes.py` (§2.14).

**Likelihood**
- `log_likelihood(rho)` equals the direct formula, for dense and sparse inputs and with and without weights.
- Invariances:
  - multiplying row i by $c_i$ leaves the posterior unchanged;
  - permuting objects leaves it unchanged;
  - duplicating a row is the same as weight 2.

**Gradients and transforms**
- The gradient of `log_density_unconstrained` agrees with finite differences for every prior family × data × selection mode (parametrised).
- `to_value` / `from_value` round-trip.
- `responsibilities` rows sum to 1, and `expected_counts` equals responsibilities summed with weights.

**Selection and diagnostics**
- The two selection modes give the likelihoods of spec §3.6, and a selection attached to the likelihoods is used.
- `kkt_gap` is zero at the MLE and positive away from it; `missing_information` behaves at its limits.
- `stack()` is one EM step.

### 2.6 Engine tests: exactness on small problems (fast), `tests/test_engines.py`

**Fixtures** (`tests/conftest.py`)
- `k3_problem` and `k3_selection_problem`: K = 3, N = 250, heteroscedastic Gaussian likelihoods, with their exact posteriors from `binference.validation.exact_posterior`.
- One-hot likelihoods, which give the conjugate posterior Dir(α + counts) exactly.

**NUTS**
- Matches the K = 3 exact posterior in mean, s.d. and quantiles within the MCSE tolerances.
- One-hot likelihoods give Dir(α + n) moments, with diagonal and dense metrics.
- The same seed gives identical draws; resuming is bitwise identical to one long run; the generator interface yields the expected shapes; parallel chains match serial ones.
- Logistic priors are reparametrised without touching the model, and match a long run.
- Arguments are validated; a given `init` is jittered only when it is one point.
- **Not implemented:** adaptation on a pure Gaussian target (step size, metric → true variances), and divergence reporting on a funnel target.

**Gibbs**
- Matches the K = 3 exact posterior, with and without selection; one-hot likelihoods give the exact conjugate posterior.
- With a selection on fractions, only the ratios of $S$ matter.
- `GammaRates` with selection, and the tree priors, agree with NUTS.
- Stored labels have shape `(n_draws, n_objects)`; non-integer weights raise; compressed integer weights and dense and sparse label sampling agree.
- A bin without likelihood support is not frozen (also for NUTS).

**MAP**
- The MLE matches `scipy.optimize` with a softmax parametrisation.
- SQUAREM uses fewer iterations than plain EM on a broad-PDF problem.
- mix-SQP matches EM and is certified, dense and sparse; it handles tall dense inputs and falls back; it stops at the round-off floor; with a selection it solves for the observed fractions.
- The fixed point equals iterated stacking.
- MAP with a prior and selection, tree priors on a grid, and logistic priors with L-BFGS match references.

**Laplace**
- The importance-corrected result matches the K = 3 exact posterior and reports $\hat k$.
- The uncorrected result is flagged approximate.
- Negative curvature is clipped to a wide positive variance; the finite-difference Hessian helpers are checked on a known quadratic.
- **Not implemented:** an analytic Hessian (Laplace uses finite differences of the exact gradient, spec §6.5).

**VB**
- One-hot likelihoods give exactly Dir(α + n); on broad PDFs the s.d. is smaller than the exact one, flagged `approximate=True` with an `ApproximationWarning`.
- With a selection the means are close to the exact posterior; rates with a selection match Gibbs.

**Bootstrap**
- Shapes, determinism and the flagging of the results; both schemes.
- At N = 2000 the bootstrap spread is within 20% of the posterior s.d.
- A logistic prior refits each reweighted catalogue.

The NUTS and Laplace audit fixes are in `tests/test_nuts_fixes.py`, the MAP fixes in `tests/test_map_fixes.py` and the Gibbs, bootstrap and validation fixes in `tests/test_gibbs_fixes.py` (§2.14).

### 2.7 `tests/test_results.py` (§7)

More result tests are in `tests/test_results_audit.py`, `tests/test_results_outputs.py`, `tests/test_best_loo.py`, `tests/test_values.py` and `tests/test_derived_rates.py` (§2.14).

**Summaries and intervals**
- `mean`, `std`, `quantile`, `interval` (central and HDI) and `cov` agree with NumPy on known arrays.
- `summary()` repr, `to_dict`, and categorical labels.
- `density()` equals ρ / volumes.

**Derived quantities**
- `functional`, and `rebin`, which is exactly the sum over groups.

**Individual objects**
- `assignment_probabilities` matches a Monte Carlo average of the responsibilities.
- `sample_assignments` shapes, and the empirical frequencies converge to the assignment probabilities.
- `realized_counts` sums to W in every draw, and is narrower than the population for precise data.
- `sample_values` stays within the bin edges.

**New objects**
- `predict_proba` and `log_predictive` on new objects; `log_predictive` beats a deliberately wrong population on held-out data.

**Diagnostics**
- `check()` warns on synthetic bad chains (stuck chains → R̂ warning). E-BFMI and the other thresholds are in `tests/test_results_audit.py`.

**Point estimates, external draws and rates**
- `PointEstimate` outputs and `laplace()`; `from_samples` and `attach`; rate results; a result without a model keeps its bins.

**Persistence and plots** (`tests/test_plotting_io.py`)
- `save` / `load` round-trip, including the metadata, the schema version and the bins.
- Plot smoke tests on the Agg backend; `to_datatree` is skipped when xarray is not installed.

### 2.8 `tests/test_diagnostics.py` (R-DIAG-1)

- On an AR(1) process with known φ:
  - the integrated autocorrelation time ≈ (1+φ)/(1−φ);
  - ESS ≈ n / τ.
- ESS, R̂ and MCSE match ArviZ on a fixed array.
- R̂ is about 1 for iid chains and above 1.1 for shifted chains.
- Degenerate inputs are validated.
- PSIS:
  - k̂ is small for well-behaved weights and large for heavy-tailed weights;
  - equal weights have no tail;
  - the smoothed weights are normalised.
- The autocorrelation function is correct.
- The KKT gap is tested with the model (§2.5).

### 2.9 `tests/test_api.py` (§8)

**`infer`**
- Accepts a raw array, a `Likelihoods` object, a model and `bins=` (R-INPUT-4).
- Accepts each method string. Unknown methods raise with the list of valid ones.
- Unknown keyword arguments raise.
- `method="map"` returns a `PointEstimate`.
- `rng` reproducibility.
- A selection passed through `infer`; `method="em"` returns a `PointEstimate`, and `"auto"` records its choice.
- **Not implemented:** with `progress=True` the run completes and prints nothing to stdout when tqdm is missing.

`method="auto"` has its own file, `tests/test_auto.py`: one test per rule, except the joint rule, tested with the joint models in `tests/test_injections.py` (DECISIONS #88).

**`PopulationEstimator`**
- `fit` returns self; `get_params` and `set_params` round-trip; `predict_proba` shapes; `score` returns a finite value.
- It is clonable: `get_params(deep=False)` reproduces the constructor.
- MAP and raw arrays.

**Top-level namespace**
- `binference.__all__` exports the public names, and `__version__` is defined.
- `CITATION.cff` carries the package version.

### 2.10 `tests/test_simulate_validation.py` (§9)

**Mocks**
- The mock generators give correct shapes, and their truths are consistent: realised counts equal the histogram of the true values.
- Interim-prior posteriors and samples round-trip and follow the interim prior given.
- Selection mocks thin each bin as expected and add the outlier fraction.
- `prior_predictive` keeps objects in proportion to the selection, takes the bin-average keep probability on the measurement, draws a Poisson number of objects for rates, and validates its inputs.

**`exact_posterior`**
- It matches the conjugate posterior for one-hot likelihoods and is accurate at the simplex boundary.

**`sbc` and `coverage` (fast smoke versions)**
- Small runs produce ranks and rates with the right shapes.
- The uniformity p-values of uniform ranks reject at the nominal rate; the ECDF p-value is the null probability of a smaller band statistic.

### 2.11 Slow statistical validation (`-m slow`)

`tests/test_slow.py`:
- **SBC**: NUTS with `Dirichlet(1)` and `LogisticRandomWalk(order=1, scale=0.8)` (200 simulations, K = 4); Gibbs with `Dirichlet(1)` (1000 simulations) and the generalised Dirichlet (500); a check that SBC rejects posteriors about 18% too narrow; VB is expected to fail. Ranks are computed for each bin fraction and for the joint log-likelihood (Modrák et al. 2023).
- **Coverage**: 68% and 95% central intervals of Gibbs for the bin fractions and the mean, over 1000 simulations, within four binomial standard errors of nominal.
- **Geweke "getting it right"** for the Gibbs conditionals, with and without selection, and a check that it has power against a broken kernel.
- **Larger cross-engine agreement**: Gibbs and NUTS agree at K = 15, N = 5000.
- **Recovery**: whitened residuals of posterior means over simulated catalogues with interim-prior posteriors, and with selection.
- **Interim-prior regression**: feeding posteriors as likelihoods biases ρ toward π; using `from_table` with the interim prior gives no bias.
- Binning bias with coarse bins is real; VB underestimates widths against NUTS.

`tests/test_sbc_cases.py` (audit T-2): SBC of an order-2 random walk with an inferred scale (NUTS), Gibbs with a known selection in both modes, rates with an exposure, the Pólya tree and the importance-corrected Laplace approximation; Geweke tests of the generalised Dirichlet, a Dirichlet tree and Gamma rates with an exposure.

`tests/test_matrix_regression.py` (DECISIONS #42): NUTS, Gibbs, Laplace and `method="auto"` reproduce stored 20,000-draw NUTS references of benchmark-matrix cells (`tests/data/matrix_reference.json`) within four combined Monte Carlo errors.

`tests/test_injections.py` and `tests/test_joint_response.py`: joint inference of the population with a selection from injections or a tallied response is calibrated, and NUTS agrees with joint Gibbs.

**Not implemented** from the original plan: cross-engine agreement at K = 30, N = 2×10⁴ (the suite uses K = 15, N = 5000); recovery with outliers.

### 2.12 Docs, examples and packaging

- `docs/tutorials/*.md` (MyST notebooks) are executed in the docs CI job (`myst-nb`, `nb_execution_mode="cache"`, fail on error). Each tutorial should run in under 2 minutes. Numbers that the prose quotes or compares are checked by hidden `assert` cells (tagged `remove-cell`).
- The code of `README.md`, `docs/start/install.md` and the guides runs in CI with `docs/_scripts/run_snippets.py`, which also runs tutorials locally.
- `python -m build` produces sdist and wheel; `twine check` passes, and the wheel is installed in a clean virtualenv and imported (CI job `dist`).
- **Not implemented:** numpydoc validation as a lint step.

### 2.13 Performance benchmarks (non-gating)

`benchmarks/` holds scripts run by hand:
- `bench_kernels.py`: timings of the likelihood kernels;
- `matrix.py`: the benchmark matrix of scenario × prior × engine against long reference runs (DECISIONS #42, #88), with results in `benchmarks/results/`. `tests/test_benchmark_matrix.py` runs its quick mode.

The spec's §10 targets (NUTS ESS/s, Gibbs iterations/s, MAP and Laplace times) have no dedicated benchmark script.

### 2.14 Tests added with later decisions and audits

| File | What it covers |
|---|---|
| `test_auto.py` | `method="auto"`: one test per rule, its options and its records (DECISIONS #88) |
| `test_basis.py` | the mixture basis and internal resolution (DECISIONS #4, #5, #47, #76) |
| `test_benchmark_matrix.py` | the benchmark matrix script runs end to end in its quick mode |
| `test_best_loo.py` | the best fit, `_best` outputs and leave-one-out posteriors (DECISIONS #35, #51, #52, #85) |
| `test_binning_bias.py` | the binning-bias check and its planning number (DECISIONS #6, #20, #86) |
| `test_constructors.py` | constructors by input form, adaptive quadrature and measures in constructors (DECISIONS #21, #23, #32, #64) |
| `test_correlated.py` | correlated Gaussian likelihoods (DECISIONS #2, #25, #66) |
| `test_density.py` | `Density`, interim priors and explicit stacking (DECISIONS #11, #24, #26) |
| `test_derived_rates.py` | totals and rates derived from fraction draws (DECISIONS #9, #78) |
| `test_gibbs_fixes.py` | Gibbs, bootstrap and validation fixes (audit G-1–G-12, DECISIONS #75) |
| `test_hygiene.py` | `Likelihoods` validation, storage and `compress` (DECISIONS #1) |
| `test_injections.py` | selections from injections inferred with the population (DECISIONS #10, #30, #79, #80) |
| `test_joint_response.py` | tallied responses inferred with the population (DECISIONS #28, #81) |
| `test_map_fixes.py` | MAP fixes (audit M-5, M-6, M-8, M-9, M-13, T-4) |
| `test_measures_masks.py` | measures and masked grids (DECISIONS #19, #21) |
| `test_model_fixes.py` | model fixes (DECISIONS #15, #56, #57; audit M-1–M-4, M-7, M-14) |
| `test_nuts_fixes.py` | NUTS and Laplace fixes (audit N-1–N-15, DECISIONS #74) |
| `test_prior_fixes.py` | prior fixes (audit P-1–P-22, DECISIONS #72, #73) |
| `test_response.py` | binned measurements and the classifier routes (DECISIONS #27, #29) |
| `test_results_audit.py` | results, io and plotting fixes (audit O-1–O-21) |
| `test_results_outputs.py` | the selected population, full-sample counts, KKT checks, data support and the prior-vs-data ratio |
| `test_review_regressions.py` | findings of the independent v0.1 code review |
| `test_samples.py` | the samples estimator (DECISIONS #31, audit I-23) |
| `test_selection.py` | selection functions (DECISIONS #7, #8, #10, #46, #47, #48) |
| `test_values.py` | value draws from each input's within-bin information (DECISIONS #50) |

## 3. Mapping from requirements to tests

| Requirement | Tests |
|---|---|
| R-THEORY-1 (population ≡ hierarchical) | 2.6 Gibbs vs NUTS vs exact; 2.11 cross-engine |
| R-THEORY-2 (selection) | 2.5 selection modes; 2.6 selection agreement; 2.11 Geweke with selection, SBC with selection |
| R-INPUT-1/2/3/4 | 2.2 (`from_table`, representations, validation); 2.9 raw arrays |
| R-OUT-1 (population vs realised) | 2.7 `realized_counts`, `sample_assignments` |
| R-OUT-2 (KKT certificate) | 2.5 KKT gap; 2.6 mix-SQP certified; `test_results_outputs.py` (`PointEstimate.check`) |
| R-DIAG-1 | 2.8 |
| R-BINS-1 | 2.1, 2.4 (smoothness priors on categories raise) |
| R-INF-MAP-2 (SQUAREM default) | 2.6 MAP iteration counts |
| Engines §6.2–6.9 | 2.6; `test_auto.py` |
| Validation tools §9 | 2.10, 2.11 |
