# binference — test plan (v0.1)

This plan turns the requirements in [`spec.md`](spec.md) into tests. It has three goals:

1. Every public behaviour is covered by a fast, deterministic test.
2. Every inference engine is shown to target the right posterior, using exact references, invariances, cross-engine agreement and simulation-based calibration.
3. The docs and tutorials stay executable.

## 1. Principles

- **Deterministic.** Every random test uses a fixed seed passed through `rng`. Tests that rely on statistical tolerances use a seed that passes, plus tolerances wide enough that the false-failure probability is below 1e-4 for any seed. Seed-robustness is checked offline with `BINFERENCE_TEST_SEEDS=…`. The test suite has a fixture that loops over seeds when that environment variable is set, in the same spirit as dynesty.
- **Tolerances scale with Monte Carlo error.** Compare against references with
  $$|\hat\theta - \theta_\text{ref}| \le z\cdot \mathrm{MCSE}(\hat\theta) + \epsilon_\text{abs}, \qquad z = 4 \text{ to } 5.$$
  Standard deviations use a relative tolerance derived from the effective sample size (ESS).
- **Two tiers.**
  - Default `pytest` runs fast tests: unit, exactness on small problems, and API tests. The target is under 2 minutes on one core.
  - `pytest -m slow` runs the statistical validation: SBC, coverage, larger cross-engine comparisons and the Geweke test. CI runs it in a separate job.
- **Warnings are errors in CI** (`-W error::binference.BinferenceWarning` plus `filterwarnings` in `pyproject.toml`). Tests that expect a warning assert it with `pytest.warns`.
- **Requirement IDs.** Test modules and docstrings cite requirement IDs (`R-…`) where they apply.

## 2. Test inventory

### 2.1 `tests/test_bins.py` (R-BINS-*)

**Constructors and geometry**
- `Bins(edges)`: edges are validated (monotone, finite, ≥ 2 values). Check `K`, `centers`, `volumes` and `ndim == 1`.
- `Bins.linear` and `Bins.log`: spacing and endpoint values.
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
- `coarsen`: groups are valid, volumes add up, and an invalid grouping raises.
- Equality and repr; immutability (attribute assignment raises).

### 2.2 `tests/test_likelihoods.py` (R-INPUT-*)

**Constructor**
- Basic construction from a dense array, a sparse CSR matrix and a list of lists.
- `log_scale` is recorded, the row max is 1, and `values * exp(log_scale)` reproduces the input.
- `from_log`: agrees with exponentiating directly, and stays stable when log values are around −1e4.

**Validation**
- Negative values, NaN and inf raise.
- All-zero rows raise. With `drop_empty=True` they are dropped and an `InputWarning` is emitted.
- A weights length mismatch raises, and non-positive weights raise.

**Representations**
- "mass" divided by the volumes equals "density" for a known Gaussian on unequal bins.
- **Unequal-bin regression test:** a known population with unequal bins is recovered by the MAP to within 1e-3 when the right representation is used. The wrong representation gives a bias above a threshold, which documents why the argument exists.

**`from_posteriors`**
- Omitting `interim_prior` raises (R-INPUT-1).
- `"flat"` is accepted.
- Mass and density interim priors give identical likelihoods.
- **Interim-prior removal:** simulate likelihoods $\ell$, form posteriors $p \propto \ell\pi$, then convert back with $\pi$. The result is proportional to $\ell$ row by row (rtol 1e-12).
- Representation sanity warnings fire: declared masses whose rows do not sum to 1 emit an `InputWarning`.

**`from_samples`**
- A histogram of known samples gives the expected masses divided by the interim masses.
- Out-of-range samples are reported in `meta`.
- Per-object effective sample counts are reported.
- A ragged list of arrays is accepted.
- N-D samples are accepted.
- A callable interim prior is accepted.

**Other constructors**
- `from_gaussian`:
  - 1-D values match the analytic CDF differences divided by the volume;
  - N-D with diagonal σ equals the product of 1-D terms;
  - σ = 0 raises.
- `from_distributions`: agrees with `from_gaussian` for scipy `norm` objects, and works for skewed distributions such as `skewnorm`.
- `from_function`: the quadrature converges to `from_gaussian` as `n_sub` grows.
- `from_binned`: rows equal the response rows, weights equal the counts, and zero-count rows are dropped. The efficiency comes back as an `"observed"` selection.
- `Response`: `from_simulation` reproduces a known response within binomial error; a migration matrix with its efficiency equals the joint matrix; route C recovers the prevalence when the posteriors are miscalibrated; the predictive check matches its pseudo-inverse form and flags label shift.
- Route A (`from_table` on categories with the training prior): $p/\pi_\text{train}$ is proportional to the row values.

**Methods**
- `compress()`: duplicate rows merge into summed weights, and the posterior is identical (MAP to 1e-10; the NUTS/Gibbs agreement is in the slow tests).
- `subset`, `to_dense` and `to_sparse` round-trip.
- `N`, `K`, `nnz` and `len`.

### 2.3 `tests/test_simplex.py` (internal helpers)

- The Helmert basis is orthonormal with columns summing to zero.
- Round trips:
  - ILR to softmax and back is the identity (1e-12);
  - softmax is stable for inputs around ±1e3.
- The log-Jacobian of the ILR map matches a finite-difference Jacobian determinant to a constant, for K = 2..6.

### 2.4 `tests/test_priors.py` (§5)

**Every prior, parametrised over the families**
- `log_prob` integrates to 1 on the 2-simplex (K = 3 quadrature, tolerance 1e-3; looser where α < 1 makes the density singular).
- Sample moments match the analytic moments (Dirichlet, GD), or match Monte Carlo moments computed from the tree definition (Dirichlet-tree, Pólya tree).
- The unconstrained log-density and its gradient agree with finite differences (rtol 1e-6) at random points.
- `prior_transform(u)`: uniform `u` gives draws whose moments match `sample`, and its Jacobian is consistent where checked.

**Conjugate priors**
- `conjugate_update(counts)` equals the brute-force posterior. For K = 3 on a grid, prior × multinomial likelihood is normalised and compared.
- `map_update` equals argmax(log prior + Σ n log ρ), computed by numerical optimisation.

**Family-specific checks**
- Generalised Dirichlet:
  - `GeneralizedDirichlet(a, b)` with b_k = Σ_{j>k} a_j reduces to `Dirichlet(a)` in its density, to 1e-12;
  - it can produce positive correlations (Monte Carlo check).
- `DirichletTree`:
  - the linear log-density form equals the node-product form;
  - a tree that satisfies the consistency condition collapses to a Dirichlet.
- `PolyaTree`: the prior mean equals the base measure, and the variance decreases as `c` grows.
- `Dirichlet.reference`, `from_counts` and `from_base_measure` give the documented α values.

**Logistic family**
- `LogisticNormal`: sample moments of η match the given mean and covariance, and the shift mode is removed.
- `LogisticRandomWalk`:
  - prior draws have the expected roughness (the variance of second differences scales with `scale²`);
  - the 2-D ICAR neighbour structure is correct;
  - categorical bins raise.
- `LogisticGP` (Hilbert-space approximation):
  - with fixed hyperparameters, the covariance matches the exact kernel to a tolerance set by the number of basis functions J;
  - the gradient with respect to the hyperparameters matches finite differences.
- Hyperpriors (`HalfNormal`, `LogNormal`, `InverseGamma`, `Fixed`): log-density and gradient checks.

**Rates**
- `GammaRates`: moments and conjugate update.

### 2.5 `tests/test_model.py` (§3, §6.1)

**Likelihood**
- `log_likelihood(rho)` equals the direct formula, for dense and sparse inputs, with and without weights and with and without selection.
- Invariances:
  - multiplying row i by $c_i$ leaves the posterior unchanged (log-likelihood changes by $\sum w_i\log c_i$ only);
  - permuting objects leaves it unchanged;
  - duplicating a row is the same as weight 2.

**Gradients and transforms**
- The gradient of `log_prob` in unconstrained coordinates agrees with finite differences for every prior family × selection × rates combination (parametrised).
- `to_rho` / `from_rho` round-trip.
- `responsibilities` rows sum to 1, and `expected_counts` equals responsibilities summed with weights.

**Selection and diagnostics**
- Selection formula (S): checked against direct enumeration for a tiny problem, where the Λ-marginalisation is done numerically with a 1/Λ prior.
- `missing_information` is 0 for one-hot likelihoods and approaches 1 for flat likelihoods.

### 2.6 Engine tests: exactness on small problems (fast)

**Fixtures**
- `tiny_problem(K=3, N=200)` with heteroscedastic Gaussian likelihoods.
- An exact posterior by quadrature (`binference.validation.exact_posterior`).
- One-hot likelihoods, which give the conjugate posterior Dir(α + counts) exactly.

**`tests/test_engine_nuts.py`**
- Matches the K = 3 quadrature in mean, s.d. and 5/50/95% quantiles within the MCSE tolerances.
- One-hot likelihoods give Dir(α + n) moments.
- Standalone checks on a pure Gaussian target:
  - the step size adapts;
  - the diagonal metric approaches the true variances;
  - dense and diagonal metrics both work.
- Divergence reporting: a pathological funnel target yields `diverging > 0`.
- The same seed gives identical draws, and different seeds give different draws.
- Resumability: one `run(500)` followed by `run(500)` produces 1000 draws.
- The generator interface yields the expected shapes.

**`tests/test_engine_gibbs.py`**
- Matches the K = 3 quadrature, and one-hot likelihoods give the exact conjugate posterior.
- **Selection:** agrees with NUTS on (S) for a K = 4 problem (moderate size, fast).
- `GammaRates` with selection: agrees with NUTS on the rate posterior.
- Generalised Dirichlet and Dirichlet-tree priors: agree with NUTS.
- Stored labels have shape `(n_draws, n_objects)`, and their histogram equals `realized_counts`.
- Non-integer weights raise.
- Sparse and dense inputs give identical chains under the same seed.

**`tests/test_engine_map.py`**
- The MLE KKT gap is below 1e-8 at convergence.
- The MLE matches a reference from `scipy.optimize` with a softmax parametrisation, to 1e-6.
- EM with SQUAREM and without it converge to the same point, and SQUAREM uses at least 5× fewer iterations on a broad-PDF problem.
- The fixed point equals iterated stacking (one EM step at the fixed point leaves it unchanged).
- MAP with a Dirichlet prior (α ≥ 1) equals a numerical-optimisation reference. α < 1 clips and emits a warning.
- Selection EM matches numerical maximisation of (S).
- Logistic priors use L-BFGS; the result matches `scipy.optimize` in unconstrained coordinates.

**`tests/test_engine_laplace.py`**
- On a large-N, well-populated problem, mean and s.d. match NUTS within 5%.
- The importance-resampled version reports k̂, and the corrected moments move toward NUTS.
- The analytic Hessian matches a finite-difference Hessian.

**`tests/test_engine_vb.py`**
- One-hot likelihoods give exactly Dir(α + n).
- On broad PDFs the s.d. comes out smaller than NUTS's. This documents the known under-dispersion, with the result flagged `approximate=True` and an `ApproximationWarning`.

**`tests/test_engine_bootstrap.py`**
- Shapes, determinism and the flagging of the results.
- For large N, the bootstrap spread is close to the posterior s.d. (loose tolerance).

### 2.7 `tests/test_results.py` (§7)

**Summaries and intervals**
- `mean`, `std`, `quantile`, `interval` (central and HDI) and `cov` agree with NumPy on known arrays.
- `summary()` repr, and `to_dict`.
- `density()` equals ρ / volumes.

**Derived quantities**
- `functional`, and `rebin`, which is exactly the sum over groups.

**Individual objects**
- `assignment_probabilities` matches a Monte Carlo average of the responsibilities.
- `sample_assignments` shapes, and the empirical frequencies converge to the assignment probabilities.
- `realized_counts` sums to W in every draw.
- `sample_values` stays within the bin edges.

**New objects**
- `predict_proba` and `log_predictive` on new objects.
- `log_predictive` beats a deliberately wrong population on held-out data.

**Diagnostics**
- Diagnostics are present, and `check()` warns on synthetic bad chains (stuck chains → R̂ warning).

**Persistence**
- `save` / `load` round-trip, including the metadata, the schema version and the bins.

**Optional dependencies**
- `to_datatree` is skipped when xarray is not installed.
- `PointEstimate.laplace()` returns a `PopulationResult`.

### 2.8 `tests/test_diagnostics.py` (R-DIAG-1)

- On an AR(1) process with known φ:
  - the integrated autocorrelation time ≈ (1+φ)/(1−φ), within 10%;
  - ESS ≈ n / τ.
- R̂ is about 1 for iid chains and above 1.1 for shifted chains. The rank-normalised version handles heavy tails (Cauchy chains).
- MCSE of the mean ≈ s.d. / √ESS.
- PSIS:
  - k̂ is below 0.5 for well-behaved weights;
  - k̂ is above 0.7 for heavy-tailed (Pareto α = 1) weights;
  - the smoothed weights are normalised.
- KKT gap: 0 at the MLE, and positive away from it.

### 2.9 `tests/test_infer_api.py` and `tests/test_estimator.py` (§8)

**`infer`**
- Accepts a raw array, a `Likelihoods` object and `bins=` (R-INPUT-4).
- Accepts each method string. Unknown methods raise with the list of valid ones.
- Unknown keyword arguments raise.
- `method="map"` returns a `PointEstimate`.
- `rng` reproducibility.
- With `progress=True` the run completes and prints nothing to stdout when tqdm is missing.

**`PopulationEstimator`**
- `fit` returns self.
- `get_params` and `set_params` round-trip.
- `predict_proba` shapes.
- `score` returns a finite value.
- It is clonable, matching `sklearn.base.clone` semantics without importing sklearn: `get_params(deep=False)` reproduces the constructor.

**Top-level namespace**
- `binference.__all__` exports what the README uses.
- `__version__` is defined.

### 2.10 `tests/test_simulate.py` and `tests/test_validation.py` (§9)

**Mocks**
- The mock generators give correct shapes, and their truths are consistent: realised counts equal the histogram of the true values.
- Interim-prior mocks round-trip through `from_posteriors`.
- Selection mocks thin as expected.

**`exact_posterior`**
- It is normalised and matches the conjugate posterior for one-hot likelihoods.

**`sbc` (fast smoke version)**
- 20 simulations with a tiny K produce ranks with the right shape.
- The uniformity statistic is computed.

### 2.11 Slow statistical validation (`-m slow`)

**SBC**
- Configurations:
  - K = 4, N = 50, 200 simulations;
  - engines NUTS and Gibbs (and Laplace with importance resampling, flagged expected-pass);
  - priors Dirichlet(1), generalised Dirichlet and `LogisticRandomWalk`.
- Ranks are computed for each bin fraction and for the joint log-likelihood (Modrák et al. 2023).
- Pass criterion: a χ² uniformity test on the binned ranks, with p > 1e-3 after a Bonferroni correction across quantities.
- VB is expected to fail SBC, recorded as `xfail(strict=True)` to document the under-dispersion.

**Coverage**
- 68% and 95% central intervals for bin fractions and for the mean-redshift functional.
- Measured over 300 simulations; within binomial error of nominal.

**Geweke "getting it right" test for the Gibbs sampler**
- On a tiny problem, alternate the Gibbs update with the data update and compare against prior draws (two-sample tests).
- Both with and without selection.

**Larger cross-engine agreement**
- K = 30, N = 2×10⁴ mocks: Gibbs (long chain) and NUTS agree within MCSE.
- Laplace with importance resampling agrees on well-populated bins.

**Recovery**
- Over 50 simulated catalogues, the posterior-mean residuals in units of the posterior s.d. have mean ≈ 0 and s.d. ≈ 1.
- The truth includes interim priors, selection and outliers.

**Interim-prior regression test**
- Feeding posteriors as likelihoods biases ρ toward π, beyond 5σ.
- Using `from_posteriors` correctly gives no bias.
- This guards against regressions of the most common user error.

### 2.12 Docs, examples and packaging

- `docs/tutorials/*.md` (MyST notebooks) are executed in the docs CI job (`myst-nb`, `nb_execution_mode="cache"`, fail on error). Each tutorial must run in under 2 minutes.
- README example: a doctest-style test extracts the snippet and runs it on a mock catalogue.
- `python -m build` produces sdist and wheel, and a smoke test installs the wheel in a clean virtualenv and imports it (CI job).
- Docstrings: numpydoc validation is run as a lint step for public functions.

### 2.13 Performance benchmarks (non-gating)

`benchmarks/` holds simple scripts (asv may come later). Each one records timings and ESS/s against the targets in spec §10:
- gradient evaluation;
- NUTS ESS/s;
- Gibbs iterations/s;
- MAP time;
- Laplace time.

## 3. Mapping from requirements to tests

| Requirement | Tests |
|---|---|
| R-THEORY-1 (population ≡ hierarchical) | 2.6 Gibbs vs NUTS vs quadrature; 2.11 cross-engine |
| R-THEORY-2 (selection) | 2.5 selection formula; 2.6 selection agreement; 2.11 Geweke with selection |
| R-INPUT-1/2/3/4 | 2.2 (from_posteriors, representations, validation); 2.9 raw arrays |
| R-OUT-1 (population vs realised) | 2.7 `realized_counts`, `sample_assignments` |
| R-OUT-2 (KKT certificate) | 2.6 MAP; 2.8 KKT |
| R-DIAG-1 | 2.8 |
| R-BINS-1 | 2.1, 2.4 (smoothness priors on categories raise) |
| R-INF-MAP-2 (SQUAREM default) | 2.6 MAP iteration counts |
| Engines §6.2–6.7 | 2.6 |
| Validation tools §9 | 2.10, 2.11 |
