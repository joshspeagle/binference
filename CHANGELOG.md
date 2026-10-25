# Changelog

All notable changes to binference are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (with the usual 0.x caveat that minor
releases may break the API).

## [Unreleased] - 0.1.0.dev0

First development version.

### Changed (landing on `main`)

- Renamed from `pastry` to `binference` (package, import and PyPI name; DECISIONS #45).
  The warning base class is `BinferenceWarning` and the environment variables start
  with `BINFERENCE_`.

- Python ≥ 3.13, NumPy ≥ 2.1, SciPy ≥ 1.14.1, matplotlib ≥ 3.9.2 (DECISIONS #41, #61).
- CI runs the pre-commit hooks, the fast suite on three systems with Python 3.13 and
  3.14, minimum dependencies, the README and user-guide code, the docs build and a
  clean-wheel import; pushes to `main` publish the docs to GitHub Pages.
- Development process: issue and pull-request templates and the rules in `CLAUDE.md` and
  `CONTRIBUTING.md`.
- The prior base class is now `binference.priors.PopulationPrior` (was `Prior`).
- On log bins, the default base measures of `Dirichlet.from_base_measure` and
  `PolyaTree`, the default mean of the logistic priors and the "flat" interim prior
  are flat in log z (they were flat in z; DECISIONS #63).

### Added

- **Measures** (DECISIONS #21): each grid axis has a `Measure`, the coordinate
  `u = T(z)` in which the population is flat within a bin: linear, log, a transform with
  its inverse, or the cumulative integral of a positive density. `Bins.log` uses the log
  measure; bin centres are midpoints in `u`; `Bins.from_centers` takes midpoints in `u`
  (positive edges for log-spaced centres, I-16). Geometric hand-built edges warn.
- **Selection from injections** (DECISIONS #10, #30, #79, #80): `Selection.from_injections`
  gives a completeness with Beta posteriors per bin, inferred jointly with the population
  by Gibbs, NUTS, Laplace and `MAP(space="unconstrained")` (logit S as extra
  coordinates); `smooth=τ` ties neighbouring bins by a random walk on logit S;
  `at_mean()` fixes it for VB and the bootstrap.
- **Responses inferred with the population** (DECISIONS #28, #81): a response tallied
  from events (`Response.from_counts`, `from_simulation`, `from_labelled`) has a
  Dirichlet posterior per column (`prior=β`, default 1 per cell) and is inferred jointly
  by Gibbs, NUTS, Laplace and unconstrained MAP; `response.fixed()` takes it as known.
- **Totals and rates from fractions** (DECISIONS #9, #78): every fractions result has
  `total` (draws of Λ ~ Gamma(W, S·ρ), exact under a scale-free prior) and `rates`
  (Λρ), with an absolute or no selection; point estimates give W/(S·ρ̂).
- **Mixture basis and internal resolution** (DECISIONS #4, #5, #47, #76):
  `binference.Basis(bins, order=0|1, resolution=r)` fits components on the bins refined
  `r` times, flat (`order=0`) or linear B-spline hats (`order=1`), and reports in the
  bins. Every constructor takes a `Basis` for `bins` (`Likelihoods.on_basis` for
  per-bin likelihoods); callable selections are averaged with each component as the
  weight. Results report per bin and keep `component_draws`; save and load keep the
  basis. `Bins.refine` is the counterpart of `coarsen`. `sample_values` on non-linear
  measures now draws flat in `u` within a bin (it ignored `T'(z)`).
- `from_samples` stores likelihoods sparsely below 10% non-zero entries (was 30%; DECISIONS
  #77).
- **Masked grids** (DECISIONS #19): `mask=` on `Bins`, `Bins.grid`, `Bins.linear` and
  `Bins.log`; masked cells leave the support and the model. Saved and loaded with
  results.
- `Bins` with edges `-0.0` and `0.0` now hash alike (I-12); `Bins.index` refuses
  multi-column values for 1-D bins (I-22).
- **`Density`** (DECISIONS #26, #63): one fixed density over the bin coordinate for
  interim priors, the Dirichlet base measure and simulation truth. `Density.flat()` is
  flat in the measure coordinate. An array interim prior on unequal bins needs
  `interim_representation`, so every constructor reads it the same way (I-1); a
  callable on categories receives the labels (I-17).
- `Likelihoods.stack()` uses the interim prior the likelihoods were built with
  (I-5) and raises on rows with no support under it (I-6).
- **Constructors by input form** (DECISIONS #32, #64): `Likelihoods.from_table` and
  `Likelihoods.from_density` replace `from_posteriors`, `from_function` and
  `from_distributions`; each needs `kind="likelihood"` or `interim=`. `interim_prior`
  is now `interim` (also in `stack()` and `simulate.mock_1d`).
- **Adaptive quadrature** (DECISIONS #23): bin averages of functions and distributions
  are refined to `tol` (default 1e-3), in the measure coordinate for likelihoods
  (DECISIONS #21); `from_gaussian` averages in u on non-linear axes. Far tails no longer
  underflow to empty rows (I-2, I-3).
- **Samples** (DECISIONS #31, #67): one estimator for every interim prior (per-sample
  weights T'(z)/π(z); a per-bin prior is flat in u within bins), an opt-in Dirichlet
  `floor` in pseudo-samples,
  `n_eff` inside the bins with a warning on the share of noisy objects (I-23),
  `Likelihoods.sample_variance` and a Monte Carlo variance check in `result.check()`;
  `Measure.log_jacobian`.
- **Binned measurements** (DECISIONS #27, #29, #68): `binference.Response` holds
  P(measured bin, observed | true bin) with real truth bins, from a matrix, tallies
  (`Response.from_counts`), simulated events (`Response.from_simulation`) or a labelled
  validation set (`Response.from_labelled`: open-ended score bins or a confusion matrix).
  `Likelihoods.from_binned` replaces `from_response`; `from_classifier` is removed
  (route A is `from_table` on categories, route C is `from_binned` with a labelled
  response). `summary()` names the inputs, and `result.check()` runs a posterior
  predictive test of the measured counts (`Response.predictive_pvalue`) that flags label
  shift. `check()` raises `InputWarning` for problems with the inputs (this test and the
  samples' Monte Carlo variance) and `ConvergenceWarning` for sampler problems. `simulate.mock_categorical` gains `temperature` and a labelled `mock.response`;
  `simulate.mock_unfolding` returns `mock.response`.
- **Selection** (DECISIONS #7, #8, #10, #46, #47, #48, #69): the modes are
  `"on_truth"` and `"on_measurement"` (the old names raise an error naming the new
  ones); `Selection.combined(on_truth, on_measurement)` selects on both; a selection may
  be a function of the true value, averaged over each bin (`Selection.from_function`,
  or a callable `selection=`); values are detection probabilities in (0, 1] unless
  declared `scale="exposure"` or `"relative"`, and rates refuse a relative selection
  (audit G-4); an array that changes by more than a factor of two between neighbouring
  bins raises an `InputWarning` (a `Selection` object or a callable is taken as averaged
  within bins).
  `simulate.mock_1d(measured_range=...)` simulates selection on the measurement, alone
  or with `selection=` (combined). `sbc`/`coverage` name rates `lambda[k]` (G-10).
- **Model** (DECISIONS #15, #56, #57, #70): `PopulationModel.log_prob` and
  `log_prob_and_grad` are renamed `log_density_unconstrained` and
  `log_density_unconstrained_and_grad`; the theory page explains the two densities
  and their modes. `kkt_gap` is infinite where an object is impossible and, for rates,
  relative to the exposure (audit M-3, M-4); `missing_information` is correct with a
  selection function and on the support, with `include_prior=` for the MAP EM (M-2).
  Bootstrap replicates drop zero-weight objects in every engine, so EM no longer
  resets to the prior's mode or collapses rates to zero (M-1); EM stops with a warning
  where an object becomes impossible. The rates starting point is W ρ / S (M-7).
  `LogisticGP()` works on two bins (M-14). `PopulationResult.log_prob` is now
  `log_density_unconstrained` (old files still load); `MAP.run` no longer takes
  `weights=`.
- **MAP** (DECISIONS #71): `tol` (default 1e-8, was 1e-10 on the objective) is the
  tolerance on the KKT gap for every simplex algorithm and on the largest gradient over
  `max(1, W)` for L-BFGS (mix-SQP bounds the gap of the problem it solves), and EM is
  converged only when the gap is below it, or below `1e3 * tol` once the objective
  stops improving (M-6, M-9).
  `GammaRates` with shapes of at least one use mix-SQP, with the total rate profiled
  out (M-6). `PointEstimate.log_posterior` is `model.log_posterior(value)` for every
  algorithm (NaN for priors without a closed-form density); L-BFGS adds
  `info["log_density_unconstrained"]` (M-8). mix-SQP drops its SVD and starts after
  10 EM steps instead of 50, 2–3 times faster on large dense inputs (M-5); a failed
  quadratic step hands over to EM instead of raising (M-13); Dirichlet priors
  mixing concentrations of one and above one no longer warn (T-4). The rates KKT gap
  is one-sided, so a rate EM drives towards zero no longer reads as a violation.
- **Priors** (DECISIONS #72): `LogisticRandomWalk(trend_scale=3.0)` is the standard
  deviation of each unpenalised mode's root-mean-square size across the grid, so its
  meaning no longer depends on K (was 10 in basis units, which tightened as 1/√K; P-6).
  Arguments are checked when a prior is made: `DirichletTree` per-node alphas of
  different lengths now work (P-4), `LogisticNormal(cov=)` must be a positive variance
  or a matrix positive semi-definite on zero-sum vectors (P-8), `LogisticGP` `n_basis`
  and `boundary_factor` are checked and `length_scale` takes an array (P-10, P-11),
  `PolyaTree` needs two bins and a non-negative power (P-9), `GeneralizedDirichlet`
  `a`, `b` and `order` are checked (P-22). Reprs show every argument (P-14); tree VB
  draws have no exact zeros (P-12); `GammaRates` has a `prior_transform` (P-15).
  `GeneralizedDirichlet` and `PolyaTree` are no longer subclasses of `DirichletTree`
  (P-13). `LogisticRandomWalk(2).bind` at K = 1000 and 20,000 GP draws are about twice
  as fast (P-20).
- **GP prior** (DECISIONS #73): `LogisticGP` is stationary in the measure coordinate
  (log z on `Bins.log`; length scales are in that coordinate; P-1). Defaults:
  `boundary_factor=3` (was 1.5; P-3), `n_basis` one per bin up to 1024 modes in all
  (was at most 48 per axis), and a
  length-scale prior whose lower 1% quantile is the shortest length scale that basis
  resolves (P-2), or `LogNormal(log(range/2), 1)` on coarse grids (P-7). Partially
  centred GP densities and gradients agree where scales underflow (P-5).
- **NUTS and Laplace** (DECISIONS #74): NUTS results (`x`, `log_density_unconstrained`,
  `model`) and the positions `sample()` yields are in the model's own coordinates, not
  the partly centred sampling ones (`info["centring"]` holds the weights; N-1, N-2).
  `diagnostics.psis_threshold(n)` gives the k̂ threshold for n importance weights. Parallel workers receive the model
  once, reuse one pool per run, never raise a BLAS thread limit already set, respect
  CPU affinity, and errors from the model are raised instead of triggering a serial
  rerun (N-3, N-7, N-13). `max_depth`, `n_warmup`, `jitter` and `init` are checked
  (N-5); Laplace returns every draw asked for without importance resampling (N-4) and
  judges k̂ against min(1 − 1/log10 S, 0.7) (N-6). Dense momentum draws use a
  precomputed triangular factor (N-9).
- **Validation, Gibbs, bootstrap** (DECISIONS #75): SBC p-values and bands use the
  expected count per rank bin (G-1) and `sbc` warns when the ESS is below `n_draws` (G-9);
  `exact_posterior` integrates by stick-breaking Gauss–Legendre quadrature that resolves
  the simplex boundary, also for concentrations below one, adds `log_density`, and bounds
  its memory, with a default `n_grid` of 200 (G-3, G-5); weighted Gibbs on sparse
  likelihoods draws counts over each row's nonzeros only (G-2); Gibbs checks `chunk`,
  `store_labels` and `init`, and chains start apart, from counts that include missed
  objects (G-6, G-7); SBC rejects `n_bins` above `n_draws + 1`; `Bootstrap(kind=)` is now
  `scheme=`, non-converged replicates are counted and the base fit warns (G-8).
- **Correlated Gaussians** (DECISIONS #2, #25, #66): `from_gaussian(..., cov=)` takes full
  per-object covariances, averaged over N-D bins by adaptive cubature in log space.
- `weights` are integer multiplicities; non-integer weights need `fractional_weights=True`
  and warn that the result is a pseudo-likelihood (DECISIONS #1, #65). Stored arrays
  are read-only (I-14); `compress()` keeps sparse storage and is fast on it (I-7, I-19);
  warnings point at the caller's line; clearer errors for bad `rtol`, Gaussian shapes,
  zero objects and 2-D arrays passed to `Bins.grid` (I-8, I-10, I-11, I-20); broad
  Gaussians keep full precision near the mean (I-15).
- Each object's likelihood mass outside the bins is recorded in
  `meta["outside_fraction"]`, with a warning when many objects lose most of it
  (DECISIONS #18).

- **Inputs.** `Bins` (linear, log, from centres, N-D grids, categories) and
  `Likelihoods` with constructors for:
  - likelihood matrices (density or mass representation, dense or sparse,
    log-space input);
  - Gaussian measurements;
  - per-object posteriors with their interim prior;
  - posterior samples;
  - SciPy distributions and arbitrary likelihood functions;
  - response matrices (unfolding) and classifier outputs (quantification).
- **Selection.** `Selection` functions with explicit `"observed"` and `"true"`
  modes, attached to the data or passed to the model.
- **Model.** `PopulationModel` for bin fractions or Poisson rates, with
  log-density and gradient in unconstrained coordinates, expected counts,
  responsibilities, an optimality (KKT) gap and the missing-information fraction.
- **Priors.**
  - Simplex: `Dirichlet` (with reference, count-based and base-measure
    constructors), `GeneralizedDirichlet`, `DirichletTree`, `PolyaTree`.
  - Smoothness: `LogisticNormal`, `LogisticRandomWalk` and `LogisticGP`
    (Hilbert-space GP) on N-D grids, with inferred hyperparameters.
  - Rates: `GammaRates`.
- **Engines.**
  - `NUTS` (default): multinomial NUTS with windowed metric adaptation.
    Smoothness priors get partial non-centring (where the data constrain
    their scales) and a curvature-based dense metric. Chains are resumable
    and can run in parallel processes, which share the cores' BLAS threads
    (via `threadpoolctl`) rather than oversubscribing them.
  - Data-augmentation `Gibbs` for conjugate priors, with selection augmentation.
  - `MAP` with an optimality certificate: mix-SQP for Dirichlet priors
    (including maximum likelihood under selection), SQUAREM-accelerated EM
    for the other conjugate priors, L-BFGS otherwise.
  - `Laplace` with PSIS importance correction.
  - `MeanFieldVB` (flagged as approximate) and `Bootstrap` (nonparametric or
    Bayesian, over objects, with variable totals for rate models).
- **Interfaces.** `binference.infer` and the scikit-learn-style `PopulationEstimator`.
- **Results.** `PopulationResult` and `PointEstimate` provide:
  - summaries, intervals (central and HDI), functionals and rebinning;
  - per-object assignment probabilities, values and realised counts;
  - posterior predictive scores, diagnostics and checks with hints;
  - plots, save/load and xarray `DataTree` export.
- **Diagnostics.** Rank-normalised split R-hat, bulk and tail ESS,
  autocorrelation time, MCSE and PSIS.
- **Validation.** Exact quadrature posteriors (K <= 3), simulation-based
  calibration and coverage studies. Mock catalogues in 1-D and 2-D, plus
  categorical, unfolding and prior-predictive mocks.
- **Design documents, tests, docs.**
  - Specification, test plan, implementation plan, and a survey of related
    software, including an audit of the `frankenz` samplers.
  - A fast and a slow (statistical) test suite.
  - Sphinx documentation with tutorials.
