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
  P(measured bin, observed | true bin) with real truth bins, from a matrix, simulated
  events (`Response.from_simulation`) or a labelled validation set
  (`Response.from_labelled`: score bins or a confusion matrix).
  `Likelihoods.from_binned` replaces `from_response`; `from_classifier` is removed
  (route A is `from_table` on categories, route C is `from_binned` with a labelled
  response). `summary()` names the inputs, and `result.check()` runs a posterior
  predictive test of the measured counts that flags label shift.
  `simulate.mock_categorical` gains `temperature` and a labelled validation set;
  `simulate.mock_unfolding` returns `mock.response`.
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
