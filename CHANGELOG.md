# Changelog

All notable changes to pastry are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (with the usual 0.x caveat that minor
releases may break the API).

## [Unreleased] - 0.1.0.dev0

First development version.

### Changed (landing on `main`)

- Python ≥ 3.13, NumPy ≥ 2.1, SciPy ≥ 1.14.1, matplotlib ≥ 3.9.2 (DECISIONS #41, #61).
- CI runs the pre-commit hooks, the fast suite on three systems with Python 3.13 and
  3.14, minimum dependencies, the README and user-guide code, the docs build and a
  clean-wheel import; pushes to `main` publish the docs to GitHub Pages.
- Development process: issue and pull-request templates and the rules in `CLAUDE.md` and
  `CONTRIBUTING.md`.

### Added

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
- **Interfaces.** `pastry.infer` and the scikit-learn-style `PopulationEstimator`.
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
