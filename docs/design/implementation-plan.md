# pastry: implementation plan (v0.1)

This document describes how to build what [`spec.md`](spec.md) specifies: the code layout, the internal interfaces, the milestones and the documentation. Tests are described in [`test-plan.md`](test-plan.md); every milestone ships with its tests.

## 1. Repository layout

```
pastry/
├── pyproject.toml            # hatchling build; deps; extras; ruff, pytest and coverage config
├── README.md                 # pitch, install, 10-line quickstart, links
├── CHANGELOG.md  CONTRIBUTING.md  CITATION.cff  LICENSE
├── .pre-commit-config.yaml   # ruff (lint + format), end-of-file, trailing whitespace, check-yaml
├── .github/workflows/ci.yml  # lint, test matrix, slow tests, docs, build
├── src/pastry/
│   ├── __init__.py           # public API re-exports, __version__
│   ├── _version.py
│   ├── warnings.py           # PastryWarning, InputWarning, ConvergenceWarning, ApproximationWarning
│   ├── _utils.py             # rng handling (SPEC 7), array validation, chunking, progress
│   ├── _simplex.py           # Helmert basis, softmax / log-softmax, ILR, CLR
│   ├── _kernels.py           # dense/sparse matvecs, expected counts, log-likelihood, label sampling
│   ├── bins.py               # Bins
│   ├── likelihoods.py        # Likelihoods + constructors
│   ├── priors/
│   │   ├── __init__.py
│   │   ├── base.py           # Prior / bound-prior protocol
│   │   ├── hyper.py          # HalfNormal, LogNormal, InverseGamma, Fixed (scalar hyperpriors)
│   │   ├── dirichlet.py      # Dirichlet, GeneralizedDirichlet, DirichletTree, PolyaTree
│   │   ├── logistic.py       # LogisticNormal, LogisticRandomWalk, LogisticGP
│   │   └── rates.py          # GammaRates
│   ├── model.py              # PopulationModel: log-densities, gradients, transforms, selection
│   ├── samplers/
│   │   ├── __init__.py
│   │   ├── base.py           # Engine base class, chain bookkeeping, result assembly
│   │   ├── _nuts_core.py     # tree building, leapfrog, U-turn checks (pure functions)
│   │   ├── nuts.py           # NUTS engine with adaptation, multi-chain, resumable
│   │   ├── gibbs.py          # data-augmentation Gibbs (selection, conjugate priors)
│   │   ├── optimize.py       # MAP: EM + SQUAREM; L-BFGS for non-conjugate priors
│   │   ├── laplace.py        # Laplace + PSIS importance resampling
│   │   ├── vb.py             # mean-field VB
│   │   └── bootstrap.py      # bootstrap over objects
│   ├── results.py            # PopulationResult, PointEstimate, Summary
│   ├── diagnostics.py        # ess, rhat, autocorr_time, mcse, psis, kkt_gap, missing_information
│   ├── inference.py          # infer(), method registry
│   ├── estimator.py          # PopulationEstimator (scikit-learn style)
│   ├── simulate.py           # mock catalogues + Truth
│   ├── validation.py         # exact_posterior, sbc, coverage
│   ├── plotting.py           # matplotlib helpers (optional dependency)
│   └── io.py                 # save/load (npz, versioned schema), to_datatree
├── tests/                    # see test-plan.md
├── benchmarks/               # timing / ESS-per-second scripts (non-gating)
└── docs/                     # Sphinx + MyST-NB (see §5)
```

## 2. Internal interfaces

### 2.1 Numerical kernels (`_kernels.py`)

All functions accept dense `ndarray` or CSR `L` of shape `(N, K)` and `w` of shape `(N,)`.

| Function | Returns | Notes |
|---|---|---|
| `matvec(L, x)` | `L @ x` | BLAS or sparse |
| `rmatvec(L, y)` | `L.T @ y` | |
| `loglike_and_counts(L, w, rho)` | `(ℓ, n̂, s)` | `s = Lρ`, `n̂ = ρ ⊙ Lᵀ(w/s)`; −∞ if any `s ≤ 0` |
| `responsibilities(L, rho, idx=None, chunk=…)` | `(n, K)` | chunked |
| `sample_labels(L, rho, rng, chunk=…)` | `(N,)` int | dense: inverse-CDF on cumulative sums; CSR: global cumulative sum + `searchsorted` |
| `sample_counts(L, w_int, rho, rng)` | `(K,)` | duplicates via `rng.multinomial(m_i, r_i)`; otherwise `bincount(labels)` |

### 2.2 Priors (`priors/base.py`)

A user-facing prior is a lightweight, picklable configuration object. `PopulationModel` calls `prior.bind(bins)`, which validates the prior against the bins and precomputes bases. It returns a **bound prior** that implements:

```text
kind: "fractions" | "rates"
dim: int                         # unconstrained dimension
conjugate: bool
forward(x) -> (eta, cache)       # softmax logits (fractions) or log-rates (rates)
log_prior_and_grad(x, cache, d_eta) -> (lp, grad_x)
                                 # lp includes the Jacobian to unconstrained space;
                                 # grad_x = J^T d_eta + grad lp (chain rule through forward)
init_from_rho(rho) -> x          # starting point (least squares for logistic priors)
sample(n, rng) -> rho            # prior draws on the simplex (or rates)
log_density_simplex(rho)         # simplex priors only (for reporting / quadrature tests)
prior_transform(u) -> rho        # where supported (nested sampling)
# conjugate priors only:
posterior_sample(counts, rng, exposure=None) -> rho or lambda
map_from_counts(nhat, exposure=None) -> rho or lambda
vb_update(nhat) -> state ; vb_expected_log(state) -> E[log rho]
```

The likelihood side always hands the prior a gradient with respect to `eta`. The prior applies the chain rule. This keeps `model.py` independent of the prior family.

### 2.3 Model (`model.py`)

```text
PopulationModel(likelihoods, prior=None, *, selection=None)
  .K .N .W .dim .kind .bins .likelihoods .prior (bound) .selection
  .log_prob(x) / .log_prob_and_grad(x)          # unconstrained, exact
  .to_rho(x) / .to_rates(x) / .from_rho(rho)
  .log_likelihood(rho_or_rates)                  # includes selection normaliser
  .grad_eta(rho_or_rates) -> d_eta               # softmax: n̂ - Wρ S/(S·ρ); rates: n̂ - S λ
  .log_prior(rho)                                # simplex priors
  .prior_transform(u)
  .responsibilities(rho, idx=None)
  .missing_information(rho)
```

### 2.4 Engines (`samplers/`)

```text
Engine(model, *, rng=None, progress=False, **config)
  .run(n_samples=None, **overrides) -> PopulationResult | PointEstimate
  .sample(n_samples) -> generator of per-iteration draws (NUTS/Gibbs)
  .state                                          # resumable chain state
```

- **NUTS**
  - Round-robin over `n_chains`. Each chain carries its own position, step size, metric and adaptation state.
  - The tree-building core is pure functions taking `(logp_grad_fn, metric)`.
  - Adaptation follows Stan's windows; metric `"diag"` or `"dense"`.
  - Chains are initialised from the MAP plus jitter, or from a user-supplied `init`.
- **Gibbs**
  - Requires `prior.conjugate`.
  - Each iteration: label step, then optional selection augmentation (negative binomial), then `posterior_sample`.
  - Optional stored labels (`store_labels=True` or an index array).
- **MAP**
  - Conjugate priors: SQUAREM-accelerated EM on ρ (or λ). The M-step uses `map_from_counts`; selection adds the expected number of missed objects.
  - Otherwise: L-BFGS-B on `-log_prob` in unconstrained coordinates, with an analytic gradient.
  - Reports the KKT gap.
- **Laplace**
  - Finds the mode in unconstrained coordinates by L-BFGS, warm-started from MAP/EM.
  - The Hessian is analytic for the ILR Dirichlet; otherwise symmetrised central differences of the gradient.
  - Draws from the Gaussian, then optionally applies PSIS importance resampling and records k̂.
- **VB**: CAVI over the conjugate prior's variational state.
- **Bootstrap**: reweights objects (multinomial or Dirichlet weights) and calls MAP repeatedly.

### 2.5 Results (`results.py`)

`PopulationResult` stores:
- draws in unconstrained space, together with the transformed `samples` (fractions) and `rates`;
- the chain layout, `log_prob`, `method`, `approximate` and `info`;
- a reference to the model, used for individual-object methods.

Derived quantities are computed lazily and cached. `PointEstimate` holds the MAP outputs. Both use the `Summary` table class for `summary()`.

## 3. Milestones (each lands with its tests and docs stubs)

| M | Content | Exit criteria |
|---|---|---|
| M0 | Scaffolding: pyproject, layout, warnings, utilities, CI workflow, pre-commit, docs skeleton | `pip install -e .[test]` works; `pytest` runs; ruff clean |
| M1 | `Bins`, `Likelihoods` (all constructors), `_simplex`, `_kernels` | tests 2.1–2.3 green |
| M2 | Priors: Dirichlet family, logistic family, rates, hyperpriors | test 2.4 green |
| M3 | `PopulationModel` (gradients, selection, transforms) | test 2.5 green (finite-difference gradients for every family) |
| M4 | Engines: MAP, NUTS, Gibbs, Laplace, VB, Bootstrap | tests 2.6 green; prototype benchmarks reproduced |
| M5 | Results, diagnostics, `infer`, estimator, io | tests 2.7–2.9 green |
| M6 | `simulate`, `validation` (exact posterior, SBC, coverage) | tests 2.10 green; slow 2.11 green locally |
| M7 | Plotting | smoke tests (Agg backend) |
| M8 | Docs: theory, user guide, tutorials (executed), API reference, README | docs build with executed notebooks, no warnings |
| M9 | Independent review, fixes, benchmarks, polish | review findings resolved; full suite and docs green |

## 4. Engineering conventions

- **Style**
  - NumPy-style docstrings with `Examples`, cross-referencing equations in `theory.md`.
  - ruff formatting with line length 99.
  - Type hints on public functions, with `from __future__ import annotations`.
- **Numerics**
  - Never form `N × K` temporaries outside chunked loops in sampler hot paths.
  - Guard `log(0)`.
  - Use `float64` throughout.
  - Draw all randomness from an explicit `Generator`; chains get independent children via `rng.spawn(n)`.
- **Errors and warnings**
  - Validate public inputs early with actionable messages.
  - Engines never swallow non-finite log-probabilities. They raise with the offending state summarised, except inside NUTS, where they count as divergences.
- **Dependencies**: core `numpy>=1.25` and `scipy>=1.11`. matplotlib, xarray, pandas and tqdm are imported lazily inside the functions that need them.

## 5. Documentation plan

- **Toolchain:** Sphinx, MyST-NB, pydata-sphinx-theme, numpydoc, sphinx-copybutton and sphinx-design.
- **Tutorials** are MyST Markdown notebooks under `docs/tutorials/`, executed at build time with caching. Each runs in under 2 minutes on a laptop CPU.
- **Pages:**
  1. *Getting started*: install, then the quickstart from the README, expanded.
  2. *User guide*:
     - inputs and conventions (likelihoods vs posteriors, interim priors, masses vs densities, samples, adapters);
     - priors;
     - selection effects and rates;
     - inference engines and how to choose one;
     - results (population vs realised counts, individual objects, functionals);
     - diagnostics and validation;
     - performance;
     - pitfalls (stacking, interim priors, unequal bins, VB under-dispersion, binning choice).
  3. *Theory*: the normative maths from spec §3, in clean notation, with derivations (EM as iterated stacking, the selection augmentation, tree densities).
  4. *Tutorials*:
     1. Quickstart: n(z) from Gaussian photo-z likelihoods, comparing stacking and pastry, with individual shrinkage.
     2. Inputs and conventions: posteriors with interim priors, samples, unequal bins (wrong vs right).
     3. Choosing an engine: NUTS vs Gibbs vs Laplace vs VB vs MAP, accuracy against speed, with diagnostics.
     4. Priors and smoothness: Dirichlet α, generalised Dirichlet, Pólya tree, random-walk and GP priors with hyperparameter inference.
     5. Selection effects and rates.
     6. Categories and unfolding: classifier quantification and a response-matrix example.
     7. Validating your setup with SBC and coverage.
     8. *(should)* A two-dimensional CMD-like population.
  5. *API reference* (autosummary).
  6. *Design notes*: spec, plans, landscape report.
  7. Changelog, contributing, citing.

## 6. CI (GitHub Actions)

| Job | Contents |
|---|---|
| `lint` | `ruff check`, `ruff format --check` |
| `test` | ubuntu and macOS × Python 3.11–3.14; `pip install -e .[test]`; `pytest -q` (fast tier); coverage report on one cell |
| `slow` | ubuntu, Python 3.13: `pytest -m slow` |
| `docs` | `pip install -e .[docs]`; `sphinx-build -W -b html docs docs/_build/html` (executes the tutorials) |
| `build` | `python -m build`; install the wheel in a fresh virtualenv; `python -c "import pastry"` |

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| NUTS bugs, which are subtle | Port the validated prototype; test it on Gaussian, Dirichlet and quadrature targets; add SBC and cross-engine tests |
| Slow mixing for broad PDFs or many bins | Document it; prefer NUTS; offer smoothness priors; report the missing-information fraction |
| Python overhead in NUTS for small N | Acceptable for v0.1; a JAX backend is on the roadmap |
| Memory for large dense L | Sparse storage, `compress()`, chunked kernels; the docs give sizing guidance |
| Ambiguous user inputs | Explicit representation and interim-prior arguments, plus sanity warnings |
