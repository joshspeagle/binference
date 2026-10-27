# Walkthrough plan

**Current position:** landing on `main` (see `landing-plan.md`): Phase C, C1–C11, C7b, C7c, C11b (derived rates, selection from injections and tallied responses inferred jointly) merged; C12 (results) merged; C13 (#72, #73) merged; C13b (#76: benchmark matrix, `method="auto"` default) merged; C14-1 (#82) and the Auto fix (#85) merged; C14-3 (#87) merged; #90 and C14-2 (#91) merged; C15 (#94: phase-wide review) merged; Phase C complete and the landing summary published; the walkthrough resumed at Phase 3 and is **paused at step 3.1** (briefing published, discussion pending) while the documentation site is designed and written (Josh, 2026-10-27). Resume at 3.1 after the docs land, and refresh the 3.1 page against the code then. Phase 1 done except 1.3's architecture decisions. Step 1.3's architecture decisions remain deferred until the relevant code has been read.

## How each step works

1. **Briefing.** Claude writes `briefings/<step>.md`: what the code does, the
   statistics or algorithm behind it, every non-obvious decision with its
   motivation and the alternatives, known weaknesses, and questions worth
   challenging. Short enough to read in a few minutes.
2. **Discussion at the depth Josh chooses** (see levels below). Questions,
   derivations, small experiments, reading code together.
3. **Outcome.** Each point ends as *keep*, *change now*, *change later* or
   *open question*, recorded in `DECISIONS.md` / `QUESTIONS.md`.
4. **Changes.** Small changes are made and committed during the step (code,
   tests and docs together). Large or cross-cutting ones, especially API and
   layout changes, are collected and done in Phase 7 so we restructure once.
5. **Close.** Log entry, plan status updated, commit.

### Levels of engagement (chosen per step, can change mid-step)

- **L1, overview.** The briefing plus discussion of the decisions. Good for
  infrastructure and parts Josh is happy to trust.
- **L2, guided tour.** Walk through the code file by file, function by function,
  with the reasoning behind each piece.
- **L3, deep dive.** Derivations, line-by-line reading, experiments to test a
  choice (e.g. re-running a validation with an alternative), comparison with
  the literature or with frankenz.

## Phases

Order is a proposal: Josh may jump around. Rough default levels in brackets.

### Phase 1. Big picture [L1–L2]
- 1.1 The problem, scope and design principles: spec §1–2, README, `docs/index.md`.
- 1.2 The landscape census and what it implied for the design: `reports/`, `docs/design/landscape.md`.
- 1.3 Architecture: module map, data flow from inputs to results, the layers of the API (spec §8.1), `docs/design/implementation-plan.md`.
- 1.4 The autonomous decision log D1–D16 (spec §13), one by one: keep, change, or revisit later.

### Phase 2. The statistical model [L3]
- 2.1 Notation and the core model: Dirichlet–multinomial hierarchy, marginal vs augmented form, EM as iterated stacking, KKT certificate (spec §3, `docs/theory.md`).
- 2.2 Bins, masses vs densities, bin-averaged likelihoods, binning bias: `bins.py`.
- 2.3 Inputs: likelihoods vs posteriors and interim priors, samples, functions, unfolding, classifiers: `likelihoods.py`, `selection.py`. **Josh wants the details of every `from_*` conversion.**
- 2.4 Selection effects: "observed" vs "true" modes, the normaliser, rates with exposure.
- 2.5 Population vs realised sample; individual-object posteriors; value draws.
- 2.6 The model object and numerical kernels: `model.py`, `_kernels.py`, `_simplex.py` (ILR coordinates, Jacobians).

### Phase 3. Priors [L2–L3]
- 3.1 Dirichlet family: Dirichlet, generalised Dirichlet, Dirichlet trees, Pólya trees; conjugacy; unbounded densities: `priors/dirichlet.py`, `priors/base.py`. **Josh needs this from first principles (GD and Pólya trees are new to him), with explorers, before deciding on small concentrations.**
- 3.2 Logistic (smoothness) family: whitened form, random-walk and HSGP bases, default mean, partial non-centring: `priors/logistic.py`.
- 3.3 Hyperpriors and defaults (e.g. GP length-scale prior): `priors/hyper.py`.
- 3.4 Rates: `priors/rates.py`.
- 3.5 Default prior choice (α = 1 vs 1/K) and prior guidance.

### Phase 4. Inference engines [L3]
- 4.1 NUTS core: multinomial NUTS, U-turn criteria, dual averaging, windowed adaptation: `samplers/_nuts_core.py`.
- 4.2 NUTS engine: initialisation, geometry adaptation and the centring rule, metric choice, resuming, parallel chains: `samplers/nuts.py`. **Josh wants to know exactly what is implemented and how much control users have over arguments and hidden constants; and to explore alternate coordinates, sampling strategies and other tricks for the complex priors (with 3.1).**
- 4.3 Gibbs and data augmentation (incl. selection via negative-binomial augmentation): `samplers/gibbs.py`, `samplers/base.py`.
- 4.4 Point estimates: EM/SQUAREM, mix-SQP, L-BFGS, the KKT gap: `samplers/optimize.py`.
- 4.5 Approximations: Laplace + PSIS, mean-field VB, bootstrap: `samplers/laplace.py`, `vb.py`, `bootstrap.py`.
- 4.6 Choosing a method; `method="auto"`; the evidence behind the defaults (spec §6.9–6.10). **Build the benchmark matrix (prior family × data regime × engine) that sets the adaptive defaults (DECISIONS #14).**

### Phase 5. Outputs and diagnostics [L2]
- 5.1 `PopulationResult` and `PointEstimate`: `results.py`.
- 5.2 Diagnostics and `check()` hints: `diagnostics.py`.
- 5.3 Validation tools and simulators: `validation.py`, `simulate.py`.
- 5.4 Plotting and I/O: `plotting.py`, `io.py`.

### Phase 6. Testing and evidence [L1–L2]
- 6.1 Test strategy: `docs/design/test-plan.md`, fast vs slow suites, exact-posterior and SBC tests, review regressions.
- 6.2 Performance: benchmarks, targets (spec §10), known limitations.

### Phase 7. API, UX and organisation [L2, Josh-led]
- 7.1 Entry points: `pastry.infer`, `PopulationEstimator`, engine classes: `inference.py`, `estimator.py`, `__init__.py`.
- 7.2 Naming, defaults, warnings and error messages; package layout.
- 7.3 Implement the restructuring collected in earlier phases (one coherent change). First build `changes.md`: every "change later" decision plus all 127 audit findings (review and simplify lenses), each triaged (adopt / superseded by a decision / reject with reason), grouped by module.

### Phase 8. Docs, tutorials and infrastructure [L1–L2]
- 8.1 User guide, theory page, tutorials: what they teach and in what order.
- 8.2 Packaging, dependencies, CI, contributing, citation, release plan (name, Python floor, publicity, JOSS).

### Phase 9. Roadmap
- 9.1 Priorities beyond v0.1: flows / continuous densities, JAX backend, hierarchical models, others (spec §12).

## Background audit

Seven auditor reports land in `audit/` (see `audit/README.md`). Per DECISIONS #13, findings are triaged after the walkthrough is complete; the mapping below is for reference: inputs → 2.2/2.3, model-map → 2.6/4.4, priors → Phase 3, nuts-laplace → 4.1/4.2/4.5, gibbs-vb-validation → 4.3/4.5/5.3, outputs-api → Phase 5/7, tests-docs → Phases 6/8.

## Status

| Step | Level | Status | Briefing | Notes |
|---|---|---|---|---|
| 0 Plan | – | done | – | Josh: follow the proposed order |
| 1.3 Architecture | L2 | overview done; decisions deferred | `briefings/1.3-architecture.md` | Josh: needs code context before judging the 8 decisions |
| 2.1 Core math (+2.6 code) | L3 | done (2026-10-23) | `briefings/2.1-core-math.html` (+ .md) | decisions #1–#15; 2.6 code review of model/kernels partly covered |
| 2.2 Bins and binning bias | L2–L3 | done (2026-10-23) | `briefings/2.2-bins.html` (+ .md) | |
| 2.3 Inputs (`from_*`) | L3 | done (decisions #22–#32) | `briefings/2.3-inputs.html`, `briefings/2.3b-redesign.html` (+ .md) | |
| 1.1 Problem, scope, principles | L1–L2 | done (decisions #33–#37; `vision.md`) | `briefings/1.1-scope.html` (+ .md) | |
| 1.2 Landscape | L1–L2 | done (decisions #38–#42; `post-v1.md`) | `briefings/1.2-landscape.html` (+ .md) | |
| 1.4 Decision log D1–D16 | L1 | done (decisions #43–#45; name → binference) | `briefings/1.4-decision-log.html` (+ .md) | |
| 2.4 Selection (code) | L2–L3 | done (decisions #46–#49) | `briefings/2.4-selection.html` (+ .md) | |
| 2.5 Population, sample, objects | L2 | done (decisions #50–#53) | `briefings/2.5-individuals.html` (+ .md) | |
| 2.6 Model and kernels | L2–L3 | done (decisions #54–#57) | `briefings/2.6-model-core.html` (+ .md) | |
| 3.1 Dirichlet family | L2–L3 | briefing published (2026-10-26); discussion pending | `briefings/3.1-dirichlet-family.html` (+ .md) | five questions in QUESTIONS |
