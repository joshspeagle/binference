# Session log

Append-only. One entry per working session (or per step within a long session).

## 2026-10-22: plan agreed in outline

- Context: the v0.1 first pass is complete on `claude/pastry-v0.1` (last code
  commit before the walkthrough: see `git log`). Josh wants to go through the
  whole repository slowly, at varying depth, with copious notes that survive
  context resets.
- Created `notes/review/` (this directory) and a pointer in `CLAUDE.md`.
- Next: Josh chooses where to start and the level for the first step.

## 2026-10-22: start of Phase 1

- Josh agreed to go through the phases in the proposed order, starting with
  step 1.3 (architecture) at L2, then 1.1, 1.2, 1.4.
- Wrote `briefings/1.3-architecture.md`: end-to-end call flow, the five core
  objects, the Prior/BoundPrior and Engine contracts, the central equations,
  cross-cutting conventions, dependency graph, size map, and eight
  architecture decisions flagged for discussion.

## 2026-10-22: step 1.3 discussion, pull 2.1 forward

- Josh asked whether N-D grids lose their structure when flattened. Answer
  (checked in code): flattening is storage only (C order). `Bins` keeps
  `shape`, `edges`, `unflatten`, `neighbors`. Smoothness priors are built
  per axis: the random walk uses a Kronecker sum of per-axis RW precisions
  (eigenbasis via `_KronBasis`), the GP a tensor-product Hilbert-space basis
  with a joint D-dimensional Matérn/RBF spectral density and one length scale
  per axis (ARD). Dirichlet priors are exchangeable (no geometry by design);
  Pólya trees reject N-D grids.
- Josh wants: the details of every `from_*` conversion (Phase 2.3); a longer
  discussion of the core math; code reading alongside the briefings before
  judging the eight architecture decisions (deferred, parked in QUESTIONS).
- Pulled step 2.1 forward; wrote `briefings/2.1-core-math.md` (math + exact
  code locations).

## 2026-10-22: background audit launched

- At Josh's request, launched seven independent background auditors (review +
  simplify lenses), one per slice of the codebase. They write only to
  `notes/review/audit/<area>.md`; source is untouched. Index: `audit/README.md`.
- Findings will be triaged when the walkthrough reaches each area; outcomes
  go to DECISIONS.md. The walkthrough continues meanwhile with step 2.1.

## 2026-10-22: step 2.1 §1–2 (generative model, binning assumption, marginal posterior)

- Walked through §1–2 with code: `Likelihoods.from_gaussian` (CDF differences
  divided by bin width = the flat-within-bin average), `_normalise_linear`
  (rows scaled to max 1, log-scales kept), `PopulationModel.__init__`
  (`_ll_const`), `_kernels.loglike_and_counts`, `log_prob_and_grad`.
- Demo of the flat-within-bin approximation: population ∝ exp(−1.5 z),
  bins of width 0.5, object at d = 0.5 (on a bin edge): ratio of pastry's
  L_ik to the exact p(d | z ∈ B_k) is 1.32 / 0.79 / 0.72 for σ = 0.1 and
  1.05 … 0.78 for σ = 0.4. Errors of 20–30% per bin when the population
  changes by ×2 across a bin.
- Raised for discussion: flat in which coordinate (log bins are flat in z,
  not log z); correction options; non-integer weights.

## 2026-10-22: briefings become interactive pages

- Josh asked for briefings as interactive HTML so equations, text, diagrams and
  figures can be explored together. Convention from now on: each step gets
  `briefings/<step>.html` (published as a private artifact, links in
  `ARTIFACTS.md`) plus the `.md` record.
- Built `2.1-core-math.html`: MathJax equations, collapsible code excerpts with
  file:line, a mermaid data-flow diagram, and two explorers: (1) flat-within-bin
  vs exact bin likelihood (sliders: population slope, K, σ, d; shows per-bin
  ratios and the effect on the object's bin posterior), (2) stacking vs EM on a
  2000-object mock (iteration slider, KKT gap curve, distance to truth).

## 2026-10-22: Josh's questions on §1–2

- Flat-within-bin applies to all priors, including smooth ones (they put a
  smooth prior on per-bin log-fractions, but the likelihood is still built
  from flat-in-bin L). Key observation: the engines only need p(d_i|ρ) to be
  linear in ρ, so any fixed mixture basis works unchanged; ran an experiment
  showing linear B-splines cut the systematic bias from up to 14 s.e. to ≤ 3
  s.e. (parked in QUESTIONS).
- N-D independence: only `from_gaussian` assumes per-axis independent errors;
  other constructors accept correlated likelihoods (parked).
- Weights: Josh questions the rationale (parked with options).
- Gradients: hand-derived NumPy, verified by finite-difference tests
  (`tests/test_model.py::test_gradients_match_finite_differences`,
  `tests/test_priors.py`), no autodiff (D1).
- Audit reports received and committed: gibbs-vb-validation (12 findings),
  inputs (23, one major), outputs-api (21, one major).

## 2026-10-22: phone-friendly briefing pages

- Josh is reading on mobile. Reworked `2.1-core-math.html` (version 2): sticky
  swipeable section chips, split equations, selection modes as stacked cards,
  thinned chart labels, ratios listed as text, two sliders per row, EM
  re-simulation on slider release. Checked at 375 px in a headless-Chromium
  iframe. Requirements recorded in README.md.
- Audit report received and committed: model-map (14 findings, one major:
  zero-weight rows make EM reset or collapse under bootstrap weights).

- Audit report received and committed: nuts-laplace (15 findings, no bug-major; NUTS matches Stan; main item: results of logistic-prior fits expose the internal centred coordinates in `x`, `log_prob`, `model`).
- Audit report received and committed: priors (22 findings; core math correct; LogisticGP defaults misbehave on log bins (P-1), K > 48 (P-2), long length scales (P-3); DirichletTree per-node alpha crash (P-4)). Note: this OpenBLAS is pathologically slow multi-threaded for eigh; use OPENBLAS_NUM_THREADS=1 for experiments.

## 2026-10-22: decisions on weights, correlated errors, gradients; continuous field

- Recorded DECISIONS #1 (weights: multiplicities public, bootstrap internal,
  non-integer opt-in with warning), #2 (correlated Gaussian constructor with
  per-object covariances), #3 (NumPy gradients for now; autodiff backend on
  roadmap).
- Josh asked why the continuous field cannot be evaluated inside bins.
  Answer: it can; it amounts to separating the model's internal resolution
  from the reporting bins. Analysis parked in QUESTIONS for the Phase 3/7
  design discussion.
- Audit report received and committed: tests-docs (20 findings; slow statistical tests too weak: pass an ~18%-too-narrow posterior at the shipped seed (T-1); SBC never covers hyperparameters, GP, selection, rates, trees, Laplace (T-2); mix-SQP RuntimeWarning with mixed alpha=1/alpha>1 (T-4); several docs-vs-code contradictions). All seven audit reports are in (135 findings in total).

## 2026-10-22: step 2.1 §3–4 (selection, rates) and design decisions

- Recorded DECISIONS #4 (separate reporting bins from internal model
  resolution; continuous field evaluated inside bins), #5 (B-spline mixture
  basis), #6 (binning-bias diagnostic), at Josh's request.
- Added a selection explorer to the 2.1 page (version 3): 6000 objects,
  selection by true bin (logistic detection curve) or by a cut on the measured
  value; MLE fits with the correct mode, the other mode, and selection ignored;
  readout of distances to truth and the rates-view population size
  N_obs / (S·ρ̂). Checked with pastry that the correct-mode MLE error
  (total variation 0.07–0.14 over 8 mocks; posterior s.d. up to 0.03 per bin)
  is statistical noise, and said so on the page.

## 2026-10-22: Josh on §3–4

- #5 clarified: B-splines are for every prior family; the goal is no
  flat-within-bin assumption anywhere (#5 updated).
- Selection: decided names on_measurement / on_truth (#7) and combined
  selection with a worked example (#8). Input form (arrays vs callables)
  parked with a proposal.
- Rates: currently opt-in via the `GammaRates` prior (separate "kind").
  Derived that fractions-with-selection equals the rates model with the
  total marginalised under 1/Λ, so rates can be drawn exactly after any
  fractions fit; options A/B/C parked for the API discussion.

## 2026-10-22: selection from samples; rates afterburner

- Josh proposed selection inputs as arrays / callables / samples, with pastry fitting selection functions from injection samples. Agreed design recorded as DECISIONS #10 (incl. joint vs cut marginalisation); rates option B + C as #9. Open: default marginalisation.

## 2026-10-22: step 2.1 §5–6 (gradient, EM, stacking, KKT)

- Josh confirmed joint marginalisation as the default (DECISIONS #10).
- Page version 4: §3–4 marked discussed; §5 gains the softmax gradient derivation and cost; §6 gains why self-consistency is not optimality, the KKT code, SQUAREM / mix-SQP pointers, and the relevant audit findings (M-1–M-4, M-6). EM explorer now reports EM's measured asymptotic rate (≈ missing information) and the implied Gibbs autocorrelation time (0.998 → ~1000 iterations at the defaults).

## 2026-10-23: KKT explained; process decision on audits

- Explained the KKT condition (derivation, meaning of g_k as the rate of improvement from moving mass to bin k, sufficiency by concavity, gap as Frank–Wolfe bound ℓ* − ℓ ≤ W·gap). Demo: EM from a prior excluding bin 7 gets stuck at ρ_7 = 0 with g_7 = 1.65 (flagged); optimum has ρ_7 = 0.11, 57 nats better; bin 10 is a legitimate zero (g ≈ 0.2). Added to the page (version 5).
- Decisions #11 (stacking explicit in docs/API), #12 (KKT gap checked by check()), #13 (audit findings handled after the walkthrough).

## 2026-10-23: step 2.1 §7 (coordinates)

- Page version 6: why ILR (zero-sum subspace, orthonormal, Aitchison distance, no reference bin), Helmert code, the other families (logistic: linear in ξ, log θ Jacobian; rates: log λ; conjugate engines work on ρ directly), and a K = 3 explorer: Dir(α + n) on the simplex vs in ILR coordinates, and the tail of log ρ for an empty bin (98% span ≈ ln 99 / α), explaining NUTS tree-depth problems for small α.

## 2026-10-23: small concentrations, adaptive defaults, Jacobian convention

- Josh wants to understand the Dirichlet family (GD, Pólya trees) before
  deciding on small concentrations; prefers dynesty-style adaptive defaults.
  Recorded #14 (adaptive defaults, evidence-based, benchmark matrix in Phase
  4) and #15 (Jacobian convention explicit everywhere). Small-concentration
  decision deferred to after 3.1 (deep dive with explorers) and Phase 4.
- Benchmark `experiments/2.1_small_concentration.py` (1-D K=30 with ~1/3
  near-empty bins, N=1000; 2-D 12×12): NUTS vs Gibbs ESS/s — Dir(1) 66 vs 59;
  Dir(0.3) 8 vs 13; Dir(1/K) 0.5 vs 3.0 (NUTS 633 s, both poor); GD(1) 52 vs
  24; Pólya(c=1) 88 vs 83; Pólya(c=0.1) 22 vs 28; 2-D Dir(1) 34 vs 36. No
  NUTS tree-depth saturation except Dir(1/K) (3%). Problem is concentrated
  at very small concentrations; trees and N-D grids are not special.
