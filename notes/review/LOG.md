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
