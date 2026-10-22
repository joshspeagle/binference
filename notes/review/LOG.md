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
