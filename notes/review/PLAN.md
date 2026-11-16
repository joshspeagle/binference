# Plan

**Current position:** 0.2.0 is released (#223). The next release is 0.3.0, tracked in #225 (DECISIONS #134). The walkthrough's phases and the teardown (DECISIONS #106) are finished; their record is `LOG.md` and `DECISIONS.md`, and their material is archived at commit `3c12aa3` by #227 (DECISIONS #135).

## Stages to 0.3.0

Each item is an issue labelled `0.3.0` and a sub-issue of #225. Stages 1 and 2 can run side by side; stage 4 follows the others.

| Stage | Issue | Item | Status |
|---|---|---|---|
| 0 Record the plan | #226 | This plan, the split of #221, the questions cross-referenced | done (#240) |
| 1 Clean-up (no behaviour change) | #227 | Fold the walkthrough notes into the docs, then archive them | done (#242, #243, #244) |
| | #228 | A contributor set-up that works outside Josh's machine | done (#241) |
| | #229 | Tests grouped by module, not by the review that added them | done (#245) |
| | #230 | Split `results.py` along its seams, if the split is clean (optional) | done (#246): no split, value drawing moved out |
| | #231 | Public-repository hygiene: conduct, security, DOI, the old name | done (#247) |
| 2 API and defaults | #204 | Derived totals under a flat Dirichlet | done (#252) |
| | #232 | Every prior takes its geometry from the bins' measure | done (#251) |
| | #233 | Samples constructor: kind and smoothing | done (#253) |
| | #234 | Selection draws: parameters only, or averaging as an opt-in | done (#249) |
| | #235 | `GeneralizedDirichlet` from a target mean; Pólya tree split points | done (#250) |
| 3 Samplers | #222 | NUTS on `LogisticGP` with large grids; ships if its acceptance is met, otherwise documented as a known limitation | done (#254): not met, documented; dense metric to 500 dimensions |
| 4 Release | #77 | Golden values against mixsqp, REBayes and Empirikos, tried in a cloud session first | done (#255): mixsqp and Empirikos; REBayes's problem through mixsqp |
| | #238 | Review of the public API and the policy for breaking changes | done (#256): options keyword-only, policy in `CONTRIBUTING.md`; renames in #257 |
| | #257 | Renames from the API review (no shims before 1.0) | in review |
| | #239 | Docs pass and the release | open |

## After 0.3.0

All labelled `deferred`, in the order they build on each other:

1. #218 field integrations and exports (qp and RAIL first).
2. #220 outliers, built on #61's joint-block interface: an outlier fraction is the third jointly inferred quantity #61 waits for.
3. #217 overdispersion, with the hierarchical models of #237.
4. #215 calibration and #216 the implicit interim prior, which share held-out or labelled inputs.
5. #219 scale and run control, then the JAX backend of #237.
6. #236 coordinates and metrics for NUTS, whenever the benchmark matrix shows the need (internal, no API change).
