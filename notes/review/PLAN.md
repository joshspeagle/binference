# Plan

**Current position:** 0.2.0 is released (#223). The next release is 0.3.0, tracked in #225 (DECISIONS #134). The walkthrough's phases and the teardown (DECISIONS #106) are finished; their record is `LOG.md` and `DECISIONS.md`, and their material is archived at commit `3c12aa3` by #227 (DECISIONS #135).

## Stages to 0.3.0

The stages below track the work agreed in #225. Stage 4 follows the earlier stages.

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
| | #257 | Renames from the API review (no shims before 1.0) | done (#258) |
| | #248 | Audit suspected duplicate tests | done (#261) |
| | #218 | Document and verify qp posterior inputs | #260 |
| | #239 | Docs pass and first beta release | beta metadata in review; publication and citation checks pending |
| | #225 | Reconcile the release tracker | close after #239 and the final evidence audit |

## Deferred ideas

The scope review closed #61, #215, #216, #217, #219, #220, #236 and #237
as not planned. The closing comments record the evidence or requirements
that would justify revisiting each idea. The design specification's
roadmap preserves the N-D tree, flow, JAX and hierarchical-model proposals.
Further integrations from #218 can be scoped around a downstream use case.
