# Landing plan: from the v0.1 branch to a tracked package on `main` (draft, 2026-10-25)

Josh's brief (2026-10-25): the repository is now `joshspeagle/binference` (default
branch `main`, docs deployed by GitHub Actions). Convert the work so far into the
package on `main`, tracked by well-specified issues and resolved by substantive pull
requests, each followed by `/simplify` then `/code-review` and fixes, merged
sequentially (stacked where useful). At the end every decision so far and every audit
finding is resolved on `main`; deferred items remain as open issues. Walkthrough
explainers are not issues; they stay in `notes/review/`. Model: telescopic's landing
(tracking issue #1, phase issues, PRs #21–#60, DECISIONS #58 there).

## Structure on GitHub

- **Tracking issue**: "Bring binference v0.1 and the walkthrough decisions to main",
  listing phases as sub-issues.
- **Phase issues** (sub-issues of the tracker); each lists its work issues.
- **Work issues**: one per pull request. Body: Scope, Decisions (#n), Audit findings
  (IDs), Acceptance criteria, Records. Labels: `phase:*`, `area:*`, `type:{bug,design,
  refactor,docs,infra}`, `deferred`.
- **Pull requests**: one per work issue, "Closes #n". Body: Scope, Commits, Checks,
  Reviews (simplify + code-review outcome), Records. Stacked: base = previous PR's
  branch, retargeted to `main` when that merges. Merge commits (keep hashes notes cite).

## Per-PR loop

1. Issue written and linked before code.
2. Branch `land/NN-topic` from the previous PR's branch; commits each one logical change.
3. Fast tests, ruff, format, docs build (and slow tests of changed modules) pass locally.
4. Open the PR; CI runs.
5. `/simplify`, then `/code-review`, on the PR diff, sized by lines changed
   (<100: one reviewer, low; 100–500: two, medium; >500: four, high; numerics or public
   API go one size up). Fix valid findings; answer the rest in `notes/pr-reviews/NN.md`.
6. CI green, then merge (who merges: question 2), log entry, plan updated.

Independent PRs may be implemented in parallel worktrees; merges stay in order.

## Phase A: land the history (no new code)

The v0.1 code, docs, research and walkthrough notes enter `main` as history-preserving
PRs (merge commits keep the hashes the notes cite). CI never ran on the branch (it only
triggers on `main` and PRs), so each landing PR may need a small "make CI green" commit.
The independent review and the background audit stand in for per-PR reviews here; their
findings are fixed in Phase C.

| PR | Commits | Content |
|---|---|---|
| A1 | 029b2fe..dd4a4dc | landscape census, research notes, spec, test and implementation plans |
| A2 | e1a0ae0..c76e60e | v0.1 core package, engines, results, tests, infra (CI, pre-commit, README) |
| A3 | 9100566..73d1930 | docs, tutorials and the fixes of the first independent review |
| A4 | 8efa643..HEAD | walkthrough notes, briefings, audit reports, this plan |

## Phase B: foundations

| PR | Content | Decisions / findings |
|---|---|---|
| B1 | Process and infrastructure: CLAUDE.md rules for issues, branches, PRs and reviews; CONTRIBUTING; issue and PR templates; labels; CI (Python ≥ 3.13 matrix, pinned ruff, timeouts, dist job, README/user-guide code execution, slow suite on demand); docs build and Pages deploy from Actions; mypy if Josh wants it | #41 (Python), T-8, T-14, T-15, T-16, T-20 |
| B2 | Rename pastry → binference: package, imports, docs, citation, URLs, notes references | #45 |

## Phase C: decisions and audit, stacked by dependency

| PR | Module | Decisions | Audit |
|---|---|---|---|
| C1 | Bins: measure (transform or density), masks, arbitrary edges | #17, #19 (Bins, Dirichlet), #21 (Bins and inputs), #22 | I-12, I-16, I-22 |
| C2 | Density class (interim prior, base measure), PopulationPrior rename, stacking made explicit | #11, #24, #26 | I-1, I-5, I-6, I-13, I-17 |
| C3 | Likelihood constructors: names by input form, kind/interim rule, adaptive quadrature, outside-support mass, weights policy | #1, #18, #22, #23, #32 | I-2–I-4, I-7–I-11, I-14, I-15, I-18, I-19, I-21 |
| C4 | Samples: one Monte Carlo estimator, pointwise interim, N/S check, Dirichlet floor | #31 | I-23 |
| C5 | Binned measurements: Response object, from_binned, classifier routes and their record, label-shift check | #27, #29 | – |
| C6 | Selection: names, absolute/relative, combined (per-bin), callables, steep-S warning; mocks for truth, measurement and both | #7, #8, #10 (arrays, callables), #46, #47 (warning), #48 | G-4, G-10, G-11 |
| C7 | Model and kernels: KKT fixes, missing information, zero-weight rows, each concept once, Jacobian naming | #15, #56, #57 | M-1–M-4, M-7, M-10–M-12, M-14 |
| C8 | MAP / EM / mix-SQP fixes | – | M-5, M-6, M-8, M-9, M-13, T-4 |
| C9 | Priors fixes (GP in the measure coordinate, K = 2, trees, validation) | #21 (GP) | P-1–P-22 |
| C10 | NUTS and Laplace fixes | – | N-1–N-15 |
| C11 | Gibbs, VB, bootstrap, validation fixes | – | G-1–G-3, G-5–G-9, G-12 |
| C12 | Results and outputs: selected population, derived total and rates, full-sample counts, value draws from within-bin information, KKT in check(), data-support flags, prior-vs-data ratio | #9 (derived rates), #12, #41, #49, #50, #53 | O-1–O-21 |
| C13 | Tests: stronger slow tests, SBC on selection/rates/GP, coverage gaps | – | T-1–T-3, T-9–T-11, T-13 |
| C14 | Docs: vision and principles, field-neutral README and landing page, user guide and tutorials updated for every change, spec and test plan refreshed, CHANGELOG | #33, #34, #36, #37 | T-5–T-7, T-12, T-17–T-19, I-20, P-21, N-15, O-21 |

Then a phase-wide `/simplify` and `/code-review` on Phase C's whole diff (C15) and a
release-candidate check (all tests, slow suite, docs).

## Phase D: deferred work, as open issues

Each becomes a well-specified issue labelled `deferred`, linked from the tracker:

- Needs Phases 3–5 of the walkthrough: B-spline basis and internal resolution (#4, #5),
  binning-bias diagnostic and planner (#6, #20), adaptive defaults (#14), measure in RW
  and tree priors (#21), masked smoothness priors (#19), remove the rates kind and the
  explicit-total option (#9), selection draws, injections and joint inference (#10, #30),
  response inferred jointly (#28), within-bin selection at internal resolution (#47),
  one result object with draws + best fit and `_best` names (#35, #51), leave-one-out
  (#52), correlated Gaussians (#2, #25), speed plan and float32 (#54, #55), benchmark
  matrix (#39, #42), golden-value tests (#41), survey into docs (#43), per-field
  tutorials (#37).
- Post-v1 (`post-v1.md`): PDF calibration, implicit interim priors, recalibration (#38),
  overdispersion (#40), chunked inputs, GPU, checkpoints, exports (#41).
- Parked questions (`QUESTIONS.md`): Bins.neighbors, N-D Pólya trees, small
  concentrations, tree-matched coordinates, how priors use bins, the 1.3 architecture
  decisions.

## Questions for Josh

1. History: land v0.1 and the notes as the four history-preserving PRs of Phase A, or
   one squashed PR?
2. Merging: Claude merges each PR when CI is green and reviews are addressed, or Josh
   approves each?
3. Scope split: confirm that items needing the Phase 3–5 walkthrough are issues, not
   implemented now (notably #9 removing the rates kind, #10/#28/#30 joint inference,
   #54/#55 speed).
4. Python ≥ 3.13 (and current NumPy/SciPy floors); add mypy?
5. PR review reports in `notes/pr-reviews/` (the walkthrough keeps `notes/review/`).
