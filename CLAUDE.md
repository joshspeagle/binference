# CLAUDE.md

binference (developed as `pastry` until the rename, DECISIONS #45) infers a population
distribution over bins, and each object's posterior under it, from noisy per-object
likelihoods, posteriors, samples, binned measurements or classifier outputs.

## Start here

- The walkthrough with Josh lives in `notes/review/`: read its `README.md`, then
  `PLAN.md` (current position), the last entries of `LOG.md`, and `QUESTIONS.md`.
- Every settled choice is in `notes/review/DECISIONS.md`; the landing of the v0.1 branch
  on `main` follows `notes/review/landing-plan.md` (tracking issue #4).
- Design background: `docs/design/spec.md`, `docs/how-it-works/theory.md`.

## Layout

| Path | What it is |
|---|---|
| `src/binference/` | the package |
| `tests/` | the pytest suite; `slow` marks statistical tests |
| `docs/` | the Sphinx site in MyST Markdown; `docs/tutorials/` are executed notebooks; `docs/design/` holds the spec and plans |
| `benchmarks/` | timing scripts, run by hand |
| `notes/review/` | walkthrough plan, log, decisions, questions, briefings, experiments, audit reports; not part of the site |
| `research_notes/`, `reports/` | the landscape census behind the design |

## Commands

Set-up, tests, lint (`pre-commit run --all-files`), docs and the snippet runner are in
`CONTRIBUTING.md`; CI runs the same commands (`.github/workflows/ci.yml`). Fast tests take
about 80 s, the slow suite about 20 min on one thread, a clean docs build about 9 min. Use
`OPENBLAS_NUM_THREADS=1` for timings in shared containers.

## Recording

- A decision is a choice that another competent developer could make differently and
  that affects behaviour, the public API, a number, a default, a dependency or the
  process. Record it in `notes/review/DECISIONS.md` with its reason. Never choose silently.
- A question only Josh can answer goes to `notes/review/QUESTIONS.md` with a recommended
  default; work continues on the default.
- Append to `notes/review/LOG.md` and update `notes/review/PLAN.md` as work lands.

## Issues and pull requests

All code reaches `main` through a pull request that closes an issue (DECISIONS #58).

1. **Issue first.** Scope, the decisions (#n) and audit findings it resolves, acceptance
   criteria. Work that is deferred or undecided is an issue labelled `deferred`.
2. **Branch** `land/NN-topic` (or `fix/…`, `feat/…` after the landing), from `main` or
   from the previous pull request's branch when they stack. Commits each make one
   logical change and pass every test that passed before them.
3. **Check locally**: fast tests, lint, format, docs build when docs change, and the slow
   tests of changed modules.
4. **Open the pull request** with the template: Closes #n, Scope, Checks, Reviews, Records.
   A stacked pull request targets the previous branch and is retargeted to `main` when
   that merges.
5. **Review**: `/simplify`, then `/code-review`, on the pull request's diff, sized by lines
   added plus removed; a change to numerics or the public API takes the next size.

   | Lines | `/simplify` | `/code-review` |
   |---|---|---|
   | under 100 | one reviewer for all angles | `low` |
   | 100 to 500 | two reviewers | `medium` |
   | over 500 | four reviewers, one per angle | `high` |

   Fix every valid finding; answer the others in the pull request's Reviews section.
   Review reports are not committed (DECISIONS #60).
6. **Merge** with a merge commit once CI is green; Claude merges (DECISIONS #58). Then the
   log entry and plan update.

## Git

- Never force-push, rebase or amend a pushed commit. Bring `main` into a branch with a
  merge commit.
- Commit subject in the imperative, at most 72 characters; a body only for why.

## Code

- Minimum code that is correct, clear and fast; removing beats adding. Each concept is
  implemented once (DECISIONS #57).
- Validate public inputs early, with errors that name the argument and say what was
  expected. Internal functions trust their callers.
- Public functions never modify their inputs. Randomness only from the caller's `rng`
  (SPEC 7). float64 unless a decision says otherwise.
- Expose every tunable with a sensible default (DECISIONS #16).
- Type hints on public signatures; NumPy-style docstrings in the notation of
  `docs/how-it-works/theory.md`.

## Tests

- Hermetic and seeded. Tolerances from Monte Carlo error, never from trying seeds.
- A test earns its place by catching a plausible bug that no other test catches; each bug
  fix adds a test that fails without it. Name tests as sentences of behaviour.
- Mark a test `slow` when it needs many replicates or takes more than 5 s.

## Docs

- Docs state what the code does now (except `docs/design/`). Each fact has one home.
- Tutorials are MyST notebooks executed at build time; keep each under two minutes.
- State only measured figures, with the setup that produced them.

## Agents

- Subagents get self-contained briefs and never commit. Read every line a subagent writes
  before committing it. Parallel writers get separate worktrees.

## Writing

- Plain and exact; literal over metaphor. Replies lead with the answer, then the reason,
  then the next step.
