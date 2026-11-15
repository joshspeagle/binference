# Project notes

The working record of binference's design: what was decided, why, and what is still
open. These notes are not part of the documentation site. The walkthrough and teardown
that produced most of them (October 2026) are finished; their briefings, audit reports,
design drafts and the research notes behind the landscape census are kept at commit
[`3c12aa3`](https://github.com/joshspeagle/binference/tree/3c12aa3d174bc335edadceb3d4cfeed25e62c94d)
(DECISIONS #134, #135).

| File | What it holds | How it changes |
|---|---|---|
| `PLAN.md` | The current position and the stages to the next release | Updated as work lands |
| `LOG.md` | Dated log of what was done, with issues and pull requests | Append only |
| `DECISIONS.md` | Every decision, with its reason | Append; edit only the status column |
| `QUESTIONS.md` | Open questions, each with a recommended default and the issue that carries it | Items close when decided |
| `api-ledger.md` | Public arguments and their defaults, as reviewed | Rows added when an argument is reviewed |
| `experiments/` | The scripts and outputs that decisions and docs cite | Add a script with the decision that uses it |

## Rules for a new session

1. Read `PLAN.md`, the last entries of `LOG.md`, and the open items in `QUESTIONS.md`.
2. Do not re-open a decision recorded in `DECISIONS.md` unless Josh does.
3. Record as you go: a log entry per piece of work, a decision entry per decision.
