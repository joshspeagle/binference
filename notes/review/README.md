# Code walkthrough: working notes

Josh and Claude are working through the whole repository together, top to bottom:
the big-picture decisions, the statistics, the samplers, every implementation
choice and why it was made, and the API/UX. The aim is for Josh to understand
and own all of it, and to change whatever he disagrees with along the way.

These notes are the shared memory of that process. They are written so that a
future session (after a context reset) can pick up exactly where we left off.

| File | What it holds | How it changes |
|---|---|---|
| `PLAN.md` | The phases, their steps, the files each step covers, and status | Updated when a step starts or finishes, or the plan changes |
| `LOG.md` | Dated session log: what we covered, what was discussed, what changed (with commits) | Append only |
| `DECISIONS.md` | Every decision made during the walkthrough (keep / change / defer), with rationale | Append; edit only the status column |
| `QUESTIONS.md` | Parking lot: open questions, ideas, proposed changes not yet decided | Items move to `DECISIONS.md` when settled |
| `briefings/` | One briefing per step: `<step>.html` is the interactive page Josh reads (equations, diagrams, live explorers, code excerpts; also published as a private artifact), `<step>.md` the plain-text record for future sessions | Written per step; amended with what we learned |
| `ARTIFACTS.md` | Links to the published briefing pages | Append when a page is published |

## Rules for a new session

1. Read `PLAN.md` (current position), the last entries of `LOG.md`, and open items in `QUESTIONS.md`.
2. Read the briefing of the current step, if one exists.
3. Do not re-open a decision recorded in `DECISIONS.md` unless Josh does.
4. Record as you go: a log entry per working session, a decision entry per decision, a commit per settled change.
