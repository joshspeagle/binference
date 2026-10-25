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
| `vision.md` | Agreed vision and design principles (step 1.1): the reference for every later decision | Edit when principles change |
| `api-ledger.md` | Every public argument: default today, proposed default, why, status (input to Phase 7) | Add rows as each module is walked through |

## Rules for a new session

1. Read `PLAN.md` (current position), the last entries of `LOG.md`, and open items in `QUESTIONS.md`.
2. Read the briefing of the current step, if one exists.
3. Do not re-open a decision recorded in `DECISIONS.md` unless Josh does.
4. Record as you go: a log entry per working session, a decision entry per decision, a commit per settled change.

## Briefing pages: requirements

- **Phone first.** Josh reads on a phone. Every page must work at ~375 px:
  single column, sticky swipeable section chips, display equations short
  enough to fit (split long ones), no multi-column tables (use stacked
  cards), chart labels that thin out, heavy recomputation on slider release.
- Check at phone width before publishing: render inside a 375 px iframe with
  headless Chromium (`/opt/pw-browsers/chromium-*/chrome-linux/chrome
  --headless=new --screenshot`); a bare `--window-size=375` is not enough
  (headless enforces a ~485 px minimum layout width).
