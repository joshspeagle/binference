# Notes for Claude sessions

- We are in a step-by-step walkthrough of the whole repository with Josh.
  Before doing anything else, read `notes/review/README.md`, then `PLAN.md`
  (current position), the latest entries of `LOG.md`, and `QUESTIONS.md`.
- Record as you go: append to `notes/review/LOG.md`, add settled decisions to
  `notes/review/DECISIONS.md`, park open items in `notes/review/QUESTIONS.md`,
  and update the status table in `PLAN.md`.
- Commit settled changes with their notes; push to the working branch.
- Design background: `docs/design/spec.md` (decision log in §13), `docs/theory.md`.
- Fast tests: `pytest -q` (~80 s). Slow statistical tests: `pytest -m slow` (~7 min).
  Docs: `sphinx-build -W --keep-going -b html docs docs/_build/html` (~9 min clean).
