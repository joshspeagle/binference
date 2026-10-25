# Contributing to pastry

Thanks for helping. pastry aims to be small, correct and pleasant to use. The
bar for a change is that it keeps the posterior exact (or clearly labels an
approximation), comes with tests, and reads like the code around it.

## Development set-up

```bash
git clone https://github.com/joshspeagle/pastry
cd pastry
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install          # ruff lint + format on every commit
```

## Tests

```bash
pytest                      # fast suite (about 80 s); warnings are errors
pytest -m slow              # statistical validation: SBC, coverage, Geweke (about 10 min)
pytest --cov=pastry         # coverage report
```

* `PASTRY_TEST_SEEDS="0,1,2" pytest` repeats the seeded tests over several seeds.
* Every engine is checked against exact posteriors from `pastry.validation.exact_posterior`.
  A new engine or prior needs the same: an exact or conjugate reference in the
  fast suite and, for new samplers, an SBC or Geweke test in `tests/test_slow.py`.
* Statistical tolerances are set from Monte Carlo standard errors (see
  `tests/conftest.py::assert_matches_reference`), never by tuning seeds until a
  test passes.

## Style

* `ruff check` and `ruff format` (line length 99) are enforced in CI.
* Use the notation of the specification ([`docs/design/spec.md`](docs/design/spec.md) §3) in code,
  docstrings and docs:
  * $L_{ik}$ for bin-averaged likelihoods;
  * $\rho$ for fractions and $\lambda$ for rates;
  * $\pi$ for interim priors and $r_{ik}$ for responsibilities.
* Randomness goes through a `rng` argument accepting a seed or
  `numpy.random.Generator` (SPEC 7), never the global NumPy state.
* User-facing problems raise `ValueError`/`TypeError` with a message that says
  how to fix them. Questionable but legal inputs emit an `InputWarning`,
  unconverged fits a `ConvergenceWarning`, and approximate engines an
  `ApproximationWarning`.
* Docstrings follow the NumPy convention and render in the API reference.

## Extending pastry

* **Priors** subclass `pastry.priors.base.Prior` (the user-facing
  specification) and return a `BoundPrior` from `bind(bins)`. A bound prior
  maps unconstrained coordinates to log-fractions or log-rates (`forward`) and
  returns the log prior and its gradient (`log_prior_and_grad`). Conjugate
  priors also implement `posterior_sample`, `map_from_counts` and the `vb_*`
  hooks, and so work with Gibbs, EM and VB.
* **Engines** subclass `pastry.samplers.base.Engine`. They take a
  `PopulationModel` and return a `PopulationResult` (draws) or a
  `PointEstimate`. Register user-facing engines in `pastry.inference.METHODS`.
* **Likelihood constructors** are `Likelihoods.from_*` class methods that
  return bin-averaged likelihoods. Inputs that carry an interim prior must
  require it explicitly.

## Documentation

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The tutorials in `docs/tutorials/` are MyST notebooks executed at build time,
so keep each under a minute or two.

## Issues and pull requests

* Every change starts from an issue that states its scope and how we will know it is
  done; bugs and changes have templates.
* A pull request closes its issue and follows the template: scope, checks, review
  outcome and records. Keep it focused and add an entry to `CHANGELOG.md` under
  *Unreleased*.
* Describe how you validated a statistical change: exact comparisons, SBC, coverage.
* Pull requests merge with a merge commit once CI is green. History is never rewritten:
  bring `main` into a branch with a merge, not a rebase.
* CI runs lint (ruff, pinned to the pre-commit version), the fast suite on Linux, macOS
  and Windows with Python 3.13 and 3.14, a minimum-dependency job, the code of the README
  and user guide, the docs build with the tutorials executed, and a wheel built and
  imported in a clean environment. The slow suite runs weekly, by hand, and on pull
  requests labelled `slow`. Pushes to `main` publish the docs to GitHub Pages.
