# Contributing to binference

Thanks for helping. binference aims to be small, correct and pleasant to use. The
bar for a change is that it keeps the posterior exact (or clearly labels an
approximation), comes with tests, and reads like the code around it.

## Development set-up

```bash
git clone https://github.com/joshspeagle/binference
cd binference
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install          # ruff lint + format on every commit
pre-commit run --all-files  # the same checks as CI's lint job
```

## Tests

```bash
pytest                      # fast suite (about 80 s); warnings are errors
pytest -m slow              # statistical validation: SBC, coverage, Geweke (about 20 min)
pytest --cov=binference         # coverage report
```

* `BINFERENCE_TEST_SEEDS="0,1,2" pytest` repeats the seeded tests over several seeds.
* Every engine is checked against exact posteriors from `binference.validation.exact_posterior`.
  A new engine or prior needs the same: an exact or conjugate reference in the
  fast suite and, for new samplers, an SBC or Geweke test in `tests/test_slow.py`.
* Statistical tolerances are set from Monte Carlo standard errors (see
  `tests/conftest.py::assert_matches_reference`), never by tuning seeds until a
  test passes.

## Style

* `ruff check` and `ruff format` (line length 99) run through the pre-commit hooks
  (`pre-commit run --all-files`), locally and in CI.
* Use the notation of the [notation page](docs/reference/notation.md) in code, docstrings
  and docs:
  * $L_{ik}$ for bin-averaged likelihoods;
  * $\rho$ for fractions and $\lambda$ for rates;
  * $\pi$ for interim priors and $r_{ik}$ for responsibilities.
* In public signatures, `value` is fractions or rates, `eta` logits or log-rates and `x` the
  unconstrained coordinates. Public densities without a qualifier (`log_prior`,
  `log_likelihood`, `log_posterior`) are in `value`; `log_density_unconstrained` is in `x`
  ([coordinates](docs/how-it-works/priors.md#coordinates)).
* Randomness goes through a `rng` argument accepting a seed or
  `numpy.random.Generator` (SPEC 7), never the global NumPy state.
* User-facing problems raise `ValueError`/`TypeError` with a message that says
  how to fix them. Questionable but legal inputs emit an `InputWarning`,
  unconverged fits a `ConvergenceWarning`, and approximate engines an
  `ApproximationWarning`.
* Docstrings follow the NumPy convention and render in the API reference.

## How the code fits together

* `infer` (`inference.py`) builds a `PopulationModel` (`model.py`) from the likelihoods,
  the prior bound to the bins and any selection, chooses an engine class from `METHODS`
  (or `Auto`), and runs it.
* `Likelihoods` (`likelihoods.py`) stores $L$ with each row scaled to a largest value of
  one and the row's log scale kept separately. Its `from_*` constructors read the inputs;
  `bins.py`, `measure.py` and `basis.py` describe the bins, `density.py` interim priors,
  `selection.py` and `response.py` the other inputs.
* `PopulationModel` owns the log density and its gradient, through the kernels in
  `_kernels.py`.
* Engines ask the bound prior what it supports, never its class: `conjugate` (Gibbs, EM,
  VB), `has_density` (MAP on the simplex and the KKT gap) and `kind` (fractions or
  rates).
* An engine returns a `PopulationResult` (draws) or a `PointEstimate` (`results.py`).
  Both keep the model, from which they compute per-object outputs, checks, plots
  (`plotting.py`) and exports (`io.py`).

## Extending binference

* **Priors** subclass `binference.priors.PopulationPrior` (the user-facing
  specification) and return a `BoundPrior` from `bind(bins)`. A bound prior
  maps unconstrained coordinates to log-fractions or log-rates (`forward`) and
  returns the log prior and its gradient (`log_prior_and_grad`). Conjugate
  priors also implement `posterior_sample`, `map_from_counts` and the `vb_*`
  hooks, and so work with Gibbs, EM and VB.
* **Engines** subclass `binference.engines.base.Engine`; register user-facing ones in
  `binference.inference.METHODS`.
* **Likelihood constructors** are `Likelihoods.from_*` class methods that
  return bin-averaged likelihoods. Inputs that carry an interim prior must
  require it explicitly.

## Documentation

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
# the code of the README, installation page and guides
python docs/_scripts/run_snippets.py README.md docs/index.md docs/start/install.md docs/guides/*.md docs/guides/inputs/*.md
# external links and the bibliography's DOIs (CI runs these when started by hand)
BINFERENCE_DOCS_EXECUTE=off sphinx-build -b linkcheck docs docs/_build/linkcheck
python docs/_scripts/check_dois.py
```

The tutorials in `docs/tutorials/` are MyST notebooks executed at build time,
so keep each under a minute or two.

The interactive explainers (V-1, V-3, …) are JavaScript modules in
`docs/_static/explainers/`, one per explainer, each with `controls`, `compute`, `readout`
and `draw`; a page shows one with ```` ```{explainer} v1 ```` and its caption, and cites it
as ``{ref}`explainer-v1` ``. After
changing a module, redraw the still frames the page shows before JavaScript runs, and
check the explainers' numbers against binference (both need Node.js 22, the version CI uses):

```bash
node docs/_scripts/explainers.mjs
pytest tests/test_explainers.py
```

## Issues and pull requests

* Every change starts from an issue that states its scope and how we will know it is
  done; bugs and changes have templates.
* A pull request closes its issue and follows the template: scope, checks, review
  outcome and records. Keep it focused and add an entry to `CHANGELOG.md` under
  *Unreleased*.
* Describe how you validated a statistical change: exact comparisons, SBC, coverage.
* Pull requests merge with a merge commit once CI is green. History is never rewritten:
  bring `main` into a branch with a merge, not a rebase.
* CI is defined in `.github/workflows/ci.yml`; label a pull request `slow` to run the
  slow statistical suite on it. Pushes to `main` publish the docs to GitHub Pages.

## Breaking changes

From 0.3.0 on, a release changes the public API (the names in each `__all__` and their
arguments) in these ways only:

- A change that breaks a call that worked before goes into a minor release (`0.4.0`),
  never a patch release (`0.3.1`).
- A renamed or removed public name or argument keeps its old spelling working for at least
  one minor release, with a `FutureWarning` that names the replacement.
- Every breaking change has a CHANGELOG entry starting **Breaking:** that says how to
  migrate: the argument value that restores the old behaviour, or why there is none.
- Saved results from the previous minor release still load.

Options after a function's primary arguments are keyword-only, and so is `rng`.
`tests/test_api.py` checks `rng` on every public callable and the other options on the
functions reviewed for 0.3.0; give a new function the same shape.

After 1.0, the same rules apply with major releases in place of minor ones.

## Releasing

Releases go to PyPI from `.github/workflows/release.yml` through trusted publishing; no
token is stored anywhere. The publisher on PyPI names that file and the environment
`pypi`. The same workflow then creates the GitHub release, with the version's CHANGELOG
section as notes and the wheel and sdist attached.

1. In a pull request, set the version to `X.Y.Z` in `src/binference/_version.py` and in
   `CITATION.cff` (with `date-released`), and turn *Unreleased* in `CHANGELOG.md` into
   `## [X.Y.Z] - YYYY-MM-DD`.
2. Merge the pull request, then tag the merge commit on `main` and push the tag:

   ```bash
   git tag -a vX.Y.Z -m "binference X.Y.Z" && git push origin vX.Y.Z
   ```

   Push the tag once CI on the merge commit has passed: a tag pushed earlier fails the
   step "Check the tag against the version, main and CI", which lists what is checked
   before anything is uploaded, and its run can be re-run once CI is green.
3. In the next pull request, set the version to the next `.dev0` and open a new
   *Unreleased* section.
