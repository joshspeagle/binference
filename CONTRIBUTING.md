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

## Extending binference

* **Priors** subclass `binference.priors.PopulationPrior` (the user-facing
  specification) and return a `BoundPrior` from `bind(bins)`. A bound prior
  maps unconstrained coordinates to log-fractions or log-rates (`forward`) and
  returns the log prior and its gradient (`log_prior_and_grad`). Conjugate
  priors also implement `posterior_sample`, `map_from_counts` and the `vb_*`
  hooks, and so work with Gibbs, EM and VB.
* **Engines** subclass `binference.samplers.base.Engine`. They take a
  `PopulationModel` and return a `PopulationResult` (draws) or a
  `PointEstimate`. Register user-facing engines in `binference.inference.METHODS`.
* **Likelihood constructors** are `Likelihoods.from_*` class methods that
  return bin-averaged likelihoods. Inputs that carry an interim prior must
  require it explicitly.

## Documentation

```bash
pip install -e ".[docs]"
sphinx-build -W --keep-going -b html docs docs/_build/html
# the code of the README, installation page and guides
python docs/_scripts/run_snippets.py README.md docs/index.md docs/start/install.md docs/guides/*.md docs/guides/inputs/*.md
# external links and the bibliography's DOIs (CI runs these weekly)
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

## Releasing

Releases go to PyPI from `.github/workflows/release.yml` through trusted publishing; no
token is stored anywhere. The publishers on PyPI and TestPyPI name that file and the
environments `pypi` and `testpypi`.

1. In a pull request, rehearse: set `src/binference/_version.py` to `X.Y.Zrc1` and run
   the Release workflow by hand on that branch (Actions → Release → Run workflow). It
   uploads to TestPyPI; check the page and install it in a clean environment:

   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ "binference==X.Y.Zrc1"
   ```

   An index accepts each version once, so after a fix rehearse again as `rc2`.
2. Once a rehearsal is clean, set the version to `X.Y.Z` in `_version.py` and in
   `CITATION.cff` (with `date-released`), and turn *Unreleased* in `CHANGELOG.md` into
   `## [X.Y.Z] - YYYY-MM-DD`.
3. Merge the pull request, then tag the merge commit on `main` and push the tag:

   ```bash
   git tag -a vX.Y.Z -m "binference X.Y.Z" && git push origin vX.Y.Z
   ```

   The workflow refuses a tag that is not `v` plus the package version, a version that
   `CITATION.cff` or `CHANGELOG.md` does not give, and a commit that is not on `main` or
   whose CI has not passed.
4. Create the GitHub release from the tag, with the version's CHANGELOG section as notes.
5. In the next pull request, set the version to the next `.dev0` and open a new
   *Unreleased* section.
