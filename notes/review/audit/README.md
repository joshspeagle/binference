# Independent audit (background sweep)

Started 2026-10-22, during the walkthrough. Seven independent reviewer agents,
each owning one slice of the codebase, apply two lenses:

- **review**: correctness bugs (wrong results, crashes, silent misuse,
  numerical problems, statistical errors), verified by running code where
  possible;
- **simplify**: reuse, simplification, efficiency, consistency, dead code,
  naming, API/docstring mismatches.

They do not change source code. Each writes `<area>.md` here. Findings are
triaged as we reach the corresponding walkthrough step; outcomes go to
`../DECISIONS.md`.

| File | Slice |
|---|---|
| `inputs.md` | `likelihoods.py`, `bins.py`, `selection.py` |
| `model-map.md` | `model.py`, `_kernels.py`, `_simplex.py`, `samplers/base.py`, `samplers/optimize.py` |
| `priors.md` | `priors/` |
| `nuts-laplace.md` | `samplers/nuts.py`, `samplers/_nuts_core.py`, `samplers/laplace.py` |
| `gibbs-vb-validation.md` | `samplers/gibbs.py`, `vb.py`, `bootstrap.py`, `validation.py`, `simulate.py` |
| `outputs-api.md` | `results.py`, `diagnostics.py`, `io.py`, `plotting.py`, `inference.py`, `estimator.py`, `__init__.py`, `warnings.py`, `_utils.py` |
| `tests-docs.md` | `tests/`, docs and README claims vs code, packaging |
