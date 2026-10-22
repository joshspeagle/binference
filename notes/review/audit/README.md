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

## Status (2026-10-22): all seven reports in

| Report | Findings | Most serious |
|---|---|---|
| inputs | 23 | I-1 interim-prior array read as density vs mass inconsistently (rows wrong up to 13× on log bins) |
| model-map | 14 | M-1 zero-weight rows make EM reset/collapse under bootstrap weights; M-2 missing information wrong with selection |
| priors | 22 | P-1–P-3 LogisticGP defaults (log bins, K > 48, long length scales); P-4 DirichletTree per-node alpha crash |
| nuts-laplace | 15 | N-1 logistic-prior results expose internal centred coordinates in x / log_prob / model |
| gibbs-vb-validation | 12 | G-1 SBC p-values miscalibrated for uneven rank bins; G-3 exact_posterior biased near the boundary (α < 1) |
| outputs-api | 21 | O-1 load() + attach() drops the selection function; O-3 estimator breaks sklearn ≥ 1.6 CV |
| tests-docs | 20 | T-1 slow statistical tests too weak; T-2 SBC misses hyperparameters, GP, selection, rates; T-4 mix-SQP warning |
