# Arguments ledger

Every public argument: current default, proposed default, why, status. Seeded
in step 2.3 (2026-10-24) with the input constructors; extended module by module
during the walkthrough; implemented in Phase 7. Principle: expose everything
with sensible defaults (DECISIONS #16).

Status: `keep`, `change` (agreed), `proposed` (awaiting Josh), `open`.

## Input constructors (`likelihoods.py`)

| Where | Argument | Today | Proposed | Why | Status |
|---|---|---|---|---|---|
| all posterior-capable constructors | `interim_prior` | `"flat"` / array / callable; array read as mass or density depending on constructor | `interim=` taking an `InterimPrior` (`flat(coordinate=)`, `from_masses`, `from_densities`, `from_function`, `from_distribution`); bare arrays refused or need a representation | one meaning everywhere (audit I-1/I-18, #24) | proposed |
| samples / density constructors | `kind` | `"likelihood"` default (function, distributions); samples require an interim | one of `kind="likelihood"` or `interim=` required, no silent default | forgetting the interim prior is the costliest input mistake | proposed |
| `from_function` / `from_distributions` | `n_sub` | 3 / 5 fixed points | `tol=` adaptive quadrature (n_sub as a fixed-rule override) | #23 | change |
| `from_samples` | division by interim | per-bin mass Π_k | pointwise π(z_s) when available | exact under the model (2.3b §2) | proposed |
| `from_samples` | `min_samples` | warn if median n_eff < 20 | log-likelihood MC-variance diagnostic Σ V_i(ρ) at posterior draws (warn > ~1) + n_eff under the fitted population | error grows like N/S | proposed |
| `from_samples` | empty bins | L = 0 | optional Dirichlet floor `a` (L ∝ c/π + a) | principled floor | proposed (opt-in) |
| `from_gaussian` | covariance | per-axis σ only | per-object covariance; exact 1-D/2-D, adaptive cubature or QMC above | #2, #25 | change |
| `from_response` | name / bins | `from_response(counts, response, bins=categories)` | `Response` object + `from_binned(counts or values, response)`, real truth-bin geometry | 2.3b §4 | proposed |
| `from_response` | response uncertainty | none | Dirichlet per column from the simulation/validation counts; sampled jointly as parameters (Gibbs) | coverage (2.3b §7) | proposed |
| `from_classifier` | route | posteriors ÷ π_train | default route C (score bins + labelled validation set); route A via `from_posteriors` + `InterimPrior` | robustness to calibration and π_train (2.3b §5) | proposed |
| every constructor | bins | some 1-D or grid only | arbitrary edges, N-D, masks, measure | #22 | change |
