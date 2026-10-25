# Audit: outputs and API (`results.py`, `diagnostics.py`, `io.py`, `plotting.py`, `inference.py`, `estimator.py`, `__init__.py`, `warnings.py`, `_utils.py`)

**Summary.** The core maths of this slice checks out. Every per-object method agrees with brute force to machine precision, in all three selection settings (none / observed / true) and for both fractions and rates: `assignment_probabilities`, `predict_proba`, `log_predictive`, `sample_assignments`, `realized_counts` and `sample_values` (truncated-Gaussian and `loglike` paths, KS p = 0.56 / 0.65 against the exact within-bin posterior). ESS (bulk/tail/mean), split R-hat, MCSE, `_gpdfit` and PSIS match ArviZ 0.22 to 4+ digits, apart from one rounding difference in the PSIS tail length. The defects sit at the edges:
- `attach()` after `load()` silently drops the selection function and accepts likelihoods on different bins, so per-object outputs come out silently wrong.
- `PopulationResult.from_samples` does no validation.
- `PopulationEstimator` cannot be used with scikit-learn ≥ 1.6 model selection (`cross_val_score` and `GridSearchCV` raise), although README and docstrings advertise it.
- `PointEstimate.plot(which="rate")` plots fractions.
- `result_to_dict` overwrites the bin labels with Gibbs label draws.
- `check()` ignores E-BFMI, tail ESS and `converged`, and computes no R-hat for single-chain runs.
- `summary(which="fractions")` on rate models reports the ESS of the rates.

The rest is consistency and UX: the `which` vocabulary (`"fractions"` vs `"fraction"`), fractions-only accessors, discoverability of `pastry.plotting`, and `to_datatree` fidelity. There is also some duplication and per-object Python loops.

Scripts and outputs: `/tmp/claude-0/-home-user-pastry/63ffb1b4-5826-5f0b-ade5-c8209761737e/scratchpad/audit/outputs-api/` (run with the pastry venv; the ArviZ/sklearn comparisons need `PYTHONPATH=ext`, a private `--target` install that leaves the venv untouched). `pytest -q`: 268 passed. Line numbers refer to `eba1285`.

---

### O-1. `attach()` after `load()` silently drops the selection function, so per-object outputs are wrong
- **Category:** bug-major
- **Location:** `results.py:392-402` (`attach`), `io.py:61-127` (selection not saved)
- **Claim:** A saved result does not store the selection function (values or mode). The documented workflow is `load(path).attach(likelihoods)`, and `attach` builds `PopulationModel(likelihoods)` with no selection. A fit with `selection_mode="true"` therefore loses the S_k column weighting in `assignment_probabilities`, `sample_*` and `realized_counts`. For both selection modes, `log_predictive` loses the 1/(S·ρ) normaliser. Nothing warns. (A rate result attached this way also gets a fractions model; that is harmless for these methods, but `model.kind` no longer matches `result.kind`.)
- **Evidence:** `i02_datatree_attach.py`. Fit with `S = linspace(1, 0.2)`, `mode="true"`, then save, load and `attach(lk)`:
  `assignment_probabilities before/after save+attach max diff: 0.018`;
  `log_predictive mean before/after: -0.967 -1.0987`.
- **Suggested fix:** Save `selection.values` and `selection.mode` (an array and a JSON field) and re-apply them in `attach` when a bare `Likelihoods` is passed. If the result has a selection and the attached likelihoods carry a different one, raise or warn. Also accept `attach(lk, selection=..., selection_mode=...)`.
- **Confidence:** confirmed by running.

### O-2. `attach`, `predict_proba` and `log_predictive` accept likelihoods on different bins (only K is compared)
- **Category:** bug-minor
- **Location:** `results.py:399` (`attach`), `results.py:279` (`predict_proba`), `results.py:299-306` (`log_predictive` has no check at all)
- **Claim:** `attach` compares only `K`, so likelihoods on a log grid can be attached to a result fitted on a linear grid with the same K. `sample_values` then draws on the *model's* bins while `result.bins` still reports the old ones. `predict_proba` likewise accepts new `Likelihoods` on unrelated bins with the same K. `log_predictive` does not even check K, so a mismatch surfaces as a matmul shape error.
- **Evidence:** `i03_attach_bins.py`: `attach with different bins (same K) accepted; result.bins: Bins(K=4, range=(0, 1)) model.bins: Bins(K=4, range=(1, 100))`; `sample_values` returns values 4.6–6.8, outside `result.bins`; `predict_proba` with likelihoods on `[10, 20]` is accepted.
- **Suggested fix:** Require `lk.bins == self.bins` (Bins has `__eq__`) in `attach`, and the same check in `predict_proba` and `log_predictive`. For a raw array, keep wrapping it with `model.bins`. Factor the "coerce to Likelihoods and validate" step into one helper used by both methods.
- **Confidence:** confirmed by running.

### O-3. `PopulationEstimator` fails with scikit-learn ≥ 1.6 model selection (`cross_val_score`, `GridSearchCV`)
- **Category:** bug-minor (crash on an advertised path)
- **Location:** `estimator.py:20-143`; README.md:131 ("For scikit-learn pipelines …"); `score` docstring ("compare priors or binnings by cross-validation")
- **Claim:** `clone()` works. Since sklearn 1.6, however, every meta-estimator calls `get_tags(est)`, which requires `__sklearn_tags__` unless the class inherits `BaseEstimator`. `cross_val_score` and `GridSearchCV` therefore raise `AttributeError`. Separately, sklearn's CV splitters cannot index a `Likelihoods`, so CV works only on raw arrays with `bins=`, and then multiplicity weights are lost.
- **Evidence:** `e01_infer_estimator.py` (sklearn 1.9.1): `cross_val_score ERR: AttributeError ... 'PopulationEstimator' object has no attribute '__sklearn_tags__'`, and the same for `GridSearchCV`. `e02_sklearn_tags_fix.py` monkeypatches the proposed fix, after which `cross_val_score: [-0.772 -0.854 -0.767]` and `GridSearchCV best: {'prior': Dirichlet(alpha=0.5)}` work. `cross_val_score(est, lk)` still fails ("All the 3 fits failed").
- **Suggested fix:** Add a lazily-importing `__sklearn_tags__` that returns `Tags(estimator_type=None, target_tags=TargetTags(required=False))`, with an input tag for 2-D arrays. Add a test that is skipped when sklearn is absent. Document that CV needs raw arrays, or make `Likelihoods` row-indexable via `__getitem__`/`__len__` (it has `subset`) so `_safe_indexing` works.
- **Confidence:** confirmed by running.

### O-4. `PopulationResult.from_samples` does not validate shape or simplex
- **Category:** bug-minor
- **Location:** `results.py:372-390`
- **Claim:** `from_samples` takes `kind=model.kind` and accepts draws of any K, and any non-simplex values for a fractions model. The docstring says "fractions or rates", which suggests rates may be passed to a fractions model. With unnormalised rows, `mean`, `summary`, `interval` and `plot` are silently wrong; the per-object methods happen to renormalise. A K mismatch surfaces later as a broadcast error.
- **Evidence:** `c01_check.py`: `from_samples with K=5 draws for K=30 model accepted`, which fails later with `operands could not be broadcast together`; `unnormalised fractions accepted; samples row sums: [34.09 30.9 29.97]`.
- **Suggested fix:** Check `draws.shape[-1] == model.K`, finiteness and non-negativity. For fractions models, renormalise when rows are within about 1e-6 of 1 and raise otherwise. Alternatively, accept `kind=` explicitly and normalise rates to fractions when the model is a fractions model.
- **Confidence:** confirmed by running.

### O-5. `PointEstimate.plot(which="rate")` plots fractions; unknown `which` raises a bare `KeyError`
- **Category:** bug-minor
- **Location:** `plotting.py:38-40` (`_draws`), `plotting.py:104`
- **Claim:** For a `PointEstimate`, any `which` other than `"density"` returns `result.rho`, so `which="rate"` on a rate MAP silently plots fractions under a "rate" axis label. An invalid value passes `_draws` and then fails with `KeyError: 'bogus'`.
- **Evidence:** `r01_results_api.py`: `PointEstimate.plot(which='rate') plotted [0.0142 0.0453 0.0378] rates are [3.857 12.29 10.25] fractions [0.0142 0.0453 0.0378]`; `pe.plot(which=bogus): KeyError 'bogus'`.
- **Suggested fix:** Route both result types through a single `_which` (O-14). For `PointEstimate`, `"rate"` should return `result.rates` (an error if `None`), and every other value should raise the same `ValueError`.
- **Confidence:** confirmed by running.

### O-6. `result_to_dict` overwrites the bin labels with the Gibbs label draws, and omits info, hyperparameters and schema
- **Category:** bug-minor
- **Location:** `io.py:40-58`
- **Claim:** `result_to_dict` merges the bins metadata (`"labels": [category labels]` for categorical bins) and then writes `out["labels"] = result.labels`, the stored per-object label draws from `Gibbs(store_labels=True)`. The category labels are lost. It also omits `info` (including `hyperparameters`), `schema` and `pastry_version`, which `save_result` does write, so the two serialisations disagree. The underlying naming clash is that `result.labels` (draws of z) shares a name with `bins.labels` (category names).
- **Evidence:** `i01_io.py`: `result_to_dict['labels'] = ndarray (2, 50, 30) (bin labels were ['x', 'y', 'z', 'w'] )`. `save`/`load` itself is fine (bin labels go in JSON meta, draws in an array).
- **Suggested fix:** Build `save_result` on `result_to_dict` (one source of truth) and rename the result attribute, e.g. `label_draws` / `assignments`, with keys `label_draws` and `bin_labels`.
- **Confidence:** confirmed by running.

### O-7. `check()` ignores E-BFMI, tail ESS and `converged`; single-chain runs never get R-hat
- **Category:** statistical
- **Location:** `results.py:545-570` (`diagnostics`), `results.py:583-636` (`check`)
- **Claim:**
  - `diagnostics` collects `ebfmi` and `converged`, but `check()` never inspects them. Low E-BFMI (< 0.3, the Betancourt/Stan threshold) is a standard NUTS warning.
  - `ess_tail` is computed but not checked. Vehtari et al. (2021) recommend both bulk and tail ESS > 100 per chain.
  - R-hat is computed only when `n_chains >= 2`, although `diag.rhat` already splits a single chain. A single-chain NUTS/Gibbs run therefore cannot fail the R-hat check, even with a trend.
  - The ESS check is skipped for `method="external"` (`from_samples`), the case where users most need it.
  - `strict=True` raises a plain `RuntimeError`; a dedicated exception would be easier to catch.
- **Evidence:** `c01_check.py`: `ebfmi=0.05..0.12, converged=False -> check() msgs: [] warnings: 0`; `single-chain NUTS diagnostics keys: ['ess_bulk', 'ess_tail']`.
- **Suggested fix:** Add `ebfmi < 0.3` (any chain), `min(ess_tail)/n_chains < ess_min`, and `converged is False` (MAP/VB) to `check()`. Compute split R-hat whenever there are at least 8 draws per chain. Apply the ESS check to any MCMC-like method, i.e. anything not marked approximate and not `laplace`/`vb`/`bootstrap`/`map`.
- **Confidence:** confirmed by running.

### O-8. `summary(which="fractions")` on a rate model reports the ESS and R-hat of the rates
- **Category:** consistency / statistical
- **Location:** `results.py:650-654` with `results.py:552`
- **Claim:** `diagnostics` is computed on the rates for rate models, which is deliberate because it covers the total. `summary(which=...)` always appends these columns, though, so the `ess_bulk` column in a fractions table is not the ESS of the numbers in that table. When the total mixes slowly the column badly understates the fractions' ESS.
- **Evidence:** `c01_check.py` (random-walk total): `summary(which='fractions').ess_bulk[:3] = [177. 750. 766.] but ESS of the fractions = [710. 775. 874.]`.
- **Suggested fix:** Compute the diagnostics columns for the `X` being summarised, cached per `which`; ESS is invariant to the per-bin `density` scaling, so `density` can share with its base. Keep `check()` on the rates.
- **Confidence:** confirmed by running.

### O-9. `plot_population`: 3-D grids crash with a cryptic error; 2-D grids silently ignore `truth`, `stack`, `levels`, `color` and `label`
- **Category:** bug-minor
- **Location:** `plotting.py:100-101`, `plotting.py:123-124`, `plotting.py:192-212`
- **Claim:**
  - `ndim >= 3` grids fall into the 1-D branch and fail inside matplotlib.
  - The 2-D branch drops all overlays and options without a word.
  - The 2-D branch returns an ndarray of axes, while the other branches return a single `Axes`.
- **Evidence:** `r01_results_api.py`: `3-D plot: ValueError 'x' has size 3, but 'y1' has an unequal size of 9`; the 2-D plot with `truth=` runs, shows no truth, and returns an `ndarray`.
- **Suggested fix:** Raise `ValueError("plot supports 1-D/2-D grids and categories; use result.density() …")` for `ndim > 2`. For 2-D, either warn when `truth`/`stack` are given or add a truth panel. Document the return type.
- **Confidence:** confirmed by running.

### O-10. Save/load gaps: float or mixed category labels become strings, a path without `.npz` does not round-trip, non-array info is dropped silently
- **Category:** bug-minor
- **Location:** `io.py:21-26`, `io.py:72-81`, `io.py:90/101`
- **Claim:**
  - Earlier review issue 14 fixed integer labels only. `Bins.categories([0.5, 1.5, 2.5])` loads as `['0.5', '1.5', '2.5']`, and mixed `[1, 'b', 3]` as strings. The loaded bins then compare unequal to the originals, which breaks `attach` once O-2 is fixed.
  - `np.savez_compressed` appends `.npz`, so `save("fit")` followed by `load("fit")` raises `FileNotFoundError`.
  - List-valued info (e.g. NUTS `inv_metric`) and nested dicts other than `hyperparameters` are dropped without notice.
- **Evidence:** `i01_io.py`: `cat_float ok=False labels [0.5, 1.5, 2.5] -> ['0.5', '1.5', '2.5']`; `cat_mixed ok=False`; `saved files: ['noext.npz']`, `load(path without .npz): FileNotFoundError`; `info_lost=['list_stat']` for every type.
- **Suggested fix:** Store labels as a JSON list with per-item type tags, or `json.dumps` them as is (JSON keeps int, float and str), and raise for unserialisable labels. In `load_result`, try `path + ".npz"` when `path` is missing, or normalise the suffix in `save`. Store lists of equal-shape arrays as stacked arrays, and list the dropped keys in `meta["info_dropped"]`.
- **Confidence:** confirmed by running.

### O-11. `sample_values(loglike=...)`: NaN crash for narrow support and slow per-object loops
- **Category:** numerical / efficiency
- **Location:** `results.py:246-261`
- **Claim:**
  - If `loglike` is `-inf` at all `n_sub` midpoints of a bin that has positive mass (a likelihood narrower than a sub-cell), `np.exp(lw - max(lw))` produces NaN and `gen.choice` raises "Probabilities contain NaN".
  - The path calls `loglike` once per (object, distinct label) in a Python double loop. One call per object over all K × n_sub points is about 12× cheaper.
- **Evidence:** `p03_loglike_perf_edge.py`: `narrow-support loglike: ValueError Probabilities contain NaN` (with a RuntimeWarning). Timing at N = 2000, K = 20, 200 draws: `loglike path ... 3.8s` against `one loglike call per object (all bins at once): 0.31s` for the loglike evaluations alone.
- **Suggested fix:** Guard `if not np.isfinite(lw).any()`, fall back to uniform in the bin and warn, or raise a clear error suggesting a larger `n_sub`. Evaluate `loglike(Z_all, i)` once per object on the (K, n_sub) sub-grid and draw the cells for all draws at once (vectorised inverse CDF via `searchsorted` on the per-bin cumulative weights).
- **Confidence:** confirmed by running.

### O-12. Several `which` vocabularies for one concept, and overloaded draw-count names
- **Category:** design / consistency
- **Location:**
  - `results.py:440-449` (`"fractions"`, `"rates"`, `"density"`)
  - `plotting.py:35-49, 85` (`"fraction"`, `"rate"`, `"density"`)
  - `io.py:151-153` (`rho`, `rate`)
  - `results.py:731-766` (`PointEstimate.value` / `.rho` / `.rates`)
  - `estimator.py` (`population_`)
- **Claim:** The same quantities go by `samples`/`fractions`/`fraction`/`rho`/`value` and `rates`/`rate`. `result.plot(which="fractions")` raises, and so does `result.summary(which="fraction")`; the user guide uses both spellings a few lines apart. Draw counts are overloaded too:
  - `n_draws` is a property giving the total number of draws;
  - `n_draws=` is a method argument giving the number of thinned population draws;
  - `infer(n_samples=)` means draws per chain for NUTS/Gibbs and in total for the other engines;
  - `PopulationEstimator.sample(n)`.
- **Evidence:** `r01_results_api.py`: `res.plot(which='fractions'): ValueError which must be 'density', 'fraction' or 'rate'.`; `res.summary(which='fraction'): ValueError which must be 'fractions', 'rates' or 'density'.`
- **Suggested fix:** Pick one spelling (plural, matching `kind`) and accept the singular as an alias in a single shared normaliser. Name the data var `fractions` in `to_datatree`, or document `rho`. Consider `PointEstimate.fractions` as an alias of `rho`.
- **Confidence:** confirmed by running.

### O-13. Fractions-only accessors on rate results
- **Category:** design
- **Location:** `results.py:451-473` (`mean`, `median`, `std`, `cov`, `corr`), `507-542` (`functional`, `rebin`), `plotting.py:215-275` (`plot_corner`, `plot_trace`)
- **Claim:** `quantile`, `interval` and `summary` take `which=`, but the moment accessors, `functional`, `rebin`, `plot_corner` and `plot_trace` are always on fractions. A rate user cannot rebin rates, compute a functional of the rates, or trace-plot the quantity the diagnostics are computed on without hand-rolling it from `result.rates`.
- **Evidence:** `r01_results_api.py`: `rebin on rates result returns fractions: True`. The signatures speak for the rest.
- **Suggested fix:** Add `which="fractions"` to `functional`, `rebin`, `cov`, `corr`, `plot_corner` and `plot_trace`, and leave the properties as fractions, documented.
- **Confidence:** confirmed by reading and running.

### O-14. `to_datatree` fidelity: hyperparameters dropped, `diverging` stored as float, non-ArviZ stat names, string bin coordinates
- **Category:** consistency
- **Location:** `io.py:130-171`
- **Claim:**
  - `info["hyperparameters"]` holds `(chain, draw)` posterior draws (e.g. `amplitude`, `length_scale`), but they do not reach the `posterior` group.
  - `diverging` is float64; ArviZ expects bool.
  - The names `accept_stat` and `n_leapfrog` differ from the ArviZ conventions `acceptance_rate` and `n_steps`, so ArviZ plots that look for them miss them.
  - The `bin` coordinates are formatted strings, which lose the numeric centres and the 2-D grid structure.
  - The attrs lack `kind`, `prior` and `pastry_version`.
- **Evidence:** `i02_datatree_attach.py`: `sample_stats vars: ['accept_stat', 'tree_depth', 'diverging', 'n_leapfrog', 'energy', 'lp'] diverging dtype: float64`; `posterior vars: ['rho'] ; hyperparameters in info: ['amplitude', 'length_scale']`; `posterior attrs: {'method': 'nuts', 'approximate': 0}`.
- **Suggested fix:** Add the hyperparameters as posterior variables, cast `diverging` to bool, rename the stats (or add aliases), and add numeric `bin_center` (and, for grids, per-axis index) coordinates plus the metadata attrs.
- **Confidence:** confirmed by running.

### O-15. Discoverability: `pastry.plotting` is not reachable from `import pastry`; no `result.plot_individual`
- **Category:** design / docs
- **Location:** `__init__.py:28`; `docs/user-guide/results.md:131` (cites `pastry.plotting.plot_individual`)
- **Claim:** `pastry.plotting` (and `pastry.io`) are not imported in `__init__`. `import pastry; pastry.plotting.plot_individual(...)` raises `AttributeError` unless something else has imported the submodule first. `plot_individual` and `plot_sbc` have no method counterparts, unlike `plot`, `plot_corner` and `plot_trace`.
- **Evidence:** `python -c "import pastry; print(hasattr(pastry,'plotting'))"` prints `False`.
- **Suggested fix:** Add a module-level `__getattr__` in `__init__` that lazily imports `plotting` and `io`, keeping matplotlib optional, and add `PopulationResult.plot_individual(i)` (and `PointEstimate.plot_individual`).
- **Confidence:** confirmed by running.

### O-16. The `ConvergenceWarning` from `infer(check=True)` points at `pastry/inference.py`
- **Category:** design
- **Location:** `results.py:635` (`stacklevel=2`), called from `inference.py:125`
- **Claim:** When `check()` is called from `infer`, the warning's location is `inference.py:125` rather than the user's call site. Default warning filters de-duplicate by location, so after the first fit, repeated fits in a session or notebook show no warning.
- **Evidence:** `i02_datatree_attach.py`: `ConvergenceWarning attributed to: src/pastry/inference.py 125`.
- **Suggested fix:** Give `check()` a private `_stacklevel` argument and call `result.check(_stacklevel=3)` from `infer`.
- **Confidence:** confirmed by running.

### O-17. `infer(method="em")` does not use EM; `method="AUTO"` does not record `auto_method`
- **Category:** consistency
- **Location:** `inference.py:17-25`, `inference.py:122`
- **Claim:** `"em"` is a pure alias of `MAP`, which with `algorithm="auto"` runs mix-SQP. Users asking for EM (for example to reproduce the classic deconvolution iteration) get a different optimiser. Also, `key` is lower-cased but `if method == "auto"` compares the raw string, so `"AUTO"` works but does not set `info["auto_method"]`.
- **Evidence:** `e01_infer_estimator.py`: `method='em' -> map {'space': 'simplex', 'algorithm': 'mixsqp'}`; `method='AUTO' auto_method recorded: False`.
- **Suggested fix:** Map `"em"` to `MAP` with `algorithm="em"`, by injecting the kwarg unless the user set it, or drop the alias. Compare against the lower-cased string.
- **Confidence:** confirmed by running.

### O-18. Defaults of `n_draws=None` (all draws) for `sample_assignments`, `sample_values` and `realized_counts`
- **Category:** design / efficiency
- **Location:** `results.py:161-207`
- **Claim:** `assignment_probabilities`, `predict_proba` and `log_predictive` default to 500 thinned draws, while the sampling methods default to *all* draws. For 4 × 1000 NUTS draws and 10^6 objects, `sample_assignments()` allocates a 32 GB int64 array, and `sample_values` a float array of the same size. `predict_proba` also densifies the whole new likelihood matrix with no chunking, unlike `assignment_probabilities`.
- **Evidence:** by inspection: memory is n_draws × n_objects × 8 bytes.
- **Suggested fix:** Default these methods to a modest `n_draws` (e.g. 100–500), or warn above a size threshold. Use int32 labels. Chunk `predict_proba` the same way as `assignment_probabilities`.
- **Confidence:** plausible (arithmetic; not run at that scale).

### O-19. Duplication and dead parameters in the per-object helpers
- **Category:** simplify
- **Location:**
  - `results.py:127` (`_select_draws(rng)` is never used)
  - `results.py:150-158` and `284-289` (re-implement `kern.responsibilities` twice)
  - `results.py:189-194` (`unit or integer` is just `integer`; it re-implements `_utils.is_integer_array`, which `bootstrap.py:75` also re-implements with a different tolerance)
  - `results.py:540-542` (`rebin` loops over groups)
  - `results.py:520` (`functional` loops in Python by default)
- **Claim:** Each of these can be shortened without changing behaviour:
  - Average `kern.responsibilities(L, rho, chunk=...)` over the draws.
  - Write `rebin` as `X @ one_hot(groups)`.
  - Drop the unused `rng` argument.
  - Use `is_integer_array` everywhere, and add it to `_utils.__all__`.
  - `rebin` with a float mapping (e.g. from arithmetic) raises an opaque `TypeError: 'numpy.float64' object is not iterable` (`r01_results_api.py`); casting integral floats, or a clear error, would help.
- **Evidence:** reading, plus `r01_results_api.py` for `rebin`.
- **Suggested fix:** as in the claim.
- **Confidence:** confirmed by reading.

### O-20. PSIS tail length rounds down, whereas ArviZ and loo round up
- **Category:** numerical (low)
- **Location:** `diagnostics.py:259`
- **Claim:** `tail_len = int(min(0.2n, 3√n))` floors, whereas ArviZ and loo use `ceil`. With 3√n non-integer the tail is one draw shorter, which changes `k_hat` slightly. Separately, the k̂ threshold is the fixed 0.7 here and in `check()`. Vehtari et al. (2024) recommend `min(1 − 1/log10 S, 0.7)`, which is stricter for small S (0.57 at S = 200 proposals).
- **Evidence:** `d01_diag_vs_arviz.py`: identical at n = 100 (max |Δlw| = 0); `n=1000 k: pastry 4.8419 arviz 4.9810`; `n=4000 k: pastry 11.2989 arviz 11.2418`.
- **Suggested fix:** Use `int(np.ceil(...))`, and use the sample-size-dependent threshold in Laplace and `check()`.
- **Confidence:** confirmed by running.

### O-21. Docs: "`sample_assignments` gives exact joint draws … whatever engine produced ρ"
- **Category:** docs
- **Location:** `docs/user-guide/results.md:64`, `results.py:166-169`
- **Claim:** The draws are exact *conditional on* ρ. They are exact joint posterior draws only when the ρ draws are, so not for VB, uncorrected Laplace, bootstrap, or Laplace with k̂ ≥ 0.7. Separately, the top-hat fallback in `sample_values` (no Gaussian meta, no `loglike`) is documented but silent. The earlier review (issue 6) showed it can be badly off (KS D = 0.47 here, `p01_per_object.py`); a one-time `InputWarning` would make it discoverable.
- **Evidence:** reading; `p01.out`.
- **Suggested fix:** Reword to "exact given exact draws of ρ (approximate results inherit their approximation)". Warn once on the top-hat fallback.
- **Confidence:** confirmed by reading.

---

## Checked and found correct
- `assignment_probabilities`, `predict_proba`, `log_predictive` against brute force, for selection none / observed / true × fractions / rates: max error ≤ 1.3e-15. `log_predictive` also matches direct numerical integration of ∫N(d; z, σ) n(z) dz to 2e-4 (quadrature error). The earlier fix for issue 12 (rates with selection) is correct.
- `sample_assignments` frequencies match `assignment_probabilities` within MC noise. `realized_counts` has the right mean (Σ_i r_ik) and its rows sum to N.
- `sample_values`, truncated-Gaussian path: matches the exact within-bin posterior (KS p = 0.56), stays finite for objects 20–170 σ outside the bins, and handles N-D cells per axis correctly. `loglike` path: KS p = 0.65. `idx` accepts None, int, list, slice and boolean masks.
- `_ess_raw`, bulk/tail/mean ESS, split rank-normalised R-hat (bulk and folded), MCSE and `_gpdfit`: identical to ArviZ 0.22 on 8 test cases. The only difference is tail ESS with odd n, where ArviZ takes the quantile of the unsplit array (261 vs 267, immaterial). `autocorr_time` gives 8.9 for AR(1) with φ = 0.8 (true value 9), and handles the all-window case better than emcee. E-BFMI (computed in `nuts.py`) uses Stan's formula.
- `_hdi` matches `az.hdi` (it includes ⌈level·n⌉ draws; ArviZ spans ⌊level·n⌋ gaps, a negligible difference). Central intervals, quantiles, summary column names (`q16`/`q84`, `q2.5`/`q97.5`), single-draw summaries, `density()` for fractions and rates, `functional` and `rebin` (integer map and list of groups) all behave as documented.
- Save/load round-trips draws, kind, method, approximate, prior repr, linear/log/2-D grids, categorical string, integer and default labels, scalar info, array info, hyperparameters, `log_prob` and Gibbs label draws with `label_index`. The fix for issue 14 holds for integer labels.
- Plotting: 1-D bands and steps, categorical bars for rates and densities, `plot_individual` (the L·V / V algebra is correct for bin-averaged likelihoods), `plot_corner`, `plot_trace`, `plot_sbc` band. Truth and stack overlays rescale to the mean total for rates, as documented.
- `infer`: dispatch, default sample counts, `n_samples` rejected for MAP, rejection of `prior`/`selection`/`bins` alongside a model, `auto` threshold. Bad engine kwargs raise `TypeError` with the engine and argument named, which is clear enough.
- `PopulationEstimator`: `get_params`/`set_params`/`clone`, `fit` returning self, `score` weighting by multiplicities (the same for a `Likelihoods` and the equivalent raw array), `predict`, `sample`.
- `_utils.as_generator` follows SPEC 7 and rejects `RandomState` clearly. `chunk_slices`, `check_positive_weights`, the warning hierarchy and the `__all__` contents are consistent.
- `pytest -q`: 268 passed, 15 deselected.

## Counts by category
| Category | Count | Findings |
|---|---|---|
| bug-major | 1 | O-1 |
| bug-minor | 7 | O-2, O-3, O-4, O-5, O-6, O-9, O-10 |
| statistical | 1 | O-7 (O-8 is listed under consistency) |
| numerical | 2 | O-11 (shared with efficiency), O-20 |
| consistency | 3 | O-8, O-14, O-17 |
| design | 4 | O-12, O-13, O-15, O-16 |
| efficiency | 1 | O-18 (O-11 is listed under numerical) |
| simplify | 1 | O-19 |
| docs | 1 | O-21 |
