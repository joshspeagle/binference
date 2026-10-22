# Audit: inputs slice (`likelihoods.py`, `bins.py`, `selection.py`)

Auditor: independent agent, 2026-10-22, branch `claude/pastry-v0.1` at `5c56942`. No source files were changed.
Scripts are in `/tmp/claude-0/-home-user-pastry/63ffb1b4-5826-5f0b-ade5-c8209761737e/scratchpad/audit/inputs/` (`aNN_*.py`). Run them with the pastry venv active.

## Summary

- The core conversions are correct. These include the interim-prior division in both representations, the exact Gaussian CDF differences (1-D and N-D), the sub-bin point layout on unequal N-D grids, sample histogramming, and the `compress` log-scale bookkeeping. The 15 earlier review fixes in this slice hold.
- **One silent statistical bug.** With `kind="posterior"`, `from_function` reads an array `interim_prior` as *densities*, while `from_distributions` reads the same array as *masses*. On log bins this skews likelihood rows by up to 13× (I-1).
- **Numerical weaknesses in `from_distributions`.** The CDF path cancels in the CDF's upper tail, and the pdf path exponentiates `logpdf` before averaging. Objects that `from_gaussian` handles fine come out with empty rows (an error) or with lost tails (I-2, I-3).
- **Quadrature is coarse for narrow likelihoods.** `from_function` defaults to `n_sub=3` (the spec says 5). Its per-entry errors reach 77% when σ is 0.1 of the bin width. The effect on inference was small in my test (I-4).
- `lk.stack()` ignores the interim prior that `from_posteriors` recorded, so it is not "the familiar stacked PDF" (I-5).
- `compress()` densifies sparse storage: 27 s and 2.4 GB where sparse row keys take 0.06 s (I-7).
- The remaining findings are minor validation gaps (`rtol>1`, `N=0`, shape misreads, the volume check skipped for small-unit bins), mutable shared storage, duplicated logic and drift between the spec and the code.

Count: 23 findings. 1 bug-major, 9 bug-minor, 2 numerical, 2 statistical, 1 efficiency, 2 consistency, 4 docs, 1 simplify, 1 design. Mixed findings are counted once, under their first label.

---

### I-1. `from_function(kind="posterior")` reads an array interim prior as densities; `from_distributions` reads it as masses

- **Category:** bug-major (silent statistical error on a documented path)
- **Location:** `src/pastry/likelihoods.py:799-810` (from_function) vs `:738-748` (from_distributions); default at `:491-492`
- **Claim:** `from_function` forwards posteriors with `representation="density"`. Since review fix #10, `interim_representation` defaults to `representation`, so an array `interim_prior` is multiplied by the bin volumes. `from_distributions` forwards `representation="mass"`, so the same array is taken as masses. Neither docstring says how an array interim prior is read, and neither method exposes `interim_representation`. On unequal bins the same `interim_prior=Pi` therefore gives likelihoods that differ by a factor ∝ 1/v_k.
- **Evidence:** `a04_kind_posterior_interim.py`, on `Bins.log(0.1, 10, 8)`, with the interim masses `Pi` passed to both:
  ```
  truth              : [0.012 0.021 0.053 0.202 0.887 1.    0.024 0.   ]
  from_distributions : [0.011 0.02  0.05  0.18  0.717 1.    0.045 0.   ]   (the residual is the within-bin-constant-π approximation)
  from_function      : [0.158 0.158 0.219 0.445 1.    0.785 0.02  0.   ]
  ratio from_function/truth: [12.97 7.44 4.16 2.20 1.13 0.79 0.83 0.72]  (∝ 1/volume)
  ```
- **Suggested fix:**
  - Add an `interim_representation` parameter (default `"mass"`) to both `from_function` and `from_distributions`, and pass it through to `from_posteriors`.
  - Document it in both docstrings.
  - Alternatively, in `from_function` call `from_posteriors(..., representation="density", interim_representation="mass")` explicitly.
  - Add a regression test with an array interim prior on log bins for both constructors.
- **Confidence:** confirmed by running.

### I-2. `from_distributions` CDF path: cancellation in the upper CDF tail gives empty rows or lost tails

- **Category:** bug-minor / numerical
- **Location:** `likelihoods.py:714-718`
- **Claim:** `np.diff(dist.cdf(edges))` cancels when every edge lies in the CDF's upper tail (cdf ≈ 1), i.e. for objects *below* the bins.
  - Objects 7–8σ below the support get an all-zero row. That raises an error (or the object is dropped), even though the likelihood is finite and `from_gaussian` handles it.
  - Closer objects lose every entry after the first.
  - The lower tail (objects above the bins) is fine, because cdf ≈ 0 there carries full relative precision.
  - The docstring says such objects "are integrated exactly".
- **Evidence:** `a01_dist_cdf_tail.py`, `norm(loc, 0.05)` on `Bins.linear(0, 1, 10)`:
  ```
  loc=-0.35: from_gaussian [1, 8.8e-08, 1.5e-16, 4.8e-27]   from_distributions [1, 0, 0, 0]
  loc=-0.6 : from_gaussian [1, 4.4e-12, 3.6e-25, ...]       from_distributions ValueError: 1 object(s) have zero likelihood in every bin
  loc=1.35, 1.6 (above the bins): identical to from_gaussian
  ```
- **Suggested fix:**
  - When the interval lies above the median, use the survival function: `-np.diff(dist.sf(e))`.
  - Better, work in log space when `logcdf`/`logsf` exist: `log(Φ(b)-Φ(a)) = lb + log(-expm1(la-lb))`, as in `_log_gauss_bin_mass`. Then build with `cls(logm, bins, log=True, representation="mass")`.
- **Confidence:** confirmed by running.

### I-3. `from_distributions` pdf path exponentiates `logpdf` before averaging, so it underflows

- **Category:** numerical
- **Location:** `likelihoods.py:721-727`
- **Claim:** `dens = np.exp(dist.logpdf(pts))` underflows to 0 for objects whose likelihood is far from the grid. The row then becomes empty and raises.
  - `from_function(log=True)` avoids this with `logsumexp`, so the two quadrature constructors are inconsistent.
  - The affected cases are N-D grids and 1-D objects that have no `cdf`.
- **Evidence:** `a03_dist_logpdf_underflow.py`. A 1-D object with only `logpdf` at N(−0.8, 0.02) on [0, 1] raises "zero likelihood in every bin". `from_gaussian` for the same object returns `[1, 4.6e-93, 3.0e-196, …]`.
- **Suggested fix:** when `logpdf` exists, compute `logm = logsumexp(logpdf.reshape(K, -1) + log(wq), axis=1) + log(volumes)` and construct with `log=True, representation="mass"`. Fall back to `pdf` only when there is no `logpdf`.
- **Confidence:** confirmed by running.

### I-4. Sub-bin quadrature is inaccurate for likelihoods narrower than the bins; `from_function` defaults to `n_sub=3`

- **Category:** statistical / consistency
- **Location:** `likelihoods.py:758` (`n_sub=3`), `:681` (`n_sub=5`); the spec §4.2 table says `from_function(loglike, bins, n_sub=5)`
- **Claim:** Gauss–Legendre with 3 points per bin badly misestimates bin averages once σ ≲ 0.2 × width. Nothing warns. The two constructors also use different defaults (3 and 5) for the same quadrature.
- **Evidence:** `a02_quadrature.py`. Gaussian likelihoods on 10 bins, against exact CDF differences, over entries above 1% of the row maximum:
  ```
  sigma/width=0.5  n_sub=3: max rel err 1.4e-3   n_sub=5: 1.5e-6
  sigma/width=0.2  n_sub=3: max rel err 0.149    n_sub=5: 2.4e-3
  sigma/width=0.1  n_sub=3: max rel err 0.774    n_sub=5: 0.186
  ```
  The effect on a 20,000-object MAP with σ = 0.1 × width was small: max |Δρ| = 8.5e-4. Most of the error falls on tail entries. Steep populations and small ρ_k may be more sensitive.
- **Suggested fix:**
  - Use `n_sub=5` for both constructors (matching the spec).
  - Add a cheap resolution check: re-evaluate a random subset of rows at `2*n_sub` and warn (`InputWarning`) if the relative change exceeds ~1%.
  - Document the "σ ≳ bin width / n_sub" rule of thumb.
- **Confidence:** confirmed by running (accuracy); the impact on inference is only lightly tested.

### I-5. `Likelihoods.stack()` defaults to a flat interim prior even when the interim prior is known

- **Category:** statistical / docs
- **Location:** `likelihoods.py:403-416`; `model.py:247-255`; `docs/user-guide/inputs.md` ("`lk.stack()` is the familiar 'stacked PDF'. It is exactly one EM step from the interim prior")
- **Claim:** For likelihoods built by `from_posteriors`, `from_samples`, `from_classifier` or kind="posterior", `meta["interim_prior_masses"]` holds the user's interim prior. `stack()` nevertheless uses `"flat"`, so it does not reproduce the user's stacked posteriors. That contradicts the guide.
- **Evidence:** `a09_stack_interim.py`:
  ```
  user's stacked posteriors   : [0.374 0.286 0.155 0.133 0.039 0.014]
  lk.stack()  (default 'flat'): [0.181 0.158 0.171 0.179 0.15  0.161]
  lk.stack(meta interim)      : [0.374 0.286 0.155 0.133 0.039 0.014]
  ```
- **Suggested fix:**
  - Change the default to `interim_prior=None`, meaning `meta["interim_prior_masses"]` if present and `"flat"` otherwise.
  - Have `PopulationModel.stack()` do the same.
  - Otherwise, fix the guide so it says the stack is formed under a flat prior.
- **Confidence:** confirmed by running.

### I-6. `stack()` returns all-NaN if an interim prior is zero where a row has support

- **Category:** bug-minor
- **Location:** `likelihoods.py:411-416`
- **Claim:** `_interim_masses` allows zero masses. A row whose support lies only in zero-mass bins gets `tot = 0`, and the division produces NaN in *every* output component.
- **Evidence:** `a08_misc2.py`: `lk.stack(np.array([0.0, 0.5, 0.5]))` with a row `[1, 0, 0]` gives `[nan nan nan]`.
- **Suggested fix:** raise a clear `ValueError` (as `from_posteriors` does for the same situation), or skip zero-total rows with a warning.
- **Confidence:** confirmed by running.

### I-7. `compress()` densifies sparse storage and runs `np.unique(axis=0)` on the dense matrix

- **Category:** efficiency
- **Location:** `likelihoods.py:389-398`
- **Claim:** `compress` is advertised for large discrete data sets, but it builds a dense `(N, K)` copy plus a rounded copy. Sparse inputs from `from_samples` or `to_sparse` therefore cost O(NK) memory and time.
- **Evidence:** `a10_compress_sparse.py` (three non-zeros per row):
  ```
  N=20000, K=500,  nnz=6e4:   compress 2.1 s,  peak 0.24 GB;  sparse-row keys 0.04 s (same groups)
  N=50000, K=2000, nnz=1.5e5: compress 27.5 s, peak 2.43 GB;  sparse-row keys 0.06 s (same groups)
  ```
- **Suggested fix:**
  - For CSR, key rows on `(indices[a:b].tobytes(), data[a:b].tobytes())`, rounding the data first when `decimals` is set. Group with a dict or `np.unique` over the keys, then build the merged CSR directly.
  - For dense input, use `np.unique` over `np.ascontiguousarray(key).view(np.void(K*8))`, which is much faster than `axis=0`.
- **Confidence:** confirmed by running.

### I-8. `to_sparse(rtol)` accepts `rtol > 1` and silently empties every row

- **Category:** bug-minor
- **Location:** `likelihoods.py:342-350`
- **Claim:** Rows are max-normalised to 1, so `rtol > 1` removes every entry.
  - `_init_from_parts` checks only `log_scale` for empty rows, so the result is accepted.
  - Every log-likelihood is then −inf, and models built on it fail obscurely.
  - Negative `rtol` is also not rejected.
- **Evidence:** `a05_misc_validation.py`: `to_sparse(rtol=1.5)` gives `nnz=0, log_evidence=[-inf -inf -inf]`, with no error.
- **Suggested fix:** require `0 <= rtol <= 1`. Keeping the row maximum (which is 1) makes `rtol <= 1` safe.
- **Confidence:** confirmed by running.

### I-9. The posterior representation check is skipped for small-unit bins

- **Category:** bug-minor
- **Location:** `likelihoods.py:999` (`np.allclose(vol, vol[0])`)
- **Claim:** `np.allclose` has a default `atol=1e-8`. Bins whose widths are all below ~1e-8, e.g. wavelengths in metres or times in years written in Gyr⁻¹ units, are treated as "equal volume" even when they differ by large factors. The mass/density sanity warning (R-INPUT-2) is then silently disabled.
- **Evidence:** `a05_misc_validation.py`. Density-looking rows declared as `"mass"`:
  - on edges `[0, 1, 3, 7]` they warn;
  - on the same edges × 1e-9 (densities × 1e9) they produce no warning.
- **Suggested fix:** `np.allclose(vol, vol[0], rtol=1e-6, atol=0.0)`. Grep other volume-equality checks, e.g. in the priors, for the same pattern.
- **Confidence:** confirmed by running.

### I-10. `from_gaussian` silently reshapes ill-shaped `mean`; `sigma` is ambiguous when N == ndim

- **Category:** bug-minor
- **Location:** `likelihoods.py:628-640`
- **Claim:**
  - For 1-D bins, `mu.reshape(-1, 1)` accepts any shape. An `(N, 2)` mean with scalar sigma silently becomes 2N objects.
  - On N-D grids, a 1-D `sigma` of length `N == ndim` is read as one sigma per object, never per axis. The behaviour is undocumented.
- **Evidence:** `a05_misc_validation.py`:
  - `from_gaussian(np.full((3, 2), .5), 0.1, 1-D bins).N` returns 6.
  - On a 2-D grid with N=2, `sigma=[0.1, 0.3]` is stored as `[[0.1, 0.1], [0.3, 0.3]]`.
- **Suggested fix:**
  - For 1-D bins, require `mu.ndim == 1` or shape `(N, 1)`.
  - For the ambiguous case, raise and ask for an explicit 2-D sigma, or document the precedence.
- **Confidence:** confirmed by running.

### I-11. Zero-object `Likelihoods` are accepted silently

- **Category:** bug-minor
- **Location:** `likelihoods.py:225-253` (no `N == 0` check), `:867-889` (from_response), `:352-364` (subset)
- **Claim:**
  - `from_response` with all-zero counts returns `Likelihoods(N=0, …)`.
  - `subset([])` does the same.
  - The `drop_empty` path raises on N=0, but other paths do not, so downstream engines meet W=0.
- **Evidence:** `a05_misc_validation.py`: `from_response([0, 0], 0.5*eye(2))` gives `Likelihoods(N=0, K=2, dense, selection='observed', …)`.
- **Suggested fix:** raise in `_init_from_parts` when `norm.shape[0] == 0` (with a specific message for `from_response`: "no observed counts").
- **Confidence:** confirmed by running.

### I-12. `Bins.__eq__` and `__hash__` disagree

- **Category:** bug-minor
- **Location:** `bins.py:219-231`
- **Claim:** Equality uses `np.array_equal` (where −0.0 == 0.0), but the hash uses `tobytes()`, which differs. Equal objects then hash differently, which breaks dict and set semantics.
- **Evidence:** `a08_misc2.py`: `Bins([-0.0, 1, 2]) == Bins([0.0, 1, 2])` is True, yet the hashes differ and a set holds both.
- **Suggested fix:** normalise in `_as_edges` (`arr = arr + 0.0` turns −0.0 into 0.0), or hash `tuple(e.tolist())`.
- **Confidence:** confirmed by running. The impact is tiny.

### I-13. Unvalidated `kind` / `interim_representation` strings

- **Category:** bug-minor / consistency
- **Location:**
  - `likelihoods.py:776-810` (`from_function` never validates `kind`)
  - `:96-104` (`interim_representation` is checked only for array priors)
- **Claim:**
  - `from_function(kind="likelihoods", interim_prior="flat")` silently runs the *posterior* branch. `from_distributions` does validate `kind`.
  - `from_samples(..., interim_representation="dens")` is accepted with a string or callable prior.
- **Evidence:** `a05_misc_validation.py`. Both calls return without error, and the first divides by the interim prior.
- **Suggested fix:**
  - Validate `kind` in `from_function`, ideally through a shared helper (see I-21).
  - Validate `representation` at the top of `_interim_masses` regardless of the type of the prior.
- **Confidence:** confirmed by running.

### I-14. Internal arrays are writable and shared between copies and models

- **Category:** consistency
- **Location:**
  - `likelihoods.py:246-253`, `:262-280`, `:366-378`
  - `model.py:105-110`
  - `selection.py:55-70`
- **Claim:**
  - `lk.values`, `lk.log_scale` and `lk.weights` are writable views of the internal storage.
  - `with_weights`, `with_selection`, `to_dense`/`_from_parts` copies and `PopulationModel._L` all share them, so an in-place edit changes every derived object. This is the same class of problem as review fix #4, reached through the public property.
  - `Selection` uses `__slots__`, but `mode` and `values` can be reassigned. `Bins`, by contrast, is immutable.
- **Evidence:** `a12_shared_storage.py`. After `lk.values[0, 1] = 0`, the copy made by `with_weights` and `PopulationModel._L` both show `[1, 0]`; `log_scale` and `weights` are writable. In `a05`, `Selection.mode = "observed"` succeeds.
- **Suggested fix:**
  - Call `setflags(write=False)` on the stored dense array, on `L.data` for CSR, and on `log_scale` and `weights`.
  - Give `Selection` the same `__setattr__` guard as `Bins`.
- **Confidence:** confirmed by running.

### I-15. `_log_gauss_bin_mass` loses relative precision when σ ≫ bin width

- **Category:** numerical (low impact)
- **Location:** `likelihoods.py:120-132`
- **Claim:** For intervals near the mean, `la` and `lb` are both O(log 0.5). Their difference is tiny and carries the absolute error of `log_ndtr`, about 1e-16. `log(-expm1)` does not help, because the cancellation already happens inside `log_ndtr`.
- **Evidence:** `a06_gauss_broad.py`. Relative error of the bin mass against a midpoint reference:

  | σ/width | current | `erf` difference |
  |---|---|---|
  | 1e6 | 1e-10 | 5e-14 |
  | 1e10 | 2.1e-6 | 3.6e-15 |
  | 1e12 | 9.8e-5 | 7e-15 |

  With σ = 1e8 on unit bins, a row that should be flat to 1e-16 shows jitter of 3e-7. That is negligible for inference, because the row carries almost no information.
- **Suggested fix:** for intervals with `|a|, |b| < ~1` use `log(0.5*(erf(b/√2) - erf(a/√2)))`, and keep the `log_ndtr` form in the tails.
- **Confidence:** confirmed by running.

### I-16. `Bins.from_centers` misbehaves on non-uniform (e.g. log-spaced) centres

- **Category:** docs / design
- **Location:** `bins.py:113-125`
- **Claim:**
  - Edges sit at arithmetic midpoints, so `bins.centers` differ from the supplied centres whenever the spacing is not uniform. PDFs tabulated at `c_k` are then treated as values for bins centred elsewhere.
  - For log-spaced centres the first edge can be **negative**, i.e. the support of a positive quantity includes negative values.
  - The docstring presents the method as the tool "when PDFs are tabulated on a grid of points".
- **Evidence:** `a05_misc_validation.py`:
  - `from_centers(geomspace(0.01, 1, 3)).edges` is `[-0.035, 0.055, 0.55, 1.45]`.
  - `from_centers([0, 1, 3]).centers` is `[0, 1.25, 3]`.
- **Suggested fix:**
  - Add `log=True` (geometric midpoints, with outer edges at `c0²/c1` and `cK²/cK-1`), or warn when the spacing is non-uniform.
  - Document that `bins.centers` need not equal the input centres.
  - Refuse to produce a lower edge ≤ 0 when every centre is positive and `log=True`.
- **Confidence:** confirmed by running.

### I-17. A callable interim prior on categorical bins receives integer codes, not labels

- **Category:** docs
- **Location:** `likelihoods.py:96-98`
- **Claim:** For categories, `interim_prior(bins.centers)` is called with `[0., 1., 2., …]`. The docstrings say "a callable is a density evaluated by sub-bin quadrature", which does not apply to categories. A user would expect the callable to receive the labels.
- **Evidence:** `a05_misc_validation.py`: on `categories(["a", "b", "c"])` the callable is called with `[0. 1. 2.]`.
- **Suggested fix:** pass `np.asarray(bins.labels, dtype=object)`, or reject callables for categories, since an array is the natural input there. Document the choice.
- **Confidence:** confirmed by running.

### I-18. Constructor surface is inconsistent

- **Category:** consistency
- **Location:** `likelihoods.py`, the `from_*` signatures
- **Claim:**
  - `from_classifier` accepts no `selection`, `selection_mode` or `drop_empty`.
  - `from_response` accepts no `drop_empty`, and neither sparse responses nor extra weights.
  - `from_gaussian` is always dense, even when σ ≪ the range of the bins makes rows mostly zeros or underflow. `from_samples` has `sparse=`.
  - The `n_sub` defaults differ (I-4).
  - `n_sub` is not passed to `_interim_masses`: a callable interim prior always uses 5 points, even when the user sets `from_distributions(n_sub=…)`.
  - `interim_representation` exists only on `from_posteriors` and `from_samples` (I-1).
  - Several warnings carry a `stacklevel` that points inside pastry when they are raised through another constructor. Examples are `_check_posterior_representation` (stacklevel=3, reached via `from_distributions`, `from_function` and `from_classifier`) and the `drop_empty` warning in `_init_from_parts` (reached from any `from_*`).
- **Evidence:** reading the code.
- **Suggested fix:** give every `from_*` the same keyword tail: `weights, selection, selection_mode, drop_empty, sparse=None`. Thread `n_sub` through to `_interim_masses`. Use one `n_sub` default. Compute the stacklevel with a small helper, or use `skip_file_prefixes` (Python ≥ 3.12, with a fallback).
- **Confidence:** confirmed by reading.

### I-19. `compress` docstring overstates exactness; it returns `self` when there are no duplicates

- **Category:** docs / consistency
- **Location:** `likelihoods.py:380-401`
- **Claim:**
  - With `decimals`, rows that differ slightly are merged and the first member's row represents the group, so "total absolute log-likelihoods are preserved exactly" no longer holds.
  - When nothing merges, `compress()` returns `self`. Every other method returns a new object. The `meta` dict is mutable, so the caller can be surprised.
- **Evidence:** `a05`: `lk.compress() is lk` is True. The first point follows from the code.
- **Suggested fix:** qualify the docstring ("exactly when `decimals is None`"), and return a shallow copy via `_from_parts`.
- **Confidence:** confirmed by running and by reading.

### I-20. Spec and docs drift from the code

- **Category:** docs
- **Location:** `docs/design/spec.md` §4.1–4.2; `bins.py:319-326`; `docs/user-guide/inputs.md`
- **Claim:**
  - The spec table still says `from_posteriors(..., interim_representation="mass")`; the code now follows `representation`.
  - It says `from_response(..., efficiency=None)`; the code defaults to `"from_response"`.
  - It says `from_function(loglike, bins, n_sub=5)`; the code uses `n_sub=3`.
  - It promises `Likelihoods.rebin(groups)` and `Bins.coarsen(factor or groups)`. Neither the `rebin` method nor the groups option exists, and the `coarsen` error for categories says "Use explicit groups to merge categories" although no API accepts groups (`result.rebin` does).
  - The first example in `inputs.md` evaluates the Gaussian at bin *centres* as `L`. With σ = 0.05 and width 0.1 that is far from the bin average: for an object at a bin centre, the neighbour/own ratio is 0.135 at the centres against 0.157/0.683 = 0.23 for the bin averages. Exact `from_gaussian` appears only further down the page.
  - `Bins.grid` argument parsing gives confusing errors. A 2-D ndarray is taken as one axis and fails "strictly increasing"; `Bins.grid([])` raises IndexError.
- **Evidence:** reading the code and docs. The ratio comes from Φ differences.
- **Suggested fix:** update the spec table. Either implement groups in `coarsen` or reword the error to point at `result.rebin`. Add a sentence to the guide that point values at centres are an approximation and `from_gaussian`/`from_function` give bin averages. Handle ndarray input in `Bins.grid` (iterate rows of a 2-D array, or raise a clear message).
- **Confidence:** confirmed by reading.

### I-21. Duplicated logic and dead code

- **Category:** simplify
- **Location:** as listed
- **Claim and suggested fix:**
  - **Interim division duplicated.** "Divide by the interim prior, checking for mass where it is zero" appears twice (`:494-505` in from_posteriors and `:570-575` in from_samples), with different messages and expressions. Factor out `_divide_interim(P, Pi)`.
  - **`kind` handling duplicated.** The `kind` validation and the likelihood-or-posterior dispatch are duplicated in `from_distributions` and `from_function`, and are already inconsistent (I-1, I-13). Factor out a helper taking `(values, representation, log)`.
  - **Quadrature mass duplicated.** `(dens.reshape(K, -1) @ w) * volumes` appears in `_interim_masses:104` and `from_distributions:727`, and also in `simulate.py:137+`. Factor out `_bin_masses_from_density`.
  - **Dense/sparse validation duplicated.** The checks in `from_posteriors:473-482` repeat `_normalise_linear`. One `_as_nonneg_matrix(values)` helper could serve both.
  - **Redundant `M.sum_duplicates()`** after `.tocsr()` (`:568`), since `tocsr` already sums duplicates.
  - **Redundant `abs(P)`** at `:497`, since P was validated non-negative.
  - **Dead branch** `np.where(b2 > a2, out, -np.inf)` at `:132`: edges are strictly increasing and σ > 0.
  - **`bins.py`:**
    - `Bins._from_parts` is only ever called with `edges=None` (`:155`), so its grid branch is dead.
    - The `_ordered` slot and `ordered` property always equal `not is_categorical` and are unused in `src/`. Drop or alias them.
    - `_subbin_points` builds index meshgrids by hand; `np.indices(bins.shape).reshape(ndim, -1)` and `np.indices((n_sub,)*ndim)` would be simpler.
  - **Late validation.** `Likelihoods.__init__` validates `representation` only after the normalisation work (`:213-219`). Validate it first.
- **Evidence:** reading the code. The grep for `_from_parts(` and `.ordered` is in my notes.
- **Confidence:** confirmed by reading.

### I-22. `Bins.index` silently flattens wrong-shaped input for 1-D bins

- **Category:** bug-minor (API)
- **Location:** `bins.py:266-268`
- **Claim:** For 1-D bins, `x.reshape(-1, 1)` accepts any shape. An `(n, 2)` array, e.g. N-D samples sent to 1-D bins, is indexed element-wise with no error. `from_samples` guards its own path, but `index` and `contains` are public.
- **Evidence:** `a05`: `Bins(1-D).index([[0.1, 0.3], [0.5, 0.9]])` returns `[0 1 2 4]`.
- **Suggested fix:** accept only `(n,)` or `(n, 1)` for 1-D bins.
- **Confidence:** confirmed by running.

### I-23. `from_samples` has no smoothing, and its noise warning is coarse

- **Category:** design
- **Location:** `likelihoods.py:563-586`
- **Claim:**
  - Histogrammed likelihoods have hard zeros wherever no sample landed, plus Poisson noise in the tails. The population MLE is sensitive to both, especially when σ is comparable to the bin width.
  - The only guard is a warning on the *median* n_eff over objects, and n_eff counts samples outside the bins.
  - An object with 5 effective samples among thousands with 1000 passes silently.
- **Evidence:**
  - Convergence check `a07_samples_bias.py`: rows converge to the exact likelihood (max relative error 1.7% at S = 4e5 on log bins), so the convention is correct.
  - The sensitivity claim comes from reasoning and an unpublished trial. In that trial the deconvolution MAP moved by up to 0.07 between S = 20, 100 and 1000. Part of that comes from the ill-conditioning of the MAP itself, so treat it as indicative only.
- **Suggested fix:**
  - Add an optional pseudo-count or kernel smoothing (`smoothing=` with a sample-bandwidth rule).
  - Warn on the *fraction* of objects with `n_eff_inside < min_samples`, not on the median.
  - Store `n_eff` of the inside samples.
- **Confidence:** plausible.

---

## Checked and found correct

- **Interim-prior division.** L ∝ P/Π in both representations, with `"flat"` meaning ∝ volumes. The convention test against the brute-force truth on log bins passes (inputs guide example, `a04` from_distributions row, `a07`).
- **`from_gaussian`.**
  - 1-D exact CDF differences divided by volume.
  - Far tails on both sides (`a01`).
  - N-D outer product in C order on an unequal 3-D grid: `from_function`/`from_distributions` at n_sub=8 agree to 1.9e-12 (`a11`).
  - Per-object `gaussian_mean`/`gaussian_sigma` survive `drop_empty`, `subset` and `compress`.
- **Quadrature internals.** `_subbin_points` cell and point ordering, the weights, and the volume factor (`a11`). A callable interim prior on N-D grids reproduces the volumes (`a11`).
- **`from_samples`.**
  - 1-D and N-D histogramming with the bin index convention (left-closed, last edge inclusive; NaN/±inf → outside).
  - Ragged lists, importance weights, Kish n_eff, outside fraction.
  - Convergence to exact likelihoods (`a07`, `a11`).
- **Normalisation and scales.**
  - `_normalise_linear`/`_normalise_log`: row-max scaling, all −inf rows, rejection of +inf/NaN, copying sparse input (fix #4 holds; `to_sparse` copies).
  - `_csr_row_max` with empty rows (the `reduceat` segments are correct because empty rows are skipped).
  - `log_scale` bookkeeping for `representation="mass"`.
  - `log_evidence_terms` units.
- **`compress`.** Without `decimals`, the weighted mean of log-scales preserves Σ w_i log L_i·ρ exactly. The representative rows match the `np.unique` groups.
- **`from_response`.** Both conventions after fix #5: R·ε in `"observed"` mode gives the per-event term M·ε with the (ε·ρ)^−W normaliser. Columns summing to more than 1 are rejected. Dropping zero-count rows is correct.
- **`from_classifier`.** The p/π_train correction (EMQ/SLD) and label handling.
- **Selection.**
  - The `Selection`/`as_selection` mode requirements and K checks.
  - `PopulationModel` scales L by S only in `"true"` mode and rescales S to max 1 for data augmentation. For fractions only ratios of S matter, so this is valid.
- **`Bins`.**
  - `index`/`contains`, `coarsen` mapping, `neighbors`, `unflatten`/`flatten`.
  - Immutability guard and read-only arrays.
  - Pickling via `__reduce__`.
  - `io` round trip of categories.
- **Targeted tests.** `tests/test_likelihoods.py` and `tests/test_bins.py`: 42 passed. The full suite was not run, since no code was changed.
