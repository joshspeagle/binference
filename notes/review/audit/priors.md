# Audit: `src/pastry/priors/`

Auditor: independent agent, 2026-10-22, branch `claude/pastry-v0.1` (clean tree). Every line of
`base.py`, `dirichlet.py`, `logistic.py`, `hyper.py`, `rates.py` and `__init__.py` was read.
Scripts are in `/tmp/claude-0/-home-user-pastry/63ffb1b4-5826-5f0b-ade5-c8209761737e/scratchpad/audit/priors/`
(abbreviated `$A/` below). `pytest -q`: 268 passed, 15 deselected (113 s).

## Summary

- The core algebra is right. Densities, Jacobians and gradients agree with finite differences for every
  prior, including partial centring at intermediate λ. Conjugate updates match brute-force importance
  sampling. `prior_transform` matches `sample`, the HSGP spectral densities are correct, and the RW
  eigenbases are correct in 1-D and N-D.
- The main problems are in the **defaults of `LogisticGP`**. The prior the user gets differs a lot
  from the one described:
  - On `Bins.log` the GP lives in the linear coordinate, so about half the bins are frozen together
    (P-1).
  - For K > 48 the `n_basis=48` cap cannot represent the lower part of the default length-scale prior
    (P-2).
  - `boundary_factor=1.5` distorts the upper part of that prior (P-3).
- Other real bugs:
  - Per-node `alpha` for a `DirichletTree` with nodes of different arity always crashes (P-4).
  - In the centred form, the gradient disagrees with the density for RBF modes below the 1e-150
    floor (P-5).
  - `trend_scale` makes "free" linear trends increasingly constrained as K grows (P-6).
  - Several bind-time crashes or silent degeneracies on coarse grids and bad inputs (P-7 to P-10).
- The rest are consistency, docs and simplification items.

Counts (by primary category): statistical 4, bug-minor 5, numerical 1, consistency 5, docs 3,
design 1, simplify 2, efficiency 1 (22 findings).

---

### P-1. `LogisticGP` on log-spaced bins: the GP is stationary in the *linear* coordinate, so most low bins are rigidly tied together
- **Category:** statistical (major for a documented path)
- **Location:** `logistic.py:502-519` (basis on `bins.edges` in linear units; default length-scale
  prior from `2*min(diff(e))` and `e[-1]-e[0]`)
- **Claim:**
  - The user guide (`docs/user-guide/priors.md:69-72`) says the logistic priors work "whatever the
    binning, `Bins.log` included". That holds for the *mean* (D16), not for the GP covariance.
  - On `Bins.log(1, 1e4, 40)` the sine basis (≤ 40 functions over a linear box of half-width
    1.5 × 5000) cannot vary across the first ~20 bins (x < 120). The length-scale prior spans
    0.5 to 1e4 (ratio 2e4), so it is not meaningful either.
  - The HSGP is also wildly inaccurate there: the exact correlation is 0, but the HSGP gives 0.86.
- **Evidence:** `$A/a08_gp_logbins.py`:
  ```
  default ell prior quantiles 1/50/99%: [5.2e-01 6.66e+00 9.999e+03]; bin centres span 1.13..8972
  q=0.5: ell=6.7; exact corr(bin0, bin20 at x=113) = 0.000; HSGP corr = 0.864
     prior SD of adjacent log-fraction steps, bins 0-5: [0.0003 0.0004 0.0004 0.0006 0.0007]  bins 34-38: [0.289 0.268 0.295 0.29 0.288]
  prior draws: SD of log(rho_1/rho_0) = 0.001  SD of log(rho_39/rho_38) = 1.194
  ```
  The prior is effectively "flat density over the first decade or two, free at the top".
- **Suggested fix:**
  - Build the basis and the default length-scale prior in a coordinate the user can choose, e.g.
    `LogisticGP(coords="log")`, or automatically in log x when the edges are geometric
    (`Bins.log`).
  - At minimum, warn at bind when `max(width)/min(width)` is large, and correct the user-guide
    sentence.
  - `LogisticRandomWalk` is unaffected: it is per bin index.
- **Confidence:** confirmed by running.

### P-2. Default `n_basis = min(K_d, 48)` cannot represent the default length-scale prior for K > 48; the effective prior is much smoother than stated
- **Category:** statistical
- **Location:** `logistic.py:496-500` (cap), `logistic.py:515-519` (lower 1% quantile = 2 bin
  widths); spec D13
- **Claim:**
  - D13 says the length-scale prior "removes length scales the grid cannot resolve". With the cap at
    48, the *basis* cannot resolve them once K > 48.
  - For short ℓ the truncated HSGP loses most of the high-frequency variance:
    - the field SD falls below `amplitude`;
    - the bin-to-bin roughness collapses.
  - ℓ then becomes weakly identified in that region: it mostly rescales the amplitude, a
    funnel-like ridge. The earlier review only noted that n_basis is "slightly below the guideline"
    for K ≤ 48.
- **Evidence:** `$A/a07_hsgp_nbasis.py`. The table shows HSGP/exact ratios for the field SD and
  for the SD of adjacent differences ("step"), over the central half of the grid, at prior
  quantiles of ℓ:

  | K | kernel | q01 sd / step | q10 sd / step | q25 sd / step |
  |---|---|---|---|---|
  | 48 | matern32 | 0.99 / 0.90 | 1.00 / 0.95 | 1.00 / 0.96 |
  | 100 | matern32 | 0.93 / 0.63 | 0.98 / 0.78 | 0.99 / 0.85 |
  | 200 | matern32 | 0.78 / 0.32 | 0.91 / 0.52 | 0.96 / 0.67 |
  | 500 | matern32 | 0.53 / 0.09 | 0.71 / 0.21 | 0.83 / 0.34 |
  | 200 | matern12 | 0.70 / 0.21 | 0.82 / 0.28 | 0.88 / 0.31 |

  Matérn-1/2 is under-resolved even at K = 48 (step-SD ratio 0.71–0.74), because its spectrum
  decays slowly.
- **Suggested fix:**
  - Tie the two defaults together. Choose `n_basis` from the lower length-scale quantile using the
    Riutort-Mayol et al. rule, m ≈ 3.42·c·S/ℓ_min for Matérn-3/2 (≈1.75 for SE, more for
    Matérn-1/2), capped by K.
  - Or, when n_basis is capped, raise the lower quantile of the default prior to what the basis
    resolves. Warn if the user's `n_basis` is below the rule.
  - Correct D13 or the docstring.
- **Confidence:** confirmed by running.

### P-3. `boundary_factor=1.5` is too small for the upper half of the default length-scale prior; edge bins are over-shrunk
- **Category:** statistical
- **Location:** `logistic.py:472, 504`
- **Claim:**
  - The default prior puts 17% of its mass above ℓ = 0.3·range and 1% above the full range.
  - Riutort-Mayol et al. (2022) require c ≳ 3.2 ℓ/S (SE) or ≳ 4.1 ℓ/S (Matérn-3/2). At ℓ = 0.3 range
    that already means c ≈ 2–2.5.
  - With c = 1.5, the Dirichlet boundary condition pulls the field towards the mean near the grid
    edges, which is where population tails live. This shows in the zero-sum part of the
    covariance, which is all that matters for ρ.
- **Evidence:** `$A/a06_hsgp_boundary.py` (K = 40 on [0, 1]; default ℓ-prior quantiles
  1/10/50/90/99% = 0.05/0.079/0.159/0.39/1.0). Each cell gives the centred-covariance max error /
  edge-bin SD ratio:

  | kernel | ℓ = 0.1 | ℓ = 0.3 | ℓ = 0.5 | ℓ = 1.0 |
  |---|---|---|---|---|
  | rbf, c = 1.5 | 0.000 / 1.00 | 0.159 / 0.89 | 0.296 / 0.68 | 0.209 / 0.41 |
  | matern32, c = 1.5 | 0.004 / 1.00 | 0.125 / 0.92 | 0.217 / 0.80 | 0.203 / 0.62 |
  | matern32, c = 3.0 | 0.025 / 0.99 | 0.001 / 1.00 | 0.004 / 1.00 | 0.039 / 0.93 |
- **Suggested fix:** default c ≈ 2.5–3, together with the n_basis rule of P-2 (m scales with c).
  Alternatively, cap the upper quantile of the default ℓ prior at what c = 1.5 supports, though that
  is a stronger modelling change.
- **Confidence:** confirmed by running.

### P-4. `DirichletTree(alpha=[per-node arrays])` crashes whenever internal nodes have different numbers of children
- **Category:** bug-minor (crash on a documented argument form; no workaround)
- **Location:** `dirichlet.py:494`. `np.ndim(alpha)` is evaluated on a ragged list before the
  `isinstance(alpha, list | tuple)` check.
- **Claim:**
  - The docstring documents "one array per internal node … with one value per child". For any tree
    with mixed arity, `np.ndim` raises `ValueError: setting an array element with a sequence`. That
    covers the user-guide tree `[[0, 1, 2], [3, [4, 5]]]`.
  - Lists, tuples and lists of `np.ndarray` all fail.
  - Uniform-arity trees (e.g. all binary) work by accident.
- **Evidence:** `$A/a02_tree_ragged_alpha.py` prints `ValueError setting an array element with a
  sequence…` for all three input forms. The tests only use scalar alpha or binary trees.
- **Suggested fix:**
  ```python
  if isinstance(alpha, list | tuple) or (isinstance(alpha, np.ndarray) and alpha.ndim > 0):
      alphas = [np.asarray(a, float) for a in alpha]; ...
  else:
      return [np.full(len(ch), float(alpha)) for ch in self._nodes]
  ```
  Also validate positivity at `__init__`; today that happens only at bind.
- **Confidence:** confirmed by running.

### P-5. Centred/partially centred form: below the 1e-150 scale floor the log-density is frozen but the gradient is not
- **Category:** numerical
- **Location:** `logistic.py:193-211`. `s_safe = max(s, 1e-150)` enters `prec` and `log_s_term`,
  but `common = … + one_m*(q-1)` still multiplies the unfloored `dlogs`.
- **Claim:**
  - For the RBF kernel, high-frequency scales underflow far below 1e-150 at moderate ℓ. At K = 48
    and ℓ = 0.6, 19 of 48 modes are below the floor and 5 are exactly 0. The default ℓ prior puts
    7% of its mass above ℓ = 0.42, where this starts.
  - For λ_j < 1, the returned ∂/∂u is then inconsistent with `lp`. In particular, the amplitude
    gradient is off by exactly one per floored mode.
  - `_conditional_log_scale_sd` (`nuts.py:52-54`) runs an FD Hessian of this *gradient* at λ = 0,
    so the D12 centring gate can be fed spurious curvature for RBF priors. Any λ_j < 1 on such
    modes during sampling would break the Hamiltonian's energy consistency.
- **Evidence:** `$A/a15_floor_grad.py` (λ = 0, amplitude 1, ℓ = 0.6):
  ```
  modes with s < 1e-150: 19 of 48 ; s==0: 5
  u_amp: analytic -27.2124  FD -8.2124
  u_ell: analytic 25738.7276  FD 2480.4374
  ```
  The earlier sweep `$A/a01_grad_fd.py` shows the same thing as "GP rbf grid lam=0 max rel err
  1.0". Every other case agrees to ≤ 2e-9.
- **Suggested fix:**
  - Have `scales()` return `log s`, which is computed analytically anyway (`log_S`), and use it
    directly: `log_s_term = one_m @ log_s`, `prec = exp(-2*one_m*log_s)`, `m = exp(lam*log_s)`.
    No floor is needed; an overflowing `prec` with z ≠ 0 is a genuine -inf.
  - If a floor is kept, zero the `dlogs` columns of floored modes so that value and gradient agree.
  - Also note `0**0 == 1` in `m = s**lam` (line 178): a mode with s == 0 and λ = 0 gets
    c = z instead of 0.
- **Confidence:** confirmed by running (the gradient mismatch); the effect on the NUTS gate is
  plausible.

### P-6. `trend_scale` is in orthonormal-basis units, so the "free" polynomial trends get tighter as K grows
- **Category:** statistical / docs
- **Location:** `logistic.py:374, 418-423`; `_orthonormal_polys` (`logistic.py:69-77`);
  user guide `priors.md:83` ("linear trends are free"); theory "broad Gaussian of width
  `trend_scale`"
- **Claim:**
  - The null modes are unit-norm vectors over K bins, so a coefficient SD of 10 gives an eta-space
    trend whose size scales as 1/√K.
  - The penalised modes are orthogonal to the trend and cannot absorb it.
  - At K = 1000, a log-fraction change of 5 across the range (a factor ~150) is a 4.6σ event, so
    `LogisticRandomWalk(2)` is far from trend-neutral there. At K = 10 the same prior is genuinely
    broad.
- **Evidence:** `$A/a11_rw_nd.py`:
  ```
  K=  10: ... prior SD of end-to-end log-fraction change from the linear mode = 9.91
  K= 100: ... = 3.43
  K=1000: ... = 1.09
  ```
- **Suggested fix:** scale the null-mode SDs by √(K_d) per axis, or define `trend_scale` as the SD
  of the end-to-end change in log-density, so the meaning does not depend on K. Document the units
  either way.
- **Confidence:** confirmed by running.

### P-7. `LogisticGP()` fails to bind when any axis has 2 bins, and becomes a near-point-mass length-scale prior with 3
- **Category:** bug-minor
- **Location:** `logistic.py:515-519`, `hyper.py:133-134`
- **Claim:**
  - With 2 bins on an axis, `2 * min width == range`, so `from_quantiles` raises
    `ValueError: Need 0 < lower < upper.` That message does not mention the GP, the axis or the
    `length_scale=` workaround. It applies to `Bins.linear(0, 1, 2)` and to an N-D grid with a
    2-cell axis, e.g. 20 × 2.
  - With 3 bins the default is IG(a = 132), i.e. ℓ is essentially fixed at 0.81·range without
    warning.
- **Evidence:** `$A/a14_coarse_grids.py`:
  ```
  linear K=2   LogisticGP  ValueError: Need 0 < lower < upper.
  grid 20x2    LogisticGP  ValueError: Need 0 < lower < upper.
  linear K=3   LogisticGP  ok ls prior IG(a=132, b=107) median 0.811
  ```
- **Suggested fix:** for axes with fewer than ~4 bins, use a fixed or broad default such as
  `Fixed(range)` or `LogNormal(log(range/2), 1)`, or raise a GP-specific error that names the axis
  and suggests `length_scale=`.
- **Confidence:** confirmed by running.

### P-8. `LogisticNormal(cov=…)` accepts negative variances and non-PSD matrices silently
- **Category:** bug-minor
- **Location:** `logistic.py:309-321`
- **Claim:**
  - A scalar `cov < 0` gives `s0 = nan`, with only a numpy RuntimeWarning, and every downstream
    density becomes NaN.
  - For a matrix, negative eigenvalues of `P C P` are dropped by `keep = lam > 1e-10*max`. The
    user's covariance is silently replaced by its PSD part, with reduced rank.
  - If `lam.max() <= 0` (e.g. `cov` = constant matrix), `keep` can be empty and the prior has no
    field at all.
- **Evidence:** `$A/a13_misc.py`:
  ```
  cov [-1.] accepted; s0 = [nan nan nan nan nan]
  cov [1. 0. 0.] accepted; s0 = [1. 1. 1. 1.]   # diag(1,1,-5,1,1,1): the -5 is dropped
  ```
- **Suggested fix:**
  - Require a scalar `cov > 0`.
  - For matrices, raise if `min eigenvalue < -1e-8 * max` or if `keep.sum() == 0`.
  - Document that only the zero-sum projection of `cov` matters.
- **Confidence:** confirmed by running.

### P-9. `PolyaTree` on one bin recurses forever
- **Category:** bug-minor
- **Location:** `dirichlet.py:620-637`. With `hi - lo == 1`, `mid == hi`, so the empty child
  `(mid, hi)` recurses without end.
- **Claim:** `PolyaTree().bind(Bins.linear(0, 1, 1))` raises `RecursionError`. `Dirichlet` raises a
  clean "needs at least two bins".
- **Evidence:** `$A/a13_misc.py`: `PolyaTree K=1: RecursionError`.
- **Suggested fix:** `if K < 2: raise ValueError(...)` in `_structure`. Validate `power` as well; a
  negative power is accepted silently, which is harmless but not meaningful.
- **Confidence:** confirmed by running.

### P-10. `LogisticGP` does not validate `n_basis` or `boundary_factor`
- **Category:** bug-minor
- **Location:** `logistic.py:476, 496-500`
- **Claim:**
  - `n_basis=0` builds a prior with no field: dim = 2, only hyperparameters, so ρ is fixed at the
    mean.
  - `n_basis=2.7` is truncated to 2.
  - A per-axis `n_basis` of the wrong length gives a raw numpy broadcast error.
  - `boundary_factor=nan` passes the `<= 1.0` check.
- **Evidence:** `$A/a13_misc.py` (inline script output: `n_basis 0 -> J = 0 dim 2`, `n_basis 2.7 -> J = 2`,
  `[5, 5] -> ValueError operands could not be broadcast…`).
- **Suggested fix:**
  - Require positive integers, one per axis.
  - Use `if not boundary_factor > 1.0`.
  - Same pattern for `GeneralizedDirichlet` `a`/`b`. A wrong-length `b` also gives a raw broadcast
    error (`dirichlet.py:546-552`).
- **Confidence:** confirmed by running.

### P-11. `LogisticGP(length_scale=np.array([...]))` is rejected, although a tuple or list works
- **Category:** consistency
- **Location:** `logistic.py:520` (`isinstance(ls, Sequence)`), `hyper.py:186-192`
- **Claim:**
  - A per-axis NumPy array, the natural form, raises
    `TypeError: Expected a Hyperprior or a positive number; got ndarray`.
  - `as_hyperprior` also rejects a 0-d `np.array(2.0)`.
  - It accepts `True`, which becomes `Fixed(1.0)`.
- **Evidence:** `$A/a12_hyper.py` output.
- **Suggested fix:** in `LogisticGP.bind`, treat `np.ndim(ls) == 1` like a sequence. In
  `as_hyperprior`, accept `np.ndim(value) == 0` reals and reject `bool`.
- **Confidence:** confirmed by running.

### P-12. Tree `vb_sample` can return exact zeros; `sample` and `posterior_sample` floor at 1e-300
- **Category:** consistency
- **Location:** `dirichlet.py:432-433` vs `357-359`, `403-407`
- **Claim:**
  - `compose` multiplies floored branch probabilities, and the product can underflow to 0.
  - `sample` and `posterior_sample` re-floor the result; `vb_sample` does not. VB draws can
    therefore contain exact zeros, so any `log(rho)` downstream (diagnostics, plotting on a log
    axis) gives -inf.
- **Evidence:** `$A/a13_misc.py`: `PolyaTree(c=0.01)` on K = 64,
  `tree vb_sample exact zeros: 2  sample zeros: 0`.
- **Suggested fix:** wrap `vb_sample` in `np.maximum(..., _TINY)`, or better, floor once inside
  `compose`.
- **Confidence:** confirmed by running.

### P-13. `GeneralizedDirichlet` and `PolyaTree` subclass `DirichletTree` but never set its attributes
- **Category:** consistency
- **Location:** `dirichlet.py:528-536`, `586-596`
- **Claim:**
  - Neither subclass calls `super().__init__`. `isinstance(gd, DirichletTree)` is True, but `.tree`,
    `.alpha`, `._nodes` and `._alphas` do not exist.
  - Any code that dispatches on `DirichletTree` and reads `prior.tree` or `prior.alpha` will raise
    `AttributeError`.
  - The `bind` method is duplicated verbatim in all three classes (`507`, `561`, `640`), although
    the subclasses override only `_structure`.
- **Evidence:** `$A/a13_misc.py`: `GD isinstance DirichletTree: True  has .tree: False  has .alpha: False`.
- **Suggested fix:** introduce a private `_TreePrior(Prior)` base with `bind` and an abstract
  `_structure`. Make `DirichletTree`, `GeneralizedDirichlet` and `PolyaTree` siblings, and delete
  the two duplicate `bind`s.
- **Confidence:** confirmed by running.

### P-14. `repr`s omit parameters that change the prior
- **Category:** consistency
- **Location:**
  - `PolyaTree._params` (`dirichlet.py:598`): `base` is missing;
  - `LogisticNormal._params` (`logistic.py:303`): `mean` is missing;
  - `LogisticRandomWalk._params` (`383`): `mean` and `trend_scale` are missing;
  - `LogisticGP._params` (`485`): `mean`, `n_basis` and `boundary_factor` are missing;
  - `Dirichlet.from_base_measure` leaves `self.alpha == 1.0` (`dirichlet.py:149`), a misleading
    public attribute.
- **Claim:** two different priors print identically, e.g. `PolyaTree(base=[1, 2, 3])` prints
  `PolyaTree(c=1.0, power=2.0)`. The repr appears in `PopulationModel.__repr__` and results, so
  saved runs lose the information.
- **Evidence:** `$A/a13_misc.py`.
- **Suggested fix:** include every non-default argument, abbreviated by `_short` for arrays. Set
  `alpha=None` (or `"base_measure"`) for the base-measure constructor.
- **Confidence:** confirmed by running.

### P-15. `GammaRates` docstring overstates the Dirichlet equivalence; rates have no `prior_transform`
- **Category:** docs / consistency
- **Location:** `rates.py:241-242`, `rates.py:276-279`, `base.py:158-160`
- **Claim:**
  - "`GammaRates(alpha, b)` implies `rho ~ Dirichlet(alpha)`" holds only for a **common** rate b.
    The class explicitly allows per-bin `rate` arrays, and with unequal b_k the implied ρ is not
    Dirichlet.
  - `_BoundGammaRates` sets `has_prior_transform = False` although the transform is a one-liner,
    `special.gammaincinv(a, u) / b`.
  - The base-class docstring hard-codes the cube as `[0,1]^(K-1)`, which is wrong for rates (K).
- **Suggested fix:** add "with a common rate". Implement `prior_transform` for rates, documenting
  the dimension as `dim` (= K-1 for fractions, K for rates).
- **Confidence:** confirmed by reasoning; the absence of a transform was checked in code.

### P-16. `GeneralizedDirichlet` docstring: the Dirichlet-equivalence condition is incomplete
- **Category:** docs
- **Location:** `dirichlet.py:515-517`
- **Claim:**
  - `a` has K-1 entries, so "`b_k = sum_{j>k} a_j`" omits the K-th Dirichlet concentration. With
    that literal choice, b_{K-1} = 0, which is invalid.
  - The correct statement is: for Dir(α₁…α_K), a_k = α_k and b_k = Σ_{j>k} α_j, including α_K.
  - The code's `b=None` default, (K-1-k)·a, is correct for symmetric α = a.
- **Suggested fix:** reword.
- **Confidence:** confirmed by reasoning.

### P-17. N-D `LogisticRandomWalk(order ≥ 2)`: the Kronecker-sum precision leaves cross terms unpenalised
- **Category:** design
- **Location:** `logistic.py:389-399`
- **Claim:**
  - The null space of Q₁⊗I + I⊗Q₂ is null(Q₁)⊗null(Q₂). For order 2 in 2-D that is {1, x, y, xy}.
    For order 3 it is 9 modes, including x²y² and the other mixed terms.
  - Those mixed terms are curved surfaces, yet they are governed only by `trend_scale`, not by the
    smoothness `scale`.
  - The code implements this correctly (`$A/a11_rw_nd.py`: |Q̂−Q| = 7e-15, 3 non-constant null modes
    for order 2).
  - The user guide's "penalises curvature" is not true for these modes. A thin-plate-type
    precision (adding mixed-difference penalties) would penalise them.
- **Suggested fix:** document the null space for N-D, or add the mixed second-difference term
  D₁⊗D₂ for order 2.
- **Confidence:** confirmed by running (the null space); the modelling choice is a design question.

### P-18. Duplicated machinery across the simplex and rate priors
- **Category:** simplify
- **Location:**
  - Stick-breaking `betaincinv` loop: `dirichlet.py:197-208` and again per node at `386-401`.
  - Log-space small-shape gamma trick: `dirichlet.py:31-48` (`_dirichlet_draw`) and `rates.py:222-232`
    (`_gamma_draw`), with two copies of `_TINY`.
  - Three identical `bind`s (P-13).
  - `_BoundLogistic._eta` (`164-168`) re-implements the first half of `forward` (`179-182`).
  - `prior_transform` of `_BoundDirichlet` is the single-node special case of `_BoundTree`.
- **Suggested fix:**
  - Add one `_stick_break(a, u)` helper and one `_gamma_variates(a, rng, shape)` helper in a small
    shared module.
  - Have `forward` call `_eta`.
- **Confidence:** n/a (quality).

### P-19. Dead code and needless branches
- **Category:** simplify
- **Location:**
  - `logistic.py:175`: `self.lam is None` can never hold (`lam` is always an array).
  - `dirichlet.py:460`: the `if n is not None` filter in `_parse_nested` (every slot is filled).
  - `logistic.py:154`: `_theta` evaluates `h.median()` for fixed hypers on every gradient call; it
    could be cached once at construction.
  - The `dlogs` rows for fixed hyperparameters are computed and never used.
- **Suggested fix:** remove the dead code; precompute the fixed θ vector in `__init__`.
- **Confidence:** n/a.

### P-20. Avoidable O(n³) and per-draw Python loops
- **Category:** efficiency
- **Location:**
  - `logistic.py:97`: three-operand `einsum("ij,jk,ki->i", …)` for the Rayleigh quotients.
  - `logistic.py:251-259`: `sample` loops over draws in Python.
- **Claim:**
  - With n = 1000 the einsum takes 0.77 s against 0.09 s for `np.sum(U * (Q @ U), 0)`. That is
    about half of `LogisticRandomWalk(2).bind` at K = 1000 (1.4 s, single-threaded BLAS).
  - `sample` could be vectorised: draw ξ as `(n, n_active)` and apply the basis to a matrix. Today
    it is 0.9 s for 20 000 GP draws at K = 48.
- **Evidence:** `$A/a10_rw_timing.py` (run with `OPENBLAS_NUM_THREADS=1`).
  - This container's multi-threaded OpenBLAS makes `eigh` on 1000 × 1000 take 24 s. That is an
    environment artefact, but a user on a similar machine would see a 40 s bind.
  - For order 1 the eigenbasis is analytic (DCT-II, λ = 2 − 2cos(πk/n)), so no `eigh` is needed.
- **Suggested fix:** replace the einsum. Optionally use the analytic order-1 basis, and vectorise
  `sample`.
- **Confidence:** confirmed by running.

### P-21. Spec table and theory text disagree with the code
- **Category:** docs
- **Location:** `docs/design/spec.md:303-304`; `logistic.py:214-249`; `base.py:130-132`
- **Claim:**
  - The spec describes `LogisticRandomWalk` as "eigenvectors of the RW/ICAR precision on
    `bins.neighbors()`" with `order=1|2`. The code uses per-axis difference matrices (a Kronecker
    sum) and allows order 3.
  - The spec limits `LogisticGP` to "1-D/2-D grids"; the code accepts any D.
  - `init_from_value` is documented as the unconstrained point "corresponding (approximately)" to
    the given fractions. For logistic priors it is a ridge-regularised fit that shrinks towards the
    prior mean. `LogisticNormal()` at K = 200 misses a Gaussian-shaped ρ by up to a factor e^1.16
    (`$A/a13_misc.py`). This is fine as an initialiser, but the name and docstring suggest an
    inverse.
- **Suggested fix:** update the spec rows and say "ridge-shrunk initial point" in the docstring.
- **Confidence:** confirmed by running and reading.

### P-22. Validation timing is inconsistent across priors
- **Category:** consistency
- **Location:**
  - `Dirichlet`, `GammaRates` and `PolyaTree(c)` validate in `__init__`.
  - `DirichletTree` alphas (positivity), `GeneralizedDirichlet` (`a`, `b`, `order`),
    `LogisticNormal(cov)` and `LogisticGP(n_basis)` validate only at `bind`, or not at all
    (P-8, P-10).
- **Claim:** errors surface late, inside `infer`, far from the user's constructor call.
- **Suggested fix:** validate everything that does not need `bins` in `__init__`.
- **Confidence:** confirmed by reading.

---

## Checked and found correct

- **Gradients.** x-space log-density and gradient match finite differences (≤ 2e-9 relative) for:
  - `Dirichlet` (scalar and vector α), `DirichletTree`, `GeneralizedDirichlet` (with `order`),
    `PolyaTree` (log bins) and `GammaRates`;
  - `LogisticNormal` (iid and full cov), RW orders 1, 2 and 3 (1-D and 2-D grid), and GP with all
    four kernels (1-D, 2-D ARD, log bins);
  - each logistic prior at λ ≡ 1, λ ≡ 0 and random intermediate λ, apart from P-5
    (`$A/a01_grad_fd.py`).
- **Normalisation and Jacobians.** The simplex densities integrate to 1 on the 2-simplex. The
  x-space density equals `log_density + Σ log ρ + ½ log K` to 1e-15. `GammaRates` marginals
  integrate to 1 (`$A/a04_conjugate.py`).
- **`prior_transform` vs `sample`.** KS p-values are 0.05–0.98 over all components for:
  - `Dirichlet`, a two-level tree, `GeneralizedDirichlet` with permuted order;
  - `PolyaTree` with K = 3 (custom base) and K = 4 (log bins);
  - a 5-leaf mixed-depth tree.
- **Conjugate updates.** Checked against importance sampling from the prior (4e5 draws):
  - `posterior_sample` means, `vb_mean`, `vb_sample` means and `vb_expected_log` all agree to MC
    error;
  - `map_from_counts` matches numerical optimisation whenever the posterior density is bounded.
    In unbounded cases it returns the documented boundary convention.
- **Tree algebra.** The node coefficients α_pa(c),c − A_c, the `slot_nodes` subset construction,
  the θ-space MAP, and the `unbounded_density` criterion (any branch coefficient < 0) were
  re-derived and checked.
- **Partial centring.** `recentre` preserves η exactly (3e-17). `lp_λ − lp_1` equals the Jacobian
  Σ(λ−1) log s exactly. The u-clip at ±350 returns −inf.
- **HSGP.** The sine eigenfunctions and eigenvalues, and the SE and Matérn spectral densities in D
  dimensions with ARD (including the ℓ^D factor and the `dlogs`), are correct. The HSGP
  covariance matches the exact kernel to < 0.01 at ℓ = 0.05–0.1, with field variance ≈ amplitude²
  (`$A/a05_hsgp_cov.py`). The errors that remain are P-2 and P-3.
- **RW eigenbases.** For orders 1–3 and n up to 3000 (order 1), 1000 (orders 2 and 3):
  - orthonormal to 2e-15, Q reconstructed to 2e-15, null residual ≤ 2e-15;
  - eigenvalues match `eigvalsh`; order 3 at n = 1000 has the smallest eigenvalue 6.26e-14 vs
    6.11e-14, an intrinsic conditioning limit.
  - The N-D implied precision equals the Kronecker sum to 7e-15, with the correct null-mode count
    (`$A/a09_rw_eigen.py`, `$A/a11_rw_nd.py`).
- **Hyperpriors.** All five integrate to 1 in u, including the Jacobian. `dlogpdf_u` matches FD,
  `median()` is the true median, and `sample` matches the density. `InverseGamma.from_quantiles`
  hits both tails to < 1e-6 for 60 ratios from 1.01 to 1e9 and for tail values 0.001–0.4, so the
  earlier fix #9 is complete (`$A/a12_hyper.py`).
- **Covariance.** `LogisticNormal(cov)` draws reproduce P C P (max error 0.004 at n = 1e5).
- **Defaults and errors.** `_resolve_mean` gives log-volume defaults, including N-D and categorical
  (ones). The error messages for categorical bins in the smoothness priors and `PolyaTree`, and for
  RW axes shorter than `order + 1`, are clear.
