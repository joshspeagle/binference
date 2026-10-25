# Audit of frankenz population and hierarchical inference

**Status:** complete (2026-10-22). Each claim below rests on a derivation given here or on a numerical
result from a script in the appendix. All scripts use fixed seeds and run single-threaded; their outputs
are saved as JSON.

**Materials audited.** `joshspeagle/frankenz` shallow clone at commit `b753e64` ("HB"):
`frankenz/samplers.py` (535 lines: `loglike_nz`, `population_sampler`, `hierarchical_sampler`),
helpers in `frankenz/pdf.py` (`gaussian`, `gaussian_bin`, `gauss_kde`), and the tutorials
`demos/5 - Population Inference with Redshifts.ipynb` (NB5) and
`demos/6 - Hierarchical Inference with Redshifts.ipynb` (NB6). Notebook cell numbers are 0-based
indices into each `.ipynb` file, counting markdown and code cells together. Line numbers refer to
`samplers.py` unless stated otherwise. The tutorials' pickled mock (`data/mock_sdss_cww_bpz.pkl`) is not
in the repository, so every test uses mocks built here (described with each test).

**Environment.** Python 3.13, NumPy 2.5.4, SciPy 1.18.1, one thread (`OMP/OPENBLAS/MKL_NUM_THREADS=1`)
on a shared 4-CPU machine. Wall-clock numbers are single measurements (not repeated) and should be
read as indicative. `samplers.py` and `pdf.py`
were copied verbatim (SHA-256 checked) into the scratch folder and imported from there. **No patches were
needed** for modern NumPy/SciPy, and the clone was not modified.

## Summary

1. **`population_sampler` is not a valid MCMC sampler (critical).** Its proposal width depends on the
   current state (gradient-scaled and capped at the smaller bin fraction). The acceptance test omits the
   Hastings factor that this requires, and the width is frozen across the inner MH steps. The effects,
   measured against exact quadrature and validated references:
   - a bin whose posterior reaches zero is absorbed at ~1e-13 within 2-15 saved samples at the notebook
     defaults, shifting the other bins by up to 1.2 posterior SD;
   - with no bin near zero, every posterior SD is inflated by 8-18%;
   - in a Geweke test the chain collapses onto a vertex of the simplex.

   The same algorithm with the Hastings factor and a per-step width is exact, which pins down the cause.
2. **`hierarchical_sampler` is correct.** It is data-augmentation Gibbs with a conjugate Dirichlet step. It
   matches the exact K=3 posteriors (rms z = 0.94 over 132 statistics), matches the K=15 reference
   (|z| <= 2.0), and passes the Geweke test.
3. **The two samplers target the same marginal posterior but do not produce the same answer.** With
   `population_sampler` given a Dir(alpha+m) prior and `hierarchical_sampler` given `alpha=alpha+m`, the
   targets are provably identical (Section 1.1). The numerical discrepancy is entirely
   `population_sampler`'s.
4. **The "flexible reference" step is not the full conditional of any joint model.** This is proved in
   Section 3.3. It fails a Geweke test (max |z| = 70, while an exact step passes at 1.7). In practice it
   discards an uncontrolled share of a representative reference sample, inflating SDs by 1.1-1.9x; the
   amount is set by N/N_ref, not by any stated degree of trust in the reference.
5. **Inputs must be likelihoods.** Posteriors computed under a non-uniform interim prior bias rho by up
   to 12.6 posterior SD and cut 68%/95% coverage to 0.29/0.52; dividing by the interim prior removes the
   bias. NB5 and the `population_sampler` docstring call the inputs "PDFs".
6. **Two equations in NB5 would bias results if implemented as written.**
   - Cell 25 normalises over objects instead of bins: biases up to 50 SD.
   - Under the top-hat basis the likelihood must be bin-averaged; cell 14 and `pdf.gaussian_bin` use the
     bin integral. With unequal bins this biases results by up to 28 SD.
7. **API bugs.**
   - `run_mcmc` never continues from the last sample; both samplers restart from the stacked PDFs.
   - `hierarchical_sampler.reset()` leaves the old samples in place.
   - `population_sampler` fails with `np.random.Generator`.
   - Bins with zero stacked mass (37 of 80 in the tutorial set-up) are frozen at exactly 0 by
     `population_sampler`.
   - There is no input validation.
8. **Cost at tutorial scale (N=20000, K=80).** `population_sampler` takes 0.42 s per saved sample and
   `hierarchical_sampler` 0.35 s at thin=5. 99% of a hierarchical sweep is a Python loop that a vectorised
   object step speeds up 6-9x. In a tutorial-like mock, the notebooks' hierarchical runs carry an
   effective sample size of only about 120 per bin.
9. **Mixing.** DA Gibbs mixes badly at the bin level once likelihoods are as wide as the bins. With a
   flat prior the median IAT goes 7 -> 74 -> 162 -> 180 sweeps for sigma/width = 0.5, 1, 2, 4; an
   informative reference prior caps it near 17. A simple Laplace-whitened HMC in additive-log-ratio
   coordinates (ALR, i.e. log-ratios to the last bin) has IAT 1-14:
   - 1.4-10x the per-bin ESS/s of vectorised DA, and 36-270x that of the original code;
   - for smooth functionals such as the mean redshift, vectorised DA remains competitive.

---

## 1. What frankenz implements, in clean notation

### 1.1 Model

| symbol | meaning |
|---|---|
| $i=1..N$ | objects, data $d_i$ |
| $k=1..K$ | bins with edges $e_0<\dots<e_K$, widths $\Delta_k=e_k-e_{k-1}$ |
| $\ell_i(z)=p(d_i\mid z)$ | per-object likelihood (an arbitrary per-object constant is allowed) |
| $L_{ik}=\Delta_k^{-1}\int_{e_{k-1}}^{e_k}\ell_i(z)\,dz$ | **bin-averaged** likelihood |
| $\rho\in\mathcal S^{K-1}$ | population bin fractions (probability mass per bin), $\sum_k\rho_k=1$ |
| $n(z\mid\rho)=\sum_k\rho_k\,\mathbb 1[z\in k]/\Delta_k$ | top-hat (histogram) population density |
| $\alpha$; $m$; $N_{\rm ref}=\sum_km_k$ | Dirichlet concentration; reference-sample bin counts (true labels) |
| $z_i\in\{1..K\}$, $n_k=\#\{i:z_i=k\}$ | latent bin labels and their counts |
| $\pi_k$, $p_{ik}$ | interim prior mass per bin, and the per-object posteriors a photo-z code reports under it |

Marginal likelihood of one object under the histogram basis:
$p(d_i\mid\rho)=\int\ell_i(z)\,n(z\mid\rho)\,dz=\sum_k\rho_k L_{ik}=L_i\cdot\rho$.

Collapsed ("population") posterior, with a representative reference sample folded into the prior:

$$p(\rho\mid d,m)\;\propto\;\prod_k\rho_k^{\alpha_k+m_k-1}\;\prod_{i=1}^N L_i\cdot\rho .$$

Augmented ("hierarchical") form: $z_i\mid\rho\sim{\rm Cat}(\rho)$ and $p(d_i\mid z_i=k)=L_{ik}$, so
$p(\rho,z\mid d)\propto{\rm Dir}(\rho;a)\prod_i\rho_{z_i}L_{iz_i}$ with $a=\alpha+m$.

Since $\sum_{z_i}\rho_{z_i}L_{iz_i}=L_i\cdot\rho$, summing over $z$ gives exactly the collapsed posterior.
Hence **the two frankenz samplers target the same marginal posterior for $\rho$** when
`population_sampler` uses a ${\rm Dir}(\alpha+m)$ log-prior (NB5 cells 68/71) and `hierarchical_sampler`
uses `alpha` $=\alpha+m$ (NB6 cell 53).

Full conditionals of the augmented form: $P(z_i=k\mid\rho,d_i)=L_{ik}\rho_k/(L_i\cdot\rho)$ and
$\rho\mid z\sim{\rm Dir}(a+n)$.

The likelihood is invariant to per-object constants, so normalising each row of $L$ is harmless. The
bin average (rather than the integral) matters as soon as bins have unequal widths (F7).

**Likelihoods vs posteriors.** Suppose a code reports $p_{ik}\propto\int_k\ell_i\pi\,dz$ under an
interim prior that is constant within bins with bin masses $\pi_k$. Then $p_{ik}\propto\pi_kL_{ik}$, and
the correct input is $L_{ik}\propto p_{ik}/\pi_k$. Feeding $p_{ik}$ instead gives
$\prod_i\sum_kL_{ik}\pi_k\rho_k$, i.e. the interim prior is counted twice.

The tutorials are unaffected because their "PDFs" are Gaussian likelihoods normalised over $z$ on
equal-width bins, i.e. posteriors under a flat interim prior.

**Stacking.** The stack $N^{-1}\sum_ip_{i}$ equals one EM step from the interim prior (checked to 3e-17).
The MLE is instead the self-consistent fixed point $\rho=N^{-1}\sum_iL_i\odot\rho/(L_i\cdot\rho)$, so
stacking is biased whenever the likelihoods are not sharp (Section 3.5).

**Gaussian mock used by the tutorials.** Data are generated as $\mu_i\sim\mathcal N(z_i,\sigma_i^2)$, so
$\ell_i(z)=\mathcal N(\mu_i;z,\sigma_i^2)$ and
$\int_k\ell_i\,dz=\Phi((e_k-\mu_i)/\sigma_i)-\Phi((e_{k-1}-\mu_i)/\sigma_i)$.

This is what `pdf.gaussian_bin` returns (verified to 3e-16). It is the bin *integral*
$\Delta_kL_{ik}$, which is harmless only for equal widths.

### 1.2 `population_sampler` (samplers.py:83-308)

State $\rho$ with cached overlaps $o_i=L_i\cdot\rho$. Target
$\pi(\rho)\propto{\rm prior}(\rho)\prod_io_i$; `loglike_nz` is at l.24-80, the prior is a user
function, and it is flat if omitted. The reported `lnpost` is $\log{\rm prior}+\sum_i\log o_i$.

One saved sample is `thin` (default 400) pair updates. Each pair update:

1. Draw an ordered pair $(j,l)$, $j\neq l$, uniformly; the direction $t=e_j-e_l$ keeps $\sum\rho=1$.
2. Set $h=10^{-4}\min(\rho_j,\rho_l,1-\rho_j,1-\rho_l)$ and take a central finite difference
   $g=[\log\pi(\rho+th/2)-\log\pi(\rho-th/2)]/h$ (l.272-282).
3. Set the scale $s(\rho)=\min(1/|g|,\;\rho_j,\rho_l,1-\rho_j,1-\rho_l)$ (l.284-287).
4. Make `mh_steps` (default 3) Metropolis steps: propose $\rho'=\rho+t\varepsilon$ with
   $\varepsilon\sim\mathcal N(0,s^2)$ and accept with $\min(1,\pi(\rho')/\pi(\rho))$
   (l.290-305). $s$ is **held at its value from the start of the pair update**.

Overlaps are updated incrementally, $o_i\leftarrow o_i+\varepsilon(L_{ij}-L_{il})$. This is exact and
costs $O(N)$ per proposal.

### 1.3 `hierarchical_sampler` (samplers.py:311-535)

Each saved sample follows `thin` (default 5) sweeps of:

1. $z_i\sim{\rm Cat}(L_i\odot\rho/L_i\cdot\rho)$ for every object, via a Python loop of
   `multinomial(1, .)`, giving $n$ (l.519-520);
2. $\rho\sim{\rm Dir}(\alpha+n+k)$, with $k=0$ when there is no reference sample (l.522);
3. the *flexible reference* step, only when `ref_sample`$=m$ is passed:
   $k\sim{\rm Mult}\big(N_{\rm ref},(m+\beta+N\rho)/(N_{\rm ref}+B+N)\big)$, with $B=\sum_k\beta_k$ and
   $\beta$ defaulting to 1 (l.524-527).

Steps 1-2 form an exact two-block Gibbs sampler (data augmentation) for $p(\rho,z\mid d)$. Step 3 is
analysed in Section 3.3 (finding F2).

The reported `lnpost` is $\log{\rm Mult}(n\mid N,\rho)+\log{\rm Dir}(\rho\mid\alpha+k)$, plus
$\log{\rm Mult}(k\mid N_{\rm ref},(m+\beta)/\sum)$ in the flexible case. This is not the log joint
posterior (F12).

### 1.4 What the tutorials do

**NB5:**
1. Builds Gaussian "PDFs" ($\sigma\sim U(0.05,0.2)$, 80 bins of width 0.1 on $[-1,7]$).
2. Shows that stacking is biased, as is a deconvolution problem.
3. Shows that Poisson, multinomial and "PDF-draw" resampling of the stack under-state the uncertainty.
4. Runs `population_sampler` with a ${\rm Dir}(1+m)$ prior from 1000 reference redshifts (750 saved
   samples).

**NB6:**
1. Repeats the set-up.
2. Runs `hierarchical_sampler` with `alpha` = 1 + m (600 saved samples, thin 5).
3. Shows hierarchical shrinkage of per-object PDFs.
4. Introduces the flexible reference step.

---

## 2. Findings

Severity: **critical** = can silently give wrong scientific answers; **major** = wrong behaviour or
documentation that is likely to cause wrong answers or large waste; **minor** = edge cases, robustness,
performance, or imprecise claims; **cosmetic** = notation or wording.

| ID | Location | Severity | Description | Evidence | Suggested fix for pastry |
|---|---|---|---|---|---|
| F1 | samplers.py:272-305 (scale l.272-287, MH loop l.290-305); NB5 cell 69 | critical | Pairwise MH whose proposal width $s(\rho)=\min(1/\lvert\partial_t\log\pi\rvert,\rho_j,\rho_l,1-\rho_j,1-\rho_l)$ depends on the state, accepted **without the Hastings factor**, with $s$ also frozen at its pre-update value for all `mh_steps` steps. The kernel does not leave the posterior invariant: bins whose posterior reaches 0 are absorbed at ~1e-13 and the other bins' SDs are inflated. | Section 3.1 (A0-A3: z up to 1.9e2; interior SDs +11-12%); Section 3.2 ablation (each defect alone breaks invariance, the corrected kernel is exact); Section 3.3 Geweke (chain ends on a simplex vertex); Section 3.4 (K=15: three tail bins absorbed, SDs +8-18%) | Do not port. Default to HMC/NUTS in unconstrained coordinates (ALR/ILR, with the log-Jacobian $\sum_k\log\rho_k$) or to vectorised DA Gibbs. A derivative-free fallback must be a provably valid kernel: slice sampling along pair directions, or MH with a per-step state-dependent scale **and** its Hastings factor. Gate every kernel with the Geweke and exact-K=3 tests (Section 5.4). |
| F2 | samplers.py:482-490, 503-505, 524-529; NB6 cells 62-63, 65, 71 | major | The flexible-reference step $k\sim{\rm Mult}(N_{\rm ref},(m+\beta+N\rho)/(N_{\rm ref}+B+N))$ is not a full conditional of any joint distribution that has $\rho\mid z,k\sim{\rm Dir}(\alpha+n+k)$, so the chain's stationary distribution is not a posterior of any model. For $N\gg N_{\rm ref}$ it ignores the reference entirely; in between it discards an uncontrolled share of it. | Section 3.3: proof; compatibility cross-difference 5.33 (0 required); Geweke max abs z 70.2, against 1.7 for an exact enumerated step; Section 3.4: for a representative reference, SDs x1.1-1.9 relative to the fixed reference, means shifted by up to 2.5 SD | Replace with an explicit model that has a representativeness parameter, e.g. $\rho\mid\rho_{\rm ref},c\sim{\rm Dir}(c\,\rho_{\rm ref})$ with $\rho_{\rm ref}\mid m\sim{\rm Dir}(\beta+m)$ and $c$ fixed or given a prior ($c\to\infty$: representative; $c\to0$: ignored). Alternatively, a power prior ${\rm Dir}(\alpha+\lambda m)$ with $\lambda\in[0,1]$. Use exact or MH steps and Geweke-test them. |
| F3 | samplers.py:171-191 and 403-420 | major | `run_mcmc` computes the continuation point (`pos = self.samples[-1]`, l.175/407) but passes `pos_init=pos_init` (None) to `sample` (l.187/418). Every call therefore restarts from the stacked PDFs, contradicting the docstring, and repeated calls silently concatenate new burn-ins. | Section 3.8: the second call's first sample equals the stacked PDF exactly; a spy shows `pos_init=None` | Explicit, resumable sampler state (rho, latent counts, RNG state, adaptation). Test that `run(a); run(b)` equals `run(a+b)` bit for bit with a fixed seed. |
| F4 | samplers.py:336-341 | major | `hierarchical_sampler.reset()` resets `samples_lnp`, creates unused `samples_prior`/`samples_counts`, and leaves `samples`; `results` then returns arrays of different lengths. | Section 3.8: 8 samples vs 3 lnp values | One results container; a reset test. |
| F5 | samplers.py:84-99 and 312-317 docstrings; NB5 cells 6, 15, 64; NB6 cell 41 | major | **Both** samplers need likelihoods, or posteriors divided by the interim prior. `population_sampler`'s docstrings and NB5 call the inputs "PDFs". The `hierarchical_sampler` docstring and NB6 cell 41 say likelihoods are required "since the prior is being explicitly modeled", which implies the population model can take posteriors; it cannot, because both models share the same likelihood for rho. Posterior inputs double-count the interim prior. | Section 3.5: up to 12.6 SD bias at N=20000; 68%/95% coverage 0.29/0.52; dividing by the interim prior restores 0.73/0.94 | API takes likelihoods. Any posterior input path requires the interim prior and divides it out internally. Document this and test it. |
| F6 | NB5 cell 25 | major (docs) | $P(g\mid h)$ is normalised by $\sum_g$ (over objects). It should be left unnormalised or normalised per object over bins, as the code does (cell 28, `axis=1`). | Section 3.5: implemented literally, MLE errors up to 49.5 SD | Document normalisation invariance (per-object constants are irrelevant) and test it. |
| F7 | NB5 cell 14; pdf.py:428-441 | minor (major for unequal bins) | Under the top-hat basis the factor multiplying $\rho_k$ is the bin **average** $\Delta_k^{-1}\int_k\ell_i$. Cell 14 says "evaluate the integral" and `gaussian_bin` returns the integral. This is harmless for the tutorials' equal widths. | Section 3.5: with unequal bins, integrated L puts alternate bins at exactly 0 (errors up to 28.3 SD); averaged L is unbiased (<= 1.5 SD) | Build L as integral/width internally; add an unequal-bin test. |
| F8 | samplers.py:498-499, 519-520 | major (performance) | The object step is a Python loop of N `multinomial(1, .)` calls plus a list-to-array sum: 68 of the 69 ms per sweep at N=20000, K=80. | Section 3.6: a vectorised inverse-CDF step takes 11 ms (float64) or 7.3 ms (float32) | Vectorised categorical sampling. Accumulate Rao-Blackwellised responsibilities on the fly. |
| F9 | samplers.py:264-305; NB5 cell 68 | minor (performance) | Each pair update costs `rstate.choice`, 2 finite-difference evaluations and 3 proposals, each O(N) in Python, plus user prior calls (the notebook's scipy Dirichlet adds ~25%). That is 1.05 ms per pair and 0.42 s per saved sample at tutorial scale. The derivative is available analytically at the same cost. | Section 3.6 | Moot given F1. Any coordinate-wise scheme should use analytic derivatives and vectorised priors. |
| F10 | samplers.py:272-282 | minor (numerics) | The finite-difference step $h=10^{-4}\min(\rho_j,\dots)$ suffers cancellation: relative error 2e-6 at rho=1e-6 and 1.5e-3 at 1e-9. At rho <= 1e-12 the difference is exactly 0, so the scale falls back to $h=10^{-4}\rho$ and the bin freezes; this is the ~1e-13 plateau in Section 3.2. | Section 3.8 | Analytic directional derivative $\sum_i(L_{ij}-L_{il})/o_i$. |
| F11 | samplers.py:178, 254, 272-287 | minor | Bins with exactly zero stacked mass at initialisation (no object has support; 37 of 80 in the tutorial set-up) get scale 0, hence NaN proposals, and stay at exactly 0 forever. Their posterior mass is about $\alpha_k/(N+A)$. | Sections 3.6 and 3.8 (0.0 vs a valid posterior mean of 0.0032) | Initialise in the interior, e.g. $({\rm stack}\cdot N+\alpha)/(N+A)$ or a Dirichlet draw; warn about bins without likelihood support. |
| F12 | samplers.py:507-513, 528-532; NB6 cells 36, 53, 65 | minor | The hierarchical "lnpost" is $\log{\rm Mult}(n\mid N,\rho)+\log{\rm Dir}(\rho\mid\alpha+k)$ [+ reference term]. It omits $\sum_i\log L_{iz_i}$ and includes the multinomial coefficient, so it is not the log joint posterior and cannot be compared with `population_sampler`'s lnpost (NB6 prints -64343 vs 224.5). It is also computed with SciPy every sweep. | Derivation; Section 3.8 | Report the collapsed log posterior $\log{\rm Dir}(\rho;a)+\sum_i\log L_i\cdot\rho$ at saved draws only. |
| F13 | samplers.py:163-164, 244-245, 292, 304, 394-395, 473-474 | minor | `rstate=None` uses the global `np.random` state. `population_sampler` needs the legacy `RandomState` API (`randn`) and fails with `np.random.Generator`. | Section 3.8 (AttributeError) | Accept a seed or Generator; use `SeedSequence.spawn` for chains. |
| F14 | samplers.py:102, 172-182, 300, 482-487 | minor | No input validation:<br>- `pos_init` as a list raises TypeError;<br>- a `pos_init` off the simplex is kept silently (samples summed to 2.0);<br>- the user prior is evaluated at infeasible proposals, so an unguarded `scipy.stats.dirichlet.logpdf` raises ValueError;<br>- integer `ref_sample` with integer `beta` raises UFuncTypeError;<br>- a row with $L_i\cdot\rho=0$ makes the object step raise ValueError (NaN probabilities). | Section 3.8 | Validate shapes, finiteness, non-negativity, row sums and simplex initialisation; never evaluate the prior outside its support. |
| F15 | samplers.py:483-513; NB6 cells 60, 70 | minor | Latent state is not persisted between calls: the reference counts are reset to $m$ and an extra initial sweep is run. $z$ and the counts are never stored (`samples_counts` is unused). The shrinkage plots therefore redraw 10 categorical samples per rho draw instead of averaging $L_j\odot\rho/(L_j\cdot\rho)$ (Rao-Blackwell). | Code reading | Store Rao-Blackwellised per-object posteriors and counts; make the state resumable. |
| F16 | pdf.py:436-439 | minor | `gaussian_bin` takes differences of $0.5(1+{\rm erf})$, whose absolute error is ~1e-16. The relative error is 28% at 8 sigma and the result is exactly 0 beyond ~8.3 sigma. An object far outside the bin range gives an all-zero row, which becomes NaN after normalisation. | Section 3.8 | Use `log_ndtr` differences with tail symmetry; flag zero rows. |
| F17 | pdf.py:499-524, 513-516 (same in `gauss_kde_dict`, pdf.py:594-597) | minor | `gauss_kde` with `wt_thresh=None`: the CDF branch sorts weights ascending and keeps `y_cdf <= 1 - cdf_thresh`, so it keeps the lowest weights and always drops the highest-weight object(s). Kernels with sigma < dx/5 are silently dropped, and the window $[c-o,c+o)$ is asymmetric. In the tutorials it is used only to plot the truth. | Section 3.8: a weight-100 object and a sigma=0.001 kernel both get zero mass | If pastry needs this, bin kernels by exact CDF differences and test mass conservation. |
| F18 | NB5 cell 62 | minor | "PDF draws ... capture the correct covariance structure overall, but severely underestimate the uncertainties". On the aggregated bin pairs plotted this is roughly right: correlations match, SDs are 1.8-3.4x too small. **Per bin it is wrong:** next-neighbour correlations have the opposite sign (-0.15 vs +0.35 to +0.48), the signature of deconvolution. | Section 3.5 (s09) | Show the posterior's per-bin covariance in the docs. |
| F19 | NB5 cells 7, 16, 75; NB6 cells 54, 66 | minor | Statements that are true but imprecise or unsupported:<br>- "no guarantee [the stack] will maximize the posterior": precisely, the stack is one EM step from the interim prior and the MLE is the EM fixed point;<br>- cell 16 calls the multinomial MLE the posterior maximum (the mode under Dir(1); the posterior mean is $(N_h+\alpha_h)/(N+A)$);<br>- cell 75 attributes the discrepancies "near zero" to the prior (garbled sentence), but F1 absorbs exactly such bins;<br>- NB6 54/66 call the hierarchical model "more natural ... easier (and quicker)": for $\rho$ it is the same model, the cost per saved sample is similar (0.35 vs 0.42 s), and the real difference is validity. | Section 3.5 (EM identity to 3e-17; stack bias 21 SD); Sections 3.1 and 3.6 | Use the precise statements in pastry's docs. |
| F20 | NB5 cells 6, 12, 15, 32, 64, 69; NB6 cells 16, 33, 45, 47; samplers.py docstrings (l.370-372, 465-467) | cosmetic | Notation and wording:<br>- $P(z\mid\mathbf g)$, $P(z\mid g)$ and $P(z_g\mid z)$ serve as population, individual posterior and likelihood;<br>- data are written as $\{p_g\}$ or $\{h_g\}$, and cell 15's left-hand side conditions on bin labels while the right-hand side sums over them;<br>- cell 64 / NB6 33: the middle term "$\ln P(p_{\mathbf g})$" should be $\ln P(\rho)$;<br>- NB6 45: ${\rm Cat}(p_g)$ should be ${\rm Cat}(p'_g)$;<br>- NB6 47: "$n'\sim{\rm Mult}(N_{\mathbf g},\rho')$" holds for the prior predictive, not for the counts the object step produces;<br>- NB5 69 calls an MH-within-Gibbs scheme "Gibbs";<br>- the PIT test (cell 32) cannot distinguish a flat-prior posterior from the correct posterior (both are PIT-uniform);<br>- the hierarchical docstrings mention "random pairs" and a nonexistent `verbose` argument for `sample`. | Reading | Adopt the notation of Section 1.1 ($i,k,\ell_i,L_{ik},\pi_k,p_{ik},\rho,\alpha,m,z_i,n_k$). |
| F21 | NB5 cells 71-72; NB6 cells 36-37, 53, 55, 65, 67 | minor | No convergence diagnostics. In a tutorial-like mock the 600-sample, thin-5 hierarchical runs have ESS of about 120 per bin. DA mixing degrades sharply for broad likelihoods without an informative prior. | Section 3.7 | Report R-hat, ESS and MCSE by default; run several chains. |

---

## 3. Numerical results

All runs are single-threaded, use fixed seeds, and state their settings with each table. MCSEs use
Geyer's initial monotone sequence ESS, summed over chains. 10% burn-in is dropped unless stated otherwise.

Each comparison is reported as a z-score:
- "z" = (estimate - exact) / MCSE;
- for the CDF checks, z = (empirical CDF at the exact p-quantile - p) / sqrt(p(1-p)/ESS of the
  indicator chain), with p in {0.025, 0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95, 0.975}.

A valid sampler should give z-values that look like N(0, 1) draws.

### 3.0 Validation of the audit's own tools (`s00_validate_tools.py`)

* **Quadrature (`exact_k3`).** Gauss-Legendre in one coordinate, Simpson / cumulative trapezoid on a
  4001-point grid in the other, with three independent orderings.
  - With no data it reproduces the analytic Beta marginals of Dir(3,5,2) and Dir(1,1,1) to < 2e-13 in
    means and SDs and < 3e-7 in nine quantiles. Dir(1,1.5,40) is reproduced to 2e-6 in moments and
    1.2e-5 in quantiles; it has an integrable square-root singularity at 0, and none of the test
    posteriors do.
  - On a data problem (K=3, N=300) the three normalisers agree to 2e-14, and the means and SDs match an
    independent 1500x1500 midpoint grid to all printed digits.
* **Re-implementation of `population_sampler`** (`common.pairwise_sampler`, `corrected=False`). It
  reproduces the original class **bit for bit** (50 saved samples, K=5, N=200, same `RandomState`).
  Ablations built on it therefore differ from frankenz only in the component switched off or on.
* **Reference samplers.** These are vectorised DA Gibbs, independence MH (IMH) in ALR space with a t5
  proposal, fixed-covariance random-walk MH (RWM) in ALR space, and Laplace-whitened HMC in ALR space.
  - Each matches the exact posteriors of Section 3.1 (calibration table below).
  - HMC was validated separately on A1 and A3: rms z 0.64 and 0.89, max abs z 2.2 (`s07a_validate_hmc.py`).

### 3.1 Task (a): exact K=3 reference (`s01_k3_exact.py`)

**Likelihoods.** Bin-averaged Gaussian likelihoods on edges (0,1,2,3); true z uniform within bins;
$\mu_i=z_i+\sigma_i\epsilon_i$.

**Runs.** Two chains per sampler, with a 25 s wall-clock budget per chain for the frankenz samplers and
the corrected control.
- frankenz `population_sampler`: thin=3 pair updates per saved sample, mh_steps=3, log-prior
  $\sum_k(\alpha_k-1)\log\rho_k$ (equal to the notebook's `scipy.stats.dirichlet.logpdf` up to a constant).
- `hierarchical_sampler`: thin=1.

**Cases.** A0 is an interior case (informative likelihoods, no bin near zero). A1-A3 each have one bin
whose posterior reaches zero.

**Case A0**: true rho = [0.5, 0.3, 0.2], N = 300, sigma_i ~ U(0.1, 0.3) (bin width 1, edges 0,1,2,3), alpha = (1,1,1), data seed 100. Exact: rho1 = 0.4318 (sd 0.0317), rho2 = 0.3636 (sd 0.0334), rho3 = 0.2045 (sd 0.0255). Quadrature checks: normaliser spread 7.6e-14, sum of means - 1 = 2.2e-16.

| sampler | n kept (2 chains) | mean rho1 (z) | mean rho2 (z) | mean rho3 (z) | sd rho1 (z) | sd rho2 (z) | sd rho3 (z) | max abs z, 27 CDF checks | min ESS |
|---|---|---|---|---|---|---|---|---|---|
| frankenz `hierarchical_sampler` | 48330 | 0.4320 (+0.6) | 0.3633 (-1.5) | 0.2047 (+1.4) | 0.0319 (+1.8) | 0.0335 (+0.8) | 0.0256 (+1.3) | 2.9 | 24823 |
| frankenz `population_sampler` | 133200 | 0.4317 (-0.7) | 0.3642 (+2.1) | 0.2042 (-2.2) | 0.0352 (+21.3) | 0.0370 (+22.0) | 0.0286 (+25.2) | 29.8 | 24648 |
| pairwise + Hastings fix (control) | 58500 | 0.4318 (-0.1) | 0.3637 (+0.3) | 0.2044 (-0.4) | 0.0315 (-1.4) | 0.0331 (-1.4) | 0.0255 (+0.0) | 2.0 | 12506 |
| vectorised DA Gibbs (mine) | 180000 | 0.4317 (-1.8) | 0.3637 (+0.5) | 0.2046 (+1.6) | 0.0317 (-1.1) | 0.0334 (+1.1) | 0.0255 (+0.3) | 2.9 | 91151 |
| IMH, ALR t5 (mine) | 900000 | 0.4318 (-0.3) | 0.3637 (+0.5) | 0.2045 (-0.3) | 0.0317 (-1.1) | 0.0333 (-1.9) | 0.0255 (-0.4) | 2.0 | 563023 |
| RWM, ALR (mine) | 270000 | 0.4318 (-0.5) | 0.3637 (+0.4) | 0.2045 (+0.1) | 0.0317 (-0.8) | 0.0333 (-0.8) | 0.0254 (-0.6) | 1.3 | 36104 |

**Case A1**: true rho = [0.5, 0.3, 0.2], N = 300, sigma_i ~ U(0.4, 1.2) (bin width 1, edges 0,1,2,3), alpha = (1,1,1), data seed 101. Exact: rho1 = 0.6333 (sd 0.0480), rho2 = 0.1019 (sd 0.0627), rho3 = 0.2648 (sd 0.0422). Quadrature checks: normaliser spread 2.1e-13, sum of means - 1 = -2.1e-14.

| sampler | n kept (2 chains) | mean rho1 (z) | mean rho2 (z) | mean rho3 (z) | sd rho1 (z) | sd rho2 (z) | sd rho3 (z) | max abs z, 27 CDF checks | min ESS |
|---|---|---|---|---|---|---|---|---|---|
| frankenz `hierarchical_sampler` | 47700 | 0.6340 (+0.8) | 0.1009 (-0.6) | 0.2651 (+0.4) | 0.0482 (+0.6) | 0.0622 (-0.7) | 0.0418 (-1.5) | 1.3 | 1651 |
| frankenz `population_sampler` | 132750 | 0.6905 (+192.5) | 0.0000 (-5e+12) | 0.3095 (+150.4) | 0.0376 (-44.4) | 0.0000 (-7e+13) | 0.0376 (-19.6) | 224.4 | 8 |
| pairwise + Hastings fix (control) | 58950 | 0.6339 (+1.2) | 0.1016 (-0.4) | 0.2646 (-0.7) | 0.0480 (-0.1) | 0.0622 (-1.2) | 0.0417 (-2.5) | 2.3 | 10018 |
| vectorised DA Gibbs (mine) | 180000 | 0.6333 (+0.0) | 0.1020 (+0.1) | 0.2647 (-0.3) | 0.0480 (-0.4) | 0.0625 (-0.5) | 0.0422 (-0.1) | 0.9 | 6547 |
| IMH, ALR t5 (mine) | 900000 | 0.6334 (+0.3) | 0.1018 (-0.8) | 0.2649 (+1.0) | 0.0481 (+0.9) | 0.0628 (+1.2) | 0.0422 (+0.1) | 2.3 | 225173 |
| RWM, ALR (mine) | 270000 | 0.6335 (+0.8) | 0.1017 (-0.5) | 0.2648 (-0.1) | 0.0478 (-1.5) | 0.0627 (-0.1) | 0.0423 (+0.7) | 2.2 | 24431 |

**Case A2**: true rho = [0.69, 0.3, 0.01], N = 200, sigma_i ~ U(0.4, 1.2) (bin width 1, edges 0,1,2,3), alpha = (1,1,1), data seed 102. Exact: rho1 = 0.7981 (sd 0.0632), rho2 = 0.1797 (sd 0.0680), rho3 = 0.0222 (sd 0.0187). Quadrature checks: normaliser spread 1.4e-11, sum of means - 1 = 6.8e-13.

| sampler | n kept (2 chains) | mean rho1 (z) | mean rho2 (z) | mean rho3 (z) | sd rho1 (z) | sd rho2 (z) | sd rho3 (z) | max abs z, 27 CDF checks | min ESS |
|---|---|---|---|---|---|---|---|---|---|
| frankenz `hierarchical_sampler` | 69255 | 0.7985 (+0.5) | 0.1793 (-0.5) | 0.0222 (+0.4) | 0.0632 (+0.1) | 0.0680 (+0.0) | 0.0187 (+0.2) | 1.1 | 4684 |
| frankenz `population_sampler` | 135000 | 0.7826 (-27.9) | 0.2174 (+67.9) | 0.0000 (-2e+13) | 0.0732 (+23.9) | 0.0732 (+12.4) | 0.0000 (-1e+13) | 72.5 | 24 |
| pairwise + Hastings fix (control) | 59850 | 0.7989 (+1.2) | 0.1789 (-1.1) | 0.0222 (+0.3) | 0.0632 (+0.2) | 0.0680 (+0.1) | 0.0187 (-0.2) | 1.9 | 8463 |
| vectorised DA Gibbs (mine) | 180000 | 0.7976 (-1.0) | 0.1805 (+1.2) | 0.0219 (-1.8) | 0.0635 (+1.3) | 0.0682 (+0.8) | 0.0185 (-2.5) | 2.6 | 13099 |
| IMH, ALR t5 (mine) | 900000 | 0.7981 (+0.3) | 0.1797 (-0.6) | 0.0222 (+0.9) | 0.0632 (+0.2) | 0.0680 (+0.4) | 0.0187 (+1.3) | 1.5 | 345572 |
| RWM, ALR (mine) | 270000 | 0.7983 (+0.4) | 0.1795 (-0.5) | 0.0222 (+0.6) | 0.0630 (-0.8) | 0.0679 (-0.1) | 0.0188 (+0.7) | 1.8 | 27146 |

**Case A3**: true rho = [0.6, 0.4, 0.0], N = 50, sigma_i ~ U(0.4, 1.2) (bin width 1, edges 0,1,2,3), alpha = (1,1,1), data seed 103. Exact: rho1 = 0.6871 (sd 0.1221), rho2 = 0.2391 (sd 0.1370), rho3 = 0.0738 (sd 0.0559). Quadrature checks: normaliser spread 1.1e-13, sum of means - 1 = 3.0e-14.

| sampler | n kept (2 chains) | mean rho1 (z) | mean rho2 (z) | mean rho3 (z) | sd rho1 (z) | sd rho2 (z) | sd rho3 (z) | max abs z, 27 CDF checks | min ESS |
|---|---|---|---|---|---|---|---|---|---|
| frankenz `hierarchical_sampler` | 191385 | 0.6859 (-1.4) | 0.2406 (+1.3) | 0.0735 (-0.7) | 0.1217 (-1.0) | 0.1368 (-0.4) | 0.0560 (+0.4) | 1.9 | 15553 |
| frankenz `population_sampler` | 142650 | 0.6472 (-35.7) | 0.3528 (+101.7) | 0.0000 (-1e+14) | 0.1509 (+36.8) | 0.1509 (+17.8) | 0.0000 (-1e+14) | 119.5 | 10 |
| pairwise + Hastings fix (control) | 64350 | 0.6877 (+0.5) | 0.2387 (-0.3) | 0.0737 (-0.2) | 0.1220 (-0.1) | 0.1367 (-0.3) | 0.0559 (+0.1) | 1.4 | 8621 |
| vectorised DA Gibbs (mine) | 180000 | 0.6870 (-0.1) | 0.2392 (+0.1) | 0.0738 (+0.1) | 0.1220 (-0.2) | 0.1366 (-0.7) | 0.0558 (-0.4) | 1.5 | 13931 |
| IMH, ALR t5 (mine) | 900000 | 0.6872 (+0.5) | 0.2390 (-0.8) | 0.0738 (+0.7) | 0.1221 (+0.2) | 0.1371 (+0.5) | 0.0560 (+1.4) | 2.4 | 339878 |
| RWM, ALR (mine) | 270000 | 0.6877 (+0.8) | 0.2386 (-0.7) | 0.0738 (-0.0) | 0.1215 (-1.5) | 0.1364 (-1.3) | 0.0557 (-0.6) | 1.8 | 29509 |

Calibration of all z-scores (means, sds, CDF at exact quantiles; 33 per case):

| sampler | number of z | rms z | fraction abs z > 2 | fraction abs z > 3 | max abs z |
|---|---|---|---|---|---|
| frankenz `hierarchical_sampler` | 132 | 0.94 | 0.01 | 0.00 | 2.9 |
| frankenz `population_sampler` | 132 | 1.7e+13 | 0.75 | 0.71 | 1.3e+14 |
| pairwise + Hastings fix (control) | 132 | 0.91 | 0.02 | 0.00 | 2.5 |
| vectorised DA Gibbs (mine) | 132 | 0.98 | 0.06 | 0.00 | 2.9 |
| IMH, ALR t5 (mine) | 132 | 0.85 | 0.02 | 0.00 | 2.4 |
| RWM, ALR (mine) | 132 | 0.77 | 0.01 | 0.00 | 2.2 |


**Reading.** `hierarchical_sampler` and all valid reference samplers agree with the exact posterior
(rms z 0.8-1.0, max abs z < 3 over 132 statistics each). frankenz `population_sampler` is wrong in
every case:
* **Absorption at zero (A1-A3).** One bin is driven to ~1e-13 and stays there. Its exact posterior is
  0.10 +/- 0.06, 0.022 +/- 0.019 and 0.074 +/- 0.056 respectively. Its mass moves to the other bins: in
  A1, rho1 is 0.690 against an exact 0.633, i.e. +1.2 posterior SD.
* **Interior variance inflation (A0, no bin near zero).** The means are right, but every posterior SD
  is 11-12% too large (z = +21 to +25).

### 3.2 Task (c): is the adaptive step a valid kernel? (`s02_pop_ablation.py`)

**Derivation.** For a pair direction $t=e_j-e_l$ the proposal is $\rho'=\rho+t\varepsilon$ with
$\varepsilon\sim\mathcal N(0,s(\rho)^2)$, so its density is $q(\rho'\mid\rho)=\phi(\varepsilon/s(\rho))/s(\rho)$.
A valid MH acceptance ratio therefore needs the factor

$$\frac{q(\rho\mid\rho')}{q(\rho'\mid\rho)}=\frac{s(\rho)}{s(\rho')}\exp\Big\{-\frac{\varepsilon^2}{2s(\rho')^2}+\frac{\varepsilon^2}{2s(\rho)^2}\Big\},$$

which samplers.py:304 omits.

In the small-step (diffusion) limit, a Metropolis walk with local step $s(x)$ and no Hastings factor has
stationary density $\propto\pi(x)/s(x)^2$; this follows from the zero-flux Fokker-Planck solution for
drift $\tfrac12s^2\nabla\log\pi$ and diffusion $s^2$. Two consequences follow.
1. **Near a boundary**, $s\le\min(\rho_j,\rho_l)$, so the density gains a factor $\rho_j^{-2}$. That is
   not integrable at 0 whenever $\pi$ is non-zero there, so the chain is pulled into the boundary.
2. **In the interior**, $s=1/|\partial_t\log\pi|$ is large near the conditional mode and small in the
   tails, so mass moves from the mode to the tails and the variance is inflated.

Separately, steps 2-3 of each pair update reuse $s$ from the state before step 1. Choosing a kernel by a
state-dependent rule from a family of individually invariant kernels does not, in general, give an
invariant kernel.

**Ablation.** This uses the bitwise-validated re-implementation on the same data as A0/A1. Variants:
- `orig`;
- `perstep_noH`: scale recomputed every step, still no Hastings factor;
- `fixed_H1`: Hastings factor on step 1 only, scale frozen for steps 2-3;
- `corrected`: per-step scale plus Hastings factor.

**Ablation, case A0** (exact sd = 0.0317, 0.0334, 0.0255; thin = 3 pair updates, 2 chains x 14.0 s)

| variant | mh_steps | accept | sd ratio to exact (rho1, rho2, rho3) | z of mean | z of sd | min rho visited |
|---|---|---|---|---|---|---|
| orig | 3 | 0.51 | 1.113, 1.115, 1.127 | -0.1, +0.6, -0.7 | +19.5, +19.2, +21.9 | 3e-01, 2e-01, 1e-01 |
| perstep_noH | 3 | 0.48 | 1.138, 1.150, 1.165 | +2.3, -1.8, -0.3 | +16.2, +16.9, +18.8 | 3e-01, 2e-01, 1e-01 |
| fixed_H1 | 3 | 0.50 | 1.078, 1.084, 1.095 | -0.0, +0.7, -1.0 | +12.0, +12.5, +14.3 | 3e-01, 2e-01, 1e-01 |
| corrected | 3 | 0.46 | 1.000, 1.008, 0.998 | +1.1, -1.7, +1.0 | -0.0, +1.0, -0.3 | 3e-01, 2e-01, 1e-01 |
| orig | 1 | 0.49 | 1.147, 1.156, 1.162 | -1.5, +1.4, -0.1 | +16.3, +16.2, +19.2 | 3e-01, 2e-01, 9e-02 |
| corrected | 1 | 0.46 | 0.996, 0.995, 0.997 | -0.6, +0.9, -0.6 | -0.5, -0.5, -0.4 | 3e-01, 2e-01, 1e-01 |

**Ablation, case A1** (exact sd = 0.0480, 0.0627, 0.0422; thin = 3 pair updates, 2 chains x 11.0 s)

| variant | mh_steps | accept | sd ratio to exact (rho1, rho2, rho3) | z of mean | z of sd | min rho visited |
|---|---|---|---|---|---|---|
| orig | 3 | 0.80 | 0.800, 0.218, 0.904 | +112.4, -56.4, +103.2 | -22.9, -5.7, -11.5 | 4e-01, 1e-13, 1e-01 |
| perstep_noH | 3 | 0.81 | 0.796, 0.000, 0.906 | +97.5, -2e+14, +75.6 | -20.3, -3e+14, -8.2 | 5e-01, 1e-13, 1e-01 |
| fixed_H1 | 3 | 0.61 | 1.143, 1.185, 1.142 | +18.7, -18.0, +16.2 | +16.7, +18.7, +18.9 | 4e-01, 2e-13, 1e-01 |
| corrected | 3 | 0.54 | 0.991, 0.997, 1.000 | +1.0, -0.7, -0.0 | -1.0, -0.3, -0.0 | 4e-01, 5e-05, 1e-01 |
| orig | 1 | 0.81 | 0.798, 0.000, 0.907 | +98.6, -9e+12, +79.3 | -20.0, -4e+13, -8.0 | 4e-01, 1e-13, 1e-01 |
| corrected | 1 | 0.53 | 1.012, 1.010, 1.012 | -0.6, +0.2, +0.5 | +1.1, +0.9, +1.2 | 4e-01, 6e-06, 8e-02 |

Trajectory of the absorbed bin with notebook defaults (thin = 400, mh_steps = 3), original code:

| chain | rho after 1, 2, 5, 10, 20, 100, 300 saved samples | first saved sample below 1e-3 / 1e-6 / 1e-12 | exact P(rho < 1e-3) |
|---|---|---|---|
| orig_defaults_chain1 (bin 2) | 5.0e-02, 9.2e-05, 2.2e-05, 1.1e-13, 1.1e-13, 1.1e-13, 1.1e-13 | 2 / 6 / 6 | 0.0035 |
| orig_defaults_chain2 (bin 2) | 1.6e-01, 9.4e-02, 5.6e-05, 2.9e-03, 8.8e-14, 9.3e-14, 9.7e-14 | 5 / 9 / 15 | 0.0035 |


**Conclusion for (c).** The adaptive scheme is **not a valid MCMC kernel**, and either defect alone
breaks invariance:
- the missing Hastings factor alone (`perstep_noH`, and `orig` with mh_steps=1) causes the absorption at
  zero, plus a 14-17% SD inflation in the interior;
- the scale frozen over the inner steps alone (`fixed_H1`) gives an 8-19% SD inflation and, in A1,
  18-sigma shifts of the means.

Restoring both (`corrected`) recovers the exact posterior.

**Does it matter in practice?** Yes. With the notebook defaults (400 pair updates per saved sample), a
bin whose posterior touches zero is absorbed within 2-15 saved samples, so burn-in cannot avoid it. This
affects any sparsely populated or poorly constrained bin: the tails of n(z), edge bins, and bins under
broad likelihoods. It fits NB5 cell 75's remark that the population result is "a little off, especially
near zero", which the notebook attributes to the prior.

### 3.3 Geweke "getting it right" tests (tasks c and d; `s03_geweke.py`)

A successive-conditional (SC) chain alternates two steps: draw data given the parameters, then apply the
kernel under test once. If the kernel leaves the posterior invariant, the SC chain's stationary
distribution is the joint prior, so its parameter marginals must match forward simulation. Positive
controls run in the same harness.

**Geweke test, collapsed model** (K=3, N=5, discrete d with C = 0.7 diagonal / 0.15 off; prior Dir(1,1,1); SC chain of 50,000 iterations, first 5% dropped). Reference values are the analytic Dir(1,1,1) moments.

| kernel | E rho1, rho2, rho3 (z) | E rho_k^2 (z) | P(rho_k < 0.05) (z) | max abs z | notes |
|---|---|---|---|---|---|
| frankenz `population_sampler` (thin 6, mh 3) | 0.000 (-1e+14), 0.000 (-8e+14), 1.000 (+4e+14) | 0.000 (-1e+26), 0.000 (-1e+27), 1.000 (+2e+14) | 1.000 (n/a), 1.000 (n/a), 0.000 (n/a) | 1.34e+27 | some rho_k < 1e-10 from iteration 164; vertex (max rho > 1-1e-10) from iteration 2152; 98.2% of iterations with a bin < 1e-10 |
| pairwise + Hastings fix (control) | 0.336 (+1.4), 0.330 (-1.3), 0.333 (-0.0) | 0.169 (+1.1), 0.164 (-1.2), 0.166 (-0.2) | 0.095 (-1.1), 0.101 (+1.5), 0.097 (-0.4) | 1.52 |  |
| frankenz `hierarchical_sampler` (control) | 0.333 (-0.2), 0.332 (-0.8), 0.335 (+1.0) | 0.166 (-0.2), 0.166 (-0.7), 0.169 (+1.3) | 0.095 (-1.6), 0.100 (+1.4), 0.098 (+0.6) | 1.62 |  |

Analytic: E rho = 0.333, E rho^2 = 0.167, P(rho < 0.05) = 0.0975.

**Geweke test, flexible reference** (K=3, N=5, m = [3, 1, 0], N_ref = 4, alpha = beta = 1; forward model k ~ DirMult(N_ref, m+beta), rho|k ~ Dir(alpha+k), z ~ Cat(rho), d ~ C[:, z]; 200,000 forward draws and 200,000 SC iterations). Re-implemented sweep is bitwise identical to frankenz: True. Compatibility cross-difference r(rho1,k1)-r(rho1,k2)-r(rho2,k1)+r(rho2,k2) with r = log p(rho|n,k) - log p(k|rho): code 5.335, exact -8.9e-16 (must be 0 for a joint to exist).

| statistic | forward (truth) | SC with code k-step (z) | SC with exact k-step (z) |
|---|---|---|---|
| E_rho1 | 0.4697 ± 0.0005 | 0.4389 (-27.2) | 0.4709 (+0.8) |
| E_rho2 | 0.3057 ± 0.0005 | 0.3115 (+5.5) | 0.3063 (+0.4) |
| E_rho3 | 0.2246 ± 0.0004 | 0.2496 (+26.6) | 0.2229 (-1.6) |
| E_k1 | 2.2878 ± 0.0026 | 2.0648 (-52.9) | 2.2913 (+0.5) |
| E_k2 | 1.1421 ± 0.0024 | 1.1863 (+11.5) | 1.1430 (+0.1) |
| E_k3 | 0.5701 ± 0.0018 | 0.7489 (+55.7) | 0.5657 (-1.1) |
| P(k=m) | 0.1909 ± 0.0009 | 0.1621 (-22.6) | 0.1911 (+0.1) |
| E_rho1*k1 | 1.2680 ± 0.0023 | 0.9792 (-70.2) | 1.2711 (+0.5) |
| max abs z (11 statistics) | | 70.2 | 1.7 |


**Task (d), derivation.** Suppose a joint $\pi(z,\rho,k)$ has the code's $\rho$-step as its full
conditional, $\rho\mid z,k\sim{\rm Dir}(\alpha+n(z)+k)$. Then
$\pi(z,\rho,k)={\rm Dir}(\rho;\alpha+n+k)\,h(z,k)$ with $h(z,k)=\pi(z,k)$, and its exact $k$-conditional
is

$$\pi(k\mid\rho,z)\;\propto\;w_z(k)\prod_j\rho_j^{k_j},\qquad w_z(k)=\frac{h(z,k)}{\prod_j\Gamma(\alpha_j+n_j+k_j)}\quad\big(\textstyle\sum_jk_j=N_{\rm ref}\big).$$

The code instead draws $k\sim{\rm Mult}(N_{\rm ref},q(\rho))$ with
$q_j=(m_j+\beta_j+N\rho_j)/(N_{\rm ref}+B+N)$.

For the two to agree for all $k$ and $\rho$, the difference $\log(q_j/\rho_j)-\log(q_l/\rho_l)$ must be
independent of $\rho$ for every pair $j,l$. Its derivative in $\rho_j$ is
$-(m_j+\beta_j)/[\rho_j(m_j+\beta_j+N\rho_j)]$, which vanishes only if $m_j+\beta_j=0$. With the default
$\beta=1$ that is impossible. Hence **no joint distribution over $(z,\rho,k)$, and so no coherent model,
has the code's three steps as its full conditionals** (Arnold-Press compatibility). The numerical
cross-difference above (5.33 vs 0) confirms this.

**What the step would need to be.** A multinomial $k$-step is compatible only when
$w_z(k)\propto\prod_jc_j^{k_j}/k_j!$, which gives ${\rm Mult}(N_{\rm ref},c\odot\rho/c\cdot\rho)$:
probabilities **proportional to** $\rho$, not affine in it.

Take the natural reading that the virtual counts are a replicate reference sample,
$k\sim{\rm DirMult}(N_{\rm ref},m+\beta)$. Its exact conditional is

$$p(k\mid\rho)\propto\prod_j\frac{\Gamma(m_j+\beta_j+k_j)}{k_j!\,\Gamma(\alpha_j+k_j)}\rho_j^{k_j},$$

which is not multinomial and needs enumeration or MH. That is the "exact" control above, which passes.

**Limits of the code's step.**
- For $N\gg N_{\rm ref}+B$, $q\to\rho$, and the pair of steps becomes a valid augmentation of
  ${\rm Dir}(\alpha+n)$: **the reference sample is ignored entirely**.
- For $N_{\rm ref}\gg N$, $k$ is a multinomial resample of $m$ independent of $\rho$. That gives a
  mixture of Dirichlets with roughly doubled reference-induced variance.
- In between (Section 3.4, $N/N_{\rm ref}=10$), it discards a large, uncontrolled fraction of a reference
  that is in fact representative.

### 3.4 Task (b): do the two samplers agree on a moderate problem? (`s04_moderate.py`)

**Moderate problem**: K = 15 bins of width 0.1 on [0, 1.5], N = 2000, n(z) = z^2 exp(-(z/0.3)^1.5) on [0,1.5], sigma_i ~ U(0.05, 0.15), reference counts m from N_ref = 200 true redshifts, prior Dir(1 + m), data seed 2024. Reference: IMH-ALR (1e6 proposals, acceptance 0.18, ESS 8879-19694 per bin); cross-checks vs IMH: RWM-ALR max |z| mean/sd = 2.6/1.5, DA Gibbs 2.0/1.4 (30 statistics each).

frankenz runs: `hierarchical_sampler` alpha = 1 + m, thin 1, 2 chains x 120 s (31504 kept); `population_sampler` Dir(1 + m) log-prior, thin = 15 pair updates, mh_steps 3, 2 chains x 120 s (88960 kept). First 20% dropped.

| bin | z range | ref mean ± sd | fz_hier mean (z) | fz_hier sd ratio | fz_pop mean (z) | fz_pop sd ratio | fz_pop min visited |
|---|---|---|---|---|---|---|---|
| 1 | 0.0-0.1 | 0.0163 ± 0.0050 | 0.0164 (+0.4) | 0.997 | 0.0163 (-0.9) | 1.147 | 1e-03 |
| 2 | 0.1-0.2 | 0.0695 ± 0.0114 | 0.0695 (+0.1) | 0.987 | 0.0697 (+1.0) | 1.123 | 2e-02 |
| 3 | 0.2-0.3 | 0.1710 ± 0.0175 | 0.1711 (+0.3) | 1.007 | 0.1711 (+0.4) | 1.101 | 1e-01 |
| 4 | 0.3-0.4 | 0.1795 ± 0.0203 | 0.1789 (-1.3) | 0.987 | 0.1794 (-0.3) | 1.084 | 9e-02 |
| 5 | 0.4-0.5 | 0.1690 ± 0.0196 | 0.1696 (+1.4) | 0.983 | 0.1692 (+0.5) | 1.099 | 9e-02 |
| 6 | 0.5-0.6 | 0.1658 ± 0.0180 | 0.1659 (+0.2) | 0.990 | 0.1664 (+1.8) | 1.108 | 9e-02 |
| 7 | 0.6-0.7 | 0.0778 ± 0.0147 | 0.0772 (-1.9) | 0.993 | 0.0781 (+1.2) | 1.117 | 2e-02 |
| 8 | 0.7-0.8 | 0.0872 ± 0.0124 | 0.0877 (+2.0) | 0.992 | 0.0870 (-1.3) | 1.120 | 3e-02 |
| 9 | 0.8-0.9 | 0.0241 ± 0.0076 | 0.0241 (-0.3) | 1.001 | 0.0241 (+0.0) | 1.142 | 2e-03 |
| 10 | 0.9-1.0 | 0.0277 ± 0.0062 | 0.0277 (-0.7) | 1.003 | 0.0281 (+3.8) | 1.179 | 5e-03 |
| 11 | 1.0-1.1 | 0.0058 ± 0.0032 | 0.0058 (-0.0) | 1.020 | 0.0068 (+20.7) | 1.267 | 5e-06 |
| 12 | 1.1-1.2 | 0.0023 ± 0.0019 | 0.0024 (+1.5) | 1.009 | 0.0000 (-123.2) | 0.000 | 6e-16 |
| 13 | 1.2-1.3 | 0.0021 ± 0.0015 | 0.0021 (-1.9) | 0.991 | 0.0038 (+74.7) | 1.302 | 3e-05 |
| 14 | 1.3-1.4 | 0.0010 ± 0.0009 | 0.0010 (+0.5) | 1.003 | 0.0000 (-150.3) | 0.000 | 7e-18 |
| 15 | 1.4-1.5 | 0.0007 ± 0.0006 | 0.0007 (+1.0) | 0.998 | 0.0000 (-42.2) | 0.232 | 7e-17 |

**Flexible reference in the same problem** (vectorised version of the frankenz kernel, ref_sample = m, beta = 1, alpha = 1; 2 chains x 20,000 sweeps each for: no reference Dir(1), fixed reference Dir(1+m), flexible reference):

| bin | fixed-ref mean ± sd | flex mean | flex mean shift (fixed-ref sd) | sd ratio flex / fixed | sd ratio none / fixed |
|---|---|---|---|---|---|
| 1 | 0.0164 ± 0.0050 | 0.0198 | +0.69 | 1.46 | 1.65 |
| 2 | 0.0693 ± 0.0112 | 0.0496 | -1.75 | 1.56 | 1.71 |
| 3 | 0.1709 ± 0.0175 | 0.2010 | +1.72 | 1.65 | 1.88 |
| 4 | 0.1799 ± 0.0202 | 0.1478 | -1.59 | 1.80 | 2.13 |
| 5 | 0.1688 ± 0.0199 | 0.2091 | +2.03 | 1.89 | 2.30 |
| 6 | 0.1661 ± 0.0180 | 0.1215 | -2.47 | 1.85 | 2.29 |
| 7 | 0.0777 ± 0.0148 | 0.1094 | +2.14 | 1.87 | 2.31 |
| 8 | 0.0873 ± 0.0123 | 0.0791 | -0.67 | 1.60 | 1.93 |
| 9 | 0.0241 ± 0.0077 | 0.0209 | -0.41 | 1.57 | 1.89 |
| 10 | 0.0278 ± 0.0062 | 0.0286 | +0.12 | 1.37 | 1.55 |
| 11 | 0.0057 ± 0.0033 | 0.0057 | +0.02 | 1.26 | 1.33 |
| 12 | 0.0024 ± 0.0019 | 0.0031 | +0.36 | 1.22 | 1.23 |
| 13 | 0.0021 ± 0.0015 | 0.0024 | +0.18 | 1.12 | 1.13 |
| 14 | 0.0010 ± 0.0009 | 0.0011 | +0.15 | 1.11 | 1.10 |
| 15 | 0.0007 ± 0.0006 | 0.0008 | +0.17 | 1.14 | 1.11 |


**Reading.** Section 1.1 shows the two samplers target the same posterior.
- `hierarchical_sampler` reproduces the reference in all 15 bins (abs z <= 2.0; SD ratios 0.98-1.02).
- `population_sampler` does not:
  - it inflates the SDs of the populated bins by 8-18%;
  - it absorbs tail bins 12, 14 and 15 at 1e-16 to 1e-18, where the reference means are 0.0023, 0.0010
    and 0.0007;
  - it moves their mass into bins 11 and 13 (z = +21 and +75).

So the answer to (b) is: **same target, different output, and the discrepancy is entirely the population
sampler's fault.**

The flexible-reference step was given a reference drawn from the same n(z), i.e. a representative one.
It still inflates SDs by 1.1-1.9x relative to the fixed reference and moves the means toward the
oscillating no-reference solution by up to 2.5 fixed-reference SDs. Note the oscillation itself: without
a reference the posterior means alternate (0.23, 0.11, 0.25, 0.08, ... in bins 3-6), the usual
ill-posedness of deconvolution.

### 3.5 Task (e) and the input-construction claims (`s05_inputs.py`, `s09_misc_claims.py`)

**Interim prior** (K = 10 bins of width 0.2 on [0, 2]; truth piecewise-constant with bin masses of z^2 exp(-(z/0.4)^1.5); sigma_i ~ U(0.1, 0.3); interim prior N(0.4, 0.4^2) either per bin (V1, bin masses 0.178, 0.229, 0.229, 0.178, 0.108, 0.051, 0.019, 0.005, 0.001, 0.000) or continuous (V2); prior on rho Dir(1); posterior from vectorised DA Gibbs.)

| input | N = 20000: (posterior mean - truth) / posterior sd, bins 1..10 | max abs | 40 replicates, N = 2000: rms standardised error | 68% coverage | 95% coverage |
|---|---|---|---|---|---|
| true likelihoods L | +1.2, -1.9, +1.8, -2.2, +2.0, -0.9, +1.3, -0.6, +0.6, -0.4 | 2.2 | 0.98 | 0.730 ± 0.023 | 0.940 ± 0.011 |
| posteriors, interim prior piecewise-constant (V1) | -12.6, -0.2, +10.4, -1.3, -2.3, -6.9, -6.8, -10.0, -10.8, -7.5 | 12.6 | 2.96 | 0.292 ± 0.023 | 0.522 ± 0.011 |
| V1 posteriors / pi_k | +1.2, -1.9, +1.8, -2.2, +2.0, -0.9, +1.3, -0.6, +0.6, -0.4 | 2.2 | 0.98 | 0.730 ± 0.023 | 0.940 ± 0.011 |
| posteriors, continuous Gaussian interim prior (V2) | -12.2, +0.3, +8.3, -0.7, -2.7, -6.1, -5.6, -7.0, -8.1, -6.8 | 12.2 | 2.74 | 0.295 ± 0.023 | 0.562 ± 0.011 |
| V2 posteriors / bin-averaged pi | +2.1, -2.2, +1.4, -2.1, +1.7, -0.3, +1.0, +0.3, +0.6, +0.0 | 2.2 | 0.96 | 0.718 ± 0.023 | 0.953 ± 0.011 |

**Input construction claims** (MLE by EM, N = 20000, same truth; standardised by the posterior sd from correct inputs):

| construction | (estimate - truth) / posterior sd, per bin | max abs |
|---|---|---|
| correct bin-averaged likelihoods (MLE) | -1.8, +0.5, +1.5, -2.0, +0.5, +0.8, -0.9, +1.0, -0.6, +0.7 | 2.0 |
| NB5 cell 25 as written: normalise over objects g (MLE) | +49.5, -20.6, -0.3, -7.0, +2.3, +1.5, +4.1, +5.6, +0.9, +18.0 | 49.5 |
| stacked likelihood-PDFs (NB5 cells 7/36) | +20.9, -3.9, -6.6, -3.2, +0.3, +2.6, +3.8, +4.8, +6.0, +8.1 | 20.9 |
| unequal bins [0.0, 0.2, 0.6, 0.8, 1.2, 1.4, 1.8, 2.0]: bin-averaged L (MLE) | -0.1, +0.2, -0.8, +0.3, +1.4, -1.5, +0.2 | 1.5 |
| same, bin-integrated L = pdf.gaussian_bin (MLE) | -13.3, +21.8, -28.3, +19.3, -6.9, +1.7, -1.2 | 28.3 |

Unequal-bin MLE with bin-integrated likelihoods: 0.000, 0.674, 0.000, 0.313, 0.000, 0.013, 0.000 (truth 0.050, 0.499, 0.226, 0.192, 0.024, 0.010, 0.001). `pdf.gaussian_bin` equals the bin integrals to 2.8e-16.



PDF-draw vs posterior covariance (s04 problem, flat prior, bins with mass > 0.01): neighbour correlation -0.38 to -0.29 (PDF draws) vs -0.80 to -0.69 (posterior); next-neighbour -0.18 to -0.14 vs 0.35 to 0.48; posterior SD / PDF-draw SD = 2.0-6.0. One EM step from a flat interim prior equals the stacked posteriors to 3e-17. For sums of two adjacent bins, (k,k+1) vs (k+2,k+3) as in the notebook's plots, k = 1..7: correlation pdf_draws -0.59 to -0.43, posterior_flat -0.54 to -0.19, posterior_Dir(1+m) -0.61 to -0.33; SD of the first pair, posterior / PDF draws: posterior_flat 2.7-3.4, posterior_Dir(1+m) 1.8-2.3.


**Reading.**
* **Interim prior.** Feeding posteriors computed under a non-uniform interim prior biases rho by up to
  12.6 posterior SD at N=20000, toward the prior's shape. Over 40 replicates at N=2000, coverage of the
  68% / 95% intervals falls to 0.29 / 0.52. Dividing by the interim prior removes the bias:
  - exactly when the prior is constant within bins (the chains coincide with the likelihood chains);
  - to within noise when a continuous prior is divided by its bin average (abs z <= 2.2; coverage
    0.72 / 0.95).
* **Cell 25 as written** (normalising over objects) biases the MLE by up to 49.5 SD.
* **Bin-integrated likelihoods with unequal bins** drive every narrow bin to exactly zero (up to 28 SD);
  bin-averaged ones are unbiased.
* **Stacking** is biased by up to 21 SD, as the notebooks say.
* **PDF-draw covariance.** The PDF draws reproduce the posterior's correlation between *aggregated*
  pairs of bins (what NB5 plots). They miss its per-bin structure: next-neighbour correlations have the
  opposite sign, and the SDs are 2-6x too small.

### 3.6 Task (f): wall-clock cost at tutorial scale (`s06_timing.py`)

**Tutorial scale**: N = 20000, K = 80 (zbins = arange(-1, 7, 0.1)), sigma ~ U(0.05, 0.2), single thread, NumPy 2.5.4, SciPy 1.18.1. 37 of 80 bins have exactly zero stacked mass.

| sampler / operation | seconds | per |
|---|---|---|
| frankenz `population_sampler`, notebook prior (scipy Dirichlet), thin=400, mh_steps=3 | 0.419 | saved sample (1.05 ms per pair update) |
| same with xlogy Dirichlet prior | 0.334 | saved sample (0.83 ms per pair update) |
| frankenz `hierarchical_sampler`, thin=5 (notebook call) | 0.346 | saved sample |
| of which: object step (Python loop of multinomial(1, .)) | 0.0681 | sweep |
| of which: Dirichlet draw + scipy logpmf/logpdf | 0.00008 | sweep |
| vectorised object step + Dirichlet (float64) | 0.0111 | sweep (0.055 per saved sample at thin=5) |
| vectorised object step + Dirichlet (float32 likelihoods) | 0.0073 | sweep |
| one evaluation of sum_i log(L_i.rho) and its gradient | 0.00103 | evaluation |

Implied notebook run times: NB5 population run (750 samples) 314 s; NB6 population run (600) 251 s; NB6 hierarchical run (600, thin 5) 208 s, vs 33 s with a vectorised object step.


### 3.7 Task (g): mixing as likelihoods broaden (`s07_mixing.py`, `s07b_tutorial_mixing.py`)

**Mixing vs likelihood width** (K = 15, N = 2000, sigma_i = s x 0.1 x U(0.5, 1.5); single chain per sampler; IAT in saved draws by Geyer IMS; ESS/s = draws/s / IAT; 'med'/'min' over the 15 bins; zbar = sum_k rho_k zmid_k). fz_hier ESS/s uses its measured iteration rate with the IAT of the long vectorised-DA chain (identical kernel; its own ~3000-iteration chains under-estimate IATs in the hundreds). fz_pop thin = 15 pair updates; its ESS refers to its own biased stationary distribution (bins absorbed near 0 show IAT in the thousands).


*Prior Dir(1 + m), N_ref = 200*

| s = sigma/width | fz_hier IAT med (own) | DA IAT med / max | fz_hier ESS/s med / min | DA-vectorised ESS/s med / min | fz_pop IAT med / max | fz_pop ESS/s med / min | RWM-ALR ESS/s med | HMC-ALR IAT med | HMC-ALR ESS/s med / min | ESS/s zbar: fz_hier / DA / fz_pop / HMC |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 2.1 | 2.0 / 2 | 69.3 / 62.13 | 1853 / 1660.4 | 10 / 44 | 40.2 / 9.05 | 102 | 0.7 | 2499 / 1560 | 132 / 3532 / 60 / 2508 |
| 0.5 | 4.8 | 4.8 / 6 | 30.3 / 23.81 | 793 / 622.6 | 16 / 1885 | 28.1 / 0.23 | 91 | 0.9 | 1665 / 755 | 129 / 3362 / 76 / 1766 |
| 1 | 12.1 | 11.0 / 17 | 12.0 / 7.85 | 337 / 220.5 | 21 / 2159 | 20.2 / 0.20 | 92 | 1.0 | 1419 / 840 | 92 / 2573 / 93 / 1546 |
| 2 | 12.7 | 14.6 / 19 | 10.4 / 7.87 | 258 / 196.0 | 16 / 3545 | 26.1 / 0.12 | 109 | 1.0 | 1525 / 1181 | 63 / 1568 / 106 / 1593 |
| 4 | 16.1 | 16.7 / 20 | 8.7 / 7.14 | 231 / 190.5 | 14 / 3939 | 31.8 / 0.11 | 111 | 1.0 | 1689 / 1357 | 29 / 775 / 99 / 1618 |

*Prior Dir(1) (flat)*

| s = sigma/width | fz_hier IAT med (own) | DA IAT med / max | fz_hier ESS/s med / min | DA-vectorised ESS/s med / min | fz_pop IAT med / max | fz_pop ESS/s med / min | RWM-ALR ESS/s med | HMC-ALR IAT med | HMC-ALR ESS/s med / min | ESS/s zbar: fz_hier / DA / fz_pop / HMC |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 7.0 | 7.2 / 9 | 19.9 / 15.13 | 536 / 408.1 | 22 / 205 | 19.4 / 2.04 | 89 | 1.0 | 1516 / 687 | 124 / 3333 / 80 / 1721 |
| 1 | 72.9 | 73.7 / 189 | 2.1 / 0.80 | 50 / 19.4 | 839 / 3972 | 0.5 / 0.11 | 25 | 7.9 | 186 / 110 | 102 / 2482 / 63 / 765 |
| 2 | 117.6 | 161.7 / 745 | 0.9 / 0.20 | 23 / 4.9 | 784 / 4547 | 0.5 / 0.09 | 20 | 14.4 | 63 / 32 | 55 / 1386 / 64 / 344 |
| 4 | 90.6 | 180.3 / 851 | 0.8 / 0.17 | 21 / 4.5 | 438 / 4862 | 0.9 / 0.08 | 46 | 4.1 | 214 / 133 | 22 / 584 / 54 / 363 |



Tutorial scale (same set-up as s06, prior Dir(1 + counts_ref), N_ref = 1000): DA kernel median / max IAT over the 12 bins with mass > 1e-3 = 24.8 / 33.1 sweeps (IAT of zbar 1.4); so the notebooks' 600 x 5 = 3000-sweep hierarchical runs carry ESS of about 121 per bin. HMC-ALR: acceptance 0.88, step 0.35, median / max IAT 0.95 / 1.53, posterior means within 0.13 posterior SD of DA. Median-bin ESS/s: original hierarchical_sampler 0.58, vectorised DA 4.5, HMC-ALR 126.


**Reading.**
* **The hierarchical (DA) kernel** mixes well when likelihoods are narrower than the bins (IAT 2-7
  sweeps). With a flat prior it degrades sharply once sigma is about the bin width or more: median IAT
  74-180 and worst bin 189-851 sweeps.
  - This is the classic slow-down of data augmentation when the fraction of missing information is
    high.
  - An informative reference prior caps it (median IAT <= 17 with N_ref/N = 0.1).
  - The slow directions are bin-to-bin contrasts; the mean redshift mixes in 1-7 sweeps.
  - frankenz's own implementation is 25-28x slower per sweep than the vectorised one at K=15, N=2000,
    and 6x slower at K=80, N=20000.
* **`population_sampler`** has median ESS/s comparable to the original hierarchical code. But its
  absorbed bins never mix (IAT in the thousands), and what it samples is the wrong distribution.
* **HMC in ALR coordinates** with the gradient $\partial/\partial\eta_j=a_j+r_j-(A+N)\rho_j$ (where
  $r_j=\rho_j\sum_iL_{ij}/o_i$), whitened by the Laplace covariance and with a fixed step size, has median
  IAT 1-14 in every setting.
  - It gives 1.4-10x the per-bin ESS/s of vectorised DA and 36-270x that of frankenz's
    `hierarchical_sampler` (about 200x at tutorial scale).
  - For the mean redshift alone, vectorised DA is comparable or better.

### 3.8 API, state handling and numerics (`s08_api_checks.py`)

| check | result |
|---|---|
| `population_sampler.run_mcmc` called twice (mh_steps=0, so a saved sample equals the start) | second call started from the stacked PDFs: True; from the last sample: False |
| `hierarchical_sampler.run_mcmc` called twice (spy on `sample`) | pos_init passed on the second call: None |
| continuation after 200 saved samples (population, one MH step) | first new sample is 0.016 (max abs) from the stacked PDFs and 0.063 from the last sample |
| `hierarchical_sampler`: run 5, `reset()`, run 3 | len(samples) = 8, len(samples_lnp) = 3; results shapes [[8, 4], [3]] |
| `population_sampler` with `np.random.default_rng` | AttributeError: 'numpy.random._generator.Generator' object has no attribute 'randn' |
| `hierarchical_sampler` with `np.random.default_rng` | runs |
| `rstate=None` | advances the global NumPy state: True |
| `pos_init` given as a list | TypeError: '<' not supported between instances of 'list' and 'float' |
| `pos_init` summing to 2, default (flat) prior | all samples sum to 2.000000-2.000000 (silently wrong simplex) |
| prior = `scipy.stats.dirichlet.logpdf` without the notebook's x<0 guard | ValueError: Each entry in 'x' must be greater than or equal to zero. |
| integer `ref_sample` and integer `beta` | UFuncTypeError: Cannot cast ufunc 'divide' output from dtype('float64') to dtype('int64') with casting rule 'same_kind' |
| bin with no likelihood support (column of zeros), K=5, N=300, flat prior | population_sampler max rho5 over 300 samples = 0.0; valid DA posterior mean = 0.0032 (approx. alpha/(N+A) = 0.0033) |
| finite-difference directional derivative (l.272-282) vs analytic, K=15, N=2000 | rho=1e-01: rel. err 7.9e-11; rho=1e-03: rel. err 1.1e-09; rho=1e-06: rel. err 1.9e-06; rho=1e-09: rel. err 1.5e-03; rho=1e-12: rel. err 1.0e+00; rho=1e-14: rel. err 1.0e+00 (FD = 0 for rho <= 1e-12, so the scale drops to 1e-4 rho) |
| `pdf.gaussian_bin` upper tail (mu=0, sigma=1, bins of width 0.1) | [4,4.1]: rel. err 5.1e-12; [6,6.1]: rel. err 1.2e-07; [8,8.1]: rel. err 2.8e-01; [9,9.1]: rel. err 1.0e+00; [10,10.1]: rel. err 1.0e+00; [20,20.1]: rel. err 1.0e+00 |
| `pdf.gaussian_bin` for an object far outside the bin range, then normalised | all NaN |
| `pdf.gauss_kde`, wt_thresh=None, weights (1, 1, 100) | mass near the three objects: 1.000, 1.000, 0.000 (highest-weight object dropped) |
| `pdf.gauss_kde` with sigma = 0.001 < dx/5 = 0.002 | total mass 0.0 |
| object step with L_i . rho = 0 | ValueError: pvals < 0, pvals > 1 or pvals contains NaNs |
| Dirichlet(0.01, 0.01, 50) draws with an exact 0 in the first component | 0.062% (frankenz accepts any alpha) |


---

## 4. What is right and worth keeping

* **The core likelihood.** The histogram (top-hat) basis with $p(d_i\mid\rho)=L_i\cdot\rho$ is correct
  (`loglike_nz`, l.24-80), as are the population posterior $\prod_iL_i\cdot\rho$ times a Dirichlet prior
  (NB5 cells 6, 12, 64, read with $L$ as likelihoods) and the observation that the per-object
  normalisation of $L$ is irrelevant.
* **`hierarchical_sampler`'s data augmentation.** The object step $z_i\sim{\rm Cat}(L_i\odot\rho/L_i\cdot\rho)$
  and the conjugate step $\rho\sim{\rm Dir}(\alpha+m+n)$ are exact. They passed every test here:
  - exact K=3 (rms z 0.94);
  - moderate K=15 (abs z <= 2.0);
  - Geweke (max abs z 1.6).

  This is a simple, dependency-free sampler that also yields per-object latent draws (shrinkage, NB6
  cell 60). It is the right default for narrow likelihoods or with an informative prior, and for smooth
  functionals such as the mean redshift.
* **Reference samples as Dirichlet pseudo-counts**, ${\rm Dir}(\alpha+m)$, are exact for a representative
  reference (NB5 cell 66, NB6 cell 49). The *idea* behind the flexible reference (a reference that might
  be unrepresentative should carry less weight) is scientifically important and worth keeping with a
  coherent model (Section 5.1).
* **The pairwise building block.** Sum-preserving moves along $e_j-e_l$ with the exact $O(N)$ overlap
  update are a valid basis for slice sampling or MH along pair directions, given a valid acceptance rule.
* **The tutorials' qualitative messages** are right and worth reproducing in pastry's docs (Section 3.5
  quantifies them):
  - stacking is not the population distribution; it is a deconvolution problem, and the stack is the
    first EM iterate from the interim prior;
  - Poisson, multinomial and PDF-draw resampling of the stack understate the uncertainty;
  - hierarchical inference requires likelihoods (NB6 cell 41).
* **The mock design.** Heteroscedastic Gaussian likelihoods, true redshifts allowed below zero to avoid
  edge effects, and a PIT check of likelihood calibration make a good template for pastry's test fixtures.
* **The stacked-PDF initialisation** is reasonable for DA. For any sampler that cannot leave exact zeros
  it should be moved into the interior.

---

## 5. Recommendations for pastry

### 5.1 Model and inputs

* **Core model.** The histogram basis with *bin-averaged* likelihoods $L_{ik}$ and the collapsed
  posterior ${\rm Dir}(\rho;\alpha+m)\prod_iL_i\cdot\rho$, with the notation of Section 1.1 throughout the
  docs and API. Build $L$ internally from per-object inputs:
  - likelihood functions or samples;
  - bin integrals divided by widths;
  - posteriors **plus** the interim prior, divided out internally (bin-averaged prior for continuous
    interim priors; Section 3.5 shows the residual is negligible at N=20000).
* **Validate inputs.** Check that values are finite and non-negative, that rows have positive sums, and
  that columns are not all zero (warn: such bins are prior-only). Accept per-object normalisation freely.
* **Reference samples.**
  - A representative reference is ${\rm Dir}(\alpha+m)$ (exact, conjugate).
  - For uncertain representativeness, offer one coherent model:
    1. a power prior ${\rm Dir}(\alpha+\lambda m)$, with $\lambda\in[0,1]$ fixed or given a Beta prior; or
    2. a two-population model $\rho\mid\rho_{\rm ref},c\sim{\rm Dir}(c\rho_{\rm ref}+\alpha)$,
       $\rho_{\rm ref}\mid m\sim{\rm Dir}(\beta+m)$, with $c$ fixed or inferred.

    Either way, document the limits ($c\to\infty$: representative; $c\to0$: ignored). Do not port
    frankenz's flexible step.
* **Regularisation.** With broad likelihoods and no reference, the posterior means oscillate between
  neighbouring bins (Section 3.4). A smoothness prior on $\log\rho$ (e.g. a Gaussian random walk or GP
  over bins, giving a logistic-Gaussian process) is the natural extension. It suits gradient-based
  samplers, not the conjugate Gibbs step.

### 5.2 Samplers

* **Default sampler: HMC/NUTS** on unconstrained coordinates (ALR or ILR; include the log-Jacobian
  $\sum_k\log\rho_k$), with an analytic gradient costing one $O(NK)$ pass (1.0 ms at N=20000, K=80) and
  mass-matrix adaptation during warm-up only.
  - In these tests even a crude fixed-step version with Laplace whitening had IAT 1-14 everywhere.
  - That is 1.4-10x the ESS/s of vectorised DA and about 200x the original code at tutorial scale.
* **Exact DA Gibbs, vectorised**, as the dependency-free alternative. It also provides per-object latent
  draws and Rao-Blackwellised per-object posteriors (running means of $L_i\odot\rho/L_i\cdot\rho$).
  - It is competitive for narrow likelihoods, with an informative prior, and for smooth functionals.
  - Warn when the bin-level ESS is low.
  - Sample the Dirichlet in log space (log-Gamma) so that small $\alpha$ never produces exact zeros: at
    $\alpha=0.01$, 0.06% of NumPy Dirichlet draws contain an exact 0. If an object's likelihood lives
    only on such bins, $L_i\cdot\rho=0$ and frankenz's object step raises (Section 3.8).
* **EM and MAP** via the fixed point $\rho\leftarrow(a-1+r)/(A-K+N)$ for fast point estimates,
  initialisation and the Laplace approximation. Note that stacking equals one EM step.
* **Do not port** the adaptive pairwise MH. If a derivative-free sampler is needed, use a slice sampler
  along random pair directions (valid and self-tuning) or MH with a fixed (warm-up-tuned) or properly
  Hastings-corrected state-dependent scale.
* **Initialise in the interior** of the simplex. Run several chains by default.

### 5.3 API and defaults

* **One state object** (rho, latent counts, RNG state, adaptation) so that `run(a); run(b)` equals
  `run(a + b)`. `reset()` clears everything, and results are a single structure (draws, log posterior,
  diagnostics).
* **Randomness.** `rng` accepts a seed or `np.random.Generator`; chains use `SeedSequence.spawn`; no
  global state.
* **Diagnostics on by default.** Split R-hat, bulk/tail ESS, MCSE on every summary, and acceptance
  statistics. The reported log density should be the collapsed log posterior.
* **Defaults.** alpha = 1 with a documented warning about deconvolution ill-posedness. float64 by
  default, with an optional float32 likelihood store (about 1.5x faster for the object step).

### 5.4 Tests to carry into the pastry test suite

1. **Exact K=3 references** (the cases A0-A3 here: interior, near-zero bin, empty bin with small N).
   Every sampler's means, SDs and CDF at nine exact quantiles must give |z| < 4. Aggregate check: rms z
   within [0.7, 1.3] over the 33 statistics per case.
2. **Geweke tests** on tiny problems (K=3, N=5, discrete observations with a confusion matrix) for every
   kernel and model extension (reference-sample models included). Include a deliberately broken kernel
   as a negative control (e.g. the Hastings factor removed) to confirm the test has power.
3. **Cross-sampler agreement** on a moderate problem (K=15, N=2000) between at least two independent
   valid samplers (e.g. NUTS vs DA), with per-bin z-scores of means and SDs.
4. **Interim-prior input path.** Posteriors plus a piecewise-constant interim prior must reproduce the
   likelihood-input result exactly (same seed); a continuous prior must agree within MCSE. A documented
   negative example shows the bias of naive posterior inputs.
5. **Unequal bins.** The bin-averaged construction is unbiased on a well-specified mock; a regression
   test catches bin-integrated inputs.
6. **Normalisation invariance.** Rescaling the rows of L leaves the log posterior unchanged up to a
   constant, and leaves the draws unchanged with a fixed seed.
7. **Zero-support bins.** The posterior mass of a bin with $L_{ik}=0$ for all $i$ matches the analytic
   prior-driven value (about $\alpha_k/(N+A)$) and is not frozen at 0.
8. **Simulation-based calibration (Talts et al. 2018).** Rank statistics over prior draws on small
   problems, complementary to Geweke.
9. **API tests.**
   - Resumability is bitwise.
   - `reset` clears all state.
   - Seeds and Generators are accepted.
   - Invalid inputs are rejected: lists, off-simplex initial points, negative/NaN/all-zero rows,
     integer dtypes.
   - The prior is never evaluated outside the support.
10. **Performance regression.** At N=20000, K=80, single thread: vectorised object step <= ~15 ms per
    sweep and log-likelihood plus gradient <= ~2 ms.
11. **Documentation tests.** Stacking, its EM-step identity, and the PDF-draw covariance comparison, as
    executable examples.

---

## Appendix: scripts and outputs

Root: `/tmp/claude-0/-home-user-pastry/63ffb1b4-5826-5f0b-ade5-c8209761737e/scratchpad/frankenz_audit/`.
Run with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 venv/bin/python scripts/<script> [args]`.
The verbatim frankenz copies are in `fz_orig/`.

| script | purpose | output(s) |
|---|---|---|
| `scripts/common.py` | mock generation, exact K=3 quadrature, Geyer ESS/MCSE, reference samplers (DA, IMH-ALR, RWM-ALR, HMC-ALR), bitwise re-implementation of `population_sampler` with ablation switches | - |
| `scripts/s00_validate_tools.py` | validates quadrature (Beta marginals, brute-force grid) and the bitwise re-implementation | `outputs/s00_validate_tools.json` |
| `scripts/s01_k3_exact.py CASE [budget]` | task (a): exact K=3 vs all samplers, cases A0-A3 | `outputs/s01_k3_A{0,1,2,3}.json` |
| `scripts/s02_pop_ablation.py CASE [budget]` | task (c): ablation of the adaptive step; notebook-default trajectory | `outputs/s02_ablation_A{0,1}.json` |
| `scripts/s03_geweke.py pop [M] [kernels]` / `flex [M]` | tasks (c, d): Geweke tests; compatibility check; bitwise check of the flexible step | `outputs/s03_geweke_pop.json`, `outputs/s03_geweke_flex.json` |
| `scripts/s04_moderate.py ref / fz_hier / fz_pop / flex [budget]` | task (b): K=15, N=2000 reference and both frankenz samplers; flexible-reference effect | `outputs/s04_reference.json`, `s04_fz_hier.json`, `s04_fz_pop.json`, `s04_flex.json` |
| `scripts/s05_inputs.py interim / construction` | task (e) and the NB5 cell 14/25 and stacking claims | `outputs/s05_interim.json`, `outputs/s05_construction.json` |
| `scripts/s06_timing.py` | task (f): wall-clock at N=20000, K=80 | `outputs/s06_timing.json` |
| `scripts/s07a_validate_hmc.py` | validates HMC-ALR against exact K=3 | `outputs/s07a_hmc_validate_A{1,3}.json` |
| `scripts/s07_mixing.py SCALE [ref/flat]` | task (g): IAT and ESS/s vs likelihood width | `outputs/s07_mixing_s{0.25,0.5,1,2,4}[_flat].json` |
| `scripts/s07b_tutorial_mixing.py` | DA and HMC mixing at tutorial scale | `outputs/s07b_tutorial_mixing.json` |
| `scripts/s08_api_checks.py` | API, state, RNG, validation and numerics checks | `outputs/s08_api_checks.json` |
| `scripts/s09_misc_claims.py` | PDF-draw covariance (NB5 cell 62) and the EM-stacking identity | `outputs/s09_misc_claims.json` |
| `scripts/make_tables.py sNN ...`, `scripts/build_notes.py`, `scripts/notes_template.md` | render the tables from the JSON outputs and assemble this note | `outputs/table_*.md` |
