# binference: specification (v0.1)

*Population and hierarchical inference with PDFs*

**Status:** draft for review, 2026-10-22.

**Inputs to this document:**
- The landscape census in `reports/Population inference software landscape.md`, with the notes behind it in `research_notes/`.
- An audit of the population samplers in `frankenz`.
- A set of prototype benchmarks run while writing this spec (Section 6.10).

**Companion documents:**
- [`test-plan.md`](test-plan.md): how each requirement here is verified.
- [`implementation-plan.md`](implementation-plan.md): module layout, milestones and the order of work.

Requirements carry IDs (`R-…`) so that tests and code can point back to them. "Must" marks something required in v0.1. "Should" marks something we intend to do but can defer. "Roadmap" marks work planned after v0.1.

---

## 1. Vision and guiding principles

`binference` turns **a catalogue of noisy, heteroscedastic, arbitrarily shaped per-object PDFs over a set of bins or categories** into **a posterior over the population distribution** (bin fractions or rates). It also returns **consistent posteriors for each individual object** under that population. Typical inputs are photo-z PDFs, positions on a colour–magnitude grid, classifier outputs over classes, or a detector response.

**Vision** (DECISIONS #33). `binference` should be as easy to pick up as scikit-learn: a novice gets a correct, calibrated answer without understanding the internals, while an expert can tune every knob. It should be fast, and it should make doing the right thing so easy that not doing it is the harder path.

The baseline model has been implemented at least four times in astronomy alone (Leistedt+2016, Leistedt & Hogg 2017, CHIPPR, `frankenz`). It reappears as iterative Bayesian unfolding in HEP, NPMLE/g-modelling in statistics, EMQ quantification in ML, and genetic stock identification in ecology. Every implementation the census found is one of the following:

- stale paper code;
- specialised to one field's input format;
- missing uncertainty quantification;
- quantifying uncertainty with approximations known to under-cover, such as mean-field VB.

None checks its own calibration. `binference`'s value is therefore **correct input conventions, calibrated uncertainty, built-in validation and speed**, plus what existing implementations leave out: inputs whose own uncertainty is propagated (selection functions from injections, responses from calibration sets), bases that remove the flat-within-bin approximation, and defaults chosen automatically from measured evidence.

Guiding principles (DECISIONS #34; the user-facing version is `docs/vision.md`):

1. **Correct by construction.** The API makes the user say what they are passing: likelihoods or posteriors, which interim prior, masses or densities, which measure. Silent convention mismatches are the most common real-world failure (stacking, interim priors, unequal bins), so the API refuses ambiguity.
2. **Easy things are one line, hard things are possible.** There is a one-call API (`binference.infer`). Beneath it sit configurable model and engine objects, and beneath those, raw log-densities for external samplers. This mirrors emcee, dynesty, nautilus and sbi.
3. **Exact inference by default; approximations are labelled**, including approximations of the model (flat within bins, Monte Carlo inputs) as well as of the sampler. Fast approximations (Laplace, mean-field VB, EM, bootstrap) carry a flag and a warning describing their known failure modes.
4. **Self-validating.** Diagnostics, optimality certificates and input checks run automatically. Simulation-based calibration (SBC) and coverage tools ship in the package, and the test suite checks engines against exact references.
5. **Lightweight core.** The core needs only NumPy, SciPy and threadpoolctl. Plotting, xarray export and future JAX or flow backends are optional extras.
6. **One way to read a result.** Every engine that draws returns the same result type with the same methods: summaries, intervals, functionals, individual-object outputs, diagnostics, plots and serialisation; `result.best` gives the best fit with the same outputs (DECISIONS #85).
7. **Everything exposed, with sensible defaults** (DECISIONS #16).
8. **Adaptive, evidence-based defaults, reported to the user** (DECISIONS #14, #88).
9. **The user's bins define the problem; internals adapt** (DECISIONS #17, #20, #21).
10. **Every uncertain input is modelled jointly by default; fixing it is an explicit choice** (DECISIONS #10, #28, #30).
11. **Fast.** Performance is a feature, measured by the benchmarks.

---

## 2. Problem statement and scope

### 2.1 The core problem

We observe data $d_i$ for $N$ objects, $i = 1..N$. Each object has a latent "true" value $z_i$ in some space $\mathcal Z$, such as redshift, a colour–magnitude position or a class label. $\mathcal Z$ is partitioned into $K$ **bins** (cells of a 1-D or N-D grid) or is a set of $K$ **categories**. For each object we can evaluate, approximately, how probable its data are given that it lies in bin $k$:

$$L_{ik} \;\propto\; p(d_i \mid z_i \in \text{bin } k).$$

The goal is the **population distribution** $\rho \in \Delta^{K-1}$ (the fraction of the population in each bin), or rates $\lambda \in \mathbb R_+^K$, with honest uncertainties. A secondary goal is each object's posterior over bins under the inferred population, the "hierarchical shrinkage" posterior.

### 2.2 Motivating use cases (all must be expressible)

| Use case | Objects | Bins | Input form |
|---|---|---|---|
| Photo-z n(z) | galaxies | 1-D redshift grid | posterior PDFs under a template-fitting or ML interim prior; or likelihoods; or posterior samples |
| CMD / HR diagram | stars | 2-D colour–magnitude grid | per-star likelihood over cells, possibly sparse |
| Quantification | items | unordered classes | calibrated classifier posteriors plus training class prior |
| HEP unfolding | events | true-spectrum bins | reconstructed counts plus response matrix plus efficiency |
| Genetic stock ID | fish | source populations | per-individual genotype likelihoods |
| Occurrence rates | planets or candidates | 1-D or 2-D period–radius grid | posterior samples plus completeness |

### 2.3 Scope

**In scope (implemented):**
- **Input conversion:** discrete and binned populations on 1-D grids, N-D grids and unordered categories, with converters from likelihoods, posteriors with interim priors, samples, functions and distributions, binned measurements with a response matrix, and classifier outputs.
- **Bases:** B-spline bases over the bins (`Basis`), which remove the flat-within-bin approximation, and a binning-bias check for that approximation (DECISIONS #47, #86).
- **Priors:** the conjugate simplex family (Dirichlet, generalised Dirichlet, Dirichlet-tree and Pólya tree), the smooth logistic-normal family (fixed covariance, random-walk/GMRF, Gaussian process) with hyperparameter inference, and Gamma rates.
- **Selection effects** for fractions and for rates, including selection functions learned from injection campaigns and inferred jointly with the population (DECISIONS #36, #80).
- **Responses** tallied from labelled calibration sets, inferred jointly with the population (DECISIONS #28, #81).
- **Rates:** derived for every fractions fit, and modelled directly with `GammaRates` (DECISIONS #9, #78).
- **Inference engines:** `method="auto"` (default, DECISIONS #88), NUTS, Gibbs, MAP/EM, Laplace with optional importance correction, mean-field VB and bootstrap.
- **Outputs:** population posteriors, best fits, realised-sample histograms, individual-object posteriors and draws (including values within bins), leave-one-out posteriors, diagnostics, SBC/coverage tools, simulation utilities, plotting and serialisation.
- **Documentation:** a theory page in clean notation, executed tutorials per field, background pages on each field's methods, and an API reference.

**Later (see `notes/review/post-v1.md` and the deferred issues):**
- A hierarchical Dirichlet model with uncertain concentration and an uncertain reference sample (the coherent replacement for frankenz's "flexible reference").
- Rates with logistic-normal priors.
- Evidence estimates.
- Continuous population densities: smooth parametric maps and normalising flows / flow matching, behind an optional JAX extra.
- Sample-variance (overdispersion) models.
- A JAX/GPU backend.
- Minibatch methods for $N \gtrsim 10^8$.
- Covariate-dependent populations.
- Exporters to qp, SACC, CosmoSIS and RAIL.

**Out of scope:**
- Computing per-object likelihoods from raw data (photometry fitting, classifier training, detector simulation). `binference` starts where each object already has a PDF, likelihood, samples or a measured bin.

---

## 3. Notation and statistical model

This section is normative. Docstrings and the theory page use exactly this notation. It also corrects the notation problems found in `frankenz` (Appendix B).

### 3.1 Objects, bins and likelihoods

- $i = 1..N$ indexes objects with data $d_i$, and $k = 1..K$ indexes bins.
- Bin $k$ is a region $B_k \subset \mathcal Z$ with **volume** $v_k = |B_k|$ (its width in 1-D). Categories have $v_k \equiv 1$.
- $\ell_i(z) = p(d_i \mid z)$ is object $i$'s likelihood function, known up to a per-object constant $c_i > 0$ that never affects inference on the population.
- $L_{ik}$ is the **bin-averaged likelihood**:
  $$L_{ik} = \frac{1}{v_k}\int_{B_k} \ell_i(z)\,dz .$$
- $w_i > 0$ are optional object weights. Integer weights are multiplicities, from compressing duplicate rows. Non-integer weights define a weighted (pseudo-)likelihood, which is documented as not a proper posterior.

### 3.2 Population model (fractions)

The population density is piecewise constant (top-hat basis):

$$n(z \mid \rho) = \sum_k \rho_k\,\frac{\mathbb 1[z\in B_k]}{v_k},\qquad \rho\in\Delta^{K-1}.$$

The **marginal likelihood of one object** is

$$p(d_i\mid\rho) = \int \ell_i(z)\,n(z\mid\rho)\,dz = \sum_k \rho_k L_{ik} \equiv L_i\cdot\rho .$$

The **population posterior** is

$$p(\rho\mid D) \;\propto\; p(\rho)\prod_i (L_i\cdot\rho)^{w_i}. \tag{P}$$

The log-likelihood $\ell(\rho)=\sum_i w_i\log(L_i\cdot\rho)$ is concave in $\rho$. It is unchanged by per-row rescaling of $L$, and its gradient is cheap: with $s = L\rho$ and $\hat n_k = \rho_k\,[L^\top(w/s)]_k$, two matrix–vector products give everything needed.

### 3.3 Augmented (hierarchical) form and its equivalence

Introduce latent labels $z_i \mid \rho \sim \mathrm{Cat}(\rho)$ with $p(d_i\mid z_i=k)\propto L_{ik}$. Then

$$p(\rho, z\mid D)\propto p(\rho)\prod_i \rho_{z_i}L_{i z_i}.$$

Summing over $z$ recovers (P) **exactly**. "Population" (marginal) samplers and "hierarchical" (data-augmentation) samplers therefore target **the same posterior for $\rho$**. They differ only in efficiency and in whether they return joint draws of $z$. Joint draws of $z$ can always be produced afterwards from any posterior draw of $\rho$, because the full conditional is exact:

$$P(z_i = k\mid \rho, D) = r_{ik}(\rho) = \frac{L_{ik}\rho_k}{L_i\cdot\rho}.$$

(R-THEORY-1) The documentation must state this equivalence. frankenz's tutorials presented the two as different models.

### 3.4 Likelihoods versus posteriors: interim priors

Many codes report **posteriors** $p_i(z)\propto \ell_i(z)\,\pi(z)$ computed under an **interim prior** $\pi$. Writing $P_{ik}=\int_{B_k}p_i$ and $\Pi_k=\int_{B_k}\pi$ for the bin masses, and assuming $\pi$ is roughly constant within bins,

$$L_{ik}\;\propto\;\frac{P_{ik}}{\Pi_k}.$$

Feeding posteriors into (P) as if they were likelihoods double-counts the interim prior. The resulting bias in $\rho$ is in the direction of $\pi$.

**R-INPUT-1 (must):** every constructor that accepts posteriors requires an explicit `interim_prior`. Valid values are bin masses, a density callable, or the explicit string `"flat"`, meaning uniform density in the coordinate.

### 3.5 Bin volumes: masses versus densities

The bin-averaged likelihood is a *function value*, not a probability mass. Inputs come in two **representations**:

- `"density"`: a value per unit volume, e.g. a likelihood or posterior PDF evaluated at bin centres.
- `"mass"`: a value integrated over the bin, e.g. CDF differences, or histogram counts of posterior samples.

Conversions:
- **Likelihoods.** $L_{ik} = \text{value}_{ik}$ for `"density"`, and $L_{ik} = \text{value}_{ik}/v_k$ for `"mass"`.
- **Posteriors with an interim prior.** First convert both the posterior and the interim prior to masses; then $L_{ik}\propto P_{ik}/\Pi_k$. Equivalently, take the ratio of densities.

With equal-volume bins the representation cannot change the inference, because it scales each row by a constant. With unequal bins it can, and mistakes here bias $\rho$ toward wide bins. (R-INPUT-2) The representation is a required, documented argument wherever it matters. Sanity warnings fire when values look inconsistent with the declared representation, e.g. declared posterior masses whose rows do not sum to 1.

### 3.6 Selection effects

Let $S_k\in(0,1]$ be the probability that an object whose true value lies in bin $k$ is detected (bin-averaged). How $S$ enters depends on what the selection acts on, so the API requires an explicit **selection mode**:

- **`"observed"`**: selection acts on the *measured* data, e.g. a flux limit applied to measured fluxes, or a reconstruction efficiency. For a detected object the selection indicator is 1, so its likelihood row $L_i$ is the ordinary likelihood $p(d_i\mid z)$ at the observed data; it must **not** be renormalised over the detectable data space. Here $S_k=\int_{\rm det}p(d\mid z\in B_k)\,dd$.
- **`"true"`**: selection acts on the *true* value, or independently of the measurement noise, e.g. a completeness map in the true parameter. Then $p(d_i,\mathrm{det}\mid z)=S(z)\,\ell_i(z)$, and the effective likelihood is $\tilde L_{ik}=S_kL_{ik}$, assuming $S$ is roughly constant within bins.

Write $\tilde L_{ik}=L_{ik}$ in the first case and $S_kL_{ik}$ in the second.

- **Fractions.** Marginalising the unknown total number of objects $\Lambda$ under $p(\Lambda)\propto 1/\Lambda$ gives, for both modes,
  $$p(\rho\mid D)\propto p(\rho)\,\frac{\prod_i (\tilde L_i\cdot\rho)^{w_i}}{(S\cdot\rho)^{W}},\qquad W=\textstyle\sum_i w_i. \tag{S}$$
  - Gibbs sampling stays conjugate if we also impute the number of missed objects: $M\mid\rho\sim\mathrm{NegBin}(W, S\cdot\rho)$, with their bins distributed $\mathrm{Mult}(M, \rho\odot(1-S)/(1-S\cdot\rho))$. Detected objects' labels are drawn from responsibilities computed with $\tilde L$.
  - EM uses the expected missed counts $W\rho_k(1-S_k)/(S\cdot\rho)$.
  - Both are verified against (S) using NUTS, exact quadrature and Geweke tests.
- **Rates.** For an inhomogeneous Poisson process with expected counts $\lambda_k$ (exposure absorbed into $S$),
  $$p(\lambda\mid D)\propto p(\lambda)\,e^{-S\cdot\lambda}\prod_i (\tilde L_i\cdot\lambda)^{w_i}.$$
  Gamma priors on $\lambda_k$ stay conjugate given the labels: $\lambda_k\mid z\sim\mathrm{Gamma}(a_k+n_k,\,b_k+S_k)$.
- **Fractions and rates are not equivalent priors.** Independent $\mathrm{Gamma}(\alpha_k,b)$ rates imply $\rho\sim\mathrm{Dir}(\alpha)$ only when there is no selection (or as $b\to 0$ with an extra $(S\cdot\rho)^{-\sum\alpha}$ factor). The documentation must say so.
- **Choosing the wrong mode biases $\rho$.** On a three-bin test with true $\rho=(0.40,0.35,0.25)$ and selection in the true bin, the correct mode gives $(0.37,0.37,0.26)$; the wrong mode gives $(0.23,0.31,0.46)$; ignoring selection gives $(0.58,0.32,0.09)$. Arrays passed as `selection` therefore require `selection_mode`; mocks and adapters attach the right mode themselves.
- **R-THEORY-2:** the documentation must cover the two ways of getting this wrong (Mandel, Farr & Gair 2019): renormalising $L$ by $S$ and then also dividing by $S\cdot\rho$, or omitting the normaliser.

### 3.7 Population quantities versus realised-sample quantities

- $\rho$ describes the **population** the objects were drawn from. Its posterior includes multinomial shot noise from having only $N$ objects.
- The **realised histogram** of the observed objects, $n_k=\sum_i w_i\mathbb 1[z_i=k]$, describes *these* objects. It is computed from joint draws of $z$.
- With perfect measurements, $n$ is known exactly while $\rho$ still carries multinomial uncertainty. With broad PDFs, both are uncertain.

For example, a weak-lensing kernel depends on the realised redshifts of the lens sample, whereas a luminosity-function or occurrence-rate study wants $\rho$. **R-OUT-1:** the result must expose both, clearly named (`samples` and `realized_counts()`). frankenz conflated them as $n=\rho N$.

### 3.8 Individual-object posteriors

- **Marginal.** $P(z_i=k\mid D) = \mathbb E_{\rho\mid D}[r_{ik}(\rho)]$. This is object $i$'s posterior under the inferred population. It uses object $i$'s own data twice only in the sense that the full joint posterior does, so it is correct under the hierarchical model.
- **Joint.** Joint draws $(\rho^{(s)}, z^{(s)})$ come from the Gibbs sampler, or from conditional draws for any other engine.
- **New objects.** The posterior predictive for a *new* object with likelihood row $L_*$ is $\mathbb E[r_*(\rho)]$, and its log predictive density is $\log \mathbb E[L_*\cdot\rho]$. The latter enables cross-validation.

### 3.9 Point estimates and the EM map

- **EM.** The EM update for the MLE is
  $$\rho^{t+1}_k = \frac{\hat n_k(\rho^t)}{W}=\frac{1}{W}\sum_i w_i\, r_{ik}(\rho^t),$$
  i.e. *stacking the posteriors computed under the current population estimate*.
  - "Naïve stacking" of interim posteriors is therefore one EM step from the interim prior.
  - The MLE is a fixed point of iterated stacking. Self-consistency is necessary but not sufficient.
  - IBU, EMQ, Richardson–Lucy and Shepp–Vardi are this same iteration.
- **Optimality certificate.** At the MLE the KKT conditions are $g_k = \frac1W\sum_i w_i L_{ik}/(L_i\cdot\hat\rho)\le 1$, with equality where $\hat\rho_k>0$. **R-OUT-2:** point estimates report the gap $\max_k g_k-1$.
- **MAP.** MAP estimates under a Dirichlet prior use $\hat n_k+\alpha_k-1$, which is valid for $\alpha_k\ge 1$. For smaller $\alpha_k$ (and Pólya trees) the simplex density is unbounded at the boundary, so `space="auto"` uses the unconstrained mode instead; `space="simplex"` clips at the boundary, with a warning.
- **Mode in unconstrained coordinates.** The mode of the density in log-ratio coordinates, used as the Laplace centre, is the MAP with exponent $\alpha_k$ instead of $\alpha_k-1$.
- **Acceleration.** EM converges slowly when per-object PDFs are broad relative to the bins (large fraction of missing information). SQUAREM acceleration is the default for EM (R-INF-MAP-2). For Dirichlet priors with $\alpha_k\ge1$ the objective is concave and mix-SQP (Kim et al. 2020) replaces EM (D15).

---

## 4. Inputs: data model and conventions

### 4.1 `Bins`

`Bins` describes the support. It is immutable.

| Constructor | Meaning |
|---|---|
| `Bins(edges)` | 1-D edges (monotone, at least two values) |
| `Bins.linear(lo, hi, n)` / `Bins.log(lo, hi, n)` | convenience 1-D grids |
| `Bins.grid(edges_x, edges_y, ...)` | N-D rectilinear grid; flattened in C order, `shape` kept |
| `Bins.categories(labels_or_K)` | unordered categories, $v_k=1$ |

Attributes and methods:
- Sizes and geometry: `K`, `shape`, `ndim`, `ordered`, `edges` (tuple of arrays), `centers` (`(K,)` in 1-D, otherwise `(K, ndim)`), `volumes` `(K,)`, `labels`.
- Lookup: `index(values) -> (n,) int` (−1 outside), `contains(values)`.
- Array shaping: `unflatten(a)` and `flatten(a)` to move per-bin arrays between flat `(K,)` and grid shape.
- Structure for priors: `neighbors()` (adjacency, used by GMRF priors), `coarsen(factor or groups)`, `__eq__`, `__repr__`.

R-BINS-1: smoothness priors require ordered grids and raise a clear error on categories.

### 4.2 `Likelihoods`

`Likelihoods` is the container for the $N\times K$ bin-averaged likelihood matrix plus metadata.

- **Storage.** A dense `float64` ndarray, or a `scipy.sparse` CSR matrix for localised PDFs.
- **Row normalisation.** Each row is normalised for numerical stability: the row maximum is 1, and `log_scale[i]` is kept so absolute log-likelihoods can be recovered for evidence and cross-validation.
- **Fields:**
  - `values`: the row-normalised matrix.
  - `log_scale`: the per-row log normalisation.
  - `weights`: `(N,)`, default ones.
  - `bins`.
  - `meta`: a dict recording provenance, including the constructor used and any interim prior applied.

**R-INPUT-3 (validation):** non-finite values and negative entries raise. All-zero rows raise, or are dropped with a warning if `drop_empty=True`. Rows with support in a single bin are allowed. Zero-volume bins raise.

**Constructors (must):**

| Constructor | Input | Conversion |
|---|---|---|
| `Likelihoods(values, bins=None, representation="density", weights=None, fractional_weights=False)` | low-level: likelihoods (bin-averaged values, or masses); integer weights unless `fractional_weights` | §3.5; `bins=None` gives `Bins.categories(K)` |
| `Likelihoods.from_log(log_values, ...)` | log-likelihoods | stable exponentiation |
| `Likelihoods.from_table(values, bins, kind="likelihood" \| interim=..., representation=None, interim_representation=None)` | a table of likelihoods or of posteriors (one of `kind`, `interim`) | $L\propto P/\Pi$ for posteriors (§3.4–3.5) |
| `Likelihoods.from_samples(samples, bins, interim, sample_weights=None)` (no default interim prior, per R-INPUT-1) | per-object posterior samples (list of arrays, or `(N, S[, ndim])`) | histogram masses divided by interim masses; out-of-range mass reported; per-object effective sample counts stored in `meta` |
| `Likelihoods.from_gaussian(mean, sigma, bins, tol=1e-3)` | Gaussian measurement model, 1-D or N-D with diagonal $\sigma$ | exact CDF differences on linear axes, adaptive quadrature in $u$ otherwise |
| `Likelihoods.from_density(f, bins, kind="likelihood" \| interim=..., tol=1e-3, max_depth=12, n_sub=None)` | a vectorised function or distribution objects | exact log-CDF masses where valid, else adaptive quadrature (DECISIONS #23, #64) |
| `Likelihoods.from_binned(response, counts=None, values=None)` | binned measurements: counts $c_j$ (or values) in measured bins and a `Response` $R_{jk}=P(\text{measured }j,\text{ observed}\mid\text{true }k)$ from a matrix, simulated events or a labelled validation set (DECISIONS #27, #29) | rows $R_{j\cdot}$ with weights $c_j$ on the response's truth bins; efficiency becomes an `"observed"` selection |

Methods:
- `compress(decimals=None)`: merge identical rows into integer weights (sparse stays sparse).
- Views and copies: `subset(idx)`, `to_dense()`, `to_sparse(rtol=…)`.
- `with_weights(w)`.
- `rebin(groups)`: only where valid. Merging bins requires the within-bin population shape, so the documentation recommends inferring on fine bins and summarising functionals.
- `__len__`, `N`, `K`, `nnz`.

**R-INPUT-4:** `infer` and the models also accept a raw `(N, K)` array, treated as `Likelihoods(array)` with categorical bins, plus an optional `bins=` keyword.

---

## 5. Priors

All priors share one interface:
- `log_prob(rho)` (density on the simplex) and `sample(n, rng)`.
- An unconstrained parametrisation, used by gradient-based engines.
- Where applicable, `conjugate_update(counts)` (Gibbs, VB) and `map_update(expected_counts)` (EM).
- Flags `conjugate` and `kind` (`"fractions"` or `"rates"`).

### 5.1 The conjugate simplex family (must)

| Prior | Parameters | Notes |
|---|---|---|
| `Dirichlet(alpha=1.0)` | scalar or `(K,)` | default prior; `Dirichlet.reference(K)` gives $\alpha=1/K$ (Berger, Bernardo & Sun 2015); `Dirichlet.from_counts(m, strength=1, base=…)` builds a reference-sample prior; `Dirichlet.from_base_measure(volumes, concentration)` gives $\alpha_k\propto v_k$, which keeps the prior consistent when bins are merged |
| `GeneralizedDirichlet(a, b)` | `(K-1,)` Beta pairs, stick-breaking order = bin order or `order=` | Connor–Mosimann; allows positive correlations |
| `DirichletTree(tree, alphas)` | nested partition plus node Dirichlet parameters | Gibbs, EM and VB stay closed-form |
| `PolyaTree(bins, c=1.0, power=2, base=None)` | dyadic tree over an ordered grid | $\alpha$ at depth $d$ is $c\,d^{\,\text{power}}$, centred on a base measure; multi-scale and conjugate |

Facts the implementation relies on, all verified numerically:
- Dirichlet ⊂ GD ⊂ Dirichlet-tree. A finite Pólya tree is a binary Dirichlet-tree.
- The tree log-density is linear in the log node masses:
  $$\log p(\rho)=\text{const}+\sum_{\text{leaves }k}(\alpha_{\mathrm{pa}(k),k}-1)\log\rho_k+\sum_{\text{internal }c\neq\text{root}}(\alpha_{\mathrm{pa}(c),c}-A_c)\log\rho_c,\qquad A_c=\textstyle\sum_{j\in\mathrm{ch}(c)}\alpha_{cj}.$$
  This gives cheap gradients for NUTS.
- Each tree node updates as $\theta_c\mid n\sim\mathrm{Dir}(\alpha_c + n_{\mathrm{ch}(c)})$.

### 5.2 The logistic-normal (smooth) family (must)

$\rho = \mathrm{softmax}(\eta)$ with a Gaussian prior on $\eta$. Each prior is written in whitened form
$$\eta=\mu+B\,(s(\theta)\odot\xi),\qquad \xi\sim\mathcal N(0,I_J),$$
so that hyperparameters $\theta$ can be sampled jointly by NUTS. The softmax shift direction is unidentified and removed (constant mode dropped).

| Prior | Basis $B$ | Scales $s(\theta)$ | Hyperparameters |
|---|---|---|---|
| `LogisticNormal(mean, cov)` | Cholesky of `cov` | 1 | none (fixed) |
| `LogisticRandomWalk(order=1\|2, scale=HalfNormal(1.0))` | eigenvectors of the RW/ICAR precision on `bins.neighbors()` (1-D and N-D) | $\tau\,\lambda_j^{-1/2}$ | scale $\tau$ (fixed or prior); null-space trend modes get a broad Gaussian |
| `LogisticGP(kernel="matern32"\|"rbf", amplitude=…, length_scale=…)` | Hilbert-space GP basis (Solin & Särkkä 2020) on 1-D/2-D grids | spectral density at $\sqrt{\lambda_j}$ | amplitude and length scale (fixed or prior) |

- `mean` defaults to the log bin volumes, a flat density (D16). It may instead be the log of a **reference distribution**, such as a calibration-sample histogram. The amplitude then controls how far the population may depart from it. This is the coherent version of frankenz's "flexible reference" (Appendix B).
- Hyperpriors are small classes with log-density and gradient in log space: `HalfNormal`, `LogNormal`, `InverseGamma`, and `Fixed(value)`.

### 5.3 Rates (must: conjugate; should: logistic)

- **Must:** `GammaRates(shape=…, rate=…)` gives conjugate rates (Gibbs, EM, NUTS on $\log\lambda$). Results expose `rates` and `fractions`.
- **Should:** a logistic-normal family with `rates=True` adds a level parameter, giving log-Gaussian-process rates as in LVK's binned GP models.

### 5.4 Hierarchical Dirichlet (should)

$\rho\sim\mathrm{Dir}(a\,m)$, where $a\sim p(a)$ and $m$ is either fixed or itself uncertain, $m\sim\mathrm{Dir}(\beta+\text{reference counts})$. Sampled by NUTS. This replaces frankenz's ad hoc reference step with a coherent joint model.

### 5.5 Defaults and guidance

- **Default prior:** `Dirichlet(1.0)` (uniform on the simplex). It is the most common choice and is robust for every engine.
- **Documented trade-offs:**
  - With many bins and few objects, $\alpha=1$ adds $K$ pseudo-counts and flattens sparse tables; `Dirichlet.reference` ($\alpha = 1/K$) avoids this.
  - Symmetric $\alpha$ over unequal bins biases grouped fractions; `from_base_measure` avoids this.
  - Very small $\alpha$ gives heavy log-space tails, which slow gradient-based samplers.
- **Smoothness:** recommend `LogisticRandomWalk(order=2)` when bins are finer than the PDFs, which is the ill-posed regime.

---

## 6. Inference engines

All engines take a `PopulationModel`, accept `rng` (SPEC 7: an int seed, `numpy.random.Generator` or `None`), and return a `PopulationResult` (or a `PointEstimate` for MAP).

### 6.1 Parametrisations

| Prior family | Unconstrained coordinates |
|---|---|
| Simplex priors | isometric log-ratio: $\rho=\mathrm{softmax}(V\eta)$, with $V$ the $K\times(K-1)$ Helmert basis and log-Jacobian $\sum_k\log\rho_k$ + const (Stan ≥ 2.37 made the same choice) |
| Logistic family | whitened $(\xi, \log\theta)$ |
| Rates | $\log\lambda$ |

- **Gradients** are analytic.
  - Softmax models: $\partial\ell/\partial\eta_k=\hat n_k-W\rho_k$, which becomes $\hat n_k - W\rho_kS_k/(S\cdot\rho)$ with selection.
  - Rates: $\partial\ell/\partial\log\lambda_k=\hat n_k-S_k\lambda_k$.
- **Cost:** one gradient costs $O(\mathrm{nnz}(L))$.

### 6.2 NUTS (default, must)

- **Algorithm:** multinomial NUTS with Stan's generalised and extra U-turn checks, dual-averaging step size and Stan-style windowed **diagonal** metric adaptation. `metric="dense"` is optional. The prototype was verified on Gaussians, Dirichlet targets and the exact K=3 posterior.
- **Defaults:** 4 chains, 1000 warm-up and 1000 draws each, `target_accept=0.8`, `max_depth=10`, initialised near the MAP with jitter.
- **Reported:** divergences, tree depths, step sizes, R̂ and ESS.
- **Behaviour:** chains run round-robin, so a generator interface and progress reporting work. `run` can be called again to extend the chains (resumable).

### 6.3 Gibbs (must)

- **When it applies:** conjugate priors (§5.1, §5.3), with or without selection.
- **Restriction:** integer weights only; it raises otherwise.
- **Speed:** label sampling is vectorised by inverse CDF (dense) or by CSR row-segment search (sparse), in chunks to bound memory. Duplicate rows with multiplicity $m$ are drawn as $\mathrm{Mult}(m, r_i)$.
- **Optional outputs:** stored labels (all, or a subset of objects) and realised counts.
- **Warning:** its mixing degrades with the fraction of missing information, so integrated autocorrelation times of hundreds are common with broad PDFs (§6.10). The documentation states this, and the result's diagnostics flag a low ESS.

### 6.4 MAP / EM (must)

- **Algorithm:** mix-SQP for Dirichlet priors with $\alpha_k\ge1$, with selection for the flat prior through the observed-population fractions $\phi_k\propto S_k\rho_k$ (D15). SQUAREM-accelerated EM for the other conjugate priors. L-BFGS in unconstrained coordinates for logistic priors (MAP of $\xi$ with hyperparameters fixed, or optimised, giving type-II MAP) and for densities unbounded at the boundary.
- **Returns:** a `PointEstimate` with the optimality certificate (§3.9), the log posterior, the iteration count and `laplace()` to obtain a Laplace approximation.

### 6.5 Laplace (must)

- **Approximation:** a Gaussian in unconstrained coordinates at the mode, using the exact Hessian. For softmax models, $H_x=\mathrm{diag}(\hat n)-R^\top\mathrm{diag}(w)R-(A+W)(\mathrm{diag}\rho-\rho\rho^\top)$ plus the prior Hessian, at $O(NK^2)$ cost; finite-differenced gradients are used for large $K$.
- **Optional importance correction (`importance=True`, default):** Pareto-smoothed importance resampling against the exact posterior, reporting $\hat k$ and an approximate-results warning when $\hat k > 0.7$.
- **Accuracy:**
  - The prototype matched NUTS to ≲3% in s.d. for well-populated bins.
  - It overestimated s.d. by up to 2.3× for nearly empty bins, because log-space tails are not Gaussian.

### 6.6 Mean-field VB (must, flagged approximate)

- **Model:** $q(\rho)=\mathrm{Dir}(\gamma)$ and $q(z_i)=\mathrm{Cat}(\phi_i)$, using coordinate-ascent updates (RAIL's `VarInfStack` is this algorithm).
- **Why it is flagged:** it underestimates posterior widths, by 2–20× in our prototype as PDFs broaden (§6.10; Wang & Titterington 2005; HSC Y3). It is useful for initialisation and quick looks.
- **Warning:** results always carry `approximate=True` and raise an `ApproximationWarning`.

### 6.7 Bootstrap (should)

- **Method:** non-parametric (multinomial-weight) or Bayesian (Dirichlet-weight) bootstrap of the MAP estimator. Integer weights are multiplicities, so objects (not rows) are resampled; rate models use Poisson (or Gamma) weights so that the total varies; replicates start from an interior point.
- **Purpose:** frequentist intervals and comparison with HEP/EB practice.
- **Labelling:** results are flagged as a sampling distribution, not a posterior.

### 6.8 External samplers (must)

`PopulationModel` exposes:
- `log_prob(x)`, `log_prob_and_grad(x)`, `to_rho(x)` and `from_rho(rho)` in unconstrained coordinates (for emcee, BlackJAX, NumPyro and similar);
- `prior_transform(u)` from the unit cube for priors that support it (Dirichlet via Beta stick-breaking quantiles; GD; trees), for dynesty, nautilus and UltraNest.

`PopulationResult.from_samples(model, rho_samples, ...)` wraps external draws, so all downstream tools work on them.

### 6.9 Choosing a method

`method="nuts"` is the default. Documentation ranks the engines by when to use them:
- NUTS: the exact default.
- Gibbs: conjugate models, when joint labels are needed and PDFs are narrow relative to the bins.
- Laplace: very large N; corrected by importance resampling.
- MAP: quick looks and initialisation.
- VB and bootstrap: approximations.

`method="auto"` (should) picks NUTS unless the estimated cost of $\mathrm{nnz}(L)$ × expected gradient count exceeds a budget, then picks Laplace. The choice is logged on the result.

### 6.10 Prototype evidence behind these choices

| Benchmark | Results |
|---|---|
| K=3, N=300, σ/bin-width 0.3–1.2 | exact quadrature, Gibbs and NUTS agree in mean, s.d. and 5/50/95% quantiles to MC error; mean-field VB s.d. too small by 1.8–2.8× |
| K=30, N=2×10⁴, σ = 0.05–0.2 (bin width 0.1) | Gibbs max τ_int ≈ 217; NUTS max τ_int ≈ 1.8; NUTS ≈ 11 min-ESS/s vs Gibbs ≈ 1 |
| K=30, N=2×10⁴, σ = 0.2–0.4 | Gibbs max τ_int ≈ 430 (did not converge in 3,500 iterations); NUTS ≈ 4 min-ESS/s with ≈180 gradients per iteration; VB s.d. too small by ≈20× |
| Selection, fractions model | Gibbs with negative-binomial augmentation = NUTS on (S) |

---

## 7. Outputs

### 7.1 `PopulationResult`

Raw draws and metadata:
- `samples`: `(n_draws, K)` fractions, flattened over chains, with `chains` as `(n_chains, n_per_chain, K)`.
- `rates`: `(n_draws, K)` or `None`.
- `log_prob`: `(n_draws,)`.
- `method`, `approximate` (bool), `model`, `bins`, `prior`, `info` (engine statistics).

Summaries:
- `mean`, `median`, `std`, `cov()`, `corr()`.
- `quantile(q)`, `interval(level=0.68, kind="central"|"hdi")`.
- `summary(level=0.68)`: a table object with `__repr__`, `to_dict()` and `to_pandas()` (pandas optional).

Densities and derived quantities:
- `density(...)`: $\rho_k/v_k$ (per unit volume, for n(z) plots).
- `functional(f)`: the per-draw derived quantity, e.g. mean redshift `lambda r: r @ bins.centers`, or tail fractions.
- `rebin(groups)`: aggregated fractions; exact for $\rho$.

Individual objects:
- `assignment_probabilities(idx=None)`.
- `sample_assignments(idx=None, n_draws=None, rng=None)`.
- `realized_counts(n_draws=None, rng=None)`.
- `sample_values(idx, ...)`: continuous values within the assigned bin, uniform by default or proportional to a user callable.

New objects and cross-validation:
- `predict_proba(likelihoods_new)` and `log_predictive(likelihoods_new)`.

Diagnostics, plotting and persistence:
- `diagnostics`: a dict of R̂, bulk/tail ESS per bin, divergences, $\hat k$, KKT gap, as relevant.
- `check(strict=False)`: emits warnings, or raises when `strict=True`.
- `plot(...)`, `plot_corner(bins=...)`, `plot_trace()`.
- `save(path)`, `PopulationResult.load(path)` (`.npz`; versioned schema).
- `to_dict()`, `to_datatree()` (xarray optional; the ArviZ ≥ 1.0 format).

### 7.2 `PointEstimate`

Fields:
- `rho`, `rates`, `log_posterior`, `log_likelihood`, `kkt_gap`, `n_iter`, `converged`.

Methods:
- `laplace(n_samples=…)` returns a `PopulationResult`.
- `assignment_probabilities()`, `predict_proba()` and `log_predictive()`, plugging in $\hat\rho$.

### 7.3 Diagnostics and warnings

- **Warning classes:** `BinferenceWarning`, with subclasses `InputWarning`, `ConvergenceWarning` and `ApproximationWarning`.
- **Default thresholds:**
  - R̂ > 1.01;
  - bulk ESS < 100 per chain;
  - any divergence;
  - $\hat k$ > 0.7;
  - KKT gap > 1e-6 relative.
- **R-DIAG-1:** `binference.diagnostics` exposes `ess`, `rhat` (rank-normalised split, Vehtari et al. 2021), `autocorr_time` (emcee-style), `mcse`, `kkt_gap` and `missing_information(model, rho)`. The last predicts slow Gibbs and EM mixing and is reported by `infer` when relevant.

### 7.4 Plotting (optional dependency)

| Grid type | Plot |
|---|---|
| 1-D | density bands (median and 68/95%) with a step or smooth display, optional overlays of the naïve stack and the truth |
| 2-D | posterior-median image plus relative-uncertainty image |
| Categories | bar chart with intervals |

Also available: a corner plot over chosen bins or functionals, trace plots, and individual-object comparison (input likelihood vs shrunk posterior).

---

## 8. API

### 8.1 Layers

1. **One call:** `res = binference.infer(likelihoods, prior=None, *, selection=None, method="nuts", n_samples=1000, rng=None, progress=False, **engine_kwargs)`.
2. **scikit-learn style:** `binference.PopulationEstimator(prior=..., method=..., ...)`. `fit(X)` returns self, where `X` is a `Likelihoods` object or an array. It also provides `predict_proba(X)`, `score(X)` (mean log predictive density, for CV over binnings and priors), `sample(n)`, `get_params`/`set_params` and `result_`.
3. **Model + engine objects:** `model = binference.PopulationModel(lk, prior, selection=...)`; `eng = binference.samplers.NUTS(model, n_chains=4, ...)`; `res = eng.run(n_samples, rng=...)`. Also `for draws in eng.sample(n): ...` (generator) and `eng.run(more)` to extend.
4. **Raw densities:** `model.log_prob_and_grad`, `model.prior_transform`, `model.to_rho`.

### 8.2 Conventions

- **Randomness:** every function that draws random numbers takes `rng` (SPEC 7). There is no global state, and the same seed gives identical results across engines and platforms (up to BLAS floating-point differences).
- **Shapes:** objects first, bins last. Draws are `(n_draws, K)` and labels `(n_draws, n_objects)`.
- **Errors:** errors are `ValueError`/`TypeError` with actionable messages. Warnings are the classes in §7.3. Long runs never silently fail; non-finite log-probabilities raise with context.
- **Progress:** progress is optional (`tqdm` if installed, otherwise periodic text).
- **Typing:** public functions are type-annotated, with numpydoc docstrings and examples.

### 8.3 Minimal example (the README must show this)

```python
import numpy as np, binference

bins = binference.Bins.linear(0.0, 3.0, 30)
lk = binference.Likelihoods.from_gaussian(mu, sigma, bins)   # or .from_posteriors(P, bins, interim_prior=pi)
res = binference.infer(lk, rng=42)                           # Dirichlet(1) prior, NUTS, 4 chains
res.summary()
res.plot()
p_i = res.assignment_probabilities()                     # shrunk per-object posteriors
n_real = res.realized_counts()                           # histogram of these objects' true bins
```

---

## 9. Validation tools (user-facing, must)

**`binference.simulate`** builds mock catalogues, returning `(Likelihoods, Truth)`. `Truth` holds `rho`, the labels, the true values and the realised counts. Functions:
- `mock_1d`, `mock_2d` and `mock_categorical`;
- `population(...)`.

The mocks can include:
- heteroscedastic Gaussian or skewed noise;
- interim-prior posteriors;
- posterior samples;
- outliers;
- selection.

**`binference.validation`:**
- `sbc(simulate_fn, infer_fn, n_sims, ...)` runs simulation-based calibration. The ranks test the bin fractions and the joint log-likelihood. It returns an `SBCResult` with uniformity tests and a plot.
- `coverage(...)` checks credible intervals for chosen functionals.
- `exact_posterior(lk, prior, n_grid)` computes the K ≤ 3 quadrature reference used by the tests.

---

## 10. Performance targets (v0.1, single CPU core, NumPy)

| Operation | Target |
|---|---|
| Gradient evaluation | ≈ the cost of two BLAS matrix–vector products (dense) or two sparse ones |
| NUTS | ≥ 5 min-ESS/s on the N = 2×10⁴, K = 30 benchmark (prototype ≈ 11) |
| Gibbs | ≥ 200 iterations/s at N = 2×10⁴, K = 30 |
| MAP | < 1 s at N = 10⁵, K = 100 (mix-SQP: 0.9 s, 4 iterations, on an idle core; a few seconds with broader PDFs; SQUAREM-EM had not converged after 2000 iterations, 39 s) |
| Laplace | < 30 s at N = 10⁶, K = 100 (dense float64 needs ~800 MB, so the documentation recommends sparse storage or compression) |

All $N\times K$ temporaries are chunked (configurable `chunk_size`), so peak extra memory is $O(\text{chunk}\times K)$.

---

## 11. Engineering requirements

- **Packaging:**
  - Layout: `pyproject.toml` (hatchling), `src/` layout, import name `binference`.
  - PyPI distribution and import name: `binference` (the package was developed as `pastry`; see §13, D5).
  - Licence: MIT (existing).
- **Python:** ≥ 3.11 (CI 3.11–3.14).
- **Dependencies:**
  - Core: `numpy>=1.25` and `scipy>=1.11`.
  - Extras: `plot` (matplotlib), `xarray`, `docs`, `test` and `dev`.
- **Quality tooling:** ruff (lint and format), pre-commit, type hints, and pytest with warnings as errors in CI.
- **Tests:**
  - Two speed tiers: fast unit tests (< 2 min total) and `@pytest.mark.slow` statistical validation.
  - Coverage of ≥ 90% lines on core modules.
- **Docs:** Sphinx + MyST-NB (tutorials stored as MyST Markdown notebooks and executed at build time), numpydoc and the pydata-sphinx-theme. Pages: Getting started, User guide, Theory, Tutorials, API reference, Design notes, Changelog and Contributing.
- **CI (GitHub Actions):**
  - lint;
  - tests on a Linux and macOS × Python matrix;
  - slow statistical tests on one job;
  - a docs build that executes the tutorials.
- **Repository files:** `CITATION.cff`, `CHANGELOG.md` and `CONTRIBUTING.md` (including the validation philosophy).

---

## 12. Roadmap beyond v0.1

1. **Continuous populations (`binference.flows`, optional JAX extra built on flowjax).** The census says to keep the exact grid likelihood whenever a grid exists:
   - Smooth population maps $\theta\mapsto\rho(\theta)$ (splines, monotone flows on bin masses, small MLPs) are fitted inside the exact sum and sampled with HMC for honest uncertainty.
   - True continuous flows are reserved for dimension ≥ 3, or for inputs given as analytic likelihoods or posterior samples. Objectives:
     - Monte Carlo estimates of the marginal likelihood (IWAE-style with K draws, or the de-biased SUMO/Russian-roulette estimator), with per-object ESS diagnostics;
     - interim-prior reweighting for sample inputs;
     - stochastic EM;
     - joint hierarchical sampling.
   - Uncertainty comes from ensembles, bootstrap or HMC over small flows, all judged by SBC against the Dirichlet reference. Meta's `flow_matching` package is CC BY-NC licensed, so it is avoided.
2. **Sample-variance (overdispersion) models:** 3sDir-like inflation, or $\rho_\text{sample}\sim\mathrm{Dir}(c\,\rho_\text{pop})$.
3. **Speed:** a JAX/GPU backend for the likelihood kernels and NUTS, and minibatch or stochastic methods for very large N.
4. **Selection:** uncertain selection functions, and selection on observed data with per-object terms.
5. **Model comparison:** evidences via nested-sampling integrations, and cross-validation utilities for choosing binnings and priors.
6. **Interoperability:** qp, SACC, CosmoSIS n(z)-realisation exports, a RAIL summariser wrapper, and a QuaPy-compatible quantifier wrapper.
7. **Covariate-dependent and multi-population models:** e.g. tomographic bins sharing structure.

---

## 13. Decisions taken autonomously, and open questions for the user

These were decided without the user's input so that work could proceed. Each is easy to revisit.

| # | Decision | Rationale | Alternatives |
|---|---|---|---|
| D1 | NumPy/SciPy-only core (plus the pure-Python `threadpoolctl`, D14); JAX later as an optional backend | install-anywhere usability (emcee-like); the gradient is just two matrix–vector products | JAX-first core |
| D2 | NUTS is the default engine, with ILR coordinates | Gibbs mixes poorly with broad PDFs (§6.10); exact; Stan made the same simplex choice | Gibbs default (frankenz, Leistedt+16) |
| D3 | Default prior `Dirichlet(1.0)` | familiar and robust for samplers | α = 1/K (BBS reference prior), α ∝ bin volume |
| D4 | Posteriors require an explicit interim prior | correctness | silent flat default |
| D5 | Distribution and import name `binference` | renamed from `pastry` (whose PyPI name was taken) in the walkthrough, DECISIONS #45 | `pastry-inference` |
| D6 | Python ≥ 3.11 | astronomy users lag SPEC 0 | ≥ 3.12 (SPEC 0) |
| D7 | Flows deferred to the roadmap, with a concrete design | the census found no evidence flows beat grid models for binned inputs, and no tool calibrates them; doing the discrete core well comes first | ship an experimental flows module now |
| D8 | Research notes (`research_notes/`) and the landscape report (`reports/`) are kept in the repository; the report is also included in the documentation | traceability | separate repository or wiki |
| D9 | Arrays passed as `selection` require an explicit `selection_mode` (`"observed"` or `"true"`) | the two modes give different likelihoods, and the wrong one biases $\rho$ badly (§3.6) | a silent default |
| D10 | Laplace importance sampling uses a Student-t (df = 5) proposal, and flags results with Pareto $\hat k\ge0.7$ as approximate | Gaussian proposals gave borderline $\hat k$ even for K = 3; for K ≳ 10, near-empty bins have one-sided tails and $\hat k\approx1$ | Gaussian proposal; no flag |
| D11 | MAP for tree priors is the mode of the density on the simplex (branch probabilities proportional to their coefficients) | matches the Dirichlet convention; the unconstrained-space mode remains available via `space="unconstrained"` | unconstrained mode everywhere |
| D12 | For logistic priors, NUTS reparametrises (partial non-centring, weights $1/(-H_{jj})$ from the curvature after 25 warm-up iterations), starts a dense metric from the Hessian, and targets acceptance 0.9. All modes stay non-centred when the median s.d. of their log-scales *given the field* exceeds 0.28 | 3–10× more effective samples per second than the whitened form, and far fewer divergences, for K ≥ 10 (7 divergences in 28 validation runs, from K = 3 to the 144-bin 2-D tutorial); without the gate, weakly constrained scales (K ≤ 6, N ≤ 40) re-create a funnel (independent review, finding 7). A gate on the *marginal* Laplace s.d. fired spuriously when a length scale was weakly identified and made GP fits 20× slower; deciding per mode instead of per prior mixed worst; the user's model object is not modified | non-centred field with a diagonal metric (Stan's default advice) |
| D13 | `LogisticGP`'s default length-scale prior is an inverse gamma with 1% mass below two bin widths and 1% above the range | removes length scales the grid cannot resolve, which caused divergences, and follows Betancourt's robust-GP advice | log-normal centred on 0.2 × range |
| D14 | NUTS chains can run in worker processes (`n_jobs`), defaulting to serial. Workers split the cores' BLAS threads between them (`threadpoolctl`, a small pure-Python dependency) | the same random streams either way (draws agree up to rounding); forked workers otherwise inherit a full BLAS thread pool each, and oversubscription made 4 chains on 4 cores slower than one serial process; process pools have pitfalls in notebooks and on platforms that spawn | parallel by default; environment variables (ignored by forked workers) |
| D15 | MAP uses mix-SQP for Dirichlet priors with $\alpha_k\ge1$ (with selection: flat prior only, solved in $\phi_k\propto S_k\rho_k$) | 5–10 second-order iterations to a KKT gap of $10^{-10}$, where SQUAREM-EM needs $10^4$ or more iterations with broad PDFs | EM everywhere |
| D16 | Logistic priors default to a mean of log bin volumes (a flat density) | the penalty then acts on the log-density whatever the binning; with log bins a zero mean would centre the prior on density ∝ 1/z | zero mean (equal fractions) |

**Open questions** (they do not block v0.1):
1. Package and distribution name (D5): settled as `binference`.
2. Should the default prior be α = 1 or the reference prior α = 1/K? This matters only in the many-bins, few-objects regime.
3. Priority among roadmap items, especially the flow-based continuous densities.
4. Whether to make the repository public early. JOSS now requires more than six months of public history.

---

## Appendix A: the same problem in other fields

| Field | Name for $\rho$ | Name for $L_{ik}$ | Their EM | Their uncertainty tool |
|---|---|---|---|---|
| Photo-z | n(z) | photo-z likelihood (posterior ÷ interim prior) | iterated stacking | Gibbs (Leistedt+16), VB (RAIL VarInfStack) |
| Stellar populations | CMD / Hess diagram | per-star CMD likelihood | — | Gibbs (Leistedt & Hogg 17) |
| GW populations | binned rate density | per-event posterior samples ÷ PE prior | — | NUTS on binned GP (PixelPop, BinnedCBCMasses) |
| HEP | true spectrum | response-matrix column | IBU / D'Agostini | FBU, bias-corrected bootstrap |
| Empirical Bayes | mixing distribution g | kernel matrix | Kiefer–Wolfowitz EM | delta method, F-localisation |
| ML quantification | class prevalence | classifier posterior ÷ training prior | EMQ / SLD | BayesianCC, bootstrap regions |
| Fisheries / genetics | stock proportions | genotype likelihood | conditional ML (SPAM, ONCOR) | Gibbs (rubias, BAYES) |

## Appendix B: lessons from the `frankenz` samplers

The full audit is in `research_notes/Population inference software landscape/frankenz_audit.md`; the landscape report summarises it. The findings that shape this spec:

1. **Same target.** The population and hierarchical samplers target the same marginal posterior for $\rho$ (§3.3), so binference offers one model with several engines rather than two models.
2. **The hierarchical (data-augmentation Gibbs) sampler is exact** and is kept, vectorised, as `Gibbs` (§6). Its mixing degrades sharply once likelihoods are as wide as the bins (median autocorrelation time 7 → 180 sweeps), which is why NUTS is the default.
3. **The pairwise population sampler is invalid**: its state-dependent proposal width lacks a Hastings factor and is frozen across inner steps, so it absorbs low-mass bins at ~1e-13 and inflates other SDs by 8–18%. Not ported; any future derivative-free kernel must pass the exact-K=3 and Geweke tests (test plan §2.6, §2.11).
4. **The "flexible reference" step is not the full conditional of a coherent joint model** (Geweke max |z| = 70). It is replaced by the power prior `Dirichlet.from_counts(m, strength=λ)`, §5.2's reference-centred logistic prior and §5.4's hierarchical Dirichlet.
5. **Inputs must be likelihoods**, and the likelihood factor is the bin *average* (§3.1). Posteriors require the interim prior (§4.2); unequal bins require an explicit `representation`.
6. **Notation.** The notebooks use $P(z\mid g)$ for the population, the individual posterior and the likelihood. §3 replaces this with $\ell_i$, $L_{ik}$, $\pi_k$, $\rho$ and $r_{ik}$ throughout.
7. **Software defects to avoid** (each now has a test): non-resumable runs, incomplete `reset`, global or legacy RNG state, bins frozen at exactly zero, exact-zero Dirichlet draws for small concentrations, and no input validation.

## Appendix C: key references

- Leistedt, Mortlock & Peiris 2016, MNRAS 460, 4258.
- Leistedt & Hogg 2017, AJ 154, 222.
- Malz & Hogg 2022, ApJ 928, 127.
- Hogg, Myers & Bovy 2010.
- Foreman-Mackey, Hogg & Morton 2014.
- Mandel, Farr & Gair 2019.
- D'Agostini 1995.
- Kuusela & Panaretos 2015.
- Stanley, Patil & Kuusela 2022.
- Efron 2016.
- Koenker & Mizera 2014.
- Kim et al. 2020 (mix-SQP).
- Ignatiadis & Wager 2022.
- Saerens et al. 2002.
- Moreo & Sebastiani (QuaPy).
- Ziegler & Czyż 2024.
- Connor & Mosimann 1969.
- Minka (Dirichlet-tree).
- Lavine 1992 (Pólya trees).
- Berger, Bernardo & Sun 2015.
- Linderman, Johnson & Adams 2015.
- Solin & Särkkä 2020.
- Hoffman & Gelman 2014.
- Betancourt 2017.
- Vehtari et al. 2021.
- Talts et al. 2018.
- Modrák et al. 2023.
- Varadhan & Roland 2008 (SQUAREM).
- Vehtari et al. 2024 (PSIS).
- Dockhorn et al. 2020.
- Vandegar et al. 2021.
- Rozet et al. 2024.
