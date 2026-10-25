# Audit of frankenz population / hierarchical inference

**Status:** IN PROGRESS (2026-10-22). Code-reading findings are drafted below; numerical checks are being
run and the tables in Section 3 will be filled as each script finishes. Anything marked *pending* has
not been verified numerically yet.

**Materials audited.** `joshspeagle/frankenz` shallow clone at commit `b753e64` ("HB"):
`frankenz/samplers.py` (535 lines: `loglike_nz`, `population_sampler`, `hierarchical_sampler`),
helpers in `frankenz/pdf.py` (`gaussian`, `gaussian_bin`, `gauss_kde`), and the tutorials
`demos/5 - Population Inference with Redshifts.ipynb` (NB5) and
`demos/6 - Hierarchical Inference with Redshifts.ipynb` (NB6). Notebook cell numbers are the 0-based cell
indices in the `.ipynb` files (markdown and code cells counted together). Line numbers refer to
`samplers.py` unless stated otherwise.

**Environment.** Python 3.13, NumPy 2.5.4, SciPy 1.18.1, single thread. `samplers.py` and `pdf.py`
were copied verbatim (SHA-256 checked) into the scratch folder and imported from there; **no patches
were needed** for modern NumPy/SciPy.

---

## 1. What frankenz implements, in clean notation

### 1.1 Model

| symbol | meaning |
|---|---|
| $i=1..N$ | objects, data $d_i$ |
| $k=1..K$ | bins with edges $e_0<\dots<e_K$, widths $\Delta_k=e_k-e_{k-1}$ |
| $\ell_i(z)=p(d_i\mid z)$ | per-object likelihood (arbitrary per-object constant allowed) |
| $L_{ik}=\Delta_k^{-1}\int_{e_{k-1}}^{e_k}\ell_i(z)\,dz$ | **bin-averaged** likelihood |
| $\rho\in\mathcal S^{K-1}$ | population bin fractions (probability mass per bin), $\sum_k\rho_k=1$ |
| $n(z\mid\rho)=\sum_k\rho_k\,\mathbb 1[z\in k]/\Delta_k$ | top-hat (histogram) population density |
| $\alpha$ | Dirichlet concentration; $m$ = reference-sample bin counts (true labels), $N_{\rm ref}=\sum_k m_k$ |
| $z_i\in\{1..K\}$, $n_k=\#\{i:z_i=k\}$ | latent bin labels and their counts |
| $\pi_k$ | interim prior mass per bin used by a photo-z code that outputs posteriors $p_{ik}$ |

Marginal likelihood of one object under the histogram basis:
$p(d_i\mid\rho)=\int\ell_i(z)\,n(z\mid\rho)\,dz=\sum_k\rho_k L_{ik}=L_i\cdot\rho$.

Collapsed ("population") posterior, with a representative reference sample folded into the prior:

$$p(\rho\mid d,m)\;\propto\;\prod_k\rho_k^{\alpha_k+m_k-1}\;\prod_{i=1}^N L_i\cdot\rho .$$

Augmented ("hierarchical") form: $z_i\mid\rho\sim{\rm Cat}(\rho)$, $p(d_i\mid z_i=k)=L_{ik}$, so
$p(\rho,z\mid d)\propto{\rm Dir}(\rho;a)\prod_i\rho_{z_i}L_{iz_i}$ with $a=\alpha+m$. Summing over $z$ gives
exactly the collapsed posterior above, so **the two frankenz samplers target the same marginal
posterior for $\rho$ when `population_sampler` uses a ${\rm Dir}(\alpha+m)$ log-prior and
`hierarchical_sampler` uses `alpha` $=\alpha+m$** (NB5 cell 68/71 vs NB6 cell 53). Full conditionals:
$P(z_i=k\mid\rho,d_i)=L_{ik}\rho_k/(L_i\cdot\rho)$ and $\rho\mid z\sim{\rm Dir}(a+n)$.

Likelihoods vs posteriors: if a code reports $p_{ik}\propto\int_k\ell_i(z)\pi(z)dz$ under an interim prior
that is constant within bins with bin mass $\pi_k$, then $p_{ik}\propto\pi_kL_{ik}$ and the correct input is
$L_{ik}\propto p_{ik}/\pi_k$. Feeding $p_{ik}$ instead gives $\prod_i\sum_k L_{ik}\pi_k\rho_k$, i.e. the
interim prior is counted twice. The tutorials are unaffected only because their "PDFs" are Gaussian
likelihoods normalised over $z$ on equal-width bins (a flat interim prior).

Gaussian mock used by the tutorials: $\mu_i\sim\mathcal N(z_i,\sigma_i^2)$, so
$\ell_i(z)=\mathcal N(\mu_i;z,\sigma_i^2)$ and
$\int_k\ell_i\,dz=\Phi((e_k-\mu_i)/\sigma_i)-\Phi((e_{k-1}-\mu_i)/\sigma_i)$, which is what
`pdf.gaussian_bin` returns (bin-*integrated*; it equals $\Delta_kL_{ik}$, harmless only for equal widths).

### 1.2 `population_sampler` (samplers.py:83-308)

State $\rho$ with cached overlaps $o_i=L_i\cdot\rho$; target $\pi(\rho)\propto{\rm prior}(\rho)\prod_io_i$
(`loglike_nz`, l.24-80; the prior is a user function, flat if omitted). One saved sample = `thin` (default
400) pair updates. Each pair update:

1. draw an ordered pair $(j,l)$, $j\neq l$, uniformly; direction $t=e_j-e_l$ (keeps $\sum\rho=1$);
2. $h=10^{-4}\min(\rho_j,\rho_l,1-\rho_j,1-\rho_l)$; central finite difference
   $g=[\log\pi(\rho+th/2)-\log\pi(\rho-th/2)]/h$ (l.272-282);
3. scale $s(\rho)=\min(1/|g|,\;\rho_j,\rho_l,1-\rho_j,1-\rho_l)$ (l.284-287);
4. `mh_steps` (default 3) Metropolis steps $\rho'=\rho+t\varepsilon$, $\varepsilon\sim\mathcal N(0,s^2)$,
   accepted with $\min(1,\pi(\rho')/\pi(\rho))$, **with $s$ held at its value from the start of the pair update** (l.290-305).

Overlaps are updated incrementally, $o_i\leftarrow o_i+\varepsilon(L_{ij}-L_{il})$, which is exact and
$O(N)$ per proposal.

### 1.3 `hierarchical_sampler` (samplers.py:311-535)

Each saved sample comes after `thin` (default 5) sweeps of
1. $z_i\sim{\rm Cat}(L_i\odot\rho/L_i\cdot\rho)$ for every object (Python loop of `multinomial(1, .)`), giving $n$;
2. $\rho\sim{\rm Dir}(\alpha+n+k)$, with $k=0$ when there is no reference sample;
3. *"flexible reference"* (only if `ref_sample`$=m$ is passed):
   $k\sim{\rm Mult}\big(N_{\rm ref},\,(m+\beta+N\rho)/(N_{\rm ref}+B+N)\big)$ with $B=\sum_k\beta_k$, $\beta$ defaulting to 1.

Steps 1-2 form an exact two-block Gibbs sampler (data augmentation) for $p(\rho,z\mid d)$. Step 3 is
analysed in finding F3.

---

## 2. Findings

Severity: **critical** = can give wrong scientific answers silently; **major** = wrong behaviour or
documentation likely to cause wrong answers or large waste; **minor** = edge cases, robustness, performance;
**cosmetic** = notation or wording.

*(Table being filled; see Section 3 for the numerical evidence.)*

---

## 3. Numerical results

*pending*

---

## 4. What is right and worth keeping

*pending*

---

## 5. Recommendations for pastry

*pending*

---

## Appendix: scripts and outputs

All under `/tmp/claude-0/-home-user-pastry/63ffb1b4-5826-5f0b-ade5-c8209761737e/scratchpad/frankenz_audit/`.
*pending*
