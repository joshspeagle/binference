# Photometric-redshift n(z) inference from per-galaxy PDFs/likelihoods: methods and software census (status 22 October 2026)

How this was checked: repository metadata came from the GitHub search API through the GitHub MCP tool (stars, `created_at`, `pushed_at`/`updated_at`, licence). The session proxy blocked github.com HTML pages and per-repository commit listings, so "last push" means the API's `pushed_at` (or `updated_at` where only that was available). File contents came from raw.githubusercontent.com. Package status came from the PyPI JSON API, checked 2026-10-22. Source code was read from PyPI sdists already in the scratchpad: pz-rail-base 2.0.11, pz-rail 2.0.0, pz-rail-som 2.0.1, pz-rail-sompz 0.0.4, pz-rail-sklearn 2.0.3, pz-rail-yaw 2.0.0 and qp-prob 1.1.4. Paper metadata came from the arXiv API, and paper content from arXiv, ar5iv or the scratchpad full texts. Where I could read only part of a paper, the entry says so. Nothing was installed or run.

## Q1. Which codes implement hierarchical or otherwise principled n(z) inference from per-galaxy PDFs, and what exactly does each do?

### Takeaway
Only one maintained, pip-installable, general-purpose tool turns a catalogue of per-galaxy p(z) into n(z) samples: the LSST DESC RAIL summarizers. Its PDF-based methods are limited to three:
- naive stacking with bootstrap errors;
- a point-estimate histogram;
- `VarInfStackSummarizer`, a mean-field variational-Bayes fit of the flat-Dirichlet–categorical model, i.e. pastry's baseline model. It uses the PDFs directly as likelihoods, evaluates them at grid points and has no interim-prior handling.

Its originators (Rau et al. 2022) and the HSC Y3 analysis both report that this mean-field Dirichlet scheme underestimates error bars.

Fully hierarchical PDF-based samplers do exist, but only as unmaintained research code:
- the Leistedt+2016 Gibbs sampler (Photoz-tools, BHM-nz);
- CHIPPR (Python 2, self-described "stale ... prototype");
- the population and hierarchical samplers in frankenz (last push 2018).

The survey pipelines (DES SOMPZ+3sDir+WZ, KiDS SOM+CC, HSC Y3 BHM, Euclid SOM/zPDF) mostly work from colour-space cells, calibration samples and clustering rather than per-galaxy PDFs. Separate tools then compress their n(z) ensembles into nuisance parameters or rank them for cosmology: CosmoSIS `nz_multirank` (hyperrank), LSSTDESC `nz_prior`, and analytic marginalisation.

### Cited Findings

#### A. LSST DESC RAIL (Python, MIT): the only maintained general "PDFs → n(z)" API
- The RAIL docs list these summarizers:
  - SOM (minisom, somoclu) in rail_som;
  - yet_another_wizz clustering-z in rail_yaw;
  - NaiveStack, VarInfStack and PointEstHist in rail_base;
  - NZDir in rail_sklearn.

  rail_sompz adds DES SOMPZ. No MCMC or hierarchical summarizer is listed. — [RAIL estimation docs (source)](https://github.com/LSSTDESC/rail/blob/main/docs/source/rail_stages/estimation.rst)
- **VarInfStackSummarizer, what the algorithm is** (rail_base 2.0.11 source):
  - Docstring: "Variational inference summarizer based on notebook created by Markus Rau ... appropriate for the likelihoods returned by template-based codes, for which the NaiveSummarizer are not appropriate".
  - Input handling: it evaluates every PDF at `nzbins` points `linspace(zmin,zmax,nzbins)` and initialises α=1.
  - Iteration (`n_iter`, default 100): E[log ρ_k] = ψ(α_k) − ψ(Σα); responsibilities γ_ik ∝ exp(E[log ρ_k])·p_i(z_k); N_k = Σ_i γ_ik; α_k = N_k + 1.
  - Output: `n_samples` (default 500) Dirichlet(α) draws as n(z) realisations, plus α itself as "single_NZ".
  - Summary: coordinate-ascent mean-field VB for a flat Dir(1) prior with categorical latent bins. There is no interim-prior division, the PDF is evaluated at grid points rather than integrated over bins, and "This algorithm needs all data in memory at once".

  — [var_inf.py](https://github.com/LSSTDESC/rail_base/blob/main/src/rail/estimation/algos/var_inf.py) (read from the [pz-rail-base 2.0.11 sdist](https://pypi.org/project/pz-rail-base/))
- **The same algorithm appears in HSC Y3** (Rau et al. 2023, arXiv:2211.16516, Appendix B), with these details:
  - "We make a mean-field ansatz for the variational distribution" q(ρ, φ_nz) ≈ q(ρ)q(φ_nz).
  - q(ρ) uses exp(ψ(α_j) − ψ(Σα) + log pz_ij), where pz_ij is "the lensing-weighted integral of each galaxy's likelihood over histogram bin j".
  - q(φ_nz) = Dir(α_0 + Σ_i γ_i).
  - Warnings: a mean-field ansatz "often lead[s] to the estimation of too narrow credibility intervals". "In our numerical experience, the under-coverage under reasonable regularization (e.g. by selecting broader histogram bins) is approximately 20%". The paper gives no test design for this figure.

  — [arXiv:2211.16516](https://arxiv.org/abs/2211.16516) (Appendix B read via [ar5iv](https://ar5iv.labs.arxiv.org/html/2211.16516))
- **NaiveStackSummarizer**: sums PDF values on a `linspace(zmin,zmax,nzbins+1)` grid. Its uncertainty is a "Poisson bootstrap, a variant of regular bootstrap" over galaxies (default 1000 samples); a masked variant handles tomographic bins. — [naive_stack.py](https://github.com/LSSTDESC/rail_base/blob/main/src/rail/estimation/algos/naive_stack.py)
- **PointEstHistSummarizer**: histogram of a point estimate (default `zmode`) with bootstrap realisations, output as qp.hist. — [point_est_hist.py](https://github.com/LSSTDESC/rail_base/blob/main/src/rail/estimation/algos/point_est_hist.py)
- **NZDirSummarizer** (rail_sklearn):
  - Described as a "simple version of TxPipe NZDir": kNN reweighting of a spectroscopic sample in colour space (Lima et al. 2008 DIR; default 10 neighbours).
  - Uncertainty: bootstrap of the spec-z sample (default 20 samples).
  - Its input is a photometric catalogue, not PDFs.

  — [nz_dir.py](https://github.com/LSSTDESC/rail_sklearn/blob/main/src/rail/estimation/algos/nz_dir.py)
- **SOM summarizers** (MiniSOM, SOMoclu; rail_som):
  - N(z) is a "weighted sum of the empirical N(z) consisting of the normalized histogram of spec-z values contained in the same SOM cluster as each photometric galaxy".
  - Cells with photometric but no spectroscopic objects go to an `uncovered_cell_file` and "should be removed from the sample".
  - Uncertainty: bootstrap over spec-z objects (default 20).

  — [somoclu_som.py](https://github.com/LSSTDESC/rail_som/blob/main/src/rail/estimation/algos/somoclu_som.py)
- **rail_sompz** (DES SOMPZ in RAIL):
  - Inputs: `spec_data`, `balrog_data` (deep plus injected wide photometry) and `wide_data`.
  - Model: n(z|b̂,ŝ) = Σ_{ĉ∈b̂} Σ_{c∈ĉ} p(z|c,ŝ) p(c|ĉ,ŝ) p(ĉ|ŝ).
  - Output: n(z) "as a qp ensemble".
  - Status: PyPI 0.0.4 (2026-03-03); last push 2026-10-19; 1 star.

  — [LSSTDESC/rail_sompz](https://github.com/LSSTDESC/rail_sompz); [PyPI](https://pypi.org/project/pz-rail-sompz/)
- **Example notebook** (S. Schmidt, "Last successfully run: Feb 9, 2026"):
  - "VarInfStackSummarizer tends to get a little wonky if you use too many bins, so we'll only use 25 bins".
  - On NZDir: "I implemented a bootstrap of the spectroscopic data rather than the photometric data ... I think that the spectroscopic variance is more important".
  - Usage: `vens = ri.estimation.algos.var_inf.var_inf_stack_summarizer(input_data=qp_data, zmin=0.0, zmax=3.0, nzbins=25, niter=10, n_samples=10)`, then `vens["output"]` (samples) and `vens["single_NZ"]`.
  - The notebook passes `niter`, but the source's config key is `n_iter`, so that argument does not set the iteration count as intended. I did not run it.

  — [Sampled_Summarizers.ipynb](https://github.com/LSSTDESC/rail/blob/main/interactive_examples/estimation_examples/Sampled_Summarizers.ipynb)
- **Attribution problems in the RAIL docs:**
  - The citation guide attributes "Variational inference stacking" to "Rau et al. (2022)" (ADS 2022MNRAS.509.4886R). This plausibly corresponds to Rau et al.'s composite-likelihood paper (arXiv:2101.01184, doi:10.1093/mnras/stab3290), but I did not confirm the mapping.
  - It attributes "Naive stacking" to "Myles et al. (2020)" with bibcode 2020arXiv200712178M. That bibcode is Malz & Hogg's CHIPPR preprint, so this attribution looks garbled.

  — [citing.rst](https://github.com/LSSTDESC/rail/blob/main/docs/source/_legacy/citing.rst); [arXiv:2101.01184](https://arxiv.org/abs/2101.01184); [arXiv:2007.12178](https://arxiv.org/abs/2007.12178)
- **RAIL v1 paper:** RAIL Team, van den Busch et al. 2025 (arXiv:2505.02928; doi:10.33232/001c.158200), "an open source Python library for at-scale probabilistic photo-z estimation". — [arXiv:2505.02928](https://arxiv.org/abs/2505.02928)
- **RAIL in use:** applied to Rubin Data Preview 1 with eight photo-z algorithms (Zhang et al. 2025, arXiv:2510.07370; MNRAS 2026).
  - n(z) method: "stacked n(z) estimates, obtained by summing per-galaxy photometric redshift PDFs", compared with point-estimate histograms. No n(z) uncertainty band is reported.
  - Caveats in the paper: "The stacked PDFs of machine learning methods show the imprint of the training sample", and several stacks "exhibit significant systematics that warrant further investigation".

  — [arXiv:2510.07370](https://arxiv.org/abs/2510.07370) (via [arXiv HTML](https://arxiv.org/html/2510.07370))

#### B. Hierarchical / Dirichlet research codes that take per-galaxy likelihoods
- **Leistedt, Mortlock & Peiris 2016** (arXiv:1602.05960; MNRAS 460, 4258; doi:10.1093/mnras/stw1304):
  - Model: a histogram f_ijk over (template type, redshift, reference magnitude) with a Dirichlet prior.
  - Sampler: a two-step Gibbs sampler (f | counts ~ Dirichlet; per-galaxy (t,z,m) | f ~ categorical).
  - Speed-up "by binning p({F}|t,z,m) in the second step of the Gibbs sampler, i.e. by using a binned likelihood", so only bin labels are drawn.
  - Cost: "Drawing 10^4 samples takes about ten minutes using a laptop". The sampler starts from the histogram of maximum-likelihood redshifts and "converged very quickly".
  - Extensions noted but not used: "prior knowledge about the correlations between bins can be incorporated by considering a prior ... following a generalised Dirichlet distribution. ... We do not consider this case in this paper".
  - Selection effects and tomographic cuts enter through a per-object selection variable s and "only affect the likelihood function, not the inference technique".

  — [arXiv:1602.05960](https://arxiv.org/abs/1602.05960) (full text read)
- **Code for Leistedt+2016:** `ixkael/Photoz-tools` contains notebooks plus `bayeshist.py`, an "MPI implementation of the hierarchical model and Gibbs sampler of Leistedt, Mortlock and Peiris (2016) for inferring histograms of underlying distributions with binned likelihoods". MIT; 3 stars; created 2016-08-29; last push 2020-04-14; not on PyPI. — [ixkael/Photoz-tools](https://github.com/ixkael/Photoz-tools)
- **Kyriacou, Mootoovaloo, Heavens, Jaffe, Leclercq & Kuijken 2026** (arXiv:2605.15078): applies the Leistedt+2016 template BHM to KiDS+VIKING-450.
  - Sampler: a 2-step Gibbs sampler that keeps the per-galaxy samples to build tomographic histograms.
  - Templates: PCA reduces the set to about 50.
  - Scale: "Applying this method to all galaxies in the KV450 sample would be too slow, so we use a subset" of 10^5 galaxies, with subset-to-subset changes "subdominant compared to the statistical error".
  - Results: better than stacked likelihoods on simulations; S8 = 0.756 ± 0.039.

  — [arXiv:2605.15078](https://arxiv.org/abs/2605.15078)
- **Code for Kyriacou+2026, likely:** `Harry45/BHM-nz`, "Bayesian Hierarchical Modelling of Redshift Distributions" (Harry45 appears to be co-author A. Mootoovaloo; the paper does not link it).
  - Contents: "reference/BL-2016 contains the original implementation of the Bayesian Hierarchical Model by Boris".
  - Use in cosmology: "The marginalisation is done by drawing random samples of the heights at each step in the likelihood calculation".
  - Status: MIT; notebooks; 1 star; last push 2023-08-27.

  — [Harry45/BHM-nz](https://github.com/Harry45/BHM-nz)
- **CHIPPR** (Malz & Hogg 2022, ApJ 928, 127, doi:10.3847/1538-4357/ac062f per [Crossref](https://api.crossref.org/works/10.3847/1538-4357/ac062f); arXiv:2007.12178):
  - Posterior (Eq. 7): p(φ|{d_j}) ∝ p(φ) Π_j ∫ p(z_j|d_j,φ*) p(z_j|φ)/p(z_j|φ*) dz_j, where φ* is the interim prior. I have written the product over galaxies outside the per-galaxy integrals; the ar5iv extraction was ambiguous on this point. This "in effect replaces the implicit prior with the sampled model hyperparameters, thereby converting the photo-z implicit posteriors into likelihoods".
  - n(z) model: a log-space histogram f(z;φ) = exp(φ_k).
  - Hyperprior: multivariate normal centred on φ*, with a squared-exponential covariance (q=1, e=100, t=q·10⁻⁵).
  - Sampler: emcee with 100 walkers; a maximum-marginal-likelihood estimate is also computed. Tests used 10^4 galaxies.
  - On stacking: "the stacked estimator is always excessively broad".
  - On a misspecified interim prior: results are "disastrous", and "Neither CHIPPR nor traditional alternatives can recover" n(z).

  — [arXiv:2007.12178](https://arxiv.org/abs/2007.12178) (details via [ar5iv](https://ar5iv.labs.arxiv.org/html/2007.12178))
- **chippr code** (`aimalz/chippr`):
  - README: "this code is stale and should be understood to be a prototype, originally scoped out for applicability to SDSS DR10-era data of low dimensionality ... will need a major upgrade for flexibility and computational scaling".
  - setup.py: version 0.1; requires emcee, matplotlib, numpy, pomegranate, scipy. The demo notebook is "for Python 2.7".
  - Status: MIT; created 2016-12-23; last push 2022-11-22; 2 stars; 19 open issues; not on PyPI. Read the Docs at chippr.readthedocs.io still resolves.
  - Predecessor: `aimalz/prob-z` (updated 2016).

  — [aimalz/chippr](https://github.com/aimalz/chippr); [aimalz/prob-z](https://github.com/aimalz/prob-z); [PyPI lookup 404](https://pypi.org/pypi/chippr/json)
- **frankenz** (`joshspeagle/frankenz`, `frankenz/samplers.py`):
  - `population_sampler` samples n(z) directly from the likelihood Π_i Σ_k pdf_ik·nz_k (`loglike_nz`), with an optional user log-prior. It uses "MH-in-Gibbs MCMC" over random pairs of bins: a move along (+1,−1) keeps the vector on the simplex, the step is scaled by a numerical gradient, and the defaults are thin=400 and mh_steps=3.
  - `hierarchical_sampler` is a Gibbs sampler for "jointly drawing redshift population distributions and individual redshift predictions". "Note that these must be *likelihoods*, since the prior is being explicitly modeled. Assumes a Dirichlet hyper-prior" (concentration α, flat by default).
  - It can take a `ref_sample` of reference counts with its own Dirichlet hyperprior β. These counts are added to the Dirichlet update and resampled each iteration.
  - Status: MIT; 18 stars; last push 2018-10-01; PyPI 0.1.5 (2018-04-30, single release, "Development Status :: 3 - Alpha"). README: "WARNING: This project is under active development and not yet stable"; "Documentation: Currently nonexistent".

  — [frankenz samplers.py](https://github.com/joshspeagle/frankenz/blob/master/frankenz/samplers.py); [frankenz README](https://github.com/joshspeagle/frankenz); [PyPI frankenz](https://pypi.org/project/frankenz/)
- **Sánchez & Bernstein 2019** (arXiv:1807.11873; doi:10.1093/mnras/sty3222): a hierarchical model over phenotype × redshift fractions f_zt.
  - Data: per-galaxy photometric likelihoods L_it, clustering against a tracer density (1 + b δ̂), and an informative prior from a spec/deep subsample, "p(f|N) ∼ Dir(N+M)".
  - "One critical approximation", replacing the stochastic density field by a point estimate δ̂, makes Gibbs sampling possible.
  - Run: 4 walkers × 2500 samples, R̂<1.03, "All the chains run in less than an hour in a two-core laptop". Sample variance of the correlations is ignored.
  - Code: no public code found.

  — [arXiv:1807.11873](https://arxiv.org/abs/1807.11873) (via [ar5iv](https://ar5iv.labs.arxiv.org/html/1807.11873))
- **Alarcon et al. 2020** (arXiv:1910.07127; doi:10.1093/mnras/staa2478) extends Sánchez & Bernstein 2019 to realistic N-body simulations. With an artificially biased spec sample (Δz̄≈0.05), the posterior bias falls to Δz̄≲0.003. — [arXiv:1910.07127](https://arxiv.org/abs/1910.07127)
- **Rau, Wilson & Mandelbaum 2020** (arXiv:1904.09988; doi:10.1093/mnras/stz3295): a hierarchical logistic Gaussian-process prior on log histogram heights, inferred from clustering cross-correlations jointly with galaxy-bias models.
  - Why an LGP: "encoding knowledge of the shape and smoothness" instead of "Dirichlet or flat priors".
  - Sampler: "Metropolis-Hastings within Gibbs" with elliptical slice sampling for the correlated LGP heights ("does not require gradients").
  - A photo-z prior enters as a multivariate normal fitted to (log) Dirichlet samples.
  - Code: no public code found.

  — [arXiv:1904.09988](https://arxiv.org/abs/1904.09988)
- **Rau et al. 2022, composite likelihood** (arXiv:2101.01184; doi:10.1093/mnras/stab3290): combines photometric redshift information with two-point cross-correlations against spectroscopic samples, reaching mean-redshift calibration of at least 0.002(1+z) on CosmoDC2. It is the likely origin of RAIL's VarInfStack:
  - "we have applied a mean field variational inference scheme that uses the Dirichlet as the variational distribution".
  - Then: "We found that the variational inference scheme yielded underestimated error bars, likely due to the restricted covariance structure of the Dirichlet that makes an independence assumption between neighboring bins". The authors moved to a logit-normal posterior with an analytic form.
  - Scaling: "the sampling approach did not scale well to the large galaxy samples expected for the first-year LSST observations", and subsampling "can lead to biased inferences".
  - Regularisation: oversmoothing with coarse bins "can lead to posteriors that can be too narrow"; Kuusela-style "merging bin regularization ... performs much better in terms of producing well-calibrated probability"; Tikhonov regularisation stabilises Hessian inversion; heuristic bin width ≈ ±2σ of individual PDFs.
  - Model checking: posterior predictive checks.

  — [arXiv:2101.01184](https://arxiv.org/abs/2101.01184) (via [ar5iv](https://ar5iv.labs.arxiv.org/html/2101.01184), first half read)

#### C. Survey calibration pipelines (DES, KiDS, HSC, Euclid)
- **DES SOMPZ** (Buchs et al. 2019, arXiv:1901.05005; Myles et al. 2021, arXiv:2012.08566; Campos et al. 2024, arXiv:2408.00922):
  - Phenotypes are wide and deep SOM cells, linked by a Balrog transfer function.
  - DES Y3 builds an ensemble of n(z) realisations: 3sDir for shot noise and sample variance ("the product of three independent Dirichlet distributions"), zero-point draws, and equal weights for three redshift samples.
  - WZ: "one can draw 3sDir n(z) samples and importance sample them by the value of their WZ likelihood", which is "an extremely inefficient process", so a modified likelihood is sampled jointly with WZ by HMC.
  - The n(z)s "are ordered with an algorithm called hyperrank".
  - Other systematics: zero-point and SOMPZ-method uncertainties are propagated by "shifting each probability integral transform by a value drawn from a Gaussian" (method σ = 0.003). Balrog uncertainty is "negligible", and shear ratios enter as "an additional data vector" in the cosmology chains.
  - Final prior widths: σ⟨z⟩ = 0.018, 0.015, 0.011, 0.017 in the four bins. The full ensemble was to be "made publicly available following publication".
  - The paper notes that the Dirichlet step alone "only reproduces the variance expected from Poisson noise".
  - (Myles+2021 was read in two passes; the final realisation count was not seen.)

  — [arXiv:2012.08566](https://arxiv.org/abs/2012.08566) (via [ar5iv](https://ar5iv.labs.arxiv.org/html/2012.08566)); [arXiv:1901.05005](https://arxiv.org/abs/1901.05005); [arXiv:2408.00922](https://arxiv.org/abs/2408.00922)
- **DES Y6:** SOM n(z) realisations are importance-sampled with a WZ likelihood from BOSS/eBOSS cross-correlations "to produce a posterior sample for each lens and source bin", which gives about a 10% S8 improvement. — [arXiv:2510.23565](https://arxiv.org/abs/2510.23565)
- **3sDir** (Sánchez, Raveri, Alarcon & Bernstein 2020; arXiv:2004.09542; doi:10.1093/mnras/staa2542):
  - Motivation: plain Dirichlet sampling of calibration counts "is only propagating shot noise", while sample variance can be "an order of magnitude larger than shot noise".
  - Key property: rescaling counts n_i→n_i/λ keeps the mean and multiplies the variance by λ.
  - Three steps:
    1. f_T ~ Dir(M_T/λ̄) over super-phenotypes;
    2. f_zT ~ Dir(M_zT/λ_T)·f_T;
    3. f_tz ~ Dir(M_tz)·f_z, with no inflation inside a redshift bin.
  - Limitations: "the Dirichlet distribution assumes no correlation between fluctuations in distinct z bins ... will hence not sample the large-scale structure faithfully on scales below Δz≈0.05", and one λ̄ is used over all z.
  - Data product: a released catalogue of resampled COSMOS2015 galaxies.

  — [arXiv:2004.09542](https://arxiv.org/abs/2004.09542) (full text read)
- **Public SOMPZ / 3sDir code** (all verified on GitHub):
  - `LSSTDESC/sompz`: "A DESC fork of the DES SOM-based photo-z code"; Python 2.7 conda environment; last push 2022-07-25; licence "Other".
  - `AndresaCampos/sompz_y6`: DES Y6 notebooks; MIT; last push 2024-08-05. It computes the transfer matrix "pcchat", pzc, pzchat and N(z).
  - `Roman-HLIS-Cosmology-PIT/Roman-SOMPZ`: "SOMPZ pipeline for Roman-HLIS"; GPL-3.0; created 2026-02; last push 2026-09-22; includes code that generates n(z) realisations.
  - `DhayaaAnbajagane/CosmicShearPhotoZ`: has `SOMPZ/ThreeSDirRunner.py`, apparently for DECADE.
  - DECADE cosmic shear itself (Anbajagane et al. 2025, arXiv:2502.17675) uses SOMPZ "including the contributions from sample variance, zeropoint calibration uncertainties, and redshift biases".

  — [LSSTDESC/sompz](https://github.com/LSSTDESC/sompz); [sompz_y6](https://github.com/AndresaCampos/sompz_y6); [Roman-SOMPZ](https://github.com/Roman-HLIS-Cosmology-PIT/Roman-SOMPZ); [CosmicShearPhotoZ](https://github.com/DhayaaAnbajagane/CosmicShearPhotoZ); [arXiv:2502.17675](https://arxiv.org/abs/2502.17675)
- **KiDS** (Wright et al. 2020, arXiv:1909.09632; van den Busch et al. 2022, arXiv:2204.02396; KiDS-Legacy: Wright et al. 2025, arXiv:2503.19440, A&A 703, A144):
  - Method: SOM direct calibration, "training one SOM per tomographic bin" (51×51 cells, 10 SOM realisations).
  - Selection: "gold weights" replace a binary gold class, and an analytic prior N(z) gives prior-volume weights.
  - Output: quoted biases are "the arithmetic means of the biases estimated per tomographic bin and pipeline setup" from simulations.
  - Photo-z use: photo-z define bin edges; in the part I could read, per-galaxy PDFs are not used in the N(z) estimate.
  - Software named: CosmoPipe (`AngusWright/CosmoPipe`, "Modular Pipeline Construction Environment for Cosmological Inference"; 7 stars; updated 2026-09-29), yet_another_wizz v2.6.0 and galselect.
  - Propagation to cosmology: "informative correlated Gaussian priors" on per-bin shifts δz (means {−2.6, 1.4, −0.2, 0.8, −1.1, −5.4}×10⁻²) with a conservative error floor σδz≈0.01. N(z) ensembles come from "different realisations of the calibration sample". CC redshifts are a consistency check (agreement except a ~2σ offset in bin four) and an alternative, "purely empirical, somewhat less constraining" prior set-up.
  - (Text read in two passes; a code-availability statement was not seen.)

  — [arXiv:2503.19440](https://arxiv.org/abs/2503.19440) (via [arXiv HTML](https://arxiv.org/html/2503.19440)); [AngusWright/CosmoPipe](https://github.com/AngusWright/CosmoPipe)
- **HSC Y3** (Rau et al. 2023, arXiv:2211.16516, doi:10.1093/mnras/stad1962):
  - Approach: a combined Bayesian hierarchical inference using photometry (DNNz; DEmP as the alternative) and LRG cross-correlations.
  - Prior: a logistic-GP / lognormal-counts model sampled with elliptical slice sampling.
  - Per-galaxy DNNz p(z): enter through an "empirical likelihood" whose root at λ=0 is the "stacked distribution", a name the authors call "conventional but misleading". There is no training-prior division.
  - Cosmic variance: taken from Sánchez+2020 COSMOS2015 measurements. It is moment-matched to lognormal parameters with a diagonal covariance ("ignoring the cross-correlations between neighboring redshift bins").
  - Output: the recommendation is "the maximum a posteriori solution for the psamp(z) and vary the mean using a Gaussian prior". No n(z) code release found.

  — [arXiv:2211.16516](https://arxiv.org/abs/2211.16516) (via [ar5iv](https://ar5iv.labs.arxiv.org/html/2211.16516))
- **Euclid, zPDF route** (Ilbert et al. 2021, Euclid preparation XI, arXiv:2101.02228; doi:10.1051/0004-6361/202040237):
  - Method: n(z) = Σ_i P_i(z) (stacked LePhare posteriors), "debiased" with the PIT histogram of a spec-z training sample. Each galaxy's PDF is reweighted as P_deb(z) = P(z)·N_P[C(z)]/∫N_P, with iterative prior updates, and photometric errors inflated ×2.
  - Plain stacking is "roughly ten and five times larger than the Euclid accuracy requirement" (DES/Euclid and LSST/Euclid depths).
  - After debiasing, the requirement is "largely satisfied" for LSST+Euclid photometry but not for DES depth.
  - Priors: "the choice of prior ... can have a significant impact on the recovered redshift distribution".
  - Missing populations: "a completely missing population ... could impact the results in an unknown manner".

  — [arXiv:2101.02228](https://arxiv.org/abs/2101.02228) (via [ar5iv](https://ar5iv.labs.arxiv.org/html/2101.02228))
- **Euclid, SOM route** (Roster et al. 2025, arXiv:2508.02779, A&A 707, A277):
  - Defining tomography by the mean spec-z per SOM cell leaves "none of the ten tomographic redshift bins satisfying the requirement".
  - Defining it by per-galaxy photo-z lets eight of ten bins meet |Δ⟨z⟩| ≤ 0.002(1+z). With a C3R2-like calibration sample, at most six do.
  - Kang et al. 2026 (arXiv:2601.02005, A&A 709, A55) add a deep-to-wide photometric transfer function to the SOM reconstruction. The key ingredient is reference objects "distributed in the colour space the same way as the wide-survey objects".

  — [arXiv:2508.02779](https://arxiv.org/abs/2508.02779); [arXiv:2601.02005](https://arxiv.org/abs/2601.02005)

#### D. Tools that turn n(z) ensembles into cosmology nuisance models
- **Hyperrank** (Cordero et al. 2022, arXiv:2109.09636, doi:10.1093/mnras/stac147):
  - Method: ranks discrete n(z) realisations by 1–4 summary values mapped onto continuous hyperparameters sampled in the cosmology chain. Mean-shifting was validated as sufficient for DES Y3.
  - Code: the CosmoSIS standard-library module `number_density/nz_multirank` (attribution "Juan P. Cordero", "Ian Harrison"; modes "mean, invchi, external"). Realisations are read from FITS extensions `NZ_{NAME}_realisation_{N}`, and the module is configured as `[hyperrank]` in `examples/des-y3.ini`.
  - Repository: cosmosis-standard-library, 21 stars, updated 2026-09-03.

  — [arXiv:2109.09636](https://arxiv.org/abs/2109.09636); [nz_multirank/module.yaml](https://github.com/cosmosis-developers/cosmosis-standard-library/blob/main/number_density/nz_multirank/module.yaml); [des-y3.ini](https://github.com/cosmosis-developers/cosmosis-standard-library/blob/main/examples/des-y3.ini)
- **LSSTDESC/nz_prior** ("A DESC Package to turn an ensemble of redshift distributions into a prior for photo-z uncertainty"):
  - Input: a qp ensemble or a sacc file with QPnz tracers.
  - Models: "Shifts", "ShiftsWidths", "PCA", "GP", "Comb" (evenly spaced Gaussians).
  - API: `X = nzp.PriorSacc(sacc_file, model=..., compute_crosscorrs="Full", zgrid=zgrid)`; `mean, cov, chol = X.get_prior()`; `X.sample_prior()`.
  - Status: install from source (`pip install .`); MIT; 2 stars; created 2024-06-21; last push 2026-05-07; not on PyPI.

  — [LSSTDESC/nz_prior](https://github.com/LSSTDESC/nz_prior)
- **Ruiz-Zapatero et al. 2026** (arXiv:2604.24425; doi:10.1093/mnras/stag1247): compares shifts, shifts & stretches, GP and PCA models built from simulated n(z) ensembles.
  - Engines: NUTS in Turing.jl with LimberJack.jl, plus Laplace/linearised analytic marginalisation, C̃ = C + J C_n Jᵀ.
  - Finding: "Stage-IV 3×2-pt analyses must go beyond simple shift & stretch models". A five-parameter PCA model "only degrades the constraint on the S8 parameter by 5 per cent" with "half the bias" in Ωm and σ8. Analytic marginalisation gives speed-ups "of up to a factor of 25".
  - Software cited: nz_prior and TJPCov.

  — [arXiv:2604.24425](https://arxiv.org/abs/2604.24425) (abstract via arXiv API; body via [arXiv HTML](https://arxiv.org/html/2604.24425))
- **Hadzhiyska et al. 2020** (arXiv:2007.14989; JCAP 2020(10)056):
  - Method: analytic marginalisation, C_M = C_c + T P Tᵀ, linearised around N̄.
  - Prior: combines analytic sample variance, Poisson noise, a photo-z-code systematic, and a smoothness prior C_P⁻¹ = Σ_n p_n vⁿᵀvⁿ with p_n = A_smooth·exp[−½(nΔz/Δz_thr)²] (A_smooth=1, Δz_thr=0.06), which gives "mild smoothing".
  - Limitations: Gaussianity ("may overestimate the variations"), linearity ("noticeably worse for auto-correlations"), and a "subjective" prior.

  — [arXiv:2007.14989](https://arxiv.org/abs/2007.14989) (via [ar5iv](https://ar5iv.labs.arxiv.org/html/2007.14989))
- **Ruiz-Zapatero et al. 2023** (arXiv:2301.11978): linearised analytic marginalisation over 160 p(z) parameters gives about a 30× speed-up. — [arXiv:2301.11978](https://arxiv.org/abs/2301.11978)
- **Stölzner et al. 2021** (arXiv:2012.07707; A&A 650, A148):
  - n(z) model: a "comb" of N_z=30 fixed, equidistant Gaussian components (σ=0.067, multiplied by z so that n(0)=0). Only the amplitudes are free, fitted as a_i = ln A_i for positivity.
  - Fitting and marginalisation: χ² fit to DIR histograms with spatial-bootstrap covariance; Gaussian analytic marginalisation over about 150 amplitudes; self-calibration by alternating optimisation.
  - Results: median shifts up to Δz≈0.02, degenerate with intrinsic-alignment amplitude.
  - Code: no release statement.

  — [arXiv:2012.07707](https://arxiv.org/abs/2012.07707) (via [ar5iv](https://ar5iv.labs.arxiv.org/html/2012.07707))
- **Zhang et al. 2023** (arXiv:2206.10169; MNRAS 518, 709):
  - Setup: the HSC n(z) prior is "modeled by 4 independent Dirichlet distributions".
  - Methods compared: shift, direct resampling (equal-weight chains, which "does not preserve Bayes' theorem"), and the proposed Bayesian resampling (250 n(z) draws, one MCMC per draw, chains weighted by MultiNest evidence). Hyperrank and comb models are discussed.
  - Findings: for HSC Y3 the shift model is adequate and about 100× cheaper.
  - For full-HSC power, the ar5iv text quotes about a 4% change in 1σ Ωm–S8 uncertainties (about 3% shift versus Bayesian resampling in the body). The current arXiv abstract says only that the choice "may significantly impact the statistical uncertainties on cosmological parameters, and a careful model selection is needed". Treat the exact percentage as version-dependent.

  — [arXiv:2206.10169](https://arxiv.org/abs/2206.10169) (via [ar5iv](https://ar5iv.labs.arxiv.org/html/2206.10169))
- **`gbernstein/nzmodes`**: a LaTeX note on sampling n(z) nuisance parameters that are known only as samples, by linear compression "ignoring directions that do not alter the likelihood of the data" and then density-estimating the compressed vector. — [gbernstein/nzmodes](https://github.com/gbernstein/nzmodes)
- **Euclid DR1 3×2pt requirement** (Bertmann et al. 2026, arXiv:2604.00805): the clustering-sample n(z) mean must be known to 0.004(1+z) to keep 80% of w0–wa constraining power. "The impact of the uncertainty on the width is negligible, provided the mean redshift is constrained with sufficient accuracy". — [arXiv:2604.00805](https://arxiv.org/abs/2604.00805)

#### E. Forward-modelling / population-model approaches (n(z) without per-galaxy PDFs)
- **Alsing et al. 2023** (arXiv:2207.05819): a forward model made of a population prior, SPS, a data/noise model and selection, used for hierarchical inference or SBI of tomographic n(z). Hierarchical sampling "requires computing the selection integral ... for every galaxy in the sample, in every likelihood evaluation ... the selection integral will require replacement with a fast emulator". — [arXiv:2207.05819](https://arxiv.org/abs/2207.05819)
- **Leistedt et al. 2023** (arXiv:2207.07673; ApJS, doi:10.3847/1538-4365/ac9d99): hierarchical SPS photo-z with neural emulators, "self-consistently inferring all model parameters, from high-level hyper-parameters to SPS parameters of individual galaxies". — [arXiv:2207.07673](https://arxiv.org/abs/2207.07673)
- **pop-cosmos:**
  - Alsing et al. 2024 (arXiv:2402.00935; ApJS 274, 12): a diffusion-model population fitted by optimal-transport distance to COSMOS photometry.
  - Thorp et al. 2024 (arXiv:2406.19437; ApJ 975, 145): per-galaxy posteriors with a GPU affine-invariant sampler, 15 GPU-s per galaxy.
  - Thorp et al. 2025 (arXiv:2506.12122; ApJ 993, 240): public software release.
  - Leistedt et al. 2026 (arXiv:2602.03935): forward-modelled KiDS-1000 n(z) "bypassing the need for spectroscopic reweighting". It finds Δz~0.05–0.1 differences in the first and last bins between population models.
  - Halder et al. 2026 (arXiv:2602.03930): 4 million KiDS galaxies at 8.2 GPU-s each.
  - Code: `Cosmo-Pop/pop-cosmos`; MIT; 14 stars; last push 2026-09-29. Depends on torch, torchdiffeq, and git dependencies `speculator@torch` and `flowfusion`. Installed with git clone + pip/poetry; not on PyPI.

  — [arXiv:2402.00935](https://arxiv.org/abs/2402.00935); [arXiv:2406.19437](https://arxiv.org/abs/2406.19437); [arXiv:2506.12122](https://arxiv.org/abs/2506.12122); [arXiv:2602.03935](https://arxiv.org/abs/2602.03935); [arXiv:2602.03930](https://arxiv.org/abs/2602.03930); [pop-cosmos pyproject](https://github.com/Cosmo-Pop/pop-cosmos/blob/main/pyproject.toml)
- **GalSBI** (Fischbacher et al. 2024, arXiv:2412.08701, JCAP 06(2025)007): an ABC/SBI population model whose output n(z) agrees with COSMOS within 1.5σ. Released as `galsbi` (PyPI 0.3.0, 2025-12-10, MIT). The 2026 extension (arXiv:2606.23783) adds clustering and therefore sample variance. — [arXiv:2412.08701](https://arxiv.org/abs/2412.08701); [arXiv:2606.23783](https://arxiv.org/abs/2606.23783); [PyPI galsbi](https://pypi.org/project/galsbi/)
- **Moser et al. 2024** (arXiv:2401.06846, JCAP 05(2024)049): SBI of deep fields for population models and n(z). **PopSED** (Li et al. 2023, arXiv:2309.16958, AJ): normalising-flow population posteriors fitted by Wasserstein distance; 10^5 GAMA galaxies in under 1 GPU-hour. Code `AstroJacobLi/popsed` (1 star; updated 2026-04-10). — [arXiv:2401.06846](https://arxiv.org/abs/2401.06846); [arXiv:2309.16958](https://arxiv.org/abs/2309.16958); [AstroJacobLi/popsed](https://github.com/AstroJacobLi/popsed)

#### F. Clustering-redshift codes (often combined with photometric n(z))
- **yet_another_wizz** (van den Busch et al. 2020): PyPI 3.1.2 (2025-05-07); GPL-3.0-or-later; 10 stars; docs at yet-another-wizz.readthedocs.io; wrapped by rail_yaw (PyPI 2.0.0, 2026-02-18). — [jlvdb/yet_another_wizz](https://github.com/jlvdb/yet_another_wizz); [PyPI](https://pypi.org/project/yet_another_wizz/)
- **the-wizz** (Morrison et al. 2017): 8 stars; updated 2023-11-13. — [morriscb/the-wizz](https://github.com/morriscb/the-wizz)
- **Tomographer** (Chiang et al. 2026, arXiv:2608.03415): an end-to-end clustering-z framework built on precomputed SDSS "activation maps". It returns the bias-weighted b(z)dN/dz for catalogues or intensity maps. — [arXiv:2608.03415](https://arxiv.org/abs/2608.03415)

### Inferences
- **Pastry's baseline model is the most-reimplemented object in this field.** The Dirichlet–categorical model with fixed per-object binned likelihoods appears in at least four places:
  - Leistedt+2016's binned-likelihood Gibbs sampler (Photoz-tools, BHM-nz);
  - frankenz's `hierarchical_sampler`;
  - HSC Y3 Appendix B;
  - RAIL's `VarInfStackSummarizer`.

  None of them is a maintained, documented, tested, general package that is honest about its uncertainties. Pastry's value is therefore in engineering and statistics hygiene rather than the core model.
- **The gap between tools is mostly about inputs and uncertainty:**
  - RAIL consumes qp ensembles but treats posteriors as likelihoods and reports mean-field-VB Dirichlet (under-dispersed) or bootstrap errors.
  - CHIPPR handles interim priors correctly but is a Python 2 prototype.
  - The survey pipelines get realistic errors (sample variance, zero-points, method bias) but are tied to SOM/deep-field set-ups rather than generic likelihood matrices.
- **The downstream interfaces are standardising:**
  - qp ensembles, used by RAIL, rail_sompz and nz_prior;
  - sacc QPnz tracers, used by nz_prior;
  - FITS files of n(z) realisations, used by CosmoSIS nz_multirank and hyperrank;
  - compressed priors (shift, stretch, PCA, GP, comb).

  Pastry should read and write qp ensembles and emit n(z) sample ensembles that these tools can consume directly.

### Gaps
- No public code found for:
  - Rau+2020 (LGP);
  - the HSC Y3 n(z) BHM;
  - Sánchez & Bernstein 2019 / Alarcon+2020;
  - Stölzner+2021;
  - Hadzhiyska+2020;
  - Zhang+2023 Bayesian resampling;
  - Leistedt+2023 hierarchical SPS (pop-cosmos is the successor code);
  - the Euclid OU-PHZ n(z) pipeline.

  "Not found" means not located by GitHub search or links in the papers I read. It does not prove the code is unavailable.
- BHM-nz is linked to Kyriacou+2026 by repository content and owner name only; I did not see the paper cite it.
- I could not resolve ADS 2022MNRAS.509.4886R to an arXiv ID directly (no ADS access). The mapping to arXiv:2101.01184 is inferred.
- For Myles+2021, KiDS-Legacy and parts of HSC Y3, only partial text was read: final realisation counts, KiDS Δz-prior correlation structure, and code statements may be in the unread sections.

## Q2. How do these handle interim priors, PDF representations, selection effects/unrepresentative calibration, smoothness/regularisation, and propagation of n(z) uncertainty to cosmology?

### Takeaway
- **Interim priors:** only CHIPPR divides the interim prior out explicitly, and CHIPPR shows that a wrong or unknown interim prior is fatal. Other tools either require true likelihoods (Leistedt+2016, frankenz, template codes such as BPZ with `no_prior`) or ignore the issue (RAIL VarInfStack and NaiveStack, HSC's DNNz "empirical likelihood", Euclid's stacked LePhare posteriors with PIT debiasing).
- **PDF representations:** standardised on qp's gridded, histogram, mixture and quantile parameterisations.
- **Selection and unrepresentative calibration:** handled mostly by weighting, SOM coverage cuts, clustering-z, or forward models with explicit selection.
- **Smoothness:** logistic-GP or Gaussian-on-log-height priors, finite-difference priors, Gaussian "comb" bases, or PCA/GP compressions of ensembles.
- **Propagation to cosmology:** either sample ensembles (hyperrank, Bayesian resampling, per-step draws) or compressed Gaussian nuisance priors (shift/stretch/PCA/GP, analytically marginalised).

### Cited Findings

#### Interim priors
- **Hogg, Myers & Bovy 2010** (arXiv:1008.4146) set out the general method: infer a population distribution from per-object "likelihood functions ... or samplings of the posterior probability distributions ... (under a given, uninformative prior)". They explicitly mention photometric redshifts. — [arXiv:1008.4146](https://arxiv.org/abs/1008.4146)
- **CHIPPR** divides each posterior by the interim prior p(z|φ*) and multiplies by the model n(z). It concludes that if the prior is implicit "the resulting posterior PDFs cannot be used for scientific inference", recommends that photo-z codes deliver likelihoods or an explicit prior, and calls a misspecified interim prior "disastrous". — [arXiv:2007.12178](https://arxiv.org/abs/2007.12178)
- **Malz 2021** (PRD 103, 083502; arXiv:2101.04675): stacking recovers N(z) only under "perfectly informative data and perfectly informative prior information". These conditions "will be strongly violated by future galaxy surveys", so "stacking must be abandoned". — [arXiv:2101.04675](https://arxiv.org/abs/2101.04675)
- **frankenz `hierarchical_sampler`:** inputs "must be *likelihoods*, since the prior is being explicitly modeled". — [frankenz samplers.py](https://github.com/joshspeagle/frankenz/blob/master/frankenz/samplers.py)
- **RAIL:**
  - VarInfStack's docstring targets "the likelihoods returned by template-based codes", but the code accepts any qp ensemble and applies no prior correction.
  - rail_bpz exposes a `no_prior` flag: "If desired, a Bayesian prior can be applied ... The product of this prior and the likelihoods is then summed over the SED types to return a marginalized posterior PDF".
  - I found no "interim" or prior-division logic anywhere in the RAIL sources read.

  — [var_inf.py](https://github.com/LSSTDESC/rail_base/blob/main/src/rail/estimation/algos/var_inf.py); [BPZ_lite.ipynb](https://github.com/LSSTDESC/rail/blob/main/interactive_examples/estimation_examples/BPZ_lite.ipynb)
- **HSC Y3:** per-galaxy DNNz conditional densities are averaged into an "empirical likelihood". The appendix does not divide out the training-set prior and argues that misspecification and cosmic variance dominate. — [arXiv:2211.16516](https://arxiv.org/abs/2211.16516)
- **Euclid:** the zPDF route stacks LePhare posteriors, notes "the choice of prior ... can have a significant impact", and corrects with the PIT of a spec-z training sample, P_deb(z) = P(z)·N_P[C(z)]/∫N_P. — [arXiv:2101.02228](https://arxiv.org/abs/2101.02228)

#### PDF representations
- **qp** (LSSTDESC; MIT; PyPI `qp-prob` 1.1.4, 2026-06-25; requires numpy≥2.0, scipy≥1.13, tables-io):
  - A `qp.Ensemble` holds many 1-D distributions of one parameterisation (metadata, objdata, optional ancil).
  - Main parameterisations: hist, interp, irregular interp, mixmod, quant. Others (packed_interp, sparse_interp, spline) are "in various stages of completeness". All scipy.stats continuous distributions are available via `qp.stats`, but cannot be converted to.
  - Known issues: conversion to mixmod "relies on sampling and then fitting ... does not provide consistent outputs"; quantile PDFs "can have negative values"; mixmod `rvs()` is "not currently functional".

  — [qp docs source](https://github.com/LSSTDESC/qp/tree/main/docs/user_guide); [qp](https://github.com/LSSTDESC/qp); [PyPI](https://pypi.org/project/qp-prob/)
- **Malz et al. 2018** (qp paper, arXiv:1806.00014, AJ 156, 35): "quantiles and samples outperform step functions"; the optimal format depends on the PDF set and the metric. — [arXiv:1806.00014](https://arxiv.org/abs/1806.00014)
- **Bin integrals versus point evaluations:**
  - Leistedt+2016 and HSC Appendix B integrate each likelihood over the histogram bins (binned likelihood; "lensing-weighted integral ... over histogram bin j").
  - RAIL VarInfStack and NaiveStack instead evaluate `pdf(zgrid)` at grid points.
  - frankenz takes an (Nobs, Nbins) array of PDFs.

  — [arXiv:1602.05960](https://arxiv.org/abs/1602.05960); [arXiv:2211.16516](https://arxiv.org/abs/2211.16516); [var_inf.py](https://github.com/LSSTDESC/rail_base/blob/main/src/rail/estimation/algos/var_inf.py); [frankenz samplers.py](https://github.com/joshspeagle/frankenz/blob/master/frankenz/samplers.py)

#### Selection effects and unrepresentative calibration samples
- **Leistedt+2016:** magnitude limits are captured as a dimension of the modelled distribution; other cuts, such as a second-band limit or tomographic selection, need a per-object selection variable s. — [arXiv:1602.05960](https://arxiv.org/abs/1602.05960)
- **Alsing+2023:** an explicit selection term whose integral is the computational bottleneck and needs an emulator. — [arXiv:2207.05819](https://arxiv.org/abs/2207.05819)
- **pop-cosmos KiDS:** a "data model for noise and selection, machine-learned from" SKiLLS. — [arXiv:2602.03935](https://arxiv.org/abs/2602.03935)
- **HSC Y3:** lensing weights are treated as a selection function inside the empirical likelihood. — [arXiv:2211.16516](https://arxiv.org/abs/2211.16516)
- **Spectroscopic incompleteness:** Hartley et al. 2020 (arXiv:2003.10454) find that colour cuts unavailable to the weak-lensing sample can bias the weighted mean redshift by Δz~0.015, and that confidence-flag selection biases DES Y1 at Δz = 0.005–0.05, "either sign". — [arXiv:2003.10454](https://arxiv.org/abs/2003.10454)
- **SOM coverage and weighting:** RAIL SOM drops "uncovered" cells; KiDS-Legacy uses gold weights and prior-volume weights; Euclid's transfer function makes reference objects match wide-survey colour space. — [somoclu_som.py](https://github.com/LSSTDESC/rail_som/blob/main/src/rail/estimation/algos/somoclu_som.py); [arXiv:2503.19440](https://arxiv.org/abs/2503.19440); [arXiv:2601.02005](https://arxiv.org/abs/2601.02005)
- **Clustering against prior bias:** Sánchez & Bernstein 2019 and Alarcon et al. 2020 combine clustering with photometry to beat a biased calibration prior (Δz̄≈0.05 reduced to ≲0.003). — [arXiv:1910.07127](https://arxiv.org/abs/1910.07127)
- **Training-set imperfection:** Crafford et al. 2026 (arXiv:2601.10797) test per-galaxy RAIL estimators on degraded training sets. They find KL divergence, Wasserstein distance and PIT the most informative metrics. — [arXiv:2601.10797](https://arxiv.org/abs/2601.10797)

#### Smoothness / regularisation
- **Prior-based approaches:**
  - Logistic-GP prior on log histogram heights: Rau+2020; HSC Y3 uses it with a diagonal cosmic-variance covariance.
  - CHIPPR: a squared-exponential covariance on log heights.
  - Hadzhiyska+2020: a finite-difference Gaussian smoothness prior.
  - Leistedt+2016: suggests a generalised Dirichlet for inter-bin correlations but does not implement it.

  — [arXiv:1904.09988](https://arxiv.org/abs/1904.09988); [arXiv:2211.16516](https://arxiv.org/abs/2211.16516); [arXiv:2007.14989](https://arxiv.org/abs/2007.14989); [arXiv:2007.12178](https://arxiv.org/abs/2007.12178); [arXiv:1602.05960](https://arxiv.org/abs/1602.05960)
- **Basis-function approaches:**
  - Stölzner's 30-component Gaussian comb with log amplitudes.
  - nz_prior "Comb"/"GP"/"PCA".
  - Ruiz-Zapatero+2026: PCA (5 per bin) and GP whose kernel is "the covariance of the calibration ensemble".

  — [arXiv:2012.07707](https://arxiv.org/abs/2012.07707); [LSSTDESC/nz_prior](https://github.com/LSSTDESC/nz_prior); [arXiv:2604.24425](https://arxiv.org/abs/2604.24425)

#### Propagation of n(z) uncertainty to cosmology
- **What each survey passes to cosmology:**
  - DES Y3: realisation ensembles, with extra systematics added by PIT shifting, sampled in the chains via hyperrank.
  - DES Y6: importance-sampled SOM realisations.
  - HSC Y3: MAP n(z) plus Gaussian mean-shift priors.
  - KiDS-Legacy: "informative correlated Gaussian priors" on per-bin δz (σ≈0.01 floor), calibrated on SKiLLS simulations, with a clustering-z alternative.

  — [arXiv:2012.08566](https://arxiv.org/abs/2012.08566); [arXiv:2510.23565](https://arxiv.org/abs/2510.23565); [arXiv:2211.16516](https://arxiv.org/abs/2211.16516); [arXiv:2503.19440](https://arxiv.org/abs/2503.19440)
- **Kyriacou+2026 / BHM-nz:** "drawing random samples of the heights at each step in the likelihood calculation", with no shift parameters. — [Harry45/BHM-nz](https://github.com/Harry45/BHM-nz)
- **Method comparisons:** Zhang+2023 Bayesian resampling versus shift versus direct resampling; Ruiz-Zapatero+2026 argues for going beyond shift & stretch to PCA. — [arXiv:2206.10169](https://arxiv.org/abs/2206.10169); [arXiv:2604.24425](https://arxiv.org/abs/2604.24425)

### Inferences
- **Interim priors (pastry design):**
  - Make the input type explicit: `likelihood` versus `posterior + interim_prior`.
  - Error (or warn loudly) when posteriors arrive without a prior.
  - Implement CHIPPR's reweighting L_ik ∝ p_ik/π_k, computed on bin integrals.
  - Provide a diagnostic for prior misspecification. CHIPPR's "disastrous" result shows the size of the risk.
- **PDF representations:**
  - Accept qp ensembles natively, converting them to bin-integrated likelihood matrices (not point evaluations). This makes pastry a drop-in, more principled replacement for RAIL's VarInfStack and NaiveStack.
  - Also accept posterior samples, mapped to counts or kernel-smoothed bin probabilities, with the interim prior divided out.
- **Overdispersion:** a single hierarchical Dirichlet captures only multinomial shot noise. Pastry should offer at least one of the following, with documented semantics:
  - a 3sDir-like concentration-rescaling knob (λ);
  - a Dirichlet-multinomial or logistic-normal overdispersion model;
  - a GP / covariance prior fed by theory or simulations, as in HSC Appendix C and Sánchez+2020.
- **Outputs for cosmology:** n(z) sample ensembles as qp/sacc/FITS-realisation files, plus optional compression to nz_prior-style shift/stretch/PCA/GP priors. This meets every downstream consumer found here.

### Gaps
- I found no paper that systematically quantifies the effect of ignoring the interim prior for modern ML photo-z codes (beyond CHIPPR's toy tests and Euclid's prior-sensitivity remark).
- I found no photo-z n(z) code that takes per-galaxy posterior samples directly. CHIPPR uses gridded posteriors, the hierarchical samplers use binned likelihood arrays, and qp can only convert samples into histograms (the hist parameterisation's `extract_hist_samples` conversion; [qp basic usage](https://github.com/LSSTDESC/qp/blob/main/docs/user_guide/basicusage.md)). Sample inputs with interim-prior reweighting (Hogg+2010-style importance sampling) would therefore be new in this domain.

## Q3. What pitfalls are documented?

### Takeaway
The documented pitfalls fall into four groups:
- **Stacking:** biased and too broad (Malz & Hogg; Malz 2021; Euclid zPDF 5–10× off requirement; Leistedt+2016; Kyriacou+2026).
- **Under-dispersed uncertainties:**
  - a plain Dirichlet propagates only shot noise, while sample variance can be 10× larger (3sDir);
  - mean-field VB under-covers by about 20% (HSC Y3 Appendix B), and Rau+2022 report "underestimated error bars" from the same Dirichlet VI;
  - bootstrap errors of stacks capture only galaxy resampling.
- **Calibration-sample incompleteness and selection biases:** Δz up to 0.05 (Hartley+2020), and Euclid's purity requirement above 99.8%.
- **Model choices:** binning (VarInfStack "wonky" with many bins; SOM-cell versus photo-z tomography), interim-prior misspecification, and nuisance models that are too rigid (shift-only).

Gibbs mixing problems are not quantitatively documented in the photo-z literature I found; the authors who report on convergence say it was fast.

### Cited Findings
- **Stacking bias:**
  - "the stacked estimator is always excessively broad" and its bias "worsens with decreasing signal-to-noise" (Malz & Hogg).
  - Stacking is valid only with "perfectly informative data" or "perfectly informative prior" (Malz 2021).
  - Leistedt+2016: stacked template likelihoods are "broad and multimodal ... stacking them does not yield a meaningful estimate"; the hierarchical model suppresses spurious secondary peaks.

  — [arXiv:2007.12178](https://arxiv.org/abs/2007.12178); [arXiv:2101.04675](https://arxiv.org/abs/2101.04675); [arXiv:1602.05960](https://arxiv.org/abs/1602.05960)
- **Stacking in practice:** Euclid zPDF stacking is biased "roughly ten and five times larger than the Euclid accuracy requirement" before PIT debiasing. Kyriacou+2026 find the BHM "performs better in almost every measure" than stacked likelihoods on simulations. — [arXiv:2101.02228](https://arxiv.org/abs/2101.02228); [arXiv:2605.15078](https://arxiv.org/abs/2605.15078)
- **Under-dispersed Dirichlet:**
  - The 3sDir paper: plain Dirichlet sampling of calibration counts propagates only shot noise; sample variance "can introduce fluctuations which are an order of magnitude larger"; even 3sDir ignores inter-bin correlations below Δz≈0.05.
  - DES Y3: 3sDir widths "within 10 per cent" of SOMPZ in three bins but "50 per cent wider" in the last.

  — [arXiv:2004.09542](https://arxiv.org/abs/2004.09542); [arXiv:2012.08566](https://arxiv.org/abs/2012.08566)
- **Mean-field VB under-coverage:** two independent statements, both about the same scheme as RAIL's VarInfStack.
  - HSC Y3 Appendix B: "a mean-field ansatz often lead[s] to the estimation of too narrow credibility intervals"; the under-coverage is "approximately 20%" with broad bins, from "numerical experience" rather than a documented test.
  - Rau et al. 2022: "the variational inference scheme yielded underestimated error bars, likely due to the restricted covariance structure of the Dirichlet that makes an independence assumption between neighboring bins".

  — [arXiv:2211.16516](https://arxiv.org/abs/2211.16516); [arXiv:2101.01184](https://arxiv.org/abs/2101.01184)
- **Binning versus calibration:** coarse-bin oversmoothing "can lead to posteriors that can be too narrow"; merging-bin regularisation produces "well-calibrated probability" (Rau et al. 2022). The DES Y3 Dirichlet step alone "only reproduces the variance expected from Poisson noise" (Myles et al. 2021). — [arXiv:2101.01184](https://arxiv.org/abs/2101.01184); [arXiv:2012.08566](https://arxiv.org/abs/2012.08566)
- **Stacking is still used in practice:** the 2025/2026 Rubin DP1 RAIL paper uses "stacked n(z) estimates, obtained by summing per-galaxy photometric redshift PDFs", without uncertainty bands. It notes that "The stacked PDFs of machine learning methods show the imprint of the training sample", an implicit-prior effect. — [arXiv:2510.07370](https://arxiv.org/abs/2510.07370)
- **Correlated bins:** Leistedt+2016 find adjacent bins strongly anticorrelated, "significantly enhanced ... due to the large support of the individual likelihoods". Appendix A shows the naive histogram estimator "fails in the regime of strong inter-bin correlations or quasi-empty bins", and that a Gaussian approximation to the Dirichlet is inadequate there. — [arXiv:1602.05960](https://arxiv.org/abs/1602.05960)
- **Binning:**
  - RAIL notebook: VarInfStack "tends to get a little wonky if you use too many bins".
  - HSC: under-coverage is quoted "under reasonable regularization (e.g. by selecting broader histogram bins)".
  - Euclid SOM: defining tomography by SOM-cell mean spec-z fails all ten bins, while per-galaxy photo-z tomography passes eight.

  — [Sampled_Summarizers.ipynb](https://github.com/LSSTDESC/rail/blob/main/interactive_examples/estimation_examples/Sampled_Summarizers.ipynb); [arXiv:2211.16516](https://arxiv.org/abs/2211.16516); [arXiv:2508.02779](https://arxiv.org/abs/2508.02779)
- **Calibration failures and outliers:**
  - Euclid direct calibration needs spectroscopic purity ">99.8%" (Ilbert+2021 abstract).
  - Hartley+2020: spectroscopic selection and redshift failures bias the mean redshift by Δz = 0.005–0.05.
  - Euclid zPDF debiasing: "a completely missing population ... could impact the results in an unknown manner".

  — [arXiv:2101.02228](https://arxiv.org/abs/2101.02228); [arXiv:2003.10454](https://arxiv.org/abs/2003.10454)
- **Interim-prior misspecification:** CHIPPR with a wrong interim prior gives "disastrous" results, with every estimator strongly biased. — [arXiv:2007.12178](https://arxiv.org/abs/2007.12178)
- **Nuisance-model rigidity:**
  - Zhang+2023: "the shift model is an inaccurate representation of the real uncertainty" at the n(z) level, and direct resampling is "overly-conservative".
  - Ruiz-Zapatero+2026: Stage-IV "must go beyond simple shift & stretch models".
  - Euclid 2026: the width is negligible if the mean is constrained to 0.004(1+z).

  — [arXiv:2206.10169](https://arxiv.org/abs/2206.10169); [arXiv:2604.24425](https://arxiv.org/abs/2604.24425); [arXiv:2604.00805](https://arxiv.org/abs/2604.00805)
- **Gaussian/linear approximations:** Hadzhiyska+2020 note that a Gaussian n(z) prior "may overestimate the variations", that a "Poisson or Dirichlet model may suit some calibration methods better", and that linearisation is "noticeably worse for auto-correlations". Stölzner+2021 find χ² = 4500 for 50 dof when fitting the DIR histograms, which they attribute to underestimated errors. — [arXiv:2007.14989](https://arxiv.org/abs/2007.14989); [arXiv:2012.07707](https://arxiv.org/abs/2012.07707)
- **Sampler cost and mixing:**
  - Leistedt+2016: "converged very quickly".
  - Sánchez & Bernstein 2019: R̂<1.03 in under an hour; they report no autocorrelations or effective sample sizes.
  - Kyriacou+2026: subsampling to 10^5 galaxies "for computational tractability".
  - DES Y3: importance-sampling 3sDir draws by the WZ likelihood is "extremely inefficient", so HMC is used.
  - Rau+2020: elliptical slice sampling because LGP heights are "quite correlated".

  — [arXiv:1602.05960](https://arxiv.org/abs/1602.05960); [arXiv:1807.11873](https://arxiv.org/abs/1807.11873); [arXiv:2605.15078](https://arxiv.org/abs/2605.15078); [arXiv:2012.08566](https://arxiv.org/abs/2012.08566); [arXiv:1904.09988](https://arxiv.org/abs/1904.09988)
- **Population-model systematics:** pop-cosmos versus shark gives Δz~0.05–0.1 in the outer KiDS bins, "comparable to the reported uncertainties from spectroscopic reweighting methods". — [arXiv:2602.03935](https://arxiv.org/abs/2602.03935)

### Inferences
- **Calibrated uncertainty is the main selling point.** Pastry should report coverage on simulated catalogues (simulation-based calibration) and state in its documentation that exact Gibbs/HMC is the reference while VB/EM are fast approximations. Both HSC (about 20% under-coverage) and Rau+2022 (underestimated error bars from the mean-field Dirichlet VI) document the failure of exactly the algorithm RAIL ships.
- **A full-covariance alternative already exists in the literature:** Rau+2022 replaced mean-field Dirichlet VI with a logit-normal posterior carrying full covariance, plus merging-bin regularisation. A logit-normal or full-rank Gaussian approximation on the simplex, or Laplace on the softmax parameterisation, is a cheap middle ground pastry could offer between mean-field VB and exact sampling.
- **Overdispersion beyond multinomial** (sample variance, calibration systematics) should be a first-class option. Otherwise users will get credible intervals as narrow as VarInfStack's.
- **Bin-count guidance:** the Dirichlet model does not care about bin order, so smoothness priors (GP, generalised Dirichlet) matter most at large K, where VarInfStack "gets wonky".

### Gaps
- I found no published study quantifying Gibbs-sampler mixing (autocorrelation versus N, K, or likelihood width) for photo-z n(z) samplers. Mixing problems for mixture-model Gibbs samplers are a plausible concern but undocumented here; this is my inference, not a cited finding.
- No published coverage test of RAIL VarInfStack was found. The RAIL tests are smoke tests only (see Q4).
- How catastrophic photo-z outliers (multimodal likelihoods) affect hierarchical versus stacked n(z) is shown only qualitatively (Leistedt+2016 secondary peaks).

## Q4. Which tools could a non-specialist install and use today (API, docs, maintenance)?

### Takeaway
- **Pip-installable and maintained in 2026:**
  - RAIL (pz-rail-base 2.0.11, Python ≥3.11, MIT, frequent releases; docs at rail-hub.readthedocs.io), which provides NaiveStack, PointEstHist, VarInfStack and SOM/NZDir/SOMPZ through a stage/pipeline API with "interactive" wrappers;
  - qp (qp-prob 1.1.4);
  - yet_another_wizz;
  - galsbi.
- **Usable from source but research-grade:** nz_prior, pop-cosmos (needs PyTorch/GPU), CosmoSIS hyperrank (needs CosmoSIS), and the SOMPZ notebook repos.
- **Stale:** the hierarchical PDF→n(z) codes CHIPPR (last push 2022, Python 2.7 demo), Photoz-tools (2020), frankenz (2018) and BHM-nz (2023).

No current package offers an emcee/scikit-learn-style `fit(L) → samples` API for the hierarchical model with honest uncertainties.

### Cited Findings
- **PyPI status** (JSON API, 2026-10-22):

  | Package | Version | Uploaded | Notes |
  |---|---|---|---|
  | pz-rail | 2.0.0 | 2026-02-27 | umbrella; `algos` extra pulls about 17 algorithm packages |
  | pz-rail-base | 2.0.11 | 2026-09-09 | 65 releases since 2023-06-13; requires Python ≥3.11, numpy, scipy≥1.9, ceci≥2.1, qp-prob≥1.0, pyyaml, structlog, click |
  | pz-rail-som | 2.0.1 | 2026-03-19 | needs minisom, somoclu, scikit-learn, pandas |
  | pz-rail-sompz | 0.0.4 | 2026-03-03 | needs numba |
  | pz-rail-sklearn | 2.0.3 | 2026-06-25 | |
  | pz-rail-yaw | 2.0.0 | 2026-02-18 | |
  | qp-prob | 1.1.4 | 2026-06-25 | |
  | yet_another_wizz | 3.1.2 | 2025-05-07 | GPL-3.0-or-later |
  | pzflow | 4.0.0 | 2026-05-06 | |
  | galsbi | 0.3.0 | 2025-12-10 | |
  | frankenz | 0.1.5 | 2018-04-30 | |

  All RAIL packages are MIT and "Development Status :: 4 - Beta". Not on PyPI under the queried names: chippr, pop-cosmos, sompz, nz_prior, hyperrank, popsed, txpipe, firecrown.

  — [PyPI pz-rail-base](https://pypi.org/project/pz-rail-base/); [PyPI qp-prob](https://pypi.org/project/qp-prob/); [PyPI frankenz](https://pypi.org/project/frankenz/); [PyPI yet_another_wizz](https://pypi.org/project/yet_another_wizz/)
- **conda-forge** (anaconda.org API, 2026-10-22): qp-prob 1.1.1 (2026-06-18), pz-rail-base 2.0.9 (2026-06-24), pzflow 4.0.0, somoclu 1.7.6, minisom 2.3.6, sacc 2.4, cosmosis 3.26 (2026-10-02). Not found: pz-rail umbrella, yet_another_wizz, frankenz. — [anaconda.org conda-forge/pz-rail-base](https://anaconda.org/conda-forge/pz-rail-base); [conda-forge/qp-prob](https://anaconda.org/conda-forge/qp-prob)
- **GitHub activity:**

  | Repository | Stars | Activity date | Notes |
  |---|---|---|---|
  | LSSTDESC/rail | 21 | updated 2026-10-13 | |
  | rail_base | 1 | updated 2026-09-09 | 11 open issues |
  | rail_sompz | 1 | pushed 2026-10-19 | |
  | TXPipe | 23 | updated 2026-10-12 | |
  | firecrown | 32 | updated 2026-10-16 | |
  | Cosmo-Pop/pop-cosmos | 14 | pushed 2026-09-29 | |
  | LSSTDESC/nz_prior | 2 | pushed 2026-05-07 | |
  | aimalz/chippr | 2 | pushed 2022-11-22 | |
  | ixkael/Photoz-tools | 3 | pushed 2020-04-14 | |
  | joshspeagle/frankenz | 18 | pushed 2018-10-01 | |
  | Harry45/BHM-nz | 1 | pushed 2023-08-27 | |
  | jlvdb/yet_another_wizz | 10 | updated 2026-09-16 | |
  | AngusWright/CosmoPipe | 7 | updated 2026-09-29 | |
  | cosmosis-standard-library | 21 | updated 2026-09-03 | |

  — [LSSTDESC org](https://github.com/LSSTDESC); [aimalz/chippr](https://github.com/aimalz/chippr); [joshspeagle/frankenz](https://github.com/joshspeagle/frankenz); [Cosmo-Pop/pop-cosmos](https://github.com/Cosmo-Pop/pop-cosmos)
- **Documentation:**
  - Working: https://rail-hub.readthedocs.io/en/latest/ (HTTP 200), qp.readthedocs.io (200), yet-another-wizz.readthedocs.io (200), chippr.readthedocs.io/en/master (200, stale).
  - Dead: lsstdescrail.readthedocs.io, the older RAIL doc host still linked from the rail_sompz README, returned 404.

  — [RAIL docs](https://rail-hub.readthedocs.io/en/latest/); [qp docs](https://qp.readthedocs.io/); [rail_sompz README](https://github.com/LSSTDESC/rail_sompz)
- **RAIL API style:** stages are built with `make_stage(**config)` and called with `.summarize(qp_ensemble)`, or through `rail.interactive` functions that return dicts of outputs (`"output"`: n(z) samples; `"single_NZ"`: fiducial). Adding a summarizer means subclassing `PZSummarizer` and implementing `run()`. — [RAIL add_stage docs](https://github.com/LSSTDESC/rail/blob/main/docs/source/development/add_stage.rst); [summarizer.py](https://github.com/LSSTDESC/rail_base/blob/main/src/rail/estimation/summarizer.py)
- **RAIL test quality:** `tests/estimation/test_summarizers.py` runs each summarizer end-to-end on a BPZ output file ("Basic end to end test") and checks that outputs are written. There are no statistical accuracy or coverage assertions. — [rail_base tests](https://github.com/LSSTDESC/rail_base/blob/main/tests/estimation/test_summarizers.py)
- **Minimal usage snippets:**
  - **RAIL VarInfStack:** see Q1. Note the example notebook passes `niter` although the config key is `n_iter`, so the iteration count it intends is not set.
  - **qp:** `ens = qp.interp.create_ensemble(xvals, yvals)`; `ens.pdf(x)`; `ens.write_to("f.hdf5")`. — [qp quickstart](https://github.com/LSSTDESC/qp/blob/main/docs/user_guide/quickstart.md)
  - **frankenz:** `population_sampler(pdfs).run_mcmc(Niter, logprior_nz=...)` or `hierarchical_sampler(pdfs).run_mcmc(Niter, alpha=..., ref_sample=...)`, with `pdfs` an (Nobs, Nbins) array. — [samplers.py](https://github.com/joshspeagle/frankenz/blob/master/frankenz/samplers.py)
  - **nz_prior:** `X = nzp.PriorSacc(sacc_file, model="PCA", compute_crosscorrs="Full", zgrid=zgrid)`; `X.get_prior()`. — [nz_prior](https://github.com/LSSTDESC/nz_prior)
- **Hardware and install burden:**
  - pop-cosmos needs torch, torchdiffeq and git-pinned speculator/flowfusion.
  - rail_shire (a JAX SPS template-fitting estimator) warns that it "is likely not the best suited code to be run on a personal laptop".
  - The DES SOMPZ repos assume NERSC conda environments (Python 2.7 for LSSTDESC/sompz).

  — [pop-cosmos pyproject](https://github.com/Cosmo-Pop/pop-cosmos/blob/main/pyproject.toml); [rail_shire](https://github.com/LSSTDESC/rail_shire); [LSSTDESC/sompz](https://github.com/LSSTDESC/sompz)
- **LSSTDESC/rail_calib** ("Tools to calibration redshift estimation", created 2025-04-29) still shows the template "TODO List" README, so no functionality is documented. — [rail_calib](https://github.com/LSSTDESC/rail_calib)

### Inferences
Usability ranking for a non-specialist wanting "PDFs → n(z) with uncertainties" (synthesised from the findings above):

| Tool | Install | Statistical model | Uncertainty honesty | Docs/tests | Status |
|---|---|---|---|---|---|
| RAIL NaiveStack / PointEstHist | `pip install pz-rail-base` | stacking / histogram | bootstrap over galaxies only; biased | RTD docs + example notebooks; smoke tests | maintained |
| RAIL VarInfStack | same | Dirichlet–categorical, mean-field VB, PDFs used as likelihoods | under-covers (~20% per HSC; "underestimated error bars" per Rau+2022); no interim prior; grid point evaluations | minimal docs, smoke test | maintained, rarely changed |
| RAIL SOM / NZDir / SOMPZ | pip extras | colour-cell reweighting of calibration samples | bootstrap of spec-z (shot noise only); SOMPZ realisations via 3sDir code elsewhere | notebooks | maintained |
| frankenz samplers | PyPI 0.1.5 / source | exact MH-in-Gibbs (population) and Gibbs (hierarchical, Dirichlet, reference counts) | exact posterior for the model; no overdispersion | no docs | stale since 2018 |
| Photoz-tools / BHM-nz | clone | Leistedt+2016 Gibbs (binned likelihoods, MPI) | exact for model | notebooks | stale (2020 / 2023) |
| chippr | clone (Python 2) | interim-prior-corrected log-histogram + GP hyperprior, emcee | exact for model; handles interim prior | stale RTD | stale (2022) |
| nz_prior | clone + `pip install .` | ensemble → shift/stretch/PCA/GP/comb Gaussian priors | depends on input ensemble | README + examples | active (2026) |
| CosmoSIS nz_multirank | CosmoSIS | hyperrank over realisations | depends on ensemble | module.yaml | maintained |
| pop-cosmos | clone + torch/GPU | forward model population prior | model-dependent | demos | active (2026) |

Per-tool feature matrix, in pastry's terms. It is synthesised from the Q1–Q4 findings; "–" means not supported or not documented.

| Tool (lang, licence) | Input | Model | Engine | Uncertainty output | Per-object heteroscedastic L | Interim-prior correction | Selection effects | Smoothness prior | Scale notes |
|---|---|---|---|---|---|---|---|---|---|
| RAIL NaiveStack (Py, MIT) | qp ensemble of PDFs | sum of PDFs | closed form | Poisson-bootstrap samples | yes (PDFs) | – | tomographic masks only | – | chunked, MPI |
| RAIL VarInfStack (Py, MIT) | qp ensemble of PDFs, evaluated at grid points | Dir(1)–categorical | mean-field CAVI (`n_iter`=100) | Dir(α) draws (default 500) | yes | – (docstring assumes likelihoods) | – | – | all data in memory; "wonky" at many bins |
| RAIL SOM / NZDir / SOMPZ (Py, MIT) | photometry + spec-z / deep / Balrog catalogues | cell or kNN reweighting; SOMPZ transfer matrix | closed form | bootstrap of spec-z; qp ensemble | n/a | n/a | weights; uncovered cells removed | – | MPI chunking |
| Leistedt+2016 / Photoz-tools `bayeshist.py` (Py, MIT) | binned likelihoods (type × z × mag) | histogram + Dirichlet; per-object labels | Gibbs (MPI) | posterior samples of f and per-object z | yes | n/a (template likelihoods) | via selection variable s | generalised Dirichlet suggested, not implemented | 10^4 samples in about 10 min (toy) |
| BHM-nz / Kyriacou+2026 (Py notebooks, MIT) | template likelihoods | as Leistedt+2016 | Gibbs | n(z) height samples drawn per cosmology step | yes | n/a | tomography via BPZ bins | PCA template reduction | subsets of 10^5 for tractability |
| chippr (Py2, MIT) | gridded posteriors + interim prior | log-histogram with GP-like hyperprior | emcee (100 walkers); MMLE | posterior samples | yes | **yes** (divides by p(z\|φ*)) | – | yes (squared-exponential covariance) | tested at 10^4 galaxies; "computational expense" |
| frankenz samplers (Py, MIT) | (N, K) PDF/likelihood array | population: Π_i Σ_k pdf_ik·n_k; hierarchical: Dirichlet + labels + reference counts | MH-in-Gibbs pair moves; Gibbs | samples + ln posterior | yes | – (requires likelihoods) | – | any user log-prior (population sampler) | pure NumPy loops |
| Rau+2020 LGP (no code) | clustering data (+ photo-z prior) | logistic GP on log heights | MH-within-Gibbs + elliptical slice | samples | – | – | – | **yes** (LGP) | – |
| HSC Y3 BHM (no code) | DNNz p(z) + LRG cross-correlations | empirical likelihood + LGP / lognormal cosmic variance | elliptical slice | samples, then MAP + shift prior | yes (weighted) | – | lensing weights as selection | via LGP covariance (diagonal CV) | – |
| 3sDir (DES / Roman / DECADE code) | SOM cell counts of calibration fields | nested Dirichlets with λ inflation | direct sampling | n(z) realisations | n/a | n/a | via Balrog transfer | – (no inter-bin correlation below Δz≈0.05) | cheap |
| hyperrank (CosmoSIS) | FITS n(z) realisations | ranking → continuous hyperparameters | inside cosmology MCMC | marginalised | n/a | n/a | n/a | n/a | needs CosmoSIS |
| nz_prior (Py, MIT) | qp ensemble or sacc QPnz | shift / shift+width / PCA / GP / comb | moment-matched Gaussian priors | mean, cov, samples | n/a | n/a | n/a | GP / comb | – |
| pop-cosmos (Py+torch, MIT) | photometry | diffusion population model + SPS emulator + noise/selection | OT fitting; GPU ensemble MCMC | n(z) by forward simulation | n/a | n/a | **yes** (learned data/selection model) | implicit | GPU; 8–15 GPU-s per galaxy for posteriors |

Design lessons for pastry:
- Target an estimator object (`pastry.fit(L or qp.Ensemble, prior=..., interim_prior=...) → samples/summary`) with:
  - a qp-compatible I/O adapter;
  - a RAIL `PZSummarizer` wrapper, so DESC users get it inside RAIL;
  - exporters to sacc/qp/CosmoSIS realisation FITS and to nz_prior-style compressed priors.
- Ship a small, fast exact sampler (collapsed Gibbs or HMC on the simplex) as the default, with VB/EM as clearly labelled approximations.
- Ship coverage tests (simulation-based calibration) in CI, which RAIL lacks.

### Gaps
- I could not list exact latest-commit dates for every repository: per-repository commit APIs were blocked, so `pushed_at` from search metadata was used, and `updated_at` where `pushed_at` was not retrieved (rail, rail_base, TXPipe, firecrown, yet_another_wizz, CosmoPipe, CSL).
- I did not install or execute any package (resource constraints and policy), so installability is inferred from metadata and dependency lists.
- TXPipe and firecrown n(z) handling (e.g. firecrown shift/stretch systematics) was not examined in detail.
