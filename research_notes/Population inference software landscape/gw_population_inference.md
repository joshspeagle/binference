# GW population-inference software and methods (census for pastry, as of Oct 2026)

STATUS: COMPLETE (all five key questions filled; last updated 2026-10-22).

Verification sources: PyPI JSON metadata (2026-10-22); shallow git clones of each repository (latest-commit dates come from these, and the code was read but not executed); the git.ligo.org GitLab API; GitHub repository search (stars); arXiv API metadata; arXiv HTML full text of GWTC-4.0 (2508.18083), GWTC-5.0 (2605.27226), Talbot & Golomb (2304.06138) and Heinzel & Vitale (2509.07221). Downloaded material is in the scratchpad gw/ folder.

## Q1. Formalism and its implementations (gwpopulation, bilby.hyper, GWInferno, icarogw, FIGARO, popsummary, LVK tooling, newer packages)

### Takeaway
The importance-reweighted hierarchical likelihood, ln L(Λ) ≈ Σ_i ln[(1/S_i) Σ_s p(θ_is|Λ)/π_PE(θ_is)] − N ln P̂_det(Λ) with P̂_det(Λ) = (1/N_draw) Σ_{found j} p(θ_j|Λ)/p_draw(θ_j) (or its Poisson-rate form, N ln R − R·VT(Λ)), is implemented in essentially every GW package with nearly identical code. gwpopulation (MIT, JOSS 2025, numpy/cupy/JAX backends) is the de facto LVK reference implementation, and GWInferno, PixelPop, gppop, gwkokab and icarogw re-implement the same estimator in NumPyro, Pyro or CuPy. All of them are GW-shaped: inputs are posterior-sample tables with a `prior` column plus an injection set with `prior`, `total_generated` and `analysis_time`. Only bilby.hyper, gwpopulation's core likelihood, FIGARO and icarogw are reasonably usable outside GW astronomy, and even these carry GW assumptions (cosmology, redshift and hypervolume) in their selection classes.

### Cited Findings
**The formalism (key papers)**
- Hogg, Myers & Bovy (2010) gave the importance-sampling hierarchical estimator. It uses per-object posterior samples drawn under an "interim prior", divides that prior out, and averages the population density over the samples. They applied it to exoplanet eccentricities (arXiv:1008.4146, ApJ 725, 2166). The abstract says the method takes "the likelihood functions for the individual-star eccentricities, or samplings of the posterior probability distributions … (under a given, uninformative prior)". It "can also be seen as a kind of heteroscedastic deconvolution" and "can be applied to any quantity measured with finite precision … including magnitudes, parallaxes, or photometric redshifts". This is exactly pastry's problem statement. — [arXiv:1008.4146](https://arxiv.org/abs/1008.4146)
- Loredo (2004) is an earlier treatment of source uncertainties and selection in survey population analyses. — [arXiv:astro-ph/0409387](https://arxiv.org/abs/astro-ph/0409387)
- Foreman-Mackey, Hogg & Morton (2014) applied the same importance-sampling estimator, with a selection (completeness) term and a Gaussian-process smoothness prior on binned occurrence rates, to Kepler planets. The abstract: "taking into account survey completeness and, for the first time, observational uncertainties … we only require that the occurrence rate density be a smooth function of period and radius (employing a Gaussian process)". This is a direct non-GW precedent for pastry's "binned rates + GP prior + selection" model. — [arXiv:1406.3020](https://arxiv.org/abs/1406.3020)
- Mandel, Farr & Gair (2019) derived the selection-corrected likelihood p({d}|Λ) ∝ Π_i ∫ p(d_i|θ) p(θ|Λ) dθ / α(Λ)^N, where α(Λ) = ∫ p_det(θ) p(θ|Λ) dθ, and the Poisson-rate version. — [arXiv:1809.02063](https://arxiv.org/abs/1809.02063)
- Thrane & Talbot (2019) is the GW tutorial that states the hyper-likelihood used in gwpopulation and bilby (its Eq. 34 is cited in both codes' docstrings). — [arXiv:1809.02293](https://arxiv.org/abs/1809.02293); [gwpopulation hyperpe.py](https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/hyperpe.py)
- Vitale, Gerosa, Farr & Taylor (2020) is a review chapter on population inference with selection effects. — [arXiv:2007.05579](https://arxiv.org/abs/2007.05579)
- The GWTC-4.0 population paper (LVK, arXiv:2508.18083) splits its methods into a "Strongly Modeled Approach" and a "Weakly Modeled Approach" and has a dedicated appendix, "Hierarchical Inference Details: Likelihood Estimation; Sampling Techniques". The GWTC-5.0 population paper (arXiv:2605.27226) uses PixelPop, FullPop, a truncated Gaussian mixture model, a default parametric model, Madau–Dickinson redshift and bivariate skewed χ_eff/χ_p models as its "overview models". (Section headings were read from the arXiv HTML.) — [arXiv:2508.18083](https://arxiv.org/abs/2508.18083); [arXiv:2605.27226](https://arxiv.org/abs/2605.27226)

**gwpopulation** (https://github.com/ColmTalbot/gwpopulation; docs https://ColmTalbot.github.io/gwpopulation; PyPI `gwpopulation`; also on conda-forge)
- Python, MIT licence. Latest PyPI release 1.3.1 (2025-12-15); first release 0.2.0 (2019-03); 24 releases. Latest commit 2026-07-17 ("FEAT: allow use of different numbers of samples per event (#142)"). 62 stars and 61 forks on GitHub. Requires bilby>=2.7, cached_interpolate and wcosmo; Python >=3.10. — [PyPI](https://pypi.org/project/gwpopulation/); [GitHub](https://github.com/ColmTalbot/gwpopulation)
- Paper: Talbot, Farah, Galaudage, Golomb & Tong (2025), "GWPopulation: Hardware agnostic population inference for compact binaries and beyond", JOSS 10(109):7753, doi:10.21105/joss.07753 (arXiv:2409.14143). — [JOSS](https://doi.org/10.21105/joss.07753); [arXiv:2409.14143](https://arxiv.org/abs/2409.14143)
- Backend switching: `gwpop.set_backend("numpy"|"cupy"|"jax")`, with JAX x64 enabled automatically. — [backend.py](https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/backend.py)
- Inputs to `HyperparameterLikelihood(posteriors, hyper_prior, selection_function, conversion_function, max_samples, maximum_uncertainty, require_equal_samples)`:
  - a list of pandas DataFrames of per-event posterior samples, with an optional `prior` column that holds the PE (interim) prior density. This prior is divided out (interim-prior correction);
  - a population model, given either as a function `f(dataset, **hyperparams)` or as a `bilby.hyper.model.Model` of several such functions.

  The log-likelihood is Σ ln mean(w) − N ln P̂_det, with w = p(θ|Λ)/π_PE. `RateLikelihood` implements the Poisson (rate) form N ln R − N_exp. — [hyperpe.py](https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/hyperpe.py)
- Monte Carlo variance: each event's ln-likelihood variance is (E[w²] − E[w]²)/(S·E[w]²), and the selection term contributes N²·Var(P̂_det)/P̂_det². `maximum_uncertainty` returns −inf when the total variance exceeds the threshold squared. `generate_extra_statistics` stores per-event ln BF, per-event variances, the selection term and the total variance for each posterior sample. — [hyperpe.py](https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/hyperpe.py)
- Selection uses one of two classes:
  - `ResamplingVT(model, data=injections, n_events)` takes found injections with a `prior` (draw density) plus `total_generated` and `analysis_time`. It returns (P̂_det, variance) and by default enforces N_eff = μ²/σ² > 4·N_events, returning inf otherwise. Its optional `marginalize_uncertainty` applies Farr's (2019) correction, P̂_det·exp(−(3+N)/(2N_eff)), but the docstring says "We recommend not to use this as it is not completely understood if this uncertainty marginalization is correct".
  - `GridVT` integrates a gridded p_det. Its docstring warns that cost "scales exponentially with the number of parameters".

  — [vt.py](https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/vt.py)
- Usage, from the shipped GWTC-3 example notebook:
  ```python
  gwpop.set_backend("jax")
  model = Model([gwpop.models.mass.two_component_primary_mass_ratio, gwpop.models.spin.iid_spin,
                 gwpop.models.redshift.PowerLawRedshift(cosmo_model="Planck15")])
  vt = gwpop.vt.ResamplingVT(model=model, data=injections, n_events=len(posteriors))
  likelihood = gwpop.hyperpe.HyperparameterLikelihood(posteriors=posteriors, hyper_prior=model, selection_function=vt)
  priors = PriorDict(); priors["alpha"] = Uniform(-2, 4); ...
  result = bilby.run_sampler(likelihood=JittedLikelihood(likelihood), priors=priors, sampler="dynesty", nlive=100)
  ```
  — [examples/GWTC3.ipynb](https://github.com/ColmTalbot/gwpopulation/blob/main/examples/GWTC3.ipynb)
- Other features:
  - `posterior_predictive_resample` produces population-informed per-event posteriors.
  - `experimental.numpyro.gwpopulation_likelihood_model` wraps the same likelihood as a NumPyro factor for NUTS. It records variances as deterministics, draws the rate from a Gamma, and can apply `varmax`.
  - `InterpolatedNoBaseModelIdentical` is a cubic-spline-node nonparametric model with an optional RMS regularisation.
  - There are 12 test modules plus codecov CI.

  — [experimental/numpyro.py](https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/experimental/numpyro.py); [models/interped.py](https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/models/interped.py); [test/](https://github.com/ColmTalbot/gwpopulation/tree/main/test)
- The model-definition contract (from the models docstring): a function takes a `dataset` dict of arrays plus named hyperparameters and returns a density. Alternatively, a class with `variable_names` can cache expensive, hyperparameter-independent quantities. — [models/__init__.py](https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/models/__init__.py)

**bilby.hyper** (bilby 2.8.2, 2026-08-10, MIT; https://github.com/bilby-dev/bilby, 129 stars; docs https://bilby.readthedocs.io)
- `bilby.hyper.likelihood.HyperparameterLikelihood(posteriors, hyper_prior, sampling_prior=None, log_evidences=None, max_samples)` takes a list of DataFrames with a `prior` or `log_prior` column, or a sampling-prior function.
- It downsamples every event to the same sample count. Its log-likelihood is Σ ln Σ_s (p/π) − N ln S.
- It has no selection term, no Monte Carlo variance and is numpy-only. It is generic, not GW-specific, but minimal.

— [bilby 2.8.2 wheel, bilby/hyper/likelihood.py](https://pypi.org/project/bilby/2.8.2/); [GitHub](https://github.com/bilby-dev/bilby)

**gwpopulation_pipe** (https://git.ligo.org/RatesAndPopulations/gwpopulation_pipe; docs https://docs.ligo.org/RatesAndPopulations/gwpopulation_pipe; PyPI `gwpopulation-pipe` 0.4.3, 2025-04-23; MIT)
- This is the HTCondor/bilby_pipe pipeline around gwpopulation. Its modules cover data collection, injection creation, the VT helper, analysis, post-plots and popsummary output (`common_format.py`).
- It depends on bilby_pipe, htcondor, popsummary and wcosmo.
- Latest commit 2026-07-23; the repo was created in 2019; 4 GitLab stars. Several personal forks are active in 2025–26.

— [PyPI](https://pypi.org/project/gwpopulation-pipe/); [GitLab](https://git.ligo.org/RatesAndPopulations/gwpopulation_pipe)

**GWInferno** (https://github.com/FarrOutLab/GWInferno; docs https://farroutlab.github.io/GWInferno/; MIT)
- Repo facts: Python on NumPyro/JAX. Version 0.0.2 in setup.py and not on PyPI (the `gwinferno` name returns 404), so it is installed from source. 21 stars and 29 open issues plus PRs (GitHub's open_issues_count). Latest commit 2026-06-22.
- `hierarchical_likelihood()` is a NumPyro factor. It computes:
  - per-event log Bayes factors by log-sum-exp importance sampling;
  - per-event n_eff = (Σw)²/Σw²;
  - per-event variance 1/n_eff − 1/S;
  - the injection detection efficiency and its n_eff.
- Its options are:
  - `min_neff_cut` (the default): injections need N_eff ≥ 4·N_obs and each event needs n_eff > N_obs;
  - `max_variance_cut`: total ln-L variance < 1;
  - `marginalize_selection`: Farr's (2019) −(3+N)/(2N_eff) correction.
- Interfaces: a YAML config plus `gwinferno_run_from_config.py` (NUTS kernel arguments set in YAML), or a user-written NumPyro model file.
- B-spline models use precomputed design matrices at the PE samples and injections. The P-spline smoothing prior is coefficients ~ Normal(0, σ) plus a factor −½·τ·||Δ^d c||², via `apply_difference_prior`.

— [GitHub](https://github.com/FarrOutLab/GWInferno); [pipeline/analysis.py](https://github.com/FarrOutLab/GWInferno/blob/main/gwinferno/pipeline/analysis.py); [models/bsplines/smoothing.py](https://github.com/FarrOutLab/GWInferno/blob/main/gwinferno/models/bsplines/smoothing.py); [docs/example_usage.rst](https://github.com/FarrOutLab/GWInferno/blob/main/docs/example_usage.rst)

**icarogw** (https://github.com/simone-mastrogiovanni/icarogw; licence EUPL-1.2)
- Repo facts: version 2.0.3 in setup.py and not on PyPI (404). The README says to install with `pip install git+https://github.com/icarogw-developers/icarogw.git`, but I could not resolve that organisation's repo. 17 stars and 19 forks; latest commit in the clone 2025-09-10; GitHub "updated" 2026-06-23.
- Backends: numpy, or CuPy toggled through a `config.py` file containing `CUPY=False`.
- Paper: Mastrogiovanni et al. (2023), arXiv:2305.17973. Zenodo hosts the tutorials (record 7846415).
- Likelihoods are bilby `Likelihood` subclasses (`hierarchical_likelihood`, a Poisson + stochastic variant, a no-events variant).
- Stability rules are either:
  - `neffPE` = 20 per event by default, with `neffINJ` defaulting to 4 × N_events; or
  - `likelihood_variance_thr`. The variance includes finite-population factors: (N²/N_eff)(1 − N_eff/N_total) + Σ_i (1/n_eff,i)(1 − n_eff,i/S_i).
- It also covers cosmology with galaxy catalogues ("dark sirens") and stochastic backgrounds.
- The repo description calls it "a pure python package to estimate population properties of noisy observations in presence of selection biases. The code is mostly used for gravitational waves observations."

— [GitHub](https://github.com/simone-mastrogiovanni/icarogw); [likelihood.py](https://github.com/simone-mastrogiovanni/icarogw/blob/main/icarogw/likelihood.py); [arXiv:2305.17973](https://arxiv.org/abs/2305.17973)

**FIGARO** (https://github.com/sterinaldi/FIGARO; docs https://figaro.readthedocs.io; PyPI `figaro`; MIT)
- Repo facts: release 1.13.0 (2026-08-25); latest commit 2026-10-21; 20 stars and 0 open issues. Dependencies include jax, numba and ray.
- Papers: JOSS (Rinaldi & Del Pozzo 2024, doi:10.21105/joss.06589); the (H)DPGMM paper (arXiv:2109.05960, MNRAS 509, 5454).
- `DPGMM(bounds, prior_pars, alpha0, probit)` reconstructs a density from samples. `HDPGMM(bounds, MC_draws, selection_function, injection_pdf, total_injections, ...)` performs the hierarchical inference: each event enters as a set of single-event DPGMM draws, the event-level integrals are done by Monte Carlo, and the selection "alpha factor" ∫ p_det p(θ|λ) is computed per component, either from a callable p_det approximant or from injections.
- The number of components is not fixed in advance (README: "does not require the user to specify a priori the maximum allowed number of mixture components").
- Inference is a sequential, collapsed-Gibbs-style assignment of points to clusters (`add_new_point`/`_assign_to_cluster` with optional `n_reassignments`). It is repeated to give independent "draws" of the density. This description is my reading of the source; the docs call the outputs "draws".
- A probit transform handles bounded domains.
- CLIs: `figaro-density` and `figaro-hierarchical`. The loaders can "resample to get likelihood samples", which removes the PE prior.

— [README](https://github.com/sterinaldi/FIGARO); [mixture.py](https://github.com/sterinaldi/FIGARO/blob/main/figaro/mixture.py); [load.py](https://github.com/sterinaldi/FIGARO/blob/main/figaro/load.py); [PyPI](https://pypi.org/project/figaro/); [arXiv:2109.05960](https://arxiv.org/abs/2109.05960)
- Note: the PyPI `figaro` project history starts with an unrelated-looking 2014 "0.1" release, so the name probably predates this package (unverified).

**popsummary** (https://git.ligo.org/RatesAndPopulations/popsummary; PyPI 0.1.0, 2026-01-13; MIT)
- This is the LVK standard HDF5 result format, with only h5py and numpy as dependencies.
- `PopulationResult` getters and setters cover `hyperparameter_samples`, `reweighted_event_samples`, `reweighted_injections`, `fair_population_draws`, `rates_on_grids` (posterior and prior groups) and metadata (units, LaTeX labels, references, event waveforms). `tex_result()` prints credible intervals.
- Latest commit 2026-04-21; 8 GitLab stars.

— [GitLab](https://git.ligo.org/RatesAndPopulations/popsummary); [PyPI](https://pypi.org/project/popsummary/)

**LVK model repos**
- `gwtc6-population-models` (public; created 2026-07-17; last activity 2026-10-21) holds models "used as part of GWTC-6 astrophysical distributions analyses". It is installable only from git and ships default BBH/CBC `.ini` configs. Its README says models implemented "publicly elsewhere (e.g., in GWPopulation or PixelPop) are not reimplemented here". — [GitLab](https://git.ligo.org/RatesAndPopulations/gwtc6-population-models)
- `gwpopulation-fullpop` (PyPI 1.0.0, 2026-06-30; CC BY 4.0) is the FullPop mass/spin model as a gwpopulation plug-in and is used in GWTC-4 and GWTC-5. — [PyPI](https://pypi.org/project/gwpopulation-fullpop/); [GitLab](https://git.ligo.org/amanda.farah/gwpopulation-fullpop)
- The public RatesAndPopulations GitLab group has only 5 public projects: gwtc6-population-models, gwpopulation_pipe, popsummary, the legacy lalinfsamplereweighting (2019–2020) and BinnedCBCMasses (2016–2019). — [GitLab API listing](https://git.ligo.org/RatesAndPopulations)
- Legacy and superseded code:
  - Daniel Wysocki's "PopModels" (`daniel.wysocki/bayesian-parametric-population-models`; created 2017, last activity 2024-11; not on PyPI) has been superseded by gwpopulation in the LVK.
  - `colm.talbot/population` (2018–2019) is gwpopulation's predecessor.

  — [GitLab](https://git.ligo.org/daniel.wysocki/bayesian-parametric-population-models); [GitLab](https://git.ligo.org/colm.talbot/population)

**Newer (2024–2026) packages**
- **GWKokab** (https://github.com/kokabsc/gwkokab; docs https://gwkokab.readthedocs.io; PyPI 0.3.1, 2026-06-03; Apache-2.0)
  - 29 stars; latest commit 2026-10-04.
  - Built on JAX, NumPyro and flowMC (normalising-flow-enhanced MCMC), with parametric models only.
  - It supports a "GWalk" event representation, in which "each event is a mean and covariance rather than a" sample set (analytic Gaussian per-event likelihoods), as well as discrete sample-based Poisson likelihoods with a variance-tapering penalty "modified" from PixelPop's Eq. 12.
  - Papers: Qazalbash, Zeeshan & O'Shaughnessy, PRD 2026 (arXiv:2509.13638), plus a 2024 software citation.

  — [GitHub](https://github.com/kokabsc/gwkokab); [PyPI](https://pypi.org/project/gwkokab/); [arXiv:2509.13638](https://arxiv.org/abs/2509.13638)
- **Vamana** (PyPI 2.0.0, 2026-06-12; MIT; https://github.com/vaibhavtewari/vamana, 4 stars)
  - Models BBHs as a mixture of weighted Gaussians over m1, m2 and aligned spins, with Dirichlet mixture weights.
  - Samplers: its own Metropolis–Hastings or dynesty.
  - Priors are given as plain-text prior files, and it ships GWTC-3 and GWTC-5 configs.
  - Papers: arXiv:2006.15047 and arXiv:2510.25579.

  — [PyPI](https://pypi.org/project/vamana/); [GitHub](https://github.com/vaibhavtewari/vamana)
- **gravpop** (PyPI 0.1.1, 2024-10-28; https://github.com/Potatoasad/gravpop, 5 stars; latest commit 2025-11-14)
  - No licence: none in the PyPI metadata and no LICENSE file in the repo.
  - It fits each event's posterior with a truncated Gaussian mixture. Where the population model is itself a truncated-normal mixture (the "analytic sector"), the population likelihood is computed analytically; elsewhere (the "sampled sector") it falls back to Monte Carlo. The aim is narrow features without edge bias.

  — [GitHub](https://github.com/Potatoasad/gravpop); [PyPI](https://pypi.org/project/gravpop/)
- **CHIMERA** (PyPI `chimera-gw` 2.0.0, 2025-09-17; GPL-3.0+; https://github.com/CosmoStatGW/CHIMERA, 16 stars) does JAX cosmology with galaxy catalogues plus population inference. — [PyPI](https://pypi.org/project/chimera-gw/)
- **gwcosmo** (PyPI 3.1.0, 2026-07-22; GPL) is the LVK cosmology code. It depends on bilby, torch and nessai. — [PyPI](https://pypi.org/project/gwcosmo/)
- **wcosmo** (PyPI 0.5.4, 2025-09-08; MIT) is array-API/JAX cosmology used by gwpopulation. — [PyPI](https://pypi.org/project/wcosmo/)
- **GWMockCat** (git.ligo.org/amanda.farah/GWMockCat; 7 stars; last activity 2026-02) is "a lightweight code to generate mock catalogs of gravitational wave sources/CBCs, from user-defined p[opulations]" and is useful for testing. It is not on PyPI. — [GitLab](https://git.ligo.org/amanda.farah/GWMockCat)

**Census table A: status** (all verified on 2026-10-22)

Release dates come from PyPI JSON. Latest-commit dates come from shallow clones made on 2026-10-21/22. Stars come from GitHub search, or from the GitLab API for git.ligo.org projects.

| Package | Where (repo · docs · index) | Language / stack | Licence | Latest release | Latest commit | Stars | Status |
|---|---|---|---|---|---|---|---|
| gwpopulation | [GitHub](https://github.com/ColmTalbot/gwpopulation) · [docs](https://ColmTalbot.github.io/gwpopulation) · [PyPI](https://pypi.org/project/gwpopulation/) · conda-forge | Python; numpy/CuPy/JAX | MIT | 1.3.1 (2025-12-15) | 2026-07-17 | 62 | active; LVK reference |
| bilby (`bilby.hyper`) | [GitHub](https://github.com/bilby-dev/bilby) · [docs](https://bilby.readthedocs.io) · [PyPI](https://pypi.org/project/bilby/) | Python; numpy | MIT | 2.8.2 (2026-08-10) | (repo updated 2026-10-20) | 129 | active; hyper module minimal |
| gwpopulation_pipe | [GitLab](https://git.ligo.org/RatesAndPopulations/gwpopulation_pipe) · [docs](https://docs.ligo.org/RatesAndPopulations/gwpopulation_pipe) · [PyPI](https://pypi.org/project/gwpopulation-pipe/) | Python; HTCondor | MIT | 0.4.3 (2025-04-23) | 2026-07-23 | 4 | active |
| popsummary | [GitLab](https://git.ligo.org/RatesAndPopulations/popsummary) · [PyPI](https://pypi.org/project/popsummary/) | Python; h5py | MIT | 0.1.0 (2026-01-13) | 2026-04-21 | 8 | active |
| GWInferno | [GitHub](https://github.com/FarrOutLab/GWInferno) · [docs](https://farroutlab.github.io/GWInferno/) | Python; NumPyro/JAX | MIT | not on PyPI (0.0.2 in setup.py) | 2026-06-22 | 21 | active; pre-1.0 |
| icarogw | [GitHub](https://github.com/simone-mastrogiovanni/icarogw) · [Zenodo tutorials](https://zenodo.org/record/7846415) | Python; numpy/CuPy | EUPL-1.2 | not on PyPI (2.0.3 in setup.py) | 2025-09-10 | 17 | moderate |
| FIGARO | [GitHub](https://github.com/sterinaldi/FIGARO) · [docs](https://figaro.readthedocs.io) · [PyPI](https://pypi.org/project/figaro/) | Python; numpy/numba/JAX/ray | MIT | 1.13.0 (2026-08-25) | 2026-10-21 | 20 | active |
| PixelPop | [GitLab](https://git.ligo.org/jack.heinzel/pixelpop) · [PyPI](https://pypi.org/project/pixelpop/) | Python; JAX/NumPyro | MIT (pyproject; no LICENSE file) | 0.3.4 (2026-10-20) | 2026-10-05 | 3 | very active; LVK O4 |
| gppop | [GitHub](https://github.com/AnaryaRay1/gppop) | Python; PyTorch/GPyTorch/Pyro | GPL | not on PyPI (0.1.0) | 2026-07-30 | 1 | active research code |
| GWKokab | [GitHub](https://github.com/kokabsc/gwkokab) · [docs](https://gwkokab.readthedocs.io) · [PyPI](https://pypi.org/project/gwkokab/) | Python; JAX/NumPyro/flowMC | Apache-2.0 | 0.3.1 (2026-06-03) | 2026-10-04 | 29 | very active |
| Vamana | [GitHub](https://github.com/vaibhavtewari/vamana) · [PyPI](https://pypi.org/project/vamana/) | Python; numpy (dynesty optional) | MIT | 2.0.0 (2026-06-12) | 2026-06-17 | 4 | active; single author |
| gravpop | [GitHub](https://github.com/Potatoasad/gravpop) · [docs](https://potatoasad.github.io/gravpop/index.html) · [PyPI](https://pypi.org/project/gravpop/) | Python; JAX | none | 0.1.1 (2024-10-28) | 2025-11-14 | 5 | low activity |
| gwax | [GitHub](https://github.com/mdmould/gwax) · [PyPI](https://pypi.org/project/gwax/) | Python; JAX/flowjax/NumPyro | MIT | 0.1.0 (2025-09-29) | 2026-09-30 | 16 | active; self-described as undocumented |
| population-error | [GitHub](https://github.com/jack-heinzel/population-error) · [docs](https://population-error.readthedocs.io/) · [PyPI](https://pypi.org/project/population-error/) | Python; JAX | MIT classifier; no LICENSE file | 0.2.3 (2026-08-07) | 2026-10-04 | 1 | active |
| gwpopulation-fullpop | [GitLab](https://git.ligo.org/amanda.farah/gwpopulation-fullpop) · [PyPI](https://pypi.org/project/gwpopulation-fullpop/) | Python; gwpopulation plug-in | CC BY 4.0 | 1.0.0 (2026-06-30) | 2026-06-30 | 0 | active |
| gwtc6-population-models | [GitLab](https://git.ligo.org/RatesAndPopulations/gwtc6-population-models) | Python; gwpopulation plug-in | not checked | git only | 2026-09-23 | 1 | active (LVK GWTC-6) |
| autoregressive-bbh-inference | [GitHub](https://github.com/tcallister/autoregressive-bbh-inference) · [docs](https://tcallister.github.io/autoregressive-bbh-inference/) | Python; NumPyro | MIT | — | 2024-08-28 | 4 | frozen paper code |
| BinnedCBCMasses | [GitLab](https://git.ligo.org/RatesAndPopulations/BinnedCBCMasses) | Python + Stan | MIT | — | 2019-07-03 | 1 | unmaintained |
| PopModels | [GitLab](https://git.ligo.org/daniel.wysocki/bayesian-parametric-population-models) | Python | not checked | — | activity 2024-11-27 | 3 | legacy; superseded |
| pdetclassifier | [GitHub](https://github.com/dgerosa/pdetclassifier) | Python; TensorFlow | no LICENSE file seen | — | 2020-09-08 | 8 | unmaintained |
| learning-p-det / pdet | [GitHub](https://github.com/tcallister/learning-p-det) / [GitHub](https://github.com/tcallister/pdet) | Python; JAX arrays (per README output) | pdet MIT | — | 2024-09-05 / 2025-03-13 | 5 / n.v. | paper code / small O3-only package |
| poplar | [GitHub](https://github.com/cchapmanbird/poplar) · [docs](https://poplar.readthedocs.io) · [PyPI](https://pypi.org/project/poplar/) | Python; PyTorch | GPL-3.0+ | 0.3 (2025-12-11) | 2025-12-11 | 4 | low activity |
| flowMC (sampler) | [GitHub](https://github.com/GW-JAX-Team/flowMC) · [PyPI](https://pypi.org/project/flowmc/) | Python; JAX | MIT | 0.6.4 (2026-10-12) | n.v. | 7 | active |
| CHIMERA (cosmology + population) | [GitHub](https://github.com/CosmoStatGW/CHIMERA) · [PyPI](https://pypi.org/project/chimera-gw/) | Python; JAX | GPL-3.0+ | 2.0.0 (2025-09-17) | n.v. | 16 | active |
| gwcosmo (cosmology) | [GitLab](https://git.ligo.org/lscsoft/gwcosmo) · [PyPI](https://pypi.org/project/gwcosmo/) | Python; torch/nessai | GPL | 3.1.0 (2026-07-22) | n.v. | n.v. | active |
| GWMockCat (mock catalogues) | [GitLab](https://git.ligo.org/amanda.farah/GWMockCat) | Python | not checked | — | activity 2026-02-06 | 7 | utility |
| Hierarchical-Bayesian-Flows | [GitLab](https://git.ligo.org/jacob.schaefer/Hierarchical-Bayesian-Flows) | Python; PyTorch | none | — | activity 2024-12-24 | 0 | undocumented research code |

(n.v. = not verified.)

**Census table B: capabilities** of the population-inference packages

| Package | Per-object input | Interim-prior correction | Selection effects | Model family | Inference engine | Uncertainty / Monte Carlo outputs | Smoothness priors | Scale / GPU | API style | Main limitations |
|---|---|---|---|---|---|---|---|---|---|---|
| gwpopulation | posterior samples (list of DataFrames; ragged since 2026) | yes (`prior` column) | injections (`ResamplingVT`) or gridded p_det (`GridVT`); Poisson-rate form | parametric + cubic-spline nodes | any bilby sampler (dynesty default in LVK); NumPyro NUTS (experimental) | per-sample σ²_lnL̂, per-event ln BF and variance; hard cut | spline RMS regulariser only | numpy/CuPy/JAX; GPU | plain functions + bilby PriorDict | bilby/wcosmo coupling; GW-specific VT semantics; autodoc-only guide |
| bilby.hyper | posterior samples (equal counts) | yes | **none** | any | bilby samplers | none | none | numpy | function | minimal |
| GWInferno | PE samples + injections (catalog JSON / PESummary) | yes | injections | parametric + separable B-splines | NumPyro NUTS | per-event n_eff/variance; N_eff and variance cuts; Farr correction | P-spline difference prior | JAX GPU | YAML + CLI, or NumPyro model | not on PyPI; separable (no correlations) splines |
| icarogw | PE samples + injections (+ galaxy catalogues) | yes | injections | parametric (+ cosmology) | bilby samplers | N_eff or variance thresholds | — | numpy/CuPy | class wrappers | EUPL; git install; no tests |
| FIGARO | samples of any dimension | yes (resample to "likelihood samples" at load) | p_det callable or injections (HDPGMM); the `figaro-density` `--selfunc` is 1D only | DPGMM / (H)DPGMM | sequential collapsed-Gibbs-style assignment giving independent draws | draws (realisations) of the density | Dirichlet process (no explicit smoothness) | CPU (numba, ray) | CLI + Python | 54-line test suite |
| PixelPop | PE samples (padded, ragged) + injections | yes (`log_prior`) | injections | binned log-rates + parametric others | NumPyro NUTS | σ²_lnL̂, N_eff tapers; Ê via population-error; mid-run r-hat/ESS | ICAR (sparse GMRF), σ marginalised | JAX GPU; ~10^4 bins | scripts + setup functions; popsummary output | improper prior needs initialisation; GW data loaders |
| gppop | per-bin weight matrix W + per-bin VT ± σ | yes (in weights) | per-bin VT from injections | binned log-rates | Pyro NUTS, multi-GPU | var_pe, var_sel; soft variance cut | Kronecker RBF GP | torch GPU; ~8×10^4 bins (4D) | CLI `run_gppop` + metadata | no tests; GPL |
| GWKokab | PE samples, or per-event Gaussian (mean, covariance) | yes | injection-based μ(Λ) with variance (other options n.v.) | parametric multi-source | flowMC; NumPyro | variance taper | — | JAX GPU | NumPyro Distribution subclasses + CLI configs | parametric only; complex CLI |
| Vamana | PE samples + injections | yes (n.v.) | injections | Gaussian mixture, Dirichlet weights | custom MH; dynesty | n.v. | — | numpy CPU | prior files + `Analysis` | BBH-specific; no tests |
| gravpop | truncated Gaussian mixture fits to PE samples, plus samples | n.v. | selection-function class (n.v.) | truncated-normal mixtures (analytic) + Monte Carlo sector | n.v. | n.v. | — | JAX | n.v. | no licence |
| gwax | PE samples (ragged "stacked") + injections | yes | injections | parametric + ICAR binned | flow VI + importance sampling; NumPyro | variance, ESS, taper hook | ICAR | JAX GPU | functions | "not … documented"; no tests |
| population-error | hyperposterior + PE samples + injections | yes | injections | any gwpopulation-style model | post-processing | Ê (bits), precision, accuracy | — | JAX | one function | depends on gwpopulation/bilby |


### Inferences
- The estimator pastry needs for "posterior samples under an interim prior" is exactly gwpopulation's: the per-object term is (1/S) Σ_s ρ(θ_s)/π_int(θ_s). In pastry's binned case this collapses to L_ik ≈ (1/S) Σ_s 1[θ_s∈k]/(π_int(θ_s)·Δ_k). Those weights can be precomputed once, after which every likelihood call is a matrix–vector product (W @ ρ), the same trick GWInferno, PixelPop and gppop use.
- The selection term generalises cleanly: with injections drawn from p_draw, P̂_det(ρ) = Σ_k ρ_k·VT_k, where VT_k = (1/N_draw)·Σ_{found j∈k} 1/(p_draw(θ_j)·Δ_k). That makes it linear in ρ, so for binned models the Monte Carlo variance of the selection term also has closed form (gppop computes per-bin `vt_vars`).
- Nobody in GW ships a lightweight, domain-neutral "hierarchical likelihood from samples" library. gwpopulation comes closest, but it is coupled to bilby, wcosmo and redshift/VT semantics. This is an open niche that pastry can fill, and it can borrow gwpopulation's function-based model contract.
- popsummary's schema (hyperposterior samples, reweighted per-object samples, reweighted injections, fair draws, rates on grids, with posterior and prior groups) is a ready-made blueprint for pastry's result object and HDF5 export.

### Gaps
- I could not verify that the `icarogw-developers/icarogw` repo named in the README exists (a GitHub search for it returned nothing).
- I could not get GitHub release counts for GWInferno or icarogw, since neither publishes on PyPI.
- I did not verify whether FIGARO has been applied outside GW in the literature.
- I did not read the GWTC-4/5 population papers' likelihood appendices in full, only their section headings.

## Q2. Binned and nonparametric population models

### Takeaway
Every GW "weakly modelled" approach that scales uses one recipe:
- pick a fixed set of bins (or basis functions, or sample locations);
- precompute, once, which bin or basis value each PE sample and injection falls in;
- put a log-Gaussian Markov or Gaussian-process prior on the log-rates, with a non-centred ("whitened") parametrisation;
- sample everything jointly with gradient-based HMC/NUTS (Stan in 2016; NumPyro, JAX or Pyro since 2022) on GPUs.

Known working scales:
- 22 log-mass bins (BGP in GWTC-4);
- 100 bins per dimension, i.e. 10^4 bins in 2D (PixelPop in GWTC-5, ICAR prior);
- about 8×10^4 bins in 4D (gppop 2026, Kronecker-factorised GP, 4 GPUs);
- one latent per PE sample and injection (AR(1) model).

Dirichlet-process mixtures (FIGARO) and Gaussian mixtures (Vamana, RJMCMC "FM") are the main non-grid alternatives. The LVK found smoothing hyperparameters poorly identified: GWTC-4 fixed the B-spline τ because it "consistently railed against prior boundaries", and the Monte Carlo variance cut was the effective regulariser.

### Cited Findings
**Binned Gaussian-process models**
- **Mandel, Farr, Colonna, Stevenson, Tiňo & Veitch (arXiv 2016; published 2017, journal not verified here)**, "Model-independent inference on compact-binary observations" (arXiv:1608.08223), introduced model-independent "clustering" of noisy observations and showed subpopulations (BNS/NSBH/BBH) are distinguishable with tens of events. — [arXiv:1608.08223](https://arxiv.org/abs/1608.08223)
- The companion LVK code **BinnedCBCMasses** (https://git.ligo.org/RatesAndPopulations/BinnedCBCMasses; MIT, © 2015–2016 Will Farr; last commit 2019-07-03; effectively unmaintained) is a Python executable plus a Stan program. The README says it fits "a binned model to 2D data using a Gaussian process to regularise the fit. The method is similar to Foreman-Mackey, Hogg & Morton (2014)".
  - Inputs are per-event weight matrices `weights[nevents][nbins,nbins]`, which are exactly pastry's L_ik, plus per-bin log-normal VT means and widths (`mu_log_alphas`, `sigma_log_alphas`).
  - Model: a non-centred GP on the log-rate (`logn = chol(K)·raw + mu`, squared-exponential kernel `cov_exp_quad`, a "white-noise fraction" jitter). Each bin's selection term is a latent log-normal (`log_alphas = mu + sigma .* rawloga`), so Monte Carlo uncertainty in VT is marginalised. The likelihood is `Σ_i log_sum_exp(logwts[i] + logn) − Σ Δ·exp(logn + log_alphas)`.
  - The length-scale prior is log-normal, spanning the smallest bin width to the full range. A `prior_flag` gives prior-only runs.

  — [GitLab](https://git.ligo.org/RatesAndPopulations/BinnedCBCMasses); [population.stan](https://git.ligo.org/RatesAndPopulations/BinnedCBCMasses/-/blob/master/population.stan)
- **Binned Gaussian process (BGP; Ray et al.)** papers:
  - Ray, Magaña Hernandez, Mohite, Creighton & Kapadia (2023), ApJ 957, 37 (arXiv:2304.08046): jointly infers the mass and redshift distributions, including correlations.
  - Ray et al. (2025), ApJ 991, 17 (arXiv:2404.03166): masses and χ_eff.
  - Sridhar, Ray & Kalogera (2026), ApJL 1005, L54 (arXiv:2511.22093): m1, q and χ_eff in GWTC-4, finding three subpopulations.
  - Ray & Kalogera (2026, arXiv:2607.28622): "first data-driven reconstruction of the joint four-dimensional distribution" (m1, q, χ_eff, χ_p).

  — [arXiv:2304.08046](https://arxiv.org/abs/2304.08046); [arXiv:2404.03166](https://arxiv.org/abs/2404.03166); [arXiv:2511.22093](https://arxiv.org/abs/2511.22093); [arXiv:2607.28622](https://arxiv.org/abs/2607.28622)
- **gppop** code (https://github.com/AnaryaRay1/gppop; GPL; v0.1.0 in pyproject; not on PyPI; 1 star; latest commit 2026-07-30, "4D model with GPU acceleration")
  - The 2026 version is PyTorch + GPyTorch/linear-operator + Pyro; the pyproject lists "Claude (Anthropic)" as a co-author, and the docstring calls it the "Pyro model equivalent to the PyMC `make_significant_model_3d_n_eff_opt`" (the earlier version was PyMC-based).
  - Model `GPPopKronModel`:
    - priors: μ (truncated normal), σ ~ HalfNormal, per-dimension length scales ~ LogNormal;
    - a whitened latent `w ~ N(0, I)`, with the log-rate equal to μ + σ·(L_kron @ w), where L is a Kronecker product of per-dimension Cholesky factors ("LKGP trick");
    - likelihood `Σ log(W @ n) − Σ n·VT`;
    - Monte Carlo variances from per-bin weight SDs and VT SDs;
    - a soft variance cut, −softplus(exponent·(log max² − log var)).

    Sampling is Pyro NUTS, with one chain per GPU (`run_nuts_multigpu`).
  - The GWTC-5 4D example uses 26 log-uniform m1 bins (3–200 M⊙), 9 q bins, 25 χ_eff bins and 14 χ_p bins (about 8×10^4 bins before physical-domain masking). It runs `--ntune=3000 --nsteps=3000 --target_accept=0.99 --maximum_uncertainty=1.0 --devices cuda:0,...,cuda:3`.
  - Tests are absent: the `tests/` directory holds only `__init__.py`.

  — [GitHub](https://github.com/AnaryaRay1/gppop); [rates.py](https://github.com/AnaryaRay1/gppop/blob/main/src/gppop/rates.py); [example](https://github.com/AnaryaRay1/gppop/tree/main/examples/GWTC5_m1qchieffchip)
- **GWTC-4 BGP settings and caveats.** The LVK used "22 bins spaced uniformly in log-mass for the mass BGP models, and 15 bins … between χ_eff ∈ [−0.7,0.7]", with a half-normal(1) prior on σ and length-scale priors "tuned to the mean and variation in the set of distances between the bin centers". Stated caveats: "it assumes an arbitrary binning scheme, which reduces the resolution … and leads to unphysical discontinuities"; rates "in some bins may be driven by the a priori assumption of a Gaussian covariance, and not the data"; and smoothing "near sharp features e.g., near the m1=m2 boundary". — [GWTC-4.0 population paper, App. C.2](https://arxiv.org/abs/2508.18083)

**PixelPop (ICAR / Gaussian-Markov-random-field binned model)**
- Papers and code:
  - Heinzel, Mould, Álvarez-López & Vitale (2025), PRD 111, 063043 (arXiv:2406.16813): "densely bins the joint parameter space and directly infers the merger rate in each bin, assuming only that bins are coupled to their nearest neighbors".
  - The GWTC-3 application is arXiv:2406.16844 (PRD 111, L061305).
  - Álvarez-López et al. (2026), ApJL 1006, L5 (arXiv:2506.20731): 400 full-PE mock detections from population synthesis; "neglecting population-level correlations biases inference".
  - Code: https://git.ligo.org/jack.heinzel/pixelpop (PyPI `pixelpop` 0.3.4, 2026-10-20; MIT; 21 releases since 2024-11; JAX + NumPyro; depends on gwpopulation, gwpopulation_pipe, popsummary and population_error).

  — [arXiv:2406.16813](https://arxiv.org/abs/2406.16813); [arXiv:2506.20731](https://arxiv.org/abs/2506.20731); [PyPI](https://pypi.org/project/pixelpop/); [GitLab](https://git.ligo.org/jack.heinzel/pixelpop)
- Implementation details (source):
  - `ICAR_length_scales` is a NumPyro distribution with sparse per-dimension adjacency matrices and optional per-dimension log σ.
  - The ICAR smoothness scale can be marginalised, either by an analytic Gamma reparametrisation of the precision or by a grid conditional for per-dimension scales.
  - Each PE sample's and injection's bin index is precomputed, so each step costs one gather (`merger_rate_density[event_bins]`).
  - Rate likelihood with smooth tapers on: the total variance ("smooth cutoff above Talbot+Golomb 2022 recommendation to retain autodifferentiability"); single-event N_eff; and selection N_eff (4·N_obs).
  - The README notes that "an initialization is required since the ICAR model is an improper prior".
  - Output is popsummary format.
  - The README states "The dimension of the inference problem can become very large (e.g. 10^4 for a 2-dimensional space with a density of 100 bins along each axis)" and relies on JAX autodiff, GPUs and NUTS.

  — [models/car.py](https://git.ligo.org/jack.heinzel/pixelpop/-/blob/main/pixelpop/models/car.py); [models/probabilistic.py](https://git.ligo.org/jack.heinzel/pixelpop/-/blob/main/pixelpop/models/probabilistic.py); [README](https://git.ligo.org/jack.heinzel/pixelpop)
- **GWTC-5 PixelPop settings.**
  - Prior: the log-normal ICAR p(ln R_β | neighbours, σ) ∝ exp[−Σ_b (ln R_β − ln R_b)²/(2σ²)], with σ ~ U(ln σ ∈ [−3, 5]).
  - "100 bins per dimension … our one- and two-dimensional PixelPop models have 10^2 and 10^4 rate parameters".
  - Nine 2D variants were run: (m1,m2), (χ,cosθ), (m1,q), (m1,χ_eff), (m1,z), (χ_eff,χ_p), (χ_eff,q), (χ_eff,z), plus 1D z. Non-binned parameters use the parametric defaults.

  — [GWTC-5.0 population paper, App. B.1](https://arxiv.org/abs/2605.27226)

**Autoregressive-process model (Callister & Farr)**
- Callister & Farr (2024), "A Parameter-Free Tour of the Binary Black Hole Population", PRX 14, 021005 (arXiv:2302.07289), models "the merger rate of binary black holes as an unknown autoregressive process over the space of binary parameters". — [arXiv:2302.07289](https://arxiv.org/abs/2302.07289)
- Code: https://github.com/tcallister/autoregressive-bbh-inference (MIT; paper-reproduction repo; last commit 2024-08-28; data on Zenodo, doi:10.5281/zenodo.7600140).
  - The AR(1)/OU process is evaluated at the sorted union of all PE-sample and injection values (`all_lnm1_samples`, `lnm1_deltas = np.diff(sorted)`), so there is one latent per sample.
  - Exact discretisation: φ = exp(−Δ/τ), innovation = sqrt(1 − e^{−2Δ/τ})·σ·ε. It is built with `jax.lax.scan` from a reference point, both forwards and backwards, with non-centred ε ~ N(0,1).
  - Priors: σ ~ HalfNormal and log τ ~ Normal, plus a "regularizing log-likelihood factor" on σ/√τ (paper Eq. B3).
  - Soft penalties: log(1/(1 + (N_eff,inj/(4N_obs))^−30)) on injections, and an analogous one on the minimum per-event log10 N_eff.
  - Sampling: NumPyro NUTS (1000 warm-up + 1500 samples) with a dense mass matrix only on the (σ, τ) pairs.

  — [code/autoregressive_mass_models.py](https://github.com/tcallister/autoregressive-bbh-inference/blob/main/code/autoregressive_mass_models.py); [code/run_ar_lnm1_q.py](https://github.com/tcallister/autoregressive-bbh-inference/blob/main/code/run_ar_lnm1_q.py)
- GWTC-4 description: "The merger rate density at each sample is its own hyperparameter … Without any further assumptions, such a model is severely underconstrained and will converge on the maximum likelihood functional distribution". The model "cannot model correlations", and "lower bounds on the merger rate in some regions may not be an accurate lower bound on the true distribution". — [GWTC-4.0 population paper, App. C.3](https://arxiv.org/abs/2508.18083)
- Payne & Thrane (2023), PRR 5, 023013 (arXiv:2210.11641): the maximum-population-likelihood distribution over all possible priors is "a linear superposition of delta functions", following from Carathéodory's theorem. This is why unregularised nonparametric fits collapse onto PE samples. — [arXiv:2210.11641](https://arxiv.org/abs/2210.11641)

**Spline models**
- Edelman, Doctor, Godfrey & Farr (2022), ApJ 924, 101 (arXiv:2109.06137): a cubic-spline perturbation of a power law.
- Edelman, Farr & Doctor (2023, arXiv:2210.12834), "Cover Your Basis": "the first complete non-parametric model" of the BBH population, with B-splines for mass, mass ratio, spins and redshift.
- Golomb & Talbot (2023, arXiv:2210.12287): cubic-spline interpolants for the spin distributions, which "substantially increases the uncertainty".

— [arXiv:2109.06137](https://arxiv.org/abs/2109.06137); [arXiv:2210.12834](https://arxiv.org/abs/2210.12834); [arXiv:2210.12287](https://arxiv.org/abs/2210.12287)
- GWTC-4 B-spline settings: the prior is p(α|τ) = exp(−½ τ αᵀ D_rᵀ D_r α), a P-spline r-th-order difference penalty, with n ≈ 30–40 bases. "We found that τ consistently railed against prior boundaries, which meant that the limits imposed on the likelihood uncertainty was the main driver of smoothness. We therefore fix τ to a reasonable value (roughly between 5−10)". — [GWTC-4.0 population paper, App. C.1](https://arxiv.org/abs/2508.18083)
- These are implemented in GWInferno: `apply_difference_prior`, coefficients ~ Normal(0, σ_c), and precomputed design matrices at PE samples and injections. See Q1. — [GWInferno](https://github.com/FarrOutLab/GWInferno)

**Dirichlet-process and mixture models**
- Rinaldi & Del Pozzo (2022), MNRAS 509, 5454 (arXiv:2109.05960), the (H)DPGMM: "a hierarchical Bayesian non-parametric method based on the Dirichlet Process Gaussian Mixture Model … without the need of fine-tuning". It is implemented in FIGARO (see Q1). — [arXiv:2109.05960](https://arxiv.org/abs/2109.05960)
- Flexible Mixtures (FM) / VAMANA (Tiwari): a Gaussian mixture in chirp mass and aligned spins, a power law in mass ratio and redshift, and "variable model dimension, sampling the trans-dimensional posterior using a reversible-jump MCMC method". GWTC-4 used 11 components. — [GWTC-4.0, App. C.4](https://arxiv.org/abs/2508.18083); [arXiv:2006.15047](https://arxiv.org/abs/2006.15047); [Vamana PyPI](https://pypi.org/project/vamana/)
- Sadiq, Dent & Wysocki (2022, arXiv:2112.12659) use adaptive-bandwidth KDE (awKDE) of event estimates with a fitted sensitivity. This is a fast, non-hierarchical deconvolution alternative, accurate only when "individual event measurement error [is low] relative to the scale of population features". — [arXiv:2112.12659](https://arxiv.org/abs/2112.12659)

**LVK usage pattern**
- GWTC-4 used B-Spline and BGP as its main weakly modelled approaches, with AR and FM in an appendix. It notes that "a unified Bayesian approach to capture generic population features is still an open problem". — [GWTC-4.0, Sec. II.2](https://arxiv.org/abs/2508.18083)
- GWTC-5 uses PixelPop (1D and 2D), a 3D BGP, the Isolated Peak model (cubic splines), and spline-correlation models. It hatches PPD regions "where we have under 1% of PE samples … as a heuristic proxy for parts of parameter space where the models may not be actively constrained by the data". — [GWTC-5.0, Sec. III](https://arxiv.org/abs/2605.27226)

### Inferences
- pastry's baseline (ρ ~ Dir(α), known L_ik) is a special case of these models with no smoothness coupling. The GW evidence argues for making ICAR/GMRF (sparse, O(K) per gradient) and Kronecker-GP (O(Σ_d K_d³) for the Cholesky) priors on log ρ first-class, and for defaulting to a non-centred parametrisation with NUTS.
- PixelPop shows that a sparse ICAR on 10^4 bins is practical under NUTS on GPU. gppop shows that a Kronecker GP on about 10^5 bins is practical with multi-GPU NUTS. A dense GP over K bins is O(K³) and does not scale beyond about 10³–10⁴ bins.
- In pastry's notation, PixelPop's and gppop's models are p({d}|ρ) = Π_i Σ_k L_ik ρ_k with a softmax(log-rate) or Poisson-rate link. The rate form (Σ log(W n) − Σ n·VT) avoids the simplex constraint entirely and makes selection linear. pastry should consider offering both the "fractions" (Dirichlet/simplex) and "rates" (Poisson, unconstrained log-rates) parametrisations.
- Smoothing hyperparameters (τ, σ, length scales) were poorly identified in LVK practice (τ railing; fixed by hand). pastry should therefore treat them carefully: marginalise analytically where conjugate (PixelPop's Gamma trick), report their posteriors, and offer empirical-Bayes or fixed options with diagnostics.
- The "data-support" hatching heuristic (below 1% of pooled PE samples) is a cheap, useful default diagnostic for binned outputs, for example flagging bins with negligible Σ_i L_ik.

### Gaps
- I did not find wall-clock benchmarks (time vs number of bins) for PixelPop or the 4D gppop in the material I read; the papers may contain them, but I did not verify.
- I did not verify how gppop masks the physical domain (the exact number of retained bins in 4D).
- I did not read Mandel et al. (2017) in full; the GP-regularised binned model is described in the companion code README, not verified against the paper text.

## Q3. Neural approaches (flows as population models, selection emulators, SBI)

### Takeaway
Neural methods enter GW population inference in five distinct roles:
1. A normalising flow as the population density itself, trained through the sample-based hierarchical likelihood (Ruhe et al. 2022; Iampieri & Mastrogiovanni 2026).
2. A flow as a variational posterior over hyperparameters, used as a fast drop-in sampler with importance-sampling repair (Mould, Wolfe & Vitale 2025; implemented in `gwax`).
3. Flows or networks emulating population-synthesis outputs (Wong et al. 2020; Mould et al. 2022; Colloms et al. 2025).
4. Neural emulators of the selection function p_det(θ) or its population average (Gerosa et al. 2020; Talbot & Thrane 2022; Callister et al. 2024; poplar).
5. Amortised simulation-based inference that bypasses per-event Monte Carlo entirely (Leyde et al. 2024; Dingo-Pop 2026).

Role 1 is exactly pastry's "smooth density trained through an IWAE-style marginal likelihood". Only research code exists for it; there is no maintained package.

### Cited Findings
**Flows as the population model (pastry's "smooth density" use case)**
- Ruhe, Wong, Cranmer & Forré (2022), "Normalizing Flows for Hierarchical Bayesian Analysis: A Gravitational Wave Population Study" (arXiv:2211.09008).
  - Approach: "parameterizing the population distribution … with a normalizing flow".
  - Application: four parameters (m1, m2, z, χ_eff) of the then-latest LVK data.
  - Their posterior predictive distributions, computed "assuming a prior over the parameters of the flow", "recover structure that agrees with robust previous phenomenological modeling results while being less susceptible to biases introduced by less flexible models".
  - I could not verify whether code is public.

  — [arXiv:2211.09008](https://arxiv.org/abs/2211.09008)
- Iampieri & Mastrogiovanni (2026-09-10), "Hierarchical Population Inference with Normalizing Flows for Binary Black Holes" (arXiv:2609.11885).
  - Model: the BBH population in (z, m1, m2) "is represented by a normalizing flow trained directly through the hierarchical likelihood, with event posterior samples propagating measurement uncertainty and detected injections accounting for selection effects".
  - Support: they "restrict population statements to the region supported by the injection campaign", noting that "conditioning on this region can itself induce an apparent association between mass and redshift".
  - They introduce "a mutual-information diagnostic" against a redshift-independent reference.
  - A May 2025 conference talk described an earlier version: the flow was optimised by maximising the hierarchical likelihood on simulated catalogues, with full posterior sampling of the flow parameters left as future work.

  — [arXiv:2609.11885](https://arxiv.org/abs/2609.11885); [IN2P3 talk, May 2025](https://indico.in2p3.fr/event/35882/contributions/154846/)
- Li, Cheng, Cui & Fang (2026), "Model-Agnostic Population Inference for Gravitational-Wave Astronomy: From LVK to LISA" (arXiv:2602.00195).
  - Model: a "Correlated Compound-Mixture Density Network" (mixtures, Gaussian copulas, flexible marginals).
  - Training: "amortized variational inference with Normalizing Flows" for the hyperparameter posterior.
  - Applications: GWTC-3 (with injection-based selection) and LISA mocks.

  — [arXiv:2602.00195](https://arxiv.org/abs/2602.00195)
- **Hierarchical-Bayesian-Flows** (https://git.ligo.org/jacob.schaefer/Hierarchical-Bayesian-Flows; public; created 2024-09, last activity 2024-12; no licence or description; the README is a one-word placeholder) is research code. It contains continuous-normalising-flow (`CNF.py`), flow-matching (`MFM.py`: a time-conditioned vector-field network with Fourier time features), mixture-density-net and importance-sampling notebooks. It shows that flow matching for hierarchical GW populations has been explored, but none of it is packaged. — [GitLab](https://git.ligo.org/jacob.schaefer/Hierarchical-Bayesian-Flows)

**Flows as variational posteriors over hyperparameters**
- Mould, Wolfe & Vitale (2025), PRD 111, 123049 (arXiv:2504.07197).
  - Method: "learns the population posterior with a normalizing flow and serves as a drop-in replacement for existing samplers".
  - Speed: "inference takes just seconds" on accelerators, "requiring up to three orders of magnitude fewer likelihood evaluations and as few as O(10^3)".
  - Repair: discrepancies "can be addressed with smoothed importance sampling".
  - Accuracy: agreement with stochastic samplers at Jensen–Shannon divergence < 0.1 nats in 14 dimensions (as summarised by search results).

  — [arXiv:2504.07197](https://arxiv.org/abs/2504.07197)
- **gwax** (https://github.com/mdmould/gwax; PyPI `gwax` 0.1.0, 2025-09-29; MIT; 16 stars; latest commit 2026-09-30)
  - The README calls it "Gravitational-wave astronomy in JAX". It lists "Flow-based variational inference, e.g., for population analyses", "Nonparametric population models", and a likelihood "that avoids single-event downsampling". It also warns: "not as a comprehensive and documented package".
  - `variational.trainer()` minimises E_q[log q − log target] with flowjax block-neural-autoregressive flows. It supports an optional `taper(variance)` term and `temper_schedule`.
  - `importance()` reports the efficiency/ESS and the log-evidence ± σ.
  - `nonparametric.py` provides ICAR priors (whitened, Gamma-marginalised τ, Student-t variants) on binned rates.
  - That gwax is the code behind arXiv:2504.07197 is likely (same author, matching API) but not confirmed.

  — [GitHub](https://github.com/mdmould/gwax); [variational.py](https://github.com/mdmould/gwax/blob/main/gwax/variational.py); [PyPI](https://pypi.org/project/gwax/)
- **flowMC** (PyPI 0.6.4, 2026-10-12; MIT; https://github.com/GW-JAX-Team/flowMC) is a normalising-flow-enhanced MCMC sampler in JAX, and GWKokab's sampler. It is a sampler, not a population model. — [PyPI](https://pypi.org/project/flowmc/)

**Flows and networks as emulators of simulated populations**
- Wong, Contardo & Ho (2020), PRD 101, 123005 (arXiv:2002.09491): a flow-based network emulating "a phenomenological model with 6 observables and 4 hyper-parameters"; "a 10-layer network can emulate the phenomenological model accurately". — [arXiv:2002.09491](https://arxiv.org/abs/2002.09491)
- Wong, Breivik, Kremer & Callister (2021), PRD 103, 083021 (arXiv:2011.03564): a deep-learning-enhanced hierarchical analysis of COSMIC+CMC population synthesis. — [arXiv:2011.03564](https://arxiv.org/abs/2011.03564)
- Mould, Gerosa & Taylor (2022), PRD 106, 103013 (arXiv:2203.03651): deep networks emulate hierarchical-merger simulations and "estimate selection effects". — [arXiv:2203.03651](https://arxiv.org/abs/2203.03651)
- Colloms, Berry, Veitch & Zevin (2025), ApJ 988, 189 (arXiv:2503.03819), and Colloms et al. (2025, arXiv:2508.19336): normalising flows emulate population-synthesis models (five models in the second paper) and "interpolate between astrophysical input parameters" to infer branching ratios. — [arXiv:2503.03819](https://arxiv.org/abs/2503.03819); [arXiv:2508.19336](https://arxiv.org/abs/2508.19336)

**Selection-function emulators**
- **pdetclassifier.** Gerosa, Pratten & Vecchio (2020), PRD 102, 103020 (arXiv:2007.06585), trained neural-network classifiers for detectability.
  - Code: https://github.com/dgerosa/pdetclassifier (8 stars). Pretrained TensorFlow `.h5` models for design sensitivity, O1O2, O3 and O4 forecasts, trained on 2×10^7 binaries with an SNR threshold.
  - Last commit 2020-09-08, so effectively unmaintained.

  — [arXiv:2007.06585](https://arxiv.org/abs/2007.06585); [GitHub](https://github.com/dgerosa/pdetclassifier)
- Talbot & Thrane (2022), ApJ 927, 76 (arXiv:2012.01317), "employ density estimation methods" on injections, with "smaller statistical errors at comparable computational cost than the method currently most widely used". — [arXiv:2012.01317](https://arxiv.org/abs/2012.01317)
- **learning-p-det / pdet.** Callister, Essick & Holz (2024), PRD 110, 123041 (arXiv:2408.16828), trained a network on O3 pipeline injections.
  - Paper code: https://github.com/tcallister/learning-p-det (5 stars; last commit 2024-09).
  - Packaged emulator: https://github.com/tcallister/pdet (MIT; GitHub-only; last commit 2025-03-13; O3 HLV only). Usage: `p = pdet_O3(); p.predict({'mass_1':…, 'mass_2':…, 'a_1':…, 'a_2':…, 'redshift':…})`. The PyPI name `pdet` belongs to an unrelated medical package.
  - Their "dynamic injections" model draws fresh injections from the current population at each likelihood call (inverse-CDF on grids), evaluates the emulator, and uses N_exp = R·mean(p_det). This avoids reweighting fixed injections.

  — [arXiv:2408.16828](https://arxiv.org/abs/2408.16828); [pdet README](https://github.com/tcallister/pdet); [population_model.py](https://github.com/tcallister/learning-p-det/blob/main/code/population_model.py)
- Lorenzo-Medina & Dent (2025), CQG 42, 045008 (arXiv:2408.13383), give a physically modelled (semi-analytic, not neural) selection function as an alternative to reweighting injections. — [arXiv:2408.13383](https://arxiv.org/abs/2408.13383)
- **poplar** (https://github.com/cchapmanbird/poplar; PyPI 0.3, 2025-12-11; GPL-3.0+; PyTorch; 4 stars) is described as "Augmenting selection bias modelling with machine learning for big speedups". It was introduced for LISA extreme-mass-ratio-inspiral populations by Chapman-Bird et al. (2023, MNRAS, doi:10.1093/mnras/stad1397). Its `PopulationLikelihood` uses a Poisson selection factor, −R·S(Λ) + N ln R, where S is evaluated as a function of the hyperparameters; this reading is from the source code. — [GitHub](https://github.com/cchapmanbird/poplar); [PyPI](https://pypi.org/project/poplar/)

**Simulation-based or amortised inference**
- Leyde, Green, Toubiana & Gair (2024), PRD 109, 064056 (arXiv:2311.12093): neural posterior estimation (NPE) with a flow gives "automatic inclusion of selection effects" and no stochastic samplers. It "scales to ∼600 events before performance degrades", and the authors split catalogues into sub-populations that are later recombined. — [arXiv:2311.12093](https://arxiv.org/abs/2311.12093)
- Leyde, Green, Dax, Mould, Fabbri & Gair (2026), "Dingo-Pop" (arXiv:2605.11274): a transformer over per-event tokens taken directly from strain, trained on simulated catalogues with selection. It offers "population inference without per-event Monte Carlo sampling noise", amortises over 25–1000 events, and runs end-to-end in about 1 s. — [arXiv:2605.11274](https://arxiv.org/abs/2605.11274)

**Joint (non-marginalised) hierarchical sampling** (relevant to pastry's joint-sampling plan)
- Mancarella & Gerosa (2025), PRD 111, 103012 (arXiv:2502.12156), sample all GWTC-3 event-level parameters and the population hyperparameters jointly. That is "a parameter space of about 500 dimensions", which gives population-informed single-event posteriors and their correlations with the population parameters directly. — [arXiv:2502.12156](https://arxiv.org/abs/2502.12156)
- Heinzel & Vitale (2025) group this with density-estimate approaches (FIGARO; Callister et al. 2022, arXiv:2205.08574; Hussain et al. 2024, arXiv:2411.02252). They describe it as a route to "avoid the single-event Monte Carlo integrals", and analytic/ML selection approaches as a route to remove the selection Monte Carlo. — [arXiv:2509.07221](https://arxiv.org/abs/2509.07221)

### Inferences
- **IWAE framing.** For pastry's planned flow or flow-matching population densities, the per-object term ln[(1/S) Σ_s q_φ(θ_s)/π_int(θ_s)] is exactly an IWAE-style log-mean-exp estimator. In expectation it is a lower bound on the per-object log marginal likelihood (Jensen).
- **The selection term breaks the bound.** The selection term enters as −N ln P̂_det(φ), which is biased the other way. The total objective is therefore not a bound, and its bias grows as N²·Var(P̂)/P̂² (see Q4).
- **Mitigations GW practice suggests:**
  - track σ²_lnL(φ) during training;
  - add a smooth taper (gwax/PixelPop style) or the Heinzel–Vitale correction factor;
  - when p_det(θ) is known analytically or by emulator, draw fresh samples from q_φ for the selection integral with a reparameterised estimator, as in learning-p-det's "dynamic injections". This avoids weight degeneracy.
- **Restrict claims to supported regions.** The Iampieri & Mastrogiovanni support-restriction caveat applies directly: flows extrapolate into regions with no injections or posterior samples. pastry should mask or report such regions (compare the GWTC-5 hatching heuristic in Q2).
- **Gibbs as joint sampling.** In pastry's discrete-bin case, joint sampling of (z_i, ρ) is the classic Dirichlet–multinomial data augmentation, with a conjugate Gibbs update. This is precisely what GW lacks: Mancarella & Gerosa needed about 500-dimensional HMC. pastry can offer collapsed or uncollapsed Gibbs as a Monte-Carlo-noise-free baseline when L_ik is given on grids.
- **SBI is out of scope.** Amortised SBI (NPE, Dingo-Pop) needs a forward simulator of the per-object data, which pastry's core use case lacks (per-object likelihoods are given). It is probably out of scope beyond citation.

### Gaps
- I could not confirm public code for Ruhe et al. (2022), Iampieri & Mastrogiovanni (2026), Li et al. (2026) or Dingo-Pop.
- I could not confirm that gwax is the released implementation of arXiv:2504.07197.
- I did not verify which density estimator Talbot & Thrane (2022) used (the abstract says only "density estimation methods").
- I did not read poplar's selection network design in depth.

## Q4. Monte Carlo accuracy of the sample-based likelihood

### Takeaway
The GW community has converged on a few standard practices:
- Compute the total variance of the estimated log-likelihood, σ²_lnL̂(Λ) = Σ_i Var(L̂_i)/L̂_i² + N²·Var(P̂_det)/P̂_det², at every hyperparameter point.
- Reject or taper points with σ²_lnL̂ > 1. This was used in GWTC-4 and GWTC-5 and replaced GWTC-3's per-integral N_eff cuts, N_eff,inj > 4N and N_eff,event > 10.
- Store the per-sample variances so results can be audited.

Heinzel & Vitale (2025) supply the theory and a package, `population-error`:
- the plug-in likelihood is biased, and a closed-form correction exists;
- even an unbiased likelihood gives a biased posterior through two-point covariances;
- total-variance thresholds are provably safe, but per-integral N_eff thresholds are not;
- information loss should be kept to Ê ≲ 0.2 bits.

Variance cuts are not innocuous. In GWTC-4 the choice of cut changed astrophysical conclusions, and it acted as the effective smoothness prior for B-splines.

### Cited Findings
**Foundational papers**
- Farr (2019), RNAAS 3, 66 (arXiv:1904.10879), gives "formulas for the accuracy to which a selection function must be measured via Monte-Carlo injections … The number of found injections scales linearly with the number of objects in the population". — [arXiv:1904.10879](https://arxiv.org/abs/1904.10879)
- The resulting rules are implemented in gwpopulation's `ResamplingVT` (N_eff > 4·N_events) and as the uncertainty-marginalised P̂_det·exp(−(3+N)/(2N_eff)). gwpopulation's docstring recommends against the latter: "it is not completely understood if this uncertainty marginalization is correct". — [gwpopulation vt.py](https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/vt.py)
- Essick & Farr (2022, arXiv:2204.00461) find "correlations between the estimators at nearby points in parameter space are crucial to the precision of the estimate". Approximate marginalisation that neglects these correlations "will either introduce a bias … or be more expensive", so "we therefore recommend that hierarchical inferences with empirically estimated target distributions use point estimates". — [arXiv:2204.00461](https://arxiv.org/abs/2204.00461)
- Talbot & Golomb (2023), "Growing Pains" (arXiv:2304.06138):
  - "Monte Carlo uncertainty in estimating the survey selection bias is the limiting factor in our ability to probe narrow population models and this will rapidly grow more problematic as the size of the observed population increases".
  - Keeping per-integral uncertainty constant as N grows "is insufficient and the actual scaling depends on the functional form".
  - They model the uncertainty as a Gaussian-process draw over parameter space.
  - They recommend these checks "be routinely performed for any population analysis", with scripts provided.
  - Heinzel & Vitale credit them with conjecturing the σ²_lnL̂ < 1 threshold.

  — [arXiv:2304.06138](https://arxiv.org/abs/2304.06138)

**Heinzel & Vitale (2025), "When (not) to trust Monte Carlo approximations for hierarchical Bayesian inference" (arXiv:2509.07221)**
- They "derive an error estimator Ê for quantifying the amount of information that is lost due to the Monte Carlo approximation and recommend that this error is limited to no more than Ê ≲ 0.2 bits". Ê is computable in post-processing.
- "the hierarchical likelihood estimator is biased but may be corrected". The corrected estimator is L̂_corr = ξ̂^{−N} exp(−½ N(N+1) σ̂²_ξ/ξ̂²) Π_i p̂(d_i|Λ) (their Eq. 24), which leaves residual bias at O(M⁻²). This has the "similar[ity] but opposite sign to the modifying term proposed [by] Farr".
- Posterior bias persists "even if the likelihood estimator is unbiased", through two-point covariance terms ⟨δ̂(Λ)δ̂(Λ′)⟩.
- "Log-likelihood variance thresholds are safe … [via] the Cauchy–Schwarz inequality". In contrast, "thresholds based on individual Monte Carlo integrals are not necessarily safe (as used in e.g., previous LVK population analyses)".
- "…However, they can be stricter than necessary": "simple models with strong assumptions can be much more stable to Monte Carlo uncertainty than those with weaker assumptions", and weak, many-parameter models "tend to require the strict log-likelihood variance threshold".
- Outlook: Monte Carlo cost "quickly becomes computationally intractable for weak models and large catalogs"; alternatives are density-estimate joint sampling, ML or analytic selection, and SBI.

— [arXiv:2509.07221](https://arxiv.org/abs/2509.07221)

**population-error** (https://github.com/jack-heinzel/population-error; docs https://population-error.readthedocs.io; PyPI `population-error` 0.2.3, 2026-08-07, first release 2025-09-08)
- Facts: MIT classifier, but no LICENSE file in the repo. JAX; depends on gwpopulation and bilby. 1 star; latest commit 2026-10-04 (the "ragged-posteriors" PR).
- API: `error_statistics(model_function, injections, event_posteriors, hyperposterior, vt_model_function=None, include_likelihood_correction=True, conversion_function, nobs, rate=False, event_counts=None)`. It returns error, precision and accuracy statistics and accepts ragged per-event sample lists, which are padded internally. There is also a `marginal_error_statistics` variant.
- It is already a dependency of PixelPop. A 2026-10-05 PixelPop commit fixed an "error statistic bug where T_live was not included for selection function component".

— [README](https://github.com/jack-heinzel/population-error); [statistics.py](https://github.com/jack-heinzel/population-error/blob/main/population_error/statistics.py); [PyPI](https://pypi.org/project/population-error/)

**LVK practice**
- GWTC-4.0 (App. A.1) gives σ²_lnL̂ = Σ_i σ²_L̂i/L̂_i² + N²σ²_ξ (and the rate-marginalised form with σ²_ξ/ξ̂²). "We adopt a threshold on σ²_lnL̂ of 1 … we ignore posterior samples with variances above this chosen threshold". The paper calls this conservative, since "we only require that the difference of log-likelihoods to be small", and notes that "for some models, a large region of hyperparameter space is removed by this threshold". — [arXiv:2508.18083](https://arxiv.org/abs/2508.18083)
- GWTC-4.0 (App. D.3) on GWTC-3 practice: GWTC-3 required "N_eff … greater than 4N_events for the sensitivity injections … and greater than 10 for every event". Switching cuts changes the inferred χ_p distribution (whether μ_p is away from zero, its width, and the χ_eff–χ_p correlation). These claims "are not necessarily astrophysical in origin. Rather, they are driven by regions of the parameter-space that our events and sensitivity injections allow us reliably probe". The paper attributes part of this to the PE prior: "a uniform and isotropic spin prior … induces a χ_p prior that goes to 0 at χ_p = 0, 1", which yields small N_eff there. — [arXiv:2508.18083](https://arxiv.org/abs/2508.18083)
- GWTC-5.0: "we require a maximum variance of 1 on the population likelihood estimator; for some models, we also show results with a more relaxed variance cut … implemented as a sharp or smoothly-tapered cutoff". — [arXiv:2605.27226](https://arxiv.org/abs/2605.27226)
- GWTC-4.0 (App. C.1): for B-splines, "the limits imposed on the likelihood uncertainty was the main driver of smoothness". — [arXiv:2508.18083](https://arxiv.org/abs/2508.18083)
- Sensitivity (injection) data products for GWTC-4.0 are on Zenodo: O4a sensitivity estimates (doi:10.5281/zenodo.16740117) and cumulative O1–O4a estimates (doi:10.5281/zenodo.16740128), as cited in the GWTC-4.0 methods paper. I did not inspect their contents. — [arXiv:2508.18082](https://arxiv.org/abs/2508.18082); [GWOSC GWTC-4.0](https://gwosc.org/GWTC-4.0/)

**How packages report or enforce accuracy**

| Package | Variance computed | Enforcement |
|---|---|---|
| gwpopulation | per-event and selection variances; `ln_likelihood_and_variance`, `generate_extra_statistics` | hard `maximum_uncertainty` (−inf); `ResamplingVT` N_eff > 4N |
| GWInferno | per-event n_eff and variances, as NumPyro deterministics | `min_neff_cut` (default), `max_variance_cut` (< 1), Farr correction |
| icarogw | variance with finite-population factors | `neffPE` = 20, `neffINJ` = 4N, or a variance threshold |
| PixelPop | — | smooth taper of variance and N_eff ("to retain autodifferentiability") |
| gppop | — | soft −softplus penalty; CLI `--maximum_uncertainty=1.0 --exponent=-75` |
| Callister AR | — | soft log-logistic N_eff penalties (exponent −30) |
| gwkokab | — | quadratic taper 100(σ² − σ²_thr)² above the threshold |
| gwax | — | `taper(variance)` hook in the variational loss; ESS reporting |

— [gwpopulation](https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/hyperpe.py); [GWInferno](https://github.com/FarrOutLab/GWInferno/blob/main/gwinferno/pipeline/analysis.py); [icarogw](https://github.com/simone-mastrogiovanni/icarogw/blob/main/icarogw/likelihood.py); [PixelPop](https://git.ligo.org/jack.heinzel/pixelpop/-/blob/main/pixelpop/models/probabilistic.py); [gppop](https://github.com/AnaryaRay1/gppop/blob/main/src/gppop/rates.py); [AR](https://github.com/tcallister/autoregressive-bbh-inference/blob/main/code/autoregressive_mass_models.py); [gwkokab](https://github.com/kokabsc/gwkokab); [gwax](https://github.com/mdmould/gwax/blob/main/gwax/likelihood.py)

### Inferences
- **Grid inputs have no Monte Carlo noise.** When pastry's per-object likelihoods L_ik are given on the same bins (grids), the data term has no Monte Carlo noise at all. Only sample-based inputs, sample-based selection and smooth (flow) models need these diagnostics. That is a genuine advantage over GW tooling, and the docs should say so.
- **Closed-form variance for binned sample inputs.** For binned models fed by posterior samples, p̂_i(ρ) = Σ_k ρ_k·m_ik, with m_ik = (1/S)Σ_s 1[θ_s∈k]/(π_int(θ_s)Δ_k). Because bin indicators are disjoint, the variance is (1/S)[Σ_k ρ_k²·q_ik − p̂_i²], with q_ik = (1/S)Σ_s 1[θ_s∈k]/(π_int(θ_s)Δ_k)². So σ²_lnL̂(ρ) is a cheap quadratic form in ρ that can be evaluated alongside the likelihood. The same holds for injection-based selection.
- **The interim prior must cover the population support.** GWTC-4's χ_p episode shows that an interim prior vanishing where the population has mass creates exploding weights. pastry should check, warn on, or reject input interim priors with near-zero density inside the bin domain.
- **Defaults pastry should adopt:**
  - compute σ²_lnL̂ by default;
  - offer a hard cut, a smooth taper and Heinzel–Vitale's bias correction as options;
  - report the fraction of posterior or optimisation mass affected;
  - provide a post-hoc Ê (bits) diagnostic, possibly calling or porting `population-error`'s formulas.
- **IWAE training.** For flows, add the same variance and taper machinery to the training objective (as gwax does), and report effective sample sizes per object and for selection.
- **N_eff cuts.** Per-object N_eff cuts should not be the primary safety criterion. Heinzel & Vitale show they are not sufficient; the total log-likelihood variance is.

### Gaps
- I did not extract Ê's exact formula from the paper. Only the summary, the correction term and the bounds are captured above.
- I did not verify Talbot & Golomb's numerical guidance on required sample counts as a function of N.
- I did not check whether gwpopulation implements the Heinzel–Vitale correction. No such option was seen in `hyperpe.py` or `vt.py` as of the 2026-07 clone.

## Q5. API/UX lessons (model, prior and selection specification; results; pain points)

### Takeaway
GW population software uses three API styles:
- **Plain functions plus an external sampler** (gwpopulation + bilby). A model is a function of a data dict and named hyperparameters. This is the easiest to learn and the most widely adopted.
- **Probabilistic-programming models** (GWInferno, PixelPop, gppop, the AR code, gwkokab). These are NumPyro or Pyro models with a `factor(...)` likelihood. They are flexible and support gradients and GPUs, but users must write PPL code.
- **Config-file and CLI pipelines** (gwpopulation_pipe `.ini` + HTCondor, GWInferno YAML, gwkokab's 20+ console scripts and config templates, FIGARO CLIs, Vamana prior files).

The repeated pain points are:
- ragged per-event sample counts (until 2026 most code downsampled every event to the smallest count);
- specifying the interim (PE) prior and its Jacobians;
- JAX/CUDA installation and x64 configuration;
- improper smoothing priors that need initialisation, and poorly identified smoothing hyperparameters;
- variance cuts that silently remove parameter space;
- non-PyPI installs and heterogeneous licences.

popsummary is the de facto result schema.

### Cited Findings
**How models and priors are specified**
- **gwpopulation** models are functions `f(dataset, **hyperparameters) -> density` (or classes with `variable_names`), combined with `bilby.hyper.model.Model([...])`. Priors are a `bilby.core.prior.PriorDict`. Sampling goes through `bilby.run_sampler(likelihood=JittedLikelihood(likelihood), priors=priors, sampler="dynesty")`. The backend is a global switch, `gwpop.set_backend("jax")`. — [models/__init__.py](https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/models/__init__.py); [examples/GWTC3.ipynb](https://github.com/ColmTalbot/gwpopulation/blob/main/examples/GWTC3.ipynb)
- **gwpopulation_pipe** `.ini` files set the run directory, the VT file and thresholds (`vt-ifar-threshold`, `vt-snr-threshold`), `vt-function=injection_resampling_vt`, the backend and GPU requests. Two features stand out:
  - Model enumeration: "A tensor product will be done with the models listed here, i.e., if you list two mass models, two spin magnitude models, one spin tilt model, and two redshift models, a total of 8 analysis jobs will be set up".
  - Interim prior: data collection needs the PE prior declared (`mass-prior=flat-detector-components`, `distance-prior=comoving`) so the `prior` column can be built.

  — [example/gwtc3-bbh.ini](https://git.ligo.org/RatesAndPopulations/gwpopulation_pipe/-/blob/master/example/gwtc3-bbh.ini)
- **GWInferno** has two levels: a YAML `models:` block naming `gwinferno.numpyro_distributions.*` classes with per-hyperparameter `prior: numpyro.distributions.X` and `prior_params`, run by `gwinferno_run_from_config.py config.yml`; or a user Python file defining a NumPyro `model`. The docs concede that "Defining models with the method above may not be suitable for more complex model declarations". — [docs/example_usage.rst](https://github.com/FarrOutLab/GWInferno/blob/main/docs/example_usage.rst)
- **gwkokab** tells users to subclass `numpyro.distributions.Distribution`, implementing `log_prob` (and `sample` for simulation). It ships CLI entry points (`discrete_*`, `analytical_gwalk_*`, `gwk_numpyro_cfg_template`, `gwk_flowMC_cfg_template`, `gwk_report`, synthetic-data generators). Its FAQ section "GWKokab is slow on GPU" lists JAX/XLA environment variables and compilation-cache settings to tune. — [FAQs.md](https://github.com/kokabsc/gwkokab/blob/main/docs/source/FAQs.md); [pyproject.toml](https://github.com/kokabsc/gwkokab/blob/main/pyproject.toml)
- **FIGARO** is CLI-first (`figaro-density`, `figaro-hierarchical`). Selection enters as a user `.py` file defining `selection_function(x)`, with the warning "check that your selection function does not return zeros". A probit coordinate change to unbounded space is "completely user-transparent", and outputs are JSON "draws" plus plots. — [quickstart.md](https://github.com/sterinaldi/FIGARO/blob/main/docs/source/quickstart.md)
- **Vamana** uses plain-text prior files with columns `NAME DEFINITION DISTRIBUTION Per_component arg1 arg2 Proposal pargs`, for example Dirichlet mixture weights with a `dirichlet_joint` proposal. — [README](https://github.com/vaibhavtewari/vamana)
- **PixelPop** examples run four phases: data loading and cleaning; `setup_probabilistic_model` (an "initialization is required since the ICAR model is an improper prior"); `inference_loop`, which "prints out mid-run chain diagnostics" (worst r-hat and n_eff sites); and `create_popsummary`. — [examples/README.md](https://git.ligo.org/jack.heinzel/pixelpop/-/blob/main/examples/README.md)

**How results are returned and plotted**
- **popsummary** `PopulationResult` stores hyperparameter samples, reweighted event samples, reweighted injections, fair population draws and rates on grids (posterior and prior groups), plus metadata and LaTeX labels. `tex_result()` gives credible intervals. It is used by gwpopulation_pipe and PixelPop. — [popsummary](https://git.ligo.org/RatesAndPopulations/popsummary)
- **gwpopulation** returns bilby `Result` objects (corner plots). `generate_extra_statistics` adds per-sample diagnostics (per-event ln BF, variances, selection) and `posterior_predictive_resample` gives population-informed event posteriors. — [hyperpe.py](https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/hyperpe.py)
- **LVK figures** show posterior predictive distribution (PPD) bands, and GWTC-5 hatches regions "where we have under 1% of PE samples … as a heuristic proxy for parts of parameter space where the models may not be actively constrained by the data". — [arXiv:2605.27226](https://arxiv.org/abs/2605.27226)

**Documented pain points**
- **Ragged sample counts.**
  - PixelPop's migration note: the common loader "downsamples every event to the smallest event's sample count … That discards samples from well-measured events". An "earlier attempt [with] a list of per-event dicts … defeats JAX vectorization and is slow. It was reverted". The fix pads to a rectangular `(Nobs, NPE)` array with padded rows given `prior=+inf` (zero weight), and divides Monte Carlo means and variances by each event's real count.
  - gwpopulation added "different numbers of samples per event (#142)" in July 2026, using cumulative-sum segment means.
  - gwax advertises a likelihood "that avoids single-event downsampling".
  - population-error accepts ragged lists.

  — [ragged_samples_migration.md](https://git.ligo.org/jack.heinzel/pixelpop/-/blob/main/docs/ragged_samples_migration.md); [gwpopulation hyperpe.py](https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/hyperpe.py); [gwax README](https://github.com/mdmould/gwax)
- **Interim prior and coordinates.**
  - The PE prior must be declared and transformed (detector- vs source-frame masses, distance vs redshift).
  - FIGARO exposes `likelihood=True` ("resample to get likelihood samples") and `keep_dVdz` ("do not remove dV/dz from the likelihood (only if the population model do NOT include it)").
  - GWTC-4 traced χ_p artefacts to a PE prior vanishing at χ_p = 0, 1.

  — [FIGARO load.py](https://github.com/sterinaldi/FIGARO/blob/main/figaro/load.py); [arXiv:2508.18083](https://arxiv.org/abs/2508.18083)
- **Installation and hardware.**
  - GWInferno, icarogw and gppop are not on PyPI.
  - GWInferno's README carries long CUDA/cuDNN-specific instructions.
  - icarogw toggles the GPU with a `config.py` file containing `CUPY=False` in the working directory.
  - gwpopulation enables JAX x64 globally.

  — [GWInferno README](https://github.com/FarrOutLab/GWInferno); [icarogw README](https://github.com/simone-mastrogiovanni/icarogw); [gwpopulation backend.py](https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/backend.py)
- **Smoothing hyperparameters and cuts.**
  - B-spline τ "consistently railed against prior boundaries" and was fixed by hand.
  - The variance threshold removes "a large region of hyperparameter space" for some models.

  — [arXiv:2508.18083](https://arxiv.org/abs/2508.18083)
- **Licences** are heterogeneous: MIT (gwpopulation, bilby, GWInferno, FIGARO, popsummary, PixelPop, Vamana, gwax, pdet); Apache-2.0 (gwkokab); GPL (gppop, poplar GPLv3, gwcosmo, CHIMERA GPL-3+); EUPL-1.2 (icarogw); CC BY 4.0 (gwpopulation-fullpop); no licence file (gravpop, Hierarchical-Bayesian-Flows; population-error declares MIT only via a classifier). — sources as in Q1 and Q3.

**Documentation and test quality** (counted from the clones on 2026-10-22)

| Package | Test files | Test lines | Docs and CI |
|---|---|---|---|
| gwpopulation | 13 | ≈2.4k | codecov; docs site; JOSS |
| GWInferno | 13 | ≈1.6k | CI and docs badges |
| gwkokab | 63 | ≈17k | readthedocs + CI |
| PixelPop | 14 | ≈3.6k | — |
| gravpop | 8 | ≈1k | — |
| FIGARO | 1 | 54 | readthedocs and JOSS paper, but a minimal test suite |
| gppop | 0 (empty `tests/`) | — | — |
| gwpopulation_pipe, popsummary, icarogw, Vamana, gwax | no test directory in the clone | — | — |
| AR and learning-p-det | paper-reproduction repos | — | docs pages, no tests |

gwpopulation's docs "guide" pages are just `automodule` directives over module docstrings.
— [gwpopulation](https://github.com/ColmTalbot/gwpopulation); [GWInferno](https://github.com/FarrOutLab/GWInferno); [gwkokab](https://github.com/kokabsc/gwkokab); [PixelPop](https://git.ligo.org/jack.heinzel/pixelpop); [FIGARO](https://github.com/sterinaldi/FIGARO); [gppop](https://github.com/AnaryaRay1/gppop)

### Inferences
- **Model contract.** pastry's emcee/scikit-learn-like ease should copy gwpopulation's lowest-friction contract (a model is a plain function or object of arrays and named parameters, so users never write PPL code for standard cases). It should avoid two things seen in GW tooling: global backend state (`set_backend`, `config.py` flags), and HPC-pipeline-first design. Config and CLI layers can come later.
- **Interim prior as a first-class input.** Make it explicit, with a validated density over the binning coordinates, automatic Jacobian handling for coordinate changes, and a coverage check. In GW this is the most error-prone, least visible step.
- **Ragged inputs.** Accept ragged per-object sample sets natively, either as a padded array with counts or as CSR-style segment sums. Users should never be forced to downsample, and padding must not leak into the counts.
- **Result object.** Store hyperparameter or ρ samples, per-object population-informed posteriors ("reweighted event samples"), predictive draws, rates or densities on grids, per-sample diagnostics (σ²_lnL̂, ESS) and metadata. Export to HDF5 (borrowing popsummary's layout) and to ArviZ `InferenceData`, so standard convergence diagnostics work.
- **Diagnostics by default.** Show r-hat, ESS, the Monte Carlo variance and the fraction of mass cut, and flag low-data-support bins (GWTC-5 hatching). Several GW codes print these mid-run, which users rely on.
- **Hardware.** Default to CPU NumPy/JAX with optional GPU, keep the core dependency-light, and put flow and GPU extras behind optional installs. Every widely used GW package is either bilby-coupled or JAX/CUDA-heavy, and install friction is a recurring complaint.

### Gaps
- GitHub issue trackers could not be searched from this session (issue search returned no results), so user-reported pain points are inferred from code, docs and changelogs rather than from issue threads.
- I did not verify the quality of the rendered documentation sites (only repository docs sources were read).

