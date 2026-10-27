# The software landscape

This page compares the tools closest to binference's scope, across fields, and records
what binference took from the design of other scientific Python packages. The field
pages describe each tool in its context; the {doc}`index <index>` gives the date of the
survey.

## Comparison

"Generic input" means an arbitrary per-object likelihood matrix, or posterior samples
with a declared interim prior. No maintained, general-purpose Python package found in
the census takes such input and returns population fractions with sampled uncertainty
and smoothness, selection and interim-prior extensions. The maintained tools restrict
their inputs to one field's data products, stop at point estimates, or are written in R
or Julia; the tools that implement the full hierarchical model with sampled uncertainty
are paper code.

| tool (field) | language, licence, status | per-object input | engine; population uncertainty | selection · smoothness · interim prior |
|---|---|---|---|---|
| [RAIL `VarInfStackSummarizer`](https://github.com/LSSTDESC/rail_base/blob/main/src/rail/estimation/algos/var_inf.py) (photo-$z$) | Python, MIT; pz-rail-base 2.0.11, September 2026 | PDFs evaluated at grid points | mean-field VB; Dirichlet draws that under-cover | none · none · none |
| [Photoz-tools](https://github.com/ixkael/Photoz-tools), [BHM-nz](https://github.com/Harry45/BHM-nz) (photo-$z$) | Python notebooks, MIT; last push 2020 and 2023 | binned template likelihoods | exact two-block Gibbs; posterior samples | selection variable · not built · not applicable |
| [CHIPPR](https://github.com/aimalz/chippr) (photo-$z$) | Python 2, MIT; "stale" prototype | gridded posteriors and interim prior | emcee; posterior samples | none · squared-exponential hyperprior · yes |
| [frankenz](https://github.com/joshspeagle/frankenz) (photo-$z$) | Python, MIT; last push 2018 | $(N,K)$ array of PDFs | pairwise Metropolis (invalid) and exact Gibbs | none · user log-prior · no |
| [Starlight](https://github.com/ixkael/Starlight) (colour–magnitude) | Python/Cython, MIT; last commit 2017 | precomputed per-kernel likelihoods | Gibbs over 1.4 million stars; posterior samples | none · none · not applicable |
| [extreme-deconvolution](https://github.com/jobovy/extreme-deconvolution), [pygmmis](https://github.com/pmelchior/pygmmis) (densities) | C and Python; XD maintained but not on PyPI; pygmmis 1.2.3, 2022 | Gaussian mean and covariance | EM maximum likelihood; none | pygmmis: known completeness · covariance regulariser · no |
| [StarFormationHistories.jl](https://github.com/cgarling/StarFormationHistories.jl) (Hess diagrams) | Julia, MIT; v2.0.0, October 2026 | binned counts and a template matrix | Poisson likelihood ratio; MAP with Hessian, HMC, MCMC | via templates · hierarchical metallicity relations · not applicable |
| [exopop](https://github.com/dfm/exopop) (occurrence rates) | Python, MIT; 2014 paper code | posterior samples | Poisson process, elliptical slice sampling; posterior samples | completeness · GP · yes |
| [PosteriorStacker](https://github.com/JohannesBuchner/PosteriorStacker) (astronomy) | Python, GPL/AGPL (inconsistent); 0.6.4, September 2026 | one-dimensional samples under a flat prior | flat-Dirichlet histogram by nested sampling; samples and evidence | none · none · flat prior only |
| [MRExo](https://github.com/shbhuk/mrexo) (exoplanets) | Python, GPL-3; 1.1.5, 2023; no tests | values with asymmetric errors, mapped to $N\times K$ | simplex MLE; bootstrap | none · basis degree by cross-validation or AIC · not applicable |
| [gwpopulation](https://github.com/ColmTalbot/gwpopulation) (GW) | Python, MIT; 1.3.1, December 2025; NumPy/CuPy/JAX | samples with a `prior` column, and injections | any bilby sampler; samples and Monte Carlo variance | injections · spline nodes · yes |
| [PixelPop](https://git.ligo.org/jack.heinzel/pixelpop), [gppop](https://github.com/AnaryaRay1/gppop) (GW) | JAX/NumPyro, MIT; PyTorch/Pyro, GPL; gppop not on PyPI, no tests | samples mapped to per-bin weights ($L_{ik}$) | NUTS on GPUs over $10^4$ (ICAR) to $8\times10^4$ (Kronecker GP) bins | injections · ICAR or GP · yes |
| [RooUnfold](https://gitlab.cern.ch/RooUnfold/RooUnfold) (HEP) | C++/ROOT, BSD-3; 3.1.0, conda-forge only | one shared response matrix | IBU, SVD, Tikhonov, GP; propagated errors, toys | efficiency · Tikhonov · simulation prior |
| [PyUnfold](https://github.com/jrbourbeau/pyunfold), [PyFBU](https://github.com/pyFBU/fbu) (HEP) | Python; frozen 2018; unmaintained (Theano) | response matrix | IBU with per-bin errors; NUTS posterior | efficiency · spline or Tikhonov · not applicable |
| [mixsqp](https://cran.r-project.org/package=mixsqp), [REBayes](https://cran.r-project.org/package=REBayes) (empirical Bayes) | R/C++; REBayes needs MOSEK | any $N\times K$ likelihood matrix | convex NPMLE; none | none · post-hoc kernel smoothing · no |
| [ashr](https://cran.r-project.org/package=ashr) (empirical Bayes) | R | any matrix and a Dirichlet prior | MAP, or mean-field VB (`mixVBEM`) | none · unimodal dictionary · no |
| [deconvolveR](https://github.com/bnaras/deconvolveR), [Empirikos.jl](https://github.com/nignatiadis/Empirikos.jl) (empirical Bayes) | R and Julia; both active in 2026 | design matrices; likelihood types | log-spline with delta-method errors; NPMLE with F-localisation and AMARI intervals | truncation · log-spline or convex prior classes · no |
| [npeb](https://github.com/jake-soloff/NPEB), [mixsqpx](https://pypi.org/project/mixsqpx/) (empirical Bayes) | Python; alpha | any kernel matrix | cvxpy/MOSEK; JAX mix-SQP; none | none |
| [QuaPy](https://github.com/HLT-ISTI/QuaPy) (ML quantification) | Python, BSD-3; 0.2.3, October 2026; 16 test modules | classifier outputs | EM; NUTS; bootstrap regions; temperature calibration | none · none · division by the training prior |
| [rubias](https://github.com/eriqande/rubias) (fisheries genetics) | R/C++, CC0; archived on CRAN February 2026 | genotypes and reference baselines | Gibbs, bias-corrected bootstrap, baseline resampling; traces | none · none · not applicable |
| [salmon](https://github.com/COMBINE-lab/salmon), [kallisto](https://github.com/pachterlab/kallisto) (RNA-seq) | Rust and C++, BSD | reads in equivalence classes | EM/VBEM; bootstrap or Gibbs replicates | none |
| [density-deconvolution](https://github.com/bayesiains/density-deconvolution), [NEB](https://github.com/MaximeVandegar/NEB), [neuralG](https://github.com/shijiew97/neuralG) (neural) | PyTorch research code, 2020–2026; several unlicensed; no tests | Gaussian noise, Monte Carlo samples or grids | marginal-likelihood training; point estimates | none · none · implicit flat prior |

For comparison, binference is a Python package (NumPy and SciPy core) that takes
likelihood matrices (dense or sparse), posteriors and posterior samples with an interim
prior, Gaussian measurements, likelihood functions, and binned counts with a response;
samples the posterior exactly with NUTS or Gibbs, or approximates it with Laplace, VB
and bootstrap engines whose results are flagged when approximate; and supports selection on the truth or on the
measurement (fixed, or inferred from injections), Dirichlet-family and logistic
smoothness priors, and rates ({doc}`../guides/index`).

## Notes on the landscape

- **Durability is the recurring failure.** CRAN archived rubias because the
  maintainer's email bounced; PyFBU died with Theano; several neural repositories carry
  no licence; and emcee kept its core maintainable by moving MPI pools and parallel
  tempering into other packages because "there aren't enough resources to maintain"
  them.
- **Tests check that code runs, not that intervals cover.** RAIL's summarizer tests are
  end-to-end smoke tests, gppop's test directory is empty, FIGARO's suite is 54 lines,
  and deconvolveR and npeb have no tests.
- **Uncertainty is where the tools differ most.** Most engines converge to the same
  maximum; their error bars mean different things. Mean-field VB under-covers, convex
  NPMLE solvers return none, bootstrap regions measure an estimator's variability, and
  Gibbs and NUTS sample the exact posterior ({doc}`simplex-priors` compares the engines).
- **Scale is limited by memory**, as for the dense likelihood matrix of 1.36 million Gaia
  stars on a $100\times100$ grid ({doc}`empirical-bayes`). Collapsing identical
  rows (salmon's equivalence classes, Empirikos's multiplicities), sparse storage
  (npeb's log-probability threshold) and computing each object's likelihood once on
  fixed bins (Starlight) are the established remedies; binference offers the first two
  as `lk.compress()` and sparse `Likelihoods` ({doc}`../guides/performance`).

## Lessons from scientific Python packages

The interface follows patterns that users of other inference packages praise:

- **emcee**: plain inputs and a sampler object that can be run again to continue
  {cite:p}`foremanmackey2013,foremanmackey2019`. binference's engines resume: a second
  `engine.run(n)` continues the chains without a new warm-up.
- **scikit-learn**: hyperparameters in the constructor, `fit` returning `self`, learned
  attributes with a trailing underscore, `get_params` and `set_params` {cite:p}`buitinck2013`. {class}`~binference.PopulationEstimator` follows these conventions without
  depending on scikit-learn. Full compliance does not fit, because the input is a
  likelihood matrix rather than a feature matrix.
- **dynesty and nautilus**: configured objects instead of piles of keyword arguments,
  and priors declared as objects. binference's priors, selections and responses are
  objects passed to one `infer` call.
- **UltraNest**: output that says which criteria were met. `result.check()` lists the
  diagnostics that failed and, where it can, a remedy ({doc}`../guides/results`).
- **ArviZ 1.x**: results that other tools read. `result.to_datatree()` returns an xarray
  `DataTree` in ArviZ's layout, and `result.samples` stays a plain array.
- **SPEC 7** {cite:p}`vanderwalt2023`: one `rng` argument that accepts a seed or a `numpy.random.Generator`, with
  a separate stream per chain; binference rejects legacy `RandomState` objects.
- **tinygp**: expose the model as functions that other samplers can call.
  {class}`~binference.PopulationModel` provides the log posterior and its gradient in
  unconstrained coordinates and a prior transform for nested samplers, and
  `PopulationResult.from_samples` wraps draws from any sampler.

Validation follows the statistical-software literature: exact posteriors for small
problems, Geweke's "getting it right" test for the Gibbs sampler, and simulation-based
calibration {cite:p}`talts2018` that ranks the joint log-likelihood as well as each
fraction, because parameter ranks alone "could never detect large classes of problems
including when the posterior is equal to the prior" {cite:p}`modrak2025`. binference
ships the exact posterior, simulation-based calibration and coverage as user-facing
tools ({doc}`../guides/checking`).
