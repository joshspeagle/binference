# The software landscape

Which tools come closest to binference, across fields? This page compares them. It
also lists the patterns binference took from other scientific Python packages. The
field pages describe each tool in its context. The {doc}`index <index>` gives the date
of the survey.

## Comparison

"Generic input" here means an arbitrary per-object likelihood matrix, or posterior
samples with a declared interim prior. The census found no maintained, general Python
package that takes such input and samples the population's posterior. The maintained
tools take one field's data products, stop at point estimates, or are written in R or
Julia. The tools that sample the full hierarchical model are paper code.

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

The {doc}`guides <../guides/index>` describe binference's own inputs, engines, priors and
selection.

## Notes on the landscape

- **Durability is the recurring failure.** CRAN archived rubias because the maintainer's
  email bounced. PyFBU died with Theano. Several neural repositories carry no licence.
  emcee moved MPI pools and parallel tempering into other packages, because "there
  aren't enough resources to maintain" them.
- **Tests check that code runs, not that intervals cover.** RAIL's summarizer tests are
  end-to-end smoke tests. gppop's test directory is empty, and FIGARO's suite is 54
  lines. deconvolveR and npeb have no tests.
- **Uncertainty is where the tools differ most.** Most engines converge to the same
  maximum, but their error bars mean different things. Mean-field VB under-covers.
  Convex NPMLE solvers return none. Bootstrap regions measure an estimator's
  variability. Gibbs and NUTS sample the exact posterior ({doc}`simplex-priors`).
- **Memory limits scale.** The established remedies collapse identical rows (salmon,
  Empirikos.jl), store sparse matrices (npeb), or compute each likelihood once on fixed
  bins (Starlight). binference offers the first two
  ([performance guide](../guides/performance.md)).

## Patterns from scientific Python packages

| package | pattern | in binference |
|---|---|---|
| emcee {cite:p}`foremanmackey2013,foremanmackey2019` | plain inputs; a sampler that runs again to continue | a second `engine.run(n)` continues the chains without a new warm-up |
| scikit-learn {cite:p}`buitinck2013` | hyperparameters in the constructor, `fit` returning `self`, `get_params` | {class}`~binference.PopulationEstimator`, without a scikit-learn dependency ([scikit-learn style](../guides/engines.md#scikit-learn-style)) |
| dynesty, nautilus | configured objects, priors declared as objects | priors, selections and responses passed to one `infer` call |
| UltraNest | output that says which criteria were met | `result.check()` lists failed diagnostics and remedies |
| ArviZ 1.x | results that other tools read | `result.to_datatree()` in ArviZ's layout |
| SPEC 7 {cite:p}`vanderwalt2023` | one `rng` argument, a seed or a `Generator` | a separate stream per chain; legacy `RandomState` rejected |
| tinygp | the model as functions other samplers can call | {class}`~binference.PopulationModel`; `PopulationResult.from_samples` wraps outside draws |

binference's validation uses exact posteriors, Geweke's "getting it right" test and
simulation-based calibration {cite:p}`talts2018`. Its calibration ranks the joint log-likelihood as well as each fraction.
Parameter ranks alone "could never detect large classes of problems including when the
posterior is equal to the prior" {cite:p}`modrak2025`. The
{doc}`validation guide <../guides/validation>` describes binference's tools.
