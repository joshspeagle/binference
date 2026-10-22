# Empirical-Bayes deconvolution (g-modeling), NPMLE and measurement-error deconvolution software

(Status: COMPLETE draft, 2026-10-22 — all five questions filled. Metadata gathered 2026-10-22 from CRAN/PyPI JSON metadata, GitHub search (star counts), shallow git clones and CRAN reference manuals stored in the scratchpad `eb/` folder. "Last commit" dates come from those clones. Code-level claims cite the file on GitHub. Items marked (UNVERIFIED) or (INFERENCE) need checking.)

## Q1. Software in R, Python and Julia: what each accepts, and which take an arbitrary per-observation likelihood matrix L_ik

### Takeaway
The mature reference implementations are R packages: REBayes (interior point via commercial MOSEK; generic `KWDual(A, d, w)` takes any n×m likelihood matrix), mixsqp (open-source SQP; `mixsqp(L, w)` takes any n×m likelihood matrix, `log = TRUE` allowed), ashr (`mixEM`/`mixSQP`/`mixVBEM(matrix_lik, prior)` take any n×k likelihood matrix *plus a Dirichlet prior*, which is exactly pastry's baseline posed as MAP or VB), deconvolveR (Efron's smooth log-spline g with delta-method SEs; generic P/Q/y interface, homoscedastic in practice) and ebnm (a unified Gaussian-only front end). The only Julia package with real inference for g is Empirikos.jl (NPMLE via JuMP conic solvers plus F-localisation/AMARI confidence intervals, with new likelihoods added through multiple dispatch). Python is thin: npeb (Gaussian, heteroscedastic, multivariate NPMLE, cvxpy/MOSEK, alpha, last release 2024) and mixsqpx (a 2025–26 JAX port of mix-SQP, alpha, Python 3.12 only) are the only PyPI packages found. No Python package returns uncertainties for g, and none accepts per-object posterior samples or interim-prior-weighted posteriors. Those features would set pastry apart.

### Cited Findings

#### Summary matrix (inputs → does it accept a generic L_ik?)
| Package (lang) | Generic L_ik input? | Native likelihoods | Engine | UQ on g | Status (Oct 2026) |
|---|---|---|---|---|---|
| REBayes (R) | Yes: `KWDual(A, d, w)` | Gaussian (hetero σ), t, binomial, Poisson, gamma/variance, Weibull, Gompertz, bivariate, etc. | MOSEK interior point (dual) | None built-in (smoothing only) | CRAN 2.60, 2026-03-01; needs MOSEK |
| mixsqp (R/C++) | Yes: `mixsqp(L, w, x0, log)` | none (solver only) | SQP + active set + low-rank (tSVD) | Returns Hessian of objective only | CRAN 0.3-54, 2023-12-20 |
| ashr (R) | Yes: `mixEM/mixSQP/mixIP/mixVBEM(matrix_lik, prior)` | normal, t, binomial, Poisson, logF, normal-mixture errors; unimodal priors | mix-SQP (default), IP, SQUAREM-EM, VB-EM | Variational Dirichlet posterior on π (`mixVBEM`) | CRAN 2.2-63, 2023-08-21; GitHub dev 2.2-69, 2026-05-04 |
| deconvolveR (R) | Partly: user `P` (n×m), `Q`, `y` | Poisson, Normal (unit-variance z-scores, binned), Binomial (per-object) | Penalised MLE (`nlm`) on log-spline g | Delta-method SE of g and G, bias of g | CRAN 1.2-2, 2026-08-23 |
| ebnm (R) | No (x, s Gaussian only) | Gaussian, heteroscedastic s | wraps mixsqp/ashr/REBayes/deconvolveR | Per-θ posterior only; no UQ on ĝ | CRAN 1.1-42, 2025-10-10 |
| nspmix (R) | Via user S3 family (`logd`, `gridpoints`, ...) | normal, Poisson, + user families | CNM (constrained Newton, support reduction) | None | CRAN 2.0-0, 2025-05-10 |
| Empirikos.jl (Julia) | Via new `EBayesSample` type (multiple dispatch); `MixturePriorClass` with arbitrary components | Normal (hetero), binomial, Poisson, χ², noncentral t, folded normal, truncated, ... | JuMP conic (MOSEK or open-source Hypatia) | F-localisation and AMARI CIs for functionals of g | v0.6.5, last commit 2026-08-30 |
| npeb (Python) | Module fns `solve_weights_cvx(K)`, `solve_weights_mosek(K)` take any n×m kernel | d-dim Gaussian, per-object precision matrices; Poisson | cvxpy or MOSEK Fusion + EM on atoms | None | PyPI 0.0.2, 2024-02-05; alpha |
| mixsqpx (Python/JAX) | Yes (mix-SQP port; `mixsolve`) | none (solver only) | mix-SQP in JAX (jit-able) | None | PyPI 0.2.1a3, 2026-05-21; alpha |
| decon (R) | No | Gaussian/Laplace errors, hetero σ allowed | Deconvolution kernel + FFT | None | ARCHIVED on CRAN (last 1.3-4, 2021) |
| deamer (R) | No | known error density / pure-noise sample / replicates | Adaptive projection (penalised contrast) | None | ARCHIVED (1.0, 2012) |

#### Feature-support matrix for pastry's requirements
Sources: the per-package findings below. Q4 has the scale evidence. "No" means no documented support found.

| Package | Heteroscedastic per-object likelihoods | Interim-prior correction | Selection effects | Smoothness/shape priors | Scale evidence | GPU | Minimal usage |
|---|---|---|---|---|---|---|---|
| REBayes | Yes (any A in `KWDual`; σ vector in `GLmix`) | No | No general mechanism found | `KWsmooth` post-hoc kernel smoothing; `BDGLmix` Efron log-spline (homoscedastic only); `medde` TV/log-concave (density estimation) | n = 10^5, m = 10^4: ~73 min; reported failure at n = 7×10^4, m = 10^4 | No (MOSEK multithreaded) | `GLmix(x, v = 300, sigma = s)` |
| mixsqp | Yes (L) | No | No | None (pure MLE) | n ≈ 2×10^6, m ≈ 10^2 in ~10 s (Julia version) | No (multithreaded BLAS only) | `mixsqp(L, w)` |
| ashr | Yes (`matrix_lik`; or betahat/se with normal/t/binomial/Poisson likelihoods) | No | No | Unimodality via a fixed dictionary; Dirichlet penalty | — | No | `ash(betahat, sebetahat, mixcompdist = "normal")` |
| deconvolveR | Partly (Binomial per-object rows; user P) | No | Zero-truncation "thinning" correction (Poisson) | Log-spline exponential family + norm penalty | Moderate n (p ≈ 6 parameters, `nlm`) | No | `deconv(tau, X, family = "Normal", pDegree = 6)` |
| ebnm | Gaussian σ_i only | No | No | Many families (unimodal, scale mixtures, deconvolver, ...) | ~linear in n up to 10^6 | No | `ebnm(x, s, prior_family = "npmle")` |
| nspmix | Via user-defined family | No | No | No | — | No | `cnm(npnorm(brca))` |
| Empirikos.jl | Yes (likelihood types; heteroskedastic normal) | No | `TruncatedSample` ([a, ∞) only), `SelectionTilted` | Convex prior classes (Gaussian/uniform scale mixtures, etc.) | — | No | `fit(NPMLE(DiscretePriorClass(), Hypatia.Optimizer), Zs)` |
| npeb | Yes (d-dim Gaussian, per-object Σ_i; module solvers take any K) | No | No | No | Gaia TGAS 1.4×10^6 (paper) | No | `GLMixture().fit(X, prec)` |
| mixsqpx | Yes (L) | No | No | No | Unknown | JAX (GPU possible in principle; UNVERIFIED) | `mixsolve(...)` (signature not inspected) |
| cebmf_torch | Gaussian (betahat, se) | No | No | ash-type priors | Minibatch EM | Yes (PyTorch) | `ash(betahat, se, prior='norm', batch_size=8192)` |
| decon | Gaussian/Laplace with per-object σ | No | No | Kernel bandwidth | FFT speed-up | No | `DeconPdf(y, sig, error = 'normal', bw = 'dboot1')` |

#### R packages
- **REBayes** (Koenker & Gu). CRAN 2.60 (published 2026-03-01; first release 2013-02-19; 24 versions), GPL (>= 2), maintainer R. Koenker; Depends `Matrix`; Suggests `Rmosek`; "SystemRequirements MOSEK (http://www.mosek.com) and MOSEK license"; there is no public development repo, only the read-only `cran/REBayes` mirror. Reference paper: Koenker & Gu (2017), JSS 82(8), doi:10.18637/jss.v082.i08 — [CRAN REBayes](https://cran.r-project.org/package=REBayes), [REBayes manual](https://cran.r-project.org/web/packages/REBayes/REBayes.pdf)
  - Generic engine: `KWDual(A, d, w, ...)`, where A is the "Linear constraint matrix" (in practice the n×m matrix of component likelihoods), d the constraint vector and w the observation weights. It returns `f` (dual solution = mixing density), `g` (mixture density at data points), `logLik` and `status`. Options include `rtol` (default 1e-6), MOSEK `control` lists (e.g. `num_threads`), and an experimental `alpha` that switches from the NPMLE to Rényi-divergence fitting criteria. MOSEK termination messages such as `MSK_RES_TRM_STALL` "are treated as warnings". `KWPrimal` is an experimental primal alternative — [REBayes manual](https://cran.r-project.org/web/packages/REBayes/REBayes.pdf)
  - Family wrappers include `GLmix(x, v = 300, sigma = 1, hist = FALSE, histm = 300, weights)`, where `sigma` "may take vector values of length(x)" (heteroscedastic Gaussian). `hist = TRUE` bins the data for large n (n > 1000) but is "only feasible when there are only a small number of distinct sigma". Others: `Bmix` (binomial, with a uniqueness check for small trial counts), `B2mix`, `Pmix`, `GVmix`, `GLVmix`, `TLmix`, `Weibullmix`, `Gompertzmix`, `WGLVmix`, and others. Return objects carry `x` (grid), `y` (mass), `g`, `logLik`, `dy` (Bayes rule) and `status`. `predict` gives posterior mean, median, mode or quantiles. `KWsmooth` and `bwKW` kernel-smooth a fitted NPMLE. `medde` gives penalised/shape-constrained density estimation (TV penalty on derivatives of log f, Rényi α) — [REBayes manual](https://cran.r-project.org/web/packages/REBayes/REBayes.pdf)
  - The manual index has no function returning standard errors or confidence bands for ĝ (INFERENCE from the function index) — [REBayes manual](https://cran.r-project.org/web/packages/REBayes/REBayes.pdf)
- **deconvolveR** (Efron & Narasimhan). CRAN 1.2-2 (2026-08-23; first 2016-12-01), GPL (>= 2), Imports only `splines` and `stats`; GitHub bnaras/deconvolveR (11 stars, last commit 2026-08-04, CI moved to GitHub Actions in 1.2-2); paper Narasimhan & Efron (2020), JSS 94(11), doi:10.18637/jss.v094.i11 — [CRAN deconvolveR](https://cran.r-project.org/package=deconvolveR), [GitHub](https://github.com/bnaras/deconvolveR), [NEWS](https://github.com/bnaras/deconvolveR/blob/master/NEWS.md)
  - Single function `deconv(tau, X, y, Q, P, n = 40, family = c("Poisson","Normal","Binomial"), ignoreZero = TRUE, deltaAt = NULL, c0 = 1, scale = TRUE, pDegree = 5, aStart = 1, ...)`. `tau` is the discrete support of θ. `X` holds the data (Poisson counts; Normal z-scores; Binomial (n_i, s_i) pairs). Alternatively "y, P and Q can be specified together" for "experimentation with other exponential families". Returns `mle`, `Q`, `P`, `S` (ratio of artificial to genuine information), `cov`, `cov.g`, `stats` (columns θ, g, SE.g, G, SE.G, Bias.g; plus a thinning-corrected `tg` for zero-truncated Poisson), `loglik` and `statsFunction` — [deconvolveR manual](https://cran.r-project.org/web/packages/deconvolveR/deconvolveR.pdf)
  - Source: the Normal family bins z-scores into `n` bins with unit-variance `pnorm` (homoscedastic only). The Binomial family builds an N×m per-object likelihood matrix `P <- sapply(tau, function(w) dbinom(X[,2], size = X[,1], prob = w))` with `y <- 1` (Efron's "unique case"). g = exp(Qα)/Σ with a natural-spline Q, penalty `c0 * sqrt(sum(a^2))` (an L2 *norm*, not its square), optimiser `stats::nlm` — [deconv.R](https://github.com/bnaras/deconvolveR/blob/master/R/deconv.R)
  - The repo has no tests directory (checked in the clone) — [GitHub](https://github.com/bnaras/deconvolveR)
- **ashr** (Stephens). CRAN 2.2-63 (2023-08-21; first 2016-12-21), GPL (>= 3), maintainer P. Carbonetto; Imports `mixsqp`, `SQUAREM`, `Rcpp`, `truncnorm`, etc. GitHub stephens999/ashr: 88 stars, 37 forks, 48 open issues, last commit 2026-05-04 (dev version 2.2-69), 30 test files. Paper: Stephens (2017), "False discovery rates: a new deal", Biostatistics 18(2):275–294 (per the ebnm JSS bibliography; the DOI 10.1093/biostatistics/kxw041 is from memory) — [CRAN ashr](https://cran.r-project.org/package=ashr), [GitHub](https://github.com/stephens999/ashr)
  - `ash(betahat, sebetahat, mixcompdist = c("uniform","halfuniform","normal","+uniform","-uniform", ...), optmethod = c("mixSQP","mixIP","cxxMixSquarem","mixEM","mixVBEM","w_mixEM"), nullweight = 10, prior = c("nullbiased","uniform","unit"), lik = ...)`. `prior` is a "Dirichlet prior on mixture proportions: 'nullbiased' c(nullweight,1,...,1) ... 'uniform' is c(1,1...,1); 'unit' is (1/K,...,1/K), for optmethod = mixVBEM". The likelihood objects are `lik_normal()`, `lik_t(df)`, `lik_binom(y, n)`, `lik_pois(y)`, `lik_logF(df1, df2)` and `lik_normalmix(pilik, sdlik)` — [ashr manual](https://cran.r-project.org/web/packages/ashr/ashr.pdf)
  - Low-level solvers take a generic likelihood matrix. `mixEM(matrix_lik, prior, pi_init, control)` ("a n by k matrix with (j,k)th element equal to f_k(x_j)"; "maximum a posteriori (MAP) estimation for a Dirichlet prior on π"; SQUAREM-accelerated). `mixIP` calls `REBayes::KWDual` and needs MOSEK. `mixSQP(matrix_lik, prior, pi_init, control, weights)`. `mixVBEM(matrix_lik, prior)` "Estimates posterior on mixture proportions π by Variational Bayes, with a Dirichlet prior on π" and returns `pipost`, the fitted Dirichlet parameters (adapted from Bishop 2009, ch. 10). `estimate_mixprop(data, g, prior, optmethod, ...)` is exported "for use by some other related packages" — [ashr manual](https://cran.r-project.org/web/packages/ashr/ashr.pdf)
  - Implementation detail: `mixSQP` encodes the Dirichlet prior as pseudo-observations, `A <- rbind(diag(k), matrix_lik); w <- c(prior - 1, weights)`, dropping zero-weight rows. This handles α_k ≥ 1 only, because mixsqp needs non-negative weights — [mix_opt.R](https://github.com/stephens999/ashr/blob/master/R/mix_opt.R)
- **mixsqp** (Kim, Carbonetto, Stephens, Anitescu). CRAN 0.3-54 (2023-12-20; first 2018-11-14), MIT; Rcpp/RcppArmadillo, `irlba`. GitHub stephenslab/mixsqp: 11 stars, last commit 2023-12-20, 9 test files, pkgdown docs. Paper: Kim et al. (2020), JCGS 29(2):261–273, doi:10.1080/10618600.2019.1689985, arXiv:1806.01412 — [CRAN mixsqp](https://cran.r-project.org/package=mixsqp), [GitHub](https://github.com/stephenslab/mixsqp)
  - `mixsqp(L, w = rep(1, nrow(L)), x0 = rep(1, ncol(L)), log = FALSE, control = list())` minimises −Σ_j w_j log(Σ_k L_jk x_k) over the simplex. "In some cases, it is easier or more natural to compute log(L) ... Setting log = TRUE". The solution "is invariant to rescaling the rows of L"; `normalize.rows = TRUE` is "the recommended setting for better stability". `tol.svd` sets a truncated-SVD low-rank approximation. `eps` is added inside the logs. `numiter.em` gives EM pre-iterations. There is a debugging mode that dumps inputs on failure. Multithreading comes only via a multithreaded BLAS. Returns `x`, `value`, `grad`, `hessian`, `status`, `progress` — [mixsqp manual](https://cran.r-project.org/web/packages/mixsqp/mixsqp.pdf)
- **ebnm** (Willwerscheid, Carbonetto, Stephens). CRAN 1.1-42 (2025-10-10; first 2022-03-08), GPL (>= 3); GitHub stephenslab/ebnm (14 stars, last commit 2025-10-08, 22 test files, pkgdown site); JSS (2025) 114(3), doi:10.18637/jss.v114.i03; arXiv:2110.00152 — [CRAN ebnm](https://cran.r-project.org/package=ebnm), [GitHub](https://github.com/stephenslab/ebnm)
  - `ebnm(x, s = 1, prior_family = c("point_normal","point_laplace","point_exponential","normal","horseshoe","normal_scale_mixture","unimodal","unimodal_symmetric","unimodal_nonnegative","unimodal_nonpositive","generalized_binary","npmle","deconvolver","flat","point_mass","ash"), mode = 0, scale = "estimate", g_init = NULL, fix_g = FALSE, output = ebnm_output_default(), optmethod = NULL, control = NULL, ...)`. Only Gaussian errors are supported; `s` may be heteroscedastic, but "when horseshoe priors are used, errors must be homoskedastic" and "the 'deconvolver' family requires that all standard errors be equal to 1". The NPMLE defaults to mixsqp, with `optmethod = "REBayes"` (MOSEK) optional. Outputs are `data`, `posterior`, `fitted_g`, `log_likelihood` and `posterior_sampler`, with S3 methods `coef`, `vcov` (posterior variances of θ_j, not of ĝ), `confint` (Monte Carlo), `simulate`, `predict`, `logLik`, `plot` — [ebnm manual](https://cran.r-project.org/web/packages/ebnm/ebnm.pdf)
- **nspmix** (Yong Wang). CRAN 2.0-0 (2025-05-10; first 2017-04-26), GPL (>= 2), Depends `lsei`. `cnm(x, init, model = c("npmle","proportions"), maxit = 100, tol = 1e-6, grid = 100, ...)` implements the constrained Newton method (CNM; Wang 2007, JRSSB, doi:10.1111/j.1467-9868.2007.00583.x) and hierarchical CNM. It "works for an arbitrary family of mixture models defined by the user", who supplies `initial`, `valid`, `logd`, `gridpoints`, `suppspace`, `weight`. It returns the discrete `mix`, `ll`, `max.gradient` (the gradient-function optimality check) and `convergence` — [CRAN nspmix](https://cran.r-project.org/package=nspmix), [nspmix manual](https://cran.r-project.org/web/packages/nspmix/nspmix.pdf)
- **decon** (X.-F. Wang & B. Wang; JSS 2011, 39(10)). Deconvolution-kernel density, CDF and regression estimators; `DeconPdf(y, sig, x, error = 'normal' | 'laplacian' | 'snormal', bw = 'dboot1', fft = FALSE, ...)`, where `sig` may be a per-observation vector (heteroscedastic); bandwidth selectors `dnrd`, `dmise`, `dboot1`, `dboot2`. ARCHIVED on CRAN; last version 1.3-4 (2021-10-20) — [CRAN decon](https://cran.r-project.org/package=decon)
- **deamer** (Stirnemann, Samson, Comte). Adaptive (penalised-contrast) deconvolution with a known error density (`deamerKE`), an auxiliary pure-noise sample (`deamerSE`) or replicates (`deamerRO`). ARCHIVED; only version 1.0 (2012) — [CRAN deamer](https://cran.r-project.org/package=deamer)
- **deconvolve** (T. Hyndman). GitHub-only (not on CRAN). Kernel deconvolution with homo- or heteroscedastic errors that are supplied or estimated from replicates; plug-in, CV and SIMEX bandwidths; last commit 2019-04-23 (unmaintained) — [GitHub deconvolve](https://github.com/TimothyHyndman/deconvolve)
- **Related Stephens-lab R tools**: mashr (multivariate adaptive shrinkage; CRAN 0.2.79, 2023-10-18), udr ("Ultimate Deconvolution", multivariate normal means building on Extreme Deconvolution; GitHub-only v0.3-158, last commit 2024-05-28, 8 stars) and flashier (CRAN 1.0.7) — [CRAN mashr](https://cran.r-project.org/package=mashr), [GitHub udr](https://github.com/stephenslab/udr), [CRAN flashier](https://cran.r-project.org/package=flashier)
- **Other R**: EbayesThresh (CRAN 1.4-12, 2017), locfdr (Efron; CRAN 1.1-8, 2015), npmlreg (NPMLE random effects in GLMs; CRAN 0.46-5, 2018), mixfdr (archived) and lpme (archived) — [CRAN locfdr](https://cran.r-project.org/package=locfdr), [CRAN npmlreg](https://cran.r-project.org/package=npmlreg), [CRAN lpme](https://cran.r-project.org/package=lpme). ebci (Armstrong, Kolesár & Plagborg-Møller robust EB intervals for θ_i, not for g; CRAN 1.0.0, 2021) — [CRAN ebci](https://cran.r-project.org/package=ebci)
- **GBnpmle** (Shijie Wang). An R wrapper (via `reticulate`) around PyTorch code for "Generative Bootstrapping for NPMLE", which trains a generator that outputs bootstrap NPMLEs in one pass (Gaussian-location, Poisson, lognormal, gamma and binomial mixtures; CUDA recommended). GitHub-only, GPL-3, 2 stars, last commit 2024-02-27 — [GitHub GBnpmle](https://github.com/shijiew97/GBnpmle)

#### Python packages
- **npeb** (Jake A. Soloff). PyPI 0.0.2 (first upload 2023-09-13, last 2024-02-05), MIT; deps numpy, scipy, scikit-learn, pandas; GitHub jake-soloff/NPEB (6 stars; last commit 2023-09-13; setup.py still says 0.0.1 and "Development Status :: 3 - Alpha"; no tests directory) — [PyPI npeb](https://pypi.org/project/npeb/), [GitHub NPEB](https://github.com/jake-soloff/NPEB)
  - `GLMixture(prec_type='general'|'diagonal', homoscedastic=False, atoms_init=None)`. `.fit(X, prec, max_iter_em=10, weight_thresh=0., n_chunks=1, log_prob_thresh=-inf, row_condition=False, solver='cvxpy'|'mosek', **solver_params)` takes an n×d data array and per-object precision matrices of shape (n, d, d) or (n, d), i.e. heteroscedastic multivariate Gaussian. It first solves the discretised NPMLE with atoms = data points by default ("exemplar" atoms), then runs EM to move atoms and weights jointly. `log_prob_thresh` sparsifies the kernel matrix. There is also `PMixture` (Poisson) and `Grenander` — [GLMixture.py](https://github.com/jake-soloff/NPEB/blob/master/npeb/GLMixture.py)
  - The module-level `solve_weights_cvx(K)` (cvxpy `Maximize(sum(log(A @ w)))`) and `solve_weights_mosek(K, use_sparse=True)` (MOSEK Fusion exponential-cone program) accept any n×m kernel matrix — [GLMixture.py](https://github.com/jake-soloff/NPEB/blob/master/npeb/GLMixture.py)
  - README usage: `from npeb import GLMixture; m = GLMixture(); m.fit(X, prec); gmleb = m.posterior_mean(X, prec)` — [GitHub NPEB](https://github.com/jake-soloff/NPEB)
- **mixsqpx** (Paul Diegert). PyPI releases from 0.1.2 (2025-06-27) to 0.2.1a3 (2026-05-21), MIT (copyright Bunting, Diegert & Maurel). `requires_python <3.13, >=3.12`; pinned `jax`/`jaxlib` >=0.5.0, <=0.6.0. It is "a jax-compatible interface to the mixSQP algorithm ... ported from the R implementation", with a function `mixsolve` that "can be transformed with Jax transformations such as jit" — [PyPI mixsqpx](https://pypi.org/project/mixsqpx/)
  - Its companion **spmlex** (GitHub pdiegert/spmlex, 3 stars, not on PyPI) profiles out a discretised F, maximising L*(θ) = max_F Σ_i log ∫ G(w_i, v; θ) dF(v). It uses implicit differentiation through the mix-SQP solution so that JAX autodiff and jit apply. The user supplies a JAX function returning the n×grid log-likelihood matrix — [GitHub spmlex](https://github.com/pdiegert/spmlex)
- **ebnmpy** (kclamar). Python port of ebnm with a scikit-learn-style API: `PointNormalEBNM(mode="estimate").fit(x=x, s=s)`. GitHub-only, not on PyPI, 0 stars, last updated 2021-12-12 (abandoned) — [GitHub ebnmpy](https://github.com/kclamar/ebnmpy)
- **mpebpm** (A. Sarkar). GPU (PyTorch) minibatch-SGD empirical-Bayes Poisson means with point-mass, gamma and point-gamma (parametric) priors; README claims "data on the order of 10⁶ cells and 10⁴ genes in parallel". Install is via conda channel `aksarkar` or git; last commit 2021-04-22 — [GitHub mpebpm](https://github.com/aksarkar/mpebpm)
- **cebmf_torch** (W. Denault; GitHub william-denault/cebmf_torch, 7 stars, created 2025-08-21, updated 2026-10-19; not on PyPI; licence not stated in docs). A pure-PyTorch EBNM/EBMF package (normal, exponential, Laplace and point-mass priors; `ash`) with GPU support. "Mini-batch EM for `pi` is implemented via Adam on logits (recommended) or online EM". Inputs are betahat/se tensors, not a likelihood matrix — [cebmf_torch docs](https://cebmf-torch.readthedocs.io/en/latest/), [GitHub](https://github.com/william-denault/cebmf_torch)
- **gradvi** (Stephens lab). GitHub-only Python gradient-descent VI with ash and point-normal priors for regression/trend filtering, not a deconvolution tool — [GitHub gradvi](https://github.com/stephenslab/gradvi)
- **Research-code repositories, not packages**:
  - jake-soloff/Multivariate-Heteroscedastic-NPMLE (figures for Soloff et al.) — [GitHub](https://github.com/jake-soloff/Multivariate-Heteroscedastic-NPMLE)
  - hanshengjiang/npmle_git (NPMLE for mixtures of regressions, arXiv:2108.09816) — [GitHub](https://github.com/hanshengjiang/npmle_git)
  - jiafengkevinchen/close-replication (Chen, "Empirical Bayes When Estimation Precision Predicts Parameters", Econometrica MS-22935): "All code is in Python, but NPMLE estimation relies on the package rpy2, R, and Mosek" — [GitHub close-replication](https://github.com/jiafengkevinchen/close-replication)
  - gli-minsuk-shin/AB-NPMLE (2026, anonymised double-blind code: amortised bootstrap NPMLE via a neural "AMEM-Net", PyTorch) — [GitHub AB-NPMLE](https://github.com/gli-minsuk-shin/AB-NPMLE)
- **PyPI absences** (checked 2026-10-22): no packages named mixsqp, pymixsqp, ashr, pyashr, ebnm, pyebnm, ebnmpy, npmle, empirikos, rebayes, deconvolver, gbnpmle, gradvi or extreme-deconvolution — [PyPI search via JSON API, e.g. pypi.org/project/npeb](https://pypi.org/project/npeb/)
- **Neural EB and astronomy deconvolution** (other researchers cover these): NEB, "Neural Empirical Bayes" source-distribution estimation (MaximeVandegar/NEB, 10 stars); astroML (XDGMM; PyPI 1.0.2.post1, 2022); pygmmis (PyPI 1.2.3, 2022; GMMs for "incomplete, truncated, and noisy data") — [GitHub NEB](https://github.com/MaximeVandegar/NEB), [PyPI astroML](https://pypi.org/project/astroML/), [PyPI pygmmis](https://pypi.org/project/pygmmis/)

#### Julia (and other)
- **Empirikos.jl** (Nikolaos Ignatiadis). MIT; Project.toml version 0.6.5; julia ≥ 1.10. Registered in the Julia General registry, with versions 0.5.1 … 0.6.5 listed. GitHub nignatiadis/Empirikos.jl: 20 stars, last commit 2026-08-30 ("add robust limma variants"); CI and codecov badges; 24 test files; Quarto docs with REBayes and prostate vignettes — [GitHub Empirikos.jl](https://github.com/nignatiadis/Empirikos.jl), [docs](https://nignatiadis.github.io/Empirikos.jl/dev/), [General registry Versions.toml](https://github.com/JuliaRegistries/General/blob/master/E/Empirikos/Versions.toml)
  - Design: "each sample is wrapped in a type that represents its likelihood ... makes it easy to add new likelihoods and have it automatically integrate with the rest of the package (say the nonparametric maximum likelihood estimator) through Julia's multiple dispatch". Sample types include Normal, StandardNormal, Binomial, Poisson, scaled χ², noncentral t, folded normal, truncated, truncated Poisson, noncentral hypergeometric, bivariate binomial and replicated samples — [samples.qmd](https://github.com/nignatiadis/Empirikos.jl/blob/master/docs/src/manual/samples.qmd)
  - Estimation: `fit(NPMLE(DiscretePriorClass(), Hypatia.Optimizer), Zs)` (or `Mosek.Optimizer`). The NPMLE is posed as a JuMP problem with a `RelativeEntropyCone` and uses multiplicity-weighted summaries. Prior classes include `DiscretePriorClass`, `MixturePriorClass`, `GaussianScaleMixtureClass`, `BetaMixtureClass`, `UniformScaleMixtureClass`, `GaussianLocationScaleMixtureClass`. Other estimators: `KolmogorovSmirnovMinimumDistance`, `MethodOfMoments`, `ParametricMLE` — [NPMLE.jl](https://github.com/nignatiadis/Empirikos.jl/blob/master/src/NPMLE.jl), [estimation.qmd](https://github.com/nignatiadis/Empirikos.jl/blob/master/docs/src/manual/estimation.qmd)
  - Inference: `FLocalizationInterval(flocalization, convexclass, solver, n_bisection=100)` with `DvoretzkyKieferWolfowitz(α=0.05)`, `ChiSquaredFLocalization` or `InfinityNormDensityBand` (the last "only works for homoskedastic Normal samples"). `AMARI` gives affine minimax Anderson–Rubin intervals. Targets include `PriorDensity`, `PriorMean`, `MarginalDensity`, `PosteriorMean`, `PosteriorProbability`, `PosteriorVariance`, `LinearOverLinear`. For heteroskedastic samples the DKW band multiplier becomes e (`multiplier = exp(1)`) — [flocalization_intervals.jl](https://github.com/nignatiadis/Empirikos.jl/blob/master/src/flocalization_intervals.jl), [flocalizations.jl](https://github.com/nignatiadis/Empirikos.jl/blob/master/src/flocalizations.jl), [flocalizations.qmd](https://github.com/nignatiadis/Empirikos.jl/blob/master/docs/src/manual/flocalizations.qmd)
- **DualALM** (leozzy13). "A Julia package for NPMLE" (dual augmented Lagrangian, semismooth-Newton CG; code accompanies Zhang, Cui, Sen & Toh). 3 stars, last commit 2024-12-12; no tests; Project.toml lacks name/uuid, so it is not a registered package — [GitHub DualALM](https://github.com/leozzy13/DualALM)
- **mixsqp-paper**: Julia implementations of mix-SQP and comparison methods (8 stars, last commit 2019-10-10) — [GitHub mixsqp-paper](https://github.com/stephenslab/mixsqp-paper)
- **PMcore** (LAPKB, Rust). Cross-domain NPMLE: nonparametric population-pharmacokinetics algorithms NPAG (Non-Parametric Adaptive Grid) and NPOD over ODE likelihoods, integrated with Pmetrics. crates.io `pmcore` 0.29.0, GPL-3.0, 5 stars, last commit 2026-09-15 (active) — [GitHub PMcore](https://github.com/LAPKB/PMcore)

### Inferences
- **Exact match to pastry's input**: these accept a user-supplied n×K matrix of per-object likelihoods over a grid:
  - REBayes `KWDual` (MOSEK)
  - mixsqp `mixsqp(L, log = TRUE)`
  - ashr `mixEM`/`mixSQP`/`mixVBEM` (with a Dirichlet prior)
  - deconvolveR `deconv(P=, Q=, y=)` (with caveats below)
  - npeb's module-level solvers
  - mixsqpx
  - nspmix (via a user family with `model = "proportions"`)

  Among the surveyed EB packages, only ashr's `mixVBEM` treats π as random under a Dirichlet prior, which is pastry's baseline Dirichlet–multinomial (as a VB approximation). Bootstrap-based randomness (GBnpmle) exists only as research code.
- **deconvolveR caveat** (INFERENCE from reading `deconv.R`; needs a test before relying on it). For user-supplied P/Q/y, the gradient uses `qw %*% y` and the information uses `yHat = sum(y) * f` unless `y == 1` scalar. So per-object rows (heteroscedastic "unique case") appear to be handled correctly only via the Binomial code path (`y <- 1`); passing a vector of ones would mis-scale the Fisher information. pastry could replicate Efron's delta-method formulas directly for the per-object case.
- **Python gap**: no maintained, documented, tested Python package covers NPMLE/g-modeling with arbitrary per-object likelihoods and UQ for g. Researchers in Python fall back on rpy2 + REBayes + MOSEK (Chen's CLOSE replication), or on research-grade, alpha-status packages (npeb, mixsqpx). This is a clear niche for pastry.
- **Commercial-solver dependency**: MOSEK (REBayes, npeb-mosek, Empirikos-MOSEK) is a recurring friction point. Open alternatives exist: mixsqp, Hypatia (Julia), cvxpy with open solvers (for example Clarabel/SCS exponential-cone support; UNVERIFIED for this problem size).

### Gaps
- I could not read the REBayes source repository (CRAN mirror access blocked in this session), so its internal tests and documentation quality are not assessed beyond the manual and vignette list.
- Empirikos.jl release *dates* were not retrievable (GitHub API restricted). Registry versions and the last commit date were verified.
- Download statistics (CRAN logs, PyPI stats) were not collected.
- mixsqpx's GitHub repository (github.com/pdiegert/mixsqpx, named in its PyPI README) did not appear in GitHub search and its raw README was not found on `main`. Its test suite and API beyond `mixsolve` are unverified.

## Q2. Methods: NPMLE algorithms, g-modeling, Dirichlet penalties, heteroscedastic/multivariate cases, smooth vs discrete g, self-consistency

### Takeaway
On a fixed grid, max over π in the simplex of Σ_i log Σ_k L_ik π_k is a finite-dimensional convex problem for *any* per-object likelihoods (Koenker & Mizera 2014). So pastry's input L_ik plugs straight into the NPMLE machinery. The solvers, in rough order of maturity:
- **EM**: simple, but slow near the optimum and not sparse in practice.
- **Interior point on the dual** (REBayes/MOSEK): accurate, but its normal equations become ill-conditioned for large n and m.
- **mix-SQP**: active set plus low-rank L; best for large n and moderate m in 1-D.
- **Dual ALM with semismooth Newton** (Zhang et al. 2024): exploits solution sparsity; handles n ≈ 10^6 with m ≈ 10^4, including 2-D Gaia data.
- **CNM / vertex-direction methods**: move support points rather than fix a grid.

The NPMLE is discrete and very sparse: O(log n) atoms in the Gaussian case, and non-unique in d > 1. This is good for point estimation and bad for interval estimates. Smooth alternatives:
- Efron's penalised log-spline exponential family (deconvolveR).
- Kernel-smoothed NPMLE (REBayes `KWsmooth`).
- Dirichlet-penalised mixtures over a fixed dictionary (ashr).
- Early-stopped EM from a smooth start.
- The 2026 "smooth NPMLE" with a Gaussian smoothing layer (Kim & Sen), which keeps convexity and adds inference.

At the optimum, the KKT conditions restricted to the support are exactly the self-consistency (EM fixed-point) equations: π_k equals the average over objects of the posterior probability of bin k under π itself.

### Cited Findings

#### Problem class and algorithms
- Discretising G to a grid A gives the convex problem max over w in the simplex of (1/n) Σ_i log Σ_j L_ij w_j. "A crucial observation made by Koenker and Mizera (2014) is that (11) is a (finite-dimensional) convex optimization problem ... they proposed solving the dual to (11) using an interior point solver". "EM can be prohibitively slow (Redner and Walker, 1984; Koenker and Mizera, 2014)". Other approaches listed: "EM (Laird, 1978), vertex direction and exchange methods (Böhning, 1985), semi-infinite methods (Lesperance and Kalbfleisch, 1992), constrained-Newton methods (Wang, 2007), and hybrid methods" — [Soloff, Guntuboyina & Sen, arXiv:2109.03466 (JRSSB 2024, doi:10.1093/jrsssb/qkae040)](https://arxiv.org/abs/2109.03466)
- Koenker & Mizera (2014), "Convex optimization, shape constraints, compound decisions, and empirical Bayes rules", JASA 109(506):674–685 (bibliographic details as cited in Soloff et al. and the REBayes manual) — [REBayes manual](https://cran.r-project.org/web/packages/REBayes/REBayes.pdf)
- mix-SQP (Kim et al. 2020) combines four ideas: a reformulation (non-negative orthant rather than simplex), SQP, an active-set QP solver that exploits sparse iterates, and low-rank (RRQR/tSVD) approximations of L. "When n and m are large, and n is larger than m, the fastest approach is NN-SQP-A-QR". The active-set QP cost "grows linearly in m, whereas the complexity of the IP solution grows quadratically"; the worst case is O(m^3). The active-set method also supports warm starts, which "is difficult to achieve with IP methods" — [Kim et al. 2020, arXiv:1806.01412](https://arxiv.org/abs/1806.01412)
- First-order methods: "even after running EM and projected gradient for 1,000 iterations on the 20,000 × 800 data set, the solution remained at least 0.35 log-likelihood units away from the solution obtained by mix-SQP"; "second-order information is crucial for obtaining good search directions near the solution" while "gradient-descent methods are able to rapidly identify good approximate solutions" — [Kim et al. 2020](https://arxiv.org/abs/1806.01412)
- Dual ALM (Zhang, Cui, Sen & Toh, JMLR 25(8):1–46, 2024) uses a semismooth-Newton ALM on the dual. Its generalised Hessian is sparse, costing O(n s min(n, s)) with s ≈ the number of active atoms, versus mix-SQP's dense O((n + m)k²) Hessian with k = rank(L). It "can handle n ≈ 10^6 data points with m ≈ 10^4 support points". A partial-EM (PEM) variant jointly moves atoms and weights. "The EM solution for x ... is not usually sparse, although theory guarantees sparsity" — [Zhang et al., arXiv:2208.07514](https://arxiv.org/abs/2208.07514), [JMLR v25/22-1120](https://www.jmlr.org/papers/v25/22-1120.html)
- CNM (Wang 2007, JRSSB) in nspmix optimises support points and weights. It reports `max.gradient`, the maximum of the gradient function, as an optimality check — [nspmix manual](https://cran.r-project.org/web/packages/nspmix/nspmix.pdf)
- New in 2026: Yan Zhang's "data-compression strategy that reduces the cost of repeated likelihood evaluations in NPMLE computation to polylogarithmic order in the sample size" (exponential-family mixtures) — [arXiv:2604.19725](https://arxiv.org/abs/2604.19725)

#### Discreteness, sparsity, uniqueness, grid choice
- An NPMLE exists with ≤ n atoms, and the fitted marginal values L̂_i are unique. G is optimal iff D(G, ϑ) = (1/n) Σ_i φ_Σi(X_i − ϑ)/f_G,Σi(X_i) − 1 ≤ 0 for all ϑ. The support lies in the zero set of D, i.e. at the global maxima of a "dual mixture density". Ĝ "is not necessarily unique when d > 1, even in the homoscedastic case", and the EB posterior means are then not necessarily unique — [Soloff et al., Lemma 1–2](https://arxiv.org/abs/2109.03466)
- In 1-D Gaussian mixtures with a subgaussian prior, the NPMLE has O(log n) atoms with high probability ("self-regularization") — [Polyanskiy & Wu, arXiv:2008.08244](https://arxiv.org/abs/2008.08244)
- A randomly reweighted NPMLE (independent Gamma weights concentrating at 1) is unique with high probability and has O((log n/log log n)^d + log n) atoms — [H. Jiang 2026, arXiv:2610.01088](https://arxiv.org/abs/2610.01088)
- Grid size:
  - Jiang & Zhang (2009, Prop. 5) show EM on a grid approximates well once m ≍ (log n)√n.
  - Dicker & Zhao (2016) show m = ⌊√n⌋ atoms suffices in 1-D.
  - The √ placement in both rates is read from a PDF text extraction.
  - Soloff et al. bound the support (1-D: the range of the data; homoscedastic d > 1: the convex hull; heteroscedastic d > 1: a "ridgeline manifold" that can lie outside the hull) and give an explicit discretisation-error bound (Prop. 6). They also propose an "exemplar+" atom-sampling scheme for moderate d, noting that plain exemplar atoms (atoms = data points) "can produce grossly suboptimal solutions".

  — [Soloff et al.](https://arxiv.org/abs/2109.03466)
- In ebnm, mixsqp was faster than REBayes for fewer than 80 grid components and REBayes faster for 80 or more. A Willwerscheid (2021) rule says when 80 components are "good enough" for homoskedastic data — [ebnm JSS paper, arXiv:2110.00152](https://arxiv.org/abs/2110.00152)

#### Smooth vs discrete g
- ebnm authors: "the NPMLE is a discrete distribution (Laird 1978), so posterior distributions ... are discrete as well. While the posterior mean ... may be perfectly adequate for point estimation, interval estimates can be unsatisfactory". In their wOBA example, "Each credible interval endpoint is constrained to lie at one of the support points of the NPMLE ĝ" and one player gets "a degenerate interval". deconvolveR "uses a penalized likelihood that encourages 'smooth' priors" and "can outperform the NPMLE in certain respects when the true prior is smooth (Koenker 2017)" — [ebnm paper, arXiv:2110.00152](https://arxiv.org/abs/2110.00152)
- Koenker's REBayes vignette: "the Efron approach is preferable when the mixing distribution has a smooth density, while Kiefer-Wolfowitz is preferable for" discrete mixing distributions; "likelihood methods appear to be preferable to (Fourier) kernel methods". In 1000 trials with a smooth G0 (n = 1000), the mean Wasserstein error was 0.188 for Efron, 0.343 for the unsmoothed KW-NPMLE and 0.190 for a biweight-smoothed KW-NPMLE (bandwidth 0.7, "an excellent bandwidth oracle") — [Koenker, "Bayesian Deconvolution: an R Vinaigrette" (REBayes vignette)](https://cran.r-project.org/web/packages/REBayes/vignettes/Bdecon.pdf)
- Efron (2016), Biometrika 103(1):1–20, doi:10.1093/biomet/asv068: traditional asymptotics give "very slow nonparametric rates", but "parametric exponential family modelling of g(θ) can give useful estimates in moderate-sized samples" — [RePEc abstract](https://ideas.repec.org/a/oup/biomet/v103y2016i1p1-20..html)
- Efron's g-model in deconvolveR: g(α) = exp(Qα)/Σ exp(Qα), with Q a centred and scaled natural-spline basis (`pDegree = 5`). The penalty is c0·‖α‖ (default c0 = 1), fit by `nlm` — [deconv.R](https://github.com/bnaras/deconvolveR/blob/master/R/deconv.R)
- f- vs g-modeling: Efron (2019), "Bayes, Oracle Bayes and Empirical Bayes", Statistical Science 34(2), doi:10.1214/18-STS674. "Only f(·) need be estimated" in many EB applications, "avoiding the difficult deconvolution problems of estimating g(·)" — [Project Euclid](https://projecteuclid.org/journals/statistical-science/volume-34/issue-2/Bayes-Oracle-Bayes-and-Empirical-Bayes/10.1214/18-STS674.pdf)
- Smooth NPMLE (Kim & Sen, arXiv 2026): "a hierarchical Gaussian smoothing layer that restricts the mixing distribution to a Gaussian location mixture". It is "computable via convex optimization", attains "a polynomial deconvolution rate that is asymptotically minimax", gives smooth posteriors and optimal-length marginal coverage sets, and extends to heteroscedastic Gaussian data — [arXiv:2603.27843](https://arxiv.org/abs/2603.27843)
- Smooth near-MLE by an algorithm whose smooth iterates converge to the NPMLE, with a data-driven stopping rule — [Chae, Martin & Walker 2018, SPL 140:142–146, arXiv:1712.03852](https://arxiv.org/abs/1712.03852)
- Gaussian-smoothed NPMLEs for Poisson mixtures (optimal-transport distance, near root-n) — [Lim & Han, arXiv:2406.08808](https://arxiv.org/abs/2406.08808). Gamma-smoothed Poisson NPMLE — [T. Kim, arXiv:2605.03594](https://arxiv.org/abs/2605.03594)

#### Dirichlet-penalised mixture weights (ashr)
- ashr optimises L(π) = Σ_j w_j log(Σ_k π_k f_jk) + h(π) with h(π) = Σ_k (prior_k − 1) log π_k. The default "nullbiased" prior is c(nullweight = 10, 1, …, 1). The dictionary is a fixed grid of mixture components (e.g. uniforms or normals of increasing width centred at the mode), so g is unimodal by construction. `mixVBEM` returns the Dirichlet variational posterior — [ashr manual](https://cran.r-project.org/web/packages/ashr/ashr.pdf)
- The penalty enters mix-SQP as pseudo-data rows (identity rows weighted prior_k − 1) — [mix_opt.R](https://github.com/stephens999/ashr/blob/master/R/mix_opt.R)

#### Heteroscedastic and multivariate cases
- Jiang & Zhang (2009), "General maximum likelihood empirical Bayes estimation of normal means", Ann. Statist. 37(4):1647–1684, introduced the GMLEB that REBayes `GLmix` implements ("Kiefer Wolfowitz MLE as proposed by Jiang and Zhang"). `GLmix` accepts a per-observation `sigma` vector — [REBayes manual](https://cran.r-project.org/web/packages/REBayes/REBayes.pdf)
- Soloff et al. extend the NPMLE to X_i | θ_i ~ N(θ_i, Σ_i) with known, arbitrary per-object covariances. They prove an oracle inequality (near-optimal denoising up to log factors), finite-sample Hellinger bounds and adaptive deconvolution. Applications include a Gaia TGAS colour–magnitude diagram of ≈1.4 million stars (initial grid of m = 10^4 support points) and 19-D APOGEE abundances (n = 27,238) — [arXiv:2109.03466](https://arxiv.org/abs/2109.03466)
- Precision-dependent priors: Chen's CLOSE ("Empirical Bayes When Estimation Precision Predicts Parameters", Econometrica) targets the failure of the standard assumption that θ_i is independent of σ_i — [close-replication README](https://github.com/jiafengkevinchen/close-replication)
- Normal approximations: the regret of the NPMLE under approximately normal noise is controlled by the exact-normal rate plus a CLT-quality term — [Chen, Deb & Ignatiadis 2026, arXiv:2605.31599](https://arxiv.org/abs/2605.31599)

#### Self-consistency / EM fixed point
- EM was "first observed by Laird (1978)" to apply to the NPMLE over discrete distributions — [Soloff et al.](https://arxiv.org/abs/2109.03466)
- mixsqp declares convergence when "the maximum dual residual in the Karush-Kuhn-Tucker (KKT) optimality conditions is less than or equal to convtol.sqp" — [mixsqp manual](https://cran.r-project.org/web/packages/mixsqp/mixsqp.pdf)
- The rescaling invariances that make L_ik usable up to per-object constants: "the gradient of the objective does not change with re-scaling" and "the solution to the optimization problem is invariant to rescaling the rows of L" — [mixsqp manual](https://cran.r-project.org/web/packages/mixsqp/mixsqp.pdf)

#### Key papers (bibliographic list; venue details from the cited source)
Foundations, EM and convex formulation:
- Kiefer & Wolfowitz (1956), Ann. Math. Statist. 27:887–906 — the NPMLE — [Soloff et al. refs](https://arxiv.org/abs/2109.03466)
- Laird (1978), JASA 73(364):805–811 — EM and the discreteness of the NPMLE — [Kim et al. refs](https://arxiv.org/abs/1806.01412)
- Jiang & Zhang (2009), Ann. Statist. 37(4):1647–1684 — GMLEB — [Soloff et al. refs](https://arxiv.org/abs/2109.03466)
- Koenker & Mizera (2014), JASA 109(506):674–685 — convex/IP formulation — [Soloff et al. refs](https://arxiv.org/abs/2109.03466)
- Wang (2007), JRSSB 69:185–198, doi:10.1111/j.1467-9868.2007.00583.x — CNM — [nspmix manual](https://cran.r-project.org/web/packages/nspmix/nspmix.pdf)
- Dicker & Zhao (2016), Biometrika 103(1):21–34 — grid size m = √n — [Soloff et al. refs](https://arxiv.org/abs/2109.03466)
- Feng & Dicker, arXiv:1606.02011 — approximate NPMLE via convex optimisation — [arXiv](https://arxiv.org/abs/1606.02011)
- Saha & Guntuboyina, arXiv:1712.02009 — NPMLE for Gaussian location mixtures, denoising — [arXiv](https://arxiv.org/abs/1712.02009)

Software papers:
- Koenker & Gu (2017), JSS 82(8), doi:10.18637/jss.v082.i08 — REBayes — [REBayes manual](https://cran.r-project.org/web/packages/REBayes/REBayes.pdf)
- Narasimhan & Efron (2020), JSS 94(11), doi:10.18637/jss.v094.i11 — deconvolveR — [JSS](https://www.jstatsoft.org/article/view/v094i11)
- Stephens (2017), Biostatistics 18(2):275–294 — ashr — [ebnm paper refs](https://arxiv.org/abs/2110.00152)
- Kim, Carbonetto, Stephens & Anitescu (2020), JCGS 29(2):261–273, doi:10.1080/10618600.2019.1689985, arXiv:1806.01412 — mix-SQP — [arXiv](https://arxiv.org/abs/1806.01412)
- Willwerscheid, Carbonetto & Stephens (2025), JSS 114(3), doi:10.18637/jss.v114.i03, arXiv:2110.00152 — ebnm — [arXiv](https://arxiv.org/abs/2110.00152)
- X.-F. Wang & B. Wang (2011), JSS 39(10) — decon — [CRAN decon](https://cran.r-project.org/package=decon)

g-modeling and smooth g:
- Efron (2016), Biometrika 103(1):1–20, doi:10.1093/biomet/asv068 — log-spline g-modeling, delta-method accuracy — [RePEc](https://ideas.repec.org/a/oup/biomet/v103y2016i1p1-20..html)
- Efron (2019), Statist. Sci. 34(2):177–201, doi:10.1214/18-STS674 — Bayes/oracle/EB; f- vs g-modeling — [Project Euclid](https://projecteuclid.org/journals/statistical-science/volume-34/issue-2/Bayes-Oracle-Bayes-and-Empirical-Bayes/10.1214/18-STS674.pdf)
- Efron & Hastie (2016), *Computer Age Statistical Inference*, CUP, ch. 21 (deconvolution) — [deconvolveR manual](https://cran.r-project.org/web/packages/deconvolveR/deconvolveR.pdf)
- Koenker (2017), "Bayesian Deconvolution: an R Vinaigrette", cemmap CWP38/17 — [REBayes vignette](https://cran.r-project.org/web/packages/REBayes/vignettes/Bdecon.pdf)
- Chae, Martin & Walker (2018), SPL 140:142–146, arXiv:1712.03852 — smooth near-MLE — [arXiv](https://arxiv.org/abs/1712.03852)
- Lim & Han, arXiv:2406.08808 — smoothed NPMLEs — [arXiv](https://arxiv.org/abs/2406.08808)
- Kim & Sen (2026), arXiv:2603.27843 — smooth NPMLE with inference — [arXiv](https://arxiv.org/abs/2603.27843)

Heteroscedastic/multivariate, scale and sparsity:
- Soloff, Guntuboyina & Sen (2024), JRSSB, doi:10.1093/jrsssb/qkae040, arXiv:2109.03466 — [arXiv](https://arxiv.org/abs/2109.03466)
- Zhang, Cui, Sen & Toh (2024), JMLR 25(8):1–46, arXiv:2208.07514 — dual ALM — [JMLR](https://www.jmlr.org/papers/v25/22-1120.html)
- Polyanskiy & Wu (2020), arXiv:2008.08244 — O(log n) atoms — [arXiv](https://arxiv.org/abs/2008.08244)
- H. Jiang (2026), arXiv:2610.01088 — reweighted NPMLE sparsity — [arXiv](https://arxiv.org/abs/2610.01088)
- Y. Zhang (2026), arXiv:2604.19725 — compression and fast computation — [arXiv](https://arxiv.org/abs/2604.19725)

Inference/UQ:
- Laird & Louis (1987), JASA 82(399):739–750 — bootstrap EB intervals — [Ignatiadis & Wager refs](https://arxiv.org/abs/1902.02774)
- Ignatiadis & Wager (2022), JASA 117(539):1149–1166, arXiv:1902.02774 — F-localisation/AMARI — [arXiv](https://arxiv.org/abs/1902.02774)
- Wang, Shin & Bai (2024), IEEE SPL 31:870–874, arXiv:2402.18748 — GB-NPMLE — [arXiv](https://arxiv.org/abs/2402.18748)
- Cui & Walker, Bayesian Analysis, doi:10.1214/24-BA1498, arXiv:2310.00880 — Bayesian bootstrap for mixtures — [arXiv](https://arxiv.org/abs/2310.00880)
- Ignatiadis & Kankanala (2026), arXiv:2602.20115 — Bayesian nonparametric compound decisions — [arXiv](https://arxiv.org/abs/2602.20115)
- Chen, Deb & Ignatiadis (2026), arXiv:2605.31599 — normal approximations — [arXiv](https://arxiv.org/abs/2605.31599)
- Ignatiadis & Sen, arXiv:2303.02887 — empirical partially Bayes / compound χ² — [arXiv](https://arxiv.org/abs/2303.02887)
- Koenker & Gu (2024), "Empirical Bayes for the Reluctant Frequentist", arXiv:2404.03422 (tutorial) — [arXiv](https://arxiv.org/abs/2404.03422)

Neural EB (other researchers cover this; listed for completeness):
- Neural-g (S. Wang, Chakraborty, Qin & Bai), arXiv:2406.05986 — [arXiv](https://arxiv.org/abs/2406.05986)
- Vandegar et al., Neural Empirical Bayes (code: MaximeVandegar/NEB) — [GitHub NEB](https://github.com/MaximeVandegar/NEB)

### Inferences
- **Self-consistency, derived (standard; consistent with Soloff Lemma 1 discretised).** With f_i = Σ_j L_ij π_j, the gradient is ∂ℓ/∂π_k = (1/n) Σ_i L_ik / f_i. The simplex KKT conditions are (1/n) Σ_i L_ik/f_i ≤ 1 for all k, with equality wherever π_k > 0. Multiplying the equality by π_k gives π_k = (1/n) Σ_i [π_k L_ik / f_i] = (1/n) Σ_i p(z_i = k | d_i, π). So the NPMLE is a fixed point of "stack the per-object posteriors computed under the current prior". Three consequences for pastry:
  - Naive stacking of posteriors computed under an interim prior is exactly *one* EM step from that prior.
  - EM fixed points with π_k = 0 can be non-optimal, because EM cannot revive a zeroed bin. Self-consistency is therefore necessary but not sufficient, and pastry should report the KKT certificate max_k (1/n) Σ_i L_ik/f_i − 1 (≤ tol), as nspmix (`max.gradient`) and mixsqp (dual residual) do.
  - With a Dir(α) MAP the fixed point becomes π_k ∝ Σ_i r_ik + α_k − 1. In the full Dirichlet–multinomial posterior, E[ρ_k | z] = (α_k + n_k)/(Σα + N).
- **Row invariance and interim priors.** The NPMLE depends on L only up to per-object (row) scale factors. Per-object evidences or normalisations of photo-z PDFs can therefore be dropped. Anything that varies across bins within a row cannot: the interim prior, bin widths, and selection functions. For posteriors under an interim prior π0, the correct input is L_ik ∝ p_ik/π0_k, a column rescaling.
- **Atom sparsity in pastry's setting.** NPMLE sparsity (O(log n) atoms) means that, on a fine K-bin grid, most fitted bin fractions will be exactly zero. For populations like n(z) or a CMD that are believed smooth, pastry should default to a smooth or regularised g (log-spline/GP/Dirichlet) and treat the NPMLE as a diagnostic or upper bound on the attainable likelihood.
- **Grid design.** Fine grids are cheap in pastry because the per-object likelihoods are given on bins. But in 2-D (CMD) settings, the singular-value spectrum of L decays slowly, so mix-SQP-style low-rank tricks are less effective (per Zhang et al.).

### Gaps
- I did not obtain full texts of Efron (2016), Efron (2019) or Koenker & Mizera (2014); claims about them rely on abstracts, citing papers, and the deconvolveR source.
- Kim & Sen (2026) and Chen–Deb–Ignatiadis (2026) are recent preprints (arXiv only, not peer-reviewed as of Oct 2026); I did not check whether code accompanies them.
- I found no source comparing NPMLE and smooth g-models specifically for non-Gaussian, multimodal per-object likelihoods (photo-z-like). Evidence is for Gaussian, Poisson and binomial noise.

## Q3. Uncertainty quantification for g

### Takeaway
Few tools return uncertainty for g itself:
- **deconvolveR**: delta-method SEs and bias for g and its CDF G, valid when the smooth parametric g-model is right.
- **Empirikos.jl**: F-localisation and AMARI confidence intervals for linear (and ratio) functionals of g, with explicit bias control.
- **ashr's `mixVBEM`**: a variational Dirichlet posterior for π, which VB theory says is typically too narrow.

ebnm, REBayes, mixsqp, nspmix and npeb return none.

On coverage:
- Efron-style plug-in intervals are the shortest but under-cover, because they ignore smoothing bias.
- F-localisation intervals are conservative (≈100% pointwise, ≥95% simultaneous).
- AMARI intervals are near nominal pointwise.
- EB posterior intervals that ignore uncertainty in ĝ under-cover, worst for the discrete NPMLE.

Bootstrap approaches (Laird–Louis; GB-NPMLE; the Bayesian bootstrap for mixtures; parametric/nonparametric bootstrap of the pmf) and fully Bayesian nonparametric approaches exist mainly as papers and research code. 2026 work (Kim & Sen's smooth NPMLE; Ignatiadis & Kankanala's Bayesian nonparametrics) is closing the theory gap.

### Cited Findings
- **deconvolveR (Efron's delta method)**, implemented in `statsFunction`:
  - Fisher information I(α) = Q^T W diag(ŷ) W^T Q, where W = g·(P/f − 1) and ŷ = N f (multinomial case) or 1 (per-object case).
  - Penalty gradient ṡ = c0 α/‖α‖ and Hessian s̈.
  - bias(α̂) = −(I + s̈)^{-1} ṡ; Cov(α̂) = (I + s̈)^{-1} I (I + s̈)^{-1}.
  - Pushed to g by D_q = (diag g − g g^T) Q: Cov(ĝ) = D_q Cov D_q^T, bias(ĝ) = D_q bias(α̂). SE(G) follows by cumulative summation.
  - The "S" ratio is tr(s̈)/tr(I) ("the ratio of artificial to genuine information").

  — [deconv.R](https://github.com/bnaras/deconvolveR/blob/master/R/deconv.R), [deconvolveR manual](https://cran.r-project.org/web/packages/deconvolveR/deconvolveR.pdf)
- **Ignatiadis & Wager**, "Confidence Intervals for Nonparametric Empirical Bayes Analysis" (JASA 2022, 117(539):1149–1166; arXiv:1902.02774). The intervals "provide asymptotic frequentist coverage of empirical Bayes estimands, such as the posterior mean or the local false sign rate", "even when the estimands are only partially identified or when empirical Bayes point estimates converge very slowly":
  - F-localisation intervals have "simultaneous coverage for all z".
  - AMARI intervals have "pointwise coverage".
  - Efron's variance-only intervals "do not account for bias and so could only achieve valid coverage via undersmoothing; and it is unclear how to achieve valid undersmoothing in practice".

  — [arXiv:1902.02774](https://arxiv.org/abs/1902.02774)
- **Their simulations** (400 replicates, 95% intervals for the posterior mean):
  - "The log-spline intervals are shortest; however they do not achieve nominal coverage, while all other methods do."
  - F-localisation pointwise coverage is "close to 100%"; AMARI pointwise coverage is "closer to the nominal 95%".
  - AMARI simultaneous coverage was 75% and 70% in two spiky-prior settings; F-localisation simultaneous coverage was above 95%.

  Worked example (Bichsel Swiss car-insurance data, n = 119,853, Poisson): for z = 4 claims the NPMLE posterior mean is 0.82, with F-localisation interval 0.52–1.64 and AMARI 0.58–1.39 — [arXiv:1902.02774](https://arxiv.org/abs/1902.02774)
- **Empirikos.jl** implements `FLocalizationInterval` (DKW, χ² and sup-norm density-band localisations; solved by Charnes–Cooper linear-fractional programming or quasi-convex bisection) and `AMARI`, for targets including `PriorDensity`, `PriorMean`, `MarginalDensity`, `PosteriorMean` and `PosteriorProbability`. Heteroskedastic samples use a compound DKW band with multiplier e — [Empirikos.jl source](https://github.com/nignatiadis/Empirikos.jl/blob/master/src/flocalizations.jl), [intervals docs](https://github.com/nignatiadis/Empirikos.jl/blob/master/docs/src/manual/intervals.qmd)
- **ebnm evaluation**: "A known limitation of empirical Bayes methods is that they often underestimate uncertainty in the posteriors, since uncertainty in the estimate of g is not taken into account". Nominal-90% credible intervals "tended to be too small" but "usually not far off". "The lone exception was the NPMLE, which tended to have much poorer coverage because ... it results in a discrete prior that can greatly underestimate uncertainty" — [ebnm paper, arXiv:2110.00152](https://arxiv.org/abs/2110.00152)
- **VB intervals for mixture weights**: "the covariance matrices from the variational Bayes approximations are normally 'too small' compared with those for the maximum likelihood estimator, so that resulting interval estimates for the parameters will be unrealistically narrow, especially if the components of the mixture model are not well separated". This is relevant to ashr's `mixVBEM` Dirichlet posterior — [Wang & Titterington, "Inadequacy of interval estimates corresponding to variational Bayesian approximations" (AISTATS 2005)](https://proceedings.mlr.press/r5/wang05a.html)
- **Bootstrap**:
  - Laird & Louis (1987), "Empirical Bayes confidence intervals based on bootstrap samples", JASA 82(399):739–750 (bibliographic details as cited in Ignatiadis & Wager) — [arXiv:1902.02774 references](https://arxiv.org/abs/1902.02774)
  - GB-NPMLE: Wang, Shin & Bai (2024), IEEE Signal Processing Letters 31:870–874, arXiv:2402.18748. A generator produces NPMLE bootstrap estimates after "a single evaluation of a novel two-stage optimization algorithm", accurate "even when there are a hundred thousand observations". Code: GBnpmle (R + PyTorch) — [arXiv:2402.18748](https://arxiv.org/abs/2402.18748), [GitHub GBnpmle](https://github.com/shijiew97/GBnpmle)
  - Bayesian bootstrap for mixtures (works off the NPMLE; Pólya-urn/gradient-ascent reinterpretation): Cui & Walker, Bayesian Analysis, doi:10.1214/24-BA1498 — [arXiv:2310.00880](https://arxiv.org/abs/2310.00880)
  - Nonparametric and parametric bootstrap confidence intervals for power-series mixture pmfs: Balabdaoui, Besdziek & Wang 2025 — [arXiv:2508.00163](https://arxiv.org/abs/2508.00163)
- **Smooth NPMLE with inference**: Kim & Sen (2026) "characterize marginal coverage sets that are optimal in expected length, construct plug-in estimators of these sets, and establish theoretical guarantees ... in terms of both coverage probability and expected length", including heteroscedastic Gaussian data — [arXiv:2603.27843](https://arxiv.org/abs/2603.27843)
- **Fully Bayesian nonparametric**: "the fully Bayesian posterior mean achieves near-optimal risk. Moreover, it is admissible as a genuine Bayes rule, whereas the corresponding NPMLE plug-in rule is inadmissible" — [Ignatiadis & Kankanala 2026, arXiv:2602.20115](https://arxiv.org/abs/2602.20115)
- **Objective Dirichlet for many-cell multinomials** (relevant to α in pastry's Dir(α) baseline): the recommended overall prior is Dir(1/m, …, 1/m), "or 0.8 × {1/m, ..., 1/m} to be slightly more precise" — [Berger, Bernardo & Sun 2015, Bayesian Analysis 10(1):189–221, arXiv:1504.02689](https://arxiv.org/abs/1504.02689)

### Inferences
- **UQ is pastry's differentiator.** No Python package, and only two R/Julia packages, return uncertainty on g. A Dirichlet–multinomial posterior (Gibbs or HMC) with calibrated coverage, plus frequentist checks, would be novel in the Python ecosystem.
- **Report functionals, not just bin marginals.** When per-object likelihoods are wide relative to the bin width, the likelihood is nearly flat along many directions of the simplex (the "very slow rates" of deconvolution). Posterior marginals of individual ρ_k will then be prior-dominated and strongly anti-correlated. pastry should foreground well-identified functionals (CDF/cumulative fractions, coarse-bin sums, moments, posterior predictive marginal f) and expose the full covariance or samples. An Efron-style "artificial vs genuine information" ratio (prior or penalty curvature versus Fisher information) is a cheap, interpretable diagnostic for prior dominance.
- **Choice of α matters when K is large.** Dir(1) adds K pseudo-counts in total, which can dominate when N is not much greater than K. The Berger–Bernardo–Sun Dir(1/K) recommendation is a principled default to offer. Note that ashr's pseudo-row MAP trick cannot represent α_k < 1, but Gibbs/HMC posteriors can.
- **Calibration testing.** Coverage claims should be tested by simulation-based calibration and by frequentist coverage of ρ and its functionals under known truths, including misspecified (smooth vs spiky) truths. The Ignatiadis–Wager simulation design (spiky/negative-spiky priors, pointwise vs simultaneous coverage) is a good template.
- **Cheap bootstrap path.** A weighted-likelihood (Bayesian) bootstrap is easy to support because mixsqp-style solvers already accept observation weights w_i: draw w ~ Dirichlet(1, …, 1)·N or Exp(1) weights, warm-start and re-solve. Jiang (2026) shows Gamma-reweighted NPMLEs are unique and sparse.

### Gaps
- I found no published coverage study of deconvolveR's delta-method intervals *for g itself* (as opposed to posterior means), nor any for ashr's `mixVBEM` intervals specifically.
- Wang & Titterington's result concerns VB for mixture models in general (verified at PMLR R5, AISTATS 2005). Its magnitude for ashr's fixed-component `mixVBEM` with large K was not quantified.
- I found no Python/Julia implementation of GB-NPMLE-style or Bayesian-bootstrap UQ beyond research code. Whether Kim & Sen (2026) or Ignatiadis & Kankanala (2026) ship software was not checked.
- How Empirikos' heteroskedastic DKW multiplier e was derived (likely a compound DKW inequality from Ignatiadis & Sen 2023, arXiv:2303.02887) is UNVERIFIED.

## Q4. Performance at scale and numerical issues

### Takeaway
The largest published runs:
- **1-D, n ≈ 2 × 10^6, m ≈ 10^2–10^3**: about 10 s with mix-SQP.
- **2-D Gaia TGAS, n ≈ 1.4 × 10^6, m = 10^4**: about 8 hours with the dual ALM on a 32-core, 256 GB workstation. The dense L alone takes about 109 GB.

Nothing in the EB software literature demonstrates N = 10^7–10^8. At pastry's target sizes, memory for a dense N×K likelihood matrix (8NK bytes in float64) is the binding constraint, not convexity. Chunked, sparse or compressed L, minibatch or streaming first-order updates (GPU-friendly), and second-order "polishing" on a reduced problem will all be needed.

Known numerical issues:
- EM converges slowly near the optimum and does not produce sparse iterates.
- Interior-point normal equations are ill-conditioned for large m and n; MOSEK can fail or stall.
- mix-SQP's speed depends on L being numerically low-rank, which fails in 2-D and higher.
- Likelihoods underflow, so log-likelihood input and row normalisation are needed.
- Zero columns in L must be avoided.
- g is non-unique for d > 1 and for binomial data with few trials.

### Cited Findings
- **mix-SQP vs MOSEK interior point (KWDual)**: "In large data sets (n ≈ 10^6 observations, m ≈ 10^3 mixture components), our implementation achieves at least 100-fold reduction in runtime compared with a state-of-the-art interior point solver". GIANT GWAS (n = 2,126,678, m > 100): "about 10 s vs. 1,000 s". "KWDual runtimes increased more rapidly with m because KWDual did not benefit from the reduced-rank matrix operations". Using a low-rank approximation of L cut runtime "by a factor of over 10" at n = 10^6. Benchmarks ran in Julia with OpenBLAS — [Kim et al. 2020, arXiv:1806.01412](https://arxiv.org/abs/1806.01412)
- **Why low rank works in 1-D**: "many of the distributions g_k are similar to one another. Hence, the matrix L is numerically rank deficient and, in our experience, many of its singular values are near floating-point machine precision" — [Kim et al. 2020](https://arxiv.org/abs/1806.01412)
- **Why it fails in 2-D**: L "has a significant proportion of nonzero singular values when d ≥ 2 ... a low rank approximation of L may lose crucial information". Interior-point methods have "inherently ill-conditioned normal equations that are extremely costly to solve by an iterative method when both m and n are large". "REBayes solver reports failure for a synthetic data set with m = 10^4 and n = 7 × 10^4" — [Zhang, Cui, Sen & Toh, JMLR 2024, arXiv:2208.07514](https://arxiv.org/abs/2208.07514)
- **Gaia TGAS CMD** (n = 1,363,432, d = 2):
  - mix-SQP "can only handle this problem with small m, up to several hundreds"; with a 20 × 20 grid it took 28 minutes and "failed to converge within 100 iterations".
  - The ALM on a 100 × 100 grid (m = 10^4) took about 472 minutes on a 32-core, 256 GB workstation and found 1,677 support points. "128 GB of RAM was not enough ... since the matrix L ∈ R^{1,363,432×10,000} alone consumes approximately 109 GB".
  - On a 10^5 subsample with m = 10^4: ALM about 5 minutes vs REBayes about 73 minutes.

  — [Zhang et al. 2024](https://arxiv.org/abs/2208.07514)
- **ebnm runtimes** (R, i9-13900KF, 32 GB): runtimes for n from 100 to 10^6 "grew linearly or close to linearly in n". The NPMLE and unimodal families were "slower than the fastest methods by multiple orders of magnitude". mixsqp was faster than REBayes for fewer than 80 grid components; REBayes was faster for 80 or more — [ebnm paper, arXiv:2110.00152](https://arxiv.org/abs/2110.00152)
- **Size reduction by binning or multiplicity**:
  - REBayes `GLmix(hist = TRUE)` bins data "for large problems, say n > 1000", but with heteroscedastic σ this is "only feasible when there are only a small number of distinct sigma" — [REBayes manual](https://cran.r-project.org/web/packages/REBayes/REBayes.pdf)
  - Empirikos summarises samples by multiplicity before fitting (`summarize`, `multiplicity(Zs) ./ nobs(Zs)` weights) — [NPMLE.jl](https://github.com/nignatiadis/Empirikos.jl/blob/master/src/NPMLE.jl)
  - A 2026 preprint compresses data so repeated likelihood evaluations cost polylog(n) for exponential-family mixtures — [arXiv:2604.19725](https://arxiv.org/abs/2604.19725)
- **Sparse kernels**: npeb's `log_prob_thresh` "set[s] all log probabilities under log_prob_thresh to -∞ (so the resulting probability is exactly zero)" and uses a sparse matrix in the MOSEK path. `n_chunks` computes the kernel in chunks. The default atoms are all data points ("Selecting all data points as atoms"), i.e. an n×n kernel — [GLMixture.py](https://github.com/jake-soloff/NPEB/blob/master/npeb/GLMixture.py)
- **GPU/minibatch**:
  - cebmf_torch (GitHub william-denault/cebmf_torch; 7 stars; created 2025-08, updated 2026-10-19; not on PyPI) is "a pure-PyTorch implementation" of EBMF/EBNM where "Mini-batch EM for `pi` is implemented via Adam on logits (recommended) or online EM", running on whatever device the tensors live on. Example: `res = ash(betahat, se, prior='norm', batch_size=8192)` — [cebmf_torch docs](https://cebmf-torch.readthedocs.io/en/latest/), [GitHub](https://github.com/william-denault/cebmf_torch)
  - mpebpm fits parametric EB Poisson means on GPU by minibatch SGD at about 10^6 × 10^4 scale — [GitHub mpebpm](https://github.com/aksarkar/mpebpm)
  - GB-NPMLE recommends CUDA — [GitHub GBnpmle](https://github.com/shijiew97/GBnpmle)
- **Numerical safeguards in mixsqp**:
  - Requirements: L must be "non-negative and finite (and not missing). Further, no column should be entirely zeros".
  - `log = TRUE` input; `normalize.rows` ("divides each row by the largest entry ... to avoid numerical overflow or underflow"); `eps` added inside the logarithms "to sidestep computing logarithms of zero".
  - If not converged, "x may not satisfy the equality constraint", i.e. may not sum to 1.
  - Multithreading comes only through a multithreaded BLAS.

  — [mixsqp manual](https://cran.r-project.org/web/packages/mixsqp/mixsqp.pdf)
- **MOSEK behaviour** in REBayes: "Mosek termination messages are treated as warnings ... solutions producing, for example, MSK_RES_TRM_STALL ... may still provide a satisfactory solution". The `rtol` default is 1e-6, "in some cases it is desirable to tighten this, say to 1e-10"; multithreading "may have unintended (undesirable) consequences when running simulations on clusters" — [REBayes manual](https://cran.r-project.org/web/packages/REBayes/REBayes.pdf)
- **Non-uniqueness**:
  - Binomial NPMLE: "When the number of trials is small the NPMLE may be non-unique", with a `unique` check in `Bmix` — [REBayes manual](https://cran.r-project.org/web/packages/REBayes/REBayes.pdf)
  - Gaussian d > 1: the NPMLE and the posterior means can be non-unique — [Soloff et al.](https://arxiv.org/abs/2109.03466)

### Inferences
- **Memory arithmetic for pastry** (simple arithmetic, not from a source). A dense N×K matrix costs 4NK bytes in float32: N = 10^8, K = 10^3 → 400 GB; N = 10^8, K = 10^2 → 40 GB; N = 10^7, K = 10^4 → 400 GB. Default fitting paths must therefore never materialise dense L for large N. Workable options:
  - Stream chunks (from disk or memory-map).
  - Store per-object likelihoods sparsely (top-k bins, or a log-threshold as in npeb).
  - Deduplicate or cluster near-identical rows into weights (multiplicity, as in Empirikos and REBayes `hist`).
- **Recommended solver tiering for pastry** (INFERENCE from the benchmarks above):
  1. Minibatch or streaming EM / mirror ascent on softmax logits (GPU, O(nnz) per pass) to get near the optimum quickly. Kim et al. note first-order methods "rapidly identify good approximate solutions".
  2. A second-order polish (mix-SQP-style active-set Newton on the support, or the dual ALM) on a deduplicated or subsampled problem.
  3. A final full-data KKT check: max_k (1/N) Σ_i w_i L_ik/f_i ≤ 1 + tol.
- **Hessian cost.** Hessian-based methods need H = L^T diag(1/f²) L/N, which costs O(nnz·K) and is K×K (800 MB at K = 10^4, factorisation O(K³)). Active-set restriction to the NPMLE support (1,677 atoms of 10^4 in Gaia) is what makes Newton-type methods viable.
- **Bayesian (Dirichlet–multinomial) runs at this N.** Augmented Gibbs costs O(nnz) per sweep plus N categorical draws. Its autocorrelation, like EM's slowness, is governed by the fraction of missing information, which is high when per-object likelihoods overlap many bins. HMC on unconstrained logits uses the same O(nnz) gradient as EM and avoids latent z_i (INFERENCE; standard EM/data-augmentation theory, e.g. Dempster, Laird & Rubin 1977, not checked here).
- **Statistical vs systematic error at large N.** At N ≥ 10^7 the statistical posterior width of ρ shrinks like N^{-1/2} in well-identified directions. Reported uncertainties will then be dominated by deconvolution ill-posedness (flat directions) and by errors in the L_ik themselves (photo-z template or calibration errors). pastry's UQ design should allow for likelihood-model uncertainty (e.g. tempering or error inflation), not only sampling noise.

### Gaps
- No published benchmark covers N ≥ 10^7 for NPMLE/g-modeling, any GPU NPMLE solver with arbitrary L, or sparse-L formulations at that scale. These are open engineering questions for pastry.
- I could not verify the internals or speed of mixsqpx (JAX). Its GitHub repository did not appear in GitHub search; only the PyPI release exists.
- Runtime figures are from different hardware and languages (Julia, MATLAB, R) and are not directly comparable.

## Q5. API/UX lessons and design insights for pastry

### Takeaway
The packages people actually use pair a tiny "matrix in, weights out" core (mixsqp `mixsqp(L, w)`, REBayes `KWDual(A, d, w)`) with domain-friendly front ends (ebnm, ash, GLmix) that return one rich, uniform result object. pastry should copy this layering:
1. An expert core taking log-L (dense, sparse or chunked) plus weights.
2. Input adapters for PDFs on grids, posterior samples, interim-prior posteriors and Gaussian (x, σ).
3. A scikit-learn/emcee-style estimator with consistent outputs: ρ̂, per-object posteriors, log-likelihood, a KKT certificate, UQ samples or intervals, and diagnostics.

It should avoid the observed pain points: commercial solvers (MOSEK), Gaussian- or z-score-only scope, interfaces that differ across prior families, untested research code, print-based logging, import-time warnings, and quadratic-memory defaults.

### Cited Findings
- **Fragmentation is the core UX problem.** ebnm's authors: "each existing package has a different interface and outputs, which hinders comparisons", hence "a unified interface for efficiently solving the EBNM problem using a wide variety of prior families". The ebnm object always contains `data`, `posterior`, `fitted_g`, `log_likelihood` and optionally `posterior_sampler`, selected via `output = ebnm_output_default()`/`ebnm_output_all()`. `g_init` with `fix_g = TRUE` reuses or fixes a prior (e.g. "to do computations with the 'true' g in simulations"). S3 methods: `coef`, `vcov`, `confint`, `simulate`, `predict`, `logLik`, `nobs`, `plot`, `summary` — [ebnm paper](https://arxiv.org/abs/2110.00152), [ebnm manual](https://cran.r-project.org/web/packages/ebnm/ebnm.pdf)
- **Minimal solver API with good diagnostics** (mixsqp):
  - Signature `mixsqp(L, w, x0, log = FALSE, control = list())`, `mixsqp_control_default()`.
  - `verbose` prints a table with columns "objective", "max(rdual)" (the KKT residual), "nnz", "max.diff", "nqp", "nls".
  - Returns `status` and `progress`.
  - A debug mode "will generate an .RData file containing the exact mixsqp inputs, and will stop execution upon convergence failure".
  - Example: `out <- mixsqp(L, w); mixobjective(L, out$x, w)`.

  — [mixsqp manual](https://cran.r-project.org/web/packages/mixsqp/mixsqp.pdf)
- **Likelihood-as-type design** (Empirikos.jl): users wrap each observation in a likelihood type (e.g. `NormalSample(Z, σ)`, `PoissonSample`). Estimation is `fit(NPMLE(DiscretePriorClass(), Hypatia.Optimizer), Zs)`. The solver is any JuMP-compatible optimiser (open-source Hypatia or MOSEK). Inference is `confint(method, target, Zs)` with estimands as objects (`PriorDensity`, `PosteriorMean`, ...) — [Empirikos docs](https://github.com/nignatiadis/Empirikos.jl/blob/master/docs/src/manual/estimation.qmd), [samples.qmd](https://github.com/nignatiadis/Empirikos.jl/blob/master/docs/src/manual/samples.qmd)
- **Single-function scientific API** (deconvolveR): `result <- deconv(tau = tau, X = z, family = "Normal", pDegree = 6); g <- result$stats[, "g"]`. The result is a list including an `loglik` closure and a `statsFunction` closure for recomputing statistics at other α. "Some warnings are emitted by the nlm routine used for optimization, but they are mostly inconsequential" — [deconvolveR manual](https://cran.r-project.org/web/packages/deconvolveR/deconvolveR.pdf), [vignette](https://github.com/bnaras/deconvolveR/blob/master/vignettes/deconvolution.Rmd)
- **Scope limits hurt**: ebnm's deconvolveR wrapper "fits a model to z-scores rather than observations and associated standard errors ... it does not seem appropriate for the wOBA data" — [ebnm paper](https://arxiv.org/abs/2110.00152)
- **Commercial solver friction**: "REBayes ... relies on the commercial interior-point solver MOSEK ... in order to provide a fully open-source toolkit that does not require installation of commercial software, we re-implemented estimation of the NPMLE in ebnm using the open-source package mixsqp" — [ebnm paper](https://arxiv.org/abs/2110.00152). Chen's Python replication still needed "rpy2, R, and Mosek", with Docker and a `mosek.lic` file — [close-replication](https://github.com/jiafengkevinchen/close-replication)
- **Domain defaults baked into the API**: ash defaults to a unimodal prior centred at 0 with a "nullbiased" Dirichlet (weight 10 on the null), suited to sparse genomic effects — [ashr manual](https://cran.r-project.org/web/packages/ashr/ashr.pdf), [ashr README](https://github.com/stephens999/ashr)
- **Research-code UX** (npeb): scikit-learn-like `GLMixture().fit(X, prec)` / `.posterior_mean(X, prec)`, but:
  - progress is reported with `print`;
  - import emits printed warnings if `mosek.fusion` or `cvxpy` is missing;
  - there is no test suite;
  - the default atoms are all n data points, i.e. an n×n kernel.

  — [GLMixture.py](https://github.com/jake-soloff/NPEB/blob/master/npeb/GLMixture.py)
- **Differentiable NPMLE**: spmlex/mixsqpx expose a JAX-transformable mix-SQP (`mixsolve`) and implicit differentiation of the profile log-likelihood L*(θ) = max_F Σ_i log ∫ G(w_i, v; θ) dF(v). The user supplies only a JAX function returning the observation × grid log-likelihood matrix — [GitHub spmlex](https://github.com/pdiegert/spmlex), [PyPI mixsqpx](https://pypi.org/project/mixsqpx/)
- **Selection/truncation precedents**:
  - deconvolveR's zero-truncated Poisson divides P by (1 − e^{−θ}) and reports a "thinning"-corrected prior `tg` alongside g — [deconv.R](https://github.com/bnaras/deconvolveR/blob/master/R/deconv.R), [vignette](https://github.com/bnaras/deconvolveR/blob/master/vignettes/deconvolution.Rmd)
  - Empirikos has `TruncatedSample` (currently "Only [a, ∞) ... intervals allowed") and `SelectionTilted` distributions — [truncated.jl](https://github.com/nignatiadis/Empirikos.jl/blob/master/src/samples/truncated.jl)
- **Documentation and test quality** (from clones, 2026-10-22):

  | Package | Test files | Documentation |
  |---|---|---|
  | ashr | 30 | vignettes |
  | ebnm | 22 | pkgdown, JSS paper |
  | mixsqp | 9 | pkgdown, vignettes |
  | Empirikos.jl | 24 | Quarto docs, CI, codecov |
  | udr | 14 | pkgdown, vignettes |
  | PMcore | 5 | API docs site, CI |
  | deconvolveR | 0 | vignette, JSS paper |
  | npeb | 0 | README, demo notebook |
  | DualALM | 0 | none |

  REBayes not assessed (no public repo). — [GitHub ashr](https://github.com/stephens999/ashr), [GitHub ebnm](https://github.com/stephenslab/ebnm), [GitHub mixsqp](https://github.com/stephenslab/mixsqp), [GitHub Empirikos.jl](https://github.com/nignatiadis/Empirikos.jl), [GitHub deconvolveR](https://github.com/bnaras/deconvolveR), [GitHub NPEB](https://github.com/jake-soloff/NPEB)
- **Canonical benchmark datasets** bundled with these packages:
  - Empirikos: Bichsel, Tacks, Shakespeare, Surgery, Prostate, EfronMorrisBaseball, LordCressie, Norberg, Butterfly, and others.
  - deconvolveR: bardWordCount, surg, disjointTheta.
  - REBayes: tacks, flies, Norberg, velo, bball.
  - ebnm: wOBA.

  — [Empirikos datasets dir](https://github.com/nignatiadis/Empirikos.jl/tree/master/src/datasets), [deconvolveR manual](https://cran.r-project.org/web/packages/deconvolveR/deconvolveR.pdf), [REBayes manual](https://cran.r-project.org/web/packages/REBayes/REBayes.pdf)

### Inferences
Concrete design recommendations for pastry, synthesised from the above:
1. **Input layer.** Make `log_L` (N×K) the canonical input: numpy, JAX, scipy.sparse, or a chunk iterator/memmap, plus optional per-object weights.
   - Normalise rows in log space (subtract the row max), as mixsqp recommends; no evidences are required, by the row-scale invariance.
   - Provide adapters:
     - `from_pdfs(pdf, grid, bin_widths, interim_prior=None)`: divide by the interim prior column-wise and handle densities vs bin masses explicitly.
     - `from_samples(samples, bins, interim_prior=None)`: histogram/KDE of posterior samples, reweighted by 1/interim prior.
     - `from_gaussian(x, sigma)`: for ebnm-style data.

   None of the surveyed EB packages supports interim-prior correction or posterior-sample inputs. This is a gap pastry fills.
2. **Selection effects.** Implement selection as column rescaling L_ik / S_k (likelihood conditional on selection, the deconvolveR thinning pattern) and return both the selected-population and intrinsic ρ, as deconvolveR returns g and `tg`.
3. **Estimator layer, one interface, consistent outputs.** Offer `NPMLE` (convex, certificate), `DirichletMAP(alpha)` (the ashr pattern, but allowing α < 1 via a proper optimiser), `SmoothG` (Efron log-spline / GP / generalised-Dirichlet penalties), and `FullBayes` (Gibbs/HMC). Each returns the same result object, ebnm-style: `rho`, `log_likelihood`, `posterior(i)` (lazy), `kkt_gap`, `samples` or `cov`, `diagnostics` (effective atoms, prior-information ratio à la Efron's S, convergence trace).
4. **Solver layer.** Open-source by default: no MOSEK or cvxpy requirement.
   - SQUAREM-accelerated EM as a robust baseline.
   - A mix-SQP-like active-set Newton for moderate N·K.
   - A JAX/GPU minibatch path for huge N.
   - Implicit differentiation through the inner solve (as spmlex does) so that outer hyperparameters (selection-function parameters, smoothness scales, flow parameters) can be learned by gradient.
5. **Diagnostics and logging.** Use the `logging` module and an mixsqp-style progress table (objective, KKT residual, nnz). Provide a debug-dump option that saves inputs on failure.
6. **Validation suite.**
   - Golden-value tests against REBayes, mixsqp and Empirikos on bundled canonical datasets (Tacks, Bichsel, Shakespeare, Surgery).
   - KKT-certificate tests.
   - Simulation-based-calibration and coverage tests for UQ, following the Ignatiadis–Wager spiky/smooth prior simulation design.
7. **Defaults.** Choose domain-appropriate defaults (smooth g for n(z) and CMDs) rather than genomics-style null-biased unimodal priors. Warn when N/K is small enough that Dir(α) dominates.

### Gaps
- No user studies or issue-tracker analyses of these packages were done; UX lessons are inferred from documentation, papers and source code.
- How spmlex/mixsqpx compute the implicit gradient at a sparse (active-set) solution was not inspected.
- REBayes' internal tests and documentation could not be assessed.
