# HEP/nuclear-physics unfolding: software, methods and uncertainty quantification (census for pastry, as of Oct 2026)

Scope: HEP and nuclear-physics unfolding, meaning the deconvolution of detector-smeared histograms or event samples, as of 2026-10-22. It covers the software, the statistical methods and the uncertainty quantification (UQ), and what transfers to pastry (per-object likelihoods L_ik, Dirichlet–multinomial population model).

Verification method: repository metadata comes from full git clones read on 2026-10-22 and from the GitHub search API (star counts). Package data comes from the PyPI JSON, conda-forge and CERN GitLab v4 APIs. APIs and features were read from source code and docstrings; **no package code was executed**. Paper claims come from arXiv full texts or abstracts, and journal references from the arXiv API. "[unverified]" marks claims I could not confirm from a primary source.

## 1. Software status in 2026 and usability (inputs, outputs, API, maintenance)

### Takeaway
Binned unfolding in production still runs on C++/ROOT tools: **RooUnfold** (BSD-3, v3.1.0 Oct 2025, now on conda-forge, active 2026 development branches) and **TUnfold** (last standalone release V17.9, Nov 2019, bundled in ROOT and RooUnfold). The Python-native classical tools are frozen: **PyUnfold** (last release Nov 2018) and **PyFBU** (unmaintained; needs PyMC3 3.5 / Theano / numpy<1.20). Since 2020, method development has been dominated by ML unbinned unfolding: OmniFold plus more than 15 generative or discriminative variants. Apart from the TensorFlow-only `omnifold` PyPI package (0.1.36, Dec 2024) and a BDT OmniFold class for RooUnfold, that work ships as single-paper research code without releases, docs or tests. **No HEP tool accepts a per-object likelihood matrix L_ik as input.** Every binned tool assumes one shared response matrix, and every unbinned tool assumes a paired (truth, reco) simulation instead of explicit likelihoods.

### Cited Findings

#### A. Classical binned toolkits (shared response matrix R_ij)
- **RooUnfold** (C++/ROOT; Python via PyROOT).
  - Master `src/` holds RooUnfoldBayes (iterative/D'Agostini), RooUnfoldSvd (Höcker–Kartvelishvili, via TSVDUnfold), RooUnfoldBinByBin, RooUnfoldInvert, RooUnfoldIds (iterative dynamically stabilised), RooUnfoldGP (Gaussian process), RooUnfoldPoisson ("a simple likelihood unfolding") and RooUnfoldTUnfold. It also has **bundled TUnfold V17 sources** (TUnfoldDensity, TUnfoldSys, TUnfoldBinning, TUnfoldIterativeEM) and RooFitUnfold (a RooFit likelihood interface). — [GitLab src tree](https://gitlab.cern.ch/RooUnfold/RooUnfold/-/tree/master/src), [README](https://gitlab.cern.ch/RooUnfold/RooUnfold)
  - The README says version "3.0.0: Inclusion of RooFit, transition to RooFitUnfold". Tags run 3.0.0 (2022-07-19) to 3.1.0 (2025-10-09); the last master commit was 2025-12-11. Branches active in 2026 include `add-bru-algorithm` (2026-05-04, i.e. Blobel's Regularized Unfolding), `response-event-storage` (2026-05-04), `root-compat-fixes` (2026-05-09) and `fix-warnings` (2026-05-14). Maintainers: Adye, Burgard, Brenner, Croft. — [GitLab API](https://gitlab.cern.ch/RooUnfold/RooUnfold)
  - Install: build with cmake against ROOT (CVMFS/LCG, conda or pixi). `pyproject.toml` uses scikit-build+cmake. conda-forge has `roounfold` 3.1.0 (updated 2026-09-20). Not on PyPI. — [README](https://gitlab.cern.ch/RooUnfold/RooUnfold), [conda-forge](https://anaconda.org/conda-forge/roounfold)
  - Docs: a User Guide PDF (2022), a CERN tutorial and Doxygen. Cite as Brenner et al., IJMPA 35 (2020) 2050145, arXiv:1910.14654. The older reference is Adye, PHYSTAT 2011, CERN-2011-006 pp. 313–318, arXiv:1105.1160. — [README](https://gitlab.cern.ch/RooUnfold/RooUnfold), [arXiv:1105.1160](https://arxiv.org/abs/1105.1160)
  - RooFitUnfold "provides a common interface to these algorithms as well as common uniform methods to evaluate their performance in terms of bias, variance and coverage". — [Brenner et al. 2019, arXiv:1910.14654](https://arxiv.org/abs/1910.14654)
  - A 2025 paper treats systematic uncertainties "in a Likelihood formalism" via the RooFit interface. It compares Tikhonov regularisation inside the likelihood with Tikhonov applied separately. — [Brenner, Burgard, Croft 2025, arXiv:2510.15129](https://arxiv.org/abs/2510.15129)
  - OmniFold in RooUnfold: a BDT-based `RooUnfoldOmnifold` class ("will be incorporated into RooUnfold") lives at https://github.com/rymilton/unbinned_unfolding (2 stars, created 2025-03). It was not in RooUnfold master `src/` on 2026-10-22. — [Milton et al., JINST 20 (2025) P05034, arXiv:2503.09720](https://arxiv.org/abs/2503.09720), [GitHub](https://github.com/rymilton/unbinned_unfolding)
  - Minimal usage, quoted from `examples/RooUnfoldExample.py` (abridged): `import ROOT; ROOT.gSystem.Load("libRooUnfold")` (the example first tries `import RooUnfold`), then `response = ROOT.RooUnfoldResponse(40, -10.0, 10.0)`, then for each MC event `response.Fill(x_meas, x_true)` or `response.Miss(x_true)` for undetected events, then `unfold = ROOT.RooUnfoldBayes(response, hMeas, 4)`. The alternatives are `RooUnfoldSvd(response, hMeas, 20)`, `RooUnfoldBinByBin`, `RooUnfoldInvert`, `RooUnfoldTUnfold` and `RooUnfoldIds(response, hMeas, 3)`. — [example](https://gitlab.cern.ch/RooUnfold/RooUnfold/-/blob/master/examples/RooUnfoldExample.py)
- **TUnfold** (C++, S. Schmitt, DESY; shipped inside ROOT and bundled in RooUnfold as `#define TUnfold_VERSION "V17.9"`).
  - It is a least-squares fit with Tikhonov regularisation; Schmitt, JINST 7 (2012) T10003. — [arXiv:1205.6201](https://arxiv.org/abs/1205.6201)
  - The V17.9 API, read from the RooUnfold-bundled headers:
    - Regularisation modes `kRegModeSize`, `kRegModeDerivative`, `kRegModeCurvature` and `kRegModeMixed`, with an optional area constraint (`kEConstraintArea`) and bin-width or user density modes.
    - Strength scans: `ScanLcurve`; `ScanTau`, which minimises the average, maximum or squared global correlation (`kEScanTauRhoAvg/RhoMax/RhoSquareAvg`, each with a `...Sys` variant that includes systematics); and `ScanSURE` (Stein's unbiased risk estimate, with `GetSURE` and `GetDF`, the effective degrees of freedom).
    - Background: `SubtractBackground`.
    - Systematics: `AddSysError` (modes `kSysErrModeMatrix/Relative/Shift`).
    - Outputs: `GetOutput`, `GetFoldedOutput`, `GetEmatrixTotal`, `GetEmatrixSysUncorr`, `GetEmatrixSysSource`, `GetDeltaSysSource`, `GetRhoItotal` (global correlations), `GetBias`, `GetChi2A` and `GetChi2L`.
    - — [TUnfold.h](https://gitlab.cern.ch/RooUnfold/RooUnfold/-/blob/master/src/RooUnfold/TUnfold/TUnfold.h), [TUnfoldDensity.h](https://gitlab.cern.ch/RooUnfold/RooUnfold/-/blob/master/src/RooUnfold/TUnfold/TUnfoldDensity.h), [TUnfoldSys.h](https://gitlab.cern.ch/RooUnfold/RooUnfold/-/blob/master/src/RooUnfold/TUnfold/TUnfoldSys.h)
  - TUnfold also contains `TUnfoldIterativeEMV17`, which "implements the iterative EM unfolding method (also called D'Agostini Method or 'iterative Bayesian method')", where "iterations can be chosen based on SURE" via `ScanSURE(nIterMax, ...)`. — [TUnfoldIterativeEM.h](https://gitlab.cern.ch/RooUnfold/RooUnfold/-/blob/master/src/RooUnfold/TUnfold/TUnfoldIterativeEM.h)
  - The latest standalone release is **V17.9 (2019-11-20)**. The page warns: "The version distributed with the root package often is outdated and does not contain the latest bug fixes." Schmitt's 2016 review compares the L-curve scan with minimum global correlation. — [TUnfold page](https://www.desy.de/~sschmitt/tunfold.html), [Schmitt 2016, arXiv:1611.01927](https://arxiv.org/abs/1611.01927)
  - The ROOT reference docs (root.cern/doc) now redirect to CERN single sign-on, so the bundled headers are the accessible API reference. — [root.cern doc (SSO redirect observed 2026-10-22)](https://root.cern/doc/master/classTUnfoldDensity.html)
- **PyUnfold** (Python, MIT, JOSS 2018).
  - It implements D'Agostini iterative Bayesian unfolding (IBU). Signature: `iterative_unfold(data, data_err, response, response_err, efficiencies, efficiencies_err, prior=None, ts='ks', ts_stopping=0.01, max_iter=100, cov_type='multinomial', return_iterations=False, callbacks=None)`. — [source unfold.py](https://github.com/jrbourbeau/pyunfold/blob/master/pyunfold/unfold.py)
  - Stopping: the test statistic between successive iterations can be KS, χ², Bayes factor or relative maximum difference.
  - Priors: uniform by default, a `jeffreys_prior` helper, or user-supplied.
  - Smoothing: a `SplineRegularizer(degree, smooth, groups)` callback smooths between iterations.
  - Errors: Adye-corrected error propagation, with a Multinomial or Poisson data covariance.
  - **Return value: a dict with `unfolded`, `stat_err`, `sys_err` (from response-matrix MC statistics), `ts_iter`, `num_iterations` and `unfolding_matrix`. Only per-bin errors are returned, not a full covariance matrix** (read from the return dict in source). — [source](https://github.com/jrbourbeau/pyunfold/blob/master/pyunfold/unfold.py), [docs](https://jrbourbeau.github.io/pyunfold/)
  - Minimal usage (docstring): `iterative_unfold(data=[100,150], data_err=[10,12.2], response=[[0.9,0.1],[0.1,0.9]], response_err=..., efficiencies=[1,1], efficiencies_err=[0.01,0.01])` returns `{'num_iterations': 4, 'unfolded': array([94.3,155.7]), 'stat_err': ..., ...}`. It has a pytest suite (test_unfold, test_mix, test_priors, test_teststat, test_callbacks) and Sphinx docs with tutorials. — [source](https://github.com/jrbourbeau/pyunfold)
  - Maintenance: last commit and release 0.5.0 on 2018-11-18/19 (PyPI). conda-forge has `pyunfold` 0.5.0. GitHub: 32 stars, 14 open issues. CI used Travis and AppVeyor. JOSS doi:10.21105/joss.00741. — [PyPI](https://pypi.org/project/PyUnfold/), [GitHub](https://github.com/jrbourbeau/pyunfold)
- **PyFBU / fbu** (Fully Bayesian Unfolding; Python on PyMC3 and Theano; GPL-2.0).
  - Model: a prior on truth bins (default 'Uniform' between user `lower` and `upper` bounds) and a Poisson likelihood `mc.Poisson('unfolded', mu=R·truth (+bkg), observed=data)`.
  - Backgrounds: normalisation nuisance parameters (Uniform on [0,3] or bounded Normal with given relative error).
  - Object systematics: standard-normal nuisance parameters that linearly morph the signal and background templates.
  - Regularisation: an optional Tikhonov potential, `-alpha*|curvature - refcurv|`.
  - Sampling: NUTS with defaults `nTune=1000`, `nMCMC=10000`, `nChains=2`, or a MAP mode (L-BFGS-B). Outputs: `trace` per truth bin and `nuisancestrace`. — [source PyFBU.py](https://github.com/pyFBU/fbu/blob/master/fbu/PyFBU.py), [tikhonov.py](https://github.com/pyFBU/fbu/blob/master/fbu/tikhonov.py)
  - Minimal usage (tutorial): `myfbu = fbu.PyFBU(); myfbu.data=[100,150]; myfbu.response=[[0.08,0.02],[0.02,0.08]]; myfbu.lower=[0,0]; myfbu.upper=[3000,3000]; myfbu.run(); trace = myfbu.trace`. Backgrounds go in `myfbu.background={'bckg1':[20,30]}` and `myfbu.backgroundsyst={'bckg1':0.5}`, and systematics in `myfbu.objsyst={...}`. — [tutorial.ipynb](https://github.com/pyFBU/fbu/blob/master/tutorial.ipynb)
  - Status: README says "no longer actively maintained and it depends on libraries (e.g. Theano) which are... no longer maintained". The legacy install is Python 3.8.16 with `pip install "numpy<1.20" pymc3==3.5 fbu`. PyPI 0.1.2 is from 2018-11-13. The original gerbaudo/fbu is **archived**; the pyFBU/fbu fork had its last commit 2023-03-23. — [GitHub](https://github.com/pyFBU/fbu), [PyPI](https://pypi.org/project/fbu/)
  - Method paper: Choudalakis 2012, arXiv:1201.4612. Its outcome is "a posterior probability density for the spectrum ... Regularization consists in choosing a non-constant prior"; it treats systematics by marginalising sampled nuisance parameters. — [arXiv:1201.4612](https://arxiv.org/abs/1201.4612)
  - A **modern reimplementation of FBU** (TensorFlow 2.9 and TensorFlow Probability 0.17, with NUTS) accompanies Neural Posterior Unfolding: "One of our goals is to provide an updated version of Fully Bayesian Unfolding that makes use of current tools." It takes about 40 s per pseudo-experiment (10k draws + 10k tuning). The code is notebooks in jp2555/NPU, with no package or licence file found. — [Torales Acosta et al. 2025, arXiv:2509.06370](https://arxiv.org/abs/2509.06370), [GitHub](https://github.com/jp2555/NPU)
- **D'Agostini's "improved" IBU (2010)**: "uncertainties are now treated at all levels by probability density functions and their propagation is performed by Monte Carlo integration", with Dirichlet priors for the smearing matrix. Implemented in R on the author's web page; not checked. — [arXiv:1010.0632](https://arxiv.org/abs/1010.0632)
- **Wiener-SVD** (C++/ROOT; MicroBooNE/BNL).
  - Inputs: model spectrum, measured spectrum, **measured covariance matrix**, response and a smoothness matrix.
  - Outputs: unfolded spectrum, **"additional smearing matrix"**, Wiener filter and unfolded covariance.
  - Last commit 2021-04-11. Reference: Tang et al., JINST 12 P10002 (2017). — [GitHub](https://github.com/BNLIF/Wiener-SVD-Unfolding), [arXiv:1705.03568](https://arxiv.org/abs/1705.03568)
- **funfolding** (TU Dortmund astroparticle physics; MIT).
  - Classes: StandardLLH (Poisson likelihood with Tikhonov), SystematicLLH with LinearModelSystematics, LLHSolutionMinimizer, LLHSolutionGradientDescent, LLHSolutionMCMC (emcee), SVDSolution and DSEAGaussianNB, plus "Decision-Tree based binning". — [GitHub](https://github.com/tudo-astroparticlephysics/funfolding)
  - PyPI 0.3.0 (2024-01-11) pins Python <3.12 and requires `pymc3`. — [PyPI](https://pypi.org/project/funfolding/)
- **CherenkovDeconvolution.py / .jl** (Bunse; GPL-3): DSEA (classifier-based spectrum estimation), IBU and RUN (Blobel-type regularised unfolding), with adaptive step-size strategies. The Python version's last commit was 2022-10-19 and it is not on PyPI. — [GitHub py](https://github.com/mirkobunse/CherenkovDeconvolution.py), [GitHub jl](https://github.com/mirkobunse/CherenkovDeconvolution.jl)
- **qunfold** (Bunse; Python and JAX).
  - Described as "A unified implementation of quantification and unfolding algorithms". Methods are composed from losses and data representations, with LikelihoodMaximizer and ExpectationMaximizer among them.
  - It has a scikit-learn-style `fit`/`predict` API, e.g. `ACC(RandomForestClassifier(oob_score=True)).fit(X,y).predict(X_tst)`.
  - Install from git (`pip install 'qunfold @ git+https://github.com/mirkobunse/qunfold'`). It has CI and docs at https://mirkobunse.github.io/qunfold; last commit 2026-07-13. — [GitHub](https://github.com/mirkobunse/qunfold)
- **QUnfold** (Quantum4HEP; MIT).
  - `QUnfolder(response, measured, binning, lam)` solves min ‖Rx−d‖² + λ‖Lx‖², where L is a discrete Laplacian. It solves this as a QUBO via D-Wave simulated annealing, hybrid or QPU solvers, or as Gurobi integer/QUBO problems.
  - Errors: `cov` is estimated from the spread of annealer reads, plus optional Poisson Monte-Carlo toys (`num_toys`). The hybrid solver returns `cov = diag(sol)`.
  - PyPI 0.3.5 (2024-09-20). The repository had 451 commits, last 2026-08-11. Tests are thin (only `tests/test_utils.py`). Docs: https://qunfold.readthedocs.io.
  - Papers: JINST 21 (2026) T08004 per the README, and arXiv:2602.22776. — [source qunfolder.py](https://github.com/Quantum4HEP/QUnfold/blob/main/src/qunfold/qunfolder.py), [PyPI](https://pypi.org/project/QUnfold/), [arXiv:2602.22776](https://arxiv.org/abs/2602.22776)
  - Minimal usage (README): `unfolder = QUnfolder(response, measured, binning, lam=0.1); unfolder.initialize_qubo_model(); sol, cov = unfolder.solve_simulated_annealing(num_reads=100)`. — [README](https://github.com/Quantum4HEP/QUnfold)
- **GP unfolding** (Bozson, Cowan, Spanò 2018). The estimator is the posterior mode; for approximately Gaussian bin counts it equals "the mean function of a GP conditioned on the maximum likelihood estimator". "Regularisation is introduced via the kernel function", which can vary along the spectrum, and "the full statistical covariance matrix for the estimator is obtained". Kernel hyperparameters (amplitude A, length scale l) are chosen by simulation closure tests or by a method from the GP literature. Code: notebooks (adambozson/gp-unfold, 2018), plus RooUnfoldGP in RooUnfold. — [arXiv:1811.01242](https://arxiv.org/abs/1811.01242), [GitHub](https://github.com/adambozson/gp-unfold)
- **Nuclear γ-ray unfolding (Oslo group)**:
  - Lima et al. 2025: regularised maximum likelihood (RMLE) with non-negativity, detector response and an explicit background model; "well-calibrated confidence intervals" with "correct coverage". — [arXiv:2511.16687](https://arxiv.org/abs/2511.16687)
  - Mjøs et al. 2026: empirical-Bayes hierarchical Poisson model with a joint ON/OFF background likelihood. The prior is "centered on an automatically selected Richardson-Lucy reference spectrum, with an adaptive width". Sampled with NUTS; reports simultaneous bands. — [arXiv:2606.24971](https://arxiv.org/abs/2606.24971)
  - Code: andreashmj/empirical-bayes-unfolding-oslo-method (PyMC, GPL-3, v1.0.0, pytest suite, 2026-09-22). It needs the `shapedev` branch of ompy (GPL-3, 7 stars, not on PyPI). — [GitHub](https://github.com/andreashmj/empirical-bayes-unfolding-oslo-method), [ompy](https://github.com/oslocyclotronlab/ompy)
- **Other 2024–2026 classical methods**:
  - Gagunashvili: entropy-regularised maximum likelihood with K-means binning and a bootstrap-chosen regularisation parameter that minimises mean integrated square error (MISE). — [CPC 308 (2025) 109473, arXiv:2402.12990](https://arxiv.org/abs/2402.12990)
  - He shows it beats Richardson–Lucy on MISE and on the condition number of the correlation matrix. — [CPC 318 (2026) 109894, arXiv:2505.10283](https://arxiv.org/abs/2505.10283)
  - Croft 2026 translates Blobel's Fortran RUN into "Blobel's Regularised Unfolding" (BRU): cubic B-splines, eigenmode filtering and automatic regularisation strength. Compare the RooUnfold branch `add-bru-algorithm`. — [arXiv:2604.01069](https://arxiv.org/abs/2604.01069)

#### B. ML / unbinned unfolding (paired simulation instead of a response matrix)
- **OmniFold**: iterative two-step classifier reweighting. Step 1 reweights reco-level simulation to data; step 2 pulls the weights back to particle level. The authors call it a generalisation of IBU. — [Andreassen et al., PRL 124 (2020) 182001, arXiv:1911.09107](https://arxiv.org/abs/1911.09107)
  - `omnifold` (PyPI). Signature: `MultiFold(name, reco_model, gen_model, data, mc, ..., niter=3, n_ensemble=1, batch_size=128, epochs=50, lr=1e-4, early_stop=10)`, then `.Unfold()` and `.reweight(x, model2)`. `DataLoader(reco, gen, weight, normalize, bootstrap)` draws Poisson(1) bootstrap weights. Models: built-in `MLP` and Point-Edge Transformer `PET`, or any Keras model. Optional Horovod for multi-GPU. No real docs beyond the README and notebooks. — [source omnifold.py](https://github.com/ViniciusMikuni/omnifold/blob/main/omnifold/omnifold.py), [dataloader.py](https://github.com/ViniciusMikuni/omnifold/blob/main/omnifold/dataloader.py), [README](https://github.com/ViniciusMikuni/omnifold)
  - hep-lbdl/OmniFold (Keras notebooks; MIT; last commit 2026-06-17) covers background subtraction by neural positive reweighting, fakes, resolution and efficiency, in its "GaussianExample" notebook. — [GitHub](https://github.com/hep-lbdl/OmniFold)
  - Real-data uses: H1 (arXiv:2108.12376). ATLAS 24-dimensional Z+jets (PRL 133 (2024) 261803, arXiv:2405.20041), published as **per-event weights** (`weights_nominal` plus `weights_*` columns for uncertainties) with notebooks. — [arXiv:2108.12376](https://arxiv.org/abs/2108.12376), [ATLAS GitLab](https://gitlab.cern.ch/atlas-physics/public/sm-z-jets-omnifold-2024)
  - Neutrinos: T2K mock-data study (PRD 112, 012008 (2025)). — [arXiv:2504.06857](https://arxiv.org/abs/2504.06857), [code](https://github.com/rhuang1/OmnifoldT2K)
  - Heavy-ion and large-background extension (OmniFoldHI), with a proof of equivalence to EM and IBU (JHEP07 (2026) 012). — [arXiv:2507.06291](https://arxiv.org/abs/2507.06291), [code](https://github.com/OmniFoldHI/OmniFoldHI)
  - Profile OmniFold adds nuisance parameters. — [Zhu et al. 2025, arXiv:2512.07074](https://arxiv.org/abs/2512.07074), [code](https://github.com/richardzhs/ProfileOmnifold)
  - AUSSIE is a non-iterative replacement for step 2 (SciPost Phys. 21, 055 (2026); MIT; PyTorch and Hydra). — [arXiv:2602.24282](https://arxiv.org/abs/2602.24282), [code](https://github.com/heidelberg-hepml/aussie)
  - RAN uses a Wasserstein critic and needs no iterations. — [arXiv:2606.06603](https://arxiv.org/abs/2606.06603), [code](https://github.com/umarsqureshi/RAN)
  - Moment Unfolding uses Boltzmann-type weights w(x)=exp(λ0 x + λ1 x²) and unfolds moments only (PRD 110, 116013 (2024)). — [arXiv:2407.11284](https://arxiv.org/abs/2407.11284), [code](https://github.com/HEP-GAN/MomentUnfolding)
  - Optimal transport: SwdFold (arXiv:2406.01635); Wasserstein loss with a provably convergent generalised Sinkhorn algorithm. — [arXiv:2406.01635](https://arxiv.org/abs/2406.01635), [arXiv:2603.20903](https://arxiv.org/abs/2603.20903), [code](https://github.com/katycraig/unfolding-Wasserstein-loss)
- **Generative (per-event posterior sampling p(truth | reco))**:
  - GAN (Datta, Kar, Roy 2018, arXiv:1806.00433); FCGAN (SciPost Phys. 8, 070 (2020), arXiv:1912.00477). — [arXiv:1806.00433](https://arxiv.org/abs/1806.00433), [arXiv:1912.00477](https://arxiv.org/abs/1912.00477)
  - cINN "allows for a per-event statistical interpretation" (SciPost Phys. 9, 074 (2020)). — [arXiv:2006.06685](https://arxiv.org/abs/2006.06685)
  - Iterative cINN (IcINN) "adjusts for deviations between simulated training samples and data". — [arXiv:2212.08674](https://arxiv.org/abs/2212.08674)
  - Event-by-event probabilities from transfer matrices, compared with IcINN. — [arXiv:2310.17037](https://arxiv.org/abs/2310.17037)
  - Latent variational diffusion. — [arXiv:2305.10399](https://arxiv.org/abs/2305.10399), [arXiv:2404.14332](https://arxiv.org/abs/2404.14332), [code](https://github.com/Alexanders101/LVD)
  - SBUnfold, a Schrödinger bridge (PRD 109 (2024) 076011). — [arXiv:2308.12351](https://arxiv.org/abs/2308.12351), [code](https://github.com/ViniciusMikuni/SBUnfold)
  - Conditional DDPM. — [arXiv:2406.01507](https://arxiv.org/abs/2406.01507), [code](https://gitlab.cern.ch/cpazosbo/cddpm-unfolder)
  - Distribution mapping (SciPost Phys. 18, 200 (2025)). — [arXiv:2411.02495](https://arxiv.org/abs/2411.02495)
  - GenFoldG/GenFoldC, which handle acceptance, efficiency and impurities. — [arXiv:2509.02708](https://arxiv.org/abs/2509.02708)
  - Top decays. — [arXiv:2501.12363](https://arxiv.org/abs/2501.12363)
  - Conditional flow matching (CFM) with Lorentz-equivariant transformers (L-GATr) in "several hundred dimensions". — [arXiv:2510.19906](https://arxiv.org/abs/2510.19906), [code](https://github.com/heidelberg-hepml/high-dim-unfolding)
  - Benchmark comparison: "The Landscape of Unfolding with Machine Learning" (SciPost Phys. 18, 070 (2025)). — [arXiv:2404.18807](https://arxiv.org/abs/2404.18807)
- **Likelihood-based and "simulation-prior-independent" neural unfolding (closest to pastry)**:
  - **Neural Empirical Bayes** (NEB): "direct maximization of the log-marginal likelihood of the observations" under a neural source distribution, comparing biased and de-biased estimators and variational approaches, with a collider-physics demonstration (AISTATS 2021). — [Vandegar et al., arXiv:2011.05836](https://arxiv.org/abs/2011.05836)
  - **SPINUP** "builds on the Neural Empirical Bayes (NEB) approach and encodes the unfolded distribution as a generative model that can be forward-mapped to match the observed data". It is made efficient by neural importance sampling, and "ensembling can be used to estimate the effect of information loss". — [Butter et al. 2025, arXiv:2507.15084](https://arxiv.org/abs/2507.15084)
  - **Unbinned Profiled Unfolding** uses as ML loss "the full likelihood function, based on binned inputs at detector-level" and profiles nuisance parameters (PRD 108, 016002 (2023)). — [Chan & Nachman, arXiv:2302.05390](https://arxiv.org/abs/2302.05390), [code](https://github.com/qwerasd903/UnbinnedProfiledUnfolding)
  - **Neural Posterior Unfolding** (NPU): a MADE/MAF flow learns p(t|m) from (truth, measured) histogram pairs, giving amortised posteriors. — [arXiv:2509.06370](https://arxiv.org/abs/2509.06370)

#### C. Metadata census (verified 2026-10-22)
| Tool | Lang / engine | Licence | Latest release | Last commit | Stars | Status |
|---|---|---|---|---|---|---|
| RooUnfold | C++/ROOT (+PyROOT) | BSD-3 | 3.1.0 (2025-10-09); conda-forge 3.1.0 | master 2025-12-11; branches 2026-05 | 14 (GitLab), 48 forks | maintained |
| TUnfold | C++ (in ROOT; bundled in RooUnfold) | not stated on TUnfold page | V17.9 (2019-11-20) | n/a | n/a | stable/frozen |
| PyUnfold | Python/numpy | MIT | 0.5.0 (2018-11-19) | 2018-11-18 | 32 | unmaintained since 2018 |
| fbu (PyFBU) | Python/PyMC3+Theano | GPL-2 | 0.1.2 (2018-11-13) | 2023-03-23 (fork) | 9 (archived) / 3 (fork) | unmaintained (README) |
| omnifold | Python/TF≥2.15 | MIT | 0.1.36 (2024-12-20) | 2025-05-28 | 7 | sporadic |
| hep-lbdl/OmniFold | Keras notebooks | MIT | none | 2026-06-17 | 34 | reference code |
| QUnfold | Python/D-Wave/JAX | MIT | 0.3.5 (2024-09-20) | 2026-08-11 | 25 | active repo, no recent release |
| qunfold (Bunse) | Python/JAX | MIT (pyproject classifier) | git only | 2026-07-13 | 4 | active, small |
| funfolding | Python/pymc3/emcee | MIT | 0.3.0 (2024-01-11) | 2024-01-11 | 1 | dormant |
| CherenkovDeconvolution.py | Python | GPL-3 | none on PyPI | 2022-10-19 | 2 | dormant |
| Wiener-SVD | C++/ROOT | none found | none | 2021-04-11 | 4 | dormant |
| gp-unfold | notebooks | GPL-3 | none | 2018-10-31 | 3 | archival (method lives in RooUnfoldGP) |
| NPU (+FBU reimpl.) | TF 2.9/TFP 0.17 notebooks | none found | none | 2025-04-10 | 0 | paper code |
| MomentUnfolding | TF notebooks | none found (has CITATION.cff) | tags v1.0.0/1.0.1 | 2026-03-08 | 1 | paper code |
| SBUnfold | Python | none found | none | 2024-02-22 | 4 | paper code |
| aussie | PyTorch+Hydra | MIT | none | 2026-06-06 | 3 | paper code |
| high-dim-unfolding | PyTorch, CFM+L-GATr | MIT-style | none | 2025-10-23 | 1 | paper code (CI tests) |
| ProfileOmnifold | notebooks | none found | none | 2026-05-07 | 0 | "actively under development" |
| RAN | PyTorch | MIT | none | 2026-04-05 | 2 | paper code |
| OmniFoldHI | Python | none found | none | 2025-08-29 | 1 | paper code |
| unbinned_unfolding (RooUnfoldOmnifold BDT) | Python/ROOT | (not checked) | none | (updated 2026-06-12) | 2 | paper code |
| EB γ-ray (andreashmj) | PyMC/NUTS | GPL-3 | v1.0.0 | 2026-09-22 | 0 | paper code with tests |
| RooUnfoldML (CERN GitLab) | Python | BSD-3 | none (PyPI 404) | 2022-04-21 | – | abandoned stub ("Features: TODO") |
— sources: the GitHub, GitLab, PyPI and conda-forge pages linked above; [GitHub search API](https://github.com/search?q=unfolding) star counts as of 2026-10-22.
- Name collisions: PyPI `unfolding` is Bloch-wave band unfolding and PyPI `unfold` unpacks life-cycle-assessment (LCA) scenario databases; neither is HEP. — [PyPI unfolding](https://pypi.org/project/unfolding/), [PyPI unfold](https://pypi.org/project/unfold/)
- Search noise: a GitHub search for "unfolding" (pushed after 2025-01-01) is dominated by image-restoration "deep unfolding networks" and band-structure tools; it surfaced no further HEP unfolding packages. — [GitHub search](https://github.com/search?q=unfolding&type=repositories)

### Inferences
- The packaged, maintained Python options for HEP unfolding in 2026 are, in effect, RooUnfold via conda-forge plus PyROOT (heavy ROOT dependency) and `omnifold` (TensorFlow-only, unbinned reweighting). There is **no maintained, lightweight, pip-installable Bayesian unfolding package**. PyFBU's niche, posterior sampling with nuisance parameters, is vacant, and the NPU authors re-implemented FBU in notebooks. pastry's Dirichlet–multinomial core with NUTS or Gibbs sampling would fill that niche even for HEP users, provided it accepts a response matrix as a special case of L_ik.
- Most ML unfolding repositories are single-paper code with no tests, docs or licences. The community has noted the need for "starting point" packages (Milton et al. 2025), so a well-engineered package would stand out.
- PyUnfold returning only diagonal errors, and QUnfold's hybrid solver returning `cov = diag(sol)`, are examples of what pastry should avoid. Always expose the full covariance and posterior samples.

### Gaps
- The TUnfold licence is not stated on its home page, and I did not check ROOT's licence for `hist/unfold`. The ROOT-shipped TUnfold version was not checked; only the RooUnfold-bundled copy (V17.9) was read.
- Licences for unbinned_unfolding, SBUnfold, NPU, OmniFoldHI, ProfileOmnifold and Wiener-SVD were not confirmed. No LICENSE file was seen in the clones (unbinned_unfolding was not cloned).
- I did not inspect the RooUnfold User Guide PDF for documented coverage or toy-error options (e.g. `kCovToy`) [unverified].
- No standalone code was found for Gagunashvili's MISE method or Croft's BRU beyond the RooUnfold branch name.

## 2. Mapping to pastry: shared response matrix vs per-object likelihoods; IBU = EM = Richardson–Lucy = Shepp–Vardi

### Takeaway
Binned HEP unfolding is pastry's model with **one shared likelihood vector per reco bin**: L_ik = R_{b(i),k}, where b(i) is the reco bin of object i. **Iterative Bayesian Unfolding is the EM algorithm for the multinomial/Poisson mixture weights**, identical to Richardson–Lucy and Shepp–Vardi. This is documented by statisticians (Kuusela & Panaretos), by the OmniFold authors (an unbinned generalised-EM proof) and by Falcão & Takacs (a full derivation). The unbinned per-event EM in the OmniFold paper (Eq. 10) is exactly pastry's per-object EM, except that OmniFold estimates p(m|t) implicitly with classifiers on paired simulation instead of taking explicit L_ik. HEP's efficiency correction, its wide-bin "response depends on the truth" problem and its "simulation prior dependence" of generative unfolding map directly onto pastry's selection effects, within-bin shape and interim-prior problems.

### Cited Findings
- Standard HEP forward model: measured histogram n with E[n] = ν = Rμ + β, where R_ij = P(observed in bin i | true bin j) (efficiency included) and β is background. Two approaches: "unfolding" (estimate μ̂ with covariance U) and "folding" (fold the theory prediction with R and compare to the data). — [PDG Statistics review §40.2.5 (Cowan, rev. 2023)](https://pdg.lbl.gov/2024/reviews/rpp2024-rev-statistics.pdf)
- IBU update: t_j^(n) = Σ_i [R_ij t_j^(n−1) / Σ_k R_ik t_k^(n−1)] m_i. "One of the most widely used methods is Iterative Bayesian Unfolding (IBU), also known as Richardson-Lucy deconvolution." — [Andreassen et al. 2020, arXiv:1911.09107](https://arxiv.org/abs/1911.09107)
- Unbinned per-event version: ν_n(t) p_Gen(t) = ∫dm p_Data(m) p(m|t) ν_{n−1}(t) p_Gen(t) / ∫dt′ p(m|t′) ν_{n−1}(t′) p_Gen(t′). The authors prove it "is a consequence of a generalized expectation-maximization (EM) algorithm ... We use the particle-level information t as the unobserved latent variables", citing Dempster–Laird–Rubin and Shepp–Vardi (1982). — [arXiv:1911.09107](https://arxiv.org/abs/1911.09107)
- Statisticians' statement: "The two main approaches are the expectation–maximization (EM) algorithm with an early stopping [D'Agostini (1995), Vardi, Shepp and Kaufman (1985), Lucy (1974), Richardson (1972)] and a certain variant of Tikhonov regularization [Höcker and Kartvelishvili (1996)]. In HEP terminology, the former is called the D'Agostini iteration." They model the problem as "estimating the intensity function of an indirectly observed Poisson point process" (AoAS 9(3) 1671–1705). — [Kuusela & Panaretos 2015, arXiv:1505.04768](https://arxiv.org/abs/1505.04768)
- Full derivation: OmniFold-HI "corresponds to the expectation-maximization algorithm and is equivalent to iterative Bayesian unfolding (IBU)". The paper gives an explicit E-step, q(C_i|E_j) = p(E_j|C_i)p_n(C_i)/Σ_i′ p(E_j|C_i′)p_n(C_i′), and M-step, and shows the methods "are maximum likelihood estimators". — [Falcão & Takacs, JHEP07 (2026) 012, arXiv:2507.06291](https://arxiv.org/abs/2507.06291)
- Software-level confirmation: TUnfold's `TUnfoldIterativeEMV17` class header states it "implements the iterative EM unfolding method (also called D'Agostini Method or 'iterative Bayesian method')". — [TUnfoldIterativeEM.h](https://gitlab.cern.ch/RooUnfold/RooUnfold/-/blob/master/src/RooUnfold/TUnfold/TUnfoldIterativeEM.h)
- **Selection as thinning (Kuusela & Panaretos)**: "the first thing that can happen is that it might not be observed at all due to the limited efficiency and acceptance of the device. Mathematically, this corresponds to thinning of the Poisson process ... The thinned process M* is a Poisson point process with intensity function f*(s) = ε(s)f(s), where ε(s) = P(Z_i = 1|X_i = s) is the efficiency". The unknown intensity is "modeled using a B-spline", with y|β ∼ Poisson(Kβ). The same paper says bin-by-bin unfolding "provably accounts for smearing effects incorrectly through a multiplicative efficiency correction", and that Choudalakis's Bayesian solution "seems to have seldom been used in practice thus far". — [arXiv:1505.04768](https://arxiv.org/abs/1505.04768)
- History: "Shepp and Vardi, and independently Kondor, have introduced it [Richardson–Lucy iteration] into physics." Zech also gives an algorithm to fix the number of iterations. — [Zech 2013, arXiv:1210.5177](https://arxiv.org/abs/1210.5177)
- Schmitt's comparison labels D'Agostini's iteration "EM method" (columns n = 0, 20, 100, 1000 iterations) next to IDS. — [Schmitt 2016, arXiv:1611.01927](https://arxiv.org/abs/1611.01927)
- Cowan on naming: "As the procedure uses Bayes' theorem in an intermediate step it has been called a 'Bayesian method'. This should imply, however, that the estimators summarize in some way a joint posterior pdf ..., and one would need to specify a joint prior." That is, IBU is not a posterior. — [Cowan 2002](https://www.ippp.dur.ac.uk/old/Workshops/02/statistics/proceedings/cowan.pdf)
- Efficiency as a separate input: IBU implementations take "Detection efficiencies for the cause distribution" (with uncertainties) separately from the migration matrix. — [PyUnfold source](https://github.com/jrbourbeau/pyunfold/blob/master/pyunfold/unfold.py)
- RooUnfold builds R event by event from simulated (reco, truth) pairs with `Fill`/`Miss`. Even the "binned" response is an empirical average over events. — [RooUnfoldExample.py](https://gitlab.cern.ch/RooUnfold/RooUnfold/-/blob/master/examples/RooUnfoldExample.py)
- **Within-bin shape (HEP's version of pastry's interim/within-bin prior)**: "This response matrix depends on the unknown shape of the true spectrum, leading to a fundamental systematic uncertainty". Regularising via wide bins "lead[s] to a non-trivial bias". The proposed fix is "unfolding with fine bins first, followed by constructing confidence intervals for linear functionals of the fine-bin counts" (JINST 17 P10013 (2022)). — [Stanley, Patil, Kuusela, arXiv:2111.01091](https://arxiv.org/abs/2111.01091)
- **Noisy response estimates**: binned response matrices estimated from finite MC are noisy, and "the noise in the estimated response matrix can inadvertently regularize the problem". The alternative is conditional density estimation of the response kernel, then binning (JINST 21 (2026) P03029). — [Zhu, Marini, Kuusela, Wasserman, arXiv:2509.00892](https://arxiv.org/abs/2509.00892)
- **Simulation prior dependence** (HEP's interim-prior problem): conditional generative unfolding learns p(truth|reco) under the training simulation. IcINN "adjusts for deviations between simulated training samples and data" by iterating. SPINUP is "independent of the prior from the simulated training data" because it fits a forward-mapped generative model. AUSSIE's loss "directly yields solutions with minimal dependence on the reference simulation". — [arXiv:2212.08674](https://arxiv.org/abs/2212.08674), [arXiv:2507.15084](https://arxiv.org/abs/2507.15084), [arXiv:2602.24282](https://arxiv.org/abs/2602.24282)
- **Per-event outputs**: cINN unfolding "allows for a per-event statistical interpretation". Backes et al. derive "probabilistic single-event unfolded distributions, together with their uncertainties and correlations" from transfer matrices. — [arXiv:2006.06685](https://arxiv.org/abs/2006.06685), [arXiv:2310.17037](https://arxiv.org/abs/2310.17037)
- **Selection, acceptance, fakes and background in unbinned methods**: OmniFold handles fakes in step 1 only and inefficiency in step 2 (with prior or average-weight fill-in). OmniFold-HI adds "background, detector acceptance, efficiency"; GenFoldG/C add impurities and acceptance/efficiency. — [hep-lbdl/OmniFold README](https://github.com/hep-lbdl/OmniFold), [arXiv:2507.06291](https://arxiv.org/abs/2507.06291), [arXiv:2509.02708](https://arxiv.org/abs/2509.02708)
- **Smooth densities trained through the marginal likelihood**: NEB maximises the log-marginal likelihood of noisy observations under a neural source density and compares "biased and de-biased estimators, and ... variational approaches". This is the HEP-adjacent precedent for pastry's IWAE-style flow training. — [arXiv:2011.05836](https://arxiv.org/abs/2011.05836)
- IBU/EM is also used outside collider physics, e.g. for quantum-computer readout-error mitigation. — [Nachman et al. 2019, arXiv:1910.01969](https://arxiv.org/abs/1910.01969)

### Inferences (design implications for pastry)
- **Binned HEP unfolding is a special case of pastry's input.** Expanding a response matrix R (K_reco × K) and reco counts n_b into per-object rows gives L_ik = R_{b(i),k}, or one row per reco bin with multiplicity weights. pastry could provide `from_response_matrix(R, counts, efficiency)` so HEP users get FBU-like posteriors and the HEP benchmarks (Kuusela–Panaretos, Stanley–Patil–Kuusela, the Landscape datasets) become test cases.
- **Selection effects ≙ HEP efficiencies.** If S_k = P(selected | bin k), IBU's division by ε_k gives the pastry EM update ρ_k ∝ ρ_k S_k^{-1} Σ_i L_ik / Σ_j L_ij ρ_j. In a Bayesian treatment this is the thinned-Poisson or conditional likelihood Π_i (Σ_k L_ik ρ_k) / (Σ_k S_k ρ_k)^N. Kuusela & Panaretos state the thinning model explicitly (f* = ε f), but the pastry-specific update is my derivation, not a quoted result.
- **Early stopping is regularisation.** HEP warns that the IBU iteration count is a hidden regulariser (Cowan; Schmitt; Kuusela–Panaretos). pastry should state that its EM/MAP baseline is the "IBU limit" and favour explicit priors (Dirichlet α, smoothness or GP) whose strength can be fit by empirical Bayes or marginalised.
- **Interim-prior correction has a direct HEP analogue.** Per-object posteriors computed under a survey "interim prior" are like generative unfolding trained on a simulation prior. HEP fixes it by iterating (IcINN, OmniFold) or by fitting the source distribution through the forward model (NEB, SPINUP). pastry's division by the interim prior plus a hierarchical fit is the NEB/SPINUP route.
- **Wide-bin bias transfers.** If pastry users choose coarse K, the per-bin L_ik implicitly assumes a within-bin shape. Following Stanley–Patil–Kuusela, recommend fine bins internally and report coarse summaries as functionals with intervals.

### Gaps
- I did not retrieve D'Agostini's 1995 NIM A paper or the Shepp–Vardi 1982 and Vardi–Shepp–Kaufman 1985 originals; the equivalence is cited through the secondary statements above.
- No HEP tool was found that takes heteroscedastic per-object likelihood vectors directly; this is an absence finding from the census, not a quoted statement.

## 3. Uncertainty quantification in unfolding

### Takeaway
HEP practice is dominated by error propagation (Adye-corrected IBU covariances) and by bootstrap or toy replicas, with systematics from re-running on varied simulations or from nuisance parameters in a likelihood (FBU, TRExFitter-style profile likelihood, RooFitUnfold, Profile OmniFold). The key statistical lesson, demonstrated by Kuusela and collaborators, is that **regularisation bias destroys frequentist coverage of both naive bootstrap and Bayesian credible intervals**. Fixes are bias-corrected bootstrap, shape-constrained strict bounds, and fine-bin unfolding with intervals on wide-bin functionals. ML unfolding adds ensembling: 4–10 networks is typical, 100 in ATLAS. Its cost runs to O(100) bootstrap or ensemble re-trainings and 500–10,000 GPU hours.

### Cited Findings
- **Bias–variance and choice of strength**:
  - In regularised unfolding, "In achieving a reduction in variance one inevitably introduces some bias"; the parameter τ "determines the balance between the bias and variance". — [PDG §40.2.5](https://pdg.lbl.gov/2024/reviews/rpp2024-rev-statistics.pdf)
  - Cowan lists criteria: minimise the bin-averaged mean squared error, use a Poisson-weighted MSE, or require the "estimated bias squared ... be not larger than its own estimated variance". For iterative methods, "The number of iterations should be decided upon before looking at the actual data, for example by using Monte Carlo test data. Continuing to iterate brings increasingly large variances". — [Cowan 2002](https://www.ippp.dur.ac.uk/old/Workshops/02/statistics/proceedings/cowan.pdf)
  - Two procedures are compared, the L-curve scan and minimisation of the average global correlation coefficient ρ_avg; "the ρ_avg minimisation yields a stronger regularisation than the L-curve method". The bin-by-bin method "is clearly disfavoured". — [Schmitt 2016, arXiv:1611.01927](https://arxiv.org/abs/1611.01927)
  - Zech gives an algorithm "to fix the number of iterations to an optimal value" for Richardson–Lucy. — [arXiv:1210.5177](https://arxiv.org/abs/1210.5177)
  - PyUnfold stops when a test statistic between successive iterations drops below `ts_stopping`. — [PyUnfold source](https://github.com/jrbourbeau/pyunfold/blob/master/pyunfold/unfold.py)
  - Gagunashvili chooses the regularisation parameter by bootstrap MISE minimisation. — [arXiv:2402.12990](https://arxiv.org/abs/2402.12990)
- **Empirical Bayes and coverage (Kuusela & Panaretos 2015)**: the regularisation strength (the scale δ of a smoothness prior on B-spline coefficients) is chosen by "marginal maximum likelihood estimation", using a Monte Carlo EM algorithm with a single-component Metropolis–Hastings sampler. The authors note that "little is known about the coverage properties" of HEP error-propagation intervals. They find that "neither Bayesian credible intervals nor standard bootstrap confidence intervals succeed in achieving good frequentist coverage ... due to the inherent bias". Their "iteratively bias-corrected bootstrap" achieves "nearly nominal frequentist coverage with only a modest increase in interval length", demonstrated on the CMS Z-boson mass spectrum. — [arXiv:1505.04768](https://arxiv.org/abs/1505.04768)
- **Shape-constrained strict bounds** (Kuusela & Stark): "Previous unfolding methods use ad hoc discretization and regularization, resulting in confidence intervals that can have significantly lower coverage than their nominal level. Instead of regularizing using a roughness penalty or stopping iterative methods early, we impose physically motivated shape constraints: positivity, monotonicity, and convexity". The uncertainty is a "nonparametric confidence set for the true spectrum, consisting of all those spectra that satisfy the shape constraints and that predict the observations within an appropriately calibrated level". — [arXiv:1512.00905](https://arxiv.org/abs/1512.00905)
- **Risk-based selection of strength**: TUnfold V17.9 offers `ScanSURE`, a Stein's-unbiased-risk-estimate scan with effective degrees of freedom `GetDF`, for the Tikhonov τ and for the IBU iteration count (`TUnfoldIterativeEMV17::ScanSURE(nIterMax, ...)`). — [TUnfold.h](https://gitlab.cern.ch/RooUnfold/RooUnfold/-/blob/master/src/RooUnfold/TUnfold/TUnfold.h), [TUnfoldIterativeEM.h](https://gitlab.cern.ch/RooUnfold/RooUnfold/-/blob/master/src/RooUnfold/TUnfold/TUnfoldIterativeEM.h)
- **GP unfolding** sets kernel hyperparameters either by simulation closure or, following "GP methods", by maximising the marginal likelihood, log P(n; θ) = log ∫ P(n|µ) P(µ; θ) dµ. That is empirical Bayes, as in Kuusela–Panaretos. — [Bozson, Cowan, Spanò, arXiv:1811.01242](https://arxiv.org/abs/1811.01242)
- **Wide-bin UQ (Stanley, Patil, Kuusela 2022)**: One-at-a-time Strict Bounds (OSB) and Prior-Optimized (PO) intervals give "well-calibrated frequentist confidence intervals even in constrained and rank-deficient settings". PO intervals have "provably correct coverage", are "only slightly wider" than OSB and "display little sensitivity to the choice of the prior". By contrast, guaranteed coverage "is not the case for Bayesian methods". — [arXiv:2111.01091](https://arxiv.org/abs/2111.01091), JINST 17 P10013
- **Per-tool UQ outputs**:
  - PyUnfold returns per-bin stat and response-MC errors with Adye propagation and a Multinomial or Poisson data covariance, but no full covariance matrix. — [source](https://github.com/jrbourbeau/pyunfold/blob/master/pyunfold/mix.py)
  - QUnfold estimates covariance from annealer reads plus optional Poisson toys. — [source](https://github.com/Quantum4HEP/QUnfold/blob/main/src/qunfold/qunfolder.py)
  - PyFBU returns MCMC traces for truth bins and nuisances. — [source](https://github.com/pyFBU/fbu/blob/master/fbu/PyFBU.py)
  - GP unfolding returns the "full statistical covariance matrix". — [arXiv:1811.01242](https://arxiv.org/abs/1811.01242)
  - Wiener-SVD returns the unfolded covariance plus an "additional smearing matrix" to apply to predictions. — [GitHub](https://github.com/BNLIF/Wiener-SVD-Unfolding)
  - D'Agostini 2010 propagates all uncertainties by Monte Carlo sampling from pdfs, so "the final uncertainty does not rely on the assumption of normality". — [arXiv:1010.0632](https://arxiv.org/abs/1010.0632)
- **Calibration of Bayesian or amortised posteriors**: NPU and FBU pulls are "well centered around zero with unit variance" at moderate smearing ("proper calibration"), but "NPU appears to underestimate the variance at low smearing". — [arXiv:2509.06370](https://arxiv.org/abs/2509.06370)
  - Nuclear RMLE intervals "maintain correct coverage". — [arXiv:2511.16687](https://arxiv.org/abs/2511.16687)
  - The empirical-Bayes γ-ray method reports simultaneous bands from NUTS. — [arXiv:2606.24971](https://arxiv.org/abs/2606.24971)
- **Nuisance parameters and systematics**:
  - FBU marginalises background-normalisation and object-systematic nuisance parameters. — [PyFBU source](https://github.com/pyFBU/fbu/blob/master/fbu/PyFBU.py), [arXiv:1201.4612](https://arxiv.org/abs/1201.4612)
  - Unbinned Profiled Unfolding profiles them. — [arXiv:2302.05390](https://arxiv.org/abs/2302.05390)
  - So does Profile OmniFold. — [arXiv:2512.07074](https://arxiv.org/abs/2512.07074)
  - The RooFit likelihood interface handles systematics. — [arXiv:2510.15129](https://arxiv.org/abs/2510.15129)
  - ATLAS keeps profile-likelihood folding utilities (C++, Apache-2.0). — [TRExStats/unfoldingcode](https://gitlab.cern.ch/TRExStats/unfoldingcode)
- **ML unfolding practice (CMS, T2K, ATLAS, STAR, H1 authors)**:
  - Statistical uncertainties come from bootstrapping and systematics from "varying the MC sample". A new "uncertainty associated with the neural network initialization" is "generally small (< 2%)".
  - "Most analyses used some form of ensembling, with numbers of ensembles typically ranging from 4 to 10", while ATLAS Z+jets and track functions "used an ensemble of 100 models".
  - Bootstrapping needs "about 50 to 100 individual bootstrap runs".
  - Totals are "approximately 500 and 10,000 GPU hours".
  - An open question is "Are there viable alternatives to bootstrapping?"
  - — [Canelli et al. 2025, arXiv:2507.09582](https://arxiv.org/abs/2507.09582)
- **Bayesian networks for ML unfolding**: the Landscape paper uses Bayesian generative networks and a Bayesian OmniFold ("bOmniFold error bar is based on drawing 20 Bayesian samples"), noting these error bars are "a learned approximation to the true uncertainty". — [arXiv:2404.18807](https://arxiv.org/abs/2404.18807)
- **Reporting correlated errors**:
  - Unfolded results are reported as "estimators µ̂ and the corresponding covariance matrix". "If only the unfolded distributions and their separate covariance matrices are reported ... information on such correlations is not retained." — [PDG §40.2.5](https://pdg.lbl.gov/2024/reviews/rpp2024-rev-statistics.pdf)
  - For unbinned results, Arratia et al. distinguish "two-point" uncertainties (a varied entry with the same structure as the nominal) from "replica" uncertainties ("many entries", the standard deviation across replicas, e.g. bootstrap). Bootstrapping is also used to correlate different measurements. — [Arratia et al., JINST 17 P01024, arXiv:2109.13243](https://arxiv.org/abs/2109.13243)
  - ATLAS publishes `weights_nominal` plus `weights_*` uncertainty columns per event. — [ATLAS GitLab](https://gitlab.cern.ch/atlas-physics/public/sm-z-jets-omnifold-2024)

### Inferences
- pastry's UQ should go beyond Bayesian credible intervals. Offer (i) posterior samples; (ii) a coverage-checking utility (simulate from a known ρ, refit, report coverage), as RooFitUnfold does for bias, variance and coverage; and (iii) optional frequentist add-ons for coarse-bin functionals, such as a bias-corrected bootstrap. HEP evidence shows regularised or prior-driven intervals can undercover where bias is large.
- Output format: a matrix of posterior draws of ρ (draws × K) is a "replica" representation in the Arratia sense. pastry should export it directly, with the covariance as a convenience, and allow joint exports across several fitted quantities so cross-correlations survive. That is the PDG warning.
- Empirical Bayes for the smoothness scale (Kuusela–Panaretos; also Bozson et al. for GP hyperparameters) is a natural default for pastry's smoothness and GP priors, with full hierarchical sampling as an option. For point-estimate (EM/MAP) modes, a SURE or effective-degrees-of-freedom criterion (TUnfold V17.9) is a cheap, documented alternative to choosing the iteration count by eye.
- Simplex-constrained outputs have rank-deficient covariances, since the fractions sum to 1. A 2026 neutrino paper shows that rank deficiency from shared events across distributions makes naive χ² "unphysically inflated" and proposes a "range-projected χ²" with N_bins − N_null degrees of freedom. pastry's goodness-of-fit and comparison utilities should therefore work in the K−1-dimensional subspace (or log-ratio coordinates) and handle several summaries computed from the same objects jointly. — [Martinez Lopez et al. 2026, arXiv:2609.24796](https://arxiv.org/abs/2609.24796)

### Gaps
- I found no published coverage study for PyUnfold, OmniFold-bootstrap or generative-unfolding intervals beyond pulls in NPU and Bayesian-network error bars in the Landscape paper.
- Kuusela & Stark's journal venue was not retrieved (the arXiv listing shows no journal reference), and neither were their numerical coverage results.

## 4. Recommendations from PHYSTAT, reviews and experiment practice

### Takeaway
The consensus across PDG and Cowan reviews, Schmitt, Kuusela and the 2022/2025 unbinned-unfolding community papers has five strands:
- Consider folding instead of unfolding, and publish what makes folding possible: the response with its nuisance dependence, plus backgrounds.
- Always report the full covariance, or replicas.
- Never use bin-by-bin corrections.
- Fix the regularisation or iteration choice before looking at data, with MC closure, stress and coverage tests.
- For unbinned results, publish per-event weights with replica and two-point uncertainty weights, run blinded, and ensemble.

### Cited Findings
- **Folding and preserving information**:
  - "In many problems it is not necessary to unfold ... if the goal is to compare the result with the prediction of an existing theory". "If a particularly important measured distribution is to retain its value, then both the data and the response matrix should be preserved." — [Cowan 2002](https://www.ippp.dur.ac.uk/old/Workshops/02/statistics/proceedings/cowan.pdf)
  - "To record enough information for a future analysis using folding one must provide all of the ingredients ...: the primary data n, the auxiliary measurements u, and also the response matrix R(θ) and expected backgrounds β(θ) as functions of the nuisance parameters." — [PDG §40.2.5](https://pdg.lbl.gov/2024/reviews/rpp2024-rev-statistics.pdf)
- **Hypothesis tests on unfolded histograms**: "Such unfolded histograms are then compared to theoretical predictions ... When doing this, even informally, one is testing hypotheses. However, a more fundamentally sound way to test hypotheses is to smear the theoretical predictions by simulating detector response and then comparing to the data without unfolding". The paper asks to what extent testing after unfolding reproduces testing by folding; its quantitative conclusions were not retrieved. — [Cousins, May, Sun 2016, arXiv:1607.07038](https://arxiv.org/abs/1607.07038)
- **Bin-by-bin**:
  - "Most of these results relied on the questionable bin-by-bin heuristic (8), while the EM algorithm (3) and various forms of penalization (6) were also used". This was the authors' survey of CMS unfolded results from 2012; "the unfolding methodology used in LHC data analysis is not well established". — [Kuusela & Panaretos 2015](https://arxiv.org/abs/1505.04768)
  - Bin-by-bin "is clearly disfavoured". — [Schmitt 2016](https://arxiv.org/abs/1611.01927)
- **PHYSTAT 2011**: the workshop "on Statistical Issues Related to Discovery Claims in Search Experiments and Unfolding" (CERN, 17–20 Jan 2011; proceedings CERN-2011-006, eds. Prosper & Lyons) included Adye's RooUnfold paper. — [arXiv:1105.1160](https://arxiv.org/abs/1105.1160), [RooUnfold README](https://gitlab.cern.ch/RooUnfold/RooUnfold)
  - Its unfolding day (20 Jan 2011), per the session programme:
    - Introduction: Lyons, "Introductory comments".
    - Method talks: Panaretos, "A statistician's view"; Blobel, "Unfolding methods for particle physics"; Zech, "Regularization and error assignment in unfolding"; Kartvelishvili, "Unfolding with SVD"; Bierwagen, "Bayesian Unfolding"; Malaescu, IDS; Tackmann, SVD experience; Schmelling, "Regularization by control of resolution function"; Dembinski, "ARU – towards optimal unfolding"; Adye, RooUnfold.
    - Experiment talks: CMS, ATLAS (Choudalakis) and ALICE unfolding.
    - Closing "Summary and discussion" led by Panaretos.
  - — [PHYSTAT2011 unfolding session programme (Indico)](https://indico.cern.ch/event/107747/sessions/2573/session-timetable.pdf)
- **Pedagogical reviews**:
  - Spanò, "Unfolding in particle physics: a window on solving inverse problems", EPJ Web Conf. 55 (2013) 03002, doi:10.1051/epjconf/20135503002 (SOS 2012 IN2P3 School of Statistics). The bibliographic record was verified via search; the full text returned HTTP 403, so its content is not summarised here. — [EPJ Web Conf. record](https://www.epj-conferences.org/10.1051/epjconf/20135503002)
  - Cowan 2002. — [Cowan 2002](https://www.ippp.dur.ac.uk/old/Workshops/02/statistics/proceedings/cowan.pdf)
  - Schmitt 2016. — [arXiv:1611.01927](https://arxiv.org/abs/1611.01927)
  - Croft 2026 (BRU). — [arXiv:2604.01069](https://arxiv.org/abs/2604.01069)
- **2026 developments**:
  - Production analyses still run high-dimensional *binned* unfolding. CMS measured 25 N-subjettiness observables simultaneously, "unfolded to the level of stable particles" (JHEP 09 (2026) 258), and published a triple-differential Z+jet cross section "after unfolding for detector effects in all three dimensions simultaneously"; the unfolding methods were not checked. — [arXiv:2604.25538](https://arxiv.org/abs/2604.25538), [arXiv:2609.39653](https://arxiv.org/abs/2609.39653)
  - An ATLAS thesis presents a "full-phase-space measurement of the Z+jets production cross section ... differential in the kinematics of every final state charged particle". — [Greif 2026, arXiv:2608.28449](https://arxiv.org/abs/2608.28449)
  - RABBIT, a TensorFlow-2 framework for "binned profile maximum likelihood fits with thousands of bins and nuisance parameters" on CPU and GPU, is relevant infrastructure for likelihood-based unfolding. Its abstract does not mention unfolding. — [Walter et al. 2026, arXiv:2608.26376](https://arxiv.org/abs/2608.26376)
  - The ScatterPrism study of conditional flow matching for "detector modeling and unfolding" in nuclear physics finds that "its standard training loss is fundamentally misleading", because the "CFM loss plateaus prematurely, obscuring ongoing physical refinement" (JINST 21, C07012 (2026)). — [arXiv:2604.01313](https://arxiv.org/abs/2604.01313)
- **Unbinned publication (Arratia et al. 2022)**: "There is currently no community standard for publishing unbinned data". The paper proposes a format and storage scheme, classifies uncertainties as two-point or replica, and discusses procedure uncertainties ("reinitializing with different starting parameters, bootstrapping, etc. Bayesian neural networks may also ... help"). — [arXiv:2109.13243](https://arxiv.org/abs/2109.13243)
- **Practical guide (2025)**: analyses were often "blinded", using MC in place of data while configuring. "Most analyses found that ensembling improved the stability". "Ensembling the weights themselves facilitates publishing a fully unbinned dataset that does not depend on any particular choice of binning". The guide also recommends putting irreducible backgrounds into the initial MC as negative-weight events. — [arXiv:2507.09582](https://arxiv.org/abs/2507.09582)
- **Software standardisation**: "we have assembled two software packages for the OmniFold unbinned unfolding method that should serve as the starting point for any future analyses" (the RooUnfold-based BDT class and PyPI `omnifold`). — [Milton et al. 2025, arXiv:2503.09720](https://arxiv.org/abs/2503.09720)
- **Response-matrix estimation**: consider conditional density estimation of the kernel rather than raw binned counts, especially with small MC samples. — [arXiv:2509.00892](https://arxiv.org/abs/2509.00892)
- **Classic method references** (verified on arXiv):
  - Höcker & Kartvelishvili SVD, NIM A372 (1996) 469. — [hep-ph/9509307](https://arxiv.org/abs/hep-ph/9509307)
  - Blobel 2002, "An Unfolding Method for High Energy Physics Experiments". — [hep-ex/0208022](https://arxiv.org/abs/hep-ex/0208022)
  - Malaescu 2009, IDS. — [arXiv:0907.3791](https://arxiv.org/abs/0907.3791)
  - Schmitt 2012, TUnfold. — [arXiv:1205.6201](https://arxiv.org/abs/1205.6201)
  - Gagunashvili 2026 discusses internal quality criteria for unfolded results. — [arXiv:2603.03168](https://arxiv.org/abs/2603.03168)

### Inferences
- For pastry: (1) make "forward-folding" checks first-class, i.e. a posterior predictive of the observed per-object data. (2) Export a reusable artefact (the L_ik matrix or its generator, plus posterior draws) so downstream users can re-fit or fold new models, mirroring the "publish R(θ)" advice. (3) Ship closure, stress and coverage test helpers. (4) Discourage bin-by-bin-like shortcuts such as stacking or averaging per-object PDFs; HEP evidence says these are biased (this analogy is my inference). (5) For pastry's flow and flow-matching density models, monitor posterior-predictive or summary-statistic metrics rather than training loss alone. ScatterPrism found the CFM loss plateaus while physical fidelity keeps improving.

### Gaps
- The texts of Spanò 2013 (HTTP 403), Blobel's PHYSTAT 2011 talk and the PHYSTAT 2011 summary discussion were not retrieved, so their specific recommendations are not captured. A web search found no document labelled as "recommendations" of the PHYSTAT 2011 unfolding workshop.
- Experiment-internal guidance (e.g. CMS Statistics Committee pages, ATLAS internal notes) was not checked; ROOT's reference documentation now sits behind CERN single sign-on.
- No PHYSTAT event devoted to unbinned or ML unfolding after 2022 was identified in this pass. The 2022 and 2025 community papers (Arratia et al.; Canelli et al.) serve as the de facto guidance.

## 5. API conventions and ease of use

### Takeaway
HEP unfolding APIs follow four patterns:
- **ROOT object pipelines**: RooUnfold's response object, then an algorithm object, then histograms and error matrices.
- **One-shot functional calls**: PyUnfold's `iterative_unfold(...)` returning a dict.
- **Mutable configuration objects with `run()`**: PyFBU, which sets attributes and stores `trace`.
- **Class-based ML drivers**: `omnifold.MultiFold(...).Unfold()`, `QUnfolder(...).solve_*()` returning `(sol, cov)`, and Bunse's sklearn-style `fit`/`predict`.

The simplest documented Python entry point is PyUnfold's single function call. ML tools trade simplicity for TensorFlow/GPU dependencies and many hyperparameters.

### Cited Findings
- PyUnfold: a pure-function call with array-like inputs and dict output; callbacks (`Logger`, `SplineRegularizer`) modify behaviour; `return_iterations=True` returns a pandas DataFrame of all iterations. — [source](https://github.com/jrbourbeau/pyunfold/blob/master/pyunfold/unfold.py)
- PyFBU: attribute-assignment configuration (`myfbu.data = ...; myfbu.run()`), with results stored on the object (`myfbu.trace`, `myfbu.nuisancestrace`). — [tutorial](https://github.com/pyFBU/fbu/blob/master/tutorial.ipynb)
- omnifold: data wrapper (`DataLoader`), user- or package-supplied Keras models (`MLP`, `PET`), driver class (`MultiFold`) and `.reweight()`, plus plotting helpers (`SetStyle`, `HistRoutine`). "The OmniFold codebase includes a suite of recommended default hyperparameters that were generally sufficient for some analyses." — [README](https://github.com/ViniciusMikuni/omnifold), [arXiv:2507.09582](https://arxiv.org/abs/2507.09582)
- QUnfold: constructor with `(response, measured, binning, lam)`, explicit `initialize_qubo_model()`, solver-specific methods returning `(sol, cov)`, and ROOT converters (`TH1_to_numpy`, `TH2_to_numpy`). — [README](https://github.com/Quantum4HEP/QUnfold)
- qunfold (Bunse): scikit-learn estimator conventions (`fit(X, y)`, `predict(X)`), composing losses and representations. — [README](https://github.com/mirkobunse/qunfold)
- RooUnfold: an event-wise `Fill`/`Miss` response-building API mirroring the simulation loop; algorithms are selected by class (`RooUnfoldBayes`, `RooUnfoldSvd`, ...) with the regularisation parameter as a constructor argument. — [example](https://gitlab.cern.ch/RooUnfold/RooUnfold/-/blob/master/examples/RooUnfoldExample.py)

### Inferences
- pastry can borrow the best of each:
  - PyUnfold's one-call functional entry point with sensible defaults.
  - An sklearn/emcee-like object API (`PopulationModel(prior=...).fit(L, ...)` returning a results object with `.samples`, `.mean`, `.cov`, `.summary()` and `.predictive()`).
  - RooUnfold-style adapters (`from_response_matrix`, `from_samples`) so HEP users can switch.
  - Always return the full covariance and the samples.
- Avoid PyFBU's mutable-attribute style and hard dependence on a single probabilistic programming backend; PyFBU's Theano/PyMC3 lock-in is the cautionary tale.

### Gaps
- I found no survey of practitioners' views on which unfolding tool is easiest to use. RooUnfold's status as the common interface is documented by its maintainers (Brenner et al.); "ease of use" comparisons remain anecdotal.
