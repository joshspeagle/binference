# Quantification, label shift and mixture-proportion estimation software (census for pastry, Oct 2026)

## ML quantification and label-shift software (QuaPy, BayesianCC/labelshift, BBSE/RLLS/MLLS, LeQua)

### Takeaway
QuaPy (v0.2.3, released 5 Oct 2026, BSD-3) is the reference Python library for quantification and, since 2025, the most complete one for uncertainty: bootstrap confidence regions for any aggregative method and four Bayesian quantifiers (BayesianCC, BayesianKDEy, BayesianMAPLS and PQ), sampled with NumPyro NUTS or Stan, plus temperature calibration of the posteriors. The Python mlquantify (v0.5.1, BSD-3) now mirrors much of this. qunfold (Bunse) links quantification to unfolding through response-matrix losses and Tikhonov smoothing but reports no uncertainty. The other label-shift codes (abstention, Czyż's labelshift, quantificationlib) are point-estimate or research code that is no longer developed.

### Cited Findings
**QuaPy (HLT-ISTI, Pisa): the reference library**
- Identity and status: QuaPy v0.2.3 was uploaded to PyPI on 2026-10-05; the first PyPI release (0.1.4) was 2021-05-10; there are 11 releases; it needs Python >=3.8 and has a BSD licence. The core dependencies are scikit-learn, pandas, tqdm, matplotlib, joblib, xlrd, abstention, ucimlrepo and certifi. A `bayes` extra adds jax, jaxlib, numpyro and pystan, and a `neural` extra adds torch and geotorch. Docs are at hlt-isti.github.io/QuaPy. — [PyPI quapy JSON](https://pypi.org/pypi/quapy/json), [PyPI](https://pypi.org/project/quapy/)
- GitHub (2026-10): 73 stars, 15 forks, 1 open issue; created 2020-12-15 and last pushed 2026-10-06. Licence BSD-3-Clause. — [GitHub HLT-ISTI/QuaPy](https://github.com/HLT-ISTI/QuaPy)
- Test and CI: 16 unit-test modules (`test_*.py`) ship in `quapy/tests`, among them `test_confidence.py`, `test_bayesian_utils.py`, `test_calibration.py`, `test_labelshift.py` and `test_replicability.py`, with integration tests kept separate. GitHub Actions runs `python -m unittest` over a Python-version matrix and builds the Sphinx docs. — [QuaPy tests dir](https://github.com/HLT-ISTI/QuaPy/tree/master/quapy/tests), [ci.yml](https://github.com/HLT-ISTI/QuaPy/blob/master/.github/workflows/ci.yml)
- Methods, from the class definitions in the 2026-10 source:
  - Aggregative: CC, PCC, ACC, PACC, EMQ (SLD), BBSEhard, BBSEsoft, RLLS, HDy, DyS, SMM, DMy, EDy, KDEyML, KDEyHD, KDEyCS, LEIP, and the threshold family (T50, MAX, X, MS, MS2).
  - Non-aggregative: MLPE, DMx, EDx, ReadMe (Hopkins & King).
  - Meta and other: Ensemble, MedianEstimator, SCMQ, MCSQ, MCMQ, a QUnfoldWrapper, and neural QuaNet, HistNetQ and GMNet.
  - Confidence and Bayesian: AggregativeBootstrap, BayesianCC, PQ, BayesianKDEy, BayesianMAPLS. — [aggregative.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/aggregative.py), [confidence.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/confidence.py), [_bayesian.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/_bayesian.py)
- API changes in 0.2.0 (CHANGE_LOG):
  - `fit(data: LabelledCollection)` became the scikit-learn-like `fit(X, y)`.
  - `predict` was added, with `quantify` kept as an alias.
  - The classifier and validation split are now fixed at construction, so they can no longer be passed to `fit`.
  - "aggregative bootstrap for deriving confidence regions (confidence intervals, ellipses in the simplex, or ellipses in the CLR space)" was added, which "applies resampling only to the aggregation phase, thus avoiding to train many quantifiers".
  - BayesianCC moved to `confidence.py`. — [CHANGE_LOG.txt](https://github.com/HLT-ISTI/QuaPy/blob/master/CHANGE_LOG.txt)
- In 0.2.1 (CHANGE_LOG):
  - Added: Bayesian KDEy and Bayesian MAPLS, "temperature calibration utilities for Bayesian confidence-aware methods", CLR and ILR compositional transforms, Aitchison and ILR kernels for KDEy, a DirichletProtocol, the ReadMe method, RLLS, and EDx/EDy ported from quantificationlib.
  - 0.2.2 added HistNetQ and 0.2.3 added GMNet. — [CHANGE_LOG.txt](https://github.com/HLT-ISTI/QuaPy/blob/master/CHANGE_LOG.txt)
- Uncertainty API:
  - Every confidence-aware method implements `predict_conf(X, confidence_level=0.95)`, which returns `(point_estimate, region)`.
  - A `ConfidenceRegionABC` region provides `point_estimate()`, `coverage(true_value)`, `simplex_portion()`, `montecarlo_proportion()`, `samples` and `__contains__`.
  - Region types: `ConfidenceIntervals` (optional Bonferroni), `ConfidenceEllipseSimplex`, `ConfidenceEllipseCLR`/`ILR` and `ConfidenceIntervalsCLR`/`ILR`. Intervals also have Winkler-score methods. — [confidence.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/confidence.py)
- `AggregativeBootstrap(quantifier, n_train_samples=1, n_test_samples=500, confidence_level=0.95, region='intervals'|'ellipse'|'ellipse-clr', bonferroni=False)` wraps any aggregative quantifier.
  - Resampling the training predictions gives a "model-based" bootstrap, resampling the test predictions gives a "population-based" one, and the two can be combined. — [confidence.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/confidence.py)
  - Docs example: `pacc = AggregativeBootstrap(PACC(), n_test_samples=500, confidence_level=0.95); pacc.fit(Xtr, ytr); pred_prev, conf = pacc.predict_conf(test_X); conf.coverage(true_prev)`. — [examples/16.confidence_regions.py](https://github.com/HLT-ISTI/QuaPy/blob/master/examples/16.confidence_regions.py)
- Moreo & Salvati (LQ 2025, pp. 12–33) compared three bootstrap schemes (model-based, population-based, combined) and three region shapes (intervals, simplex ellipses, CLR ellipses).
  - They conclude that "the combined approach paired with confidence intervals is the most effective".
  - They also warn that "standard bootstrap methods rely on the i.i.d. assumption and therefore cannot provide valid coverage guarantees under prior probability shift … bootstrap intervals merely quantify the internal variability of the quantifier". — [LQ 2025 proceedings PDF](https://lq-2025.github.io/proceedings/CompleteVolume.pdf)
- BayesianCC (Ziegler & Czyż), as implemented in QuaPy, puts Dirichlet priors on the test prevalence π and on each row of P(C|Y).
  - The labelled confusion counts are modelled as `F_yc ~ Multinomial(n_y, P(C|Y))` and the unlabelled predicted-class counts as `N_c ~ Multinomial(N, P(C|Y)ᵀπ)`.
  - Sampling uses NumPyro NUTS (default 500 warm-up and 1000 samples), and an optional `temperature` divides the log-likelihood.
  - Parameters: `prior='uniform'|scalar|array` (the Dirichlet α), `region`, and `bonferroni`. — [_bayesian.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/_bayesian.py), [confidence.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/confidence.py)
- BayesianKDEy's NumPyro model is `prev ~ Dirichlet(α)` with `loglik = Σ_i logsumexp_k(log prev_k + log KDE_k(s_i))`, divided by the temperature and sampled with NUTS.
  - KDE_k is a kernel density estimate of the classifier outputs for class k.
  - This is the Dirichlet-mixture likelihood that pastry uses, with L_ik estimated nonparametrically. — [_bayesian.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/_bayesian.py)
- BayesianMAPLS is a Bayesian version of MLLS/EMQ, after Ye et al., WACV 2024 (code adapted from ChangkunYe/MAPLS).
  - Its log-likelihood is `Σ_i logsumexp_k(log p(k|x_i) + log π_test,k − log π_train,k)`, sampled with NUTS.
  - Priors are 'uniform', 'map' or 'map2'; the chain is initialised from the EM solution. — [_bayesian.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/_bayesian.py)
- PQ ("Precise Quantifier") is a binary Bayesian histogram-matching quantifier (a Bayesian HDy) run through PyStan. It comes from Igiraneza, Fraser & Hinch (2025), who report "narrow prediction intervals with sufficient coverage". — [confidence.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/confidence.py), [arXiv:2507.06061](https://arxiv.org/abs/2507.06061)
- `calibrate_temperature(method, train, val_prot, temp_grid=(0.5,1,1.5,2,5,10,100), nominal_coverage=0.95, criterion='winkler'|'auto')` chooses the temperature by evaluating on validation samples from a protocol.
  - With 'winkler' it takes the lowest mean Winkler score among candidates whose average region size is under a threshold; with 'auto' it takes the candidate whose empirical coverage is closest to nominal. — [_bayesian.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/_bayesian.py)
- Evaluation protocols and LeQua data:
  - QuaPy implements sampling protocols: APP (artificial prevalence), NPP, UPP, DirichletProtocol and DomainMixer.
  - It includes loaders for LeQua 2022 and LeQua 2024 and the normalised match distance error. — [CHANGE_LOG.txt](https://github.com/HLT-ISTI/QuaPy/blob/master/CHANGE_LOG.txt)

**BayesianCC / Czyż's labelshift repository**
- Ziegler & Czyż, "Bayesian Quantification with Black-Box Estimators", arXiv:2302.09159 (Feb 2023), published in TMLR 2024 (openreview Ft4kHrOawZ).
  - They show that adjusted classify and count, BBSE and invariant-ratio estimators "are closely related to the inference in a particular Bayesian model". They give an MCMC scheme with an asymptotic consistency guarantee. — [arXiv:2302.09159](https://arxiv.org/abs/2302.09159)
- The repository pawel-czyz/labelshift (BSD-3; 1 star; last commit 2024-06-13) carries the badge "Project Status: Concept"; it is the code to reproduce the paper's experiments.
  - Installation is `pip install -e .` plus a Snakemake workflow. setup.cfg has version 0.0.1.
  - Modules: `algorithms/bayesian_discrete.py` (NumPyro), `expectation_maximization.py`, `bbse.py`, `ratio_estimator.py` (InvariantRatioEstimator), `classify_and_count.py` and `recalibrate.py`, with pytest tests. — [GitHub pawel-czyz/labelshift](https://github.com/pawel-czyz/labelshift)
- The original model in that repo also puts a Dirichlet prior on P_train(Y). It treats the labelled class counts as Multinomial and the labelled confusion counts as row-wise Multinomial, so the confusion matrix is inferred jointly rather than plugged in. — [bayesian_discrete.py](https://github.com/pawel-czyz/labelshift/blob/main/labelshift/algorithms/bayesian_discrete.py)
- Naming hazard: the PyPI name `labelshift` belongs to a different project, pm25/Label-Shift-Estimation, "A pytorch-based toolbox for label shift estimation" (v0.2.11, 2023-04-08, MIT). — [PyPI labelshift JSON](https://pypi.org/pypi/labelshift/json)

**Other quantification libraries**
- mlquantify (Python; Luiz Fernando Luth Junior & André Maletzke)
  - PyPI: v0.5.1 stable, with a pre-release 0.5.2b3 uploaded 2026-08-03; first release 2024-08-13; 53 releases; BSD-3-Clause; Python >=3.9. Docs are at luizfernandolj.github.io/mlquantify, with the website at coenlab.github.io/mlquantify. — [PyPI mlquantify JSON](https://pypi.org/pypi/mlquantify/json)
  - README: "33 quantification methods".
    - Counting: CC, PCC, ACC, TAC, TX, TMAX, T50, MS, MS2, FM, GACC, GPACC.
    - Matching: DyS, HDy, HDx, SORD, SMM, MMD_RKHS, the KDEy variants, EDy, EDx and others.
    - Likelihood: EMQ, CDE, MLPE.
  - It also has modular solvers and representations, a `ClassifierCalibrator`, confidence regions (`ConfidenceInterval`, `ConfidenceEllipseSimplex`, `ConfidenceEllipseCLR`), `GridSearchQ`, APP/NPP/UPP/PPP protocols and scikit-learn-style `Display` plots. — [mlquantify README](https://github.com/luizfernandolj/mlquantify)
  - The code has an `AggregativeBootstrap` meta-quantifier (`n_train_bootstraps`, `n_test_bootstraps`) and 21 pytest modules (`test_*.py`). The last commit in the clone is dated 2026-10-16. — [mlquantify meta/_classes.py](https://github.com/luizfernandolj/mlquantify/blob/master/mlquantify/meta/_classes.py)
- R mlquantify (Maletzke): a separate CRAN package, "Algorithms for Class Distribution Estimation", v0.2.0 published 2022-01-20, GPL. — [CRAN mlquantify](https://cran.r-project.org/package=mlquantify)
- QuantificationLib (AICGijon, Oviedo)
  - PyPI v0.1.2 (2024-01-31), GPLv3; last commit 2024-04-06; 13 stars; Sphinx docs and codecov.
  - Classes: CC, PCC, AC, PAC, EM, HDy/HDX, MMy, EDy/EDX, CvMy, PWKQuantifier, FriedmanME, SORDy, QUANTy, DeBias and EoQ, plus ordinal ACOrdinal, PDFOrdinaly and an OrdinalQuantificationTree.
  - It is point-estimate only. — [PyPI quantificationlib JSON](https://pypi.org/pypi/quantificationlib/json), [GitHub AICGijon/quantificationlib](https://github.com/AICGijon/quantificationlib)
- qunfold (Mirko Bunse): "a unified framework of algorithms for quantification and unfolding".
  - Install it with `pip install 'qunfold @ git+https://github.com/mirkobunse/qunfold'`; it needs a JAX backend. The code version is 0.1.6-rc4 and the last commit is 2026-07-13. Its pyproject carries an MIT classifier but the repository has no LICENSE file, and GitHub reports no licence. There is one small test module of 8 tests plus performance scripts.
  - Usage: `acc = ACC(RandomForestClassifier(oob_score=True)); acc.fit(X_trn, y_trn); p_hat = acc.predict(X_tst)`. — [qunfold README](https://github.com/mirkobunse/qunfold)
  - Methods: LinearMethod (ACC, PACC, RUN, EDx, EDy, HDx, HDy, KMM), with losses LeastSquares, Blobel, Energy, HellingerSurrogate and TikhonovRegularization. It also has LikelihoodMaximizer (ML with ordinal smoothness penalties) and ExpectationMaximizer.
  - A grep of the package finds no bootstrap or confidence code. — [qunfold methods](https://github.com/mirkobunse/qunfold/tree/main/qunfold/methods)
  - Naming hazard: PyPI `qunfold` (v0.3.5, 2024-09-20, MIT) is Quantum4HEP/QUnfold, "A quantum-based approach to the distribution unfolding problem". It is a different package. — [PyPI qunfold JSON](https://pypi.org/pypi/qunfold/json)
- abstention (Kundaje lab)
  - PyPI v0.1.3.1, 2020-01-22; last commit 2020-11-14. It is unmaintained but still a hard dependency of QuaPy. Neither the repository nor setup.py declares a licence; GitHub reports none. — [GitHub search metadata](https://github.com/kundajelab/abstention)
  - It implements Platt, isotonic, temperature, vector, bias-corrected temperature scaling (BCTS) and no-bias vector scaling calibrators. For label shift it implements EM (Saerens 2002), BBSL (Lipton 2018) and RLLS (Azizzadenesheli 2019). — [abstention README](https://github.com/kundajelab/abstention), [PyPI abstention JSON](https://pypi.org/pypi/abstention/json)
- skada (scikit-adaptation; v0.6.0, 2026-06-17; BSD-3; 167 stars) is a general domain-adaptation toolbox.
  - Its target/label-shift tools are `MMDTarSReweight` (Zhang et al. ICML 2013) and JCPOT (Redko et al. AISTATS 2019).
  - Both are point reweightings, not prevalence estimators with uncertainty. — [skada _reweight.py](https://github.com/scikit-adaptation/skada/blob/main/skada/_reweight.py), [PyPI skada JSON](https://pypi.org/pypi/skada/json)

**Label-shift estimators (ML literature)**
- BBSE (Lipton, Wang & Smola 2018, arXiv:1802.03916) "works even when predictors are biased, inaccurate, or uncalibrated, so long as their confusion matrices are invertible". — [arXiv:1802.03916](https://arxiv.org/abs/1802.03916)
- RLLS: Azizzadenesheli, Liu, Yang & Anandkumar 2019, "Regularized Learning for Domain Adaptation under Label Shifts", arXiv:1903.09734. — [arXiv:1903.09734](https://arxiv.org/abs/1903.09734)
- Alexandari, Kundaje & Shrikumar (ICML 2020; arXiv:1901.06852) show that "combining maximum likelihood with … bias-corrected calibration outperforms both BBSL and RLLS". They prove that the maximum-likelihood objective is concave and give a strategy for estimating the source-domain priors. — [arXiv:1901.06852](https://arxiv.org/abs/1901.06852)
- Garg, Wu, Balakrishnan & Lipton (NeurIPS 2020; arXiv:2003.07554) show that MLLS consistency requires calibration and an invertibility condition, and that BBSE is "roughly equivalent to MLLS for a particular choice of calibration". They decompose the MLLS error into miscalibration and estimation terms. — [arXiv:2003.07554](https://arxiv.org/abs/2003.07554)
- Vaz, Izbicki & Stern (arXiv:1807.03929; JMLR 2019) introduce "ratio estimators" and use a testable weak prior-shift assumption to "build confidence intervals". — [arXiv:1807.03929](https://arxiv.org/abs/1807.03929)
- Other foundational papers:
  - Tasche, "Fisher consistency for prior probability shift" (JMLR 18, 2017; arXiv:1701.05512). — [arXiv:1701.05512](https://arxiv.org/abs/1701.05512)
  - Forman (2008), "Quantifying counts and costs via classification", DMKD 17:164–206. — [doi:10.1007/s10618-008-0097-y](https://doi.org/10.1007/s10618-008-0097-y)
  - González et al., "A Review on Quantification Learning", ACM Computing Surveys 50 (2017/18). — [doi:10.1145/3117807](https://doi.org/10.1145/3117807)
  - Esuli, Fabris, Moreo & Sebastiani, *Learning to Quantify* (Springer Information Retrieval Series, 2023). — [doi:10.1007/978-3-031-20467-8](https://doi.org/10.1007/978-3-031-20467-8)
- KDEy: Moreo, González & del Coz, "Kernel density estimation for multiclass quantification", Machine Learning 114 (2025); arXiv:2401.00490. — [doi:10.1007/s10994-024-06726-5](https://doi.org/10.1007/s10994-024-06726-5)
- Garg et al. (2021), "Mixture Proportion Estimation and PU Learning: A Modern Approach" (arXiv:2111.00980): in ML, "mixture proportion estimation" usually means this PU-learning setting, where one component distribution is unknown. That is a different problem from pastry's, where the components are known. — [arXiv:2111.00980](https://arxiv.org/abs/2111.00980)
- RLSbench (Garg et al. 2023, arXiv:2302.03020) benchmarks domain adaptation under "relaxed" label shift, where p(x|y) also changes. — [arXiv:2302.03020](https://arxiv.org/abs/2302.03020)

**LeQua competitions and LQ workshops**
- LeQua 2022 (CLEF 2022) was the first quantification lab (Esuli, Moreo, Sebastiani; arXiv:2111.11249). — [arXiv:2111.11249](https://arxiv.org/abs/2111.11249)
- LeQua 2024, "the 2nd data challenge on Learning to Quantify", had four tasks:
  - T1: binary quantification under prior shift.
  - T2: multiclass quantification under prior shift.
  - T3: ordinal quantification under prior shift (new).
  - T4: binary quantification under covariate shift (new). — [LeQua 2024 tasks](https://lequa2024.github.io/tasks/)
- LQ 2024 was held in Vilnius on 2024-09-13 at ECML/PKDD. — [lq-2024.github.io](https://lq-2024.github.io/)
- LQ 2025 was held in Porto on 2025-09-15. Its programme included "An Overview of LeQua 2025" (Esuli, Moreo, Sebastiani, Sperduti) and team reports (UniLeiden, CSE UNSW, Lamarr, UniOvi). — [lq-2025.github.io](https://lq-2025.github.io/)
- The LQ 2025 proceedings contain:
  - conformal "Mondrian Inductive Conformal Quantification" (Smirnov & Schlembach);
  - Moreo & Salvati's bootstrap confidence intervals;
  - prior-shift detection (Kaftan et al.);
  - recalibration of binary classifiers (Tasche);
  - QuaDapt. — [LQ 2025 proceedings PDF](https://lq-2025.github.io/proceedings/CompleteVolume.pdf)
- No lq-2026.github.io or lequa2026.github.io site exists (HTTP 404 on 2026-10-22). — [lq-2026.github.io (404)](https://lq-2026.github.io/)

### Inferences
- Since 2025 the quantification community has treated uncertainty as a priority. QuaPy's `predict_conf` → `(estimate, region)` pattern, with region objects that know their coverage, simplex volume and Winkler score, is a good template for pastry's result object.
- The Bayesian methods in QuaPy implement the same marginal likelihood that pastry uses, Σ_i log Σ_k ρ_k L_ik under a Dirichlet prior. QuaPy therefore already shows that "NUTS on the simplex through NumPyro" works well enough to ship.
- Pastry can differ from QuaPy in three ways:
  - it accepts L_ik directly, rather than deriving it from a scikit-learn classifier;
  - it scales to large N and K;
  - it adds smoothness, GP and selection terms, which QuaPy lacks.
- QuaPy uses temperature calibration to widen posteriors that are too tight. This is an admission that likelihoods built from classifier outputs are misspecified. Pastry should expect the same with real photo-z PDFs and offer a principled alternative, for example a likelihood temperature or a hierarchical "PDF-calibration" model that is checked through simulation-based coverage tests.

### Gaps
- The LeQua 2025 tasks and results were not found online beyond the overview title in the LQ 2025 programme. The LeQua 2025 site URL is unknown, and lequa2025.github.io returns 404.
- Download counts for QuaPy and mlquantify were not retrieved; pepy.tech was not checked.
- I could not confirm whether an LQ 2026 workshop or a LeQua 2026 challenge took place.
- QuaPy's documentation site was not rendered to check its tutorial quality; I relied on the docstrings and examples in the repository.
- The test-suite coverage percentage of QuaPy is unknown.

## Formal links: EMQ = EM for mixture weights; interim-prior division; calibration; confusion matrix = response matrix

### Takeaway
Quantification under label shift is the same mathematical problem as pastry's. A classifier posterior divided by the training ("interim") prior gives a per-item likelihood L_ik ∝ p(k|x_i)/π_train,k. EMQ/SLD is then exactly the EM algorithm for the mixture weights ρ in Π_i Σ_k L_ik ρ_k, and the Bayesian versions (BayesianMAPLS, BayesianKDEy) put a Dirichlet prior on ρ and sample with NUTS. The confusion-matrix family (ACC, BBSE, BayesianCC) is unfolding with an estimated response matrix. It is consistent but loses information by binning, whereas likelihood methods need calibrated inputs.

### Cited Findings
- QuaPy's EMQ code is literally EM for mixture weights.
  - E-step: `ps_unnormalized = (qs / Ptr) * Px`, where Px are the classifier posteriors and Ptr the training prevalence. M-step: `qs = ps.mean(axis=0)`. Iteration starts from the training prevalence and stops when MAE < ε.
  - Zero training-prevalence entries are smoothed by adding ε. — [QuaPy aggregative.py EMQ.EM](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/aggregative.py)
  - The EMQ docstring describes SLD as iterating "the posterior probabilities … and the class prevalence estimates obtained via maximum-likelihood estimation, in a mutually recursive way". It exposes the Alexandari et al. heuristics: a cross-validated estimate of the training prevalence (`exact_train_prev=False`) and recalibration (`calib='bcts'` and others). — [QuaPy aggregative.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/aggregative.py)
- The original reference is Saerens, Latinne & Decaestecker (2002), "Adjusting the Outputs of a Classifier to New a Priori Probabilities: A Simple Procedure", Neural Computation 14:21–41. — [doi:10.1162/089976602753284446](https://doi.org/10.1162/089976602753284446)
- qunfold's LikelihoodMaximizer makes the interim-prior division explicit.
  - The code is `pXY = classifier.predict_proba(X) / p_trn  # proportional to P(X|Y)`, then normalised.
  - Its loss is `−mean log(pXY @ p) + τ0·½Σ(p_{k+1}−p_k)² + τ1·½Σ(−p_{k−1}+2p_k−p_{k+1})²`, i.e. first- and second-difference smoothness penalties for ordinal classes. It is minimised with scipy trust-ncg on a softmax parametrisation.
  - The docstring states that it "is proven to be asymptotically equivalent to the ExpectationMaximizer by Saerens et al. (2002)". — [qunfold likelihood.py](https://github.com/mirkobunse/qunfold/blob/main/qunfold/methods/likelihood.py)
- QuaPy's BayesianMAPLS likelihood is `Σ_i logsumexp_k(log p(k|x_i) + log π_test,k − log π_train,k)`. This is the interim-prior-corrected marginal likelihood inside a NUTS posterior with a Dirichlet prior.
  - With `exact_train_prev=False`, π_train is estimated as the mean of the posteriors on training data. — [QuaPy _bayesian.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/_bayesian.py)
  - MAPLS paper: Ye, Tsuchida, Petersson & Barnes, "Label Shift Estimation for Class-Imbalance Problem: A Bayesian Approach", WACV 2024, pp. 1062–1071. — [doi:10.1109/WACV57701.2024.00111](https://doi.org/10.1109/WACV57701.2024.00111)
- KDEyML minimises the KL divergence between the test distribution of classifier outputs and a mixture of class-conditional KDEs, Σ_k α_k p_k(x̃). The docstring states that this "corresponds to the maximum likelihood estimate". In this method L_ik = KDE_k(s_i) is estimated from labelled data. — [QuaPy _kdey.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/_kdey.py), [arXiv:2401.00490](https://arxiv.org/abs/2401.00490)
- Calibration is the key requirement for likelihood methods.
  - Alexandari et al. note that the limiting assumption of Saerens' EM "is that p(y|x) is calibrated, which is not true of modern neural networks". They show that ML plus bias-corrected calibration (BCTS) beats BBSL and RLLS, and they propose estimating the source-domain priors. — [arXiv:1901.06852](https://arxiv.org/abs/1901.06852)
  - Garg et al. show that MLLS is consistent only with calibration plus a confusion-matrix invertibility condition. Its finite-sample error splits into miscalibration and estimation terms, and they attribute "BBSE's statistical inefficiency to a loss of information due to coarse calibration". — [arXiv:2003.07554](https://arxiv.org/abs/2003.07554)
  - The BCTS and related calibrators are implemented in the abstention package, which QuaPy imports. — [abstention calibration.py](https://github.com/kundajelab/abstention/blob/master/abstention/calibration.py)
- The confusion matrix acts as a response matrix.
  - BBSE works with "arbitrary black box predictors … so long as their confusion matrices are invertible". — [arXiv:1802.03916](https://arxiv.org/abs/1802.03916)
  - Ziegler & Czyż show that ACC, BBSE and invariant ratio estimators approximate inference in one Bayesian model in which the confusion matrix P(C|Y) is a latent parameter, learned from labelled counts at the same time as the prevalence. — [arXiv:2302.09159](https://arxiv.org/abs/2302.09159), [QuaPy _bayesian.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/_bayesian.py)
- qunfold implements ACC, PACC, HDx, HDy, EDx, EDy, KMM and RUN as one "LinearMethod": a data representation plus a loss over the linear system q = M p.
  - RUN is "Regularized Unfolding by Blobel (1985)" with `TikhonovRegularized(BlobelLoss)`. The Tikhonov term "promotes smooth solutions … often required in ordinal quantification and in unfolding problems". — [qunfold linear/__init__.py](https://github.com/mirkobunse/qunfold/blob/main/qunfold/methods/linear/__init__.py), [qunfold losses.py](https://github.com/mirkobunse/qunfold/blob/main/qunfold/methods/linear/losses.py)
  - The unification paper is Bunse, "Unification of algorithms for quantification and unfolding", INFORMATIK 2022, doi:10.18420/inf2022_37, which resolves to dl.gi.de handle 20.500.12116/39536. — [doi:10.18420/inf2022_37](https://doi.org/10.18420/inf2022_37)
- Bunse, Moreo, Sebastiani & Senz (arXiv:2310.09210) compare ordinal quantification algorithms "proposed by authors from very different research fields, such as data mining and astrophysics, who were unaware of each others' developments". They propose regularised ordinal quantifiers that "prevent ordinally implausible estimates, assuming that ordinal distributions tend to be smooth in practice". — [arXiv:2310.09210](https://arxiv.org/abs/2310.09210)
- Firat (2016), "Unified Framework for Quantification" (arXiv:1606.00868), is an earlier unification of quantifiers as constrained regressions. — [arXiv:1606.00868](https://arxiv.org/abs/1606.00868)
- In astronomy the same interim-prior idea appears as importance-reweighting of posterior samples drawn under an interim prior:
  - Hogg, Myers & Bovy 2010, ApJ 725:2166, arXiv:1008.4146. — [arXiv:1008.4146](https://arxiv.org/abs/1008.4146)
  - Foreman-Mackey, Hogg & Morton 2014, arXiv:1406.3020, doi:10.1088/0004-637X/795/1/64. — [arXiv:1406.3020](https://arxiv.org/abs/1406.3020)

### Inferences
- **Dictionary between the two fields.**
  - pastry's L_ik equals, up to a per-object constant, p(k|x_i)/π_interim,k.
  - Summing or stacking per-object PDFs is PCC, which is biased under prior shift.
  - EM on Π_i Σ_k L_ik ρ_k is EMQ/SLD/MLLS.
  - The Dirichlet–multinomial posterior over ρ is BayesianMAPLS/BayesianKDEy.
  - Hard-assignment histograms followed by inversion of a calibration "response matrix" are CC/ACC/BBSE. The Bayesian version that also learns the response matrix is BayesianCC.
- **Interim prior.** pastry should take the prior under which input PDFs or samples were produced as an explicit argument, the counterpart of π_train. That prior is the model's effective prior, not necessarily the nominal training label frequencies. Alexandari et al. and QuaPy's `exact_train_prev=False` estimate it as the mean held-out posterior. With photo-z PDFs produced under an implicit template or training prior, pastry should offer the same estimate as a fallback.
- **Calibration.** Garg et al.'s decomposition implies that miscalibrated per-object PDFs bias the hierarchical estimate. This bias is in addition to the noise that pastry's posterior already reports, so pastry should provide calibration hooks and checks.
  - Examples of hooks: temperature scaling of log L_ik, and per-bin log-bias terms (BCTS-style).
  - Examples of checks: PIT and coverage on a labelled validation set.
  - A per-bin additive bias in log-posterior space is the same thing as a change of interim prior. That is why BCTS's bias terms help EM.
- **Response matrix.** For users with hard labels or point estimates rather than PDFs (e.g., photo-z point estimates or classifier labels), pastry could offer a BBSE/BayesianCC-style path in which a confusion or response matrix learned from labelled calibration data, and its own uncertainty, enter the likelihood. This is the same as unfolding with a statistically uncertain response matrix.
- **Smoothness.** The ordinal-quantification work (RUN/Tikhonov, and τ0/τ1 in LikelihoodMaximizer) is a direct precedent for pastry's smoothness priors on ordered bins. The second-difference penalty corresponds to a Gaussian (intrinsic random-walk) prior.

### Gaps
- I did not access the full texts of Saerens et al. (2002) or Alexandari et al. (2020) to quote their derivations; the EM equivalence above is shown from implementations and abstracts.
- I did not confirm whether any quantification package accepts *likelihoods* directly (rather than a classifier) as a documented input. QuaPy's aggregative methods can take pre-computed posteriors through `fit_classifier=False` and the aggregation API, but no "likelihood-matrix" entry point was found.

## Ecology and genetics: genetic stock identification, mixed-stock analysis, source tracking, isotope mixing (and Python equivalents)

### Takeaway
Genetic stock identification (GSI) and mixed-stock analysis (MSA) solve pastry's problem exactly, and solved it decades earlier. Each fish has a likelihood L_ik = p(genotype_i | population k) computed from a reference "baseline", and the mixture proportions are estimated in one of two ways:
- by EM, as conditional maximum likelihood (SPAM, ONCOR, mixstock), with bootstrap confidence intervals;
- by Gibbs sampling, with a Dirichlet prior on the proportions and multinomial latent assignments (BAYES, rubias, Ms.GSI, mixstock).

rubias is the modern reference implementation, but it was archived from CRAN in February 2026 and must be installed from GitHub. Its main lessons are about bias from priors and reporting-unit aggregation, propagating baseline uncertainty, and posterior-predictive totals. Microbial source tracking (SourceTracker, FEAST) and RNA-seq transcript quantification (salmon, kallisto) solve the same EM or Gibbs problem at far larger N. Stable-isotope mixing models (MixSIAR, simmr) solve a different, within-individual linear-mixing problem. I found no Python GSI or MSA package; the only Python tool in this family is SourceTracker2.

### Cited Findings
**Genetic stock identification / mixed-stock analysis**

*rubias (R/C++; Moran & Anderson)*
- Identity: "Bayesian Inference from the Conditional Genetic Stock Identification Model". GitHub version 0.4.1, licence CC0. Imports Rcpp and RcppParallel. — [rubias DESCRIPTION](https://github.com/eriqande/rubias/blob/master/DESCRIPTION)
- Status:
  - GitHub (2026-10): 6 stars, 5 forks, 15 open issues; last push 2026-06-16. — [GitHub eriqande/rubias](https://github.com/eriqande/rubias)
  - CRAN: "Archived on 2026-02-21 as email to the maintainer is undeliverable". — [CRAN rubias](https://cran.r-project.org/package=rubias)
  - The README says the CRAN version "has not kept up" because the fully Bayesian code "encounters some memory problems" on one chip/compiler combination, and points users to `remotes::install_github("eriqande/rubias")`. Docs are at eriqande.github.io/rubias. — [rubias README](https://github.com/eriqande/rubias)
- API: `infer_mixture(reference, mixture, gen_start_col, method = "MCMC"|"PB"|"BR", alle_freq_prior = list("const_scaled"=1), pi_prior = NULL, pi_init = NULL, reps = 2000, burn_in = 100, pb_iter = 100, pi_prior_sum = 1, total_catch_tib = NULL, ...)`. The three methods are:
  - "MCMC": MCMC conditional on the reference allele frequencies.
  - "PB": the same plus a parametric-bootstrap bias correction.
  - "BR": baseline resampling, i.e. "simulating reference allele frequencies using the genotypes of mixture individuals and allocations from the previous sweep". — [rubias infer_mixture.R](https://github.com/eriqande/rubias/blob/master/R/infer_mixture.R)
- Prior: the default mixing-proportion prior is a symmetric Dirichlet with elements W/C, where W = `pi_prior_sum` (default 1) and C is the number of collections. Users can override the prior per collection. The docs warn that if W is "made much smaller than 1, things could start to mix more poorly". — [rubias infer_mixture.R](https://github.com/eriqande/rubias/blob/master/R/infer_mixture.R)
- Outputs are tidy data frames:
  - `mixing_proportions` (posterior means);
  - `indiv_posteriors` (per-fish assignment probabilities);
  - `mix_prop_traces` ("useful for computing credible intervals");
  - `bootstrapped_proportions` (PB);
  - optional posterior-predictive "stock-specific total catch" traces. — [rubias infer_mixture.R](https://github.com/eriqande/rubias/blob/master/R/infer_mixture.R)
- Usage: `mixf <- infer_mixture(chinook, chinook_mix, 5)`. — [rubias vignette ppns_less_than_1_over_n.Rmd](https://github.com/eriqande/rubias/blob/master/vignettes/ppns_less_than_1_over_n.Rmd)
- Release notes (NEWS):
  - 0.3.0 added the fully Bayesian option, in which fish allocated to a reference sample on a given MCMC step "have their alleles added to that reference sample".
  - 0.4.0 added posterior-predictive simulation of total catch.
  - 0.4.1 added "tools to understand whether mixing proportions < 1/N are the result of fish in the mixture that actually look like they belong to a certain reporting unit or … from the accumulation of a lot of uncertainty". — [rubias NEWS.md](https://github.com/eriqande/rubias/blob/master/NEWS.md)
- The bias rubias corrects "occurs when mixture proportion estimates are desired for groups of populations (reporting units) and the number of populations within each reporting unit are uneven". — [rubias README](https://github.com/eriqande/rubias)
  - The bias "increases with the rate of misassignment between reporting units (decreases with genetic differentiation), and increases as the number of collections within reporting units becomes more uneven". — [rubias assess_pb_bias_correction.R](https://github.com/eriqande/rubias/blob/master/R/assess_pb_bias_correction.R)
- Paper: Moran & Anderson (2019), "Bayesian inference from the conditional genetic stock identification model", Canadian Journal of Fisheries and Aquatic Sciences 76:551–560. — [doi:10.1139/cjfas-2018-0016](https://doi.org/10.1139/cjfas-2018-0016)

*BAYES (Pella & Masuda)*
- Pella & Masuda (2001), "Bayesian methods for analysis of stock mixtures from genetic characters", Fishery Bulletin 99:151–167.
- An informative prior on the stocks' genetic characters is derived from baseline samples, and a "neutral, low-information prior is used for the stock proportions". — [NOAA SPO record](https://spo.nmfs.noaa.gov/content/bayesian-methods-analysis-stock-mixtures-genetic-characters), [Fishery Bulletin PDF](https://fisherybulletin.nmfs.noaa.gov/sites/default/files/13.pdf)

*SPAM (Alaska Department of Fish and Game)*
- "Statistics Program for Analyzing Mixtures", from the ADF&G Gene Conservation Laboratory, Windows. Version 3.7 is dated 25 June 2003.
- It uses conditional maximum likelihood with bootstrap resampling (keyword RESA, default 100), likelihood confidence intervals and studentized bootstrap intervals.
- Version 3.7 added two Bayesian (Dirichlet-posterior) models for baseline allele frequencies, drawn repeatedly from their posteriors during resampling, to handle low-frequency alleles. — [SPAM v3.7 manual](https://www.adfg.alaska.gov/static/fishing/software/sp15/spam_v37.pdf)
- Citation: Debevec et al. (2000), J. Heredity 91:509–511. — [doi:10.1093/jhered/91.6.509](https://doi.org/10.1093/jhered/91.6.509)

*ONCOR (Kalinowski, Manlove & Taper, Montana State; manual dated 21 Oct 2007)*
- Uses conditional maximum likelihood (Millar 1987) and "the EM algorithm … to estimate mixture proportions", iterating until the summed change is below 10⁻⁶.
- Bootstrapped 95% confidence intervals are optional and "slower, especially if the baseline is large". — [ONCOR manual](https://www.montana.edu/kalinowski/software/documents/ONCOR_Manual_21Oct2007.pdf)
- Millar (1987), CJFAS 44:583–590. — [doi:10.1139/f87-071](https://doi.org/10.1139/f87-071)

*gsi_sim (Anderson)*
- A C program for doing and simulating GSI. GitHub eriqande/gsi_sim: 6 stars, last updated 2019-03-28.
- It goes with Anderson, Waples & Kalinowski (2008), "An improved method for predicting the accuracy of genetic stock identification", CJFAS 65:1475–1486. — [GitHub eriqande/gsi_sim](https://github.com/eriqande/gsi_sim), [doi:10.1139/f08-049](https://doi.org/10.1139/f08-049)

*Ms.GSI (Hsu & Habicht 2024, Evolutionary Applications 17:e13621)*
- Multistage GSI: two Pella–Masuda models are chained so that a broad-scale baseline and a regional baseline can be combined.
- It is an R Gibbs sampler with Dirichlet priors (flat by default). It reports posterior means, SDs and 90% credible intervals, with R̂/ESS diagnostics, and supports the conditional GSI formulation.
- Code is at GitHub boppingshoe/Ms.GSI: R, 0 stars, updated 2026-07-02. — [PMC10853661](https://pmc.ncbi.nlm.nih.gov/articles/PMC10853661/), [doi:10.1111/eva.13621](https://doi.org/10.1111/eva.13621), [GitHub boppingshoe/Ms.GSI](https://github.com/boppingshoe/Ms.GSI)

*mixstock (Bolker)*
- R/C package for mixed-stock analysis of sea turtles, v0.9.9, GPL.
- Functions include conditional and unconditional maximum likelihood, a Gibbs sampler (`gibbs`, `gibbsC`), `BUGS.out`, and `confint.mixstock.est`. — [GitHub bbolker/mixstock](https://github.com/bbolker/mixstock)
- Status: CRAN archived it on 2014-08-30 "as long-term memory-access errors were not corrected". The README says it is installable only via devtools, with compilation tools. Last commit 2024-07-21. — [CRAN mixstock](https://cran.r-project.org/package=mixstock), [mixstock README](https://github.com/bbolker/mixstock)
- Papers:
  - Bolker, Okuyama, Bjorndal & Bolten (2007), "many-to-many" MSA, Molecular Ecology 16:685–695. — [doi:10.1111/j.1365-294x.2006.03161.x](https://doi.org/10.1111/j.1365-294x.2006.03161.x)
  - Okuyama & Bolker (2005), Ecological Applications 15:315–325. — [doi:10.1890/03-5063](https://doi.org/10.1890/03-5063)

*Hierarchical extension*
- Guo, Dey & Holsinger, "A hierarchical Bayesian approach for estimating the origin of a mixed population" (IMS Collections 3, 2008; arXiv:0805.3269). — [arXiv:0805.3269](https://arxiv.org/abs/0805.3269)

*Evidence that proportional methods beat classify-and-count*
- An ADF&G WASSIP technical document summarises Koljonen et al. (2005): "proportional assignment (CML or BAYES) performs better than classical individual assignment (IA)", and a Bayesian individual-assignment method slightly outperformed BAYES.
- The same document says "informed priors provide less biased BAYES results than flat priors, especially where there is little genetic [differentiation]". — [ADF&G WASSIP Tech Doc 17](https://www.adfg.alaska.gov/static/fishing/pdfs/research/wassip/documents/techdoc_17_chum_reporting_group_exploratory_methods_v1.pdf)

*Adjacent CRAN tools (different problems)*
- assignPOP (v1.3.1, 2025-09-12) does individual population assignment with ML classifiers (caret, randomForest, SVM). — [CRAN assignPOP](https://cran.r-project.org/package=assignPOP)
- stockR (v1.0.76, 2023-04-26) identifies stocks by clustering, an unsupervised problem. — [CRAN stockR](https://cran.r-project.org/package=stockR)

**Microbial source tracking (same latent-assignment structure, at the read level)**

*SourceTracker (Knights et al. 2011)*
- "Bayesian community-wide culture-independent microbial source tracking", Nature Methods 8:761–763. — [doi:10.1038/nmeth.1650](https://doi.org/10.1038/nmeth.1650)
- The R original is danknights/sourcetracker (82 stars). — [GitHub danknights/sourcetracker](https://github.com/danknights/sourcetracker)

*SourceTracker2 (Python)*
- GitHub caporaso-lab/sourcetracker2 (the clone remote biota/sourcetracker2 redirects there): 92 stars, 48 forks, 55 open issues; last commit 2024-07-10; version string 2.0.1-dev; "modified BSD". — [GitHub caporaso-lab/sourcetracker2](https://github.com/caporaso-lab/sourcetracker2)
- On PyPI it exists only as `sourcetracker` 2.0.1, from 2016-05-09. — [PyPI sourcetracker JSON](https://pypi.org/pypi/sourcetracker/json)
- API: `gibbs(sources, sinks=None, alpha1=.001, alpha2=.1, beta=10, restarts=10, draws_per_restart=1, burnin=100, delay=1, jobs=1)`.
  - `alpha1` and `alpha2` are Dirichlet pseudo-counts for the known and Unknown source feature distributions; `beta` is a pseudo-count for test sequences.
  - There is also a CLI, `sourcetracker2 gibbs`. On the CLI, rarefaction to 1000 sequences per sample is the default; the API does not rarefy. — [sourcetracker2 _sourcetracker.py](https://github.com/caporaso-lab/sourcetracker2/blob/master/sourcetracker/_sourcetracker.py), [README](https://github.com/caporaso-lab/sourcetracker2)

*FEAST (Shenhav et al. 2019)*
- "Fast expectation-maximization for microbial source tracking", Nature Methods 16:627–632.
- R with Rcpp/RcppArmadillo; GitHub cozygene/FEAST has 152 stars and is installed with devtools.
- It returns a sink × source proportion matrix that includes an "unknown" source. The README still calls it a "beta version". — [doi:10.1038/s41592-019-0431-x](https://doi.org/10.1038/s41592-019-0431-x), [GitHub cozygene/FEAST](https://github.com/cozygene/FEAST)

**RNA-seq transcript quantification: the largest-scale instance of the same EM problem**

*salmon (COMBINE-lab)*
- 938 stars; BSD-3. It is now "salmon 2.0 — a from-scratch Rust rewrite"; the C++ line ended at 1.12.0, on the `cpp` branch. The workspace was bumped to v2.8.0 on 2026-09-25.
- Its statistical model is "EM/VBEM over equivalence classes", with VBEM as the default.
- Uncertainty comes as inferential replicates from `--numBootstraps` or `--numGibbsSamples`. — [salmon README](https://github.com/COMBINE-lab/salmon), [salmon cli-flag-status.md](https://github.com/COMBINE-lab/salmon/blob/master/docs/cli-flag-status.md)
- Paper: Patro et al. (2017), Nature Methods 14:417–419. — [doi:10.1038/nmeth.4197](https://doi.org/10.1038/nmeth.4197)

*kallisto (Pachter lab)*
- 775 stars; BSD-2; last commit 2026-01-20. It has a `--bootstrap-samples` (`-b`) option. — [kallisto main.cpp](https://github.com/pachterlab/kallisto/blob/master/src/main.cpp), [GitHub pachterlab/kallisto](https://github.com/pachterlab/kallisto)
- Paper: Bray et al. (2016), "Near-optimal probabilistic RNA-seq quantification", Nature Biotechnology 34:525–527. — [doi:10.1038/nbt.3519](https://doi.org/10.1038/nbt.3519)

**Stable-isotope mixing models (related, but a different likelihood)**

*MixSIAR*
- CRAN v3.1.12, 2020-10-20; GPL-3; needs JAGS ≥ 4.3. GitHub: 125 stars, 157 open issues, last push 2024-07-19.
- It is "not one model, but a framework" with "fixed/random effects, source data types, priors, and error terms". — [CRAN MixSIAR](https://cran.r-project.org/package=MixSIAR), [GitHub brianstock/MixSIAR](https://github.com/brianstock/MixSIAR)
- Paper: Stock et al. (2018), PeerJ 6:e5096. — [doi:10.7717/peerj.5096](https://doi.org/10.7717/peerj.5096)

*simmr*
- CRAN v0.5.3, 2026-07-23; GPL ≥ 2; uses JAGS plus Rcpp/RcppArmadillo.
- Version 0.5.0 "Implemented new Fixed Form Variational Bayes method for fitting SIMMs". The README shows codecov and R-CMD-check badges. — [CRAN simmr](https://cran.r-project.org/package=simmr), [simmr NEWS](https://github.com/andrewcparnell/simmr/blob/master/NEWS.md), [simmr README](https://github.com/andrewcparnell/simmr)
- Papers: Parnell et al. 2010 (PLoS ONE 5:e9672) and 2013 (Environmetrics 24:387–399). — [doi:10.1371/journal.pone.0009672](https://doi.org/10.1371/journal.pone.0009672), [doi:10.1002/env.2221](https://doi.org/10.1002/env.2221)
- In these models each consumer's tracer values are a proportion-weighted mixture of source tracer distributions (the CRAN descriptions speak of "dietary proportions" inferred from "stable isotope values taken from the organisms' tissue samples"). They do not assign each item to a latent class. — [CRAN simmr](https://cran.r-project.org/package=simmr)

**Python equivalents**
- None of these PyPI names exist: `rubias`, `mixstock`, `mixsiar`, `pymixsiar`, `simmr`, `pysimmr`. PyPI `gsi` is an unrelated OpenSSL wrapper. — [PyPI JSON API checks](https://pypi.org/pypi/rubias/json)
- A GitHub search for "mixed stock analysis language:Python" returned 0 repositories, and the "genetic stock identification" search returned only R/C/HTML repositories. — [GitHub search](https://github.com/search?q=genetic+stock+identification&type=repositories)

### Inferences
- GSI/MSA is pastry's model almost exactly:
  - L_ik is a per-object likelihood under each source;
  - ρ ~ Dir(α);
  - z_i ~ Cat(ρ);
  - the conditional model uses EM for the MLE (CML/ONCOR/SPAM) and data-augmentation Gibbs (BAYES/rubias) for the posterior.
- Decades of practice there have produced several lessons for pastry:
  - **Dirichlet prior strength.** Prior mass matters when likelihoods are weakly informative ("little genetic differentiation" corresponds to broad or degenerate photo-z PDFs). Informed or hierarchical priors reduce bias.
  - **Aggregation bias.** A symmetric Dirichlet over fine bins (collections) gives grouped bins (reporting units) prior mass in proportion to the number of fine bins they contain. That biases the group-level proportions when groups are uneven, as rubias documents. pastry's default α should be tied to bin width or a base measure, for example α_k ∝ Δz_k, so that re-binning is prior-consistent. Otherwise pastry should warn when users aggregate bins after the fact.
  - **Training-set uncertainty.** Uncertainty in the "baseline", i.e. the class-conditional models that generate L_ik, should be propagated. rubias "BR" and SPAM's Dirichlet resampling of baseline allele frequencies do this; in pastry it maps to template or training-set uncertainty in photo-z likelihoods.
  - **Interpretable outputs.** Posterior predictive totals and the "proportion < 1/N" diagnostics are user-facing outputs that pastry could copy, e.g. predicted counts in bins for a finite survey and flags for "empty" bins.
- salmon and kallisto show that the EM/VBEM and Gibbs approach scales to 10⁸ reads by collapsing items with identical compatibility patterns into equivalence classes. pastry can likewise deduplicate or cluster nearly identical L_i rows, for example objects with the same photometry and noise, into weighted pseudo-objects to cut the O(NK) cost.
- I found no Python GSI or MSA tool. pastry would be the first general Python implementation of this model class with a scikit-learn or emcee-like API, provided its input is a generic likelihood matrix and not a genotype table.

### Gaps
- The BAYES software (Pella & Masuda) and cBAYES (DFO Canada) distribution pages could not be verified. The download of an ADF&G BAYES manual failed with a connection reset, and cBAYES returned no results. They are therefore omitted from the census apart from the BAYES paper.
- Koljonen et al. (2005) was cited only through the ADF&G technical document; the Crossref lookup was rate-limited.
- I did not verify whether MixSIAR uses ILR/ALR-transformed proportions for its covariate effects; the PeerJ page did not render.
- No scale benchmarks (N, K, run time) were found for rubias or Ms.GSI.

## Epidemiology and survey statistics: prevalence with misclassification

### Takeaway
Epidemiology's "true prevalence from an imperfect test" is binary quantification:
- the Rogan–Gladen estimator is Adjusted Classify-and-Count, with sensitivity and specificity playing the role of the confusion or response matrix;
- Bayesian latent-class models are BayesianCC with priors on sensitivity and specificity.

The field's tools (epiR, prevalence, testerror; Gelman & Carpenter's Stan models) are R or Stan code for small K, usually K=2. I found no Python package. The transferable lesson is to propagate calibration (Se/Sp) uncertainty hierarchically, which also handles the truncation problem of plug-in inversion.

### Cited Findings
- Foundational papers:
  - Rogan & Gladen (1978), "Estimating prevalence from the results of a screening test", American Journal of Epidemiology 107:71–76. — [doi:10.1093/oxfordjournals.aje.a112510](https://doi.org/10.1093/oxfordjournals.aje.a112510)
  - Joseph, Gyorkos & Coupal (1995), "Bayesian estimation of disease prevalence and the parameters of diagnostic tests in the absence of a gold standard", Am. J. Epidemiol. 141:263–272. — [doi:10.1093/oxfordjournals.aje.a117428](https://doi.org/10.1093/oxfordjournals.aje.a117428)
  - Hui & Walter (1980), "Estimating the error rates of diagnostic tests", Biometrics 36:167, the classic latent-class model. — [doi:10.2307/2530508](https://doi.org/10.2307/2530508)
  - Gelman & Carpenter (2020), "Bayesian analysis of tests with unknown specificity and sensitivity", JRSS C 69:1269–1283. — [doi:10.1111/rssc.12435](https://doi.org/10.1111/rssc.12435)
- Code for the Gelman & Carpenter work is at bob-carpenter/diagnostic-testing ("statistical models to analyze diagnostic tests"; 16 stars; updated 2025-10-03). — [GitHub bob-carpenter/diagnostic-testing](https://github.com/bob-carpenter/diagnostic-testing)
- epiR (CRAN v3.0.0, published 2026-10-13, GPL)
  - `epi.prev(pos, tested, se, sp, method = "wilson", tp.method = "rogan.gladen", bayes.variant = NULL, units = 100, conf.level = 0.95)` estimates true prevalence by "the Rogan Gladen correction or a simplified Bayesian approach (Kopacka and Fuchs 2026)" that "accounts for uncertainty in test sensitivity and specificity".
  - Confidence-interval options are Clopper–Pearson, Sterne, Blaker and Wilson.
  - The docs warn: "Where AP < (1 - Sp) or AP > Se a Bayesian approach for estimation of true prevalence is recommended". — [CRAN epiR](https://cran.r-project.org/package=epiR), [epiR reference manual](https://cran.r-project.org/web/packages/epiR/refman/epiR.html)
- prevalence (CRAN v0.4.1, 2022-06-03; needs JAGS; GitHub last push 2022-06-03)
  - `truePrev(x, n, SE = 1, SP = 1, prior = c(1, 1), nchains = 2, burnin = 10000, update = 10000)` puts priors on SE and SP given as formulas, e.g. `SE = ~dunif(0.60, 1.00)`.
  - `truePrevMulti`/`truePrevMulti2` handle multiple tests, the latter with conditional dependence (Dendukuri & Joseph 2001), and `truePrevPools` handles pooled samples.
  - Outputs are MCMC chains with diagnostics. — [CRAN prevalence](https://cran.r-project.org/package=prevalence), [prevalence reference manual](https://cran.r-project.org/web/packages/prevalence/refman/prevalence.html), [GitHub brechtdv/prevalence](https://github.com/brechtdv/prevalence)
- testerror (R, Bristol Vaccine Centre r-universe) reportedly offers Rogan–Gladen and Bayesian true-prevalence functions. That description comes from a search-result summary; I did not inspect the functions, but the r-universe page returned HTTP 200 on 2026-10-22. — [r-universe testerror](https://bristol-vaccine-centre.r-universe.dev/testerror)
- BayesLCA (CRAN v1.9, 2020-05-06) is "Bayesian Latent Class Analysis using several different methods". — [CRAN BayesLCA](https://cran.r-project.org/package=BayesLCA)
- A web search for a Python true-prevalence or Rogan–Gladen package found only R tools (epiR, testerror). — [rdocumentation epi.prev](https://www.rdocumentation.org/packages/epiR/versions/2.0.95/topics/epi.prev), [testerror](https://bristol-vaccine-centre.r-universe.dev/testerror)
- Crossover between the fields:
  - The Bayesian quantifier PQ in QuaPy comes from Igiraneza, Fraser & Hinch (2025), "Estimating prevalence with precision and accuracy". — [arXiv:2507.06061](https://arxiv.org/abs/2507.06061)
  - Moreo & Salvati cite epidemiology as a motivating discipline for confidence regions in quantification. — [LQ 2025 proceedings](https://lq-2025.github.io/proceedings/CompleteVolume.pdf)
- Survey statistics / social science: Hopkins & King's ReadMe (AJPS 54:229–247, 2010) estimates category proportions of documents without classifying individual documents. QuaPy implements it as `ReadMe`, with `predict_conf`. — [doi:10.1111/j.1540-5907.2009.00428.x](https://doi.org/10.1111/j.1540-5907.2009.00428.x), [QuaPy non_aggregative.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/non_aggregative.py)

### Inferences
- **Rogan–Gladen is ACC.** For K=2, π̂ = (AP + Sp − 1)/(Se + Sp − 1) is the same as ACC's (CC − FPR)/(TPR − FPR) with TPR = Se and FPR = 1 − Sp. (This identity is my derivation from the standard definitions, not quoted from a source.)
- **Truncation.** epiR's warning about AP < 1 − Sp is the K=2 case of the negative or out-of-simplex solutions of unconstrained response-matrix inversion. The Bayesian answer is a simplex-constrained prior plus uncertain Se/Sp; pastry gets this for free from its Dirichlet/simplex parametrisation.
- **Hierarchical calibration.** Gelman & Carpenter's hierarchical pooling of Se/Sp across calibration studies is a template for pastry's treatment of per-survey or per-population calibration of PDFs. Examples are pooling PIT-based calibration parameters across fields, or across spectroscopic calibration samples with different selections.

### Gaps
- Survey-statistics misclassification literature, such as misclassified categorical responses in complex surveys, was not researched beyond ReadMe; no software was verified there.
- I did not read the Kopacka & Fuchs (2026) "simplified Bayesian" method in epiR beyond the manual's description.
- The Gelman & Carpenter Stan code was not inspected; the repository language is listed as HTML.

## Lessons for pastry

### Takeaway
The working designs are:
1. a scikit-learn-like `fit`/`predict` core that returns a rich uncertainty object (point estimate, posterior samples or bootstrap replicates, region types, coverage helpers), as QuaPy and mlquantify do;
2. a fast EM/VBEM point estimate with cheap resampling or Gibbs "inferential replicates", as salmon, kallisto and rubias do;
3. full Bayesian sampling with explicit priors, diagnostics and calibration checks, as rubias, Ms.GSI and QuaPy's Bayesian methods do.

The recurring failure modes are:
- heavy system dependencies (JAGS, Stan, JAX extras);
- posterior samples that are hard to get at;
- abandoned CRAN or PyPI releases;
- prior effects that users do not understand (Dirichlet mass, reporting-unit aggregation);
- over-confident posteriors when likelihoods are miscalibrated.

pastry can fill a real gap. No existing package takes a generic per-object likelihood matrix (or posterior samples under an interim prior) and returns calibrated population fractions with smoothness or selection extensions in Python.

### Cited Findings
**Census summary (verified 2026-10-22)**

| Tool | Domain | Lang / licence | Release; last commit | Input | Model / engine | Uncertainty | Status |
|---|---|---|---|---|---|---|---|
| [QuaPy](https://github.com/HLT-ISTI/QuaPy) | ML quantification | Python / BSD-3 | 0.2.3 (2026-10-05); 2026-10-06 | X,y + sklearn classifier (or precomputed posteriors) | CC…KDEy; EM; NUTS (NumPyro); Stan | bootstrap regions, Bayesian samples, temperature calibration | active |
| [mlquantify](https://github.com/luizfernandolj/mlquantify) | ML quantification | Python / BSD-3 | 0.5.1 (0.5.2b3 2026-08-03); 2026-10-16 | X,y + classifier | 33 methods | bootstrap regions | active |
| [qunfold](https://github.com/mirkobunse/qunfold) | quantification + unfolding | Python+JAX / MIT classifier, no LICENSE file | git only (0.1.6-rc4); 2026-07-13 | X,y + classifier | linear methods, RUN/Tikhonov, ML, EM | none | maintained, small |
| [labelshift (Czyż)](https://github.com/pawel-czyz/labelshift) | Bayesian quantification | Python+NumPyro / BSD-3 | none; 2024-06-13 | confusion counts | BayesianCC; NUTS | posterior samples | research code |
| [abstention](https://github.com/kundajelab/abstention) | calibration + label shift | Python / none declared | 0.1.3.1 (2020-01-22); 2020-11-14 | softmax outputs | EM, BBSE, RLLS, BCTS | none | unmaintained |
| [quantificationlib](https://github.com/AICGijon/quantificationlib) | ML quantification | Python / GPL-3 | 0.1.2 (2024-01-31); 2024-04-06 | classifier | many, incl. ordinal | none | dormant |
| [rubias](https://github.com/eriqande/rubias) | GSI | R+C++ / CC0 | GitHub 0.4.1; 2026-06-16 (CRAN archived 2026-02-21) | genotype tables | conditional GSI; Gibbs, PB, BR | traces, credible intervals, posterior-predictive totals | GitHub-only |
| [Ms.GSI](https://github.com/boppingshoe/Ms.GSI) | multistage GSI | R / not checked | 2026-07-02 | 2 baselines + mixture | Pella–Masuda Gibbs | posterior summaries, R̂/ESS | niche |
| [mixstock](https://github.com/bbolker/mixstock) | MSA (turtles) | R+C / GPL | 0.9.9; 2024-07-21 (CRAN archived 2014) | haplotype counts | CML/UML; Gibbs; BUGS | confint, MCMC | dormant |
| [SPAM](https://www.adfg.alaska.gov/static/fishing/software/sp15/spam_v37.pdf) / [ONCOR](https://www.montana.edu/kalinowski/software/documents/ONCOR_Manual_21Oct2007.pdf) | GSI | desktop GUI programs (the SPAM manual references Windows) | 2003 / 2007 | genotypes | CML (EM) | bootstrap CIs | legacy |
| [SourceTracker2](https://github.com/caporaso-lab/sourcetracker2) | microbial source tracking | Python / modified BSD | PyPI 2016; 2024-07-10 | count tables | Dirichlet latent assignment; Gibbs | posterior draws | dormant |
| [FEAST](https://github.com/cozygene/FEAST) | microbial source tracking | R+C++ | GitHub (beta) | count tables | EM | point estimate | beta |
| [salmon](https://github.com/COMBINE-lab/salmon) / [kallisto](https://github.com/pachterlab/kallisto) | RNA-seq | Rust BSD-3 / C++ BSD-2 | 2.8.0 (2026-09-25) / 2026-01-20 | reads → equivalence classes | EM/VBEM | bootstrap / Gibbs replicates | active |
| [MixSIAR](https://github.com/brianstock/MixSIAR) / [simmr](https://github.com/andrewcparnell/simmr) | isotope mixing | R+JAGS (simmr also FFVB) / GPL | 3.1.12 (2020) / 0.5.3 (2026-07-23) | tracer values | linear mixing; MCMC/VB | posterior samples | different problem |
| [epiR](https://cran.r-project.org/package=epiR) / [prevalence](https://cran.r-project.org/package=prevalence) | epidemiology | R (prevalence needs JAGS) / GPL | 3.0.0 (2026-10-13) / 0.4.1 (2022) | counts + Se/Sp | Rogan–Gladen; Bayesian latent class | CIs, credible intervals | K=2 only |

**User-facing design evidence**
- *A familiar API lowers adoption barriers.*
  - QuaPy 0.2.0 moved to scikit-learn-style `fit(X, y)` and `predict`, keeping `quantify` as an alias.
  - QuaPy 0.1.9 added a default classifier (`LogisticRegression(max_iter=3000)`) so methods work without arguments, and let `N_JOBS` be set from an environment variable. — [QuaPy CHANGE_LOG.txt](https://github.com/HLT-ISTI/QuaPy/blob/master/CHANGE_LOG.txt)
  - qunfold has the same three-line pattern: construct, `fit`, `predict`. — [qunfold README](https://github.com/mirkobunse/qunfold)
- *Uncertainty as an object.* QuaPy's `predict_conf` returns `(estimate, region)`, and the region objects carry `coverage`, `simplex_portion`, `samples` and Winkler scores. mlquantify adds `ConfidenceRegionDisplay` plots. — [QuaPy confidence.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/confidence.py), [mlquantify README](https://github.com/luizfernandolj/mlquantify)
- *Bootstrap caveat.* Aggregation-phase bootstrap is cheap, but "cannot provide valid coverage guarantees under prior probability shift". — [LQ 2025 proceedings (Moreo & Salvati)](https://lq-2025.github.io/proceedings/CompleteVolume.pdf)
- *Over-confident posteriors.* QuaPy added `calibrate_temperature`, which picks a likelihood temperature from a grid using validation coverage or Winkler score. — [QuaPy _bayesian.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/_bayesian.py)
- *Dependency friction.*
  - QuaPy's `bayes` extra pulls in jax, jaxlib, numpyro and pystan, and pins `setuptools<82`. Its code also silences "upstream pystan noise" from `pkg_resources`. — [PyPI quapy JSON](https://pypi.org/pypi/quapy/json), [QuaPy _bayesian.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/_bayesian.py)
  - EMQ has an `on_calib_error` policy because "abstention's calibration functions failed (which happens sometimes)". — [QuaPy CHANGE_LOG.txt](https://github.com/HLT-ISTI/QuaPy/blob/master/CHANGE_LOG.txt)
  - MixSIAR and prevalence require a system JAGS install. — [CRAN MixSIAR](https://cran.r-project.org/package=MixSIAR), [CRAN prevalence](https://cran.r-project.org/package=prevalence)
- *Support load.*
  - MixSIAR's most-discussed issues include "Error in JAGS model" (#105), "Accessing posterior chains with attach.jags function" (#80), "accessing the output to get objects" (#229) and "Fatal error when loading MixSIAR GUI on Mac OS X" (#183).
  - Issue #242 shows the model monitors an `ilr.global` node, i.e. proportions are ILR-parametrised.
  - The repository has 157 open issues. — [MixSIAR #105](https://github.com/brianstock/MixSIAR/issues/105), [#80](https://github.com/brianstock/MixSIAR/issues/80), [#229](https://github.com/brianstock/MixSIAR/issues/229), [#183](https://github.com/brianstock/MixSIAR/issues/183), [#242](https://github.com/brianstock/MixSIAR/issues/242)
- *Release fragility.*
  - CRAN archived rubias in 2026 because of an undeliverable maintainer email, and mixstock in 2014 because of C memory errors.
  - The PyPI names `labelshift` and `qunfold` belong to projects other than the well-known GitHub repos of the same name. — [CRAN rubias](https://cran.r-project.org/package=rubias), [CRAN mixstock](https://cran.r-project.org/package=mixstock), [PyPI labelshift JSON](https://pypi.org/pypi/labelshift/json), [PyPI qunfold JSON](https://pypi.org/pypi/qunfold/json)
- *Name clash.* The PyPI name `pastry` is taken by an unrelated project ("A simple api wrapper for chef", v0.1.31, last upload 2018-03-12), and `pypastry` is an ML experimentation framework (v0.3.0, 2021). `pastry-inference` was free on 2026-10-22. — [PyPI pastry JSON](https://pypi.org/pypi/pastry/json), [PyPI pypastry JSON](https://pypi.org/pypi/pypastry/json)
- *Tidy, inspectable outputs.* rubias returns tidy tibbles: mixing proportions, per-individual posteriors, full traces and posterior-predictive totals. Ms.GSI reports R̂ and ESS, and prevalence requires `nchains >= 2`. — [rubias infer_mixture.R](https://github.com/eriqande/rubias/blob/master/R/infer_mixture.R), [PMC10853661](https://pmc.ncbi.nlm.nih.gov/articles/PMC10853661/), [prevalence manual](https://cran.r-project.org/web/packages/prevalence/refman/prevalence.html)
- *Scale.* salmon's EM/VBEM runs "over equivalence classes" and offers `--numBootstraps` and `--numGibbsSamples` replicates. BayesianCC works on K×K confusion counts plus K predicted-class counts, so its cost does not depend on N. — [salmon README](https://github.com/COMBINE-lab/salmon), [QuaPy _bayesian.py](https://github.com/HLT-ISTI/QuaPy/blob/master/quapy/method/_bayesian.py)
- *Smoothness.*
  - Regularisation that "prevents ordinally implausible estimates" outperformed the unregularised ordinal quantifiers. — [arXiv:2310.09210](https://arxiv.org/abs/2310.09210)
  - LeQua 2024 Task T3 is an ordinal quantification benchmark, and QuaPy ships LeQua loaders. — [LeQua 2024 tasks](https://lequa2024.github.io/tasks/), [QuaPy CHANGE_LOG.txt](https://github.com/HLT-ISTI/QuaPy/blob/master/CHANGE_LOG.txt)
- *Evaluation protocols.*
  - QuaPy evaluates under sampling protocols APP (artificial prevalence), NPP, UPP, DirichletProtocol and DomainMixer; mlquantify uses APP, NPP, UPP and PPP.
  - QuaPy has a `test_replicability.py` test module. — [QuaPy tests](https://github.com/HLT-ISTI/QuaPy/tree/master/quapy/tests), [mlquantify README](https://github.com/luizfernandolj/mlquantify)

### Inferences
- **Core API**
  - Something like `pastry.fit(L, prior=..., interim_prior=None, method="nuts"|"gibbs"|"em"|"vb")`, or an estimator class with `.fit(L)`, accepting:
    - an N×K likelihood matrix;
    - N×S posterior samples plus the interim prior;
    - per-object PDFs on a grid.
  - It should return a result object with:
    - `.mean`, `.samples` (an ArviZ-compatible InferenceData), `.interval(level)`, `.region(kind="intervals"|"ellipse-clr")` and `.coverage(true_rho)`;
    - per-object membership posteriors (rubias' PofZ, EMQ's corrected posteriors);
    - posterior-predictive counts.
  - Diagnostics (R̂, ESS) should be computed automatically.
- **Layered inference:**
  1. EM/MAP as a fast default, with optional fast bootstrap "inferential replicates" (salmon-style);
  2. Gibbs data augmentation (rubias, BAYES, SourceTracker) for the plain Dirichlet case;
  3. gradient-based HMC/VI in an unconstrained (ALR/ILR) space for smoothness, GP and selection extensions, as QuaPy's NumPyro models do.
  - The core should be NumPy/SciPy only, with JAX as an optional accelerator, to avoid QuaPy- and MixSIAR-style dependency friction.
- **Interim prior and calibration as first-class inputs.**
  - `interim_prior=` corresponds to π_train in EMQ and MAPLS.
  - Optional calibration includes temperature and per-bin log-bias terms, the analogue of BCTS.
  - Coverage checks on labelled validation data or simulations, in the spirit of QuaPy's `calibrate_temperature` and Winkler scores. Prefer a hierarchical calibration model over ad hoc tempering where possible.
- **Prior semantics.**
  - Make the Dirichlet concentration explicit, and document that α should scale with bin width, i.e. a base measure.
  - Provide tools for "is this small fraction real or prior-driven?", like rubias' proportions-below-1/N diagnostics.
  - Warn about aggregation bias when fine bins are summed into coarse ones (rubias' reporting-unit bias).
- **Response-matrix path.** For users with only point estimates or hard labels, a BBSE/BayesianCC-style module could learn the confusion matrix from a labelled calibration set and propagate its uncertainty, rather than inverting it. This aligns pastry with unfolding practice; the HEP researchers cover the details.
- **Scaling.** Deduplicate or cluster identical or near-identical likelihood rows into weighted pseudo-objects (salmon's equivalence classes). Use sufficient statistics where the model allows; with hard labels and a confusion matrix the cost does not depend on N.
- **Testing and benchmarks**
  - Adopt quantification-style protocols: an APP-like sweep over true ρ, coverage and Winkler/interval-score metrics, and replicability tests.
  - LeQua 2022/2024 (especially T3, ordinal) and rubias' simulation tools give external benchmarks beyond astronomy.
- **Packaging**
  - Choose a distribution name that is free on PyPI (`pastry` is taken).
  - Keep the release pipeline simple and multi-maintainer, so that pastry does not suffer rubias- or mixstock-style archiving.
  - Declare a licence explicitly; abstention and qunfold show how often this is missed.

### Gaps
- No user surveys or usability studies of quantification or GSI software were found. The usability lessons above come from changelogs, issue titles and API design, not systematic user research.
- QuaPy's and rubias' issue trackers returned no results in semantic issue search, so pain points are evidenced mainly by MixSIAR.
- Performance numbers (run time against N and K) for QuaPy's Bayesian methods, rubias or SourceTracker2 were not found. Scalability claims rest on the algorithms (per-item likelihood versus sufficient statistics), not on benchmarks.
- I did not check whether any tool supports selection effects (truncation or completeness functions) in the mixture likelihood. None of the docs I read mention it, so pastry's selection-effect support appears novel in this family, but this is unverified beyond those docs.
