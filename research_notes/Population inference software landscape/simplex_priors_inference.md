# Priors over histograms (the probability simplex) and inference for the hierarchical Dirichlet–multinomial mixture-weights model

Scope: (a) priors on bin fractions ρ ∈ Δ^{K-1} beyond the plain Dirichlet; (b) algorithms for p(ρ | {L_i}) ∝ p(ρ) Π_i Σ_k L_ik ρ_k (known per-object likelihoods L_ik = p(d_i | z_i = k)); (c) Python library support as of 22 October 2026. Domain tools (astro, GW, HEP unfolding, EB/NPMLE packages, ML quantification), neural density models and API/testing practice are covered by other notes. Versions and dates come from PyPI/CRAN/conda-forge metadata pulled on 2026-10-22. Function and class availability was checked by import and attribute lookup in clean virtualenvs (numpy 2.5.4, jax 0.12.0.1). Repository existence and activity come from GitHub search. Papers come from arXiv or Crossref, and every DOI below resolved in Crossref. "(derived)" marks my own algebra or numerical checks, not literature claims. Notation: N objects, K bins, L the N×K likelihood matrix, r_ik = L_ik ρ_k / Σ_j L_ij ρ_j the per-object "responsibilities".

## 1. Priors over the simplex for bin fractions: definitions, properties, conjugacy and covariance

### Takeaway
There are two families of priors.
- **Conjugate family.** Dirichlet ⊂ generalised Dirichlet (GD) ⊂ Dirichlet-tree, plus finite-depth Pólya trees, which are binary Dirichlet-trees. Given the latent counts n_k = #{i : z_i = k}, every member keeps Gibbs data augmentation and EM in closed form. Each adds structure only through an ordering or tree topology (variance per node, positive correlation within subtrees). None of them encodes smoothness across neighbouring bins.
- **Gaussian priors on the log-ratio vector** (logistic-normal, RW/ICAR/GMRF, P-splines, (HS)GP, i.e. the logistic Gaussian process). This is the natural home for smoothness and arbitrary correlations. They are not conjugate, so they need gradient-based samplers on unconstrained coordinates. If Gibbs is required, Pólya–gamma stick-breaking augmentation (Linderman, Johnson & Adams 2015) makes Gaussian priors conditionally conjugate.

On fixed bins a Dirichlet process adds nothing beyond a Dirichlet. For sparse tables (n ≪ K), the choice of α matters more than people expect.

### Cited Findings

#### Dirichlet, and how to choose α
- **Definition and moments.** Dir(α) on the simplex has mean α_k/α_0 (α_0 = Σα_k), Var(ρ_k) = α_k(α_0−α_k)/(α_0²(α_0+1)) and Cov(ρ_j,ρ_k) = −α_jα_k/(α_0²(α_0+1)). Every off-diagonal covariance is therefore negative, and a single precision α_0 controls all variances (K parameters in total). SciPy's `scipy.stats.dirichlet.cov` reproduces this closed form (checked numerically for α=(0.5,2,3.5): off-diagonal −0.00397 in both) — [SciPy dirichlet docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet.html) (derived/checked).
- **Conjugacy and use in this model.** The Dirichlet is conjugate to multinomial counts: the posterior is Dir(α + n). Leistedt, Mortlock & Peiris (2016) build the photometric n(z) hierarchical model on exactly this: Dirichlet prior on bin fractions, Dirichlet posterior given the counts, sampled with a two-step Gibbs sampler — [Leistedt et al. 2016, MNRAS 460:4258](https://doi.org/10.1093/mnras/stw1304) ([arXiv:1602.05960](https://arxiv.org/abs/1602.05960)).
- **The same model and sampler in 2026.** The KiDS+VIKING-450 BHM n(z) paper uses a uniform prior on f_ijk "(the Dirichlet distribution)" and "a 2-step Gibbs sampler" that alternates per-galaxy (t,z,m) draws with the population fractions — [arXiv:2605.15078](https://arxiv.org/abs/2605.15078).
- **Objective priors for sparse tables.** Berger, Bernardo & Sun (2015) treat the multinomial with n ≪ m cells. With n=3 and m=1000, the Jeffreys prior (a=1/2) "effectively added 1/2 to the 998 zero cells, making them more important than the cells with data", leaving the two observed cells with only 0.008 total posterior mass. The uniform prior (a=1) "is even worse". The a→0 (Haldane) prior gives an improper posterior if any cell is empty. Their recommended "overall" prior is "roughly" a = 1/m — [Berger, Bernardo & Sun 2015, Bayesian Analysis 10:189](https://doi.org/10.1214/14-BA915) ([arXiv:1504.02689](https://arxiv.org/abs/1504.02689)).
- **Reference-sample-informed α.** Calibration-sample counts have been used directly as Dirichlet parameters for n(z) uncertainty. Sánchez et al. (2020) show that the plain count-Dirichlet underestimates the uncertainty when sample variance (large-scale structure) dominates, as does a bootstrap, and they inflate it (see "Overdispersion" below) — [Sánchez, Raveri, Alarcon & Bernstein 2020, MNRAS 498:2984](https://doi.org/10.1093/mnras/staa2542) ([arXiv:2004.09542](https://arxiv.org/abs/2004.09542)).
- **Generic machinery for informative priors from earlier data.** Power priors raise the historical likelihood to a power a_0 — [Ibrahim & Chen 2000, Stat. Sci. 15:46](https://doi.org/10.1214/ss/1009212673). Dirichlet α can be estimated from observed proportion vectors by Minka's fixed-point MLE — [Minka, "Estimating a Dirichlet distribution"](https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf) — which the Python package `dirichlet` implements (v1.0.0, 2025-08-09, MIT) — [PyPI](https://pypi.org/project/dirichlet/), [GitHub ericsuh/dirichlet](https://github.com/ericsuh/dirichlet). PreliZ 0.28.0 (2026-09-07, Apache-2.0) has a `dirichlet_mode` elicitation helper (verified by import) — [PyPI preliz](https://pypi.org/project/preliz/).

#### Generalised Dirichlet (Connor & Mosimann 1969; Wong 1998)
- **Definition.** Set z_1 = p_1 and z_i = p_i/(1 − Σ_{j<i} p_j). If the z_i are independent Beta(a_i, b_i) (a "cascade" or stick-breaking with free Betas), then p has the GD distribution. The mean is E[p_j] = a_j/(a_j+b_j) · Π_{m<j} b_m/(a_m+b_m). It has about twice as many parameters as the Dirichlet (2(K−1) free) and "a more general covariance structure". It reduces to the Dirichlet when b_{i−1} = a_i + b_i. It is "completely neutral" — [Wikipedia: Generalized Dirichlet distribution](https://en.wikipedia.org/wiki/Generalized_Dirichlet_distribution), summarising [Connor & Mosimann 1969, JASA 64:194](https://doi.org/10.1080/01621459.1969.10500963) and [Wong 1998, Appl. Math. Comput. 97:165](https://doi.org/10.1016/S0096-3003(97)10140-0).
- **Conjugacy.** It is conjugate to the multinomial: a'_j = a_j + y_j and b'_j = b_j + Σ_{i>j} y_i — [Wikipedia GD](https://en.wikipedia.org/wiki/Generalized_Dirichlet_distribution). Leistedt et al. (2016) note that a GD prior on the n(z) fractions gives a GD posterior, and that it reduces to the Dirichlet case for many galaxies or weak interbin-correlation priors — [arXiv:1602.05960](https://arxiv.org/abs/1602.05960).
- **Positive correlations are possible** (derived). With p_1=V_1, p_2=V_2(1−V_1), p_3=(1−V_2)(1−V_1), take a broad V_1 ~ Beta(0.5,0.5) and a tight V_2 ~ Beta(200,200). A Monte Carlo check (2×10⁶ draws) gives corr(p_2,p_3) = +0.985 and corr(p_1,p_2) = −0.996. Positive dependence arises between components that share a highly variable upstream "stick" and a tight downstream split. The ordering of bins matters.
- **Related.** The GD-multinomial is used for overdispersed counts — [Bouguila 2008, IEEE TKDE 20:462](https://doi.org/10.1109/TKDE.2007.190726).

#### Dirichlet-tree, nested Dirichlet, hyper-Dirichlet
- **Definition and properties.** A Dirichlet-tree is a product of independent Dirichlets on the branch probabilities at each internal node of a tree whose leaves are the K bins. Minka's note says:
  - it is "simply the conjugate distribution to the multinomial" for tree-structured sampling;
  - "the Dirichlet-tree distribution can give an independent variance to each p_k";
  - "the leaves in a subtree are correlated since they all depend on the ancestors of that subtree";
  - any tree can realise a plain Dirichlet with suitably chosen α (all β_j = 0);
  - Connor & Mosimann's distribution is the special "cascade-tree" case.

  The number of parameters is one α per edge — [Minka 1999/2004, "The Dirichlet-tree distribution"](https://tminka.github.io/papers/dirichlet/minka-dirtree.pdf). An earlier related form is the hyper-Dirichlet type 1 — [Dennis 1991, Commun. Stat. Theory Methods 20:4069](https://doi.org/10.1080/03610929108830757).
- **Applications showing it is practical**:
  - Dirichlet-tree multinomial mixtures for microbiome compositions — [Mao & Ma 2022, AoAS 16:1476](https://doi.org/10.1214/21-AOAS1552) ([arXiv:2008.00400](https://arxiv.org/abs/2008.00400));
  - IRV ballot-polling audits — [Everest et al. 2022, arXiv:2209.03881](https://arxiv.org/abs/2209.03881);
  - Latent Dirichlet-Tree Allocation with vectorised mean-field VI and EP "for all DT" — [Wang & Bouguila 2026, arXiv:2602.18795](https://arxiv.org/abs/2602.18795);
  - Dirichlet-tree priors in Bayesian causal forests — [Aguilar et al. 2026, arXiv:2610.26728](https://arxiv.org/abs/2610.26728).
- **Hyper-Dirichlet in R.** Hankin's hyper-Dirichlet (a different generalisation, aimed at Bradley–Terry/Plackett–Luce data) — [Hankin 2010, JSS 33(11)](https://doi.org/10.18637/jss.v033.i11). It lives on in R `hyper2` v3.2-3 (2026-06-23, GPL-2|3), which "supersedes the 'hyperdirichlet' package" — [CRAN hyper2](https://cran.r-project.org/package=hyper2). `hyperdirichlet` itself was archived on CRAN on 2017-06-28 — [CRAN hyperdirichlet](https://cran.r-project.org/package=hyperdirichlet).
- **Monograph on Dirichlet relatives.** [Ng, Tian & Tang 2011, "Dirichlet and Related Distributions", Wiley](https://doi.org/10.1002/9781119995784). I did not re-check its chapter list (it is thought to cover grouped and nested Dirichlet and Liouville families).

#### Beta-Liouville, scaled and shifted-scaled Dirichlet, flexible Dirichlet
- **Beta-Liouville multinomial.** It is presented as an alternative to the multinomial and Dirichlet-multinomial, whose "strict assumptions about the mean, variance, and covariance" can fail. Lakin & Abdo give fast MLE and classification for it — [Lakin & Abdo 2020, arXiv:2006.07454](https://arxiv.org/abs/2006.07454). It is also used as a prior in unigram-mixture clustering — [Bilancia & Magro 2024, arXiv:2410.21862](https://arxiv.org/abs/2410.21862). Not verified: the exact parametrisation (a Liouville family with a Beta generator, about K+1 parameters) and its conjugacy to the multinomial.
- **Scaled and shifted-scaled Dirichlet.** These come from the compositional-data (Aitchison-geometry) school: the scaled Dirichlet is a perturbed Dirichlet composition, and the shifted-scaled version adds powering by a scalar. They are motivated by the Dirichlet's "almost completely negative correlation and a strong independence structure" — [Monti, Mateu-Figueras & Pawlowsky-Glahn 2011, "Notes on the scaled Dirichlet distribution", in *Compositional Data Analysis*, Wiley, pp. 128–138](https://doi.org/10.1002/9781119976462.ch10); [shifted-scaled Dirichlet, CoDaWork 2011 record](https://www.recercat.cat/handle/10256/13637?show=full). I found no claim of multinomial conjugacy, and expect none (inference).
- **Flexible Dirichlet** (a structured mixture of Dirichlets) — [Ongaro & Migliorati 2013, JMVA 114:412](https://doi.org/10.1016/j.jmva.2012.07.007). I did not re-read its covariance properties.

#### Logistic-normal (Aitchison) and other log-ratio Gaussian priors
- **Origins.** Logistic-normal distributions: map y ~ N(μ, Σ) in R^{K−1} to the simplex by the additive-logistic transform — [Aitchison & Shen 1980, Biometrika 67:261](https://doi.org/10.2307/2335470). Aitchison (1982) argued that compositional analysis had been hampered by "a lack both of concepts of independence and of rich enough parametric classes of distributions in the simplex", and introduced "transformed-normal distributions in the simplex" — [Aitchison 1982, JRSSB 44:139](https://doi.org/10.1111/j.2517-6161.1982.tb01195.x).
- **Isometric log-ratio (ILR) coordinates.** An orthonormal basis in Aitchison geometry, so the simplex becomes a Euclidean space in which Gaussian priors are natural — [Egozcue et al. 2003, Math. Geol. 35:279](https://doi.org/10.1023/A:1023818214614).
- **Properties.** There are (K−1) + K(K−1)/2 parameters, so arbitrary-sign correlations are possible. The prior is not conjugate to the multinomial. Linderman et al. note that inference for logistic-normal multinomial models "is challenging due to the lack of conjugacy, and previous work has relied upon variational approximations" (e.g. the correlated topic model) — [Linderman, Johnson & Adams 2015, arXiv:1506.05843](https://arxiv.org/abs/1506.05843); [Blei & Lafferty 2007, AoAS 1:17](https://doi.org/10.1214/07-AOAS114). A logistic-normal multinomial regression is used for microbiome counts — [Xia et al. 2013, Biometrics 69:1053](https://doi.org/10.1111/biom.12079).

#### Logistic Gaussian process (LGP) and smoothness priors on log-histograms
- **LGP lineage.** A GP f on the log-density, normalised by exp/∫exp:
  - [Leonard 1978, JRSSB 40:113](https://doi.org/10.1111/j.2517-6161.1978.tb01655.x);
  - [Lenk 1988, JASA 83:509](https://doi.org/10.1080/01621459.1988.10478625);
  - [Lenk 1991, Biometrika 78:531](https://doi.org/10.1093/biomet/78.3.531);
  - [Tokdar 2007, JCGS 16:633](https://doi.org/10.1198/106186007X210206) (grid-based faster implementation).

  For K bins, the LGP restricted to bins is simply ρ = softmax(f), with f ~ N(m, K_θ) on bin centres.
- **Laplace approximation for LGP density estimation on a grid.** Riihimäki & Vehtari (2014) note that "the computational complexity of the proposed LA approach is dominated by the covariance matrix operations, which scale … as O(m³)" in the number of grid points m. They exploit the discretisation (FFT and Kronecker structure for dense 2-D grids), and report that with the default grid size of 400, "1D and 2D density estimation takes one to two seconds" — [Riihimäki & Vehtari 2014, Bayesian Analysis 9:425](https://doi.org/10.1214/14-BA872) ([arXiv:1211.0174](https://arxiv.org/abs/1211.0174)).
- **Astronomy use.** A hierarchical LGP for photometric redshift distributions — [Rau, Wilson & Mandelbaum 2020, MNRAS 491:4768](https://doi.org/10.1093/mnras/stz3295) ([arXiv:1904.09988](https://arxiv.org/abs/1904.09988)).
- **2026 activity.** Paulin, Jung & Benczúr sample a finite-feature LGP for conditional densities (per-galaxy photo-z is a motivating example): Fourier basis along the response and Nyström features in covariates. "Given the hyperparameters, its posterior is strongly log-concave with a uniformly bounded Hessian", and they sample it with kinetic Langevin dynamics — [arXiv:2610.09591](https://arxiv.org/abs/2610.09591). Other 2026 LGP papers: [Chen et al. arXiv:2606.22915](https://arxiv.org/abs/2606.22915) (generalised-Bayes LGP density regression) and [Kim et al. arXiv:2610.09452](https://arxiv.org/abs/2610.09452) (finite-rank LGP with exact likelihood). I checked only their titles and dates.
- **Discrete smoothness priors on log ρ.**
  - **Random walks (RW1/RW2) and intrinsic GMRFs** — [Rue & Held 2005, *Gaussian Markov Random Fields*](https://doi.org/10.1201/9780203492024).
  - **ICAR/BYM spatial priors** (a 1-D RW1 is an ICAR on a chain; a 2-D lattice ICAR suits colour–magnitude-diagram histograms) — [Besag, York & Mollié 1991, AISM 43:1](https://doi.org/10.1007/BF00116466). There is an efficient pairwise-difference ICAR implementation in Stan — [Morris et al. 2019, Spat. Spatio-temporal Epidemiol. 31:100301](https://doi.org/10.1016/j.sste.2019.100301).
  - **P-splines** — [Eilers & Marx 1996, Stat. Sci. 11:89](https://doi.org/10.1214/ss/1038425655); Bayesian P-splines — [Lang & Brezger 2004, JCGS 13:183](https://doi.org/10.1198/1061860043010).
  - **Penalised composite link model.** Ill-posed count problems with a smoothness penalty, structurally the same "blurred histogram" problem — [Eilers 2007, Stat. Modelling 7:239](https://doi.org/10.1177/1471082X0700700302).
  - **Efron's g-modelling.** A penalised spline exponential family on log g, the empirical-Bayes analogue — [Efron 2016, Biometrika 103:1](https://doi.org/10.1093/biomet/asv068).
  - **Hilbert-space approximate GPs (HSGP).** A reduced-rank basis makes GP priors cheap in probabilistic programming languages — [Riutort-Mayol et al. 2023, Stat. Comput. 33](https://doi.org/10.1007/s11222-022-10167-2).
- **Implementations of these priors** (verified by import):
  - PyMC 6.3.2: `pm.ICAR`, `pm.CAR`, `pm.ZeroSumNormal`, `pm.gp.HSGP`, `pm.gp.LatentKron`;
  - NumPyro 0.22.0: `CAR`, `ZeroSumNormal`, `ZeroSumTransform`, `numpyro.contrib.hsgp`;
  - Stan: `sum_to_zero_vector` (2.36) and `sum_to_zero_matrix` (2.37).

  Sources: [PyMC ICAR docs](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.ICAR.html), [NumPyro distributions](https://num.pyro.ai/en/stable/distributions.html), [NumPyro contrib](https://num.pyro.ai/en/stable/contrib.html), [Stan reference manual: transforms](https://mc-stan.org/docs/reference-manual/transforms.html), [Stan 2.37 release post](https://blog.mc-stan.org/2025/09/02/release-of-cmdstan-2-37/).

#### Pólya trees
- **Definition and theory.** A random measure built from a binary tree of partitions, with independent Beta splits at each node — [Ferguson 1974, Ann. Stat. 2:615](https://doi.org/10.1214/aos/1176342752); [Lavine 1992, Ann. Stat. 20:1222](https://doi.org/10.1214/aos/1176348767); [Lavine 1994, Ann. Stat. 22:1161](https://doi.org/10.1214/aos/1176325623); [Mauldin, Sudderth & Williams 1992, Ann. Stat. 20:1203](https://doi.org/10.1214/aos/1176348766). Level-dependent Beta parameters growing with depth (commonly c·j²) give absolutely continuous draws. That is the standard choice; I did not re-verify which paper it should be attributed to.
- **Extensions:**
  - mixtures of finite Pólya trees — [Hanson 2006, JASA 101:1548](https://doi.org/10.1198/016214506000000384);
  - optional Pólya trees (adaptive partitions) — [Wong & Ma 2010, Ann. Stat. 38:1433](https://doi.org/10.1214/09-AOS755);
  - Pólya-tree hierarchical model for related densities — [Christensen & Ma 2020, JRSSB 82:127](https://doi.org/10.1111/rssb.12346).
- **Software.** The main R implementation, DPpackage, was "Archived on 2019-08-04" on CRAN — [CRAN DPpackage](https://cran.r-project.org/package=DPpackage). The PyPI names `polyatree` and `pypolyatree` do not exist (PyPI JSON 404).

#### Dirichlet process, Pitman–Yor, DP mixtures
- **DP.** Defined by Dirichlet finite-dimensional marginals on every measurable partition — [Ferguson 1973, Ann. Stat. 1:209](https://doi.org/10.1214/aos/1176342360). So DP(α, G_0) restricted to K fixed bins is exactly Dir(αG_0(B_1), …, αG_0(B_K)).
- **Pitman–Yor** two-parameter Poisson–Dirichlet — [Pitman & Yor 1997, Ann. Probab. 25:855](https://doi.org/10.1214/aop/1024404422). Stick-breaking Gibbs samplers — [Ishwaran & James 2001, JASA 96:161](https://doi.org/10.1198/016214501750332758). DP mixtures for smooth densities — [Escobar & West 1995, JASA 90:577](https://doi.org/10.1080/01621459.1995.10476550); [Neal 2000, JCGS 9:249](https://doi.org/10.1080/10618600.2000.10474879).
- **Python:**
  - `pm.StickBreakingWeights` (PyMC 6.3.2, verified) — [PyMC docs](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.StickBreakingWeights.html);
  - variational DP Gaussian mixture in scikit-learn (1.9.1, 2026-09-10) — [sklearn BayesianGaussianMixture](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html);
  - `bnpy` 0.1.7 (last release 2022-04-07) — [PyPI bnpy](https://pypi.org/project/bnpy/);
  - `dpmmpython` 0.1.10 (2022-08-05; wraps Julia) — [PyPI dpmmpython](https://pypi.org/project/dpmmpython/).

#### Covariance structure: Dirichlet vs alternatives
- **Dirichlet.** All covariances are negative and set by the means and one precision (formula above, checked with SciPy). The CoDa literature lists "an almost completely negative correlation and a strong independence structure" as its known drawbacks — [shifted-scaled Dirichlet record](https://www.recercat.cat/handle/10256/13637?show=full); see also [Aitchison 1982](https://doi.org/10.1111/j.2517-6161.1982.tb01195.x).
- **Generalised Dirichlet.** "More general covariance structure" — [Wikipedia GD](https://en.wikipedia.org/wiki/Generalized_Dirichlet_distribution). Positive correlations are achievable, as in the numerical example above (derived).
- **Dirichlet-tree.** Independent variance per leaf, and positive dependence within subtrees through shared ancestors — [Minka](https://tminka.github.io/papers/dirichlet/minka-dirtree.pdf).
- **Logistic-normal / stick-breaking Gaussian.** Correlated Gaussians on the stick-breaking logits ψ put mass near the π_1 = π_2 diagonal; anti-correlated ones put mass on the edges — [Linderman et al. 2015, Fig. 1](https://arxiv.org/abs/1506.05843).

#### Overdispersion and sample-variance extensions
- **Dirichlet-multinomial (compound multinomial).** Captures extra-multinomial variation in counts — [Mosimann 1962, Biometrika 49:65](https://doi.org/10.2307/2333468). Available as `scipy.stats.dirichlet_multinomial`, NumPyro `DirichletMultinomial`, PyMC `DirichletMultinomial` and TFP `DirichletMultinomial` (all verified by import).
- **Sample variance in redshift calibration (3sDir).** "The 3-step Dirichlet sampling method consists of drawing, in sequence, values of f_T, then f_zT, then f_tzT from individual Dirichlet distributions", with variances inflated to the theoretically predicted sample-variance level. Bootstrap and basic Dirichlet resampling "are underestimated and do not cover" the truth, while 3sDir gives pulls consistent with N(0,1) — [Sánchez et al. 2020](https://doi.org/10.1093/mnras/staa2542) ([arXiv:2004.09542](https://arxiv.org/abs/2004.09542)).
- **Other routes:**
  - GD-multinomial (MGLM in R implements "generalized Dirichlet multinomial model") — [CRAN MGLM v0.2.3, 2026-08-23](https://cran.r-project.org/package=MGLM);
  - logistic-normal multinomial — [Xia et al. 2013](https://doi.org/10.1111/biom.12079);
  - Beta-Liouville multinomial — [Lakin & Abdo 2020](https://arxiv.org/abs/2006.07454).

#### Conjugacy under data augmentation, and the Pólya–gamma trick
- **Conjugate given latent counts n** (so Gibbs and EM updates are closed-form):
  - Dirichlet;
  - GD — [Wikipedia GD](https://en.wikipedia.org/wiki/Generalized_Dirichlet_distribution), [Leistedt et al. 2016](https://arxiv.org/abs/1602.05960);
  - Dirichlet-tree, including finite-depth Pólya trees as binary Dirichlet-trees — [Minka](https://tminka.github.io/papers/dirichlet/minka-dirtree.pdf);
  - DP on fixed bins (= Dirichlet) — [Ferguson 1973](https://doi.org/10.1214/aos/1176342360).
- **Not conjugate:**
  - logistic-normal and LGP — [Linderman et al.](https://arxiv.org/abs/1506.05843), [Riihimäki & Vehtari](https://arxiv.org/abs/1211.0174);
  - Gaussian smoothness priors on log ρ (by the same argument);
  - scaled Dirichlet (inference).
- **Pólya–gamma stick-breaking augmentation** (Linderman, Johnson & Adams 2015). Rewrite Mult(x | N, π) as K−1 binomials on the stick-breaking fractions π̃_k = σ(ψ_k). Using the Pólya–gamma identity of [Polson, Scott & Windle 2013, JASA 108:1339](https://doi.org/10.1080/01621459.2013.829001), p(ψ | x, ω) ∝ p(ψ) e^{κψ} e^{−ωψ²/2}, "which is Gaussian when p(ψ) is Gaussian", with ω | ψ, x ~ PG(b(x), ψ). This gives "a conditionally conjugate augmentation scheme for Gaussian priors". The transform "is asymmetric" (order-dependent), but the authors argue "this lack of symmetry does not impair the representational capacity of the model" — [arXiv:1506.05843](https://arxiv.org/abs/1506.05843). An earlier auxiliary-variable approach for multinomial logit is [Held & Holmes 2006, Bayesian Analysis 1:145](https://doi.org/10.1214/06-BA105). A tree-structured relative (logistic-tree normal: Gaussian on binary-split logits) is [LeBlanc & Ma 2023, Biometrics 79:2321](https://doi.org/10.1111/biom.13772) ([arXiv:2109.05386](https://arxiv.org/abs/2109.05386)).
- **Pólya–gamma samplers in Python** (all verified):
  - `polyagamma` 2.0.2 (2025-03-21, BSD-3, NumPy/SciPy-compatible) — [PyPI](https://pypi.org/project/polyagamma/), [GitHub zoj613/polyagamma](https://github.com/zoj613/polyagamma);
  - `pypolyagamma` 1.2.3 (2020-06-14, GPL-3) — [PyPI](https://pypi.org/project/pypolyagamma/), [GitHub slinderman/pypolyagamma](https://github.com/slinderman/pypolyagamma);
  - PyMC `PolyaGamma` and NumPyro `TruncatedPolyaGamma` (verified by import; the NumPyro version is a truncated approximation by name).

#### Summary table (each entry backed by the sources above; "derived" = my algebra)

| Prior on ρ | Free params | Covariance | Conjugate given counts? | Smoothness across bins? | Python (verified) |
|---|---|---|---|---|---|
| Dirichlet | K | all negative, rigid | yes | no | SciPy, NumPyro, PyMC, TFP |
| Generalised Dirichlet | 2(K−1) | order-dependent, can be positive (derived) | yes | no (ordering only) | efax (JAX, exp.-family form only); none in PPLs |
| Dirichlet-tree / finite Pólya tree | #edges | per-leaf variance, positive within subtrees | yes | only via tree (adjacent-bin dyadic trees) | none found |
| Beta-Liouville | ~K+1 (unverified) | more flexible than DM | reported (unverified) | no | none found |
| Scaled / shifted-scaled Dirichlet | 2K−1 (inference) | perturbation of Dirichlet | no (inference) | no | none found |
| Logistic-normal (ALR/ILR Gaussian) | (K−1)+(K−1)K/2 | arbitrary | no (PG-SB trick) | via Σ | any PPL (MvNormal + softmax) |
| RW/ICAR/GMRF, P-spline, (HS)GP on log ρ | few hyperparams | smooth, local positive corr. | no (PG-SB trick) | yes | PyMC ICAR/CAR/HSGP; NumPyro CAR/hsgp; Stan |
| DP on fixed bins | 1 + base measure | = Dirichlet | yes | no | n/a (= Dirichlet) |

### Inferences
- For pastry, the prior API can be split into two back-ends.
  - **"Conjugate" priors** (Dirichlet, GD, Dirichlet-tree/Pólya tree) get exact Gibbs and EM/MAP updates through a single interface: a `posterior_given_counts(n)` method plus a log-density.
  - **"Gaussian log-ratio" priors** (logistic-normal, RW/ICAR, P-spline, GP/HSGP, LGP) work on an unconstrained sum-to-zero vector f with ρ = softmax(f), sampled by NUTS. A Pólya–gamma stick-breaking Gibbs option is useful only if the user insists on Gibbs.
- **Identifiability of smoothness priors on log ρ.** Softmax is invariant to adding a constant. The natural parametrisation is a sum-to-zero vector: Stan's `sum_to_zero_vector` with an orthonormal basis, or `ZeroSumNormal`. This also makes ICAR/RW priors proper on the K−1-dimensional subspace (derived; consistent with Stan 2.37 building its simplex from a sum-to-zero vector).
- **Defaults for α.** BBS 2015 shows that "uninformative" Dirichlet choices are not innocuous when the effective number of counts is small relative to K. In this model the effective counts are reduced further by broad likelihoods (see Section 2), so pastry should expose α explicitly. Suggested presets are "uniform" (1), "Jeffreys" (1/2) and "BBS overall" (1/K), with a prior-sensitivity diagnostic, rather than one silent default.
- **Sample variance.** Cosmic or sample variance needs either an extra hierarchical layer (per-field ρ_f ~ Dir(λρ̄), or a logistic-normal with a theory-informed covariance) or the 3sDir-style variance inflation. A plain Dirichlet–multinomial misstates the uncertainty, per Sánchez et al.

### Gaps
- I did not obtain Wong (1998) or Connor & Mosimann (1969) full texts. The GD covariance formula is not quoted here; positive correlation is shown only by my own Monte Carlo check.
- Beta-Liouville parametrisation and conjugacy, and scaled-Dirichlet density details, were not verified from primary text (only abstracts and catalogue records).
- I could not confirm the attribution of the c·j² Pólya-tree rule, or the chapter contents of Ng, Tian & Tang (2011).
- Pitman–Yor bin marginals: I found no source stating their form. My belief that they are not Dirichlet (unless the discount is 0) is unverified.
- I found no literature comparing GD, Dirichlet-tree and Gaussian-log-ratio priors on the same deconvolution problem.

## 2. Inference algorithms for p(ρ | {L_i}) ∝ p(ρ) Π_i Σ_k L_ik ρ_k: how they compare and scale

### Takeaway
The marginal likelihood is a cheap, concave-in-ρ function: one N×K matrix–vector product per evaluation or gradient. So marginal gradient-based samplers (NUTS on an ILR/softmax-of-sum-to-zero parametrisation, which Stan made its default in 2.37) are the most robust general choice.

The others each have a place:
- **Gibbs data augmentation** is exact and simple but slows down sharply as per-object likelihoods broaden: its rate is governed by the fraction of missing information, as is EM's.
- **EM/MAP** (whose fixed point is "iterated stacking") plus a Laplace approximation is the fast first look; second-order and SQP solvers beat EM by orders of magnitude.
- **Mean-field VB** is closed-form but underestimates variance.
- **Nested sampling or SMC** only for evidences at modest K.

For N up to 10⁹, compress the likelihood matrix (group objects, sparsify) before reaching for stochastic-gradient MCMC. SG-MCMC on the simplex is biased near the boundary unless you use SCIR-type schemes. I found no published head-to-head benchmark of Gibbs vs NUTS vs VI for this exact model.

### Cited Findings

#### Structure of the target (derived unless cited)
- **Log-likelihood and derivatives.** ℓ(ρ) = Σ_i log(L_i·ρ), with gradient ∂ℓ/∂ρ_k = Σ_i L_ik/(L_i·ρ) and Hessian −Σ_i L_i L_iᵀ/(L_i·ρ)², which is negative semidefinite. So ℓ is concave on the simplex, and with a log-concave prior (Dirichlet with α_k ≥ 1) the MAP problem is convex (derived).
- **Convex-optimisation literature.** Maximum likelihood for mixture proportions with a fixed n×m likelihood matrix L is the convex problem studied by [Koenker & Mizera 2014, JASA 109:674](https://doi.org/10.1080/01621459.2013.869224) and [Kim et al. 2020, JCGS 29:261](https://doi.org/10.1080/10618600.2019.1689985) ([arXiv:1806.01412](https://arxiv.org/abs/1806.01412)).
- **Cost.** One likelihood or gradient evaluation costs O(nnz(L)), i.e. O(NK) dense (derived).

#### Gibbs data augmentation, mixing, and remedies
- **The two-block sampler.** z_i | ρ ~ Cat(r_i), then ρ | z ~ Dir(α + n). This is the Tanner–Wong data-augmentation / Diebolt–Robert mixture Gibbs sampler — [Tanner & Wong 1987, JASA 82:528](https://doi.org/10.1080/01621459.1987.10478458); [Diebolt & Robert 1994, JRSSB 56:363](https://doi.org/10.1111/j.2517-6161.1994.tb01985.x). Astronomical n(z) analyses use exactly this sampler — [Leistedt et al. 2016](https://arxiv.org/abs/1602.05960); [arXiv:2605.15078](https://arxiv.org/abs/2605.15078).
- **Convergence theory.** EM's convergence rate is governed by the fraction of missing information — [Dempster, Laird & Rubin 1977, JRSSB 39:1](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x); [Meng & van Dyk 1997, JRSSB 59:511](https://doi.org/10.1111/1467-9868.00082). Tan (2023) summarises that augmentation schemes are designed "by minimizing the fraction of missing information governing the EM algorithm convergence rate", and carries the result over to Gibbs samplers — [Tan 2023, Stat. Sci. 38:240](https://doi.org/10.1214/22-STS867) ([arXiv:1712.08887](https://arxiv.org/abs/1712.08887)). The covariance and convergence structure of two-block DA Gibbs samplers is analysed in [Liu, Wong & Kong 1994, Biometrika 81:27](https://doi.org/10.1093/biomet/81.1.27) (I did not re-read the full text).
- **Remedies (general literature).** I found none tailored to mixture weights.
  - parameter-expanded DA — [Liu & Wu 1999, JASA 94:1264](https://doi.org/10.1080/01621459.1999.10473879);
  - conditional and marginal augmentation — [Meng & van Dyk 1999, Biometrika 86:301](https://doi.org/10.1093/biomet/86.2.301);
  - review — [van Dyk & Meng 2001, JCGS 10:1](https://doi.org/10.1198/10618600152418584);
  - ancillarity–sufficiency interweaving (ASIS) — [Yu & Meng 2011, JCGS 20:531](https://doi.org/10.1198/jcgs.2011.203main).
- **Analogous evidence that DA can be far worse than marginal Metropolis.** For imbalanced logit/probit data, DA spectral gaps are bounded above by order n^{−1/2}(log n)^k, while a Metropolis sampler "cannot be worse than (log n)³". When latent data produce "very poor mixing … one can instead avoid introducing latent data", and the non-DA sampler "had dramatically better mixing" — [Johndrow et al. 2019, JASA 114:1394](https://doi.org/10.1080/01621459.2018.1505626) ([arXiv:1605.05798](https://arxiv.org/abs/1605.05798)).
- **Marginal likelihood from Gibbs output** for conjugate priors (useful for comparing α or trees) — [Chib 1995, JASA 90:1313](https://doi.org/10.1080/01621459.1995.10476635).

#### EM / MAP, and the link to iterated posterior stacking
- **EM for mixture weights with fixed components.** The update is ρ_k ← (1/N) Σ_i r_ik. EM can be "intolerably slow" — [Redner & Walker 1984, SIAM Rev. 26:195](https://doi.org/10.1137/1026034), as cited and summarised by Kim et al. 2020.
- **Faster solvers.** Interior-point methods (REBayes::KWDual via MOSEK) are "substantially faster and more accurate than EM". Mix-SQP (sequential quadratic programming with active sets and low-rank/RRQR approximations of a numerically rank-deficient L) is ~100× faster than KWDual for n > 10⁶ and m > 100 ("about 10 s vs. 1,000 s") — [Kim, Carbonetto, Stephens & Anitescu 2020](https://arxiv.org/abs/1806.01412). The R package mixsqp exists on CRAN ([Crossref record](https://doi.org/10.32614/CRAN.package.mixsqp)). Python ports belong to the EB/NPMLE note.
- **EM as mirror descent.** For exponential families, EM "can be interpreted as a mirror descent update", which gives non-asymptotic KL convergence — [Kunstner, Kumar & Schmidt 2021, arXiv:2011.01170](https://arxiv.org/abs/2011.01170).
- **Acceleration and streaming.** EM acceleration: SQUAREM — [Varadhan & Roland 2008, Scand. J. Stat. 35:335](https://doi.org/10.1111/j.1467-9469.2007.00585.x). Online/stochastic EM for streaming large N — [Cappé & Moulines 2009, JRSSB 71:593](https://doi.org/10.1111/j.1467-9868.2009.00698.x). Standard errors from EM: Louis's method — [Louis 1982, JRSSB 44:226](https://doi.org/10.1111/j.2517-6161.1982.tb01203.x).
- **Fixed point = iterated stacking** (derived). The EM fixed point satisfies ρ = (1/N) Σ_i r_i(ρ). The population estimate is the average ("stack") of per-object posteriors computed under prior ρ, iterated to self-consistency. MAP with a Dirichlet prior gives ρ_k = (α_k − 1 + Σ_i r_ik)/(N + α_0 − K), which is valid only if α_k ≥ 1; otherwise the mode is on the boundary.

#### Marginal samplers: HMC/NUTS on transformed simplices (transforms, Jacobians, geometry)
- **Stan.** Since 2.37, Stan's simplex is "the softmax of a sum-to-zero vector of length K", built from an orthonormal (ILR-type) basis, with "det(J) = ∏ x_i", i.e. log|J| = Σ_k log x_k. "All versions of Stan pre-2.37 used the stick-breaking transform." The sum-to-zero vector transform is linear, so its Jacobian is constant, and the construction "gives all marginal variances equal" — [Stan reference manual, transforms](https://mc-stan.org/docs/reference-manual/transforms.html); older stick-breaking description: [Stan 2.36 manual](https://mc-stan.org/docs/2_36/reference-manual/transforms.html). The 2.37 release says simplex and stochastic matrices now use "a new transformation under the hood" that should be "faster and more stable in different posterior geometries". It cites no paper or benchmark — [Stan blog, 2025-09-02](https://blog.mc-stan.org/2025/09/02/release-of-cmdstan-2-37/).
- **NumPyro.** The default `biject_to(simplex)` is `StickBreakingTransform`, which shifts x by log(K − k) so that 0 ↦ uniform (source read) — [NumPyro distributions/transforms](https://num.pyro.ai/en/stable/distributions.html).
- **PyMC.** The default Dirichlet transform is `SimplexTransform`: forward = log x_{1:K−1} − mean(log x) (CLR, drop last); backward = softmax([y, −Σy]). This is a sum-to-zero parametrisation in a non-orthonormal basis (source read) — [PyMC Dirichlet docs](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Dirichlet.html).
- **TFP.** The default event-space bijector for Dirichlet is `SoftmaxCentered`: "Y = exp([X 0]) / sum(exp([X 0]))", i.e. inverse ALR. `IteratedSigmoidCentered` "applies a Stick Breaking procedure", centred so 0 ↦ (1/K, …) — [TFP SoftmaxCentered](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/SoftmaxCentered), [TFP IteratedSigmoidCentered](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/IteratedSigmoidCentered).
- **Betancourt (2012).** "A series of transformations that reshape the canonical Dirichlet distribution into a form more amenable to efficient MCMC", via hyperspherical coordinates, turning the Dirichlet into "a simple product of independent Beta distributions" — [Betancourt 2012, AIP Conf. Proc. 1443:157](https://doi.org/10.1063/1.3703631) ([arXiv:1010.3436](https://arxiv.org/abs/1010.3436)).
- **Geometry in ALR/ILR coordinates** (derived). In ALR/ILR coordinates y, the Dirichlet prior plus Jacobian gives log p(y) = Σ_k α_k log ρ_k(y) + const. Here log ρ_k(y) = y_k − logsumexp(y), which is concave, so the prior is log-concave for any α_k > 0. Its tail along log ρ_k → −∞ decays like ρ_k^{α_k}, so for α_k ≪ 1 (sparse priors, empty bins) the tails become very long in y. That means long NUTS trajectories or divergences. Stick-breaking adds order dependence. A plausible explanation for Stan's switch is that the orthonormal basis makes all coordinates equivalent (inference).
- **Transform benchmark paper not found.** A Stan Discourse post (16 Nov 2022) says only that "@mjhajharia was an intern for @Bob_Carpenter this summer working on different transforms, one of which was the simplex", and links a stick-breaking Stan implementation. It gives no comparison results — [Stan Discourse](https://discourse.mc-stan.org/t/how-to-manually-code-unit-simplexes-in-stan/29275/9). A web-search snippet (not verified on the page) mentions a forthcoming paper "comparing different simplex transforms" by Carpenter, Pinkney, Axen, Siccha and Jhajharia. I could not find it on arXiv (author searches for Carpenter, Jhajharia and Pinkney returned nothing relevant).
- **Learned simplex bijections.** Aitchison-geometry bijections are used to move simplex densities to Euclidean space for flow matching — [Williams et al. 2025, arXiv:2510.27480](https://arxiv.org/abs/2510.27480) (relevant to pastry's planned flows).

#### Langevin and SG-MCMC on the simplex (minibatch scaling)
- **Patterson & Teh (2013).** They compare four parametrisations: reduced-mean, reduced-natural, expanded-mean (Gamma-expanded, πk = |θk|/Σ|θ|) and expanded-natural. They find "an unnormalized parameterisation, using a mirroring trick to remove boundaries, coupled with a natural gradient update, achieves the best mixing performance" (SGRLD). The tests use a sparse Dir(0.1) prior with K = 10 and counts (90, 5, 5, 0, …) — [Patterson & Teh 2013, NeurIPS](https://papers.nips.cc/paper/2013/hash/309928d4b100a5d75adff48a9bfc1ddb-Abstract.html).
- **Baker, Fearnhead, Fox & Nemeth (2018).** For SGRLD "the time-discretization error can dominate when we are near the boundary", producing "asymptotic biases which dominate at the boundary". Their stochastic Cox–Ingersoll–Ross process (SCIR) "removes all discretization error" and is "asymptotically unbiased" — [arXiv:1806.07137](https://arxiv.org/abs/1806.07137) (NeurIPS 2018).
- **Supporting methods and reviews:**
  - control variates for SG-MCMC — [Baker et al. 2019, Stat. Comput. 29:599](https://doi.org/10.1007/s11222-018-9826-2);
  - review — [Nemeth & Fearnhead 2021, JASA 116:433](https://doi.org/10.1080/01621459.2020.1847120);
  - mirror-map Langevin, specialised to Dirichlet posteriors, with a Õ(ε⁻²d²) rate — [Hsieh et al. 2018, arXiv:1802.10174](https://arxiv.org/abs/1802.10174).
- **Exact subsampling HMC.** HMC with energy-conserving subsampling — [Dang et al. 2019, arXiv:1708.00955](https://arxiv.org/abs/1708.00955) (JMLR) — is implemented as NumPyro `HMCECS`, which accepts a `proxy` such as `taylor_proxy` (control variates) and block updates of subsample indices (docstring read) — [NumPyro MCMC docs](https://num.pyro.ai/en/stable/mcmc.html). Related: [Quiroz et al. 2019, JASA 114:831](https://doi.org/10.1080/01621459.2018.1448827).

#### Laplace / Fisher-information approximations
- **Library support.** Laplace approximations are available as NumPyro `AutoLaplaceApproximation` and `pymc_extras.fit_laplace` (verified by import) — [NumPyro autoguide](https://num.pyro.ai/en/stable/autoguide.html), [pymc-extras](https://www.pymc.io/projects/extras/en/latest/).
- **Latent Gaussian tooling in BlackJAX 1.7.1.** An "adjoint-differentiated Laplace marginal log-density" (`laplace_hmc` family) that integrates out latent Gaussian variables, aimed at hierarchical models with latent GPs. Also `marginal_latent_gaussian`, the auxiliary-gradient sampler of Titsias & Papaspiliopoulos (2018) (docstrings read) — [BlackJAX docs](https://blackjax-devs.github.io/blackjax/).
- **LGP precedent.** Laplace for LGP density estimation: [Riihimäki & Vehtari 2014](https://doi.org/10.1214/14-BA872).
- **Observed information** (derived). I_obs = Σ_i L_i L_iᵀ/(L_i·ρ)², to be projected onto ILR coordinates; the complete-data information is N·diag(1/ρ). When L_i ≈ constant across k, I_obs → 0 and the Laplace posterior becomes prior-dominated and ill-conditioned. With empty bins the Gaussian approximation fails at the boundary.

#### Variational inference
- **Variance underestimation.** "Variational inference generally underestimates the variance of the posterior density; this is a consequence of its objective function" — [Blei, Kucukelbir & McAuliffe 2017, JASA 112:859](https://doi.org/10.1080/01621459.2017.1285773) ([arXiv:1601.00670](https://arxiv.org/abs/1601.00670)). Mixture-specific evidence (Gaussian mixtures with all parameters unknown, including Dirichlet-distributed mixing weights): "the covariance matrices from the variational Bayes approximations are normally 'too small' compared with those for the maximum likelihood estimator, so that resulting interval estimates for the parameters will be unrealistically narrow, especially if the components of the mixture model are not well separated" — [Wang & Titterington 2005, AISTATS, PMLR R5:373](https://proceedings.mlr.press/r5/wang05a.html). Poorly separated components correspond to broad, overlapping L_i here.
- **Scalable and automatic VI.** Stochastic VI for large N — [Hoffman et al. 2013, JMLR 14](https://jmlr.org/papers/v14/hoffman13a.html). ADVI in transformed space — [Kucukelbir et al. 2017, JMLR 18](https://jmlr.org/papers/v18/16-107.html). Pathfinder — [Zhang et al. 2022, JMLR 23](https://jmlr.org/papers/v23/21-0889.html), available in BlackJAX (`pathfinder`, `multipathfinder`) and pymc-extras (`fit_pathfinder`) (verified by import).
- **Flow-based guides.** NumPyro ships `AutoIAFNormal`, `AutoBNAFNormal`, `AutoDAIS` and `AutoSurrogateLikelihoodDAIS` (verified by import) — [NumPyro autoguide](https://num.pyro.ai/en/stable/autoguide.html).
- **Closed-form CAVI for this model** (derived). q(ρ) = Dir(α̃) with α̃_k = α_k + Σ_i r̃_ik and r̃_ik ∝ L_ik exp(ψ(α̃_k) − ψ(α̃_0)). Its fixed point is a "digamma-shrunk" stacking. Because q factorises z from ρ, it ignores the posterior coupling responsible for the missing-information inflation, so it is overconfident precisely when L is broad (inference, consistent with Blei et al.).

#### SMC and nested sampling for evidences
- **Foundations.** Nested sampling — [Skilling 2006, Bayesian Analysis 1:833](https://doi.org/10.1214/06-BA127). SMC samplers — [Del Moral, Doucet & Jasra 2006, JRSSB 68:411](https://doi.org/10.1111/j.1467-9868.2006.00553.x).
- **Python packages and current versions:**
  - dynesty 3.1.0 (2026-07-17) — [Speagle 2020, MNRAS 493:3132](https://doi.org/10.1093/mnras/staa278), [PyPI](https://pypi.org/project/dynesty/);
  - UltraNest 4.6.3 (2026-10-13, GPL-3.0) — [Buchner 2021, JOSS 6:3001](https://doi.org/10.21105/joss.03001), [PyPI](https://pypi.org/project/ultranest/);
  - nautilus 1.0.6 (2025-12-29, MIT) — [Lange 2023, MNRAS 525:3181](https://doi.org/10.1093/mnras/stad2441), [PyPI](https://pypi.org/project/nautilus-sampler/);
  - jaxns 3.0.0 (2026-09-30, Apache-2.0) — [Albert 2020, arXiv:2012.15286](https://arxiv.org/abs/2012.15286), [PyPI](https://pypi.org/project/jaxns/). It is wrapped by `numpyro.contrib.nested_sampling` (verified by import).
- **BlackJAX 1.7.1** exposes:
  - `ns` and `nss` ("Nested slice sampling, with hit-and-run"; the `nss.py` docstring cites "Yallup, Kroupa & Handley, 2026, arXiv:2601.23252") — [Yallup, Kroupa & Handley 2026, arXiv:2601.23252](https://arxiv.org/abs/2601.23252) (GPU-vectorised NS);
  - tempered, adaptive-tempered, persistent-sampling and `partial_posteriors_smc` SMC, the last carrying a `data_mask` for data-tempering (source read) — [BlackJAX docs](https://blackjax-devs.github.io/blackjax/).

#### Scaling to N ~ 10⁶–10⁹ and K ~ 10²–10⁴ (incl. 2-D histograms)
- **Low-rank structure of L.** At n > 10⁶ with m > 100 the matrix L is often numerically rank-deficient, and exploiting this (RRQR / low-rank) is the main speed-up in mix-SQP — [Kim et al. 2020](https://arxiv.org/abs/1806.01412).
- **Counts suffice in the conjugate model.** "The Dirichlet model only requires the bin counts" — [Leistedt et al. 2016](https://arxiv.org/abs/1602.05960). Grouping objects into cells with a shared likelihood vector (as SOM-based photo-z pipelines do; see photo-z note) turns the likelihood into a weighted sum over cells.
- **Minibatch and subsampling tools.**
  - BlackJAX `sgmcmc` (`sgld`, `sghmc`, `sgnht`, `csgld`; verified by import) — [BlackJAX sgmcmc docs](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/sgmcmc/index.html);
  - SGMCMCJax 0.2.13 (last PyPI release 2023-08-07; GitHub repo not archived) — [Coullon & Nemeth 2022, JOSS 7:4113](https://doi.org/10.21105/joss.04113), [PyPI](https://pypi.org/project/sgmcmcjax/), [GitHub](https://github.com/jeremiecoullon/SGMCMCJax);
  - SCIR reference code at github.com/jbaker92/scir (link given in the paper; not independently checked);
  - NumPyro `HMCECS` and `plate(subsample_size=…)` for SVI.
- **GP scaling in K.** Dense LGP on m grid points costs O(m³). For dense 2-D grids, use FFT/Kronecker structure — [Riihimäki & Vehtari 2014](https://arxiv.org/abs/1211.0174) — or HSGP / GMRF (sparse precision) priors — [Riutort-Mayol et al. 2023](https://doi.org/10.1007/s11222-022-10167-2); [Rue & Held 2005](https://doi.org/10.1201/9780203492024).

#### Empty bins and support
- **Boundary bias in Langevin samplers.** Near-zero components ("sparse simplex") are where Langevin-type samplers fail: SGRLD bias "dominate[s] at the boundary" — [Baker et al. 2018](https://arxiv.org/abs/1806.07137); [Patterson & Teh 2013](https://papers.nips.cc/paper/2013/hash/309928d4b100a5d75adff48a9bfc1ddb-Abstract.html).
- **Improper posterior.** An improper Dirichlet (a → 0) gives an improper posterior when cells are empty — [Berger, Bernardo & Sun 2015](https://arxiv.org/abs/1504.02689).
- **Further behaviour (derived):**
  - a zero is absorbing for EM, since ρ_k = 0 ⇒ r_ik = 0;
  - MAP with α_k < 1 is not interior;
  - a bin with L_{·k} ≈ 0 for every object is not identified by the data, so its posterior equals its prior conditional;
  - softmax-type parametrisations can never put exact zeros, which is acceptable for posterior sampling but means MAP-in-y diverges if the MLE sits on the boundary.

#### Benchmarks found (none for the exact model with Gibbs vs NUTS vs VI)
- MLE solvers for mixture proportions (EM vs interior point vs mix-SQP) — [Kim et al. 2020](https://arxiv.org/abs/1806.01412).
- Langevin parametrisations on the simplex — [Patterson & Teh 2013](https://papers.nips.cc/paper/2013/hash/309928d4b100a5d75adff48a9bfc1ddb-Abstract.html).
- SGRLD vs SCIR — [Baker et al. 2018](https://arxiv.org/abs/1806.07137).
- DA vs Metropolis in imbalanced categorical models — [Johndrow et al. 2019](https://arxiv.org/abs/1605.05798).
- Uncertainty calibration of bootstrap vs Dirichlet vs 3sDir — [Sánchez et al. 2020](https://arxiv.org/abs/2004.09542).
- Estimator comparisons (stacking vs marginal-likelihood vs full posterior) for photo-z n(z) are in [Malz & Hogg 2022, ApJ 928:127](https://doi.org/10.3847/1538-4357/ac062f); details are in the photo-z note.

### Inferences
- **Missing information as the mixing diagnostic** (derived). The complete-data information is N diag(1/ρ). The observed information is Σ_i L_iL_iᵀ/(L_i·ρ)².
  - With one-hot L (perfect measurements) the fraction of missing information is 0, and Gibbs mixes in one step.
  - With L_i nearly flat across bins it tends to 1, so Gibbs autocorrelation and EM iteration counts blow up.

  This is the regime of broad photo-z PDFs and of objects near the CMD noise floor. Pastry can compute an estimate of this fraction cheaply at the MAP (from I_obs and I_com) and use it to recommend Gibbs vs NUTS automatically.
- **Recommended default stack:**
  1. Compress L: deduplicate or cluster identical or near-identical rows into weights w_c, and use sparse storage for compact PDFs.
  2. MAP via EM accelerated with SQUAREM, or a convex solver; report the fraction of missing information and a Laplace approximation in ILR coordinates.
  3. NUTS (NumPyro/BlackJAX) on a sum-to-zero ILR vector with ρ = softmax(Hy), adding Σ_k log ρ_k for the Jacobian (Stan 2.37 convention). This handles Dirichlet, logistic-normal, ICAR and GP priors identically.
  4. Optional exact Gibbs for the conjugate priors (Dirichlet, GD, Dirichlet-tree), used as a cross-check.
  5. Closed-form mean-field VB as a "fast but overconfident" mode, clearly labelled.
- **Compute and memory at the top of the range** (derived). One dense evaluation at N = 10⁹, K = 10³ is ~2×10¹² flops; float32 storage of L would be ~4 TB. So for N ≳ 10⁸ the bottleneck is memory and I/O, not flops. Compression (cells or clusters, sparsity, low rank as in mix-SQP) is mandatory, and streaming sufficient statistics or minibatching comes second.
- **2-D histograms with K up to 10⁴.**
  - The likelihood side is unchanged: O(nnz(L)).
  - Smoothness priors must have sparse precision (ICAR/RW2 on the lattice) or Kronecker/HSGP structure. A dense GP at K = 10⁴ (O(K³) ≈ 10¹² per Cholesky, per hyperparameter step) is impractical.
  - NUTS in 10⁴ dimensions with a diagonal mass matrix is routine if the posterior is reasonably conditioned.
- **Evidences.** Nested sampling in a K−1 = 10²–10⁴ dimensional space is expensive. Use NS (dynesty/UltraNest/nautilus/jaxns/BlackJAX NSS) for small K or few hyperparameters. Prefer Chib's estimator (conjugate Gibbs), SMC/annealed importance sampling, or bridge sampling for larger K (inference, not benchmarked).
- **Empty or unsupported bins.**
  - Detect bins with Σ_i L_ik ≈ 0 and either drop them or warn that they are prior-determined.
  - Default α_k ≥ 1 for MAP.
  - For sampling with α_k < 1, prefer Gibbs or SCIR-type schemes, or use a smoothness prior that pools neighbouring bins.

### Gaps
- I found no published benchmark comparing Gibbs DA, NUTS (by transform), mean-field VB and EM+Laplace on Π_i Σ_k L_ik ρ_k as likelihood breadth varies. Pastry will need its own; this note's FMI-based framing gives the axis to vary.
- The Stan team's simplex-transform comparison could not be located as a paper; the only verified trace is the 2022 Discourse note about Jhajharia's internship work on transforms. No quantitative ESS-per-gradient comparison of stick-breaking vs ALR vs ILR vs expanded-Gamma under NUTS was found.
- I did not find parameter-expansion or ASIS schemes specific to finite-mixture weights with known components.
- The scaling statements (flop and memory estimates) are back-of-envelope (derived), not measured.
- Liu, Wong & Kong (1994) and van Dyk & Meng (2001) full texts were not re-read. The precise "DA rate = maximal fraction of missing information" statement is carried here via Tan (2023) and general EM theory.

## 3. Python library support in October 2026

### Takeaway
Every major JAX/PyTensor/TF probabilistic programming language ships the Dirichlet, Dirichlet-multinomial and multinomial distributions and a simplex bijector, but they disagree on the transform:

| Library | Default simplex transform |
|---|---|
| NumPyro | stick-breaking |
| PyMC | CLR-drop-last softmax |
| TFP | ALR / SoftmaxCentered |
| Stan ≥ 2.37 | orthonormal sum-to-zero (ILR) + softmax |

None ships a generalised Dirichlet, a Dirichlet-tree, a Pólya tree, Beta-Liouville or a scaled Dirichlet. The only Python GD I found is the exponential-family implementation in `efax` (JAX). Pastry should therefore implement GD and Dirichlet-tree itself, which is straightforward via Beta/Dirichlet factors. It can rely on NumPyro/BlackJAX for samplers, HSGP/ICAR/ZeroSum components and the Pólya–gamma, nested-sampling and SG-MCMC tooling. Note that stable TensorFlow Probability has not been released since November 2024, and its JAX substrate failed to import with current JAX in this environment.

### Cited Findings

#### Library-by-library (versions from PyPI JSON on 2026-10-22; capabilities verified by import/attribute lookup)

| Library (version, date, licence) | Simplex distributions | Simplex transforms | Relevant inference | Status |
|---|---|---|---|---|
| SciPy 1.18.1 (2026-08-21, BSD) | `dirichlet` (pdf, logpdf, mean, var, cov, entropy, rvs); `dirichlet_multinomial` (pmf, logpmf, mean, var, cov; no rvs); `multinomial` | — | — | active |
| NumPyro 0.22.0 (2026-09-18, Apache-2.0) | `Dirichlet`, `DirichletMultinomial`, `Multinomial`, `Categorical`, `MixtureSameFamily`, `MixtureGeneral`, `CAR`, `ZeroSumNormal`, `TruncatedPolyaGamma` | `StickBreakingTransform` (default), `ZeroSumTransform`, `SimplexToOrderedTransform` | NUTS, HMC, HMCECS, MixedHMC, DiscreteHMCGibbs, HMCGibbs, AIES, ESS, BarkerMH, SA; SVI + autoguides (Delta, Normal, MVN, LowRank, Laplace, IAF, BNAF, DAIS); contrib: hsgp, nested_sampling, funsor, einstein, stochastic_support | active (GitHub `updated_at` 2026-10-22) |
| PyMC 6.3.2 (2026-09-08, Apache-2.0); pymc-extras 0.15.1 (2026-09-16) | `Dirichlet`, `DirichletMultinomial`, `Multinomial`, `Categorical`, `Mixture`, `StickBreakingWeights`, `ICAR`, `CAR`, `ZeroSumNormal`, `PolyaGamma`; `gp.HSGP` | `SimplexTransform` (default; CLR-drop-last) | NUTS etc.; pymc-extras: `marginalize`/`recover_marginals` (discrete latents), `fit_laplace`, `fit_pathfinder`, `find_MAP`, `histogram_approximation` | active |
| TensorFlow Probability 0.25.0 (2024-11-08, Apache-2.0); tfp-nightly 0.26.0.dev20261021 | `Dirichlet`, `FlatDirichlet`, `DirichletMultinomial`, `Multinomial`, `GaussianProcess`, `VariationalGaussianProcess` | `SoftmaxCentered` (Dirichlet default), `IteratedSigmoidCentered` (stick-breaking) | `tfp.mcmc` HMC, NUTS, SliceSampler, MALA, RWM, ReplicaExchangeMC, `sample_annealed_importance_chain`; `tfp.vi` | no stable release for ~2 years; nightly still published; repo not archived |
| BlackJAX 1.7.1 (2026-10-02, Apache-2.0) | (none; user supplies log-density) | (none) | nuts, hmc, mclmc/adjusted_mclmc, mala, barker, elliptical_slice, slice_sampling, rmhmc; window/chees/meads adaptation; pathfinder, meanfield_vi, fullrank_vi, svgd; sgmcmc (sgld, sghmc, sgnht, csgld); SMC (tempered, adaptive, persistent, partial_posteriors); ns/nss; laplace_hmc; marginal_latent_gaussian | active |
| Stan (latest conda-forge CmdStan 2.40.0, listing modified 2026-09-24) via cmdstanpy 1.3.0 (2025-10-20), BridgeStan 2.9.0 (2026-07-06), PyStan 3.14.0 (2026-09-11) | `dirichlet`, `multinomial`, `categorical` (and more) | `simplex` = softmax(sum_to_zero) since 2.37; `sum_to_zero_vector` / `_matrix`; row/column stochastic matrices; `*_constrain`/`*_unconstrain`/`*_jacobian` functions (2.37) | NUTS, ADVI, Pathfinder (Stan's algorithm list not re-verified in this session); BridgeStan ("Access the methods of a Stan model in Python") exposes log-density + gradient to Python samplers | active |

Sources:
- [SciPy dirichlet](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet.html), [dirichlet_multinomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet_multinomial.html), [multinomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multinomial.html), [PyPI scipy](https://pypi.org/project/scipy/).
- [NumPyro distributions](https://num.pyro.ai/en/stable/distributions.html), [MCMC](https://num.pyro.ai/en/stable/mcmc.html), [autoguide](https://num.pyro.ai/en/stable/autoguide.html), [contrib](https://num.pyro.ai/en/stable/contrib.html), [PyPI numpyro](https://pypi.org/project/numpyro/), [GitHub](https://github.com/pyro-ppl/numpyro).
- [PyMC Dirichlet](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Dirichlet.html), [PyPI pymc](https://pypi.org/project/pymc/), [pymc-extras docs](https://www.pymc.io/projects/extras/en/latest/), [PyPI pymc-extras](https://pypi.org/project/pymc-extras/), [GitHub](https://github.com/pymc-devs/pymc).
- [TFP Dirichlet](https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Dirichlet), [SoftmaxCentered](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/SoftmaxCentered), [IteratedSigmoidCentered](https://www.tensorflow.org/probability/api_docs/python/tfp/bijectors/IteratedSigmoidCentered), [PyPI tensorflow-probability](https://pypi.org/project/tensorflow-probability/), [PyPI tfp-nightly](https://pypi.org/project/tfp-nightly/), [GitHub](https://github.com/tensorflow/probability).
- [BlackJAX docs](https://blackjax-devs.github.io/blackjax/), [PyPI blackjax](https://pypi.org/project/blackjax/), [GitHub](https://github.com/blackjax-devs/blackjax).
- [Stan transforms](https://mc-stan.org/docs/reference-manual/transforms.html), [Stan 2.37 release](https://blog.mc-stan.org/2025/09/02/release-of-cmdstan-2-37/), [conda-forge cmdstan](https://anaconda.org/conda-forge/cmdstan), [PyPI cmdstanpy](https://pypi.org/project/cmdstanpy/), [PyPI bridgestan](https://pypi.org/project/bridgestan/), [PyPI pystan](https://pypi.org/project/pystan/).

Further verified details:
- **TFP's stable JAX substrate is broken with current JAX.** In this session `tensorflow_probability.substrates.jax` from stable 0.25.0 failed to import under JAX 0.12.0.1 (`AttributeError: module 'jax.interpreters.xla' has no attribute 'pytype_aval_mappings'`). tfp-nightly 0.26.0.dev20261021 imported fine (local import test; release dates from [PyPI tensorflow-probability](https://pypi.org/project/tensorflow-probability/) and [PyPI tfp-nightly](https://pypi.org/project/tfp-nightly/)).
- **No relatives of the Dirichlet in the installed libraries.** A grep of installed numpyro, pymc, pymc-extras, pytensor, tensorflow_probability (stable and nightly), scipy, blackjax, scikit-bio, pyrolite and preliz found no `GeneralizedDirichlet`, `DirichletTree`, `BetaLiouville`, `ScaledDirichlet` or `LogisticNormal` class (local source grep, 2026-10-22). The libraries' distribution listings agree — [NumPyro distributions](https://num.pyro.ai/en/stable/distributions.html), [PyMC Dirichlet docs](https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.Dirichlet.html).
- **PyMC and NumPyro can marginalise z for you, but known L needs no enumeration.**
  - In PyMC, `pymc_extras.marginalize` handles discrete z_i automatically.
  - NumPyro offers funsor-based enumeration (`TraceEnum_ELBO`, `contrib.funsor`) and `DiscreteHMCGibbs`/`MixedHMC` for sampling z.
  - With known L, the simplest marginal model is a single factor Σ_i logsumexp(log L_i + log ρ), so neither is needed (inference from the APIs verified above).

#### Generalised Dirichlet and Dirichlet-tree availability
- **efax 2.6.0** (2026-10-10, Apache-2.0; "Exponential families for JAX") provides `GeneralizedDirichletNP` and `GeneralizedDirichletEP` "based off … Wong 1998". It includes natural and expectation parametrisations, log-normaliser (sum of betaln), sufficient statistics (log x, log(1 − cumsum x)) and entropy, on a "subsimplex" support. I saw no sampling method in that file — [PyPI efax](https://pypi.org/project/efax/), [GitHub NeilGirdhar/efax](https://github.com/NeilGirdhar/efax), [source file](https://raw.githubusercontent.com/NeilGirdhar/efax/master/efax/_src/distributions/gen_dirichlet.py). GitHub code search for "GeneralizedDirichlet" in Python otherwise found only an actuarial library (ProteusLLP) and an unrelated neuroimaging repo (search on 2026-10-22).
- **No Dirichlet-tree on PyPI.** The names `generalized-dirichlet`, `gdirichlet`, `dirichlet-tree`, `dirichlettree`, `beta-liouville`, `logistic-normal`, `polyatree`, `pypolyatree`, `dirichlet-process` and `pitman-yor` all return 404 on PyPI. `pydtm` exists but is an unrelated API client (PyPI JSON, 2026-10-22). GitHub repository search for "dirichlet tree" in Python found nothing relevant.
- **R/C++ references usable for cross-validation:**
  - **MGLM** v0.2.3 (2026-08-23, GPL-2|3) implements Dirichlet-multinomial, generalised Dirichlet-multinomial and negative-multinomial models with MM/Newton MLE — [CRAN MGLM](https://cran.r-project.org/package=MGLM).
  - **elections.dtree** (R/C++ Dirichlet-tree for IRV audits) was "Archived on 2025-02-09 as requires archived package 'prefio'"; the GitHub repo still exists (C++) — [CRAN](https://cran.r-project.org/package=elections.dtree), [GitHub fleverest/elections.dtree](https://github.com/fleverest/elections.dtree).
  - **hyper2** (hyper-Dirichlet) is current — [CRAN hyper2](https://cran.r-project.org/package=hyper2).
  - **DPpackage** (Pólya trees, DP) was archived in 2019 — [CRAN](https://cran.r-project.org/package=DPpackage).

#### Compositional-data (CoDa) packages (log-ratio transforms, zero handling)

| Package (version, date, licence) | What it provides | Status / sources |
|---|---|---|
| scikit-bio 0.7.4 (2026-09-21, BSD-3) | `skbio.stats.composition`: `closure`, `clr`/`clr_inv`, `ilr`/`ilr_inv` (custom basis), `alr`/`alr_inv`, `sbp_basis`, `tree_basis`, `multi_replace` (zero replacement), `perturb`, `power`, `inner`, `rclr`, `vlr`, `ancom`, `ancombc`, `dirmult_ttest`, `dirmult_lme` | active — [docs](https://scikit.bio/docs/latest/generated/skbio.stats.composition.html), [PyPI](https://pypi.org/project/scikit-bio/) |
| pyrolite 0.3.8.post1 (2026-09-29, MIT) | `pyrolite.comp.codata`: ALR/CLR/ILR and inverses, `helmert_basis`, Box–Cox, sphere transforms | active — [docs](https://pyrolite.readthedocs.io/en/main/), [PyPI](https://pypi.org/project/pyrolite/) |
| composition_stats 2.0.0 (2021-10-10, BSD-3) | small CoDa package | last release 2021; repo still updated — [PyPI](https://pypi.org/project/composition-stats/), [GitHub](https://github.com/ntessore/composition_stats) |
| compositional 2026.5.11 (BSD-3) | CLR/ILR, proportionality | [PyPI](https://pypi.org/project/compositional/), [GitHub](https://github.com/jolespin/compositional) |
| pycodamath 1.1.4 (2026-06-30, MIT; Bitbucket) | CoDa tools | [PyPI](https://pypi.org/project/pycodamath/) |
| gneiss 0.4.6 (2019) | balance trees | stale — [PyPI](https://pypi.org/project/gneiss/) |

None of these packages defines priors or samplers on the simplex.

### Inferences
- **Transform.** Pastry should not inherit a library's default simplex transform implicitly. Defaults differ (stick-breaking in NumPyro, CLR in PyMC, ALR in TFP, ILR in Stan) and they change: Stan switched in 2.37 with no published benchmark. A pastry-owned `ILRSimplex` bijector (orthonormal Helmert-type basis H, ρ = softmax(Hy), log|J| = Σ_k log ρ_k + const) built on NumPyro's `ZeroSumTransform` or plain JAX would match Stan ≥ 2.37 and keep results reproducible across back-ends.
- **Back-end.** JAX is the most practical: NumPyro supplies distributions, SVI, HMCECS, hsgp and nested-sampling wrappers, and BlackJAX supplies samplers, SG-MCMC, SMC and NS. TFP should be avoided as a hard dependency given the stale stable release and the import failure observed with current JAX. PyMC is a reasonable optional front-end (ICAR, HSGP, marginalize).
- **New code pastry must own:**
  - GD log-density, sampler and conjugate update;
  - Dirichlet-tree and Pólya-tree classes, parametrised by a tree over adjacent bins with per-node Dirichlet α, a conjugate posterior and a sampler;
  - the closed-form CAVI and EM/MAP for the mixture-weights model.

  Cross-checks can use efax (GD normaliser/entropy), MGLM (GD-multinomial MLE, in R), SciPy (`dirichlet`, `dirichlet_multinomial`) and polyagamma (Pólya–gamma draws).

### Gaps
- I could not run the GitHub API in this session (403 except for the pastry repo), so maintenance signals for some repos (efax, SGMCMCJax, polyagamma, composition_stats, elections.dtree) rest on GitHub search metadata (`updated_at`), which also changes with stars, not on commit history.
- I did not check PyTorch/Pyro simplex transforms (pyro-ppl 1.9.2, 2026-10-03, exists on PyPI) in this session; the docs fetch failed.
- I did not test BlackJAX `nss` or `laplace_hmc` beyond reading their docstrings.
- I did not verify whether Stan's embedded Laplace approximation (if any) is available from Python.
