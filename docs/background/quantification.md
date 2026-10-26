# Quantification and mixture proportions

Machine learning calls the problem *quantification*, or estimation under *label shift*:
a classifier trained on one set of labelled data is applied to an unlabelled set whose
class proportions differ, and the task is to estimate those proportions rather than to
label each item. Fisheries genetics solved the same problem decades earlier as genetic
stock identification, and epidemiology, microbial ecology and RNA-seq quantification
solve it under other names. These fields ship the Bayesian version of binference's model
for classifiers and genotypes.

## The problem in these fields' terms

**Quantification.** A classifier gives each test item $x_i$ a posterior $p(k\mid x_i)$
learned under the training prevalence $\pi_\text{train}$. Under label shift the
class-conditional distributions $p(x\mid k)$ are the same in training and test, and only
the prevalence changes. Methods divide into:

- *classify and count* (CC) and its probabilistic version (PCC), which average hard or
  soft predictions;
- *adjusted* methods (ACC, BBSE), which correct the counts with the confusion matrix
  estimated on labelled data (Lipton, Wang & Smola 2018);
- *likelihood* methods: EMQ or SLD (Saerens, Latinne & Decaestecker 2002), also called
  MLLS, which iterates the class prevalences and the posteriors corrected for them;
- *Bayesian* methods that sample the prevalence: BayesianCC (Ziegler & Czyż 2024),
  which learns the confusion matrix jointly with the prevalence, and BayesianMAPLS and
  BayesianKDEy in QuaPy.

**Genetic stock identification** (GSI) estimates the proportions of source populations
in a mixed fishery catch. Each fish's genotype has a likelihood under each baseline
population's allele frequencies, and the mixture proportions are estimated by EM as a
conditional maximum likelihood (SPAM, ONCOR) or by Gibbs sampling with a Dirichlet prior
and latent assignments (BAYES, rubias). Microbial source tracking (SourceTracker, FEAST)
and RNA-seq transcript quantification (salmon, kallisto) solve the same EM or Gibbs
problem over reads.

**Epidemiology** estimates true prevalence from an imperfect test. The Rogan–Gladen
estimator corrects the apparent prevalence with the test's sensitivity and specificity,
which is ACC for two classes; Bayesian latent-class models put priors on sensitivity
and specificity.

## In binference's terms

The {doc}`user guide <../user-guide/inputs>` describes two routes from a classifier to
the population, with the conditions under which each is exact.

- **Route A: posteriors and the training prevalence.**
  `Likelihoods.from_table(probs, Bins.categories(classes), interim=train_prior)` divides
  the training prevalence out of each posterior, $L_{ik}\propto p(k\mid x_i)/
  \pi_{\text{train},k}$. Maximising the likelihood is then EMQ, and sampling the
  posterior is BayesianMAPLS.
- **Route C: scores and a labelled validation set.**
  `Likelihoods.from_binned(Response.from_labelled(val_scores, val_labels),
  values=scores)` bins the scores (or uses the predicted labels) and uses the
  validation set's confusion matrix as a response, inferred together with the
  proportions. This is BayesianCC.
- **Genotype likelihoods** under each baseline population are a likelihood matrix on
  `Bins.categories(populations)`: `Likelihoods(L, bins)` ({doc}`mixed-stock tutorial
  <../tutorials/10_mixed_stocks>`).
- **PCC** is stacking, one EM step from the training prevalence (`lk.stack()`).
- **Ordinal classes** are ordered bins, so the smoothness priors apply
  (`LogisticRandomWalk`), the Bayesian counterpart of the first- and second-difference
  penalties of ordinal quantifiers.

The {doc}`categories tutorial <../tutorials/06_categories_unfolding>` runs both routes on
a mock classifier under prior shift.

## Methods and software

- **QuaPy** ([HLT-ISTI/QuaPy](https://github.com/HLT-ISTI/QuaPy); BSD-3, version 0.2.3)
  is the reference Python library. Its EMQ code divides the classifier posteriors by the
  training prevalence and averages them, which is EM with the interim prior corrected.
  BayesianMAPLS samples $\sum_i\operatorname{logsumexp}_k(\log p(k\mid x_i)+
  \log\pi_{\text{test},k}-\log\pi_{\text{train},k})$ under a Dirichlet prior with NumPyro
  NUTS; BayesianKDEy uses class-conditional kernel density estimates of the classifier
  outputs as $L_{ik}$; BayesianCC follows Ziegler & Czyż. Every confidence-aware method
  returns an estimate and a region from `predict_conf`, with regions that report their
  coverage, and `calibrate_temperature` chooses a likelihood temperature against
  validation coverage. Its inputs are features and a scikit-learn classifier; no
  documented entry point takes a likelihood matrix, and none adds smoothness or
  selection.
- **mlquantify** (Python; BSD-3) implements 33 methods with bootstrap confidence
  regions. **qunfold** ([mirkobunse/qunfold](https://github.com/mirkobunse/qunfold))
  implements quantification and unfolding methods in one framework (Bunse 2022),
  including a likelihood maximiser with ordinal smoothness penalties, but reports no
  uncertainty. **labelshift** is the research code of BayesianCC; **abstention** provides
  calibration (temperature and bias-corrected temperature scaling) and label-shift
  estimators.
- **rubias** ([eriqande/rubias](https://github.com/eriqande/rubias); R/C++) runs
  Dirichlet–multinomial Gibbs sampling with a parametric-bootstrap bias correction and
  baseline resampling, and returns traces, per-fish posteriors and posterior-predictive
  totals. CRAN archived it in February 2026 because the maintainer's email was
  undeliverable. ONCOR and SPAM are legacy desktop programs for the EM estimate;
  Ms.GSI chains two Pella–Masuda models; mixstock (R) offers EM and Gibbs for sea-turtle
  mixtures.
- **SourceTracker2** ([caporaso-lab/sourcetracker2](https://github.com/caporaso-lab/sourcetracker2))
  is the only Python tool in this family; its last PyPI release is from 2016.
  **salmon** and **kallisto** scale EM, variational EM and Gibbs replicates to $10^8$
  reads by collapsing reads with the same compatibility pattern into equivalence classes.
- **epiR** and **prevalence** (R) implement Rogan–Gladen and Bayesian true-prevalence
  estimates for two classes.

No Python GSI or mixed-stock tool was found, and no package in this family documents a
selection term.

## How binference relates

**Shared.** The likelihood of EMQ, BayesianMAPLS and BayesianKDEy, the Dirichlet prior,
and NUTS on the simplex; the confusion-matrix model of BayesianCC, with the matrix
inferred from labelled counts; the Gibbs sampler of BAYES and rubias, with per-object
assignment probabilities and draws of the realised counts in each class
(`result.realized_counts`).

**Added.** A likelihood matrix as direct input, independent of how it was produced;
selection effects; smoothness priors for ordered classes; compression of duplicated
rows, as salmon's equivalence classes do (`lk.compress()`); and simulation-based
calibration of the whole analysis.

**Not done.** binference does not train or calibrate classifiers, has no likelihood
temperature, does not estimate the effective training prevalence from held-out
posteriors (it must be given), returns Bayesian draws rather than bootstrap confidence
regions in CLR coordinates, and does not model uncertainty in a genetic baseline beyond
what a tallied response carries.

## Pitfalls known in these fields

- **Calibration.** The EM quantifier assumes calibrated posteriors, "which is not true of
  modern neural networks"; maximum likelihood with bias-corrected calibration
  outperforms BBSE and RLLS (Alexandari, Kundaje & Shrikumar 2020). MLLS is consistent
  only with calibration and an invertible confusion matrix (Garg et al. 2020). QuaPy's
  temperature calibration is a response to classifier-derived likelihoods that are too
  confident.
- **Bootstrap regions.** Standard bootstrap methods "cannot provide valid coverage
  guarantees under prior probability shift"; they "merely quantify the internal
  variability of the quantifier" (Moreo & Salvati 2025).
- **Classify and count** is biased under prior shift; in fisheries, proportional methods
  (CML or BAYES) perform better than assigning each individual (Koljonen et al. 2005, as
  summarised by the Alaska Department of Fish and Game).
- **Prior strength.** Informed priors give less biased results than flat ones "especially
  where there is little genetic [differentiation]", the analogue of broad likelihoods.
  rubias warns that a Dirichlet prior with total mass much below one can make the
  sampler "mix more poorly".
- **Aggregation bias.** A symmetric Dirichlet over fine groups gives each coarse group
  prior mass in proportion to its number of fine groups, which biases coarse proportions
  when groups are uneven (rubias's "reporting-unit" bias). In binference, sum fine
  fractions per draw with `result.rebin` and choose the prior's concentration per bin
  ({doc}`simplex-priors`).
- **Inversion leaves the simplex.** Plug-in corrections such as Rogan–Gladen can return
  negative prevalences; epiR recommends a Bayesian approach whenever the apparent
  prevalence is below one minus the specificity. A posterior on the simplex cannot leave
  it.
- **Fields unaware of each other.** A 2023 comparison of ordinal quantifiers found
  methods "proposed by authors from very different research fields, such as data mining
  and astrophysics, who were unaware of each others' developments" (Bunse et al. 2023).

## References

- Alexandari, Kundaje & Shrikumar (2020), *ICML* ([arXiv:1901.06852](https://arxiv.org/abs/1901.06852)): maximum likelihood with calibration for label shift.
- Bunse (2022), *INFORMATIK 2022* ([doi:10.18420/inf2022_37](https://doi.org/10.18420/inf2022_37)): unification of quantification and unfolding.
- Bunse, Moreo, Sebastiani & Senz (2023) ([arXiv:2310.09210](https://arxiv.org/abs/2310.09210)): ordinal quantification.
- Garg, Wu, Balakrishnan & Lipton (2020), *NeurIPS* ([arXiv:2003.07554](https://arxiv.org/abs/2003.07554)): a unified view of label-shift estimation.
- Lipton, Wang & Smola (2018) ([arXiv:1802.03916](https://arxiv.org/abs/1802.03916)): black-box shift estimation.
- Moran & Anderson (2019), *Can. J. Fish. Aquat. Sci.* 76, 551 ([doi:10.1139/cjfas-2018-0016](https://doi.org/10.1139/cjfas-2018-0016)): rubias.
- Moreo & Salvati (2025), [LQ 2025 proceedings](https://lq-2025.github.io/proceedings/CompleteVolume.pdf): bootstrap confidence regions for quantification.
- Patro et al. (2017), *Nature Methods* 14, 417 ([doi:10.1038/nmeth.4197](https://doi.org/10.1038/nmeth.4197)): salmon.
- Pella & Masuda (2001), *Fishery Bulletin* 99, 151: BAYES.
- Rogan & Gladen (1978), *Am. J. Epidemiol.* 107, 71 ([doi:10.1093/oxfordjournals.aje.a112510](https://doi.org/10.1093/oxfordjournals.aje.a112510)): prevalence from an imperfect test.
- Saerens, Latinne & Decaestecker (2002), *Neural Computation* 14, 21 ([doi:10.1162/089976602753284446](https://doi.org/10.1162/089976602753284446)): EM for class priors.
- Ye, Tsuchida, Petersson & Barnes (2024), *WACV* ([doi:10.1109/WACV57701.2024.00111](https://doi.org/10.1109/WACV57701.2024.00111)): MAPLS.
- Ziegler & Czyż (2024), *TMLR* ([arXiv:2302.09159](https://arxiv.org/abs/2302.09159)): Bayesian quantification with black-box estimators.
