# Quantification and prevalence estimation

What does machine learning call binference's problem? It calls it quantification, or
estimation under label shift. A classifier trained on labelled data is applied to an
unlabelled set whose class proportions differ. The task is to estimate those proportions,
not to label each item. Epidemiology solves the same problem for an imperfect test.
Ecology and genetics solve it for genotypes and reads; see {doc}`ecology`.

## The problem in these fields' terms

**Quantification.** A classifier gives each test item $x_i$ a posterior $p(k\mid x_i)$,
learned under the training prevalence $\pi_\text{train}$. Under label shift the
class-conditional distributions $p(x\mid k)$ are the same in training and test. Only the
prevalence changes. The methods fall into four groups:

- **Classify and count** (CC) and its probabilistic version (PCC) average hard or soft
  predictions.
- **Adjusted** methods (ACC, BBSE) correct the counts with a confusion matrix estimated
  on labelled data {cite:p}`lipton2018`.
- **Likelihood** methods iterate the prevalences and the posteriors corrected for them.
  The method is EMQ or SLD {cite:p}`saerens2002`, also called MLLS.
- **Bayesian** methods sample the prevalence. BayesianCC {cite:p}`ziegler2024` learns
  the confusion matrix jointly with the prevalence. BayesianMAPLS builds on MAPLS
  {cite:p}`ye2024`. BayesianKDEy is a third, in QuaPy.

**Epidemiology** estimates true prevalence from an imperfect test. The Rogan–Gladen
estimator {cite:p}`rogan1978` corrects the apparent prevalence with the test's
sensitivity and specificity. It is ACC for two classes. Bayesian latent-class models put
priors on sensitivity and specificity.

## In binference's terms

| quantification term | binference |
|---|---|
| prevalence | population fractions $\rho$ on `Bins.categories(classes)` |
| training prevalence | the interim prior, `interim=` |
| confusion matrix | a {class}`~binference.Response` |
| EMQ, SLD, MLLS | the MAP with posteriors and the training prevalence |
| BayesianMAPLS | the posterior with posteriors and the training prevalence |
| BayesianCC | `from_binned` with `Response.from_labelled`, inferred with the fractions |
| PCC | stacking, `lk.stack()` |
| ordinal penalties | `LogisticRandomWalk` on ordered bins |
| the model behind Rogan–Gladen | a two-class `Response` with known sensitivity and specificity |

A classifier reaches binference by one of two routes. The
[inputs guide](../guides/inputs.md#binned-measurements-and-classifiers) gives the
conditions under which each is exact.

- **Posteriors and the training prevalence.**
  `Likelihoods.from_table(probs, Bins.categories(classes), interim=train_prior)` divides
  the training prevalence out: $L_{ik}\propto p(k\mid x_i)/\pi_{\text{train},k}$.
  Maximising the likelihood is then EMQ. Sampling the posterior is BayesianMAPLS.
- **Scores and a labelled set.** `Likelihoods.from_binned(Response.from_labelled(
  val_scores, val_labels), values=scores)` bins the scores. It uses the labelled set's
  confusion matrix as a response, inferred with the proportions. This is BayesianCC.

The {doc}`categories tutorial <../tutorials/06_categories_unfolding>` runs both routes on
a mock classifier under prior shift.

## Methods and software

- **QuaPy** ([HLT-ISTI/QuaPy](https://github.com/HLT-ISTI/QuaPy); BSD-3, version 0.2.3)
  is the reference Python library. Its EMQ divides the posteriors by the training
  prevalence and averages them. BayesianMAPLS samples the log-likelihood below under a
  Dirichlet prior with NumPyro NUTS:

  $$
  \sum_i\operatorname{logsumexp}_k\big(\log p(k\mid x_i)+\log\pi_{\text{test},k}-\log\pi_{\text{train},k}\big).
  $$

  BayesianKDEy uses class-conditional kernel density estimates of the classifier outputs as $L_{ik}$.
- QuaPy's confidence-aware methods return an estimate and a region from `predict_conf`.
  `calibrate_temperature` chooses a likelihood temperature against validation coverage.
  QuaPy takes features and a scikit-learn classifier, not a likelihood matrix. It adds no
  smoothness or selection.
- **mlquantify** (Python; BSD-3) implements 33 methods with bootstrap confidence regions.
- **qunfold** ([mirkobunse/qunfold](https://github.com/mirkobunse/qunfold)) joins
  quantification and unfolding {cite:p}`bunse2022`. It includes a likelihood maximiser
  with ordinal smoothness penalties, but reports no uncertainty.
- **labelshift** is the research code of BayesianCC. **abstention** provides calibration
  and label-shift estimators.
- **epiR** and **prevalence** (R) implement Rogan–Gladen and Bayesian true-prevalence
  estimates for two classes.

## How binference relates

**Shared.** The likelihood of EMQ, BayesianMAPLS and BayesianKDEy, the Dirichlet prior,
and NUTS on the simplex. The confusion-matrix model of BayesianCC, with the matrix
inferred from labelled counts.

**Added.** A likelihood matrix as direct input, however it was produced. Selection
effects. Smoothness priors for ordered classes. Simulation-based calibration of the
whole analysis.

**Not done.** binference does not train or calibrate classifiers and has no likelihood
temperature. It does not estimate the effective training prevalence; you must give it.
It returns Bayesian draws, not bootstrap confidence regions.

## Pitfalls known in these fields

- **Calibration.** The EM quantifier assumes calibrated posteriors, "which is not true
  of modern neural networks" {cite:p}`alexandari2020`. Maximum likelihood with
  bias-corrected calibration outperforms BBSE and RLLS. MLLS is consistent only with
  calibration and an invertible confusion matrix {cite:p}`garg2020`.
- **Bootstrap regions.** Standard bootstrap methods "cannot provide valid coverage
  guarantees under prior probability shift". They "merely quantify the internal
  variability of the quantifier" {cite:p}`moreo2025`.
- **Classify and count** is biased under prior shift. The {doc}`ecology` page gives the
  fisheries evidence.
- **Inversion leaves the simplex.** Plug-in corrections such as Rogan–Gladen can return
  negative prevalences. epiR recommends a Bayesian approach whenever the apparent
  prevalence is below one minus the specificity. A posterior on the simplex cannot leave
  it.
- **Fields unaware of each other.** A comparison of ordinal quantifiers found methods
  "proposed by authors from very different research fields, such as data mining and
  astrophysics, who were unaware of each others' developments" {cite:p}`bunse2024`.
