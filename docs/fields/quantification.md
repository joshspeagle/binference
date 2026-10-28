# Quantification and prevalence estimation

What does machine learning call binference's problem? It calls it quantification, or
estimation under {term}`label shift`. You train a classifier on labelled data and apply
it to an unlabelled set whose class proportions differ. You want those proportions, not
a label for each item. Epidemiology solves the same problem for an imperfect test.
Ecology and genetics solve it for genotypes and reads; see {doc}`ecology`.

## The problem in these fields' terms

**Quantification.** A classifier gives each test item, with data $d_i$, a posterior
$p(k\mid d_i)$. It learned this posterior under the training prevalence
$\pi_\text{train}$. Under label shift the class-conditional distributions $p(d\mid k)$
are the same in training and test. Only the prevalence changes. The methods fall into
four groups:

- **Classify and count** (CC) and its probabilistic version (PCC) average hard or soft
  predictions.
- **Adjusted** methods (ACC, BBSE) correct the counts with a confusion matrix estimated
  on labelled data {cite:p}`lipton2018`.
- **Likelihood** methods iterate the prevalences and the posteriors corrected for them.
  The field calls the method EMQ, SLD {cite:p}`saerens2002` or MLLS.
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
| training prevalence | the {term}`interim prior` |
| confusion matrix | a {class}`~binference.Response` |
| EMQ, SLD, MLLS | the MAP from posteriors and the training prevalence |
| BayesianMAPLS | the posterior from posteriors and the training prevalence |
| BayesianCC | the posterior from scores and a labelled set |
| PCC | {term}`stacking`, `lk.stack()` |
| ordinal penalties | `LogisticRandomWalk` on ordered bins |
| the model behind Rogan–Gladen | a two-class `Response` with known sensitivity and specificity |

A classifier reaches binference by one of two routes:

- **posteriors and the training prevalence**, which needs classifier calibration;
- **scores and a labelled set**, which needs only label shift.

The [inputs guide](../guides/inputs/binned.md) gives the
code for each and the conditions under which each is exact. The
{doc}`categories tutorial <../tutorials/06_categories_unfolding>` runs both routes on a
mock classifier under prior shift.

## Methods and software

The [comparison table](software.md#comparison) gives each tool's language, licence and
status.

- **QuaPy** is the reference Python library. Its EMQ divides the posteriors by the
  training prevalence and averages them. BayesianMAPLS samples this log-likelihood under
  a Dirichlet prior with NumPyro NUTS:

  $$
  \sum_i\operatorname{logsumexp}_k\big(\log p(k\mid d_i)+\log\pi_{\text{test},k}-\log\pi_{\text{train},k}\big).
  $$

  BayesianKDEy uses kernel density estimates of each class's classifier outputs as
  $L_{ik}$.
- QuaPy's uncertainty-aware methods return an estimate and a region from
  `predict_conf`. Its `calibrate_temperature` applies temperature scaling, chosen
  against validation coverage. QuaPy takes features and a scikit-learn classifier, not
  a likelihood matrix.
- **mlquantify** implements 33 methods with bootstrap confidence regions.
- **qunfold** joins quantification and unfolding {cite:p}`bunse2022`. It includes a
  likelihood maximiser with ordinal smoothness penalties, but reports no uncertainty.
- **labelshift** is the research code of BayesianCC. **abstention** provides classifier
  calibration and label-shift estimators.
- **epiR** and **prevalence** (R) implement Rogan–Gladen and Bayesian true-prevalence
  estimates for two classes.

## How binference relates

binference shares the likelihood of EMQ, BayesianMAPLS and BayesianKDEy, the Dirichlet
prior, and NUTS on the simplex. It shares BayesianCC's confusion-matrix model, with the
matrix inferred from labelled counts.

For quantification, binference adds a likelihood matrix as direct input, however it was
produced. It also adds smoothness priors for ordered classes.

Not done:

- binference does not train classifiers or apply classifier calibration.
- It has no likelihood temperature.
- It does not estimate the effective training prevalence; you must give it.
- It returns Bayesian draws, not bootstrap confidence regions.

## Pitfalls known in these fields

- **Classifier calibration.** The EM quantifier assumes calibrated posteriors, "which is
  not true of modern neural networks" {cite:p}`alexandari2020`. Maximum likelihood with
  bias-corrected calibration outperforms BBSE and RLLS. MLLS is consistent only with
  calibrated posteriors and an invertible confusion matrix {cite:p}`garg2020`.
- **Bootstrap regions.** Standard bootstrap methods "cannot provide valid coverage
  guarantees under prior probability shift". They "merely quantify the internal
  variability of the quantifier" {cite:p}`moreo2025`.
- **Classify and count** is biased under prior shift. The {doc}`ecology` page gives the
  fisheries evidence.
- **Inversion leaves the simplex.** Plug-in corrections such as Rogan–Gladen can return
  negative prevalences. epiR recommends a Bayesian approach whenever the apparent
  prevalence is below one minus the specificity. A posterior on the simplex cannot leave
  it.
- **Fields unaware of each other.** Ordinal quantifiers came from "very different
  research fields, such as data mining and astrophysics". Their authors "were unaware of
  each others' developments" {cite:p}`bunse2024`.
