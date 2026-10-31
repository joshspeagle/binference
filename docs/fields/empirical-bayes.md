# Empirical Bayes deconvolution and g-modelling

What does statistics call binference's problem? It calls it deconvolution, g-modelling
or estimation of a mixing distribution. An unknown prior $g$ generates many true values,
and noisy data measure each one. The task is to estimate $g$, and with it each true
value's posterior; this is empirical Bayes. Most of the software is in R or Julia.

## The problem in statistical terms

The field writes $\theta_i$ for the true value and $X_i$ for the data. This page writes
$z_i$ and $d_i$, as the rest of the documentation does. It writes $w$ for the mixture
proportions, binference's $\rho$.

The nonparametric maximum-likelihood estimator (NPMLE) of $g$ {cite:p}`kiefer1956`
maximises the log-likelihood over all distributions. On a fixed grid of support points
$z_1,\dots,z_K$ this becomes

$$
\max_{w\in\Delta^{K-1}}\ \sum_{i=1}^{N}\log\sum_{k=1}^{K}L_{ik}w_k,\qquad L_{ik}=p(d_i\mid z_k).
$$

It is "a (finite-dimensional) convex optimization problem" for any likelihoods
({cite:alp}`koenker2014`, as summarised by {cite:alp}`soloff2025`). Its optimality
conditions are the EM fixed-point ("self-consistency") equations. The NPMLE is discrete
and sparse. In one-dimensional Gaussian problems it has $O(\log N)$ atoms with high
probability {cite:p}`polyanskiy2020`.

Efron's g-modelling fits a smooth, penalised log-spline $g$ on the grid, with
delta-method standard errors {cite:p}`efron2016`. f-modelling estimates only the
marginal density of the data, "avoiding the difficult deconvolution problems of
estimating $g(\cdot)$" {cite:p}`efron2019`.

The classic application is the normal-means problem with known, unequal standard errors
{cite:p}`jiang2009`. {cite:t}`soloff2025` extend the NPMLE to multivariate Gaussian
errors with per-object covariances. Their application is a Gaia colour–magnitude diagram
of about 1.4 million stars.

## In binference's terms

| empirical-Bayes term | binference |
|---|---|
| prior $g$, mixture proportions $w$ | population fractions $\rho$ |
| grid of support points | narrow {class}`~binference.Bins` with `Likelihoods.from_gaussian` or `from_density`; or `Bins.categories` with `Likelihoods.from_table(L, bins, kind="likelihood")` |
| NPMLE | `infer(lk, method="map")` with the default flat Dirichlet |
| self-consistency check | the {term}`KKT gap`, `PointEstimate.kkt_gap` |
| ashr's Dirichlet-penalised mixture | the MAP with a `Dirichlet` prior |
| Efron's penalised log-spline $g$ | `LogisticRandomWalk`, `LogisticGP` |
| empirical-Bayes posterior | `result.assignment_probabilities()` |
| plug-in empirical-Bayes posterior | `result.assignment_probabilities_best()` |
| Efron's ratio of "artificial to genuine information" | `result.prior_information_ratio()` |
| nonparametric or Bayesian bootstrap | `Bootstrap(scheme="nonparametric")`, `scheme="bayesian"` |

The smoothness priors need ordered bins. The
[point estimates guide](../guides/engines.md#point-estimates) says which solver
`method="map"` uses. The {doc}`empirical Bayes tutorial <../tutorials/empirical-bayes>`
works through the NPMLE, the posterior over $g$ and the shrinkage estimates.

## Methods and software

The [comparison table](software.md#comparison) gives each tool's language, licence and
status. REBayes, mixsqp, ashr, npeb and mixsqpx accept a generic likelihood matrix.
deconvolveR accepts one in part; ebnm takes Gaussian errors only.

- **mix-SQP** {cite:p}`kim2020` "achieves at least 100-fold reduction in runtime" over
  the MOSEK interior-point solver. The test used about $10^6$ objects and $10^3$
  components. mix-SQP exploits a numerically low-rank $L$ in one dimension. The mixsqp
  package implements it.
- **REBayes** {cite:p}`koenker2017` solves the NPMLE by an interior-point method through
  the commercial MOSEK solver.
- **The dual augmented Lagrangian method** {cite:p}`zhang2024` exploits the sparsity of
  the solution. It handled the Gaia TGAS diagram on a $100\times100$ grid in about 472
  minutes on a 32-core workstation. The dense $L$ alone takes about 109 GB. In two
  dimensions mix-SQP "can only handle this problem with small m".
- **ashr** {cite:p}`stephens2017` offers mix-SQP, EM and an interior-point method. Its
  `mixVBEM` is the only routine surveyed that treats the proportions as random, by
  mean-field variational Bayes under a Dirichlet prior. ashr's MAP routines encode the
  prior as pseudo-observations. That encoding handles only $\alpha_k\ge1$.
- **deconvolveR** {cite:p}`narasimhan2020` fits Efron's log-spline $g$ with delta-method
  errors. These errors assume the smooth model is right.
- **Empirikos.jl** gives F-localisation intervals with simultaneous coverage, and AMARI
  intervals with pointwise coverage {cite:p}`ignatiadis2022`.
- **ebnm** wraps mixsqp, ashr, REBayes and deconvolveR. Its authors name the cost of many
  packages: "each existing package has a different interface and outputs, which hinders
  comparisons" {cite:p}`willwerscheid2025`.

## How binference relates

binference solves the discretised NPMLE with mix-SQP, as mixsqp does. It accepts the
same input: any $N\times K$ likelihood matrix, in log space if wanted, with per-object
weights.

For empirical Bayes, binference adds:

- a posterior over $g$, not only a point estimate;
- priors with $\alpha_k<1$, which pseudo-observations cannot represent;
- sparse storage and `lk.compress()` for duplicated rows
  ({doc}`performance guide <../guides/performance>`).

Not done:

- binference gives Bayesian credible intervals, not frequentist intervals for
  functionals of $g$.
- It does not move support points.
- It uses neither MOSEK nor a convex-programming layer.
- It has no f-modelling estimators.

## Pitfalls known in empirical Bayes

- **Discrete NPMLEs give degenerate intervals.** "The NPMLE is a discrete distribution
  ... interval estimates can be unsatisfactory" {cite:p}`willwerscheid2025`. In ebnm's
  baseball example each interval endpoint "is constrained to lie at one of the support
  points". One player gets "a degenerate interval".
- **Plug-in empirical Bayes understates uncertainty.** Nominal 90% intervals that ignore
  the uncertainty in $\hat g$ "tended to be too small". The NPMLE "tended to have much
  poorer coverage" {cite:p}`willwerscheid2025`.
- **Plug-in smoothing intervals under-cover.** Efron's log-spline intervals were the
  shortest in {cite:author}`ignatiadis2022`'s simulations. But they "do not achieve
  nominal coverage, while all other methods do", because they ignore smoothing bias.
- **Variational intervals are too narrow,** "especially if the components of the mixture
  model are not well separated" {cite:p}`wang2005`. Broad, overlapping likelihoods give
  exactly that case.
- **EM cannot move a fraction away from zero.** A fixed point with $w_k=0$ need not be
  optimal. Self-consistency is therefore necessary but not sufficient; the KKT gap is
  the check ([theory](../how-it-works/point-estimates.md#em-is-iterated-stacking)).
- **Deconvolution converges slowly.** Its nonparametric convergence rates are "very
  slow" {cite:p}`efron2016`. Single bin fractions are then often prior-dominated and
  anti-correlated. Report well-identified functionals instead: cumulative fractions,
  coarse sums, moments.
- **Numerics.** Likelihoods underflow, so solvers take log input and normalise rows.
- **Commercial solvers** are a recurring obstacle. ebnm re-implemented the NPMLE with
  mixsqp "to provide a fully open-source toolkit".
