# Empirical Bayes deconvolution and g-modelling

What does statistics call binference's problem? It calls it deconvolution, g-modelling
or estimation of a mixing distribution. Many parameters are drawn from an unknown prior
$g$, and each is observed through noisy data. The task is to estimate $g$, and with it
each parameter's posterior; this is empirical Bayes. Statistics owns the best solvers
and the only interval estimates for $g$. Almost all of them are in R or Julia.

## The problem in statistical terms

The field writes $\theta_i$ for the true value and $X_i$ for the data. This page writes
$z_i$ and $d_i$, as the rest of the documentation does, and $w$ for $\rho$.

The nonparametric maximum-likelihood estimator (NPMLE) of $g$ {cite:p}`kiefer1956`
maximises the log-likelihood over all distributions. On a fixed grid of support points
$z_1,\dots,z_K$ this becomes

$$
\max_{w\in\Delta^{K-1}}\ \sum_{i=1}^{N}\log\sum_{k=1}^{K}L_{ik}w_k,\qquad L_{ik}=p(d_i\mid z_k).
$$

It is "a (finite-dimensional) convex optimization problem" for any likelihoods
({cite:alp}`koenker2014`, as summarised by {cite:alp}`soloff2025`). Its optimality
conditions are the EM fixed-point ("self-consistency") equations. The NPMLE is discrete
and sparse. In one-dimensional Gaussian problems it has $O(\log n)$ atoms with high
probability {cite:p}`polyanskiy2020`.

Efron's g-modelling fits a smooth, penalised log-spline $g$ on the grid, with
delta-method standard errors {cite:p}`efron2016`. f-modelling estimates only the marginal density of
the data, "avoiding the difficult deconvolution problems of estimating $g(\cdot)$"
{cite:p}`efron2019`.

The classic application is the normal-means problem with known, unequal standard errors
{cite:p}`jiang2009`. {cite:t}`soloff2025` extend the NPMLE to multivariate Gaussian
errors with per-object covariances. Their application is a Gaia colour–magnitude diagram
of about 1.4 million stars.

## In binference's terms

| empirical-Bayes term | binference |
|---|---|
| prior $g$, weights $w$ | population fractions $\rho$ |
| grid of support points | narrow {class}`~binference.Bins` with `Likelihoods.from_gaussian` or `from_density`; or `Bins.categories` with `Likelihoods(L, bins)` |
| NPMLE | `infer(lk, method="map")` with the default flat Dirichlet |
| self-consistency check | the {term}`KKT gap`, `binference.diagnostics.kkt_gap` |
| ashr's Dirichlet-penalised mixture | the MAP with a `Dirichlet` prior |
| Efron's penalised log-spline $g$ | `LogisticRandomWalk`, `LogisticGP` |
| empirical-Bayes posterior | `result.assignment_probabilities()` |
| plug-in empirical-Bayes posterior | `result.assignment_probabilities_best()` |
| Efron's ratio of "artificial to genuine information" | `result.prior_information_ratio()` |
| nonparametric or Bayesian bootstrap | `Bootstrap(scheme="nonparametric")`, `scheme="bayesian"` |

The smoothness priors need ordered bins. The
[point estimates guide](../guides/engines.md#point-estimates) says which solver
`method="map"` uses.

The {doc}`empirical Bayes tutorial <../tutorials/09_empirical_bayes>` works through the
NPMLE, the posterior over $g$ and the shrinkage estimates.

## Methods and software

| package | language | generic $L_{ik}$ input | engine | uncertainty on $g$ |
|---|---|---|---|---|
| [REBayes](https://cran.r-project.org/package=REBayes) {cite:p}`koenker2017` | R | yes, `KWDual(A, d, w)` | interior point via the commercial MOSEK solver | none |
| [mixsqp](https://cran.r-project.org/package=mixsqp) | R/C++ | yes, `mixsqp(L, w)`, log input allowed | SQP with active sets and a low-rank $L$ | none |
| [ashr](https://cran.r-project.org/package=ashr) {cite:p}`stephens2017` | R | yes, with a Dirichlet prior | mix-SQP, EM, interior point, variational EM (`mixVBEM`) | variational Dirichlet posterior |
| [deconvolveR](https://github.com/bnaras/deconvolveR) {cite:p}`narasimhan2020` | R | partly | penalised log-spline $g$ | delta-method standard errors and bias |
| [ebnm](https://cran.r-project.org/package=ebnm) | R | no; Gaussian errors only | wraps mixsqp, ashr, REBayes and deconvolveR | none for $g$ |
| [Empirikos.jl](https://github.com/nignatiadis/Empirikos.jl) | Julia | through likelihood types | conic solvers (MOSEK or open-source Hypatia) | F-localisation and AMARI intervals |
| [npeb](https://github.com/jake-soloff/NPEB) | Python | yes, through module functions | cvxpy or MOSEK; alpha, no tests | none |
| [mixsqpx](https://pypi.org/project/mixsqpx/) | Python/JAX | yes | mix-SQP in JAX; alpha, Python 3.12 only | none |

- **mix-SQP** {cite:p}`kim2020` "achieves at least 100-fold reduction in runtime" over
  the MOSEK interior-point solver, at about $10^6$ objects and $10^3$ components. It
  exploits a numerically low-rank $L$ in one dimension.
- **The dual augmented Lagrangian method** {cite:p}`zhang2024` exploits the sparsity of
  the solution. It handled the Gaia TGAS diagram on a $100\times100$ grid in about 472
  minutes on a 32-core workstation. The dense $L$ alone takes about 109 GB. In two
  dimensions mix-SQP "can only handle this problem with small m".
- **ashr's `mixVBEM`** is the only routine surveyed that treats the weights as random.
  It uses mean-field variational Bayes under a Dirichlet prior. ashr's MAP routines
  encode the prior as pseudo-observations, which handles only $\alpha_k\ge1$.
- **Interval estimates for $g$.** deconvolveR's delta-method errors assume the smooth
  model is right. Empirikos.jl's F-localisation intervals have simultaneous coverage,
  and its AMARI intervals pointwise coverage {cite:p}`ignatiadis2022`.
- **Python** has only npeb and mixsqpx. ebnm's authors name the cost: "each existing package has a different interface
  and outputs, which hinders comparisons" {cite:p}`willwerscheid2025`.

## How binference relates

**Shared.** binference solves the discretised NPMLE with mix-SQP, as mixsqp does. It
accepts the same input: any $N\times K$ likelihood matrix, in log space if wanted, with
per-object weights.

**Added.** A posterior over $g$, sampled exactly and checked by simulation-based
calibration. Priors with $\alpha_k<1$, which pseudo-observations cannot represent.
Interim-prior correction and posterior-sample inputs, which no surveyed EB package
supports. Selection effects. Sparse storage and `lk.compress()` for duplicated rows
({doc}`../guides/performance`).

**Not done.** binference gives Bayesian credible intervals, not frequentist intervals
for functionals of $g$. It does not move support points. It uses neither MOSEK nor a
convex-programming layer. It has no f-modelling estimators.

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
  model are not well separated" {cite:p}`wang2005`. Broad, overlapping likelihoods are
  that case.
- **EM never revives a zeroed weight.** A fixed point with $w_k=0$ need not be optimal.
  Self-consistency is therefore necessary but not sufficient; the KKT gap is the check
  ([theory](../how-it-works/theory.md#point-estimates-em-is-iterated-stacking)).
- **Deconvolution is slow.** Nonparametric rates are "very slow" {cite:p}`efron2016`.
  Single bin fractions are then often prior-dominated and anti-correlated. Report
  well-identified functionals instead: cumulative fractions, coarse sums, moments.
- **Numerics.** Likelihoods underflow, so solvers take log input and normalise rows.
- **Commercial solvers** are a recurring obstacle. ebnm re-implemented the NPMLE with
  mixsqp "to provide a fully open-source toolkit".
