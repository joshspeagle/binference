# Empirical Bayes deconvolution and g-modelling

In statistics the same problem is called *deconvolution*, *g-modelling* or estimation of
a mixing distribution. Many parameters $\theta_i$ are drawn from an unknown prior $g$;
each is observed through noisy data $X_i\sim p(\cdot\mid\theta_i)$; the task is to
estimate $g$, and with it each $\theta_i$'s posterior (empirical Bayes). Statistics owns
the best solvers and the only interval estimates for $g$, almost all of them in R or
Julia.

## The problem in statistical terms

The nonparametric maximum-likelihood estimator (NPMLE) of $g$ (Kiefer & Wolfowitz 1956)
maximises $\sum_i\log\int p(X_i\mid\theta)\,dg(\theta)$ over all distributions. On a
fixed grid of support points $\theta_1,\dots,\theta_K$ this becomes

$$
\max_{w\in\Delta^{K-1}}\ \sum_{i=1}^{N}\log\sum_{k=1}^{K}L_{ik}w_k,\qquad L_{ik}=p(X_i\mid\theta_k),
$$

which is "a (finite-dimensional) convex optimization problem" for any per-object
likelihoods (Koenker & Mizera 2014, as summarised by Soloff, Guntuboyina & Sen 2024).
At the optimum, the conditions for optimality on the simplex restricted to the support
are the EM fixed-point ("self-consistency") equations: $w_k$ is the average over objects
of the posterior probability of point $k$ under $w$ itself. The NPMLE is discrete and
sparse; in one-dimensional Gaussian problems it has $O(\log n)$ atoms with high
probability (Polyanskiy & Wu 2020).

Two smooth alternatives are common. Efron's g-modelling fits $g\propto\exp(Q\alpha)$, a
penalised log-spline exponential family on the grid, and gives delta-method standard
errors (Efron 2016). Kernel-smoothing the NPMLE is the other. *f-modelling* estimates
only the marginal density of the data, which many empirical-Bayes rules need, "avoiding
the difficult deconvolution problems of estimating $g(\cdot)$" (Efron 2019). The classic
application is the normal-means problem with known, possibly different, standard errors
(Jiang & Zhang 2009); Soloff et al. extend the NPMLE to multivariate Gaussian errors
with per-object covariances, with a Gaia colour–magnitude diagram of about 1.4 million
stars as an application.

## In binference's terms

The {doc}`empirical Bayes tutorial <../tutorials/09_empirical_bayes>` works through
the NPMLE, the posterior over $g$ and the shrinkage estimates on simulated data.

The likelihood matrix on a grid is binference's $L_{ik}$, and $w$ is $\rho$
([theory](../theory.md#the-population-model)). The grid maps onto
{class}`~binference.Bins` as narrow bins around the support points, with bin-averaged
likelihoods from `Likelihoods.from_gaussian` or `from_density`, or, when only the values
at the points matter, onto `Bins.categories` with `Likelihoods(L, bins)`; the smoothness
priors need ordered bins.

- **The NPMLE.** With a flat Dirichlet prior the posterior mode is the maximum-likelihood
  estimate, so `binference.infer(lk, method="map")` computes the NPMLE on the grid. For
  Dirichlet priors it uses mix-SQP (Kim et al. 2020); `method="em"` runs SQUAREM-
  accelerated EM. The result carries its KKT gap, the largest violation of the
  optimality conditions, which is zero at the optimum (`binference.diagnostics.kkt_gap`).
- **The Dirichlet-penalised mixture** of ashr, $\sum_iw_i\log(\sum_k\pi_kL_{ik})+
  \sum_k(\alpha_k-1)\log\pi_k$, is the MAP of the same model with a Dirichlet prior.
- **Smooth $g$.** `LogisticRandomWalk` and `LogisticGP` put a Gaussian prior on
  $\log\rho$ and infer its smoothness, the Bayesian counterpart of Efron's penalised
  log-spline ({doc}`../user-guide/priors`).
- **Empirical-Bayes posteriors** of each object are `result.assignment_probabilities`,
  which average over the posterior draws of $\rho$; the `_best` variants, such as
  `assignment_probabilities_best`, use the point estimate alone, as plug-in empirical
  Bayes does ({doc}`../user-guide/results`).
- **Efron's ratio of "artificial to genuine information"** is
  `result.prior_information_ratio()`.
- **Bootstrap.** The `Bootstrap` engine resamples objects (`scheme="nonparametric"`) or
  reweights them with Dirichlet weights (`scheme="bayesian"`) and refits the MAP.

## Methods and software

| package | language | generic $L_{ik}$ input | engine | uncertainty on $g$ |
|---|---|---|---|---|
| [REBayes](https://cran.r-project.org/package=REBayes) | R | yes, `KWDual(A, d, w)` | interior point via the commercial MOSEK solver | none |
| [mixsqp](https://cran.r-project.org/package=mixsqp) | R/C++ | yes, `mixsqp(L, w)`, log input allowed | SQP with active sets and a low-rank $L$ | none |
| [ashr](https://cran.r-project.org/package=ashr) | R | yes, with a Dirichlet prior | mix-SQP, EM, interior point, variational EM (`mixVBEM`) | variational Dirichlet posterior |
| [deconvolveR](https://github.com/bnaras/deconvolveR) | R | partly | penalised log-spline $g$ | delta-method standard errors and bias |
| [ebnm](https://cran.r-project.org/package=ebnm) | R | no; Gaussian errors only | wraps mixsqp, ashr, REBayes and deconvolveR | none for $g$ |
| [Empirikos.jl](https://github.com/nignatiadis/Empirikos.jl) | Julia | through likelihood types | conic solvers (MOSEK or open-source Hypatia) | F-localisation and AMARI intervals |
| [npeb](https://github.com/jake-soloff/NPEB) | Python | yes, through module functions | cvxpy or MOSEK; alpha, no tests | none |
| [mixsqpx](https://pypi.org/project/mixsqpx/) | Python/JAX | yes | mix-SQP in JAX; alpha, Python 3.12 only | none |

- **mix-SQP** (Kim et al. 2020) "achieves at least 100-fold reduction in runtime" over
  the MOSEK interior-point solver at about $10^6$ observations and $10^3$ components,
  because many columns of $L$ are similar and $L$ is numerically low-rank in one
  dimension. After 1000 iterations on a $20{,}000\times800$ problem, EM and projected
  gradient were still at least 0.35 log-likelihood units short of the mix-SQP solution.
- **The dual augmented Lagrangian method** (Zhang et al. 2024) exploits the sparsity of
  the solution and handled the two-dimensional Gaia TGAS diagram on a $100\times100$
  grid in about 472 minutes on a 32-core, 256 GB workstation, where the dense $L$ alone
  takes about 109 GB. In two dimensions $L$ is not low-rank, and mix-SQP "can only
  handle this problem with small m".
- **ashr's `mixVBEM`** is the only EB routine surveyed that treats the mixture weights as
  random, by mean-field variational Bayes under a Dirichlet prior. Its MAP routines
  encode the prior as pseudo-observations, which handles $\alpha_k\ge1$ only.
- **Interval estimates for $g$.** deconvolveR's delta-method errors assume the smooth
  model is right; Empirikos.jl's F-localisation intervals have simultaneous and its
  AMARI intervals pointwise frequentist coverage (Ignatiadis & Wager 2022).
- **Python** has only npeb and mixsqpx; researchers fall back on R and MOSEK through
  rpy2. ebnm's authors name the cost of the fragmentation: "each existing package has a
  different interface and outputs, which hinders comparisons" (Willwerscheid et al.
  2025).
- **Size reduction.** Empirikos.jl fits multiplicity-weighted summaries, REBayes can bin
  the data, and npeb can drop kernel entries below a log-probability threshold.

## How binference relates

**Shared.** binference solves the discretised NPMLE problem with the mix-SQP algorithm,
as mixsqp does, and accepts the same input: any $N\times K$ likelihood matrix, in log space if
wanted, with row scaling irrelevant and per-object weights.

**Added.** A posterior over $g$ rather than a point estimate, sampled exactly and
checked with simulation-based calibration; priors with $\alpha_k<1$, which ashr's
pseudo-observation encoding cannot represent; interim-prior correction and posterior-sample inputs, which
none of the surveyed EB packages supports; selection effects; and sparse storage and
`lk.compress()` for duplicated rows ({doc}`../user-guide/performance`).

**Not done.** binference gives Bayesian credible intervals, not frequentist intervals
for functionals of $g$ such as Empirikos.jl's; it does not move support points, as the
constrained-Newton and dual-ALM methods do; it uses neither MOSEK nor a convex-programming
layer; and it has no f-modelling estimators.

## Pitfalls known in empirical Bayes

- **Discrete NPMLEs give degenerate intervals.** "The NPMLE is a discrete distribution
  ... interval estimates can be unsatisfactory"; in ebnm's baseball example each
  credible interval endpoint "is constrained to lie at one of the support points" and one
  player gets "a degenerate interval" (Willwerscheid et al. 2025). On a fine grid most
  fitted fractions are exactly zero.
- **Plug-in empirical Bayes understates uncertainty.** Nominal 90% intervals that ignore
  the uncertainty in $\hat g$ "tended to be too small", and the NPMLE "tended to have much
  poorer coverage" (Willwerscheid et al. 2025).
- **Plug-in smoothing intervals under-cover.** Efron's log-spline intervals were the
  shortest in Ignatiadis & Wager's simulations but "do not achieve nominal coverage,
  while all other methods do", because they ignore the smoothing bias.
- **Variational intervals are too narrow,** "especially if the components of the mixture
  model are not well separated" (Wang & Titterington 2005), which is the case of broad,
  overlapping likelihoods.
- **EM is slow near the optimum and never revives a zeroed weight.** A fixed point with
  $w_k=0$ need not be optimal, so self-consistency is necessary but not sufficient; the
  KKT gap is the check ([theory](../theory.md#point-estimates-em-is-iterated-stacking)).
- **Deconvolution is slow.** Nonparametric rates are "very slow" (Efron 2016), so
  individual bin fractions are often prior-dominated and strongly anti-correlated;
  well-identified functionals (cumulative fractions, coarse sums, moments) are the
  quantities to report.
- **Numerics.** Likelihoods underflow, so solvers take log input and normalise rows; no
  column of $L$ may be all zeros (mixsqp); the NPMLE can be non-unique in more than one
  dimension and for binomial data with few trials.
- **Commercial solvers** (MOSEK, for REBayes) are a recurring obstacle; ebnm
  re-implemented the NPMLE with mixsqp "to provide a fully open-source toolkit".

## References

- Efron (2016), *Biometrika* 103, 1 ([RePEc](https://ideas.repec.org/a/oup/biomet/v103y2016i1p1-20..html)): empirical Bayes deconvolution estimates.
- Efron (2019), *Statistical Science* 34, 177 ([doi:10.1214/18-STS674](https://projecteuclid.org/journals/statistical-science/volume-34/issue-2/Bayes-Oracle-Bayes-and-Empirical-Bayes/10.1214/18-STS674.pdf)): Bayes, oracle Bayes and empirical Bayes.
- Ignatiadis & Wager (2022), *JASA* 117, 1149 ([arXiv:1902.02774](https://arxiv.org/abs/1902.02774)): confidence intervals for nonparametric empirical Bayes.
- Jiang & Zhang (2009), *Annals of Statistics* 37, 1647: general maximum-likelihood empirical Bayes for normal means.
- Kiefer & Wolfowitz (1956), *Annals of Mathematical Statistics* 27, 887: the NPMLE.
- Kim, Carbonetto, Stephens & Anitescu (2020), *JCGS* 29, 261 ([arXiv:1806.01412](https://arxiv.org/abs/1806.01412)): mix-SQP.
- Koenker & Gu (2017), *JSS* 82(8): REBayes.
- Koenker & Mizera (2014), *JASA* 109, 674: convex optimisation for empirical Bayes.
- Narasimhan & Efron (2020), *JSS* 94(11): deconvolveR.
- Polyanskiy & Wu (2020) ([arXiv:2008.08244](https://arxiv.org/abs/2008.08244)): self-regularisation of the NPMLE.
- Soloff, Guntuboyina & Sen (2024), *JRSSB* ([arXiv:2109.03466](https://arxiv.org/abs/2109.03466)): multivariate heteroscedastic NPMLE.
- Stephens (2017), *Biostatistics* 18, 275: ashr.
- Wang & Titterington (2005), *AISTATS* ([PMLR R5](https://proceedings.mlr.press/r5/wang05a.html)): variational interval estimates.
- Willwerscheid, Carbonetto & Stephens (2025), *JSS* 114(3) ([arXiv:2110.00152](https://arxiv.org/abs/2110.00152)): ebnm.
- Zhang, Cui, Sen & Toh (2024), *JMLR* 25(8) ([arXiv:2208.07514](https://arxiv.org/abs/2208.07514)): dual ALM for the NPMLE.
