# Priors on the simplex and inference for mixture weights

What does the statistics literature know about priors and algorithms for fractions on
the simplex? The fractions $\rho$ lie on the {term}`simplex`. The posterior is
$p(\rho\mid D)\propto p(\rho)\prod_i\sum_kL_{ik}\rho_k$. This page lists the known
priors and algorithms, and which of them binference provides. The {doc}`priors guide
<../guides/priors>` and the {doc}`engines guide <../guides/engines>` say how to choose.

## Priors

Priors on the simplex fall into two families.

**The conjugate family** contains the Dirichlet, the generalised Dirichlet
{cite:p}`connor1969`, the Dirichlet tree {cite:p}`minka1999` and finite Pólya trees
{cite:p}`lavine1992`. Given the counts per bin, each has a closed-form posterior, so
Gibbs sampling and EM stay simple.

- **Dirichlet.** All covariances are negative. The means and the total concentration
  $\alpha_0=\sum_k\alpha_k$ fix them:
  $\mathrm{Cov}(\rho_j,\rho_k)=-\alpha_j\alpha_k/(\alpha_0^2(\alpha_0+1))$.
- **Generalised Dirichlet.** It breaks a stick with independent Beta variables. It has
  $2(K-1)$ parameters and "a more general covariance structure". Positive correlations
  are possible, and the bin order matters. {cite:t}`leistedt2016` noted that it could
  encode correlations between redshift bins but did "not consider this case".
- **Dirichlet tree.** A Dirichlet at every internal node of a tree whose leaves are the
  bins. It "can give an independent variance to each $p_k$". Leaves in the same subtree
  are positively correlated {cite:p}`minka1999`.
- **Dirichlet process.** Restricted to fixed bins, it is a Dirichlet with $\alpha_k$
  proportional to the base measure of bin $k$ {cite:p}`ferguson1973`. It adds nothing on
  fixed bins.

**Gaussian priors on log-ratios** include the logistic-normal {cite:p}`aitchison1980`,
random walks and intrinsic Gaussian Markov random fields on $\log\rho$
{cite:p}`rue2005`, and P-splines {cite:p}`eilers1996`. Gaussian processes on bins give
the logistic Gaussian process {cite:p}`leonard1978,lenk1988`. Hilbert-space
approximations speed it up {cite:p}`riutortmayol2023`. These priors express smoothness
and arbitrary correlations. They are not conjugate, so they need gradient-based samplers
in unconstrained coordinates. Pólya–gamma stick-breaking augmentation allows Gibbs
sampling {cite:p}`linderman2015`.

**Overdispersion.** The Dirichlet–multinomial {cite:p}`mosimann1962` and the three-step
Dirichlet of redshift-calibration pipelines {cite:p}`sanchez2020` add variance beyond the
multinomial.

| prior | binference |
|---|---|
| Dirichlet; $\alpha=1/K$; $\alpha$ proportional to a base measure; reference counts | `Dirichlet(alpha)`, `Dirichlet.reference()`, `Dirichlet.from_base_measure(...)`, `Dirichlet.from_counts(counts, strength)` |
| generalised Dirichlet | `GeneralizedDirichlet(a, b)` |
| Dirichlet tree | `DirichletTree(tree, alpha)` |
| Pólya tree | `PolyaTree(c, power)` |
| logistic-normal | `LogisticNormal(mean, cov)` |
| random walk or ICAR on $\log\rho$ | `LogisticRandomWalk(order)`, per axis on grids |
| logistic Gaussian process | `LogisticGP(kernel)`, in a Hilbert-space basis |
| independent priors on rates | `GammaRates(shape, rate)` |
| Dirichlet–multinomial or 3sDir overdispersion | not provided |
| Pólya–gamma augmentation, Dirichlet-process mixtures | not provided |

No major probabilistic programming library ships a generalised Dirichlet, a Dirichlet
tree or a Pólya tree.

## Inference algorithms

The log-likelihood $\sum_i\log(L_i\cdot\rho)$ is concave on the simplex. Its value and
gradient cost one pass over $L$
([theory](../how-it-works/model.md#the-population-model)).

- **Gibbs data augmentation** alternates the objects' labels and a conjugate draw of
  $\rho$ {cite:p}`tanner1987,diebolt1994`. It is exact. The fraction of
  {term}`missing information` sets its speed of convergence, as it does EM's
  {cite:p}`tan2023`. In related
  categorical models, data augmentation mixed far worse than a marginal Metropolis
  sampler {cite:p}`johndrow2019`.
- **EM** can be "intolerably slow" {cite:p}`redner1984`. SQUAREM accelerates it
  {cite:p}`varadhan2008`. mix-SQP is much faster
  ([empirical Bayes](empirical-bayes.md#methods-and-software)).
- **Hamiltonian Monte Carlo** in transformed coordinates samples any differentiable
  prior. Libraries disagree on the transform: stick-breaking (NumPyro), centred log-ratio
  (PyMC), additive log-ratio (TensorFlow Probability). Stan, since version 2.37, uses an
  isometric log-ratio, adopted without a published benchmark. For concentrations well
  below one, the tails in log-ratio coordinates become very long.
- **Mean-field variational Bayes** has a closed form here, a Dirichlet with
  "digamma-shrunk" stacking. It factorises the labels from $\rho$ and so ignores the
  coupling of broad likelihoods. Variational inference "generally underestimates the
  variance of the posterior density" {cite:p}`blei2017`.
- **Laplace approximations** fail at the boundary of the simplex.
- **Stochastic-gradient Langevin samplers** have discretisation biases that "dominate
  at the boundary". The stochastic Cox–Ingersoll–Ross process removes them
  {cite:p}`baker2018`.
- **Evidences** come from nested sampling at small $K$, or from Chib's estimator
  {cite:p}`chib1995` on conjugate Gibbs output.

No published benchmark compares these algorithms on this model as the likelihoods
broaden. The table maps them onto binference's engines; the
{doc}`engines guide <../guides/engines>` describes each engine and how `method="auto"`
chooses.

| algorithm | binference |
|---|---|
| HMC | `NUTS`, in binference's own isometric log-ratio coordinates |
| Gibbs data augmentation | `Gibbs` |
| EM, SQUAREM, mix-SQP | `MAP`, which reports a {term}`KKT gap` |
| Laplace | `Laplace`, corrected by Pareto-smoothed importance sampling |
| mean-field VB | `MeanFieldVB`, the algorithm of RAIL's `VarInfStackSummarizer`; flagged approximate |
| stochastic-gradient samplers | not provided |
| evidences | not provided; `result.log_predictive` compares models instead |

## Pitfalls known for simplex priors

- **"Uninformative" Dirichlets are informative for sparse tables.** Take 3 observations
  in 1000 cells. The Jeffreys prior ($\alpha=1/2$) "effectively added 1/2 to the 998 zero
  cells". The two observed cells kept only 0.008 of the posterior mass. The uniform prior
  "is even worse". {cite:t}`berger2015` recommend $\alpha\approx1/K$, which
  `Dirichlet.reference()` sets. See
  [too many bins](../guides/troubleshooting.md#too-many-bins-for-a-flat-dirichlet).
- **A symmetric $\alpha$ is not consistent under merging bins.** See the aggregation
  bias on the {doc}`ecology` page and the {doc}`priors guide <../guides/priors>`.
- **Sparse priors need samplers.** The MAP with $\alpha_k<1$ lies on the boundary of the
  simplex ({doc}`empirical-bayes`).
- **Exact zeros.** At $\alpha=0.01$, 0.06% of NumPy's Dirichlet draws contain an exact
  zero (frankenz audit). Such a draw can make an object's $L_i\cdot\rho$ vanish. EM
  cannot move such a bin away from zero
  ([empirical Bayes](empirical-bayes.md#pitfalls-known-in-empirical-bayes)).
- **Bins without data** are not identified; their posterior is their prior. Their
  {term}`data support` shows it ([the population](../guides/results.md#the-population)).
- **Smoothing scales** are hard to learn ({doc}`gravitational-waves`).
- **A single Dirichlet sees only shot noise** ({doc}`astronomy`).
