# Priors on the simplex and inference for mixture weights

The fractions $\rho$ live on the simplex, and the statistics literature offers both a
range of priors there and a range of algorithms for the posterior
$p(\rho\mid D)\propto p(\rho)\prod_i\sum_kL_{ik}\rho_k$. This page summarises what is
known about each, and which of them binference provides. The {doc}`theory page
<../how-it-works/index>` defines binference's priors and engines; the {doc}`priors
<../guides/priors>` and {doc}`engines <../guides/engines>` pages say how to
choose among them.

## Priors

Priors on the simplex fall into two families.

**The conjugate family**: the Dirichlet, the generalised Dirichlet {cite:p}`connor1969`, the Dirichlet tree {cite:p}`minka1999` and finite Pólya trees {cite:p}`lavine1992`, which are binary
Dirichlet trees. Given the latent counts per bin each has a closed-form posterior, so
Gibbs sampling and EM stay simple. Each adds structure only through an ordering of the
bins or a tree over them:

- **Dirichlet.** All covariances are negative and set by the means and a single
  precision, $\mathrm{Cov}(\rho_j,\rho_k)=-\alpha_j\alpha_k/(\alpha_0^2(\alpha_0+1))$.
- **Generalised Dirichlet.** Independent Beta variables in a stick-breaking cascade,
  with $2(K-1)$ parameters and "a more general covariance structure"; positive
  correlations are possible, and the bin order matters. {cite:t}`leistedt2016` noted that
  it could encode correlations between redshift bins but did "not consider this case".
- **Dirichlet tree.** A Dirichlet at every internal node of a tree whose leaves are the
  bins; it "can give an independent variance to each $p_k$", and leaves in the same
  subtree are positively correlated {cite:p}`minka1999`.
- **A Dirichlet process** restricted to fixed bins is a Dirichlet with
  $\alpha_k=\alpha G_0(B_k)$ {cite:p}`ferguson1973`, so it adds nothing on fixed bins.

**Gaussian priors on log-ratios**: the logistic-normal {cite:p}`aitchison1980`, random
walks and intrinsic Gaussian Markov random fields on $\log\rho$ {cite:p}`rue2005`,
P-splines {cite:p}`eilers1996`, and Gaussian processes, which on bins give the logistic
Gaussian process {cite:p}`leonard1978,lenk1988`, with Hilbert-space approximations for speed
{cite:p}`riutortmayol2023`. These are where smoothness and arbitrary correlations live.
They are not conjugate, so they need gradient-based samplers on unconstrained
coordinates, or Pólya–gamma stick-breaking augmentation when Gibbs sampling is required
{cite:p}`linderman2015`.

**Overdispersion.** The Dirichlet–multinomial {cite:p}`mosimann1962` and the three-step
Dirichlet of redshift calibration {cite:p}`sanchez2020` add variance beyond the
multinomial.

| prior | binference |
|---|---|
| Dirichlet; $\alpha=1/K$; $\alpha$ proportional to bin widths; reference counts | `Dirichlet(alpha)`, `Dirichlet.reference()`, `Dirichlet.from_base_measure(c)`, `Dirichlet.from_counts(m, strength)` |
| generalised Dirichlet | `GeneralizedDirichlet(a, b)` |
| Dirichlet tree | `DirichletTree(tree, alpha)` |
| Pólya tree | `PolyaTree(c)` |
| logistic-normal | `LogisticNormal(mean, cov)` |
| random walk or ICAR on $\log\rho$ | `LogisticRandomWalk(order)`, per axis on grids |
| logistic Gaussian process | `LogisticGP(kernel)`, a Hilbert-space Gaussian process |
| independent priors on rates | `GammaRates(shape, rate)` |
| Dirichlet–multinomial or 3sDir overdispersion | not provided |
| Pólya–gamma augmentation, Dirichlet-process mixtures | not provided |

No major probabilistic programming library ships a generalised Dirichlet, a Dirichlet
tree or a Pólya tree; the only Python generalised Dirichlet found, in efax, has no
sampler. binference implements these three itself.

## Inference algorithms

The log-likelihood $\sum_i\log(L_i\cdot\rho)$ is concave on the simplex, and its value
and gradient cost one pass over $L$ ([theory](../how-it-works/model.md#the-population-model)).

- **Gibbs data augmentation** alternates per-object labels and a conjugate draw of
  $\rho$ {cite:p}`tanner1987,diebolt1994`. It is exact, but its convergence,
  like EM's, is governed by the fraction of missing information {cite:p}`tan2023`: close to zero
  for sharp likelihoods, close to one when each likelihood spreads over many bins. In
  related categorical models data augmentation mixed far worse than a marginal
  Metropolis sampler {cite:p}`johndrow2019`.
- **EM and MAP.** The EM fixed point is "iterated stacking". EM can be "intolerably slow"
  {cite:p}`redner1984`; SQUAREM accelerates it {cite:p}`varadhan2008`, and mix-SQP is
  orders of magnitude faster {cite:p}`kim2020`.
- **Hamiltonian Monte Carlo** on transformed coordinates samples any differentiable
  prior. The transform matters, and libraries disagree: NumPyro's default simplex
  transform is stick-breaking, PyMC's a centred log-ratio with the last coordinate
  dropped, TensorFlow Probability's an additive log-ratio, and Stan's, since version
  2.37, the softmax of an orthonormal sum-to-zero vector (isometric log-ratio), adopted
  without a published benchmark. For Dirichlet concentrations well below one the tails in
  log-ratio coordinates become very long.
- **Mean-field variational Bayes** has a closed form for this model, $q(\rho)=
  \mathrm{Dir}(\tilde\alpha)$ with "digamma-shrunk" stacking, but it factorises the labels
  from $\rho$ and so ignores the coupling that broad likelihoods create; variational
  inference "generally underestimates the variance of the posterior density" {cite:p}`blei2017`.
- **Laplace approximations** fail at the boundary of the simplex and become
  prior-dominated when the likelihoods are flat across bins.
- **Stochastic-gradient Langevin samplers** on the simplex have discretisation biases
  that "dominate at the boundary"; the stochastic Cox–Ingersoll–Ross process removes them
  {cite:p}`baker2018`.
- **Evidences** come from nested sampling at small $K$, or from Chib's estimator {cite:p}`chib1995`
  applied to conjugate Gibbs output.

No published benchmark compares Gibbs, NUTS, variational Bayes and EM with a Laplace
approximation on this model as the likelihoods broaden.

**In binference**, NUTS runs on its own isometric log-ratio coordinates, independent of
any library's default; `Gibbs` is the conjugate sampler; `MAP` uses mix-SQP or
SQUAREM-accelerated EM and reports a KKT gap; `Laplace` corrects its Gaussian draws by
Pareto-smoothed importance sampling and reports the Pareto $\hat k$; `MeanFieldVB` is
the closed-form scheme above (the algorithm of RAIL's `VarInfStackSummarizer`), flagged
as approximate. `method="auto"` chooses among them by rules measured on a benchmark
({doc}`../guides/engines`). binference has no stochastic-gradient samplers and computes
no evidences; held-out predictive densities (`result.log_predictive`) compare models
instead.

## Pitfalls known for simplex priors

- **"Uninformative" Dirichlets are informative for sparse tables.** With 3 observations
  in 1000 cells, the Jeffreys prior ($\alpha=1/2$) "effectively added 1/2 to the 998 zero
  cells", leaving the two observed cells only 0.008 of the posterior mass; the uniform
  prior "is even worse", and {cite:t}`berger2015` recommend $\alpha\approx1/K$.
  Broad likelihoods reduce the effective counts further. `Dirichlet.reference()` sets
  $\alpha=1/K$ ({doc}`../guides/pitfalls`).
- **The Haldane limit** $\alpha\to0$ gives an improper posterior if any bin is empty.
- **A symmetric $\alpha$ is not consistent under merging bins** of unequal width or
  number: coarse groups get prior mass in proportion to the number of fine bins they
  contain (rubias's reporting-unit bias; see {doc}`quantification`).
  `Dirichlet.from_base_measure` ties $\alpha$ to bin widths.
- **Pseudo-observation encodings cannot represent $\alpha_k<1$** (ashr's MAP routines);
  sparse priors need samplers rather than penalised solvers, and the MAP with
  $\alpha_k<1$ lies on the boundary.
- **Exact zeros.** At $\alpha=0.01$, 0.06% of NumPy's Dirichlet draws contain an exact
  zero, which can make an object's $L_i\cdot\rho$ vanish (frankenz audit); EM never
  revives a bin it has set to zero.
- **Bins without data** are not identified: their posterior is their prior conditional.
  `result.data_support` and `result.prior_information_ratio()` show how much the data
  determine ({doc}`../guides/results`).
- **Smoothing scales** are hard to learn: the strength of a smoothness prior railed
  against its prior bounds in GWTC-4 and was fixed by hand (see
  {doc}`gravitational-waves`).
- **A single Dirichlet sees only shot noise.** Its width reflects counting statistics
  alone; sample variance and calibration errors need an extra layer (see
  {doc}`astronomy`).
