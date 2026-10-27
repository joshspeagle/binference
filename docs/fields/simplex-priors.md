# Priors on the simplex and inference for mixture weights

The fractions $\rho$ live on the simplex, and the statistics literature offers both a
range of priors there and a range of algorithms for the posterior
$p(\rho\mid D)\propto p(\rho)\prod_i\sum_kL_{ik}\rho_k$. This page summarises what is
known about each, and which of them binference provides. The {doc}`theory page
<../how-it-works/theory>` defines binference's priors and engines; the {doc}`priors
<../guides/priors>` and {doc}`engines <../guides/engines>` pages say how to
choose among them.

## Priors

Priors on the simplex fall into two families.

**The conjugate family**: the Dirichlet, the generalised Dirichlet (Connor & Mosimann
1969), the Dirichlet tree (Minka) and finite Pólya trees (Lavine 1992), which are binary
Dirichlet trees. Given the latent counts per bin each has a closed-form posterior, so
Gibbs sampling and EM stay simple. Each adds structure only through an ordering of the
bins or a tree over them:

- **Dirichlet.** All covariances are negative and set by the means and a single
  precision, $\mathrm{Cov}(\rho_j,\rho_k)=-\alpha_j\alpha_k/(\alpha_0^2(\alpha_0+1))$.
- **Generalised Dirichlet.** Independent Beta variables in a stick-breaking cascade,
  with $2(K-1)$ parameters and "a more general covariance structure"; positive
  correlations are possible, and the bin order matters. Leistedt et al. (2016) noted that
  it could encode correlations between redshift bins but did "not consider this case".
- **Dirichlet tree.** A Dirichlet at every internal node of a tree whose leaves are the
  bins; it "can give an independent variance to each $p_k$", and leaves in the same
  subtree are positively correlated (Minka).
- **A Dirichlet process** restricted to fixed bins is a Dirichlet with
  $\alpha_k=\alpha G_0(B_k)$ (Ferguson 1973), so it adds nothing on fixed bins.

**Gaussian priors on log-ratios**: the logistic-normal (Aitchison & Shen 1980), random
walks and intrinsic Gaussian Markov random fields on $\log\rho$ (Rue & Held 2005),
P-splines (Eilers & Marx 1996), and Gaussian processes, which on bins give the logistic
Gaussian process (Leonard 1978; Lenk 1988), with Hilbert-space approximations for speed
(Riutort-Mayol et al. 2023). These are where smoothness and arbitrary correlations live.
They are not conjugate, so they need gradient-based samplers on unconstrained
coordinates, or Pólya–gamma stick-breaking augmentation when Gibbs sampling is required
(Linderman, Johnson & Adams 2015).

**Overdispersion.** The Dirichlet–multinomial (Mosimann 1962) and the three-step
Dirichlet of redshift calibration (Sánchez et al. 2020) add variance beyond the
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
and gradient cost one pass over $L$ ([theory](../how-it-works/theory.md#the-population-model)).

- **Gibbs data augmentation** alternates per-object labels and a conjugate draw of
  $\rho$ (Tanner & Wong 1987; Diebolt & Robert 1994). It is exact, but its convergence,
  like EM's, is governed by the fraction of missing information (Tan 2023): close to zero
  for sharp likelihoods, close to one when each likelihood spreads over many bins. In
  related categorical models data augmentation mixed far worse than a marginal
  Metropolis sampler (Johndrow et al. 2019).
- **EM and MAP.** The EM fixed point is "iterated stacking". EM can be "intolerably slow"
  (Redner & Walker 1984); SQUAREM accelerates it (Varadhan & Roland 2008), and mix-SQP is
  orders of magnitude faster (Kim et al. 2020).
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
  inference "generally underestimates the variance of the posterior density" (Blei,
  Kucukelbir & McAuliffe 2017).
- **Laplace approximations** fail at the boundary of the simplex and become
  prior-dominated when the likelihoods are flat across bins.
- **Stochastic-gradient Langevin samplers** on the simplex have discretisation biases
  that "dominate at the boundary"; the stochastic Cox–Ingersoll–Ross process removes them
  (Baker et al. 2018).
- **Evidences** come from nested sampling at small $K$, or from Chib's (1995) estimator
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
  prior "is even worse", and Berger, Bernardo & Sun (2015) recommend $\alpha\approx1/K$.
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

## References

- Aitchison & Shen (1980), *Biometrika* 67, 261: logistic-normal distributions.
- Baker, Fearnhead, Fox & Nemeth (2018), *NeurIPS* ([arXiv:1806.07137](https://arxiv.org/abs/1806.07137)): large-scale simplex sampling.
- Berger, Bernardo & Sun (2015), *Bayesian Analysis* 10, 189 ([arXiv:1504.02689](https://arxiv.org/abs/1504.02689)): overall objective priors.
- Blei, Kucukelbir & McAuliffe (2017), *JASA* 112, 859 ([arXiv:1601.00670](https://arxiv.org/abs/1601.00670)): variational inference.
- Chib (1995), *JASA* 90, 1313: marginal likelihood from Gibbs output.
- Connor & Mosimann (1969), *JASA* 64, 194: the generalised Dirichlet.
- Diebolt & Robert (1994), *JRSSB* 56, 363: mixture estimation by Gibbs sampling.
- Eilers & Marx (1996), *Statistical Science* 11, 89: P-splines.
- Ferguson (1973), *Annals of Statistics* 1, 209: the Dirichlet process.
- Johndrow et al. (2019), *JASA* 114, 1394 ([arXiv:1605.05798](https://arxiv.org/abs/1605.05798)): mixing of data augmentation.
- Lavine (1992), *Annals of Statistics* 20, 1222: Pólya trees.
- Lenk (1988), *JASA* 83, 509: the logistic normal for density estimation.
- Leonard (1978), *JRSSB* 40, 113: density estimation with logistic Gaussian processes.
- Linderman, Johnson & Adams (2015) ([arXiv:1506.05843](https://arxiv.org/abs/1506.05843)): Pólya–gamma stick-breaking augmentation.
- Minka, [The Dirichlet-tree distribution](https://tminka.github.io/papers/dirichlet/minka-dirtree.pdf).
- Mosimann (1962), *Biometrika* 49, 65: the compound multinomial.
- Redner & Walker (1984), *SIAM Review* 26, 195: mixture densities and EM.
- Riutort-Mayol et al. (2023), *Statistics and Computing* 33: Hilbert-space approximate Gaussian processes.
- Rue & Held (2005), *Gaussian Markov Random Fields*.
- Sánchez, Raveri, Alarcon & Bernstein (2020), *MNRAS* ([arXiv:2004.09542](https://arxiv.org/abs/2004.09542)): 3sDir.
- Tan (2023), *Statistical Science* 38, 240 ([arXiv:1712.08887](https://arxiv.org/abs/1712.08887)): fraction of missing information and data augmentation.
- Tanner & Wong (1987), *JASA* 82, 528: data augmentation.
- Varadhan & Roland (2008), *Scandinavian Journal of Statistics* 35, 335: SQUAREM.
