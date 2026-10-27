# Glossary

Each entry defines a term as the documentation and the API use it. Symbols follow the
{doc}`notation` page. Where another field names the same thing differently, the entry
says so; the {doc}`fields pages <../fields/index>` give the full mapping.

```{glossary}
:sorted:

approximate
  A result whose draws are not from the exact posterior: from mean-field VB, the
  bootstrap, or a Laplace fit whose importance check failed. Such a result has
  `result.approximate` set, and its summary says so. See
  [approximations](../guides/engines.md#approximations).

auto
  The default engine, `method="auto"`. It runs Gibbs where Gibbs mixes well, then
  importance-corrected Laplace where its check passes, and NUTS otherwise. The rules are
  in the {doc}`engines guide <../guides/engines>`.

autocorrelation time
  The number of draws of a Markov chain that carry as much information as one
  independent draw. The {term}`effective sample size` is the number of draws divided by
  it.

basis
  A set of components finer than the bins, for a population that is not flat within
  bins. The fit uses the components; the fractions are still reported in the bins. See
  [`Basis`](../guides/inputs.md#within-bin-shape-and-internal-resolution-basis).

bin
  One of the $K$ cells $B_1,\dots,B_K$ that partition the true value $z$. A bin is an
  interval, a cell of a grid, or a category. See
  [theory](../how-it-works/model.md#objects-bins-and-likelihoods).

bin volume
  $v_k$: the width of bin $k$ in one dimension, its area or volume on a grid, and one
  for a category. `bins.volumes` holds it in $z$, and `bins.u_volumes` in the
  {term}`measure` coordinate.

bin-averaged likelihood
  $L_{ik}$: the likelihood of object $i$ averaged over bin $k$. It is the only per-object
  input the model uses, and only its shape across the bins matters. See
  [theory](../how-it-works/model.md#objects-bins-and-likelihoods).

binned measurement
  A measurement that records only the measured bin of each object, such as a
  reconstructed energy bin or a predicted class. Its likelihood rows come from a
  {term}`response matrix`. See
  [binned measurements](../guides/inputs.md#binned-measurements-and-classifiers).

binning bias
  The bias from assuming the population is flat within each bin. It does not shrink as
  objects are added. `result.binning_bias()` measures it; see
  [theory](../how-it-works/model.md#binning-bias).

bootstrap
  An engine that refits the {term}`MAP` estimate to resampled or reweighted objects.
  Its draws describe the spread of the estimate over repeated data sets, not a
  posterior. See [approximations](../guides/engines.md#approximations).

chain
  One sequence of draws from a Markov-chain engine, NUTS or Gibbs. Several chains from
  different starting points make {term}`R-hat` possible.

concentration
  $\alpha_k$, the parameter of a {term}`Dirichlet prior`. Large values pull the fractions
  toward the prior mean; values below one favour nearly empty bins.

conjugate prior
  A prior whose posterior has the same form once every object's bin is known. The
  Dirichlet family and `GammaRates` are conjugate, so Gibbs, EM and mean-field VB apply
  to them. See [priors](../how-it-works/priors.md#the-dirichlet-family-as-trees).

coverage
  The share of simulated data sets in which a credible interval contains the true value.
  A calibrated 68% interval covers the truth in 68% of them.
  `binference.validation.coverage` measures it; see the
  {doc}`validation guide <../guides/validation>`.

credible interval
  An interval that holds a stated posterior probability of a quantity.
  `result.interval(level)` returns equal-tailed or highest-density intervals per bin.

data support
  The number of objects' worth of likelihood in each bin, `result.data_support`. The
  prior sets the fraction of a bin with less than about one. See
  [the population](../guides/results.md#the-population).

Dirichlet prior
  The default prior on {term}`population fractions`, `Dirichlet(alpha)`. With every
  {term}`concentration` equal to one it is uniform on the {term}`simplex`. See
  [the Dirichlet family](../how-it-works/priors.md#the-dirichlet-family-as-trees).

Dirichlet tree
  A prior with a Dirichlet at every node of a tree over the bins, `DirichletTree`. Bins
  that share a node vary together. It is a {term}`conjugate prior`.

divergence
divergent transition
  A NUTS trajectory whose energy error became very large. Divergences mark regions the
  sampler cannot explore, so the draws there may be biased. See
  [NUTS](../guides/engines.md#nuts).

draw
  One value of the population, or of another quantity, from a posterior or an
  approximation of it. `result.samples` holds the draws of the fractions.

E-BFMI
  The energy Bayesian fraction of missing information of a NUTS chain. Low values mean
  the chain explores the energy distribution poorly, often because of heavy tails.

effective sample size
ESS
  The number of independent draws that would estimate a quantity as precisely as the
  correlated draws do. The bulk ESS concerns the centre of a distribution and the tail
  ESS its 5% and 95% quantiles. See
  [diagnostics](../guides/results.md#diagnostics).

efficiency
  $\epsilon_k$: the probability that an object in true bin $k$ is observed at all. It is
  the column sum of a {term}`response matrix`.

EM
expectation-maximisation
  The iteration that finds the {term}`MAP` or maximum-likelihood population by
  repeated {term}`stacking` under the current estimate. See
  [EM is iterated stacking](../how-it-works/point-estimates.md#em-is-iterated-stacking).
  It is also called iterative Bayesian unfolding or Richardson–Lucy in particle
  physics, and EMQ in quantification.

engine
  An algorithm that turns a model into draws or a point estimate: auto, NUTS, Gibbs,
  MAP, Laplace, mean-field VB or the bootstrap. `infer(method=...)` chooses it; see the
  {doc}`engines guide <../guides/engines>`.

exposure
  An absolute {term}`selection` for {term}`rates`, such as observing time times area. It
  is declared with `scale="exposure"` and may exceed one. See
  [rates and exposure](../guides/selection.md#rates-and-exposure).

expected counts
  $\hat n_k$: the number of objects the data assign to bin $k$ under a given population,
  the sum of the {term}`responsibilities`. EM divides them by the total weight.

Gamma rates prior
  `GammaRates(shape, rate)`: independent Gamma priors on the {term}`rates`. It is a
  {term}`conjugate prior`. See [rates instead of fractions](../guides/priors.md#rates-instead-of-fractions).

generalized Dirichlet
  A stick-breaking product of Beta distributions, `GeneralizedDirichlet`. It allows
  correlations and variances between bins that a Dirichlet cannot express.

Gibbs
  An engine that alternates drawing every object's bin given the population and the
  population given the bins. It is exact for {term}`conjugate priors <conjugate prior>`
  and slow when {term}`missing information` is high. See
  [Gibbs sampling](../guides/engines.md#gibbs-sampling).

hierarchical posterior
  Object $i$'s posterior over bins under the inferred population, $P(k_i=k\mid D)$. It
  averages the {term}`responsibilities` over the population's posterior.
  `result.assignment_probabilities()` returns it; see
  [individual objects](../guides/results.md#individual-objects). It is also called a
  population-informed posterior in gravitational-wave astronomy.

hyperparameter
  A parameter of a prior that is itself inferred, such as the amplitude or length scale
  of a {term}`logistic prior`. Its prior is a {term}`hyperprior`.

hyperprior
  The prior of a {term}`hyperparameter`: `HalfNormal`, `LogNormal`, `InverseGamma` or
  `Gamma`. A plain number fixes the hyperparameter instead. See
  [hyperpriors](../how-it-works/priors.md#hyperpriors).

importance sampling
  Reweighting draws from a simple proposal so that they represent the exact posterior.
  The Laplace engine and leave-one-out posteriors use its Pareto-smoothed form.
  {term}`Pareto k-hat` says whether it is reliable.

injections
  Simulated objects passed through the detection pipeline to measure the
  {term}`selection`. `Selection.from_injections` counts the injected and recovered
  objects per bin. See
  [selection measured by injections](../guides/selection.md#selection-measured-by-injections).

interim prior
  $\pi$: the prior under which per-object posteriors or samples were computed.
  binference divides it out to recover likelihoods. It is also called the PE prior in
  gravitational-wave astronomy and the training prevalence in quantification. See
  [posteriors](../guides/inputs.md#posteriors-computed-under-an-interim-prior).

KKT gap
  How far a point estimate is from the optimality conditions of its problem. It is zero
  at the optimum and infinite where an object has zero likelihood.
  `PointEstimate.kkt_gap` reports it; see
  [point estimates](../how-it-works/point-estimates.md#em-is-iterated-stacking).

label
  $k_i$: the bin that object $i$ truly lies in. The model treats it as unknown; `Gibbs`
  can draw it jointly with the population.

Laplace
  An engine that fits a Gaussian at the posterior mode in {term}`unconstrained
  coordinates`. By default it corrects the Gaussian by {term}`importance sampling` and
  flags the result approximate when {term}`Pareto k-hat` is too large. See
  [approximations](../guides/engines.md#approximations).

leave-one-out
  Each object's posterior under the population fitted without it.
  `result.assignment_probabilities_loo()` estimates it by importance sampling, for
  calibration checks.

L-BFGS
  A quasi-Newton optimiser. The MAP engine uses it in {term}`unconstrained coordinates`
  for priors that mix-SQP and EM cannot handle, and to find the Laplace centre.

label shift
  The assumption that the classes differ in prevalence between a labelled set and a
  target set, while each class's scores keep their distribution. A response from
  `Response.from_labelled` relies on it.

likelihood function
  $\ell_i(z)=p(d_i\mid z)$: the probability of object $i$'s data if its true value is $z$.
  Its average over each bin is the {term}`bin-averaged likelihood`.

likelihood matrix
  The $N\times K$ table of {term}`bin-averaged likelihoods <bin-averaged likelihood>`
  $L_{ik}$, one row per object. `binference.Likelihoods` holds it with its bins, weights
  and selection.

logistic prior
smoothness prior
  A prior that makes neighbouring bins similar: `LogisticNormal`,
  `LogisticRandomWalk` or `LogisticGP`. It puts a Gaussian on the log-fractions and
  infers how smooth they are. See
  [smoothness](../how-it-works/priors.md#the-logistic-family).

log predictive density
  $\log\mathbb E[L_*\cdot\rho]$ for new objects with likelihood rows $L_*$. It scores
  models on held-out data; `result.log_predictive` returns it.

MAP
point estimate
  The single most probable population, `method="map"`, found by mix-SQP, EM or L-BFGS.
  With the default flat Dirichlet prior it is the maximum-likelihood estimate. See
  [point estimates](../guides/engines.md#point-estimates).

mass
density
  The two ways an input can give values per bin. A mass is integrated over the bin; a
  density is a value per unit {term}`bin volume`. See
  [masses versus densities](../how-it-works/model.md#masses-versus-densities).

Monte Carlo standard error
MCSE
  The standard deviation of an estimate computed from a finite number of draws.
  `binference.diagnostics.mcse` computes it for a posterior mean.

mean-field VB
  Variational Bayes that treats the population and every object's bin as independent,
  `method="vb"`. It is fast, but its intervals are too narrow, so its results are flagged
  {term}`approximate`.

measure
  The coordinate $u=T(z)$ in which the population is taken as flat within each bin,
  such as $z$ itself or $\log z$. See
  [measures](../guides/inputs.md#measures-flat-in-which-coordinate).

missing information
  $f$, between 0 and 1: how much the data leave each object's bin uncertain. It is near
  one when likelihoods are broad compared with the bins, and then EM and Gibbs are slow.
  `PopulationModel.missing_information` computes it.

mix-SQP
  A second-order method for the {term}`MAP` {cite:p}`kim2020`. It applies to a Dirichlet
  prior with concentrations of at least one (exactly one with a selection), and to
  `GammaRates` with shapes of at least one. It needs far fewer iterations than EM when
  likelihoods are broad.

mock catalogue
  A simulated set of objects with a known population, from `binference.simulate`. It
  tests whether a fit recovers the truth. See
  [mock catalogues](../guides/validation.md#mock-catalogues).

NUTS
  The No-U-Turn Sampler, a Hamiltonian Monte Carlo engine, `method="nuts"`. It samples
  the exact posterior for every prior. See [NUTS](../guides/engines.md#nuts).

object
  One measured thing: a galaxy, an event, a patient, a classified document. Object $i$
  has data $d_i$ and a true value $z$ that lies in one bin.

Pareto k-hat
  $\hat k$: the estimated tail shape of a set of importance weights. Above a threshold
  that depends on their number, the {term}`importance sampling` estimate is unreliable.

Pólya tree
  A tree prior whose Beta concentrations grow with depth, `PolyaTree`. Its fluctuations
  shrink at finer scales, and it is centred on a base measure.

population
  The distribution that the objects are drawn from, described by
  {term}`population fractions` or {term}`rates`. It differs from the
  {term}`realised sample`.

population fractions
fractions
  $\rho_k$: the share of the population in bin $k$. They sum to one, and they are the
  quantity binference infers. They are also called the prevalence in quantification,
  the prior $g$ in empirical Bayes, the true spectrum in particle physics, and mixture
  proportions in genetics. In astronomy their density per unit redshift is the redshift
  distribution $n(z)$.

posterior predictive p-value
  The probability of counts at least as discrepant as those observed, under the fitted
  model. `result.check()` uses it to test binned counts against the
  {term}`response matrix`.

posterior samples
  Draws from each object's own posterior, computed under an {term}`interim prior`.
  `Likelihoods.from_samples` turns them into likelihoods. See
  [posterior samples](../guides/inputs.md#posterior-samples).

R-hat
  $\hat R$: a convergence diagnostic that compares the spread within chains with the
  spread between them. Values near one mean the chains agree. See
  [diagnostics](../guides/results.md#diagnostics).

rates
  $\lambda_k$: the expected number of objects in bin $k$, per unit {term}`exposure`.
  Their sum is the {term}`total`. See [rates and exposure](../guides/selection.md#rates-and-exposure).

realised sample
realised histogram
  $n_k$: the number of the observed objects whose true value lies in bin $k$. Its
  uncertainty leaves out the noise of drawing $N$ objects from the population, so it is
  narrower than that of the {term}`population fractions`.
  `result.realized_counts()` returns its draws.

reference prior
  `Dirichlet.reference()`, with every concentration $1/K$ {cite:p}`berger2015`.
  It adds one pseudo-object in all, rather than $K$.

representation
  Whether input values are a {term}`mass` or a {term}`density`, `representation="mass"`
  or `"density"`. With unequal bins the wrong choice biases the fractions.

response matrix
  $R_{jk}$: the probability that an object in true bin $k$ is observed in measured bin
  $j$. `binference.Response` holds it. Classification calls it the confusion matrix.

responsibilities
  $r_{ik}(\rho)$: object $i$'s posterior over bins for a given population $\rho$. EM
  averages them over objects. See
  [the augmented form](../how-it-works/model.md#the-augmented-form).

SBC
simulation-based calibration
  A test of a whole pipeline. It ranks the true values of simulated data sets among
  their posterior draws; calibrated fits give uniform ranks. See
  [SBC](../guides/validation.md#simulation-based-calibration).

selected population
  $\phi_k$: the fractions of the objects that are observed, $S_k\rho_k/S\cdot\rho$.
  `result.selected` holds its draws.

selection
selection function
  $S_k$: the probability that an object in true bin $k$ enters the catalogue. It is
  also called completeness in astronomy and detection probability in
  gravitational-wave astronomy. See the {doc}`selection guide <../guides/selection>`.

selection mode
  Whether selection acts on the true value (`"on_truth"`) or on the measured data
  (`"on_measurement"`). The two give different likelihoods, so it must be stated. See
  [three ways selection can act](../guides/selection.md#three-ways-selection-can-act).

shrinkage
  The narrowing of each object's posterior when the inferred population informs it. The
  {term}`hierarchical posterior` is narrower than the object's own likelihood wherever
  the population is informative.

simplex
  The set of $K$ non-negative numbers that sum to one. The {term}`population
  fractions` live on it.

SQUAREM
  An acceleration of EM {cite:p}`varadhan2008`, on by default in `MAP(accelerate=True)`.

stacking
  Averaging per-object posteriors computed under an {term}`interim prior`. It is one EM
  step from that prior, so it is biased when likelihoods are broad. It is also called
  probabilistic classify and count in quantification. See
  [stacking](../guides/pitfalls.md#stacking).

total
  $\Lambda$: the expected number of objects in the population, including those the
  selection missed. `result.total` holds its draws.

tree depth
  The number of doublings of a NUTS trajectory. A chain that often reaches `max_depth`
  explores slowly.

unconstrained coordinates
  Coordinates in which every real vector is a valid population: isometric log-ratios for
  fractions, log-rates for rates. NUTS, Laplace and L-BFGS work in them. See
  [unconstrained coordinates](../how-it-works/priors.md#coordinates).

warm-up
  The first iterations of a NUTS or Gibbs chain, discarded while the sampler adapts or
  forgets its start. `n_warmup` sets their number.

weights
  $w_i$: integer multiplicities of the objects. A row of weight 3 counts as three
  identical objects; non-integer weights give a pseudo-likelihood.
```
