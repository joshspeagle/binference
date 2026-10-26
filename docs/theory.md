# Theory

This page collects the mathematics behind binference: the model, its equivalent
forms, what the engines compute, and why a few common shortcuts go wrong. The
notation is used throughout the code and docstrings.

## Objects, bins and likelihoods

We observe $N$ objects $i=1,\dots,N$ with data $d_i$. The quantity of interest
$z$ (a redshift, a position in a colour–magnitude diagram, a class label) is
partitioned into $K$ bins $B_1,\dots,B_K$ with volumes $v_k=|B_k|$ (widths in
one dimension; $v_k\equiv1$ for categories).

- $\ell_i(z)=p(d_i\mid z)$ is object $i$'s **likelihood function**. Only its
  shape matters: any per-object constant cancels from every population
  quantity.
- The **bin-averaged likelihood** is

  $$
  L_{ik}=\frac{1}{v_k}\int_{B_k}\ell_i(z)\,dz .
  $$ (eq-L)

- Optional **weights** $w_i>0$ (with $W=\sum_iw_i$) count duplicated rows:
  integer weights are multiplicities. Non-integer weights define a weighted
  pseudo-likelihood rather than a proper posterior.

## The population model

The population is a histogram: a piecewise-constant density with fractions
$\rho$ on the simplex $\Delta^{K-1}$,

$$
n(z\mid\rho)=\sum_k\rho_k\,\frac{\mathbb 1[z\in B_k]}{v_k}.
$$

Each object's marginal likelihood is then a dot product,

$$
p(d_i\mid\rho)=\int\ell_i(z)\,n(z\mid\rho)\,dz=\sum_k\rho_kL_{ik}\equiv L_i\cdot\rho ,
$$

and the posterior for the population is

$$
p(\rho\mid D)\;\propto\;p(\rho)\prod_{i=1}^{N}\left(L_i\cdot\rho\right)^{w_i}.
$$ (eq-P)

Three properties make this tractable for very large $N$:

- **Concavity.** The log-likelihood $\ell(\rho)=\sum_iw_i\log(L_i\cdot\rho)$ is
  concave on the simplex, so the maximum-likelihood problem is convex.
- **Invariance.** Rescaling any row of $L$ leaves the posterior unchanged.
- **Cheap gradients.** Two matrix–vector products give everything. With
  $s=L\rho$, the gradient is $\partial\ell/\partial\rho_k=\sum_iw_iL_{ik}/s_i$,
  and the **expected counts** are

  $$
  \hat n_k(\rho)=\rho_k\sum_i\frac{w_iL_{ik}}{L_i\cdot\rho}.
  $$

### The augmented (hierarchical) form

Give every object a latent bin label, $z_i\mid\rho\sim\mathrm{Cat}(\rho)$, with
$p(d_i\mid z_i=k)\propto L_{ik}$. The joint posterior

$$
p(\rho,z\mid D)\propto p(\rho)\prod_i\rho_{z_i}L_{iz_i}
$$

sums over $z$ to exactly {eq}`eq-P`. The "population" view (sample $\rho$
directly) and the "hierarchical" view (sample $\rho$ and the labels together)
are therefore **the same model** for $\rho$. They differ only in how
efficiently they can be sampled, and in whether labels come out jointly. The
conditional distribution of a label is exact:

$$
P(z_i=k\mid\rho,D)=r_{ik}(\rho)=\frac{L_{ik}\rho_k}{L_i\cdot\rho},
$$ (eq-resp)

so joint draws of labels can be generated afterwards from any posterior draw
of $\rho$. These **responsibilities** are object $i$'s posterior over bins
when the population is $\rho$.

### Population versus realised sample

$\rho$ describes the population the objects were drawn from, and its
posterior includes the multinomial noise of having drawn only $N$ objects. The
**realised histogram** $n_k=\sum_iw_i\mathbb 1[z_i=k]$ describes the objects
actually observed. With perfect measurements $n$ is known exactly while $\rho$
is still uncertain. A lensing kernel depends on the realised redshifts of the
lens sample; an occurrence rate or luminosity function is a property of the
population. binference returns both: `result.samples` (draws of $\rho$) and
`result.realized_counts()` (draws of $n$).

### Individual objects

Object $i$'s posterior under the inferred population averages
{eq}`eq-resp` over the posterior,
$P(z_i=k\mid D)=\mathbb E_{\rho\mid D}[r_{ik}(\rho)]$. It is narrower than its
own likelihood wherever the population is informative: this is hierarchical
**shrinkage**. For a *new* object with likelihood row $L_*$, the posterior
predictive is $\mathbb E[r_*(\rho)]$, and its log predictive density
$\log\mathbb E[L_*\cdot\rho]$ scores models on held-out data
(`result.log_predictive`).

## Point estimates: EM is iterated stacking

The EM algorithm for {eq}`eq-P` with a flat prior is

$$
\rho_k^{(t+1)}=\frac{\hat n_k(\rho^{(t)})}{W}
=\frac1W\sum_iw_i\,r_{ik}(\rho^{(t)}),
$$

which **stacks the per-object posteriors computed under the current
population**. Two consequences follow:

- **Stacking is one EM step.** Stacking posteriors computed under an interim
  prior $\pi$ is one EM step started from $\pi$. It is biased whenever the
  likelihoods are broader than the features of the population, which is
  exactly when error bars matter.
- **The same iteration recurs across fields.** Iterative Bayesian unfolding,
  Richardson–Lucy deconvolution, EM quantification (Saerens et al. 2002) and
  the Shepp–Vardi algorithm are all this update. Stopping it early is an
  implicit regulariser, not a prior.

At the maximum the Karush–Kuhn–Tucker conditions read

$$
g_k=\frac1W\sum_i\frac{w_iL_{ik}}{L_i\cdot\hat\rho}\le1,\quad\text{with equality where }\hat\rho_k>0,
$$

so binference reports the **optimality gap** $\max_kg_k-1$ for every point
estimate (`PointEstimate.kkt_gap`); it is infinite where an object has zero
likelihood. For rates, with $G$ the gradient of the log posterior in $\lambda$,
the conditions are $G_k\le0$ with equality where $\hat\lambda_k>0$, and the gap is
the larger of $\max_kG_k/S_k$ and $|\lambda\cdot G|/(S\cdot\lambda)$, so it does
not depend on the exposure's units. `MAP(tol=)` bounds this gap for EM and
the same gap of the problem mix-SQP solves (below); see
{class}`~binference.samplers.MAP` for the rule in full. With
a $\mathrm{Dir}(\alpha)$ prior the MAP update adds $\alpha_k-1$ pseudo-counts. EM
is slow when likelihoods are broad (a large fraction of missing information), so
binference accelerates it with SQUAREM (Varadhan & Roland 2008).

The **fraction of missing information** (`PopulationModel.missing_information`) is
$1-\lambda_{\min}(I_{\rm obs}, I_c)$, the rate of EM: $I_{\rm obs}$ is the
observed information in the logits and $I_c$ the complete-data information, which
with a selection function counts the imputed missed objects,
$W/(S\cdot\rho)$ objects in all. It is computed on the support of $\rho$ (a
fraction at zero is an active constraint, not missing information), and with
`include_prior=True` it adds a Dirichlet prior's term, as in the MAP EM.

For $\alpha_k\ge1$ the log posterior is concave on the simplex, and binference
uses mix-SQP instead (Kim et al. 2020). Each iteration solves a quadratic
model of the objective under $\rho\ge0$ as a non-negative least-squares
problem, so it converges in a handful of iterations however broad the
likelihoods are. A selection function makes the likelihood non-concave in
$\rho$, but not in the fractions of the *observed* population,
$\phi_k=S_k\rho_k/S\cdot\rho$:

$$
p(x_i\mid\text{observed},\rho)=\frac{\sum_kL_{ik}\rho_k}{S\cdot\rho}=\sum_k\frac{L_{ik}}{S_k}\,\phi_k
$$

in `"on_measurement"` mode, and $\sum_kL_{ik}\phi_k$ in `"on_truth"` mode. The
maximum-likelihood estimate is invariant under this change of variables, so
with a flat prior binference solves for $\phi$ and maps back,
$\rho_k\propto\phi_k/S_k$. Rates with a `GammaRates` prior of shapes
$a_k\ge1$ and rates $b_k$ reduce to the same problem: writing
$\lambda_k=\Lambda\psi_k/(S_k+b_k)$ with $\psi$ on the simplex, the total
$\Lambda=W+\sum_k(a_k-1)$ profiles out, leaving mix-SQP in $\psi$ with
$S_k+b_k$ in place of $S_k$ and a Dirichlet term $\sum_k(a_k-1)\log\psi_k$.

## Likelihoods versus posteriors

Many codes report **posteriors** $p_i(z)\propto\ell_i(z)\pi(z)$ computed
under an **interim prior** $\pi$. With bin masses $P_{ik}=\int_{B_k}p_i$ and
$\Pi_k=\int_{B_k}\pi$, and $\pi$ roughly constant within bins,

$$
L_{ik}\propto\frac{P_{ik}}{\Pi_k}.
$$

`Likelihoods.from_table`, `from_density` and `from_samples` require the
interim prior and divide it out. Using posteriors directly replaces
$L_i\cdot\rho$ by $L_i\cdot(\pi\odot\rho)$, up to a per-object constant:

$$
\sum_iw_i\log(P_i\cdot\rho)=\ell\!\left(\frac{\pi\odot\rho}{\pi\cdot\rho}\right)+W\log(\pi\cdot\rho)+\text{const}.
$$

The data then constrain the product $\pi\odot\rho$ rather than $\rho$, and the
extra factor $(\pi\cdot\rho)^W$ rewards mass where $\pi$ is large. In the
limit of uninformative data the estimate collapses onto the mode of the
interim prior.

### Masses versus densities

$L_{ik}$ is a value per unit volume, an *average* over the bin. Inputs come
in two representations:

- `"density"`: values per unit volume, such as a PDF evaluated at bin centres.
- `"mass"`: values integrated over the bin, such as CDF differences or
  histograms of samples.

Likelihood masses are divided by $v_k$. For posteriors, both the posterior and
the interim prior are converted to masses before taking the ratio. With equal
bins the distinction is a constant per row and harmless. With unequal bins,
the wrong choice biases $\rho$ toward wide bins.

## Selection effects

Let $S_k\in(0,1]$ be the probability that an object in bin $k$ makes it into
the catalogue. For fractions only the ratios of the $S_k$ matter
({eq}`eq-S` is unchanged when $S$ is scaled). For rates the absolute $S$ sets
the total, and with an exposure it may exceed one (below).
The likelihood depends on what the selection acts on:

- **`selection_mode="on_measurement"`.** Selection acts on the *measured* data: a
  flux limit on measured fluxes, a reconstruction efficiency. A detected
  object's likelihood row is unchanged: $\tilde L_{ik}=L_{ik}$.
- **`selection_mode="on_truth"`.** Selection acts on the *true* value, or
  independently of the measurement noise, such as a completeness map in the
  true parameter. Then $p(d_i,\mathrm{det}\mid z)=S(z)\ell_i(z)$, so
  $\tilde L_{ik}=S_kL_{ik}$.

The two modes give different likelihoods, so binference requires the mode to be
stated whenever a selection array is passed.

**Fractions.** Let the total number of objects $\Lambda$ be unknown. The
detected objects then form a Poisson process with expected number
$\Lambda(S\cdot\rho)$. Integrating $\Lambda$ against the scale-invariant
prior $1/\Lambda$ gives

$$
p(\rho\mid D)\propto p(\rho)\,\frac{\prod_i(\tilde L_i\cdot\rho)^{w_i}}{(S\cdot\rho)^W}.
$$ (eq-S)

The two classic mistakes (Mandel, Farr & Gair 2019) are renormalising each
object's likelihood over the detectable data *and* dividing by $S\cdot\rho$,
or omitting the normaliser altogether. Both bias $\rho$ toward
well-detected bins.

Data augmentation makes {eq}`eq-S` conjugate again. Given $\rho$, the
posterior of $\Lambda$ is $\mathrm{Gamma}(W,S\cdot\rho)$, and the number of
missed objects is Poisson given $\Lambda$. Marginalising $\Lambda$ gives

$$
M\mid\rho\sim\mathrm{NegBin}(W,\,S\cdot\rho),\qquad
\text{bins of the missed objects}\sim\mathrm{Mult}\!\left(M,\frac{\rho\odot(1-S)}{1-S\cdot\rho}\right).
$$

Adding these imputed counts to the labelled detections gives a complete,
unselected sample, so $\rho\mid\cdot\sim\mathrm{Dir}(\alpha+n+m)$. EM uses the
expected missed counts $W\rho_k(1-S_k)/(S\cdot\rho)$.

**Rates.** With expected numbers of objects $\lambda_k$ per bin (any exposure
absorbed into $S$), the data are an inhomogeneous Poisson process:

$$
p(\lambda\mid D)\propto p(\lambda)\,e^{-S\cdot\lambda}\prod_i(\tilde L_i\cdot\lambda)^{w_i}.
$$

Independent $\mathrm{Gamma}(a_k,b_k)$ priors are conjugate given the labels:
$\lambda_k\mid z\sim\mathrm{Gamma}(a_k+n_k,\,b_k+S_k)$. Fractions and rates
are not interchangeable priors: independent $\mathrm{Gamma}(a_k,b)$ rates
imply $\rho\sim\mathrm{Dir}(a)$ only when every bin has the same rate
parameter $b$ and there is no selection.

## Priors

### The Dirichlet family

These priors are conjugate to the augmented model, so they work with every
engine.

- **`Dirichlet(alpha)`.** Uniform on the simplex for $\alpha=1$ (the default).
  - `Dirichlet.reference()` uses $\alpha_k=1/K$ (Berger, Bernardo & Sun 2015),
    which matters when there are many bins and few objects, since $\alpha=1$
    adds $K$ pseudo-counts.
  - `from_base_measure` makes $\alpha_k$ proportional to the bin masses of a base
    density (by default flat in the measure coordinate, $\alpha_k\propto v_k$ on
    linear bins), which stays consistent when bins are merged.
  - `from_counts(m, strength=λ)` encodes a reference sample with counts $m$
    as $\alpha=1+\lambda m$, a power prior that down-weights a reference that
    may not be representative.
- **`GeneralizedDirichlet`** (Connor & Mosimann 1969) is a stick-breaking
  product of Betas. It allows positive correlations between bins and
  per-bin variances that a Dirichlet cannot express.
- **`DirichletTree`** (Minka) places a Dirichlet at every node of a tree over
  the bins, so bins that share a node co-vary. The log density is linear in
  the logs of leaf and node masses, so Gibbs and MAP stay closed-form.
- **`PolyaTree`** (Lavine 1992) is the dyadic tree whose Beta concentrations
  grow with depth $m$ as $c\,m^{p}$, centred on a base measure. Fluctuations
  shrink at finer scales, and $p=2$ gives continuous limits.

### Smoothness: the logistic family

A Dirichlet treats bins as exchangeable, so with broad likelihoods
neighbouring bins trade mass and the posterior is jagged (see the figure on the
[front page](index.md)). Smoothness priors put a Gaussian prior on the
log-fractions $\eta=\log\rho+\text{const}$, so that $\rho=\mathrm{softmax}(\eta)$:

$$
\eta=\mu+B\,(s(\theta)\odot\xi),\qquad\xi\sim\mathcal N(0,I).
$$

Here $B$ is a fixed basis and $s(\theta)$ are scales set by hyperparameters
$\theta$ (amplitude, length scale). The mean $\mu$ defaults to the log bin
volumes, $\mu_k=\log v_k$ up to a constant, which is a flat density. Then
$\eta_k-\mu_k=\log(\rho_k/v_k)+\text{const}$ is the log-density, so the
smoothness penalty does not depend on the binning. Passing the log of
reference fractions as $\mu$ centres the prior on a reference distribution
instead.

- **`LogisticRandomWalk(order)`.** The order-$r$ differences of $\eta$
  between adjacent bins are $\mathcal N(0,\tau^2)$.
  - The basis is the eigenvectors of the difference precision $D_r^\top D_r$,
    with $s_j=\tau/\sqrt{\lambda_j}$.
  - Polynomial null modes, such as the linear trend for $r=2$, get a broad
    Gaussian whose root-mean-square size across the grid has standard deviation
    `trend_scale`, whatever the number of bins ($s_j=$ `trend_scale` $\sqrt K$
    for these unit-norm modes).
  - N-D grids use the Kronecker sum (ICAR-type) of the per-axis precisions. Its
    null modes are the products of the per-axis ones, so for $r=2$ in 2-D the
    mixed surface $xy$ is not penalised either.
- **`LogisticGP(kernel)`.** A Hilbert-space reduced-rank Gaussian process
  (Solin & Särkkä 2020).
  - The basis is sines on a box larger than the grid,
    $\phi_j(x)=L^{-1/2}\sin\big(\pi j(x+L)/2L\big)$, and $s_j^2$ is the
    kernel's spectral density at frequency $\pi j/2L$.
  - Matérn-1/2, 3/2 and 5/2 and squared-exponential kernels are available.
  - The field is stationary in the measure coordinate $u$ ($\log z$ on
    log bins), and the box is three times the grid's half-width in $u$.
  - Amplitude and length scales (one per axis) are inferred. The default
    basis has one sine per bin, up to 1024 modes in all, and the default length-scale prior is an
    inverse gamma with 1% mass below the shortest length scale the grid and
    the basis resolve (the larger of two bin widths and
    $f\,cS/m$, with $c$ the box factor, $S$ the half-range and $m$ the number
    of sines; $f$ = 1.3, 1.8, 2.2, 8.5 for the RBF and Matérn-5/2, -3/2, -1/2
    kernels) and 1% above the range; on coarse grids, where that bound
    exceeds a quarter of the range, a log-normal around half the range.
- **`LogisticNormal(mean, cov)`** takes a general covariance.

The overall level of $\eta$ is not identified by $\rho$, so the constant mode
is dropped.

**Partial non-centring.** The whitened form above is efficient when the data
are weak and inefficient when they are strong, because then $s(\theta)$ and
$\xi$ trade off along a curved ridge. binference's NUTS engine re-parametrises
each basis coefficient $c_j=s_j\xi_j$ as

$$
c_j=s_j^{\lambda_j}\zeta_j,\qquad\zeta_j\sim\mathcal N\big(0,s_j^{2(1-\lambda_j)}\big),
$$

which interpolates between the non-centred form ($\lambda_j=1$) and the centred
form ($\lambda_j=0$). For a Gaussian likelihood with Fisher information $F_j$
the best choice is $\lambda_j=1/(1+s_j^2F_j)$ (Papaspiliopoulos, Roberts &
Sköld 2007; Gorinova et al. 2020). This equals $1/(-H_{jj})$, where $H$ is the
Hessian of the log posterior in whitened coordinates, so binference estimates it
from finite differences of the gradient early in warm-up. The posterior is
unchanged; only the coordinates the sampler moves in change.

Centring is safe only if the scales $s_j$ stay in a range where the data
dominate; otherwise the centred coordinates have a funnel of their own. With
few basis coefficients per hyperparameter (a handful of bins), the field
constrains the hyperparameters weakly, so binference keeps the non-centred form
when the median spread of $\log s_j$ given the field exceeds 0.28. That
spread comes from the curvature of the log posterior in the hyperparameters
at fixed field.

### Hyperpriors

Hyperparameters are sampled on the log scale, with `HalfNormal`, `LogNormal`,
`InverseGamma` or `Gamma` priors. Pass a number to fix one.

## Unconstrained coordinates

Gradient-based engines need $\mathbb R^d$. For simplex priors binference uses the
**isometric log-ratio** (ILR) transform, $\eta=Vx$ with
$x\in\mathbb R^{K-1}$, where $V$ is the orthonormal Helmert basis of the
zero-sum subspace, and $\rho=\mathrm{softmax}(\eta)$. The density of $x$ picks
up the Jacobian

$$
\log\left|\frac{\partial\rho_{1:K-1}}{\partial x}\right|=\sum_k\log\rho_k+\tfrac12\log K .
$$

For a $\mathrm{Dir}(\alpha)$ prior the unconstrained log density is therefore
$\sum_k\alpha_k\log\rho_k$, whose mode is the MAP with $\alpha_k$ in place of
$\alpha_k-1$. This is the centre used by the Laplace engine. Logistic priors
are already unconstrained, with log-hyperparameters appended. Rates use
$\log\lambda$.

### Two densities, two modes

The same posterior has a density in each coordinate system, and the two differ by
the Jacobian above, so they have different modes (DECISIONS #15):

| density | coordinates | Jacobian | mode | used by |
|---|---|---|---|---|
| `PopulationModel.log_posterior(value)` | fractions $\rho$ on the simplex (rates $\lambda$) | no | `MAP(space="simplex")` | EM, mix-SQP, the `log_posterior` of every point estimate (NaN where the prior has no closed-form density) |
| `PopulationModel.log_density_unconstrained(x)` | unconstrained $x$ | yes | `MAP(space="unconstrained")` | NUTS, Laplace, L-BFGS |

Neither mode is "the" answer: the posterior *mean* and quantiles are the same in
any coordinates. The simplex mode can sit on the boundary (a fraction exactly
zero); the unconstrained mode is always interior, which is why the Laplace
approximation is centred there.

The gradient of the log-likelihood with respect to the logits is
$\hat n-W\rho$, or $\hat n-W\rho\odot S/(S\cdot\rho)$ with selection, or
$\hat n-S\odot\lambda$ for rates. It costs the same two matrix–vector products
as the likelihood itself.

## The engines

- **NUTS** (Hoffman & Gelman 2014; Betancourt 2017), exact for every prior and
  the fallback of the default `method="auto"`.
  - Multinomial trajectory sampling with the generalised U-turn criterion
    and Stan's extra checks across merged subtrees.
  - Dual-averaging step-size adaptation, with a target acceptance of 0.8, or
    0.9 for logistic priors.
  - Windowed adaptation of a diagonal metric; logistic priors get a dense
    metric that starts from the curvature.
  - Chains start at the posterior mode plus jitter, and can run in parallel
    processes.
- **Gibbs** alternates $z\mid\rho$ (vectorised categorical draws) and
  $\rho\mid z$ (conjugate), with the negative-binomial augmentation under
  selection. It is exact but mixes slowly when likelihoods are broader than
  the bins: the autocorrelation time grows with the fraction of missing
  information.
- **MAP** uses mix-SQP (Dirichlet priors with $\alpha_k\ge1$, `GammaRates` with
  shapes $\ge1$) or
  SQUAREM-accelerated EM on the simplex, or L-BFGS in unconstrained
  coordinates (non-conjugate priors, densities unbounded at the edge of the
  simplex, and the Laplace centre), with the KKT gap as a certificate.
- **Laplace** fits a Gaussian at the unconstrained mode, using a
  finite-difference Hessian.
  - It is corrected by importance sampling from a Student-$t$ proposal with
    Pareto-smoothed weights (Vehtari et al. 2024).
  - Results with Pareto $\hat k\ge\min(1-1/\log_{10}S,\,0.7)$ for $S$
    proposals are flagged as approximate. This
    happens routinely for $K\gtrsim10$, because near-empty bins have one-sided
    tails.
- **MeanFieldVB** applies coordinate ascent to $q(\rho)\prod_iq(z_i)$.
  Factorising the labels from $\rho$ ignores exactly the coupling that broad
  likelihoods create, so its intervals are too narrow. It is always flagged
  as approximate.
- **Bootstrap** draws the sampling distribution of the MAP estimate by
  resampling objects (nonparametric) or by Dirichlet-weighting them (Bayesian
  bootstrap).

## Diagnostics and validation

- **Convergence.** Rank-normalised split $\hat R$ and bulk and tail effective
  sample sizes (Vehtari et al. 2021).
- **NUTS health.** Divergences, saturated tree depth and E-BFMI.
- **Monte Carlo error.** Standard errors for posterior means.

`result.check()` turns these problems into warnings, together with an importance
check's Pareto $\hat k$, an optimiser that did not converge, and problems with the
inputs.

The validation module answers a different question: is the *whole pipeline*
calibrated?
- `exact_posterior` integrates {eq}`eq-P` by quadrature for $K\le3$.
- `sbc` runs simulation-based calibration (Talts et al. 2018): ranks of true
  parameters among posterior draws, including the joint log-likelihood as a
  test quantity, tested for uniformity by a histogram chi-square or an ECDF
  test (Säilynoja et al. 2022).
- `coverage` measures the frequentist coverage of credible intervals, for
  bins and for any functional.

## Binning is a modelling choice

The histogram model assumes the population is flat within each bin. When it
is not, and bins are coarse compared with the measurement errors, the
bin-averaged likelihood {eq}`eq-L` weights the wrong parts of each bin. The
resulting **binning bias** does not shrink with $N$ and eventually dominates
the error bars. On a six-bin mock with errors comparable to the bin width it
exceeds four posterior standard deviations at $N=20\,000$. Use bins narrower
than the features you care about, add a smoothness prior to control the
noise, and report coarse summaries as functionals of the fine posterior
(`result.rebin`, `result.functional`). Stanley, Patil & Kuusela (2022) reach
the same conclusion for unfolding.

## References

- Berger, Bernardo & Sun (2015), *Bayesian Analysis* 10, 189: reference priors for the multinomial.
- Betancourt (2017), arXiv:1701.02434: a conceptual introduction to Hamiltonian Monte Carlo.
- Connor & Mosimann (1969), *JASA* 64, 194: the generalised Dirichlet.
- Gorinova, Moore & Hoffman (2020), *ICML*: automatic reparameterisation.
- Hoffman & Gelman (2014), *JMLR* 15, 1593: the No-U-Turn Sampler.
- Hogg, Myers & Bovy (2010), *ApJ* 725, 2166: population inference from per-object samples.
- Kim, Carbonetto, Stephens & Anitescu (2020), *Journal of Computational and Graphical Statistics* 29, 261: mix-SQP.
- Lavine (1992), *Annals of Statistics* 20, 1222: Pólya trees.
- Leistedt, Mortlock & Peiris (2016), *MNRAS* 460, 4258: hierarchical $n(z)$.
- Mandel, Farr & Gair (2019), *MNRAS* 486, 1086: selection effects.
- Papaspiliopoulos, Roberts & Sköld (2007), *Statistical Science* 22, 59: non-centred parametrisations.
- Saerens, Latinne & Decaestecker (2002), *Neural Computation* 14, 21: EM for priors under shift.
- Säilynoja, Bürkner & Vehtari (2022), *Statistics and Computing* 32, 32: ECDF tests of rank uniformity.
- Solin & Särkkä (2020), *Statistics and Computing* 30, 419: Hilbert-space GPs.
- Stanley, Patil & Kuusela (2022), *JINST* 17, P10013: binning in unfolding.
- Talts et al. (2018), arXiv:1804.06788: simulation-based calibration.
- Varadhan & Roland (2008), *Scandinavian Journal of Statistics* 35, 335: SQUAREM.
- Vehtari et al. (2021), *Bayesian Analysis* 16, 667: rank-normalised $\hat R$ and ESS.
- Vehtari et al. (2024), *JMLR* 25, 72: Pareto-smoothed importance sampling.
