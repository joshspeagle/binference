# The model

What does binference assume about the objects and their population, and which posterior
does it compute? This page states the model in full, gives its equivalent form with one
latent label per object, and lists every assumption in one place. The symbols are those
of the {doc}`notation page <../reference/notation>`.

```{plain-words}
Each object says how well each bin explains its data. The population says how common
each bin is. The model multiplies the two and asks which populations make all the
objects likely at once.
```

## Objects, bins and likelihoods

We observe $N$ {term}`objects <object>` $i=1,\dots,N$ with data $d_i$. The true value
$z$ of an object (a redshift, a position in a colour–magnitude diagram, a class) is
unknown. Its range is split into $K$ {term}`bins <bin>` $B_1,\dots,B_K$ with volumes
$v_k$: widths in one dimension, areas or volumes on a grid, and one for a category.

- The {term}`likelihood function` of object $i$ is $\ell_i(z)=p(d_i\mid z)$. Only its
  shape matters: a constant factor per object cancels from every population quantity.
- The {term}`bin-averaged likelihood` is its average over each bin,

  $$
  L_{ik}=\frac{1}{v_k}\int_{B_k}\ell_i(z)\,dz .
  $$ (eq-L)

  With a {term}`measure` $u=T(z)$, such as $\log z$, the average is taken in $u$
  instead ([measures](../guides/inputs.md#measures-flat-in-which-coordinate)).
- Optional {term}`weights` $w_i>0$, with total $W=\sum_iw_i$, count identical objects.
  Without weights every $w_i=1$ and $W=N$.

The $N\times K$ table of $L_{ik}$ is the only per-object input the model uses. Row $i$
is written $L_i$.

## The population model

The population is a histogram: a density that is flat within each bin. Its
{term}`population fractions <population fractions>` $\rho$ lie on the
{term}`simplex` $\Delta^{K-1}$, and its density is

$$
p(z\mid\rho)=\sum_k\rho_k\,\frac{\mathbb 1[z\in B_k]}{v_k}.
$$

The marginal likelihood of one object is then a dot product:

$$
p(d_i\mid\rho)=\int\ell_i(z)\,p(z\mid\rho)\,dz=\sum_k\rho_kL_{ik}=L_i\cdot\rho .
$$

With $D$ the data of all objects and $p(\rho)$ the prior, the posterior of the
population is

$$
p(\rho\mid D)\;\propto\;p(\rho)\prod_{i=1}^{N}\left(L_i\cdot\rho\right)^{w_i}.
$$ (eq-P)

Three properties keep this tractable for very large $N$:

- **Concavity.** The log-likelihood $\log p(D\mid\rho)=\sum_iw_i\log(L_i\cdot\rho)$ is
  concave on the simplex. Finding its maximum is therefore a convex problem.
- **Invariance.** Multiplying a row $L_i$ by a constant leaves the posterior unchanged.
- **Cheap gradients.** Two matrix–vector products give everything. The gradient is
  $\partial\log p(D\mid\rho)/\partial\rho_k=\sum_iw_iL_{ik}/(L_i\cdot\rho)$, and the
  {term}`expected counts` are

  $$
  \hat n_k(\rho)=\rho_k\sum_i\frac{w_iL_{ik}}{L_i\cdot\rho}.
  $$

The {doc}`priors` page describes the choices for $p(\rho)$. The {doc}`point estimates
<point-estimates>` and {doc}`engines` pages describe how the posterior is computed.

## The augmented form

Give every object a latent {term}`label` $k_i$, the bin it truly lies in. Let
$k_i\mid\rho\sim\mathrm{Cat}(\rho)$ and $p(d_i\mid k_i=k)\propto L_{ik}$. The joint
posterior of the population and the labels is

$$
p(\rho,k_1,\dots,k_N\mid D)\propto p(\rho)\prod_i\left(\rho_{k_i}L_{ik_i}\right)^{w_i}.
$$

Summing over the labels gives {eq}`eq-P` exactly. The "population" view samples $\rho$
alone, and the "hierarchical" view samples $\rho$ and the labels together. They are
**the same model** for $\rho$. They differ only in how efficiently they can be sampled,
and in whether labels come out jointly.

The distribution of a label given the population is exact:

$$
P(k_i=k\mid\rho,D)=r_{ik}(\rho)=\frac{L_{ik}\rho_k}{L_i\cdot\rho}.
$$ (eq-resp)

These {term}`responsibilities` are object $i$'s posterior over bins when the population
is $\rho$. The expected counts are their weighted sum, $\hat n_k=\sum_iw_ir_{ik}$.
Joint draws of the labels can be made afterwards from any draw of $\rho$, so every
engine's output supports them.

### Population versus realised sample

The fractions $\rho$ describe the population the objects were drawn from. Their
posterior includes the noise of having drawn only $N$ objects. The
{term}`realised histogram <realised sample>`
$n_k=\sum_iw_i\mathbb 1[k_i=k]$ describes the objects actually observed. With perfect
measurements $n$ is known exactly, while $\rho$ is still uncertain.

Which one to report depends on the question. A lensing kernel depends on the realised
redshifts of the lens sample. An occurrence rate or a luminosity function is a property
of the population. `result.samples` holds draws of $\rho$, and
`result.realized_counts()` draws of $n$.

### Individual objects

Object $i$'s {term}`hierarchical posterior` averages {eq}`eq-resp` over the posterior
of the population:

$$
P(k_i=k\mid D)=\mathbb E_{\rho\mid D}\left[r_{ik}(\rho)\right].
$$

It is narrower than the object's own likelihood wherever the population is informative:
this is {term}`shrinkage`. For a new object with likelihood row $L_*$, the posterior
predictive over bins is $\mathbb E[r_*(\rho)]$. Its {term}`log predictive density`,
$\log\mathbb E[L_*\cdot\rho]$, scores models on held-out data
(`result.log_predictive`).

## Likelihoods versus posteriors

Many codes report posteriors $p_i(z)\propto\ell_i(z)\pi(z)$, computed under an
{term}`interim prior` $\pi$. Let $P_{ik}$ be the mass of object $i$'s posterior in
bin $k$ and $\Pi_k$ that of the interim prior. If $\pi$ is roughly constant within each
bin,

$$
L_{ik}\propto\frac{P_{ik}}{\Pi_k}.
$$

`Likelihoods.from_table`, `from_density` and `from_samples` take the interim prior and
divide it out. Using the posteriors as likelihoods instead replaces $L_i\cdot\rho$ by
$P_i\cdot\rho\propto L_i\cdot(\Pi\odot\rho)$, up to a constant per object. Then

$$
\sum_iw_i\log(P_i\cdot\rho)=\log p\!\left(D\,\middle|\,\frac{\Pi\odot\rho}{\Pi\cdot\rho}\right)+W\log(\Pi\cdot\rho)+\text{const}.
$$

The data then constrain the product $\Pi\odot\rho$ rather than $\rho$. The extra factor
$(\Pi\cdot\rho)^W$ also rewards mass where the interim prior is large. As the data
become uninformative, the estimate collapses onto the mode of the interim prior.

Averaging the posteriors $P_{ik}$ over objects is known as {term}`stacking`. It is one
step, started from the interim prior, of the iteration that finds the most probable
population ([EM is iterated stacking](point-estimates.md#em-is-iterated-stacking)).

### Masses versus densities

$L_{ik}$ is an average over the bin, a value per unit volume. Inputs come in two
{term}`representations <representation>`:

- `"density"`: values per unit volume, such as a density evaluated at bin centres;
- `"mass"`: values integrated over the bin, such as differences of a cumulative
  distribution, or histograms of samples.

Likelihood masses are divided by $v_k$. For posteriors, both the posterior and the
interim prior are converted to masses before taking their ratio. With equal bins the
choice changes each row by a constant, which is harmless. With unequal bins the wrong
choice biases $\rho$ towards wide bins.

## Selection effects

Let $S_k$ be the probability that an object in bin $k$ enters the catalogue: the
{term}`selection`. The likelihood depends on what the selection acts on, its
{term}`selection mode`:

- **`"on_measurement"`.** Selection acts on the measured data, such as a flux limit on
  measured fluxes. A detected object's likelihood row is unchanged:
  $\tilde L_{ik}=L_{ik}$.
- **`"on_truth"`.** Selection acts on the true value, or independently of the
  measurement noise, such as a completeness map in the true parameter. Then
  $p(d_i,\text{detected}\mid z)=S(z)\,\ell_i(z)$, so $\tilde L_{ik}=S_kL_{ik}$.

The two modes give different likelihoods, so binference requires the mode whenever a
selection is given. The {doc}`selection guide <../guides/selection>` explains how to
tell them apart and how to combine them.

**Fractions.** Let the expected total number of objects $\Lambda$ be unknown. The
detected objects then form a Poisson process with expected number $\Lambda\,(S\cdot\rho)$.
Integrating $\Lambda$ out under the scale-invariant prior $p(\Lambda)\propto1/\Lambda$
gives

$$
p(\rho\mid D)\propto p(\rho)\,\frac{\prod_i(\tilde L_i\cdot\rho)^{w_i}}{(S\cdot\rho)^W}.
$$ (eq-S)

Scaling $S$ by a constant leaves {eq}`eq-S` unchanged, so for fractions only the ratios
of the $S_k$ matter. The two classic mistakes {cite:p}`mandel2019` are to renormalise
each object's likelihood over the detectable data *and* divide by $S\cdot\rho$, or to
omit the normaliser altogether. Both bias $\rho$ towards well-detected bins.

Given $\rho$, the posterior of the {term}`total` is
$\Lambda\mid\rho\sim\mathrm{Gamma}(W,S\cdot\rho)$, which `result.total` draws. Data
augmentation makes {eq}`eq-S` conjugate again. With $S$ scaled to a maximum of one, the
number of missed objects $M$ and their bins are

$$
M\mid\rho\sim\mathrm{NegBin}(W,\,S\cdot\rho),\qquad
(m_1,\dots,m_K)\mid M,\rho\sim\mathrm{Mult}\!\left(M,\frac{\rho\odot(1-S)}{1-S\cdot\rho}\right).
$$

Adding the missed objects to the labelled detections gives a complete sample without
selection. Gibbs draws them, and EM adds their expected number
$W\rho_k(1-S_k)/(S\cdot\rho)$ to the expected counts.

**Rates.** With {term}`rates` $\lambda_k$, the expected number of objects in bin $k$,
the data are an inhomogeneous Poisson process. An {term}`exposure` is absorbed into $S$,
which must then be absolute:

$$
p(\lambda\mid D)\propto p(\lambda)\,e^{-S\cdot\lambda}\prod_i(\tilde L_i\cdot\lambda)^{w_i}.
$$

Independent priors $\lambda_k\sim\mathrm{Gamma}(a_k,b_k)$ are conjugate given the labels:
$\lambda_k\mid n\sim\mathrm{Gamma}(a_k+n_k,\,b_k+S_k)$. Fractions and rates are not
interchangeable priors. Independent $\mathrm{Gamma}(a_k,b)$ rates imply
$\rho\sim\mathrm{Dir}(a)$ only when every bin has the same rate parameter $b$ and there
is no selection.

**Inferred selections and responses.** A selection on the truth measured by
{term}`injections` can be inferred with the population. The unconstrained coordinates
then gain $\mathrm{logit}\,S_k$, with each $S_k$'s posterior given the injections as its
prior ([injections](../guides/selection.md#selection-measured-by-injections)). A
{term}`response matrix` tallied from simulated events is inferred the same way, with a
Dirichlet posterior for each of its columns
([binned measurements](../guides/inputs.md#binned-measurements-and-classifiers)).

## Assumptions

Every result of binference rests on these assumptions. The pages linked from each one
show how to check it or what to do when it fails.

1. **Independent objects.** Given the population, the objects' true values are
   independent draws from it. Each object's data depend only on its own true value.
2. **Correct likelihoods.** $\ell_i(z)$ is the probability of the observed data given
   $z$. With a selection, it is not renormalised over the data that could have been
   detected.
3. **Bins that cover the true values.** Every true value lies in one of the bins.
   Likelihood outside the bins is dropped, and binference warns when many objects lose
   most of theirs ([warnings](../reference/warnings.md)).
4. **A flat population within each bin.** The population density is constant within
   each bin in the measure coordinate $u$. A {term}`basis` replaces this with fixed
   shapes inside the bins
   ([basis](../guides/inputs.md#within-bin-shape-and-internal-resolution-basis)). When
   the assumption fails, the result has a {term}`binning bias` (below).
5. **A known interim prior.** Posterior inputs were computed under the interim prior
   given, and that prior is roughly constant within each bin.
6. **Exact bin averages.** The model treats $L_{ik}$ as exact. Likelihoods estimated from
   {term}`posterior samples` carry Monte Carlo noise, which `result.check()` measures
   ([posterior samples](../guides/inputs.md#posterior-samples)).
7. **A selection of the stated kind.** The probability of entering the catalogue depends
   only on the true bin (`"on_truth"`) or only on the measured data
   (`"on_measurement"`). It is known, or inferred from injections or tallies.
8. **Poisson numbers of objects.** With a selection, the number of detected objects is
   Poisson given an unknown total with the prior $1/\Lambda$. With rates, the objects
   form a Poisson process with expected numbers $S_k\lambda_k$.
9. **A transferable response.** For {term}`binned measurements <binned measurement>`,
   the response matrix describes the observed objects as well as the events it was
   built from. For classifiers this is {term}`label shift`.
10. **Integer weights.** Integer weights count identical objects. Non-integer weights
    define a weighted pseudo-likelihood, whose posterior is not a proper posterior.

### Binning bias

Assumption 4 is a modelling choice. Suppose the population is not flat within the bins,
and the bins are coarse compared with the measurement errors. Then {eq}`eq-L` weights
the wrong parts of each bin. The resulting bias does not shrink as $N$ grows, so it
eventually dominates the error bars. binference's slow tests fit a continuous bimodal
population on six bins with errors of 0.6 to 1.2 bin widths. At $N=20\,000$ the error
of the worst fraction exceeds four posterior standard deviations.

Use bins narrower than the features you care about, and a smoothness prior to control
the noise. Report coarse summaries as functionals of the fine posterior (`result.rebin`,
`result.functional`). `result.binning_bias()` estimates the bias of a fit.
{cite:t}`stanley2022` reach the same conclusion for unfolding.
