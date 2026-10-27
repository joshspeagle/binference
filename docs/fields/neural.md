# Neural and simulation-based deconvolution

Can a neural network learn a population from noisy objects without bins? Normalising
flows, diffusion and flow-matching models represent densities in several continuous
dimensions. Since 2019 a line of work trains them on noisy, heteroscedastic per-object
data. The network then learns the population, not the noise-broadened distribution of
the measurements. binference has no neural models. This page maps the methods onto its
model and lists their known pitfalls.

## The problem in machine-learning terms

Each object has data $d_i$ of a true value $z$ with known noise. A network gives the
population density $p_\text{net}(z)$. Maximum marginal likelihood needs, for every
object, the integral $p(d_i)=\int p(d_i\mid z)\,p_\text{net}(z)\,dz$. Methods estimate
it by Monte Carlo, with $n_\text{draw}$ draws per object. The draws depend on what each
object provides:

- **a likelihood you can draw from**: average the density over draws from the
  likelihood {cite:p}`yan2025`;
- **posterior samples under an interim prior** $\pi$: average $p_\text{net}(z)/\pi(z)$
  over the samples, as in {cite:t}`hogg2010` and gravitational-wave flows
  {cite:p}`ruhe2022`;
- **a simulator**: average the likelihood over draws from the population model
  {cite:p}`vandegar2021`, or train a recognition network for an importance-weighted
  bound {cite:p}`dockhorn2020`.

The methods form four families:

1. **Flows fitted by marginal likelihood**: {cite:t}`dockhorn2020`, Neural Empirical
   Bayes {cite:p}`vandegar2021`, and {cite:t}`yan2025` on Gaia velocities.
2. **Neural empirical Bayes on a grid**: Neural-g {cite:p}`wang2024` maps fixed grid
   points through a network with a softmax output. It trains on exactly binference's
   likelihood $\sum_i\log\sum_kL_{ik}\rho_k$.
3. **Diffusion and flow-matching priors learned from corrupted observations**. They use
   EM with posterior sampling (DiEM; {cite:alp}`rozet2024`) or corruption-aware
   objectives, and they target images.
4. **Likelihood-free population fits** compare simulated and observed data. They use
   optimal transport (pop-cosmos; {cite:alp}`alsing2024`), the Wasserstein distance
   (PopSED) or the maximum mean discrepancy. Amortised population inference
   {cite:p}`leyde2024` also belongs here.

## In binference's terms

| machine-learning term | binference |
|---|---|
| flexible population density | a smoothness prior on $\log\rho$: `LogisticRandomWalk`, `LogisticGP` |
| continuous density | {class}`~binference.Basis` with `order=1`, piecewise linear |
| posterior samples with the interim prior divided out | `Likelihoods.from_samples(..., interim=...)` |
| Neural-g's softmax over grid points | `LogisticNormal` on `Bins.categories` or a grid |

On bins, a flexible density needs no Monte Carlo. Any smooth function of the fractions
enters the exact likelihood, as Neural-g's network does. binference's smoothness priors
infer their hyperparameters, and NUTS samples the density's posterior
({doc}`priors guide <../guides/priors>`). `from_samples` histograms the samples once,
not at every training step.

Neural densities may be worth their cost in three or more continuous dimensions.
binference's grids work there too, but the number of bins is the product of the bins
per axis.

## Methods and software

The [comparison table](software.md#comparison) gives each tool's language, licence and
status.

- **density-deconvolution** is the code of {cite:t}`dockhorn2020`, a flow trained by
  amortised variational inference.
- **NEB** offers direct Monte Carlo, a de-biased Russian-roulette estimator and two
  bounds.
- **Denoising-Milky-Way-with-NF** is the code of {cite:t}`yan2025`, a spline flow in
  zuko.
- **neuralG** wraps Neural-g for R. **diem** is DiEM's experiment code.
- **Infrastructure.** flowjax (JAX) and zuko (PyTorch) are flow libraries. pzflow offers
  tabular flows with error columns.

None of the deconvolution code bases surveyed checks its uncertainty for coverage.

## How binference relates

binference shares the marginal likelihood. On bins, every family above reduces to
$\prod_i\sum_kL_{ik}\rho_k$. The interim-prior division for posterior samples is the
same.

For populations that bins can represent, binference adds an exact likelihood without
Monte Carlo bias. It also adds a posterior over the smooth density.

Not done:

- binference has no flows, diffusion or flow-matching models.
- It has no importance-weighted training objectives.
- It has no unbinned densities.
- It does no simulation-based inference.

## Pitfalls known in neural deconvolution

- **Monte Carlo objectives are biased.** The log of a Monte Carlo average is biased low
  by $O(1/n_\text{draw})$. A Russian-roulette estimator removes the bias. In Neural
  Empirical Bayes tests, direct Monte Carlo with 1024 draws beat the ELBO and the
  importance-weighted bound {cite:p}`vandegar2021`.
- **Bounds are loose.** "The variational bound is not tight, and biases us towards worse
  prior models", with high variance between runs {cite:p}`dockhorn2020`.
- **Selection makes training unstable.** The selection term enters with the opposite
  bias, so the objective is no longer a bound. {cite:t}`ruhe2022` dropped selection,
  because "Optimization was highly unstable when including the selection function".
- **Noise augmentation is not deconvolution.** pzflow's `convolve_errs` trains on
  extra-noisy draws of the data. A flow fitted to pooled posterior samples, with no
  interim prior divided out, learns the noise-broadened, prior-weighted density.
- **Identifiability.** "Distinct source distributions may result in the same
  distribution over observations" {cite:p}`vandegar2021`. {cite:t}`rozet2024` call
  empirical Bayes "an ill-posed problem". A conservative initial prior "is therefore
  preferable for scientific applications".
- **Small samples.** Spline flows show gridded artefacts at small $N$
  {cite:p}`yan2025`. Flows can collapse to point masses with very few observations
  {cite:p}`vandegar2021`.
- **Extrapolation.** Flows extrapolate beyond the samples and injections. Population
  statements should stay in supported regions {cite:p}`iampieri2026`.
- **Uncertainty.** Authors report point estimates, at most with the scatter between five
  training runs. {cite:t}`ruhe2022` ran Hamiltonian Monte Carlo over the weights of a
  90-parameter flow and judged it "not scalable". Amortised neural posteriors are often
  overconfident {cite:p}`hermans2022`.
- **Cost.** {cite:author}`yan2025`'s training took 4 to 16 hours on one T4 GPU. GPU
  extreme deconvolution took about one hour.
- **Training losses.** For conditional flow matching the loss can plateau while
  physical fidelity still improves (ScatterPrism; {cite:alp}`xia2026`).
