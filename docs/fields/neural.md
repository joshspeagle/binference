# Neural and simulation-based deconvolution

Neural density models (normalising flows, diffusion and flow-matching models) can
represent a population density in several continuous dimensions without bins. A line of
work since 2019 trains them on noisy, heteroscedastic per-object data, so that the
network learns the population rather than the noise-broadened distribution of the
measurements. binference has no neural models; this page describes the methods, what
they share with binference's model, and what is known about their pitfalls.

## The problem in machine-learning terms

Each object has an observation $w_i=v_i+n_i$ of a true value $v_i$ with known noise, and
the population density $p_\theta(v)$ is a network with parameters $\theta$. Fitting by
maximum marginal likelihood needs, for every object, the convolution integral
$p(w_i\mid\theta)=\int p(w_i\mid v)\,p_\theta(v)\,dv$, which is estimated by Monte Carlo.
How it is estimated depends on what each object provides:

- **a likelihood that is easy to sample**: average the density over draws from the
  likelihood, $\frac1K\sum_jp_\theta(v_{ij})$ {cite:p}`yan2025`;
- **posterior samples under an interim prior** $\pi_0$: average
  $p_\theta(v_{ij})/\pi_0(v_{ij})$ over the samples, as in {cite:t}`hogg2010` and
  in gravitational-wave flows {cite:p}`ruhe2022`;
- **a simulator**: average the likelihood over draws from the population model itself
  {cite:p}`vandegar2021`, or train a recognition network for an importance-weighted
  bound {cite:p}`dockhorn2020`.

The census groups the methods in four families:

1. **Flows fitted by marginal likelihood**: {cite:t}`dockhorn2020`,
   Neural Empirical Bayes {cite:p}`vandegar2021`, and {cite:t}`yan2025` on Gaia
   velocities.
2. **Neural empirical Bayes on a grid**: Neural-g {cite:p}`wang2024` maps fixed grid
   points through a network with a softmax output and trains it on exactly binference's
   likelihood $\sum_i\log\sum_kL_{ik}\rho_k$.
3. **Diffusion and flow-matching priors learned from corrupted observations**, by EM
   with posterior sampling (DiEM; {cite:alp}`rozet2024`) or by corruption-aware objectives
   (ambient diffusion; noise-robust conditional flow matching), aimed at images.
4. **Likelihood-free population fits** that compare simulated and observed data:
   optimal transport (pop-cosmos; {cite:alp}`alsing2024`), the Wasserstein distance
   (PopSED), and the convolutional maximum mean discrepancy. Hierarchical neural
   posterior estimation and amortised population inference {cite:p}`leyde2024` belong
   here too.

## In binference's terms

On bins, a flexible density needs no Monte Carlo: any smooth parameterisation of the
bin fractions $\rho(\theta)$ plugs into the exact likelihood, which is what Neural-g does
with a network. binference's smooth parameterisations are the logistic priors, a
Gaussian random walk or a Gaussian process on $\log\rho$ whose hyperparameters are
inferred, sampled with NUTS so that the density comes with a posterior
({doc}`../guides/priors`); `binference.Basis` with `order=1` gives a continuous,
piecewise-linear density on refined components ({doc}`../guides/inputs/advanced`). Posterior
samples enter through `Likelihoods.from_samples`, which histograms them once with the
interim prior divided out, rather than re-estimating a Monte Carlo average at every
step of training.

The Monte Carlo estimators above are needed only when per-object information comes as
samples or as continuous likelihoods in several dimensions, where bins are expensive.
That is where neural densities may earn their cost: populations in three or more
continuous dimensions. binference's grids work in any number of dimensions, but the
number of bins grows as the product of the bins per axis.

## Methods and software

- **density-deconvolution** ([bayesiains/density-deconvolution](https://github.com/bayesiains/density-deconvolution)),
  the code of {cite:t}`dockhorn2020`: a masked autoregressive flow trained by amortised
  variational inference with the ELBO or an importance-weighted bound. It has no licence
  and a single 2020 commit.
- **NEB** ([MaximeVandegar/NEB](https://github.com/MaximeVandegar/NEB); MIT notebooks,
  last touched 2022): direct Monte Carlo, a de-biased Russian-roulette estimator, the
  ELBO and an importance-weighted bound behind a one-line estimator interface.
- **Denoising-Milky-Way-with-NF** ([Ziyang-Yan/Denoising-Milky-Way-with-NF](https://github.com/Ziyang-Yan/Denoising-Milky-Way-with-NF);
  MIT), the code of {cite:t}`yan2025`: a spline flow in zuko, trained with fresh
  likelihood draws every epoch after pre-training on the noisy points. It has no
  packaging and no tests.
- **neuralG** ([shijiew97/neuralG](https://github.com/shijiew97/neuralG)): an R wrapper
  around PyTorch for Neural-g, without tests.
- **diem** ([francois-rozet/diem](https://github.com/francois-rozet/diem); MIT, JAX):
  DiEM's experiment code, which needs a Slurm cluster.
- **pop-cosmos** and **PopSED** fit galaxy populations to photometry without per-object
  likelihoods; CondXD outputs Gaussian-mixture parameters from a network trained with
  the extreme-deconvolution likelihood.
- **Infrastructure.** [flowjax](https://github.com/danielward27/flowjax) (JAX, MIT)
  accepts arbitrary training losses and works with NumPyro;
  [zuko](https://github.com/probabilists/zuko) (PyTorch, MIT) is the maintained PyTorch
  flow library; [pzflow](https://github.com/jfcrenshaw/pzflow) (JAX) offers tabular flows
  with error columns and ensembles; NumPyro supplies NUTS for joint sampling; sbi and
  BayesFlow supply simulation-based inference. nflows has had no release since 2020, and
  Meta's flow_matching is licensed CC BY-NC 4.0, which forbids commercial use.

None of the deconvolution code bases surveyed is packaged with a test suite, several have
no licence, and none reports calibrated uncertainty on the learned density.

## How binference relates

**Shared.** The marginal likelihood: on bins, every family above reduces to
$\prod_i\sum_kL_{ik}\rho_k(\theta)$, and the interim-prior division for posterior
samples is the same.

**Added.** For populations that bins can represent, binference gives an exact
likelihood without Monte Carlo bias and a posterior over the smooth density, checked
with simulation-based calibration ({doc}`../guides/checking`).

**Not done.** binference has no flows, diffusion or flow-matching models, no
importance-weighted training objectives, no unbinned densities in many dimensions, and
no simulation-based inference.

## Pitfalls known in neural deconvolution

- **Monte Carlo objectives are biased.** The log of a Monte Carlo average is biased low
  by $O(1/K)$ in the number of samples $K$; a Russian-roulette estimator removes the
  bias at small $K$, and in Neural Empirical Bayes tests direct Monte Carlo with
  $K=1024$ beat the ELBO and the importance-weighted bound {cite:p}`vandegar2021`.
  "The variational bound is not tight, and biases us towards worse prior models", with
  high variance between runs {cite:p}`dockhorn2020`.
- **Selection makes training unstable.** The selection term enters with the opposite
  bias, so the objective is no longer a bound; {cite:t}`ruhe2022` dropped selection
  because "Optimization was highly unstable when including the selection function".
- **Noise augmentation is not deconvolution.** pzflow's `convolve_errs` option trains on
  extra-noisy draws of the data; it does not maximise the marginal likelihood. Fitting a
  flow to pooled posterior samples without dividing out the interim prior learns the
  noise-broadened, prior-weighted density, not the population.
- **Identifiability.** "Distinct source distributions may result in the same
  distribution over observations" {cite:p}`vandegar2021`, and "empirical Bayes is an
  ill-posed problem", so a conservative initial prior "is therefore preferable for
  scientific applications" {cite:p}`rozet2024`.
- **Small samples.** Spline flows show gridded artefacts at small $N$, controlled mainly
  by the number of spline bins {cite:p}`yan2025`; flows can collapse to point masses
  with very few observations {cite:p}`vandegar2021`; flows could not represent an
  expected sharp edge {cite:p}`ruhe2022`.
- **Extrapolation.** Flows extrapolate beyond the support of the samples and injections;
  population statements should be restricted to supported regions {cite:p}`iampieri2026`.
- **Uncertainty.** Results are reported as point estimates, at most with scatter between
  five training runs. Hamiltonian Monte Carlo over the weights worked for a flow with 90
  parameters and was judged "not scalable" {cite:p}`ruhe2022`, and amortised neural
  posteriors are often overconfident {cite:p}`hermans2022`.
- **Cost.** {cite:author}`yan2025`'s training took 4 hours at $K=32$ and 16 hours at $K=128$ on
  one T4 GPU, against about one hour for GPU extreme deconvolution.
- **Training losses.** For conditional flow matching the training loss can plateau while
  physical fidelity is still improving (ScatterPrism; {cite:alp}`xia2026`).
