# Gravitational-wave populations

How do gravitational-wave astronomers infer the population of merging black holes and
neutron stars? The LIGO–Virgo–KAGRA (LVK) catalogues list mergers whose masses, spins and
redshifts carry large, correlated errors. Population studies infer the distribution of
these parameters, and the merger rate, from the whole catalogue. The field has the most
developed software and the most careful treatment of Monte Carlo error on these pages.

## The problem in gravitational-wave terms

This page uses the field's symbols. $\theta$ denotes an event's parameters, $\Lambda$ the
population's hyperparameters, and $R$ the merger rate.

Parameter estimation (PE) gives each event $i$ a set of $S_i$ posterior samples
$\theta_{is}$, drawn under a PE prior $\pi_\text{PE}$. The population model is
$p(\theta\mid\Lambda)$. An event is detected when its signal-to-noise or false-alarm
statistic passes a threshold. The field measures the detection probability with
injections. These are $N_\text{draw}$ simulated signals, drawn from a known density
$p_\text{draw}$ and passed through the search pipelines. The field's hierarchical
likelihood is

$$
\ln\mathcal L(\Lambda)\approx\sum_{i=1}^{N}\ln\Big[\frac{1}{S_i}\sum_{s}
\frac{p(\theta_{is}\mid\Lambda)}{\pi_\text{PE}(\theta_{is})}\Big]-N\ln\hat P_\text{det}(\Lambda),
\qquad
\hat P_\text{det}(\Lambda)=\frac{1}{N_\text{draw}}\sum_{j\,\text{found}}
\frac{p(\theta_j\mid\Lambda)}{p_\text{draw}(\theta_j)} .
$$

Its Poisson-rate form is $N\ln R-R\,VT(\Lambda)$, where $VT$ is the sensitive
volume–time {cite:p}`mandel2019,thrane2019`. Both Monte Carlo sums are noisy. Much of
the field's recent work controls that noise.

"Weakly modelled" analyses bin the parameter space. The 2016 LVK code BinnedCBCMasses
takes a per-event weight matrix, which is $L_{ik}$. It puts a Gaussian-process prior on
the log-rates in the bins. PixelPop {cite:p}`heinzel2025a` couples each of $10^4$ bins to
its neighbours with an intrinsic conditional autoregressive (ICAR) prior. The binned
Gaussian process of {cite:t}`ray2023` fits masses and redshift jointly. Its gppop code
covered about $8\times10^4$ bins in four dimensions in a 2026 example.

## In binference's terms

| gravitational-wave term | binference |
|---|---|
| PE samples and the PE prior | `Likelihoods.from_samples(samples, bins, interim=...)`, ragged and weighted |
| detection probability, measured by injections | `Selection.from_injections(..., mode="on_measurement").at_mean()`, a fixed selection on the measurement |
| merger rate; $VT$ | `result.rates`, `result.total`, `priors.GammaRates`; `scale="exposure"` |
| ICAR prior on binned rates | `LogisticRandomWalk(order=1)` on a grid |
| binned Gaussian process | `LogisticGP`, one length scale per axis |
| reweighted event samples | `result.assignment_probabilities()`, `result.sample_values()` |
| joint draws of population and events | the `Gibbs` engine, `result.sample_assignments()` |
| Monte Carlo variance of the per-event term | `lk.sample_variance(rho)`, checked by `result.check()` |

Detection depends on the measured signal, so the selection acts on the measurement, and
binference keeps it fixed. A completeness in the true parameters acts on the truth, and
binference can infer it from injections with the population, as the selection tutorial
does. The {doc}`inputs guide <../guides/inputs/index>` describes posterior samples. The
[selection guide](../guides/selection.md#selection-measured-by-injections) describes
injections and selection on the measurement. The
{doc}`selection tutorial <../tutorials/11_populations_with_selection>` infers a mass
distribution with a selection measured by injections.

## Methods and software

The [comparison table](software.md#comparison) gives each tool's language, licence and
status.

- **gwpopulation** {cite:p}`talbot2025` is the LVK reference implementation of the
  likelihood above. It divides a `prior` column out of each event's samples.
  `ResamplingVT` estimates the detection probability from injections, with its Monte
  Carlo variance.
- **bilby.hyper** gives a domain-neutral version with no selection term.
- **GWInferno**, **icarogw**, **FIGARO**, **GWKokab** and **Vamana** re-implement the
  same estimator. Their model families are B-splines, cosmology with galaxy catalogues,
  Dirichlet-process Gaussian mixtures {cite:p}`rinaldi2022`, parametric models, and
  Gaussian mixtures with Dirichlet weights.
- **Binned models.** BinnedCBCMasses marginalises a log-normal uncertainty in each bin's
  sensitivity. PixelPop runs NUTS on GPUs. It needs an explicit starting point, because
  the ICAR prior is improper. gppop runs NUTS with a whitened Kronecker GP on a per-bin
  weight matrix.
- **{cite:t}`callister2024`** model the merger rate as an autoregressive process with one
  latent per PE sample and injection.
- **popsummary** is the LVK's HDF5 result format. **population-error** computes the
  information lost to Monte Carlo approximation {cite:p}`heinzel2025b`.
- **Neural methods** include flows for the population density
  {cite:p}`ruhe2022,iampieri2026`, flow-based variational posteriors {cite:p}`mould2025`
  and amortised inference {cite:p}`leyde2024`. See {doc}`neural`.
- **Joint sampling.** {cite:t}`mancarella2025` sampled all GWTC-3 event parameters and
  the population together, "a parameter space of about 500 dimensions".

## How binference relates

Binned GW models are binference's model with a softmax or Poisson-rate link.
BinnedCBCMasses, PixelPop and gppop all evaluate $\sum_i\ln\sum_kL_{ik}\rho_k$ from
precomputed weights. binference's `on_measurement` selection is the field's
normalisation by the detection probability.

For gravitational waves, binference adds exact Gibbs sampling of the population with
every event's bin. It also adds a cheap Monte Carlo variance check, because the binned
estimate is linear in $\rho$.

Not done:

- binference has no GPU backend and has not run at PixelPop's or gppop's scale.
- It fits no parametric models such as power laws.
- It does not reweight injections by their draw density within a bin.
- It does not taper or cut the likelihood by its Monte Carlo variance during sampling;
  it warns afterwards.
- It does not infer a selection on the measurement from injections.
- It does not write popsummary files.

## Pitfalls known in gravitational-wave population inference

- **Monte Carlo error.** Selection Monte Carlo is "the limiting factor in our ability to
  probe narrow population models" {cite:p}`talbot2023`. Thresholds on each integral's
  effective sample size "are not necessarily safe" {cite:p}`heinzel2025b`. Thresholds on
  the variance of the total log-likelihood are safe. The information lost should stay
  below about 0.2 bits {cite:p}`heinzel2025b`.
- **Variance cuts.** GWTC-4 rejects points whose log-likelihood variance exceeds 1
  {cite:p}`lvk2026a`. For its B-spline models this limit "was the main driver of
  smoothness".
- **PE priors that vanish.** A uniform, isotropic spin prior goes to zero at $\chi_p=0$
  and 1. It produced features that "are not necessarily astrophysical in origin"
  {cite:p}`lvk2026a`.
- **Smoothing hyperparameters** are poorly identified. GWTC-4 fixed the B-spline penalty
  by hand after it "consistently railed against prior boundaries". Rates in some
  binned-GP bins "may be driven by the a priori assumption of a Gaussian covariance, and
  not the data" {cite:p}`lvk2026a`.
- **Unregularised fits.** The maximum-likelihood population puts all its mass on the
  samples. It is "a linear superposition of delta functions" {cite:p}`payne2023`.
- **Regions without data.** GWTC-5 {cite:p}`lvk2026b` hatches regions with under 1% of PE
  samples. They are "parts of parameter space where the models may not be actively
  constrained by the data". binference's {term}`data support` gives the matching number
  per bin.
- **Ragged sample counts.** Downsampling every event to the smallest count "discards
  samples from well-measured events" (PixelPop's migration note). `from_samples` takes
  ragged lists.
