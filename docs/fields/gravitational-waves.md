# Gravitational-wave populations

How do gravitational-wave astronomers infer the population of merging black holes and
neutron stars? The LIGO–Virgo–KAGRA (LVK) catalogues list mergers whose masses, spins and
redshifts carry large, correlated errors. Population studies infer the distribution of
these parameters, and the merger rate, from the whole catalogue. Its software and its
handling of Monte Carlo error are the most developed on these pages.

## The problem in gravitational-wave terms

Each event $i$ comes as $S_i$ posterior samples $\theta_{is}$ from parameter estimation
(PE). The samples were drawn under a PE prior $\pi_\text{PE}$. The population model
$p(\theta\mid\Lambda)$ has hyperparameters $\Lambda$. Detection depends on the data,
through a signal-to-noise or false-alarm threshold. Its probability is measured with
injections: simulated signals drawn from a known density $p_\text{draw}$ and passed
through the search pipelines. The field's hierarchical likelihood is

$$
\ln\mathcal L(\Lambda)\approx\sum_{i=1}^{N}\ln\Big[\frac{1}{S_i}\sum_{s}
\frac{p(\theta_{is}\mid\Lambda)}{\pi_\text{PE}(\theta_{is})}\Big]-N\ln\hat P_\text{det}(\Lambda),
\qquad
\hat P_\text{det}(\Lambda)=\frac{1}{N_\text{draw}}\sum_{j\,\text{found}}
\frac{p(\theta_j\mid\Lambda)}{p_\text{draw}(\theta_j)} .
$$

Its Poisson-rate form is $N\ln R-R\,VT(\Lambda)$ {cite:p}`mandel2019,thrane2019`. Both
Monte Carlo sums are noisy. Much of the field's recent work controls that noise.

"Weakly modelled" analyses bin the parameter space. The 2016 LVK code BinnedCBCMasses
takes a per-event weight matrix, which is $L_{ik}$. It puts a Gaussian-process prior on
the log-rates in the bins. PixelPop {cite:p}`heinzel2025a` couples each of $10^4$ bins to
its neighbours with an intrinsic conditional autoregressive (ICAR) prior. The binned
Gaussian process of {cite:t}`ray2023` covers about $8\times10^4$ bins in four dimensions.

## In binference's terms

| gravitational-wave term | binference |
|---|---|
| PE samples and the PE prior | `Likelihoods.from_samples(samples, bins, interim=...)`, with the PE prior as a {class}`~binference.Density`; ragged and weighted |
| detection probability, measured by injections | `selection_mode="on_measurement"`, $S_k$ from `Selection.from_injections` |
| merger rate; $VT$ | `result.rates`, `result.total`, `priors.GammaRates`; `scale="exposure"` |
| ICAR prior on binned rates | `LogisticRandomWalk(order=1)` on a grid |
| binned Gaussian process | `LogisticGP`, one length scale per axis |
| reweighted event samples | `result.assignment_probabilities()`, `result.sample_values()` |
| joint draws of population and events | the `Gibbs` engine, `result.sample_assignments()` |

`from_samples` histograms each event's samples with the PE prior divided out. This is
the binned form of the per-event term above. Because selection acts on the data, the fit
divides each likelihood by the detectable fraction
([theory](../how-it-works/theory.md#selection-effects)). `Selection.from_injections`
counts injections per bin, without reweighting by their draw density. For a selection
on the measurement it is a fixed plug-in
([selection guide](../guides/selection.md#selection-measured-by-injections)).

A likelihood given on bins carries no Monte Carlo noise. For likelihoods from samples,
`lk.sample_variance(rho)` returns each event's relative variance $V_i(\rho)$.
`result.check()` warns when $\sum_iV_i$ exceeds a [threshold](../reference/defaults.md),
the per-event part of the LVK's variance criterion.

The {doc}`selection tutorial <../tutorials/11_populations_with_selection>` infers a
mass distribution with a selection measured by injections.

## Methods and software

- **gwpopulation** ([ColmTalbot/gwpopulation](https://github.com/ColmTalbot/gwpopulation);
  {cite:alp}`talbot2025`) is the LVK reference implementation of the likelihood above,
  with NumPy, CuPy and JAX backends. The PE prior is a `prior` column divided out of each
  event's samples. `ResamplingVT` estimates the detection efficiency from injections,
  with its Monte Carlo variance.
- **bilby.hyper** gives a domain-neutral version with no selection term.
- **GWInferno**, **icarogw**, **FIGARO**, **GWKokab** and **Vamana** re-implement the
  same estimator. Their model families are B-splines, cosmology with galaxy catalogues,
  Dirichlet-process Gaussian mixtures {cite:p}`rinaldi2022`, parametric models, and
  Gaussian mixtures with Dirichlet weights.
- **Binned models.**
  [BinnedCBCMasses](https://git.ligo.org/RatesAndPopulations/BinnedCBCMasses) (Stan; last
  commit 2019) marginalises a log-normal uncertainty in each bin's sensitivity.
  [PixelPop](https://git.ligo.org/jack.heinzel/pixelpop) runs NUTS in NumPyro on GPUs.
  It needs an explicit initialisation, because the ICAR prior is improper.
  [gppop](https://github.com/AnaryaRay1/gppop) runs Pyro NUTS on four GPUs with a
  whitened Kronecker GP. It takes a per-bin weight matrix, and its test directory is
  empty.
- **{cite:t}`callister2024`** model the merger rate as an autoregressive process with one
  latent per PE sample and injection.
- **popsummary**
  ([RatesAndPopulations/popsummary](https://git.ligo.org/RatesAndPopulations/popsummary))
  is the LVK's HDF5 result format. **population-error**
  ([jack-heinzel/population-error](https://github.com/jack-heinzel/population-error))
  computes the information lost to Monte Carlo approximation {cite:p}`heinzel2025b`.
- **Neural methods** include flows for the population density
  {cite:p}`ruhe2022,iampieri2026`, flow-based variational posteriors {cite:p}`mould2025`
  and amortised inference {cite:p}`leyde2024`. See {doc}`neural`.
- **Joint sampling.** {cite:t}`mancarella2025` sampled all GWTC-3 event parameters and
  the population together, "a parameter space of about 500 dimensions".

## How binference relates

**Shared.** Binned GW models are binference's model with a softmax or Poisson-rate
link. BinnedCBCMasses, PixelPop and gppop all evaluate $\sum_i\ln\sum_kL_{ik}\rho_k$ from
precomputed weights. The interim-prior correction is the same. binference's
`on_measurement` selection is the field's normalisation by the detection probability.

**Added.** A domain-neutral interface. Exact Gibbs sampling of the population with
every event's bin. A Monte Carlo variance check for inputs from samples.

**Not done.** binference has no GPU backend and has not run at PixelPop's or gppop's
scale. It fits no parametric models such as power laws. It does not taper or
cut the likelihood by its Monte Carlo variance during sampling; it warns afterwards. It
does not infer a selection on the measurement from injections. It does not write
popsummary files.

## Pitfalls known in gravitational-wave population inference

- **Monte Carlo error.** Selection Monte Carlo is "the limiting factor in our ability to
  probe narrow population models" {cite:p}`talbot2023`. Thresholds on each integral's
  effective sample size "are not necessarily safe". Thresholds on the variance of the
  total log-likelihood are; the information lost should stay below about 0.2 bits
  {cite:p}`heinzel2025b`. GWTC-4 rejects points whose log-likelihood variance exceeds 1
  {cite:p}`lvk2026a`.
- **Cuts are not innocent.** For GWTC-4's B-spline models the variance limit "was the
  main driver of smoothness".
- **PE priors that vanish.** A uniform, isotropic spin prior goes to zero at $\chi_p=0$
  and 1. It produced features that "are not necessarily astrophysical in origin"
  {cite:p}`lvk2026a`.
- **Smoothing hyperparameters** are poorly identified. GWTC-4 fixed the B-spline penalty
  by hand after it "consistently railed against prior boundaries". Rates in some
  binned-GP bins "may be driven by the a priori assumption of a Gaussian covariance, and
  not the data".
- **Unregularised fits** collapse onto the samples. The maximum-likelihood population is
  "a linear superposition of delta functions" {cite:p}`payne2023`.
- **Regions without data.** GWTC-5 {cite:p}`lvk2026b` hatches regions with under 1% of PE
  samples. They are "parts of parameter space where the models may not be actively
  constrained by the data". binference's `result.data_support` gives the matching number
  per bin.
- **Ragged sample counts.** Downsampling every event to the smallest count "discards
  samples from well-measured events" (PixelPop's migration note). `from_samples` takes
  ragged lists.
