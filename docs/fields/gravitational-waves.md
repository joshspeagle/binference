# Gravitational-wave populations

The LIGO–Virgo–KAGRA (LVK) catalogues list compact-binary mergers whose masses, spins
and redshifts are each measured with large, correlated errors. Population studies infer
the distribution of these parameters, and the merger rate, from the whole catalogue.
The field has the best-engineered software of all the fields on these pages, and its
practice for Monte Carlo error is the most developed.

## The problem in gravitational-wave terms

Each event $i$ comes as $S_i$ posterior samples $\theta_{is}$ from parameter estimation
(PE), drawn under a PE prior $\pi_\text{PE}$. The population model $p(\theta\mid\Lambda)$
has hyperparameters $\Lambda$. Detection depends on the data, through a
signal-to-noise or false-alarm threshold, and its probability is measured with
*injections*: simulated signals drawn from a known density $p_\text{draw}$ and passed
through the search pipelines. The hierarchical likelihood used throughout the field is

$$
\ln\mathcal L(\Lambda)\approx\sum_{i=1}^{N}\ln\Big[\frac{1}{S_i}\sum_{s}
\frac{p(\theta_{is}\mid\Lambda)}{\pi_\text{PE}(\theta_{is})}\Big]-N\ln\hat P_\text{det}(\Lambda),
\qquad
\hat P_\text{det}(\Lambda)=\frac{1}{N_\text{draw}}\sum_{j\,\text{found}}
\frac{p(\theta_j\mid\Lambda)}{p_\text{draw}(\theta_j)},
$$

or its Poisson-rate form $N\ln R-R\,VT(\Lambda)$ {cite:p}`mandel2019,thrane2019`. Both Monte Carlo sums are noisy, and much of the field's recent work is
about controlling that noise.

"Weakly modelled" analyses bin the parameter space. The 2016 LVK code BinnedCBCMasses
takes a per-event weight matrix, $L_{ik}$ by another name, and puts a Gaussian-process
prior on the log-rates in the bins; PixelPop {cite:p}`heinzel2025a` puts an intrinsic
conditional autoregressive (ICAR) prior coupling each bin to its nearest neighbours on
$10^4$ bins; and the binned Gaussian process of {cite:t}`ray2023` runs a Kronecker-factorised
GP on about $8\times10^4$ bins in four dimensions.

## In binference's terms

The {doc}`selection tutorial <../tutorials/11_populations_with_selection>` infers a
mass distribution with a selection measured by injections.

- **Per-event samples** become a likelihood matrix through
  `Likelihoods.from_samples(samples, bins, interim=...)`: each sample is histogrammed
  onto the bins with the PE prior divided out, which is the binned form of the
  per-event term above. The interim prior is a {class}`~binference.Density`; ragged
  sample counts and importance weights are accepted, and each event's effective number
  of samples is stored ({doc}`../guides/inputs/samples`).
- **Selection** acts on the measured data, so it is `selection_mode="on_measurement"`:
  each event's likelihood is used as is and divided by the detectable fraction, with
  $S_k$ the detection probability averaged over bin $k$
  ([theory](../how-it-works/theory.md#selection-effects)). `Selection.from_injections` estimates
  $S_k$ from injections counted per bin, without reweighting by their draw density
  within a bin; for a selection on the measurement it is a plug-in, not inferred with
  the population ({doc}`../guides/selection`).
- **Rates.** A fit of fractions also gives the total number and the rates under the
  scale-free prior on the total (`result.total`, `result.rates`), and `GammaRates` puts
  priors on the rates directly; `scale="exposure"` declares a selection with absolute
  values such as $VT$.
- **Smoothness.** `LogisticRandomWalk(order=1)` on a grid penalises the squared
  differences between neighbouring bins along each axis, the form of PixelPop's ICAR
  prior; `LogisticGP` is a Gaussian process with one length scale per axis, computed in
  a Hilbert-space basis rather than PixelPop's sparse precision or gppop's Kronecker
  factorisation. Their hyperparameters are inferred by default.
- **Population-informed event posteriors**, the field's "reweighted event samples", are
  `result.assignment_probabilities` and `result.sample_values`; `Gibbs` gives joint draws
  of the population and every event's bin.

Two differences follow from binning. First, a likelihood given on the bins carries no
Monte Carlo noise at all; only likelihoods built from samples do. Second, the
binned Monte Carlo estimate is linear in $\rho$, so its variance is a cheap quadratic
form in $\rho$: `lk.sample_variance(rho)` returns each object's relative variance
$V_i(\rho)$ (with its effective number of samples), and `result.check()` warns when $\sum_iV_i>1$, the per-event part of the
LVK's variance criterion ({doc}`../guides/inputs/samples`).

## Methods and software

- **gwpopulation** ([ColmTalbot/gwpopulation](https://github.com/ColmTalbot/gwpopulation);
  {cite:alp}`talbot2025`) is the LVK reference implementation of the likelihood above,
  with NumPy, CuPy and JAX backends. A population model is a plain function of a
  dictionary of arrays and named hyperparameters; the PE prior is a `prior` column
  divided out of each event's samples; `ResamplingVT` estimates the detection
  efficiency from injections and returns its Monte Carlo variance; and the likelihood
  reports per-event variances and can reject points whose total variance exceeds a
  threshold. It is coupled to bilby and wcosmo and switches array backends through
  global state.
- **bilby.hyper** (bilby) provides a domain-neutral version with no selection term and
  no Monte Carlo variance.
- **GWInferno**, **icarogw**, **FIGARO**, **GWKokab** and **Vamana** re-implement the
  same estimator in NumPyro, CuPy or NumPy with different model families: B-splines with
  difference penalties (GWInferno), cosmology with galaxy catalogues (icarogw), Dirichlet-
  process Gaussian mixtures (FIGARO; {cite:alp}`rinaldi2022`), parametric models with
  flow-enhanced MCMC (GWKokab), and Gaussian mixtures with Dirichlet weights (Vamana).
  GWInferno, icarogw and gppop are not on PyPI.
- **Binned models.** [BinnedCBCMasses](https://git.ligo.org/RatesAndPopulations/BinnedCBCMasses)
  (Stan; last commit 2019) marginalises a log-normal uncertainty in each bin's
  sensitivity. [PixelPop](https://git.ligo.org/jack.heinzel/pixelpop) runs NUTS in
  NumPyro on GPUs over $10^4$ ICAR-coupled bins, can marginalise the ICAR scale
  analytically, and needs an explicit initialisation because the ICAR prior is improper.
  [gppop](https://github.com/AnaryaRay1/gppop) runs Pyro NUTS over about $8\times10^4$
  bins on four GPUs, with a whitened Kronecker GP, taking a per-bin weight matrix as
  input; its test directory is empty.
- **{cite:t}`callister2024`** model the merger rate as an autoregressive process
  evaluated at every PE sample and injection, one latent per sample.
- **popsummary** ([RatesAndPopulations/popsummary](https://git.ligo.org/RatesAndPopulations/popsummary))
  is the LVK's HDF5 result format: hyperparameter samples, reweighted event samples,
  reweighted injections, population draws and rates on grids.
- **population-error** ([jack-heinzel/population-error](https://github.com/jack-heinzel/population-error))
  computes the information lost to Monte Carlo approximation in post-processing
  {cite:p}`heinzel2025b`.
- **Neural methods** appear as flows for the population density {cite:p}`ruhe2022,iampieri2026`, flow-based variational posteriors over
  hyperparameters {cite:p}`mould2025`, emulators of population synthesis and of
  the detection probability, and amortised simulation-based inference {cite:p}`leyde2024`. The {doc}`neural` page covers the first.
- **Joint sampling.** {cite:t}`mancarella2025` sampled all GWTC-3 event parameters and
  the population hyperparameters together, "a parameter space of about 500
  dimensions", to obtain population-informed event posteriors and their correlations
  with the population.

## How binference relates

**Shared.** Binned GW models are binference's model with a softmax or Poisson-rate link:
BinnedCBCMasses, PixelPop and gppop all evaluate $\sum_i\ln\sum_kL_{ik}\rho_k$ (or its
rate form) from precomputed per-bin weights. The interim-prior correction is the same,
and binference's `on_measurement` selection is the field's normalisation by the
detection probability.

**Added.** A domain-neutral interface: no PE-specific coordinate conventions, cosmology
or injection-file format. Exact Gibbs sampling of the population with every event's bin,
which in the binned case needs no high-dimensional HMC. A Monte Carlo variance check
that applies only when samples are used, and inputs on bins that need none.

**Not done.** binference has no GPU backend and has not been run at PixelPop's or
gppop's numbers of bins. It does not fit parametric population models such as power laws
or mixtures in continuous parameters, does not reweight injections by their draw density
within a bin, does not taper or cut the likelihood by its Monte Carlo variance during
sampling (it warns afterwards), does not infer a selection on the measurement from
injections (it is a plug-in there), and does not write popsummary files.

## Pitfalls known in gravitational-wave population inference

- **Monte Carlo error in the likelihood.** Selection Monte Carlo is "the limiting factor
  in our ability to probe narrow population models" {cite:p}`talbot2023`. Thresholds
  on each Monte Carlo integral's effective sample size "are not necessarily safe", while
  thresholds on the variance of the total log-likelihood are; the plug-in likelihood is
  biased but correctable, and the information lost should stay below about 0.2 bits
  {cite:p}`heinzel2025b`. GWTC-4 {cite:p}`lvk2026a` rejects hyperparameter points whose log-likelihood
  variance exceeds 1.
- **Cuts are not innocent.** In GWTC-4, changing the cuts changed the inferred $\chi_p$
  distribution, and for B-spline models the variance limit "was the main driver of
  smoothness".
- **PE priors that vanish where the population has mass.** A uniform, isotropic spin
  prior gives a $\chi_p$ prior that goes to zero at $\chi_p=0$ and 1, producing
  population features that "are not necessarily astrophysical in origin" (GWTC-4; {cite:alp}`lvk2026a`).
- **Smoothing hyperparameters** are poorly identified: GWTC-4 fixed the B-spline penalty
  by hand after it "consistently railed against prior boundaries", and noted that rates
  in some binned-GP bins "may be driven by the a priori assumption of a Gaussian
  covariance, and not the data".
- **Unregularised nonparametric fits** collapse onto the samples: the population that
  maximises the likelihood over all distributions is "a linear superposition of delta
  functions" {cite:p}`payne2023`.
- **Regions without data.** GWTC-5 {cite:p}`lvk2026b` hatches regions with under 1% of PE samples as "parts
  of parameter space where the models may not be actively constrained by the data";
  binference's `result.data_support` gives the corresponding number per bin.
- **Ragged sample counts.** Downsampling every event to the smallest event's count
  "discards samples from well-measured events" (PixelPop's migration note); binference's
  `from_samples` takes ragged lists.
