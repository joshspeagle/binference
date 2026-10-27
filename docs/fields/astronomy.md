# Photometric redshifts and astronomical populations

How do astronomers infer a population from objects measured with large errors? They
need the redshift distribution $n(z)$ of a galaxy sample, the colour–magnitude diagram
of Milky Way stars, planet occurrence rates, luminosity and mass functions. Each object's
quantity carries its own error. {cite:t}`hogg2010` set out the general method. It takes
per-object likelihoods, or posterior samples drawn under a known prior. It then fits
the population as "a kind of heteroscedastic deconvolution", which applies to
"magnitudes, parallaxes, or photometric redshifts". binference implements that method
and its extensions.

## The problem in astronomical terms

**Redshift distributions.** A weak-lensing survey splits its galaxies into tomographic
bins. The cosmological analysis needs each bin's $n(z)$ with its uncertainty. A
photometric-redshift (photo-$z$) code gives each galaxy a $p(z)$. Template codes can
return likelihoods, for example BPZ run without its prior. Machine-learning codes return
posteriors under the redshift distribution of their training sample, an implicit prior.
The common practice sums the $p(z)$, which is called stacking. The hierarchical
alternative puts a prior on the histogram heights of $n(z)$. It infers them together
with each galaxy's redshift {cite:p}`leistedt2016`.

**Stellar populations.** {cite:t}`leistedt2017` modelled the colour–magnitude diagram of
1.4 million Gaia TGAS stars. They used a mixture of fixed kernels with amplitudes on the
simplex. Each star's likelihood in each kernel, marginalised over distance, is computed
once. Gibbs sampling then draws the population and the stars' kernel labels: "4 million
parameters in total, most of which are marginalized out". Resolved stellar populations
are also fitted as binned colour–magnitude ("Hess") diagrams. The counts in each cell
are Poisson, with a mean that is a weighted sum of template diagrams.

**Occurrence rates.** {cite:t}`foremanmackey2014` treated a planet catalogue as an
inhomogeneous Poisson process. Its observable rate is the survey completeness times the
occurrence-rate density. They made the density piecewise constant, with a
Gaussian-process prior on the log step heights. They used per-planet posterior samples
with the interim prior divided out. Luminosity- and mass-function codes use the same
Poisson-process likelihood with a selection function {cite:p}`loredo2004,kelly2008`.

## In binference's terms

| astronomical input | binference |
|---|---|
| per-galaxy likelihoods on a redshift grid | {class}`~binference.Likelihoods` from an array, or `Likelihoods.from_density(..., kind="likelihood")` |
| per-galaxy posteriors under a known prior | `Likelihoods.from_table(P, bins, interim=...)` |
| posterior samples (eccentricities, planet radii) | `Likelihoods.from_samples(samples, bins, interim=...)`, ragged and weighted |
| a measurement with Gaussian errors (parallax, colour) | `Likelihoods.from_gaussian`, with a covariance per object on grids |
| a colour–magnitude diagram | `Bins.grid(...)`, with `mask=` for unphysical cells |
| completeness, or an injection–recovery test | `selection_mode="on_truth"`, `Selection.from_injections` |
| a flux or signal-to-noise cut | `selection_mode="on_measurement"` |
| a spectroscopic reference sample | `priors.Dirichlet.from_counts(counts, strength)`, or a logistic prior with `mean=np.log(reference)` |
| smoothness of $n(z)$ or of a rate density | `priors.LogisticGP`, `priors.LogisticRandomWalk` |
| expected numbers, occurrence rates | `result.rates`, `result.total`, `priors.GammaRates` |

The model needs the bin average of each likelihood. {cite:t}`leistedt2016` and the HSC
Y3 analysis integrate over the bins; RAIL evaluates each PDF at grid points. The
{doc}`inputs guide <../guides/inputs>` says how binference averages each input. Posterior
samples enter through {cite:author}`hogg2010`'s importance-sampling estimate. The
occurrence-rate likelihood is the rate form of the same model {cite:p}`mandel2019`.

The {doc}`quick start <../tutorials/01_quickstart>` fits a redshift distribution. The
{doc}`selection tutorial <../tutorials/05_selection>` fits an occurrence rate.

## Methods and software

### Redshift distributions

- **RAIL** (LSST DESC) is the only maintained, pip-installable, general tool that turns
  per-galaxy $p(z)$ into $n(z)$ samples. Its `VarInfStackSummarizer` is mean-field
  variational Bayes for a flat Dirichlet, binference's baseline model. It applies no
  interim-prior correction. Its originators found that it "yielded underestimated error
  bars, likely due to the restricted covariance structure of the Dirichlet"
  {cite:p}`rau2022`. HSC Y3 put the under-coverage at about 20% {cite:p}`rau2023`.
- **{cite:t}`leistedt2016`** sampled the hierarchical model by Gibbs sampling, in
  [Photoz-tools](https://github.com/ixkael/Photoz-tools) (last push 2020). A 2026 KiDS
  application used only $10^5$ galaxies, because all of them "would be too slow"
  {cite:p}`kyriacou2026`.
- **CHIPPR** {cite:p}`malz2022` is the only photo-$z$ code found that divides out the
  interim prior. Its README calls it "stale" and "a prototype"
  ([aimalz/chippr](https://github.com/aimalz/chippr)).
- **Survey calibration pipelines** get wider errors by other routes. DES draws $n(z)$
  with a three-step Dirichlet ("3sDir") that adds sample variance
  {cite:p}`sanchez2020,myles2021`. HSC Y3 adds clustering cross-correlations
  {cite:p}`rau2020,rau2023`. Euclid corrected stacked posteriors with a spectroscopic
  sample {cite:p}`ilbert2021`.
- **Downstream formats.** Cosmology codes read $n(z)$ ensembles as
  [qp](https://github.com/LSSTDESC/qp) ensembles, CosmoSIS realisation files
  {cite:p}`cordero2022`, or [nz_prior](https://github.com/LSSTDESC/nz_prior) priors.

### frankenz

binference descends from [frankenz](https://github.com/joshspeagle/frankenz) (last push
2018). An audit during binference's design tested its two samplers against exact
posteriors and Geweke "getting it right" tests
([`frankenz_audit.md`](https://github.com/joshspeagle/binference/blob/main/research_notes/Population%20inference%20software%20landscape/frankenz_audit.md)).

- **The hierarchical Gibbs sampler is exact.** It matched the exact posteriors
  (root-mean-square z-score 0.94 over 132 statistics). binference keeps it as `Gibbs`.
- **Gibbs mixes slowly with broad likelihoods.** Its median autocorrelation time grew
  from 7 to 180 sweeps as the likelihood width grew from 0.5 to 4 bin widths. A simple
  HMC stayed at 1–14 ({doc}`engines guide <../guides/engines>`).
- **The "population" sampler is not valid MCMC.** Its acceptance test omits a Hastings
  factor. Its standard deviations were 8–18% too large, and bins near zero collapsed.
- **The "flexible reference" step belongs to no joint model.** It inflated standard
  deviations by a factor of 1.1–1.9 ({doc}`priors guide <../guides/priors>`).
- **Posteriors used as likelihoods** biased $\rho$ by up to 12.6 posterior standard
  deviations. They cut the coverage of 68% intervals to 0.29
  ([pitfalls](../guides/pitfalls.md#feeding-posteriors-in-as-likelihoods)).

### Colour–magnitude diagrams and other densities

- **Starlight** ([ixkael/Starlight](https://github.com/ixkael/Starlight)) is the code of
  {cite:t}`leistedt2017`. It is binference's baseline model almost exactly. Its last
  commit was in 2017, and it was never released as a package.
- **Extreme deconvolution** (XD; {cite:alp}`bovy2011`) fits a Gaussian mixture with free
  means and covariances to data with Gaussian errors. It returns the maximum-likelihood
  mixture, with no posterior uncertainty. The reference C code
  ([jobovy/extreme-deconvolution](https://github.com/jobovy/extreme-deconvolution)) is
  maintained but not on PyPI. Only [pygmmis](https://github.com/pmelchior/pygmmis)
  handles truncation, with a completeness known everywhere. XD with fixed means and
  covariances is binference's model.
- **StarFormationHistories.jl**
  ([cgarling/StarFormationHistories.jl](https://github.com/cgarling/StarFormationHistories.jl);
  Julia) fits Hess-diagram counts with a Poisson likelihood ratio. It offers a MAP fit
  with inverse-Hessian errors, HMC and MCMC. Version 2.0.0 added fitted background
  templates, because "contaminants such as background galaxies and foreground stars
  otherwise bias the fit". PANCAKE {cite:p}`zheng2025` fits Poisson Hess diagrams with a
  mean-squared-error loss.

### Occurrence rates and distribution functions

- **exopop** ([dfm/exopop](https://github.com/dfm/exopop)) is the code of
  {cite:t}`foremanmackey2014`. It combines completeness, interim-prior weights, a
  piecewise-constant rate density and a GP prior. It was "written specifically for our
  paper" and last touched in 2014.
- **Kepler DR25** occurrence code is "not intended to be used by the community"
  ([stevepur/DR25-occurrence-public](https://github.com/stevepur/DR25-occurrence-public)).
- **PosteriorStacker**
  ([JohannesBuchner/PosteriorStacker](https://github.com/JohannesBuchner/PosteriorStacker))
  fits a flat-Dirichlet histogram to one-dimensional samples drawn under a flat prior. It
  uses nested sampling through a command-line script.
- **MRExo** ([shbhuk/mrexo](https://github.com/shbhuk/mrexo)) builds an $N\times K$ matrix
  of error-convolved basis integrals internally. It returns maximum-likelihood simplex
  weights with bootstrap errors.
- **dftools** ([obreschkow/dftools](https://github.com/obreschkow/dftools); R) fits
  distribution functions with arbitrary per-object PDFs and selection. Its last commit
  was in 2021.

## How binference relates

**Shared.** binference's model is the one in {cite:t}`leistedt2016`, Starlight,
frankenz, RAIL's `VarInfStackSummarizer` and PosteriorStacker's histogram mode. Its
`Gibbs` engine is their sampler.

**Added.** binference requires the interim prior for posteriors and computes bin
averages from each input type. It samples exactly by default and flags approximations.
It adds both selection modes, smoothness priors, rates, derived quantities per draw and
calibration tools. The {doc}`guides <../guides/index>` describe each.

**Not done.** binference sees only the uncertainty in its model. A prior over
independent objects carries shot noise, not the sample variance or calibration
systematics that 3sDir adds. binference does not use clustering cross-correlations. It
does not estimate an implicit interim prior. It has no background or contaminant
component and no free-mean Gaussian mixtures. It does not export to qp, CosmoSIS or
nz_prior.

## Pitfalls known in astronomy

- **Stacking.** "The stacked estimator is always excessively broad" {cite:p}`malz2022`.
  Stacking recovers $n(z)$ only with "perfectly informative data" or a "perfectly
  informative prior", and "must be abandoned" {cite:p}`malz2021`. Euclid's stacked
  posteriors missed the accuracy requirement by five to ten times before correction
  {cite:p}`ilbert2021`. Rubin's first data-preview paper still reports stacked $n(z)$
  without an uncertainty band {cite:p}`zhang2026`. See
  [stacking](../guides/pitfalls.md#stacking).
- **Posteriors used as likelihoods.** A misspecified interim prior was "disastrous" for
  every estimator CHIPPR tested {cite:p}`malz2022`. The stacked PDFs of machine-learning
  codes "show the imprint of the training sample" {cite:p}`zhang2026`.
- **Shot noise only.** Dirichlet resampling of calibration counts "is only propagating
  shot noise". Sample variance can be "an order of magnitude larger"
  {cite:p}`sanchez2020`. DES Y3's Dirichlet step "only reproduces the variance expected
  from Poisson noise" {cite:p}`myles2021`.
- **Approximate error bars.** The mean-field Dirichlet scheme under-covers
  {cite:p}`rau2022,rau2023`. Bootstrapped maximum-likelihood intervals for luminosity
  functions "can be too narrow" {cite:p}`kelly2008`.
- **Correlated bins.** Broad likelihoods make adjacent bins anti-correlated. The naive
  histogram estimator "fails in the regime of strong inter-bin correlations or
  quasi-empty bins" {cite:p}`leistedt2016`. See
  [per-bin error bars](../guides/pitfalls.md#reading-per-bin-error-bars-as-independent).
- **Binning.** Over-smoothing with coarse bins "can lead to posteriors that can be too
  narrow" {cite:p}`rau2022`. RAIL's example notebook warns that `VarInfStack` "tends to
  get a little wonky if you use too many bins".
- **Importance sampling** from posterior samples "could be a high variance estimator"
  {cite:p}`foremanmackey2014`. The interim prior's support must be broader than the
  population's. Few samples can suffice: 50 per star were "only slightly worse" than
  $10^5$ {cite:p}`hogg2010`.
- **Selection.** Dividing a histogram by the detection efficiency is "not motivated
  probabilistically" {cite:p}`foremanmackey2014`. Selection functions should be expressed
  in observed attributes {cite:p}`rix2021`. {cite:t}`mandel2019` name the two classic
  mistakes ({doc}`../guides/selection`).
- **Contamination and systematics** can dwarf counting noise. Correcting Kepler
  candidates for reliability moved the Earth-analogue rate for GK stars from 0.034 to
  0.015 {cite:p}`bryson2020`. Isochrone systematics run "2–4 times larger than the random
  uncertainties" {cite:p}`zheng2025`.
- **Overestimated errors** "typically cause excessive deconvolution"
  {cite:p}`leistedt2017`.
- **Objects with no support.** A 2026 occurrence-rate script
  ([sblunt/eccentricities](https://github.com/sblunt/eccentricities)) replaces $\log 0$ by zero
  and so drops such objects silently. binference raises an error for a likelihood row of
  zeros unless `drop_empty=True` is passed.
