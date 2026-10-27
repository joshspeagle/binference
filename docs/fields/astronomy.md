# Photometric redshifts and astronomical populations

How do astronomers infer a population from objects measured with large errors? They
need the redshift distribution $n(z)$ of a galaxy sample, the colour–magnitude diagram
of Milky Way stars, planet occurrence rates, and luminosity and mass functions. Each
object's quantity carries its own error. {cite:t}`hogg2010` set out the general method.
It takes per-object likelihoods, or posterior samples drawn under a known prior. It then
fits the population as "a kind of heteroscedastic deconvolution". The method applies to
"magnitudes, parallaxes, or photometric redshifts". binference implements it and its
extensions.

## The problem in astronomical terms

**Redshift distributions.** A weak-lensing survey splits its galaxies into tomographic
bins. The cosmological analysis needs each bin's $n(z)$ with its uncertainty. A
photometric-redshift (photo-$z$) code gives each galaxy a $p(z)$. Template codes can
return likelihoods, for example BPZ run without its prior. Machine-learning codes return
posteriors under the redshift distribution of their training sample, an implicit prior.
Common practice sums the $p(z)$; this is {term}`stacking`. The hierarchical alternative
puts a prior on the histogram heights of $n(z)$. It infers them together with each
galaxy's redshift {cite:p}`leistedt2016`.

**Stellar populations.** {cite:t}`leistedt2017` modelled the colour–magnitude diagram of
1.4 million Gaia TGAS stars. They used a mixture of fixed kernels with amplitudes on the
simplex. Their code computes each star's likelihood in each kernel once, marginalised
over distance. Gibbs sampling then draws the population and the stars' kernel labels:
"4 million parameters in total, most of which are marginalized out". Studies of resolved
stellar populations also fit binned colour–magnitude ("Hess") diagrams. They model the
counts in each cell as Poisson, with a mean that is a weighted sum of template diagrams.

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
{doc}`inputs guide <../guides/inputs>` says how binference averages each input.

The {doc}`quick start <../tutorials/01_quickstart>` fits a redshift distribution. The
{doc}`selection tutorial <../tutorials/05_selection>` fits an occurrence rate.

## Methods and software

The [comparison table](software.md#comparison) gives each tool's language, licence and
status.

### Redshift distributions

- **RAIL** (LSST DESC) is the only general tool that turns per-galaxy $p(z)$ into
  $n(z)$ samples. Its `VarInfStackSummarizer` runs mean-field variational Bayes for a
  flat Dirichlet, binference's baseline model. It does not divide out an interim prior.
- **{cite:t}`leistedt2016`** sampled the hierarchical model by Gibbs sampling, in
  Photoz-tools. BHM-nz is a later implementation. A 2026 KiDS application used only $10^5$ galaxies, because
  all of them "would be too slow" {cite:p}`kyriacou2026`.
- **CHIPPR** {cite:p}`malz2022` is the only photo-$z$ code found that divides out the
  interim prior.
- **Survey calibration pipelines** widen the errors by other routes. DES draws $n(z)$
  with a three-step Dirichlet ("3sDir") that adds sample variance
  {cite:p}`sanchez2020,myles2021`. HSC Y3 adds clustering cross-correlations
  {cite:p}`rau2020,rau2023`. Euclid corrected stacked posteriors with a spectroscopic
  sample {cite:p}`ilbert2021`.
- **Downstream formats.** Cosmology codes read $n(z)$ ensembles as
  [qp](https://github.com/LSSTDESC/qp) ensembles, CosmoSIS realisation files
  {cite:p}`cordero2022`, or [nz_prior](https://github.com/LSSTDESC/nz_prior) priors.
- **frankenz**, binference's predecessor, had an exact Gibbs sampler and an invalid
  Metropolis sampler. An audit made during binference's design tested both against
  exact posteriors.

### Colour–magnitude diagrams and other densities

- **Starlight** is the code of {cite:t}`leistedt2017`. It fits binference's baseline
  model almost exactly.
- **Extreme deconvolution** (XD; {cite:alp}`bovy2011`) fits a Gaussian mixture with free
  means and covariances to data with Gaussian errors. It returns the maximum-likelihood
  mixture, with no posterior uncertainty. pygmmis adds truncation, with a completeness
  known everywhere. XD with fixed means and covariances is binference's model.
- **StarFormationHistories.jl** fits Hess-diagram counts with a Poisson likelihood ratio.
  It offers a MAP fit with inverse-Hessian errors, HMC and MCMC. It fits background
  templates, because "contaminants such as background galaxies and foreground stars
  otherwise bias the fit". PANCAKE {cite:p}`zheng2025` fits Poisson Hess diagrams with a
  mean-squared-error loss.

### Occurrence rates and distribution functions

- **exopop** is the code of {cite:t}`foremanmackey2014`. It combines completeness,
  interim-prior weights, a piecewise-constant rate density and a GP prior.
- **PosteriorStacker** fits a flat-Dirichlet histogram to one-dimensional samples drawn
  under a flat prior, by nested sampling.
- **MRExo** builds an $N\times K$ matrix of error-convolved basis integrals internally.
  It returns maximum-likelihood simplex weights with bootstrap errors.
- **dftools** (R) fits distribution functions with arbitrary per-object PDFs and
  selection.

## How binference relates

binference shares the model of {cite:t}`leistedt2016`, Starlight, frankenz, RAIL's
`VarInfStackSummarizer` and PosteriorStacker's histogram mode. Its `Gibbs` engine is
their sampler.

For astronomy, binference adds:

- masked grids for colour–magnitude diagrams;
- priors centred on a spectroscopic reference;
- rates and totals from a fit of fractions ({doc}`selection guide <../guides/selection>`).

Not done:

- binference does not add the sample variance or redshift-calibration systematics that
  3sDir adds.
- It does not use clustering cross-correlations.
- It does not estimate an implicit interim prior.
- It has no background or contaminant component.
- It does not fit Gaussian mixtures with free means.
- It does not export to qp, CosmoSIS or nz_prior.

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
  codes "show the imprint of the training sample" {cite:p}`zhang2026`. In the frankenz
  audit, such posteriors cut the coverage of 68% intervals to 0.29
  ([pitfalls](../guides/pitfalls.md#feeding-posteriors-in-as-likelihoods)).
- **Shot noise only.** Dirichlet resampling of calibration counts "is only propagating
  shot noise". Sample variance can be "an order of magnitude larger"
  {cite:p}`sanchez2020`. DES Y3's Dirichlet step "only reproduces the variance expected
  from Poisson noise" {cite:p}`myles2021`.
- **Approximate error bars.** RAIL's mean-field scheme "yielded underestimated error
  bars, likely due to the restricted covariance structure of the Dirichlet"
  {cite:p}`rau2022`. HSC Y3 reports its under-coverage as "approximately 20%", from
  "numerical experience" {cite:p}`rau2023`. Bootstrapped maximum-likelihood intervals
  for luminosity functions "can be too narrow" {cite:p}`kelly2008`.
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
- **Selection.** Dividing a histogram by the detection probability is "not motivated
  probabilistically" {cite:p}`foremanmackey2014`. Selection functions should use
  observed attributes {cite:p}`rix2021`. {cite:t}`mandel2019` name the two classic
  mistakes ({doc}`../guides/selection`).
- **Contamination and systematics** can exceed counting noise. Correcting Kepler
  candidates for reliability moved the Earth-analogue rate for GK stars from 0.034 to
  0.015 {cite:p}`bryson2020`. Isochrone systematics run "2–4 times larger than the
  random uncertainties" {cite:p}`zheng2025`.
- **Overestimated errors** "typically cause excessive deconvolution"
  {cite:p}`leistedt2017`.
- **Objects with no support.** A 2026 occurrence-rate script
  ([sblunt/eccentricities](https://github.com/sblunt/eccentricities)) replaces $\log 0$ by zero.
  It therefore drops such objects without a warning. binference raises an error for a
  likelihood row of zeros, unless you pass
  [`drop_empty=True`](../guides/inputs.md#likelihood-matrices).
