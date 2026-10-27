# Photometric redshifts and astronomical populations

Astronomers infer population distributions from catalogues in which each object's
quantity is measured with its own, often large, error: the redshift distribution
$n(z)$ of a galaxy sample, the colour–magnitude diagram of the stars in the Milky Way,
the occurrence rate of planets as a function of period and radius, luminosity and mass
functions, and distributions of orbital eccentricities. {cite:t}`hogg2010` set
out the general method: take per-object likelihood functions, or samples of per-object
posteriors drawn under a known uninformative prior, and fit the population as "a kind of
heteroscedastic deconvolution", which they note applies to "magnitudes, parallaxes, or
photometric redshifts". binference implements that method and its extensions.

## The problem in astronomical terms

**Redshift distributions.** A weak-lensing survey divides its galaxies into tomographic
bins and needs each bin's $n(z)$, with its uncertainty, for the cosmological analysis.
A photometric-redshift (photo-$z$) code gives each galaxy a $p(z)$. Template-fitting
codes can return likelihoods, for example BPZ run without its prior; machine-learning
codes return posteriors under the redshift distribution of their training sample, an
*implicit* interim prior. The common practice is to sum the $p(z)$ ("stacking"). The
hierarchical alternative puts a prior on the histogram heights of $n(z)$ and infers
them together with each galaxy's redshift {cite:p}`leistedt2016`.

**Stellar populations.** {cite:t}`leistedt2017` modelled the colour–magnitude diagram
of 1.4 million Gaia TGAS stars as a mixture of fixed kernels with amplitudes on the
simplex. Because the kernels are fixed, each star's likelihood in each kernel,
marginalised over its distance, is computed once; the population and the stars' kernel
labels are then sampled by Gibbs sampling, "4 million parameters in total, most of
which are marginalized out". Resolved stellar populations are also fitted as binned
colour–magnitude ("Hess") diagrams: the counts in each cell are Poisson with mean
$\sum_j r_j c_{ij}$, a weighted sum of template diagrams.

**Occurrence rates and distribution functions.** {cite:t}`foremanmackey2014`
treated a planet catalogue as a draw from an inhomogeneous Poisson process with
observable rate $Q(w)\,\Gamma(w)$, the survey completeness times the occurrence-rate
density, modelled $\Gamma$ as a piecewise-constant function with a Gaussian-process
prior on the log step heights, and used per-planet posterior samples with the interim
prior divided out. Luminosity- and mass-function codes use the same Poisson-process
likelihood with a selection function {cite:p}`loredo2004,kelly2008`.

## In binference's terms

| astronomical input | binference |
|---|---|
| per-galaxy likelihoods on a redshift grid | {class}`~binference.Likelihoods` from an array, or `Likelihoods.from_density(..., kind="likelihood")` from a function |
| per-galaxy posteriors under a known prior | `Likelihoods.from_table(P, bins, interim=...)` |
| posterior samples (eccentricities, planet radii) | `Likelihoods.from_samples(samples, bins, interim=...)`, ragged and weighted |
| a measurement with Gaussian errors (parallax, colour) | `Likelihoods.from_gaussian`, with a covariance matrix per object on grids |
| a colour–magnitude diagram | `Bins.grid(...)`, with `mask=` for unphysical cells |
| a completeness function or injection–recovery test | `selection_mode="on_truth"`, `Selection.from_injections` |
| a flux or signal-to-noise cut | `selection_mode="on_measurement"` |
| a spectroscopic reference sample | `priors.Dirichlet.from_counts(m, strength)`, or a logistic prior with `mean=np.log(reference)` |
| smoothness of $n(z)$ or of a rate density | `priors.LogisticGP`, `priors.LogisticRandomWalk` |
| expected numbers, occurrence rates | `result.rates`, `result.total`, `priors.GammaRates` |

The model needs bin averages of each likelihood ([theory](../how-it-works/theory.md#objects-bins-and-likelihoods)).
{cite:t}`leistedt2016` and the HSC Y3 analysis integrate each likelihood over the
histogram bins; RAIL's summarizers evaluate each PDF at grid points instead.
`from_gaussian` and `from_density` compute the bin averages, and `from_table` asks
whether a table holds values per bin or masses (`representation=`)
({doc}`../guides/inputs/index`).

Posterior samples enter as {cite:author}`hogg2010`'s importance-sampling estimate: each object's
samples are histogrammed onto the bins, each weighted by the inverse of the interim
prior.
binference stores each object's effective number of samples, and `result.check()` warns
when the Monte Carlo noise of these estimates is large enough to bias the total
log-likelihood. The Poisson-process likelihood of occurrence-rate studies is the rate
form of the same model: under the scale-free prior $p(\Lambda)\propto1/\Lambda$ on the
total it reduces to the normalised likelihood in fractions {cite:p}`mandel2019`,
which is how binference obtains `result.total` and `result.rates` from a fit of
fractions ({doc}`../guides/selection`). The {doc}`quick start
<../tutorials/01_quickstart>` fits a redshift distribution, the {doc}`selection tutorial
<../tutorials/05_selection>` an occurrence rate, and the {doc}`two-dimensional tutorial
<../tutorials/08_two_dimensions>` a population on a grid.

## Methods and software

### Redshift distributions

- **RAIL** (LSST DESC) is the only maintained, pip-installable, general tool that turns
  a catalogue of per-galaxy $p(z)$ into $n(z)$ samples. Its summarizers that use PDFs
  are a stack with a Poisson bootstrap over galaxies, a histogram of point estimates,
  and `VarInfStackSummarizer`: coordinate-ascent mean-field variational Bayes for a
  flat Dirichlet prior with latent bin labels, which is binference's baseline model. It
  evaluates each PDF at grid points, applies no interim-prior correction, and holds all
  data in memory. Its originators found that the scheme "yielded underestimated error
  bars, likely due to the restricted covariance structure of the Dirichlet" {cite:p}`rau2022`, and the HSC Y3 analysis put the under-coverage at about 20% from "numerical
  experience" {cite:p}`rau2023`. RAIL's other summarizers (SOM, NZDir, SOMPZ) reweight
  spectroscopic calibration samples in colour space rather than use per-galaxy PDFs.
- **{cite:t}`leistedt2016`** sampled the hierarchical model with a two-step
  Gibbs sampler on binned likelihoods, drawing $10^4$ samples in about ten minutes on a
  laptop. Their code is in
  [Photoz-tools](https://github.com/ixkael/Photoz-tools) (last push 2020) and
  [BHM-nz](https://github.com/Harry45/BHM-nz) (2023). A 2026 KiDS application still
  used a subset of $10^5$ galaxies because running on all of them "would be too slow"
  {cite:p}`kyriacou2026`.
- **CHIPPR** {cite:p}`malz2022` is the only photo-$z$ code found that divides out the
  interim prior. It models $n(z)$ as a log-space histogram with a squared-exponential
  hyperprior and samples it with emcee; its README calls it "stale" and "a prototype",
  and its demonstration notebook is for Python 2.7
  ([aimalz/chippr](https://github.com/aimalz/chippr)).
- **Survey calibration pipelines** obtain wider, more realistic errors by other routes.
  DES SOMPZ samples $n(z)$ realisations with a three-step Dirichlet ("3sDir") that
  inflates the variance to include sample variance {cite:p}`sanchez2020,myles2021`; KiDS calibrates with self-organising maps and passes Gaussian priors on
  per-bin shifts to cosmology; HSC Y3 combines photometry with clustering
  cross-correlations under a logistic Gaussian-process prior {cite:p}`rau2020,rau2023`; Euclid
  studied stacked posteriors corrected with the PIT histogram of a spectroscopic sample
  {cite:p}`ilbert2021`. These work from colour cells, calibration samples and
  clustering rather than from per-galaxy likelihoods.
- **Downstream formats.** $n(z)$ ensembles are passed to cosmology as
  [qp](https://github.com/LSSTDESC/qp) ensembles, as FITS realisation files ranked by
  hyperrank (CosmoSIS `nz_multirank`; {cite:alp}`cordero2022`), or compressed into shift,
  PCA or GP priors ([nz_prior](https://github.com/LSSTDESC/nz_prior)).

### frankenz

binference descends from the hierarchical sampler in
[frankenz](https://github.com/joshspeagle/frankenz) (last push October 2018; one PyPI
release, 0.1.5), which offered two samplers for $n(z)$: a "population" sampler that
moves $\rho$ directly, and a "hierarchical" Gibbs sampler that alternates per-galaxy
bin labels and a Dirichlet draw of $\rho$. An audit made while designing binference
([`frankenz_audit.md`](https://github.com/joshspeagle/binference/blob/main/research_notes/Population%20inference%20software%20landscape/frankenz_audit.md) in the research notes) ran both against exact posteriors for three bins,
Geweke "getting it right" tests and a bit-for-bit re-implementation. Its findings shaped
binference:

- **The two samplers target the same posterior for $\rho$.** Summing the hierarchical
  model over the labels gives the population model exactly
  ([theory](../how-it-works/theory.md#the-augmented-hierarchical-form)), so binference has one model
  with several engines.
- **The hierarchical sampler is exact.** It matched the exact posteriors (root-mean-square
  z-score 0.94 over 132 statistics) and passed the Geweke test. binference keeps it,
  vectorised, as the `Gibbs` engine. Its mixing slows once the likelihoods are as wide
  as the bins: with a flat prior the median autocorrelation time grew from 7 to 74, 162
  and 180 sweeps as the likelihood width grew from 0.5 to 1, 2 and 4 bin widths, while
  a simple HMC in log-ratio coordinates stayed at 1–14. binference therefore leaves
  Gibbs for an importance-corrected Laplace approximation or NUTS when the fraction of
  missing information is high ({doc}`../guides/engines`).
- **The population sampler is not a valid MCMC sampler.** Its proposal width depends on
  the current state, but the acceptance test omits the Hastings factor this requires,
  and the width is held fixed across three inner steps. A bin whose posterior reaches
  zero is driven to about $10^{-13}$ within 2–15 saved samples and its mass moves to
  other bins; with no bin near zero every posterior standard deviation is still 8–18%
  too large. It was not ported.
- **The "flexible reference" step belongs to no joint model.** It is not the full
  conditional of any distribution in which $\rho$ given the counts is Dirichlet, it
  fails the Geweke test, and it inflates standard deviations by a factor of 1.1–1.9
  even with a representative reference sample. binference replaces it with the power
  prior `Dirichlet.from_counts(m, strength)` and with logistic priors centred on a
  reference distribution ({doc}`../guides/priors`).
- **Wrong inputs cause the largest errors.** Posteriors computed under a non-uniform
  interim prior biased $\rho$ by up to 12.6 posterior standard deviations and cut the
  coverage of 68% and 95% intervals to 0.29 and 0.52; dividing out the interim prior
  removed the bias. Using the bin integral of each likelihood where the model needs the
  bin average biased $\rho$ as well ([masses versus densities](../guides/troubleshooting.md#masses-versus-densities-with-unequal-bins)). binference
  requires the interim prior for posteriors and asks what a table holds
  ({doc}`../guides/inputs/index`).
- **Software defects** (runs that never resumed, a `reset()` that left old samples,
  legacy random state, bins frozen at exactly zero, no input validation) are each
  covered by a test in binference: engines resume (`engine.run(a); engine.run(b)`),
  randomness comes only from the `rng` argument, and malformed inputs raise errors.

### Colour–magnitude diagrams and other densities

- **Starlight** ([ixkael/Starlight](https://github.com/ixkael/Starlight)), the code of
  {cite:t}`leistedt2017`, is binference's baseline model almost exactly; its last commit
  was in 2017 and it was never released as a package.
- **Extreme deconvolution** (XD; {cite:alp}`bovy2011`) fits a Gaussian mixture with
  free means and covariances to data with Gaussian errors and returns the
  maximum-likelihood mixture, with no posterior uncertainty. The reference C code
  ([jobovy/extreme-deconvolution](https://github.com/jobovy/extreme-deconvolution)) is
  maintained but not on PyPI; astroML's `XDGMM` lacks projection matrices; only
  [pygmmis](https://github.com/pmelchior/pygmmis) handles truncation, and only with a
  completeness known over the whole feature space. Fixing XD's means and covariances and
  fitting only the amplitudes gives binference's model; {cite:author}`leistedt2017`'s fixed kernels
  are this case, sampled by conjugate Gibbs sampling.
- **StarFormationHistories.jl** ([cgarling/StarFormationHistories.jl](https://github.com/cgarling/StarFormationHistories.jl);
  Julia) fits Hess-diagram counts with a Poisson likelihood ratio and offers a MAP fit
  with inverse-Hessian errors, HMC and MCMC. Its version 2.0.0 added quantiles of
  per-sample cumulative quantities, which "account for the correlations between time
  bins", and fitted background templates, because "contaminants such as background
  galaxies and foreground stars otherwise bias the fit". PANCAKE {cite:p}`zheng2025`
  instead fits Poisson Hess diagrams with a mean-squared-error loss.

### Occurrence rates and distribution functions

- **exopop** ([dfm/exopop](https://github.com/dfm/exopop)), the code of {cite:t}`foremanmackey2014`, combines completeness, interim-prior importance weights, a
  piecewise-constant rate density and a GP prior with marginalised hyperparameters,
  sampled by elliptical slice sampling. It was "written specifically for our paper" and
  last touched in 2014.
- **Kepler DR25** occurrence code is "not intended to be used by the community"
  ([stevepur/DR25-occurrence-public](https://github.com/stevepur/DR25-occurrence-public)).
  A 2026 hierarchical-histogram script for giant-planet occurrence
  ([sblunt/eccentricities](https://github.com/sblunt/eccentricities)) uses
  completeness-weighted Poisson likelihoods on posterior samples.
- **PosteriorStacker** ([JohannesBuchner/PosteriorStacker](https://github.com/JohannesBuchner/PosteriorStacker))
  fits a flat-Dirichlet histogram to one-dimensional posterior samples drawn under a
  flat prior, by nested sampling, through a command-line script.
- **MRExo** ([shbhuk/mrexo](https://github.com/shbhuk/mrexo)) builds an $N\times K$
  matrix of each object's error-convolved basis integrals internally and returns
  maximum-likelihood simplex weights with bootstrap errors.
- **dftools** ([obreschkow/dftools](https://github.com/obreschkow/dftools); R) fits
  distribution functions with arbitrary per-object PDFs and selection; its last commit
  was in 2021, and its Python port never implemented non-parametric models.

## How binference relates

**Shared.** binference's model is the one in {cite:t}`leistedt2016`, Starlight,
frankenz's hierarchical sampler, RAIL's `VarInfStackSummarizer` and PosteriorStacker's
histogram mode, and its `Gibbs` engine is their sampler.

**Added.** Explicit input types, with the interim prior required for posteriors and bin
averages computed from Gaussians, densities and samples; exact sampling by default, with
approximations flagged as such ({doc}`../guides/engines`); the two selection modes, and
completeness measured by injections and inferred with the population; smoothness priors;
rates and totals; per-draw derived quantities (`result.functional`, `result.rebin`); a
check of the within-bin flatness assumption (`result.binning_bias`); and
simulation-based calibration and coverage tools ({doc}`../guides/checking`).

**Not done.** binference propagates only the uncertainty that its model contains: a
Dirichlet or logistic prior over independent objects sees shot noise, not the sample
variance or calibration systematics that 3sDir adds, and binference has no layer for
these. It does not use clustering cross-correlations, does not estimate an implicit
interim prior, has no background or contaminant component, does not fit Gaussian
mixtures with free means as XD does, and does not export to qp, CosmoSIS realisation
files or nz_prior (results save to `.npz` and convert to an xarray `DataTree`).

## Pitfalls known in astronomy

- **Stacking.** "The stacked estimator is always excessively broad" {cite:p}`malz2022`;
  stacking recovers $n(z)$ only with "perfectly informative data" or a "perfectly
  informative prior", and "must be abandoned" {cite:p}`malz2021`. Euclid's stacked posteriors
  missed the accuracy requirement by five to ten times before correction {cite:p}`ilbert2021`, and Rubin's first data-preview paper still reports stacked $n(z)$ without an
  uncertainty band {cite:p}`zhang2026`. The stack is one EM step from the interim prior
  ([stacking](../guides/troubleshooting.md#stacking)).
- **Posteriors used as likelihoods.** A misspecified interim prior was "disastrous" for
  every estimator CHIPPR tested {cite:p}`malz2022`, and the stacked PDFs of
  machine-learning codes "show the imprint of the training sample" {cite:p}`zhang2026`.
- **Shot noise only.** Dirichlet resampling of calibration counts "is only propagating
  shot noise", while sample variance can be "an order of magnitude larger" {cite:p}`sanchez2020`; DES Y3's Dirichlet step "only reproduces the variance expected from Poisson
  noise" {cite:p}`myles2021`.
- **Approximate error bars.** The mean-field Dirichlet scheme under-covers {cite:p}`rau2022,rau2023`, and bootstrapped maximum-likelihood intervals for luminosity functions
  "can be too narrow" {cite:p}`kelly2008`.
- **Correlated bins.** Broad likelihoods make adjacent bins strongly anti-correlated, and
  the naive histogram estimator "fails in the regime of strong inter-bin correlations or
  quasi-empty bins" {cite:p}`leistedt2016`. Derived quantities must be computed per
  draw, not from per-bin quantiles.
- **Binning.** Over-smoothing with coarse bins "can lead to posteriors that can be too
  narrow" {cite:p}`rau2022`, and RAIL's example notebook warns that `VarInfStack` "tends
  to get a little wonky if you use too many bins".
- **Importance sampling from posterior samples** "could be a high variance estimator",
  depending on the number of samples and the interim prior, whose support must be
  broader than the population's {cite:p}`foremanmackey2014`. Few samples can suffice:
  50 per star were "only slightly worse" than $10^5$ {cite:p}`hogg2010`.
- **Selection.** Dividing a histogram by the detection efficiency is "not motivated
  probabilistically" and biased {cite:p}`foremanmackey2014`. Selection functions should
  be expressed in observed attributes and defined beyond the region where objects are
  catalogued {cite:p}`rix2021`. {cite:t}`mandel2019` name the two classic mistakes
  ({doc}`../guides/selection`).
- **Contamination and systematics** can dwarf counting noise: correcting Kepler
  candidates for reliability moved the Earth-analogue rate for GK stars from 0.034 to
  0.015 {cite:p}`bryson2020`, and isochrone systematics in colour–magnitude fitting run
  "2–4 times larger than the random uncertainties" {cite:p}`zheng2025`.
- **Overestimated errors** "typically cause excessive deconvolution" {cite:p}`leistedt2017`.
- **Objects with no support.** A 2026 occurrence-rate script replaces $\log 0$ by zero
  and so silently drops such objects; binference raises an error for a likelihood row of
  zeros unless `drop_empty=True` is passed.
