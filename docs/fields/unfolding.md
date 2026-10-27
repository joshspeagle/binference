# Unfolding in particle physics

A particle detector smears every measured quantity and loses some events altogether.
*Unfolding* estimates the distribution of the true quantity, for example a
cross-section in bins of jet momentum, from the histogram of the measured one. Binned
unfolding is binference's model with one shared likelihood row per measured bin, and
its most widely used algorithm is the EM algorithm, one of binference's solvers for
point estimates.

## The problem in particle-physics terms

The measured histogram $n$ has expectation $\nu=R\mu+\beta$, where $\mu$ holds the
expected counts in the true bins, $\beta$ the expected backgrounds, and the response
matrix $R_{bk}=P(\text{observed in bin }b\mid\text{true bin }k)$ includes the
efficiency (PDG statistics review, §40.2.5; {cite:alp}`navas2024`). $R$ is built from simulated events: in
RooUnfold each simulated event fills the matrix with its (measured, true) pair, or is
recorded as missed. The two ways to use the result are *unfolding*, estimating $\mu$
with a covariance, and *folding*, smearing a theory prediction with $R$ and comparing
it with $n$.

Inverting $R$ amplifies the noise, so every method regularises. Iterative Bayesian
unfolding (IBU, {cite:alp}`dagostini1995`) updates $\mu$ by
$t^{(n)}_k=\sum_b\big[R_{bk}t^{(n-1)}_k/\sum_jR_{bj}t^{(n-1)}_j\big]m_b$ and is stopped
after a few iterations. It is the EM algorithm for the mixture weights, identical to
Richardson–Lucy deconvolution and to Shepp & Vardi's algorithm for emission tomography
{cite:p}`kuusela2015`; TUnfold's implementation describes itself as "the iterative
EM unfolding method (also called D'Agostini Method or 'iterative Bayesian method')".
The other classical methods are Tikhonov regularisation (TUnfold; {cite:alp}`schmitt2012`), the
singular-value decomposition method {cite:p}`hocker1996`, Blobel's regularised
unfolding, iterative dynamically stabilised unfolding, and Gaussian-process unfolding
{cite:p}`bozson2018`. Fully Bayesian Unfolding {cite:p}`choudalakis2012` samples the
posterior of the true spectrum, with regularisation as "a non-constant prior" and
systematic uncertainties as nuisance parameters.

Since 2020 method development has moved to unbinned machine-learning unfolding.
OmniFold {cite:p}`andreassen2020` reweights simulated events with classifiers in two
steps, and its authors prove that its per-event iteration is a generalised EM algorithm
with the true values as latent variables; more than fifteen generative and
discriminative variants followed.

## In binference's terms

Expanding the histogram into objects, an event measured in bin $b$ has the likelihood
row $L_{ik}=R_{b(i),k}$, and all events in the same measured bin share a row
([theory](../how-it-works/theory.md#objects-bins-and-likelihoods)).
{meth}`Likelihoods.from_binned(response, counts=...) <binference.Likelihoods.from_binned>`
builds these rows, one per measured bin weighted by its count, and accounts for the
efficiency $\epsilon_k=\sum_bR_{bk}$. The {class}`~binference.Response` comes from a
known matrix, from tallied events, or from the simulated events themselves, where a
missed event plays the role of RooUnfold's `Miss`; a response tallied from a finite
simulation is inferred with the spectrum ({doc}`../guides/inputs` gives the
details). Regularisation is a
prior, not an iteration count: `LogisticGP` or `LogisticRandomWalk` infer how smooth the
spectrum is ({doc}`../guides/priors`). The total and the expected counts per true
bin come from `result.total` and `result.rates`. When there are more measured bins than
true bins, `result.check()` compares the measured counts with those each posterior draw
predicts through the response: a folding check of the fit. The {doc}`unfolding tutorial
<../tutorials/06_categories_unfolding>` unfolds a smeared spectrum with a Dirichlet and a
Gaussian-process prior and compares both with matrix inversion.

## Methods and software

- **RooUnfold** ([RooUnfold/RooUnfold](https://gitlab.cern.ch/RooUnfold/RooUnfold);
  C++/ROOT, BSD-3, version 3.1.0, on conda-forge but not PyPI) bundles IBU, SVD,
  bin-by-bin, matrix inversion, iterative dynamically stabilised unfolding, GP unfolding,
  a Poisson likelihood unfolding and TUnfold, and its RooFit interface evaluates bias,
  variance and coverage {cite:p}`brenner2020`.
- **TUnfold** {cite:p}`schmitt2012` is a least-squares fit with Tikhonov regularisation of the
  size, derivative or curvature, with the strength chosen by an L-curve scan, by
  minimising global correlations, or by Stein's unbiased risk estimate; its last
  standalone release dates from 2019.
- **PyUnfold** ([jrbourbeau/pyunfold](https://github.com/jrbourbeau/pyunfold); last
  release 2018) implements IBU with stopping on a test statistic between iterations and
  Adye-corrected error propagation. It returns per-bin errors, not a covariance matrix.
- **PyFBU** ([pyFBU/fbu](https://github.com/pyFBU/fbu)) sampled the posterior of Fully
  Bayesian Unfolding with background and systematic nuisance parameters using PyMC3 and
  Theano. It is "no longer actively maintained" and needs `numpy<1.20`; the authors of
  Neural Posterior Unfolding re-implemented FBU in TensorFlow notebooks "to provide an
  updated version" {cite:p}`toralesacosta2026`.
- **Other Python tools.** QUnfold solves Tikhonov-regularised unfolding as a quadratic
  unconstrained binary optimisation problem on annealers; Bunse's qunfold implements
  quantification and unfolding methods in one framework (see {doc}`quantification`);
  funfolding fits a Poisson likelihood with Tikhonov regularisation.
- **Unbinned ML unfolding.** The `omnifold` package (TensorFlow) and the reference
  OmniFold notebooks are the maintained entry points; nearly every variant since is
  single-paper code. A 2025 practitioners' guide reports ensembles of 4–10 networks
  (100 for ATLAS), 50–100 bootstrap runs and 500 to 10,000 GPU hours per analysis
  {cite:p}`canelli2026`.

No unfolding tool found accepts a heteroscedastic likelihood matrix with a different
row per event: binned tools assume one shared response matrix, and unbinned tools assume
paired (true, measured) simulations.

## How binference relates

**Shared.** The likelihood of binned unfolding with a Poisson or multinomial count
model, the EM algorithm (`method="em"`; `method="map"` chooses its solver,
{doc}`../guides/engines`), and the posterior that FBU samples, with regularisation
as a prior.

**Added.** A maintained, pip-installable Bayesian unfolding: posterior draws of the
spectrum with its full covariance, a smoothness scale inferred rather than fixed, the
response's own simulation uncertainty sampled with the spectrum, a folding check, a
lightweight Python stack, and inputs beyond a shared response matrix (per-event
likelihoods with a different row for every event).

**Not done.** binference has no background term $\beta$ and no nuisance parameters for
systematic variations of the response or backgrounds; it does not unfold unbinned
event samples, does not provide frequentist intervals with guaranteed coverage (its
`Bootstrap` engine gives the sampling distribution of the MAP estimate, flagged as
approximate), and does not read ROOT histograms.

## Pitfalls known in unfolding

- **The iteration count is a hidden regulariser.** IBU is called Bayesian because it
  uses Bayes' theorem in an intermediate step, but that "should imply ... that the
  estimators summarize in some way a joint posterior pdf", which they do not {cite:p}`cowan2002`. "The number of iterations should be decided upon before looking at the actual
  data ... Continuing to iterate brings increasingly large variances" {cite:p}`cowan2002`.
- **Regularisation bias breaks coverage.** "Neither Bayesian credible intervals nor
  standard bootstrap confidence intervals succeed in achieving good frequentist
  coverage ... due to the inherent bias"; an iteratively bias-corrected bootstrap
  restored nearly nominal coverage {cite:p}`kuusela2015`. binference's
  {func}`~binference.validation.coverage` measures the coverage of an analysis on
  simulations ({doc}`../guides/validation`).
- **Wide bins.** The response "depends on the unknown shape of the true spectrum", and
  regularising with wide bins introduces "a non-trivial bias"; the remedy is to unfold
  on fine bins and build intervals for functionals of them {cite:p}`stanley2022`. In binference: fine bins with a smoothness prior, coarse summaries through
  `result.rebin` and `result.functional`, and `result.binning_bias` to test the
  within-bin assumption ({doc}`../guides/results`).
- **Noisy response matrices.** A response estimated from a finite simulation "can
  inadvertently regularize the problem" {cite:p}`zhu2026`.
- **Bin-by-bin correction** "provably accounts for smearing effects incorrectly"
  {cite:p}`kuusela2015` and "is clearly disfavoured" {cite:p}`schmitt2017`.
- **Correlations must be reported.** Results should be given as estimates with their
  covariance; "if only the unfolded distributions and their separate covariance
  matrices are reported ... information on such correlations is not retained" {cite:p}`navas2024`. To
  allow folding later, the PDG review asks for the data, the response matrix and the
  backgrounds as functions of the nuisance parameters. Comparing a smeared theory
  prediction with the data is "a more fundamentally sound way to test hypotheses" than
  comparing with the unfolded histogram {cite:p}`cousins2016`.
- **Training losses mislead.** For conditional flow matching, "its standard training loss
  is fundamentally misleading": the loss plateaus while the physical fidelity keeps
  improving (ScatterPrism; {cite:alp}`xia2026`).
