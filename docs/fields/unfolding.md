# Unfolding in particle physics

How do particle physicists recover a true spectrum from a smeared one? A detector smears
every measured quantity and loses some events. Unfolding estimates the distribution of
the true quantity, such as a cross-section in bins of jet momentum. Binned unfolding is
binference's model with one shared likelihood row per measured bin. Its most used
algorithm is the EM algorithm.

## The problem in particle-physics terms

The response matrix gives the probability of measured bin $j$ for an event in true bin
$k$, including the efficiency. The expected measured histogram is the response times the
true spectrum, plus backgrounds (PDG review, §40.2.5; {cite:alp}`navas2024`). Physicists
build the response from simulated events. Unfolding estimates the true spectrum with a
covariance. Folding smears a theory prediction with the response and compares it with
the data.

Inverting the response amplifies noise, so every method regularises.

- **Iterative Bayesian unfolding** (IBU; {cite:alp}`dagostini1995`) stops after a few
  iterations. It is the EM algorithm for the mixture weights. It is identical to
  Richardson–Lucy deconvolution and to Shepp & Vardi's algorithm for emission tomography
  {cite:p}`kuusela2015`. TUnfold calls it "the iterative EM unfolding method (also called
  D'Agostini Method or 'iterative Bayesian method')".
- **Penalised and filtered methods** include Tikhonov regularisation (TUnfold;
  {cite:alp}`schmitt2012`) and the singular-value decomposition method
  {cite:p}`hocker1996`. Others are Blobel's method, iterative dynamically stabilised
  unfolding and Gaussian-process unfolding {cite:p}`bozson2018`.
- **Fully Bayesian Unfolding** {cite:p}`choudalakis2012` samples the posterior of the
  true spectrum. It treats regularisation as "a non-constant prior" and systematics as
  nuisance parameters.

Since 2020 method development has moved to unbinned machine-learning unfolding. OmniFold
{cite:p}`andreassen2020` reweights simulated events with classifiers in two steps. Its
authors prove that its per-event iteration is a generalised EM algorithm. More than
fifteen variants followed.

## In binference's terms

| unfolding term | binference |
|---|---|
| true spectrum | population fractions $\rho$; expected counts `result.rates`, `result.total` |
| response matrix | {class}`~binference.Response`, $R_{jk}$ |
| efficiency | $\epsilon_k=\sum_jR_{jk}$, a selection `"on_measurement"` |
| measured histogram | `Likelihoods.from_binned(response, counts=...)` |
| missed event (RooUnfold's `Miss`) | a NaN measured value in `Response.from_simulation` |
| regularisation | a smoothness prior, `LogisticGP` or `LogisticRandomWalk` |
| IBU | `method="em"` run to convergence |
| folding check | `result.check()`, when there are more measured than true bins |

An event measured in bin $j$ has the likelihood row $R_{j\cdot}$, shared by all events in
that bin. The [inputs guide](../guides/inputs.md#binned-measurements-and-classifiers)
describes responses and the folding check. The {doc}`priors guide <../guides/priors>`
describes the smoothness priors. The
{doc}`unfolding tutorial <../tutorials/06_categories_unfolding>` unfolds a smeared
spectrum with a Dirichlet and a Gaussian-process prior.

## Methods and software

The [comparison table](software.md#comparison) gives each tool's language, licence and
status.

- **RooUnfold** bundles the classical methods, IBU and TUnfold among them. Its RooFit
  interface evaluates bias, variance and coverage {cite:p}`brenner2020`.
- **TUnfold** {cite:p}`schmitt2012` is a least-squares fit with Tikhonov regularisation.
  It chooses the strength by an L-curve scan, global correlations or Stein's unbiased
  risk estimate.
- **PyUnfold** implements IBU with a stopping test and Adye-corrected errors. It returns
  per-bin errors, not a covariance matrix.
- **PyFBU** sampled Fully Bayesian Unfolding. The authors of Neural Posterior Unfolding
  re-implemented it "to provide an updated version" {cite:p}`toralesacosta2026`.
- **Other Python tools.** QUnfold runs Tikhonov unfolding on quantum annealers. Bunse's
  qunfold joins quantification and unfolding ({doc}`quantification`).
- **Unbinned ML unfolding.** The `omnifold` package is the main entry point; most
  variants are single-paper code. A 2026 practitioners' guide reports 500 to 10,000 GPU
  hours per analysis {cite:p}`canelli2026`.

None of the unfolding tools surveyed accepts a different likelihood row per event.
Binned tools assume one shared response. Unbinned tools assume paired (true, measured)
simulations.

## How binference relates

binference shares the binned likelihood with a Poisson or multinomial count model, the
EM algorithm (`method="em"`), and the posterior that FBU samples. It treats
regularisation as a prior, as FBU does.

For unfolding, binference adds:

- a smoothness scale inferred from the data rather than an iteration count;
- the full posterior covariance of the spectrum;
- per-event likelihoods beyond a shared response.

Not done:

- binference has no background term.
- It has no nuisance parameters for systematic variations.
- It does not unfold unbinned events.
- It gives no frequentist intervals with guaranteed coverage. Its `Bootstrap` engine
  gives approximate ones ([approximations](../guides/engines.md#approximations)).
- It does not read ROOT histograms.

## Pitfalls known in unfolding

- **The iteration count is a hidden regulariser.** IBU uses Bayes' theorem in one step.
  But that "should imply ... that the estimators summarize in some way a joint posterior
  pdf", which they do not {cite:p}`cowan2002`. "The number of iterations should be
  decided upon before looking at the actual data ... Continuing to iterate brings
  increasingly large variances" {cite:p}`cowan2002`.
- **Regularisation bias breaks coverage.** "Neither Bayesian credible intervals nor
  standard bootstrap confidence intervals succeed in achieving good frequentist coverage
  ... due to the inherent bias" {cite:p}`kuusela2015`. An iteratively bias-corrected
  bootstrap restored nearly nominal coverage. {func}`~binference.validation.coverage`
  measures coverage on simulations ({doc}`../guides/validation`).
- **Wide bins.** The response "depends on the unknown shape of the true spectrum".
  Regularising with wide bins introduces "a non-trivial bias" {cite:p}`stanley2022`. The
  remedy is fine bins and intervals for functionals of them. In binference, see
  [is the binning fine enough?](../guides/results.md#is-the-binning-fine-enough).
- **Noisy response matrices.** A response from a finite simulation "can inadvertently
  regularize the problem" {cite:p}`zhu2026`.
- **Bin-by-bin correction** "provably accounts for smearing effects incorrectly"
  {cite:p}`kuusela2015` and "is clearly disfavoured" {cite:p}`schmitt2017`.
- **Correlations must be reported.** The PDG review warns: "if only the unfolded distributions and their
  separate covariance matrices are reported ... information on such correlations is not
  retained" {cite:p}`navas2024`.
- **Folding.** Comparing a smeared theory prediction with the data is "a more
  fundamentally sound way to test hypotheses" {cite:p}`cousins2016`.
