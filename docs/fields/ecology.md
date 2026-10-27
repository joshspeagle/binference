# Mixed-stock analysis and source tracking

What do ecologists and geneticists call binference's problem? A fishery catches salmon
from several spawning populations, or stocks, that mix at sea. Managers need the stock
proportions of the catch. Fisheries genetics calls this genetic stock identification
(GSI) or mixed-stock analysis (MSA). It solved binference's problem decades ago. The same
analysis finds the rookeries of foraging sea turtles {cite:p}`bolker2007`. Microbial
source tracking and RNA-seq transcript quantification solve it again over reads.

## The problem in these fields' terms

**Genetic stock identification.** Fish sampled on the spawning grounds form a baseline,
with known stock. The baseline gives each stock's allele frequencies. Each fish in the
mixture then has a genotype likelihood under each stock. The mixture proportions are
estimated in one of two ways:

- **Conditional maximum likelihood** by EM {cite:p}`millar1987`, with bootstrap
  intervals. SPAM, ONCOR and mixstock do this.
- **Gibbs sampling** with a Dirichlet prior on the proportions and a latent stock for
  each fish. BAYES {cite:p}`pella2001`, rubias {cite:p}`moran2019`, Ms.GSI and mixstock
  do this.

Groups of stocks are reported together as reporting units. A model that conditions on
the baseline frequencies is called conditional.

**Microbial source tracking** estimates the share of a sample's microbes that comes from
each source community. SourceTracker {cite:p}`knights2011` samples it by Gibbs.
FEAST {cite:p}`shenhav2019` fits it by EM, with an "unknown" source.

**RNA-seq transcript quantification** estimates the abundance of each transcript from
reads. Each read is compatible with some transcripts. salmon {cite:p}`patro2017` and
kallisto {cite:p}`bray2016` solve the same EM problem over $10^8$ reads.

**Stable-isotope mixing models** (MixSIAR, simmr) are a different problem. Each
consumer's tracer values are a proportion-weighted mixture of source distributions. No
item belongs to a latent source.

## In binference's terms

| term in these fields | binference |
|---|---|
| stock, source population, transcript | a bin of `Bins.categories(stocks)` |
| mixture proportions | population fractions $\rho$ |
| genotype likelihood under each baseline | a likelihood row $L_{i\cdot}$; `Likelihoods.from_table(..., kind="likelihood", log=True)` |
| conditional maximum likelihood | `method="map"` |
| BAYES, rubias (conditional, MCMC) | the `Gibbs` engine with a `Dirichlet` prior |
| per-fish posteriors | `result.assignment_probabilities()` |
| sampled fish from each stock | `result.realized_counts()` |
| reporting units | `result.rebin`, which sums fine fractions per draw |
| equivalence classes of reads | `lk.compress()`, which merges identical rows |

The {doc}`mixed-stock tutorial <../tutorials/10_mixed_stocks>` estimates the stock
proportions of a simulated catch. Genotype likelihoods from baseline frequencies treat
those frequencies as exact. Intervals can then be too narrow for stocks the baseline
separates poorly. The tutorial carries the baseline's uncertainty instead with a
confusion matrix, tallied by self-assignment of the baseline (`Response.from_labelled`).

## Methods and software

- **rubias** ([eriqande/rubias](https://github.com/eriqande/rubias); R/C++, CC0) is the
  modern reference implementation {cite:p}`moran2019`. It runs Dirichlet–multinomial
  Gibbs sampling conditional on the baseline. It offers a parametric-bootstrap bias
  correction and baseline resampling. It returns traces, per-fish posteriors and
  posterior-predictive catch totals. CRAN archived it in February 2026 because the
  maintainer's email was undeliverable.
- **SPAM** (Alaska Department of Fish and Game) and **ONCOR** are legacy desktop programs
  for the conditional maximum-likelihood estimate. SPAM 3.7 draws baseline allele
  frequencies from Dirichlet posteriors during resampling.
- **Ms.GSI** (R) chains two Pella–Masuda models, so a broad-scale and a regional baseline
  can be combined.
- **mixstock** ([bbolker/mixstock](https://github.com/bbolker/mixstock); R/C) offers
  maximum likelihood and Gibbs sampling for sea-turtle mixtures {cite:p}`bolker2007`.
  CRAN archived it in 2014.
- **SourceTracker2**
  ([caporaso-lab/sourcetracker2](https://github.com/caporaso-lab/sourcetracker2)) is the
  only Python tool in this family. Its last PyPI release is from 2016.
- **salmon** and **kallisto** scale EM, variational EM and Gibbs or bootstrap replicates
  to $10^8$ reads. They collapse reads with the same compatibility pattern into
  equivalence classes.

No Python GSI or mixed-stock tool was found. No package in this family documents a
selection term.

## How binference relates

**Shared.** The conditional model: fixed likelihoods per stock, a Dirichlet prior, and
the Gibbs sampler of BAYES and rubias. Per-fish assignment probabilities and draws of the
realised counts in each stock.

**Added.** A likelihood matrix as direct input, however it was produced. Selection
effects. Simulation-based calibration of the whole analysis.

**Not done.** binference does not compute genotype likelihoods. It does not infer
baseline allele frequencies, as rubias's baseline resampling does. The baseline's
uncertainty enters only through a tallied confusion matrix, as in the tutorial.

## Pitfalls known in these fields

- **Classify and count is biased.** Proportional methods (CML or BAYES) perform better
  than assigning each fish ({cite:alp}`koljonen2005`, as summarised by the Alaska
  Department of Fish and Game).
- **Prior strength matters.** Informed priors give less biased results than flat ones,
  "especially where there is little genetic [differentiation]". Little differentiation
  plays the role of broad likelihoods. rubias warns that a Dirichlet prior with total
  mass much below one can make the sampler "mix more poorly".
- **Aggregation bias.** A symmetric Dirichlet over stocks gives each reporting unit prior
  mass in proportion to its number of stocks. This biases the proportions of uneven
  reporting units. In binference, sum fine fractions per draw with `result.rebin`. Choose
  the prior's concentration per bin ({doc}`simplex-priors`).
- **The baseline is an estimate.** rubias's baseline resampling and SPAM's Dirichlet
  draws of allele frequencies both carry its uncertainty. A fit that treats baseline
  frequencies as exact can give intervals that are too narrow, as the
  {doc}`tutorial <../tutorials/10_mixed_stocks>` shows.
