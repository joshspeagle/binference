# Mixed-stock analysis and source tracking

What do ecologists and geneticists call binference's problem? A fishery catches salmon
from several spawning populations, or stocks, that mix at sea. Managers need the stock
proportions of the catch. Fisheries genetics calls this genetic stock identification
(GSI) or mixed-stock analysis (MSA), and solved it decades ago. The same analysis finds
the rookeries of foraging sea turtles {cite:p}`bolker2007`. Microbial source tracking and
RNA-seq transcript quantification solve it for reads.

## The problem in these fields' terms

**Genetic stock identification.** Geneticists sample fish of known stock on the spawning
grounds. This baseline gives each stock's allele frequencies. Each fish in the mixture
then has a genotype likelihood under each stock. GSI estimates the mixture proportions
in one of two ways:

- **Conditional maximum likelihood** by EM {cite:p}`millar1987`, with bootstrap
  intervals. SPAM, ONCOR and mixstock do this.
- **Gibbs sampling** with a Dirichlet prior on the proportions and a latent stock for
  each fish. BAYES {cite:p}`pella2001`, rubias {cite:p}`moran2019`, Ms.GSI and mixstock
  do this.

Managers report groups of stocks together, as reporting units. A conditional model takes
the baseline frequencies as known.

**Microbial source tracking** estimates the share of a sample's microbes that comes from
each source community. SourceTracker {cite:p}`knights2011` samples it by Gibbs. FEAST
{cite:p}`shenhav2019` fits it by EM, with an "unknown" source.

**RNA-seq transcript quantification** estimates the abundance of each transcript from
reads. Each read is compatible with some transcripts. salmon {cite:p}`patro2017` and
kallisto {cite:p}`bray2016` solve the same EM problem over $10^8$ reads.

**Stable-isotope mixing models** (MixSIAR, simmr) solve a different problem. Each
consumer's tracer values are a proportion-weighted mixture of source distributions. No
item belongs to a latent source.

## In binference's terms

| term in these fields | binference |
|---|---|
| stock, source population, transcript | a bin of `Bins.categories(stocks)` |
| mixture proportions | population fractions $\rho$ |
| genotype likelihood under each baseline | a likelihood row; `Likelihoods.from_table(..., kind="likelihood", log=True)` |
| conditional maximum likelihood | `method="map"` |
| BAYES, rubias (conditional, MCMC) | the `Gibbs` engine with a `Dirichlet` prior |
| per-fish posteriors | `result.assignment_probabilities()` |
| sampled fish from each stock | `result.realized_counts()` |
| reporting units | `result.rebin`, which sums fine fractions per draw |
| equivalence classes of reads | `lk.compress()`, which merges identical rows |

The {doc}`mixed-stock tutorial <../tutorials/mixed-stocks>` estimates the stock
proportions of a simulated catch. It carries the baseline's uncertainty with a confusion
matrix, tallied by self-assignment of the baseline (`Response.from_labelled`).

## Methods and software

The [comparison table](software.md#comparison) gives each tool's language, licence and
status.

- **rubias** {cite:p}`moran2019` is the modern reference implementation. It runs
  Dirichlet–multinomial Gibbs sampling conditional on the baseline. It offers a
  parametric-bootstrap bias correction and baseline resampling. It returns traces,
  per-fish posteriors and posterior-predictive catch totals.
- **SPAM** and **ONCOR** compute the conditional maximum-likelihood estimate. SPAM 3.7
  draws baseline allele frequencies from Dirichlet posteriors during resampling.
- **Ms.GSI** chains two Pella–Masuda models. Users can then combine a broad-scale and a
  regional baseline.
- **mixstock** offers maximum likelihood and Gibbs sampling for sea-turtle mixtures
  {cite:p}`bolker2007`.
- **SourceTracker2** is the only Python tool in this family.
- **salmon** and **kallisto** scale EM, variational EM and Gibbs or bootstrap replicates
  to $10^8$ reads. They merge reads with the same compatibility pattern into equivalence
  classes.

The census found no Python GSI or mixed-stock tool. No package in this family documents
a selection term.

## How binference relates

binference shares the conditional model: fixed likelihoods per stock, a Dirichlet prior
and the Gibbs sampler of BAYES and rubias. It shares their per-fish assignment
probabilities and draws of the realised counts in each stock.

For these fields, binference adds a likelihood matrix as direct input, however it was
produced.

Not done:

- binference does not compute genotype likelihoods.
- It does not infer baseline allele frequencies, as rubias's baseline resampling does.
  Only a tallied confusion matrix carries the baseline's uncertainty, as in the tutorial.

## Pitfalls known in these fields

- **Classify and count is biased.** Proportional methods (CML or BAYES) perform better
  than assigning each fish ({cite:alp}`koljonen2005`, as summarised by the Alaska
  Department of Fish and Game).
- **Prior strength matters.** Informed priors give less biased results than flat ones,
  "especially where there is little genetic [differentiation]". Little differentiation
  acts like broad likelihoods. rubias warns that a total concentration
  $\sum_k\alpha_k$ much below one can make the sampler "mix more poorly".
- **Aggregation bias.** A symmetric Dirichlet over stocks gives each reporting unit prior
  mass in proportion to its number of stocks. It thus biases the proportions of uneven
  reporting units. Sum fine fractions per draw with `result.rebin`, and set the prior's
  concentration per bin ({doc}`priors guide <../guides/priors>`).
- **The baseline is an estimate.** A fit that treats baseline frequencies as exact can
  give intervals that are too narrow, as the
  {doc}`tutorial <../tutorials/mixed-stocks>` shows. rubias's baseline resampling and
  SPAM's Dirichlet draws of allele frequencies carry this uncertainty.
