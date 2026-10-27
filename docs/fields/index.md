# Fields

Which name does your field give to binference's model, and which tools does it use?
Several fields arrived at the same model on their own. Each gave it its own names and
wrote its own software. Each page in this section takes one field. It states the problem
in that field's terms and maps it onto binference. It lists the field's methods and
software, and the mistakes the field has documented.

The census behind these pages recorded the facts about other packages on 22 October
2026. Releases of binference do not update them. Statements about binference describe
the current code.

## One model under many names

Every field below has $N$ {term}`objects <object>`, $K$ {term}`bins <bin>` or classes,
and for each object $i$ a {term}`bin-averaged likelihood` $L_{ik}$. The
{term}`population fractions` $\rho$ lie on the {term}`simplex`, and the likelihood of
the whole catalogue is

$$
p(D\mid\rho)=\prod_{i=1}^{N}\sum_{k=1}^{K}L_{ik}\,\rho_k .
$$

The {doc}`model page <../how-it-works/model>` derives this model and its extensions.
The two tables below collect each field's names for binference's terms in one place. A
dash means the field has no common name for the term. Each field page uses its own names
only on that page.

### Vocabulary

| field | object | bin | fractions $\rho$ | likelihood $L_{ik}$ | interim prior | response | selection | hierarchical posterior |
|---|---|---|---|---|---|---|---|---|
| astronomy | galaxy, star, planet | redshift bin, colour–magnitude cell, Hess-diagram cell | redshift distribution $n(z)$ (a density), occurrence rate, luminosity function | binned photo-$z$ likelihood; a star's likelihood in a kernel | prior of the photo-$z$ code; implicit training-set prior | — | completeness | posterior shrunk by the population |
| gravitational waves | event | bin of masses, spins and redshift | population model; rates in bins | per-event weights: samples divided by the PE prior | PE prior | — | detection probability; $VT$ | population-informed (reweighted) event posterior |
| particle-physics unfolding | event | true bin | true spectrum | row of the response matrix | simulation prior | response matrix; migration matrix with efficiency | efficiency and acceptance | — |
| empirical Bayes | unit | support point | prior $g$, mixing distribution | likelihood matrix | — | — | truncation | empirical-Bayes posterior |
| ML quantification | item | class | prevalence, class proportions | classifier posterior divided by the training prevalence | training prevalence | confusion matrix | — | corrected per-item posterior |
| ecology and genetics | fish, turtle, read | stock, source, transcript | mixture proportions, stock proportions | genotype likelihood under each baseline; read compatibility | — | — | — | per-fish assignment probability |

### Methods

| field | point estimate of $\rho$ | posterior over $\rho$ | smoothness prior | {term}`stacking` |
|---|---|---|---|---|
| astronomy | EM | Dirichlet Gibbs sampling; mean-field VB (RAIL) | logistic GP, GP, random walk | stacked $p(z)$ |
| gravitational waves | maximum hyper-likelihood | NUTS over binned log-rates | ICAR, GP, P-splines | — |
| particle-physics unfolding | iterative Bayesian unfolding (D'Agostini, Richardson–Lucy) | Fully Bayesian Unfolding | Tikhonov, GP unfolding | — |
| empirical Bayes | NPMLE (Kiefer–Wolfowitz) | variational Dirichlet posterior (ashr `mixVBEM`) | log-spline $g$ (Efron) | — |
| ML quantification | EMQ, SLD, MLLS | BayesianCC, BayesianMAPLS, BayesianKDEy | ordinal Tikhonov penalties | probabilistic classify and count (PCC) |
| ecology and genetics | conditional maximum likelihood (SPAM, ONCOR); EM (salmon) | BAYES, rubias, SourceTracker (Gibbs) | — | — |

Each field's point estimator is EM, or a truncation or acceleration of it
([theory](../how-it-works/point-estimates.md#em-is-iterated-stacking)).

## What binference adds in every field

Each field page lists only what binference adds for that field. These additions hold
for all of them:

- binference samples the exact posterior of $\rho$ with NUTS or Gibbs, and flags
  approximations ({doc}`engines guide <../guides/engines>`).
- It divides the {term}`interim prior` out of posteriors and posterior samples
  ({doc}`inputs guide <../guides/inputs/index>`).
- It models {term}`selection` on the truth or on the measurement
  ({doc}`selection guide <../guides/selection>`).
- It infers a tallied {term}`response matrix` jointly with the fractions
  ([binned measurements](../guides/inputs/binned.md)).
- It checks a whole analysis by simulation-based calibration
  ({doc}`checking guide <../guides/checking>`).

## Pages

```{toctree}
:maxdepth: 1

astronomy
gravitational-waves
unfolding
empirical-bayes
quantification
ecology
neural
simplex-priors
software
```

## Source material

These pages condense a census of the software and literature made while binference
was designed. The census keeps every citation, the code read and the numerical checks.
The repository holds ten research notes in
[`research_notes/`](https://github.com/joshspeagle/binference/tree/main/research_notes)
and a synthesis report in
[`reports/`](https://github.com/joshspeagle/binference/tree/main/reports). The design
notes include the report as {doc}`../design/landscape`. Nobody updates the notes; they
describe each project as it was in October 2026.
