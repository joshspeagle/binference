# Background

binference fits one statistical model. Several fields arrived at the same model
independently, gave it their own names, and wrote their own software for it. The pages
in this section describe, field by field, the problem in that field's terms, how it maps
onto binference's inputs, the methods and software the field uses, and the mistakes the
field has documented. They are reference material for users who know one of these
fields and want to see where binference fits, and for anyone choosing between tools.

The facts about other packages (versions, release dates, maintenance status, test
suites) were recorded on 22 October 2026 and are not updated with each release of
binference. Statements about binference describe the current code.

## One model under many names

Every field below works with $N$ objects, a set of $K$ bins or classes, and for each
object $i$ a likelihood $L_{ik}$: the probability of its data if its true value lies in
bin $k$. The population is a vector of fractions $\rho$ on the simplex, and the
likelihood of the whole catalogue is

$$
p(D\mid\rho)=\prod_{i=1}^{N}\sum_{k=1}^{K}L_{ik}\,\rho_k .
$$

The {doc}`theory page <../how-it-works/theory>` derives this model, its equivalent hierarchical form
with a latent bin label per object, and its extensions (priors, selection effects,
rates). The table names the same pieces in each field.

| binference | astronomy | gravitational waves | particle-physics unfolding | empirical Bayes | ML quantification | genetics and genomics |
|---|---|---|---|---|---|---|
| likelihood $L_{ik}$ | binned photo-$z$ likelihood; a star's likelihood in a colour–magnitude kernel | per-event weights: posterior samples divided by the sampling prior | row of the response matrix for the measured bin | likelihood matrix $L$ on a grid of support points | classifier posterior divided by the training prevalence | genotype likelihood under each baseline population; read compatibility |
| point estimate of $\rho$ | stacking (one EM step); EM | maximum hyper-likelihood | iterative Bayesian unfolding (D'Agostini, Richardson–Lucy) | NPMLE (Kiefer–Wolfowitz) | EMQ, SLD, MLLS | conditional maximum likelihood (SPAM, ONCOR); EM and VBEM (salmon) |
| posterior over $\rho$ | Dirichlet Gibbs sampling; mean-field VB (RAIL) | NUTS over binned log-rates | Fully Bayesian Unfolding | variational Dirichlet posterior (ashr `mixVBEM`) | BayesianMAPLS, BayesianKDEy | BAYES, rubias, SourceTracker (Gibbs) |
| interim prior | CHIPPR; importance weights | `prior` column divided out | dependence on the simulation prior | none found | division by the training prevalence | none found |
| selection | completeness; Poisson-process likelihood | injections, detection probability | efficiency and acceptance | truncation (deconvolveR) | none found | none found |
| smoothness | logistic GP, GP, random walk | ICAR, GP, P-splines | Tikhonov, GP unfolding | log-spline $g$ (Efron) | ordinal Tikhonov penalties | none found |

One identity links the point estimators across fields. The maximum-likelihood $\rho$
satisfies $\rho_k=N^{-1}\sum_i r_{ik}$, where $r_{ik}\propto L_{ik}\rho_k$ is object
$i$'s posterior over bins under the population $\rho$: it is the fixed point of
"average the per-object posteriors computed under the current population". Averaging
posteriors computed under an interim prior (stacking) is one step of this iteration
from that prior ([theory](../how-it-works/theory.md#point-estimates-em-is-iterated-stacking)).
Iterative Bayesian unfolding stopped early is the same iteration truncated, and EMQ in
quantification is the same iteration with the training prevalence as the starting point.
binference's `method="map"` finds the fixed point itself (by mix-SQP or EM) and reports
how far it is from the optimum; its default engine samples the posterior of $\rho$
instead.

## Pages

```{toctree}
:maxdepth: 1

astronomy
gravitational-waves
unfolding
empirical-bayes
quantification
neural
simplex-priors
software
```

## Source material

These pages are condensed from a census of the software and literature made while
binference was designed. The census itself, with every citation, the code read and the
numerical checks behind it, is kept in the repository as downloadable source material:
the ten research notes in
[`research_notes/`](https://github.com/joshspeagle/binference/tree/main/research_notes)
(one per field, plus notes on interface design and validation practice and an audit of
the `frankenz` samplers) and the synthesis report in
[`reports/`](https://github.com/joshspeagle/binference/tree/main/reports). The report
is also included in the design notes as {doc}`../design/landscape`. The notes record
the state of each project in October 2026 and are not maintained.
