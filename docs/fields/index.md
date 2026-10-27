# Fields

Which name does your field give to binference's model, and which tools does it use?
Several fields arrived at the same model on their own. Each gave it its own names and
wrote its own software. The pages in this section take one field each. They state the problem in that field's terms and map it onto binference.
They list the field's methods and software, and the mistakes it has documented.

The facts about other packages (versions, release dates, maintenance, tests) were
recorded on 22 October 2026. They are not updated with each release of binference.
Statements about binference describe the current code.

## One model under many names

Every field below has $N$ {term}`objects <object>`, $K$ {term}`bins <bin>` or classes,
and for each object $i$ a {term}`bin-averaged likelihood` $L_{ik}$. The
{term}`population fractions` $\rho$ lie on the {term}`simplex`, and the likelihood of
the whole catalogue is

$$
p(D\mid\rho)=\prod_{i=1}^{N}\sum_{k=1}^{K}L_{ik}\,\rho_k .
$$

The {doc}`theory page <../how-it-works/theory>` derives this model and its extensions.
This table is the one home of each field's names for its pieces. Field pages use these
names only on their own page.

| binference | astronomy | gravitational waves | particle-physics unfolding | empirical Bayes | ML quantification | ecology and genetics |
|---|---|---|---|---|---|---|
| object | galaxy, star, planet | event | event | unit, observation | item, document | fish, turtle, read |
| bin | redshift bin, colour–magnitude cell | bin of masses, spins and redshift | true bin | support point, grid point | class | stock, source population, transcript |
| population fractions $\rho$ | redshift distribution $n(z)$ (a density), occurrence rate, luminosity function | population model; rates in bins | true spectrum | prior $g$, mixing distribution | prevalence, class proportions | mixture proportions, stock proportions |
| likelihood $L_{ik}$ | binned photo-$z$ likelihood; a star's likelihood in a colour–magnitude kernel | per-event weights: samples divided by the PE prior | row of the response matrix for the measured bin | likelihood matrix $L$ | classifier posterior divided by the training prevalence | genotype likelihood under each baseline; read compatibility |
| interim prior $\pi$ | prior of the photo-$z$ code; implicit training-set prior | PE prior | simulation prior | none | training prevalence | none |
| response matrix $R_{jk}$ | none | none | response matrix; migration matrix with efficiency | none | confusion matrix | none |
| selection $S_k$ | completeness | detection probability; $VT$ | efficiency and acceptance | truncation | none | none |
| hierarchical posterior | posterior shrunk by the population | population-informed (reweighted) event posterior | none | empirical-Bayes posterior | corrected per-item posterior | per-fish assignment probability |
| stacking | stacked $p(z)$ | none | none | none | probabilistic classify and count (PCC) | none |
| point estimate of $\rho$ | stacking (one EM step); EM | maximum hyper-likelihood | iterative Bayesian unfolding (D'Agostini, Richardson–Lucy) | NPMLE (Kiefer–Wolfowitz) | EMQ, SLD, MLLS | conditional maximum likelihood (SPAM, ONCOR); EM and VBEM (salmon) |
| posterior over $\rho$ | Dirichlet Gibbs sampling; mean-field VB (RAIL) | NUTS over binned log-rates | Fully Bayesian Unfolding | variational Dirichlet posterior (ashr `mixVBEM`) | BayesianCC, BayesianMAPLS, BayesianKDEy | BAYES, rubias, SourceTracker (Gibbs) |
| smoothness prior | logistic GP, GP, random walk | ICAR, GP, P-splines | Tikhonov, GP unfolding | log-spline $g$ (Efron) | ordinal Tikhonov penalties | none |

One identity links the point estimators across fields. The maximum-likelihood $\rho$
satisfies $\rho_k=N^{-1}\sum_i r_{ik}(\rho)$, where $r_{ik}$ are the
{term}`responsibilities`. In words, $\rho$ is the average of the objects' posteriors
computed under $\rho$ itself. {term}`Stacking` is one step of this iteration from the
interim prior. Iterative Bayesian unfolding stopped early is the iteration truncated.
EMQ is the iteration started from the training prevalence. The
[theory page](../how-it-works/theory.md#point-estimates-em-is-iterated-stacking)
derives the identity.

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
It is kept in the repository: ten research notes in
[`research_notes/`](https://github.com/joshspeagle/binference/tree/main/research_notes)
and a synthesis report in
[`reports/`](https://github.com/joshspeagle/binference/tree/main/reports). The report
also appears in the design notes as {doc}`../design/landscape`. The notes record each
project as it was in October 2026 and are not maintained.
