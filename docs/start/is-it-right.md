# Is binference right for my data?

binference fits when three things hold. You have many {term}`objects <object>`. Each
object carries its own noisy information about one quantity, which you can divide into
{term}`bins <bin>` (intervals, grid cells or categories). And you want the
{term}`population` of that quantity, or each object's value informed by it. The tables
below match what you have and what you want to binference's inputs and outputs.

## What you have

Each row names the constructor that turns your data into a
{class}`~binference.Likelihoods`, then the guide section that explains it and the
tutorials that use it.

| You have, for each object | binference reads it with | Read more |
|---|---|---|
| a measured value and its Gaussian error | `Likelihoods.from_gaussian` | [guide](../guides/inputs.md#gaussian-measurements), {doc}`quick start <quickstart>`, {doc}`tutorial 9 <../tutorials/09_empirical_bayes>` |
| likelihoods or log-likelihoods on the bins | `Likelihoods.from_table` | [guide](../guides/inputs.md#likelihood-matrices), {doc}`tutorial 10 <../tutorials/10_mixed_stocks>` |
| a likelihood function or a SciPy distribution | `Likelihoods.from_density` | [guide](../guides/inputs.md#likelihood-functions-and-distributions) |
| a posterior on the bins, and the prior it was computed under | `Likelihoods.from_table` | [guide](../guides/inputs.md#posteriors-computed-under-an-interim-prior), {doc}`tutorial 2 <../tutorials/02_inputs>` |
| posterior samples, and the prior they were drawn under | `Likelihoods.from_samples` | [guide](../guides/inputs.md#posterior-samples), {doc}`tutorial 2 <../tutorials/02_inputs>` |
| the bin it was measured in, and the detector's {term}`response matrix` | `Likelihoods.from_binned` | [guide](../guides/inputs.md#binned-measurements-and-classifiers), {doc}`tutorial 6 <../tutorials/06_categories_unfolding>` |
| a classifier's probabilities, and the class prevalence it was trained on | `Likelihoods.from_table` | [guide](../guides/inputs.md#binned-measurements-and-classifiers), {doc}`tutorial 6 <../tutorials/06_categories_unfolding>` |
| a classifier's scores, and a labelled validation set | `Likelihoods.from_binned` with `Response.from_labelled` | [guide](../guides/inputs.md#binned-measurements-and-classifiers), {doc}`tutorial 6 <../tutorials/06_categories_unfolding>`, {doc}`tutorial 10 <../tutorials/10_mixed_stocks>` |

Posteriors and posterior samples need their {term}`interim prior`, which binference
divides out ([pitfalls](../guides/pitfalls.md#feeding-posteriors-in-as-likelihoods)).

## What you want

| You want | binference gives | Read more |
|---|---|---|
| the {term}`population`, with its uncertainty | `result.samples`, `result.summary`, `result.plot` | [guide](../guides/results.md#the-population), {doc}`tutorial 1 <../tutorials/01_quickstart>` |
| a number computed from the population, such as its mean | `result.functional` | [guide](../guides/results.md#the-population) |
| each object's value, informed by the population: its {term}`hierarchical posterior` | `result.assignment_probabilities`, `result.sample_values` | [guide](../guides/results.md#individual-objects), {doc}`tutorial 9 <../tutorials/09_empirical_bayes>` |
| the histogram of these objects, the {term}`realised sample` | `result.realized_counts` | [guide](../guides/results.md#population-versus-realised-sample) |
| totals and rates from an incomplete catalogue, given its {term}`selection function` | `infer`'s `selection`, then `result.total`, `result.rates` | {doc}`guide <../guides/selection>`, {doc}`tutorial 5 <../tutorials/05_selection>`, {doc}`tutorial 11 <../tutorials/11_populations_with_selection>` |
| a quick single answer, a {term}`point estimate` | `method="map"` | [guide](../guides/engines.md#point-estimates) |

## What binference does not do

- **Compute the per-object inputs.** binference starts where each object already has
  a likelihood, a posterior, samples or a measured bin. Fitting photometry, training a
  classifier or simulating a detector happens before it.
- **Use posteriors whose prior is unknown.** Without its interim prior, binference
  cannot turn a posterior back into a likelihood.
- **Fit a population without bins.** The population is a histogram on your bins,
  optionally with a finer B-spline {class}`~binference.Basis`. binference has no
  parametric population models, such as a power law, and no neural density models
  ({doc}`../fields/neural`).
- **Let the population depend on other variables.** All objects share one population;
  binference has no covariate-dependent populations.
- **Model correlated objects.** The model multiplies the objects' likelihoods, so it
  treats their data as independent given the population
  ([theory](../how-it-works/theory.md#the-population-model)).
