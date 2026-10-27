# binference

**Population and hierarchical inference with PDFs.**

Many objects are each measured with noise. For each one you have a likelihood,
a posterior computed under some prior, posterior samples, counts in measured
bins with a response, or a classifier's output. binference infers the
distribution of the population the objects come from, with calibrated
uncertainty, and the posterior of each object under that population. The same
problem appears under different names in several fields
([background](background/index.md)); binference treats them as one model
([why binference](vision.md)).

```{image} _static/quickstart.png
:alt: Population posteriors for a mock catalogue under a flat Dirichlet prior and a Gaussian-process prior
:width: 100%
```

```python
import binference

mock = binference.simulate.mock_1d(2000, rng=42)
result = binference.infer(mock.likelihoods, rng=1)   # method="auto" picks the engine
result.summary()
result.plot(truth=mock.truth.rho)
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Why binference
:link: vision
:link-type: doc
What it does, for whom, and the principles behind its design.
:::

:::{grid-item-card} Getting started
:link: install
:link-type: doc
Install binference and run your first fit.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Executed, end-to-end examples from several fields.
:::

:::{grid-item-card} User guide
:link: user-guide/index
:link-type: doc
Inputs and conventions, priors, selection, engines, results, performance and pitfalls.
:::

:::{grid-item-card} Theory
:link: theory
:link-type: doc
The model, its augmented form, EM as iterated stacking, selection effects and the samplers.
:::

:::{grid-item-card} API reference
:link: api/index
:link-type: doc
Every public class and function.
:::

:::{grid-item-card} Design notes
:link: design/index
:link-type: doc
Specification, test and implementation plans, and a survey of related software.
:::

:::{grid-item-card} Background
:link: background/index
:link-type: doc
The same model in astronomy, gravitational waves, unfolding, empirical Bayes, quantification and neural deconvolution.
:::
::::

```{toctree}
:hidden:
:maxdepth: 2

vision
install
tutorials/index
user-guide/index
theory
background/index
api/index
design/index
changelog
contributing
citing
```
