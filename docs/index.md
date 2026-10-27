# binference

**Population and hierarchical inference with PDFs.**

Many objects are each measured with noise. For each one you have a likelihood,
a posterior computed under some prior, posterior samples, counts in measured
bins with a response, or a classifier's output. binference infers the
distribution of the population the objects come from, with calibrated
uncertainty, and the posterior of each object under that population. The same
problem appears under different names in several fields
([background](fields/index.md)); binference treats them as one model
([what binference does](start/what-it-does.md)).

```{image} _static/quickstart.png
---
alt: >-
  Population posteriors for a mock catalogue under a flat Dirichlet prior and a
  Gaussian-process prior
width: 100%
---
```

```python
import binference

bins = binference.Bins.linear(0.0, 3.0, 15)

# a simulated stand-in for your arrays: measured values and their Gaussian errors
mock = binference.simulate.mock_1d(2000, bins, rng=42)
z_obs, z_err = mock.measurements["mean"], mock.measurements["sigma"]

lk = binference.Likelihoods.from_gaussian(z_obs, z_err, bins)
result = binference.infer(lk, rng=1)  # method="auto" chooses the engine
print(result.summary())
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} What binference does
:link: start/what-it-does
:link-type: doc
The problem in pictures, what binference infers, and the same model under many names.
:::

:::{grid-item-card} Getting started
:link: start/install
:link-type: doc
Install binference and check that it runs.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Executed, end-to-end examples from several fields.
:::

:::{grid-item-card} Guides
:link: guides/index
:link-type: doc
Inputs and conventions, priors, selection, engines, results, performance and
pitfalls.
:::

:::{grid-item-card} Theory
:link: how-it-works/theory
:link-type: doc
The model, its augmented form, EM as iterated stacking, selection effects and the
samplers.
:::

:::{grid-item-card} API reference
:link: reference/api/index
:link-type: doc
Every public class and function.
:::

:::{grid-item-card} Fields
:link: fields/index
:link-type: doc
The same model in astronomy, gravitational waves, unfolding, empirical Bayes,
quantification, ecology and neural deconvolution.
:::
::::

```{toctree}
:caption: Start
:hidden:

start/what-it-does
start/is-it-right
start/install
start/quickstart
start/principles
```

```{toctree}
:caption: Tutorials
:hidden:

tutorials/index
```

```{toctree}
:caption: Guides
:hidden:

guides/index
```

```{toctree}
:caption: How it works
:hidden:

how-it-works/theory
```

```{toctree}
:caption: Fields
:hidden:

fields/index
```

```{toctree}
:caption: Reference
:hidden:

reference/api/index
reference/glossary
reference/notation
reference/defaults
reference/warnings
reference/bibliography
reference/changelog
reference/citing
```
