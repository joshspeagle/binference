# binference

**Population and hierarchical inference with PDFs.**

binference infers the distribution of a population, and the posteriors of the
individual objects in it, from noisy per-object information over a set of
bins: likelihoods, posteriors computed under an interim prior, posterior
samples, response matrices or classifier outputs. It samples the exact
hierarchical posterior by default, so the error bars on every bin, every
derived quantity and every object are honest.

```{image} _static/quickstart.png
:alt: Population posteriors for a mock catalogue under a flat Dirichlet prior and a Gaussian-process prior
:width: 100%
```

```python
import binference

mock = binference.simulate.mock_1d(2000, rng=42)
result = binference.infer(mock.likelihoods, rng=1)   # exact posterior with NUTS
result.summary()
result.plot(truth=mock.truth.rho)
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} Getting started
:link: install
:link-type: doc
Install binference and run your first fit.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
Executed, end-to-end examples: n(z), engines, priors, selection, unfolding, validation.
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
::::

```{toctree}
:hidden:
:maxdepth: 2

install
tutorials/index
user-guide/index
theory
api/index
design/index
changelog
contributing
citing
```
