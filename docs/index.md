---
layout: home
---

# binference

**Population inference from noisy measurements.**

Use likelihoods, posteriors, samples, binned counts or classifier outputs to
estimate a population distribution and its uncertainty. The inferred population
also informs each object's posterior. The same model applies across
[several fields](fields/index.md).

:::{container} paths
- **Measurements with errors.** Start with one value and its uncertainty per object.
  [Quick start](start/quickstart.md) · [Gaussian inputs](guides/inputs/gaussian.md)
- **Posteriors or samples.** Supply the interim prior used for each object.
  [Posterior inputs](guides/inputs/posteriors.md) · [Samples tutorial](tutorials/posterior-samples.md)
- **Counts or classifier outputs.** Use a detector response or a labelled set.
  [Unfolding](tutorials/unfolding.md) · [Classifiers](tutorials/classifier-outputs.md) ·
  [Input guide](guides/inputs/binned.md)
:::

## Fit a population distribution

::::{container} home-example
:::{container} home-example-code
Start with a measured value and its Gaussian error for each object.
The example simulates those two arrays.

```python
import binference as bf

bins = bf.Bins.linear(0, 3, 15)
mock = bf.simulate.mock_1d(
    2000, bins, rng=42
)
values = mock.measurements["mean"]
errors = mock.measurements["sigma"]

lk = bf.Likelihoods.from_gaussian(
    values, bins, errors
)
result = bf.infer(lk, rng=1)
print(result.summary())
```
:::

```{figure} _static/quickstart.png
:alt: Population posteriors under a Dirichlet prior and a Gaussian-process prior. Each panel compares the inferred density and credible intervals with the true and stacked densities.
:class: figure-max

For example, the simulated values can represent galaxy redshifts. The panels show
the population fitted under a Dirichlet prior and a Gaussian-process prior.
```
::::

:::{container} sections
- **[Start](start/what-it-does.md).** What binference does, whether it suits your data, how
  to install it, and a first fit.
- **[Concepts](concepts/index.md).** Bins and likelihoods, posteriors, the population and its
  objects, priors, selection and engines, in words and figures.
- **[Tutorials](tutorials/index.md).** Executed examples from end to end, grouped by the
  kind of input.
- **[Guides](guides/index.md).** Prepare inputs, choose bins and priors, correct for
  selection, then run, read and check a fit.
- **[How it works](how-it-works/index.md).** The model, point estimates, priors, the engines,
  how `auto` chooses, and validation.
- **[Fields](fields/index.md).** The same model in astronomy, gravitational waves, unfolding,
  empirical Bayes, quantification, ecology and neural deconvolution.
- **[Reference](reference/api/index.md).** Every public class and function, the glossary,
  notation, defaults, warnings, bibliography and changelog, and how to cite binference.
:::

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
:caption: Concepts
:hidden:

concepts/index
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

how-it-works/index
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
