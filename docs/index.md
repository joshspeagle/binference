# binference

**Infer the population, then every object.**

Many objects are each measured with noise. For each one you have a likelihood,
a posterior computed under some prior, posterior samples, counts in measured
bins with a response, or a classifier's output. binference infers the
distribution of the population the objects come from, with calibrated
uncertainty, and the posterior of each object under that population. The same
problem appears under different names in several fields
([background](fields/index.md)); binference treats them as one model
([what binference does](start/what-it-does.md)).

:::{container} paths
- **Measurements with errors.** A value and its uncertainty for each object: the
  [quick start](start/quickstart.md), then [Gaussian measurements](guides/inputs/gaussian.md).
- **Posteriors or posterior samples.** Each computed under some prior: the
  [posterior-samples tutorial](tutorials/posterior-samples.md), then
  [posteriors](guides/inputs/posteriors.md) and [samples](guides/inputs/samples.md).
- **Counts or classifier outputs.** Counts in measured bins with a response, or scores
  from a classifier: the [unfolding](tutorials/unfolding.md) and
  [classifier](tutorials/classifier-outputs.md) tutorials, then
  [binned measurements and classifiers](guides/inputs/binned.md).
:::

```{image} _static/quickstart.png
---
alt: >-
  Population posteriors for a mock catalogue under a flat Dirichlet prior and a
  Gaussian-process prior
class: figure-max
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

:::{container} sections
- **[Start](start/what-it-does.md)** What binference does, whether it suits your data, how
  to install it, and a first fit.
- **[Concepts](concepts/index.md)** Bins and likelihoods, posteriors, the population and its
  objects, priors, selection and engines, in words and figures.
- **[Tutorials](tutorials/index.md)** Executed examples from end to end, grouped by the
  kind of input.
- **[Guides](guides/index.md)** Prepare inputs, choose bins and priors, correct for
  selection, then run, read and check a fit.
- **[How it works](how-it-works/index.md)** The model, point estimates, priors, the engines,
  how `auto` chooses, and validation.
- **[Fields](fields/index.md)** The same model in astronomy, gravitational waves, unfolding,
  empirical Bayes, quantification, ecology and neural deconvolution.
- **[Reference](reference/api/index.md)** Every public class and function, the glossary,
  notation, defaults and warnings.
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
