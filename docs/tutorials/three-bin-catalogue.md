---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# A three-bin catalogue from input to posterior

Two galaxies have uncertain redshifts. Their photo-redshift fitter supplies a
posterior mass in each bin and the interim prior it used. Follow the first
galaxy's numbers into a likelihood row, then fit the population and check the result.

## Recover the likelihoods

The redshift bins have widths 1, 1, and 2. The table entries are probabilities
integrated over each bin. For the first galaxy, the mass in the last bin is
$1/8$, even though its density there is only $1/16$ per unit redshift.

```{code-cell} python
from itertools import product

import numpy as np

import binference

bins = binference.Bins([0., 1., 2., 4.])
interim = np.array([.2, .3, .5])
posterior_mass = np.array([[4, 3, 1], [4, 15, 50]]) / np.array([[8], [69]])
lk = binference.Likelihoods.from_table(
    posterior_mass, bins, interim=interim,
    representation="mass", interim_representation="mass",
)
print("relative likelihoods:", lk.toarray(), sep="\n")
```

Divide the first row's masses by $(0.2,0.3,0.5)$, then divide by its
largest entry, $2.5$. The row is $(1,0.5,0.1)$. These likelihoods can
be rescaled per object. Their entries need not add to one.

```{code-cell} python
L = np.array([[1., .5, .1], [.2, .5, 1.]])
np.testing.assert_allclose(lk.toarray(), L, rtol=1e-14)
```

## Put a population behind the rows

Suppose the population fractions are $(1/4,1/2,1/4)$. For the first
galaxy the three products $L_{1k}\rho_k$ are $(1/4,1/4,1/40)$.
They add to $21/40$. Normalising gives probabilities
$(10/21,10/21,1/21)$ for its true bin.

```{code-cell} python
trial = np.array([.25, .5, .25])
prior = binference.priors.Dirichlet([1., 1., 1.])
model = binference.PopulationModel(lk, prior)
responsibilities = model.responsibilities(trial)
by_hand = np.array([[10/21, 10/21, 1/21], [1/11, 5/11, 5/11]])
np.testing.assert_allclose(responsibilities, by_hand, rtol=1e-14)
np.testing.assert_allclose(model.expected_counts(trial), by_hand.sum(axis=0), rtol=1e-14)
gradient = (L / (L @ trial)[:, None]).sum(axis=0)
np.testing.assert_allclose(gradient[0] - gradient[-1], 20/77, rtol=1e-14)
print("conditional bin probabilities:", responsibilities.round(3), sep="\n")
print("one EM step:", (model.expected_counts(trial) / model.W).round(3))
```

The Dirichlet prior is uniform over the fractions. Equal prior mean fractions put
less density per unit redshift in the wider bin.

One EM step averages the two rows. It gives a new population estimate for
the next step. To quantify population uncertainty, fit the posterior over
the fractions and average each galaxy's probabilities over those draws.

## Diagnose a short fit

Here we report the posterior mean fraction in the last bin. Start with
seven draws per chain to see what an inadequate run returns.

```{code-cell} python
short = binference.infer(
    lk, prior, method="gibbs", n_chains=4, n_warmup=80, n_samples=7, rng=4, check=False,
)
short.check()
```

The chains are too short for R-hat and effective sample size. The simulation
seed and model can stay fixed while we collect more draws. Look at the
checks and at the Monte Carlo error of the quantity we want to report.

```{code-cell} python
fit = binference.infer(
    lk, prior, method="gibbs", n_chains=4, n_warmup=80, n_samples=2000, rng=4,
)
print("check:", fit.check())
print("draws:", fit.samples.shape, "chains:", fit.chains.shape)
print("last-bin fraction:", round(fit.mean[-1], 3))
print("Monte Carlo standard error:", round(binference.diagnostics.mcse(fit.chains[..., -1]), 4))
print("object posteriors:", fit.assignment_probabilities(n_draws=None).round(3), sep="\n")
```

The Monte Carlo error measures the precision of the reported mean due to the
finite fit. The posterior standard deviation measures uncertainty about the
fraction itself. Longer chains reduce the first of those uncertainties.

### Check against the exact posterior

With this Dirichlet prior and two objects, there are nine pairs
of labels. For a pair with counts $n$, its contribution is proportional to
the product of its two likelihood entries times
$\mathbb E[\rho_{k_1}\rho_{k_2}]$ under the prior. That moment is $1/6$
for equal labels and $1/12$ for different labels. Conditional on the pair,
the posterior is $\mathrm{Dirichlet}(1+n)$.

```{code-cell} python
weights, conditional_means = [], []
for first, second in product(range(bins.K), repeat=2):
    counts = np.bincount([first, second], minlength=bins.K)
    moment = (1 + int(first == second)) / 12
    weights.append(L[0, first] * L[1, second] * moment)
    conditional_means.append((1 + counts) / 5)
exact_mean = np.average(conditional_means, axis=0, weights=weights)
print("exact posterior means:", exact_mean.round(3))
print("sampled minus exact:", (fit.mean - exact_mean).round(3))
```

The exact mean isolates sampling error because the likelihoods and prior are
identical in the two calculations. The two galaxies have uncertain true bins.
Their accumulated count in the last bin is an integer from zero to two.
The population fraction in that bin is a different quantity.

```{code-cell} python
last_counts = fit.realized_counts(n_draws=500, rng=8)[:, -1]
print("possible last-bin counts in the draws:", np.unique(last_counts).astype(int))
```

## Check the observation model

Sampler diagnostics and exact enumeration assess the specified model. For a
separate check of the observation model, suppose a detector records the same
three true bins in four measured bins. Its calibration predicts a 3% chance
of the fourth measured bin for every true bin. No choice of population fractions
can change that prediction.

```{code-cell} python
response = binference.Response(
    [[.70, .16, .11], [.17, .66, .16], [.10, .15, .70], [.03, .03, .03]],
    bins, binference.Bins.categories(["A", "B", "C", "D"]),
)
for measured_counts in ([28, 41, 28, 3], [18, 26, 16, 40]):
    observed = binference.Likelihoods.from_binned(response, counts=measured_counts)
    checked = binference.infer(
        observed, prior, method="gibbs", n_chains=4, n_warmup=80, n_samples=2000,
        rng=6, check=False,
    )
    print("fourth-bin count:", measured_counts[-1])
    print("predictive p-value:", f"{response.predictive_pvalue(measured_counts, checked.samples[::20]):.3g}")
    print("check:", checked.check())
```

Forty of a hundred detections in the fourth bin conflict with the response even
if the chains mix. Longer runs cannot repair that discrepancy. Check that the
detector calibration applies to the target catalogue, including its binning and
selection. For a classifier, compare the scores at fixed true class using
labelled target data. Use {doc}`SBC and coverage <validating-an-analysis>` to check
repeated fits under a specified data generator.

## Follow the calculation in the code

| Step | Public entry point | Implementation |
|---|---|---|
| Divide out the interim prior | {meth}`~binference.Likelihoods.from_table` | [Likelihood construction](https://github.com/joshspeagle/binference/blob/14edf32b8e8fce9e0c4d02bb1478ea87c8494082/src/binference/likelihoods.py#L794) |
| Evaluate likelihoods and expected counts | {meth}`~binference.PopulationModel.expected_counts` | [Model call](https://github.com/joshspeagle/binference/blob/14edf32b8e8fce9e0c4d02bb1478ea87c8494082/src/binference/model.py#L709) and [numerical kernel](https://github.com/joshspeagle/binference/blob/14edf32b8e8fce9e0c4d02bb1478ea87c8494082/src/binference/_kernels.py#L84) |
| Draw latent counts and update the fractions | {class}`~binference.engines.Gibbs` | [Gibbs update](https://github.com/joshspeagle/binference/blob/14edf32b8e8fce9e0c4d02bb1478ea87c8494082/src/binference/engines/gibbs.py#L158) |
| Average the conditional probabilities over draws | {meth}`~binference.PopulationResult.assignment_probabilities` | [Result method](https://github.com/joshspeagle/binference/blob/14edf32b8e8fce9e0c4d02bb1478ea87c8494082/src/binference/results.py#L498) |

The [model derivation](../how-it-works/model.md#three-bins-by-hand) takes these same
rows through the gradient. The {doc}`larger redshift example <posteriors-on-a-grid>`
compares fits with and without the interim-prior correction.
