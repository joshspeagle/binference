# binference

**Population and hierarchical inference with PDFs.**

binference infers the distribution of a population, and the posteriors of the
individual objects in it, from noisy per-object information over a set of
bins: likelihoods, posteriors computed under an interim prior, posterior
samples, response matrices or classifier outputs. It samples the exact
hierarchical posterior by default, so the error bars on every bin, every
derived quantity and every object are honest.

![Population posteriors for a mock catalogue](https://raw.githubusercontent.com/joshspeagle/binference/main/docs/_static/quickstart.png)

*2000 objects whose Gaussian measurement errors are comparable to the bin
width. Stacking the per-object PDFs (dotted) gives one curve and no
uncertainty. With the default flat Dirichlet prior (left), binference shows how
poorly the data pin down individual bins, since neighbours trade off against
each other. A Gaussian-process prior on the log-density (right) shares
strength between bins.*

## The model

For objects $i = 1..N$ with data $d_i$ and bins $k = 1..K$, let
$L_{ik}$ be object $i$'s likelihood averaged over bin $k$ and $\rho$ the
population fractions. Then

$$p(\rho \mid \{d_i\}) \propto p(\rho)\prod_{i=1}^{N}\Big(\sum_{k=1}^{K} L_{ik}\,\rho_k\Big)^{w_i}.$$

This is the hierarchical model behind photometric-redshift $n(z)$ inference
(Leistedt, Mortlock & Peiris 2016) and Gaia colour–magnitude diagrams (Leistedt & Hogg 2017). The same
model appears elsewhere as iterative Bayesian unfolding in particle physics,
the NPMLE / $g$-modelling of empirical Bayes, EM quantification in machine
learning, and mixed-stock analysis in genetics. Stacking the per-object
posteriors computed under an interim prior is exactly one EM step from that
prior. binference runs the iteration to convergence and quantifies its uncertainty.

## Installation

binference is in alpha. Until it is released on PyPI as `binference`, install it
from GitHub:

```bash
pip install "binference[plot] @ git+https://github.com/joshspeagle/binference"
```

It needs Python ≥ 3.13, NumPy, SciPy and threadpoolctl; the `plot` extra adds matplotlib and
`xarray` adds export to ArviZ-style `DataTree`s.

## Quick start

```python
import binference

mock = binference.simulate.mock_1d(2000, rng=42)   # likelihoods on 30 bins + the true population
result = binference.infer(mock.likelihoods, rng=1)  # NUTS on the exact posterior (the default)

print(result.summary())
result.plot(truth=mock.truth.rho)
```

```text
binference fractions posterior (nuts, 4000 draws)
       bin       mean         sd     median        q16       q84  ess_bulk    rhat
  [0, 0.1)   0.005989   0.003696   0.005572   0.002163  0.009778      2117       1
[0.1, 0.2)    0.01365    0.00797    0.01306   0.005212   0.02178      1873   1.001
[0.2, 0.3)    0.02388      0.013    0.02345   0.009582   0.03735      1768   1.001
...
```

Smoothness priors infer their own amplitude and length scale:

```python
smooth = binference.infer(mock.likelihoods, binference.priors.LogisticGP(), rng=1)
```

## Your own data

```python skip
bins = binference.Bins.linear(0.0, 3.0, 30)          # also Bins.log, Bins.grid (N-D), Bins.categories

lk = binference.Likelihoods.from_gaussian(z_obs, z_err, bins)              # Gaussian errors
lk = binference.Likelihoods.from_table(P, bins, interim=interim)       # posteriors under a known prior
lk = binference.Likelihoods.from_samples(samples, bins, interim="flat")  # posterior samples
lk = binference.Likelihoods.from_table(L, bins, kind="likelihood")         # any (N, K) likelihood table

result = binference.infer(lk)
```

Other constructors cover arbitrary likelihood functions and SciPy
distributions (`from_density`), and binned measurements with a response
(`from_binned`: detector unfolding, or classifier scores with a labelled validation
set).
Per-object posteriors are only accepted with the prior they were computed
under, because feeding them in as likelihoods biases the population toward
that prior.

## What you get back

```python skip
result.mean, result.std, result.interval(0.68)     # per-bin summaries (also kind="hdi")
result.functional(lambda rho: rho @ bins.centers)  # any function of rho, evaluated per draw
result.rebin(mapping)                              # coarser bins with correlations propagated
result.assignment_probabilities()                  # each object's posterior over bins
result.sample_values(idx)                          # draws of individual objects' true values
result.realized_counts()                           # histogram of *this* sample vs. the population
result.diagnostics, result.check()                 # R-hat, ESS, divergences, warnings
result.save("fit.npz"); result.to_datatree()       # round-trip to disk; ArviZ-compatible export
```

## Priors, selection and rates

* **Dirichlet family.** `Dirichlet(alpha)` is the default with `alpha=1`. `Dirichlet.reference()`
  gives `alpha=1/K`, and there are `from_counts` and `from_base_measure` constructors.
  `GeneralizedDirichlet`, `DirichletTree` and `PolyaTree` are conjugate too, so Gibbs sampling works.
* **Smoothness** on ordered bins and N-D grids: `LogisticRandomWalk`, `LogisticGP`
  and `LogisticNormal`, with hyperparameters inferred.
* **Selection effects.** Pass `selection=S` with `selection_mode="on_measurement"` (detection
  efficiency) or `"on_truth"` (completeness in the true bin).
* **Rates.** `GammaRates` gives a Poisson process with exposure, for counts as well as fractions.

## Engines

| `method=` | returns | use it for |
|---|---|---|
| `"nuts"` (default) | exact posterior draws | any prior; mixes well even when likelihoods are broader than the bins |
| `"gibbs"` | exact draws and latent bin labels | conjugate priors; fast when likelihoods are narrow |
| `"map"` / `"em"` | point estimate (maximum likelihood by default) with an optimality certificate | quick looks, initialisation; seconds for 10⁵ objects |
| `"laplace"` | Gaussian approximation, importance-corrected (PSIS) | very large N, checked by Pareto k̂ |
| `"vb"` | mean-field variational fit | speed only: it is over-confident and flagged as approximate |
| `"bootstrap"` | sampling distribution of the MAP | frequentist comparisons |

For scikit-learn pipelines there is `binference.PopulationEstimator(prior, method=...)`,
with `fit`, `predict_proba`, `score` and `sample`. NUTS can run its chains in
parallel with `binference.infer(lk, n_jobs=-1)`.

## Validation is part of the package

* `binference.validation.exact_posterior` computes exact posteriors by quadrature (K ≤ 3).
* `binference.validation.sbc` runs simulation-based calibration.
* `binference.validation.coverage` measures frequentist coverage.
* `binference.simulate` builds mock catalogues in 1-D and 2-D, plus categorical, unfolding and
  prior-predictive mocks.

The test suite checks every engine against exact posteriors, and the slow suite
(`pytest -m slow`) adds simulation-based calibration, coverage and Geweke tests.

## Documentation

The [documentation](https://joshspeagle.github.io/binference/) contains tutorials, a user
guide, the theory and the API reference. Its source is in
[`docs/`](https://github.com/joshspeagle/binference/tree/main/docs); build it with:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs docs/_build/html
```

The design documents are in [`docs/design/`](https://github.com/joshspeagle/binference/tree/main/docs/design): the
[specification](https://github.com/joshspeagle/binference/blob/main/docs/design/spec.md), the
[test plan](https://github.com/joshspeagle/binference/blob/main/docs/design/test-plan.md) and the
[implementation plan](https://github.com/joshspeagle/binference/blob/main/docs/design/implementation-plan.md). The survey of
existing tools, including an audit of the samplers in `frankenz`, is in
[`reports/`](https://github.com/joshspeagle/binference/tree/main/reports).

## Status

binference is alpha software, and the API may change before 1.0.
Planned work:
* a JAX backend;
* normalising-flow and flow-matching populations for unbinned data;
* hierarchical models for reference samples;
* GPU-scale likelihood matrices.

## Citing

If you use binference, please cite it using [`CITATION.cff`](https://github.com/joshspeagle/binference/blob/main/CITATION.cff), and cite
the method papers that apply to your problem. For photometric redshifts that is
Leistedt, Mortlock & Peiris (2016, MNRAS 460, 4258), and the general framework
is Hogg, Myers & Bovy (2010, ApJ 725, 2166).

## License

MIT; see [LICENSE](https://github.com/joshspeagle/binference/blob/main/LICENSE).
