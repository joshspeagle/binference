# Tutorials

Each tutorial is a complete analysis that runs from simulated data to a result. They are
grouped by what you have for each object. Start with the group that matches your data;
the last group applies to any data.

The tutorials are executed every time the documentation is built, so their outputs are
exactly what the code produces. Each runs in under two minutes on a shared four-core
container with one BLAS thread per process.

## Gaussian measurements

A measured value and its error for each object.

- {doc}`gaussian-quickstart`: a redshift distribution, compared with stacking, with
  derived quantities and single galaxies.
- {doc}`priors-and-smoothness`: four priors on a sparse catalogue, and held-out objects
  to choose between them.
- {doc}`two-dimensions`: a population on a grid of two measured quantities.
- {doc}`empirical-bayes`: deconvolution, the NPMLE and shrinkage estimates for each unit.

## Posteriors on a grid

- {doc}`posteriors-on-a-grid`: a table of posteriors computed under an interim prior,
  and masses versus densities.

## Posterior samples

- {doc}`posterior-samples`: black-hole masses from parameter-estimation samples and their
  prior.

## Classifier outputs

- {doc}`classifier-outputs`: class fractions under label shift, from posteriors and the
  training prevalence, or from scores and a labelled set.

## Binned counts and a response

- {doc}`unfolding`: a spectrum unfolded from detector counts with a response matrix.

## Any likelihood table

- {doc}`mixed-stocks`: stock proportions of a fishery's catch from genotype likelihoods.

## Incomplete catalogues, and checking an analysis

- {doc}`selection-effects`: occurrence rates with a known completeness, and a cut on the
  measured value.
- {doc}`selection-from-injections`: a mass distribution with a selection measured by
  injections.
- {doc}`choosing-an-engine`: every engine on one problem.
- {doc}`validating-an-analysis`: exact posteriors, simulation-based calibration and
  coverage.

```{toctree}
:hidden:

gaussian-quickstart
priors-and-smoothness
two-dimensions
empirical-bayes
posteriors-on-a-grid
posterior-samples
classifier-outputs
unfolding
mixed-stocks
selection-effects
selection-from-injections
choosing-an-engine
validating-an-analysis
```
