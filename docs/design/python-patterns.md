# Patterns from scientific Python packages

binference's interface follows patterns that users of other inference packages praise.
The census behind the {doc}`field pages <../fields/index>` recorded them.

| package | pattern | in binference |
|---|---|---|
| emcee {cite:p}`foremanmackey2013,foremanmackey2019` | plain inputs; a sampler that runs again to continue | a second `engine.run(n)` continues the chains without a new warm-up |
| scikit-learn {cite:p}`buitinck2013` | hyperparameters in the constructor, `fit` returning `self`, `get_params` | {class}`~binference.PopulationEstimator`, without a scikit-learn dependency |
| dynesty, nautilus | configured objects, priors declared as objects | priors, selections and responses passed to one `infer` call |
| UltraNest | output that says which criteria were met | `result.check()` lists failed diagnostics and remedies |
| ArviZ 1.x | results that other tools read | `result.to_datatree()` in ArviZ's layout |
| SPEC 7 {cite:p}`vanderwalt2023` | one `rng` argument, a seed or a `Generator` | a separate stream per chain; legacy `RandomState` rejected |
| tinygp | the model as functions other samplers can call | {class}`~binference.PopulationModel`; `PopulationResult.from_samples` wraps outside draws |

A scikit-learn estimator takes a feature matrix, and binference takes a likelihood
matrix. Full compliance with scikit-learn therefore does not fit.

Validation follows the statistical-software literature: exact posteriors for small
problems, Geweke's "getting it right" test for Gibbs, and simulation-based calibration
{cite:p}`talts2018`. The calibration ranks the joint log-likelihood as well as each
fraction. Parameter ranks alone "could never detect large classes of problems including
when the posterior is equal to the prior" {cite:p}`modrak2025`.
