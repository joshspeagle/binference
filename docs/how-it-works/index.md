# How it works

These pages give the mathematics behind binference, for readers who want to know exactly
what the code computes. They use the symbols of the {doc}`notation page
<../reference/notation>`, and the {doc}`defaults page <../reference/defaults>` holds every
threshold they mention.

- {doc}`model`: the model, its form with one latent label per object, and every
  assumption in one list.
- {doc}`point-estimates`: EM as iterated stacking, mix-SQP, the optimality certificate,
  missing information, and why the mode depends on the coordinates.
- {doc}`priors`: the Dirichlet family as trees, the logistic family, the unconstrained
  coordinates and the hyperpriors.
- {doc}`engines`: NUTS, Gibbs, Laplace with importance sampling, mean-field VB and the
  bootstrap.
- {doc}`auto`: the rules of `method="auto"` and the benchmark behind them.
- {doc}`validation`: diagnostics, simulation-based calibration, coverage, and the
  statistical checks of the test suite.

```{toctree}
:hidden:

model
point-estimates
priors
engines
auto
validation
```
