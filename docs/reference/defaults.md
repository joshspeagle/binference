# Defaults

This page lists the thresholds and run-time defaults that the code sets as named
constants; each function's parameter defaults are in its signature on the
{doc}`API pages <api/index>`. The tables are generated from the code. Every engine
takes `rng=None`, which draws fresh entropy from the operating system: pass a seed for
reproducible draws. Names that start with an underscore are internal: they are listed to
explain behaviour, not as settings to change.

<!-- defaults:start -->

## Automatic engine choice

| Name | Value | Module | What it controls |
|---|---|---|---|
| `EXTEND_TARGET` | `1.2` | `binference.inference` | An extension of a Gibbs run aims for this many times `target_ess`. |
| `EXTEND_MAX_FACTOR` | `3` | `binference.inference` | One extension of a Gibbs run adds at most this many times the draws so far. |
| `EXTEND_MIN_FACTOR` | `0.5` | `binference.inference` | One extension of a Gibbs run adds at least this many times the draws so far. |
| `PROBE_PROPOSALS` | `1000` | `binference.inference` | Proposals of the first pass that can stop a Laplace trial early. |
| `PROBE_DRAWS` | `500` | `binference.inference` | Draws of the first pass that can stop a Laplace trial early. |
| `PROBE_ESS_SHARE` | `0.5` | `binference.inference` | A Laplace trial stops when its first pass projects an importance ESS below this share of `target_ess`. |

## Results and diagnostics

| Name | Value | Module | What it controls |
|---|---|---|---|
| `TAIL_QUANTILES` | `(0.05, 0.95)` | `binference.diagnostics` | Quantiles whose indicators define the tail ESS. |
| `K_HAT_MAX` | `0.7` | `binference.diagnostics` | Pareto k-hat at or above which importance sampling is unreliable, reached from about 2,000 weights; with n weights the limit is min(1 - 1/log10(n), K_HAT_MAX). |
| `PSIS_MIN_WEIGHTS` | `5` | `binference.diagnostics` | Pareto smoothing needs at least this many importance weights. |
| `PSIS_TAIL` | `(0.2, 3)` | `binference.diagnostics` | The generalised Pareto is fitted to the ceil(min(a n, b sqrt(n))) largest of n weights, for (a, b) these. |
| `N_DRAWS` | `500` | `binference.results` | Population draws the per-object methods use by default. |
| `RHAT_MAX` | `1.01` | `binference.results` | Largest split R-hat that `PopulationResult.check` accepts by default. |
| `ESS_MIN_PER_CHAIN` | `100` | `binference.results` | Smallest bulk and tail ESS per chain that `PopulationResult.check` accepts by default. |
| `MIXING_MIN_DRAWS` | `8` | `binference.results` | Chains shorter than this many draws get no R-hat or ESS. |
| `EBFMI_MIN` | `0.3` | `binference.results` | `PopulationResult.check` warns when a NUTS chain's E-BFMI is below this. |
| `MAX_DEPTH_SHARE` | `0.05` | `binference.results` | `PopulationResult.check` warns when a larger share of NUTS iterations hit `max_depth`. |
| `SAMPLE_VARIANCE_MAX` | `1` | `binference.results` | `PopulationResult.check` warns when the summed Monte Carlo variance of likelihoods from samples exceeds this. |
| `EVAL_BUDGET` | `16777216` | `binference.results` | Most values a density function returns per call when values are drawn (2**24 floats are 128 MB). |

## Engines

| Name | Value | Module | What it controls |
|---|---|---|---|
| `_INIT_BUFFER` | `75` | `binference.samplers._nuts_core` | Iterations of the first fast phase of Stan-style warm-up. |
| `_BASE_WINDOW` | `25` | `binference.samplers._nuts_core` | Iterations of the first slow window of Stan-style warm-up; each next one is twice as long. |
| `_TERM_BUFFER` | `50` | `binference.samplers._nuts_core` | Iterations of the last fast phase of Stan-style warm-up. |
| `INTERIOR_START_ITER` | `50` | `binference.samplers.base` | Most regularised EM steps of the start of Gibbs and of MAP's EM and L-BFGS paths when `init` is `None`. |
| `PROPOSAL_FACTOR` | `2` | `binference.samplers.laplace` | Importance proposals per draw when `n_proposals` is `None` (at least `diagnostics.PSIS_MIN_WEIGHTS` in all, the fewest Pareto smoothing takes). |
| `EIGEN_FLOOR` | `1e-08` | `binference.samplers.laplace` | Eigenvalues of a Hessian that is not positive definite are raised to this times the largest (at least one). |
| `_TARGET_ACCEPT` | `0.8` | `binference.samplers.nuts` | `target_accept` when it is `None`, except for logistic priors. |
| `_TARGET_ACCEPT_LOGISTIC` | `0.9` | `binference.samplers.nuts` | `target_accept` for logistic priors when it is `None`. |
| `_DENSE_MAX_DIM` | `200` | `binference.samplers.nuts` | `metric="auto"` is dense for logistic priors up to this many dimensions, and up to this many the warm-up curvature is averaged over every chain's point, not taken at one. |
| `_HESSIAN_MAX_DIM` | `500` | `binference.samplers.nuts` | Up to this many dimensions, warm-up starts the metric from the curvature at the mode. |
| `_CENTRING_MAX_SD` | `0.28` | `binference.samplers.nuts` | Logistic priors stay non-centred when the median spread of the modes' log-scales given the field exceeds this. |
| `_GEOMETRY_ITER` | `25` | `binference.samplers.nuts` | Warm-up iteration of the partial centring of logistic priors, or the first slow window's start if that is earlier. |
| `_DUAL_AVERAGING` | `(10, 0.05, 0.75)` | `binference.samplers.nuts` | `t0`, `gamma` and `kappa` of the dual-averaging step-size adaptation. |
| `_ROUNDOFF` | `1000` | `binference.samplers.optimize` | A MAP run that can no longer improve its objective in floating point is converged when its optimality gap is below this many times `tol`. |
| `_MIXSQP_START_ITER` | `10` | `binference.samplers.optimize` | Regularised EM steps of mix-SQP's start when `init` is `None`. |

## Priors

| Name | Value | Module | What it controls |
|---|---|---|---|
| `_TINY` | `1e-300` | `binference.priors.base` | Floor of drawn fractions and rates, so that their logarithms stay finite. |
| `_KERNELS` | `{'rbf': (None, 1.3), 'matern52': (2.5, 1.8), 'matern32': (1.5, 2.2), 'matern12': (0.5, 8.5)}` | `binference.priors.logistic` | Smoothness `nu` (`None` for the RBF) and resolution factor `f` of each kernel of `LogisticGP`: `m` sines resolve length scales down to `f` times the box half-width over `m`. |
| `_MAX_BASIS` | `1024` | `binference.priors.logistic` | Most basis functions of `LogisticGP` in all when `n_basis` is `None`. |

## Inputs

| Name | Value | Module | What it controls |
|---|---|---|---|
| `DEFAULT_CHUNK` | `4096` | `binference._kernels` | Objects per block of the chunked likelihood kernels. |
| `MIN_EVIDENCE` | `1e-250` | `binference._kernels` | An object whose normalised evidence is below this is impossible. |
| `SPARSE_FILL` | `0.1` | `binference._kernels` | Likelihoods are stored sparse below this share of non-zero entries. |
| `MAX_ELEMENTS` | `4194304` | `binference._quadrature` | Most function values a quadrature evaluates in one call. |
| `EDGE_INSET` | `1e-09` | `binference._quadrature` | How far inside a cell, as a fraction of its width, the check rule reads its ends: a jump closer to an edge goes unseen, and one exactly on an edge reads as the inside value. |
| `DEFAULT_OVERSAMPLE` | `{1: 8, 2: 4}` | `binference.basis` | Quadrature cells per order-1 component and axis when `oversample` is `None`, by dimension (2 above two dimensions; order 0 takes 1). |
| `OUTSIDE_THRESHOLD` | `0.9` | `binference.likelihoods` | An object with more than this share of its likelihood outside the bins is mostly outside. |
| `OUTSIDE_SHARE` | `0.1` | `binference.likelihoods` | A warning is raised when more than this share of the objects are mostly outside the bins. |
| `FEW_SAMPLES_SHARE` | `0.1` | `binference.likelihoods` | `from_samples` warns when more than this share of the objects have fewer than `min_samples` effective samples. |
| `REPRESENTATION_SHARE` | `0.5` | `binference.likelihoods` | A warning is raised when more than this share of posterior rows sum to one only under the other `representation`. |
| `REPRESENTATION_RTOL` | `0.01` | `binference.likelihoods` | Relative tolerance within which a posterior row counts as summing to one. |
| `MIGRATION_ATOL` | `0.001` | `binference.response` | With `efficiency`, columns that sum to one within this are divided by their sum (a migration matrix off by rounding); a warning is raised for the others. |

## Selection

| Name | Value | Module | What it controls |
|---|---|---|---|
| `STEEP_RATIO` | `2` | `binference.selection` | A selection that changes by more than this factor between neighbouring bins raises a warning. |
| `S_FLOOR` | `1e-12` | `binference.selection` | Smallest detection probability a draw of an inferred selection takes. |

## Validation

| Name | Value | Module | What it controls |
|---|---|---|---|
| `SBC_MAX_BINS` | `20` | `binference.validation` | Most bins of an SBC rank histogram when `n_bins` is `None`. |
| `SBC_SIMULATIONS_PER_BIN` | `5` | `binference.validation` | Fewest simulations per bin of an SBC rank histogram when `n_bins` is `None`. |

<!-- defaults:end -->
