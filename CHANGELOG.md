# Changelog

All notable changes to binference are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (with the usual 0.x caveat that minor
releases may break the API).

## [Unreleased]

### Added

- `describe(bins, counts=None)` on `Dirichlet` and the tree priors draws the tree as
  text: each branch's concentration, mean share and mean mass, or the posterior given
  counts.
- `DirichletTree` takes named groups (nested dicts, with bin labels on categorical bins),
  and `alpha` as a dict from group names, and bins as the tree writes them, to the
  concentration of the branch into each.
- `BoundPrior.long_log_tails()`: whether a bin without data has a long tail in the
  unconstrained coordinates NUTS samples.
- `Selection.from_function` and `Selection.combined` take `max_depth`, the most
  refinements of their adaptive quadrature; the default, 20, allows the halvings a jump
  inside a bin needs.
- `Likelihoods.from_density`, `Density.from_function` and `Density.coerce` take
  `bounded=`: the bin averages of a bounded density, such as a box, and of
  `from_density`'s interim prior given as a function, are checked for jumps near bin
  edges as a selection function's are, at 13/8 the points per cell in 1-D.
- `Density` compares by value and is hashable: flat densities by coordinate, arrays by
  their values and reading, frozen SciPy distributions by family and parameters,
  functions by identity.

### Changed

- The `test` extra installs scikit-learn 1.6 or later, so the test of
  `PopulationEstimator` with scikit-learn's model selection runs.
- `Selection.at_mean()` is renamed `Selection.fixed()`, the name the same operation has
  on `Response`.
- A selection function's bin averages are also checked against Lobatto's 5-point rule
  read just inside each cell's ends, so a jump near a cell's edge is refined to `tol`
  instead of being missed without a warning.
- `Likelihoods.from_samples` refuses NaN and infinite samples, which counted as samples
  outside the bins.
- `Likelihoods.from_density` evaluates frozen SciPy distributions of one family in one
  call, for their exact bin masses and for their densities; the densities' check for one
  family never matched, so each distribution was evaluated separately.
- `plot(stack=...)` takes the stack as `Likelihoods.stack` returns it, and on a fit with a
  `Basis` shares its component weights among the bins, so `result.plot(stack=lk.stack())`
  works on every fit; it failed with more components than bins and, with as many, drew
  the component weights as bin fractions. A stack mapped to the bins first, as
  `Likelihoods.stack`'s docstring advised, is now refused when the numbers of components
  and bins differ and mapped twice when they agree. A `truth` or `stack` of the wrong
  length is refused by name.
- `PopulationResult` and `PointEstimate` keep read-only copies of every array they are
  given (draws, total, `value`, `x`, labels, and the arrays in `info`), freeze what they
  cache, and stay read-only through pickling: writing into `samples`, `chains`,
  `component_draws`, `total`, `selected`, `data_support` or a column of `summary()`
  raises, `diagnostics` is a read-only mapping, and changing the array passed in no
  longer changes the result.
- `check()` flags Markov chains that give no R-hat or ESS, shorter than 8 draws
  (`MIXING_MIN_DRAWS`) or constant; they passed with no message.
- `PointEstimate` refuses rates with the same message as `PopulationResult`.
- `Measure` objects are immutable, as `Bins` are; renaming the shared `Measure.log()`
  broke the saving of bins. `Bins`, `Selection`, `Response` and `Measure` also refuse
  `del` of an attribute, after which it could be set again.
- `PopulationModel.log_likelihood` refuses a value that is not finite and non-negative,
  or is all zero; a negative fraction gave a finite log-likelihood.
- `sbc` ranks the log-likelihood of a model that infers its selection or response in one
  pass over the draws: the same values, about 11 times faster.
- `method="auto"` records first in `info["auto_reason"]` why a rule left Gibbs out, and
  keeps an extended Gibbs result rather than rebuilding it.
- `binference.io` and `binference.plotting` are imported with the package and listed by
  `dir()`; they stay out of `__all__`.
- The partial-centring weights of the logistic priors' bound form are named `gamma` (was
  `lam`), also in `with_centering` and `recentre`; `info["centring"]` is unchanged.
- Tree priors take their EM weights from the concentrations, and binding no longer
  builds a slot-by-node matrix: a `PolyaTree` on 2048 bins binds in 0.06 s (was 0.64 s)
  (#166).
- `BoundPrior` documents the interface a new prior implements.
- `method="auto"` builds every engine it may run when it is made, so a bad engine
  option, or options that no engine takes, raise before any fit (#174).
- Breaking: `init` is fractions or rates of shape (K,) for `method="auto"` and `NUTS`, as
  for Gibbs and MAP; NUTS converts it with `PopulationModel.from_value` to one point that
  each chain jitters. Starts in NUTS's unconstrained coordinates, shape (dim,) or
  (n_chains, dim), are passed as `init_x`. Before, auto refused a fraction for NUTS's
  start, read a rate as a log-rate, and passed a point in NUTS's coordinates on as given
  (#174, #190).
- Above 200 dimensions, NUTS takes a logistic prior's hyperparameter curvature at one
  chain, like the rest of its warm-up curvature.
- The defaults page lists NUTS's divergence threshold, `_MAX_ENERGY_ERROR`.

### Removed

- `PopulationModel.prior_transform`, `BoundPrior.prior_transform` (on the Dirichlet, tree
  and `GammaRates` priors) and `BoundPrior.has_prior_transform`: the unit-cube map for
  nested samplers, which nothing in binference used.
- `Engine.name` and `Engine.approximate`, which nothing read.
- `Response.measured_fractions`, which only a test used and which ignored the posterior
  means of a tallied response.
- `binference.diagnostics.kkt_gap` and `binference.diagnostics.missing_information`: use
  `PopulationModel.kkt_gap` and `PopulationModel.missing_information`, or
  `PointEstimate.kkt_gap`.
- `simulate.Truth.extra`, which nothing set or read.
- `Bins.widths`, which nothing used; per-axis widths are `np.diff(bins.edges[d])`.
- `Dirichlet("reference")` and the `K` argument of `Dirichlet.reference()`, which takes `K`
  from the bins.
- `PopulationPrior.kind` and `GammaRates.kind`; the bound prior's `kind` remains.
- The lazy `binference.__getattr__`.

### Fixed

- Docstrings, error messages, attribute comments on the API pages and this changelog no
  longer cite design records, which the site does not include.
- The NUTS tree-depth hint of `check()` names long log tails only for priors that have
  them where NUTS samples, such as a tree with a branch concentration below one; it no
  longer blames trees whose concentrations are all at least one, such as
  `GeneralizedDirichlet(1, 1)` or the default `PolyaTree()` on 4 bins, and suggests
  `method="gibbs"` only when Gibbs can fit the model.
- A tree prior with a branch whose exponent on the simplex is exactly zero no longer
  counts as unbounded through rounding, so MAP keeps it on the simplex.
- The sdist no longer includes walkthrough notes, and its tests skip the checks that
  need `docs/` or `benchmarks/`, which it does not ship.
- `plot_individual` on a fit with a `Basis` drew the object's likelihood over the
  components as if they were bins, or failed when there were more components than bins;
  it is now shared among the bins as the hierarchical posterior is.
- The stack overlay of `plot(which="selected")` is no longer reweighted by the selection:
  an average over the observed objects, it already estimates the selected population.
- `plot_sbc` always shows the joint log-likelihood; with 12 or more other quantities it
  was left out. A `max_panels` below one is refused by name.
- A list or array of colours is accepted for the bars of categories only, one per
  category; the other plots failed with a matplotlib error and now raise one naming the
  argument, and `plot_sbc`, which coloured its rank bars from a list, refuses it too.
- A truth overlay on `which="selected"` of a fit with a `Basis` is drawn as given
  without a selection and refused with one, unless the basis is the bins; it failed
  with a broadcasting error or, with as many components as bins, drew a wrong overlay.
- `MAP` checks `init`'s shape and sign, and that it is not all zero, when the engine is
  made; a wrong shape failed inside NumPy. EM raises zeros in its start to 1e-12 of the
  largest value, so a zero, from `init` or from mix-SQP's hand-off to EM, no longer
  leaves a wrong mode with only a convergence warning.
- `NUTS` checks `model`, and the shape and finiteness of `init_x`, when it is made.
- The start for rates with a selection is near the mode for broad likelihoods too.
- Gibbs refuses a negative `n_warmup`; `Laplace.run` with fewer than 3 samples draws 5
  importance proposals rather than failing in Pareto smoothing; the
  engines that can infer a selection are each named once in the refusal of those that
  cannot.
- `Response(matrix, efficiency=...)` divides a migration column that sums to one within
  `MIGRATION_ATOL` by its sum, so the efficiency given is exact.
- Boolean category labels reload as booleans; they came back as 0 and 1.
- Errors name the input at fault: a single distribution passed to `from_density`;
  objects in a measured bin whose response row is zero (the error suggested
  `drop_empty`); a truth bin whose response column sums to zero (the error spoke of
  selection values), and any selection value that is zero or not finite;
  `Response.from_counts` with more counts than generated events;
  `Response.from_simulation` with `open_ended` and categorical measured bins (an
  `AttributeError`); labelled scores that are not finite. `Selection.combined` checks
  `scale` before evaluating its functions.
- Adaptive bin averages refine a row whose 5-point estimates are all zero where a check
  rule reads something, until its cells are below `tol` of their bin, and warn where
  the function is still unread there; such a row was read as zero, so a selection
  function, or a bounded density, that is non-zero only within 4.7% of a bin edge was
  missed without a warning.
- `check()` computes and checks the tail ESS of a selection or response inferred with the
  population (`ess_tail_selection`, `ess_tail_response`); it looked for these keys, which
  were never computed.
- `mock_1d` draws posterior samples flat in u within each bin when the interim prior is
  omitted or known only per bin (masses or "flat"), as `Likelihoods.from_samples` reads
  them; without an interim, `mock.interim` holds `Density.flat()`'s masses. On log bins
  the likelihoods built from these samples were biased; linear bins are unchanged.
- `summary(level=)`, `coverage(levels=)` and `plot(levels=)` refuse levels outside
  (0, 1) by name, as `interval` does.
- `rhat` of a constant chain is NaN without NumPy's "All-NaN" warning.
- `exact_posterior` refuses a `prior` given with a `PopulationModel`, which ignored it.
- `psis` refuses +inf log weights; it gave them zero weight.
- `autocorr_time` and `autocorr_function` return NaN for a constant chain; they gave 1,
  or the chain's length when its mean was off by round-off.
- `mock_categorical` checks that `prevalence` and `train_prior` have `n_classes`
  non-negative entries with a positive sum, naming them.
- The ECDF p-value is documented as the Bonferroni bound below 1e-12, and Pareto
  smoothing as needing at least 21 weights; with fewer, k̂ = inf and the weights are not
  smoothed.
- Unpickled `Bins`, and copies of grids with a non-linear measure, had writeable edges.
- `Bins.linear`, `Bins.log` and `Bins.coarsen` refuse a count or factor that is not a
  positive integer, naming it; they truncated it, and `coarsen(1.5)` returned the bins
  unchanged. `Bins.categories(2.0)` raises a TypeError naming `labels`.
- `PopulationModel.expected_counts` raises, naming the objects of zero likelihood, and
  checks `value`; it returned zeros for them.
- `missing_information(include_prior=True)` adds a `GammaRates` prior's pseudo-counts as
  the MAP EM does: 0.49 instead of 0.89 for `GammaRates(200, 1)`, which corrects the
  missing information reported when the rates EM does not converge. It is NaN where
  shapes below one outweigh the data.
- A distribution with `cdf` and `sf` but no `logsf` keeps its upper tail.
- `Likelihoods.from_samples` names the measure when a sample falls where a
  `Measure.transform` has no finite slope, and `Measure.log_jacobian` no longer emits
  NumPy warnings there.
- The docs of `tol`, of `PopulationModel(bins=)` and of `kkt_gap(include_prior=)` say
  what the code does.
- A prior's fingerprint, which `attach` compares after `load`, includes the values of its
  base measure: priors with different base arrays of the same length no longer pass for
  each other, and a named function as base matches in another process.
- MAP with a tree prior one of whose branches has exponent zero, such as
  `PolyaTree(c=K/2)`, converges when that branch's bins are empty: the KKT gap is checked
  on the tree's branch probabilities, where it keeps its precision; EM warned that it had
  not converged (at K = 32 also in 0.1.0).
- A value that is not numbers, such as `Dirichlet("reference")`, raises an error naming
  the argument rather than NumPy's cast error.
- `infer` and `Auto.run` refuse an `n_samples` that is not a positive integer, by name;
  `0` under auto ran at the automatic length.
- `Dirichlet.from_counts` refuses a negative `strength` or `base`, naming them; a negative
  strength subtracted the reference sample.
- auto reported "the only engine that takes these options" when Gibbs took them but a
  rule left it out.
- `amplitude`, `scale` and `length_scale` (and `length_scale[d]`) are named in their
  errors.
- The docs say which priors have unbounded densities, give `GeneralizedDirichlet`'s
  default `b_k = (K - k) a`, and the length scale below which `LogisticGP` warns.
- Docs: the Laplace Student-t's (−H)⁻¹ is its scale matrix, not its covariance; also
  `_HESSIAN_MAX_DIM`'s description, bootstrap's `tol`, MAP's `max_iter` (which covers
  mix-SQP) and the engines' notation.

## [0.1.0] - 2026-10-29

First release.

### Changed (landing on `main`)

- Renamed from `pastry` to `binference` (package, import and PyPI name).
  The warning base class is `BinferenceWarning` and the environment variables start
  with `BINFERENCE_`.

- Python ≥ 3.13, NumPy ≥ 2.1, SciPy ≥ 1.14.1, matplotlib ≥ 3.9.2.
- CI runs the pre-commit hooks, the fast suite on three systems with Python 3.13 and
  3.14, minimum dependencies, the README and user-guide code, the docs build and a
  clean-wheel import; pushes to `main` publish the docs to GitHub Pages.
- Releases: a tag `vX.Y.Z` on `main` publishes to PyPI by trusted publishing.
- Development process: issue and pull-request templates and the rules in `CLAUDE.md` and
  `CONTRIBUTING.md`.
- The prior base class is now `binference.priors.PopulationPrior` (was `Prior`).
- On log bins, the default base measures of `Dirichlet.from_base_measure` and
  `PolyaTree`, the default mean of the logistic priors and the "flat" interim prior
  are flat in log z (they were flat in z).
- Results (audit O-1–O-21): a result keeps its selection through `save`,
  `load` and `attach`, which refuses other bins or another selection;
  `result_to_dict` is what `save` writes (`meta` plus arrays, `label_draws` apart from
  the bin labels); `check()` also flags tail ESS, E-BFMI, one-chain R-hat and
  unconverged optimisers and raises `DiagnosticsError` when strict; `summary` reports
  the mixing of the quantity it summarises; `which=` takes `"fraction(s)"`/`"rate(s)"`
  everywhere, including `functional`, `rebin`, `cov`, `corr`, `plot_corner` and
  `plot_trace`; `method="em"` runs EM on the simplex, and an explicit
  `MAP(algorithm=)` raises where it cannot run instead of falling back to L-BFGS
  (#100); `sample_assignments`, `sample_values` and `realized_counts` default to 500
  draws; `binference.plotting` and `binference.io` load on first use;
  `result.plot_individual`; `PopulationEstimator` works with
  scikit-learn model selection (`Likelihoods.shape`); the xarray export uses ArviZ's
  statistic names, numeric bin centres and hyperparameter draws.
- `infer` defaults to `method="auto"` (was `"nuts"`): Gibbs for
  joint conjugate models and for conjugate priors with at most 0.98 missing
  information (run long enough for a bulk ESS of about 400), else Laplace when its
  importance check passes with an ESS of at least 400, else NUTS. The choice and
  reason are in `result.info["auto_method"]` and `["auto_reason"]`. The rules are one
  engine, `binference.inference.Auto`, whose thresholds (`target_ess`,
  `gibbs_max_missing`, `nnz_limit`) are options; `PopulationEstimator` defaults to it
  too. Pass `method="nuts"` for the previous behaviour. Without `n_samples`, a Gibbs run
  chosen by auto is extended until its ESS and R-hat pass, or gives way to the next
  engine beyond `max_gibbs_draws`. Laplace draws that auto keeps
  without passing its check (on very large inputs, or as the last candidate) are flagged
  `approximate`.
- `Laplace` finds its mode and curvature once per engine (and `step`), not on every
  `run`.
- `PopulationResult.log_predictive` works through the objects `chunk` at a time (a new
  `chunk` argument, 4096 by default, as `predict_proba` and
  `assignment_probabilities` already did), so its memory no longer grows with the
  number of objects times the number of draws.

- Names: `sbc` and `coverage` take `progress=` (was `progress_bar=`);
  `CoverageResult.coverage_rates()` (was `rates()`); `Likelihoods.with_weights(
  fractional_weights=)` (was `fractional=`); `simulate.mock_unfolding(n_objects=)` (was
  `n_events=`). Plots label rates `lambda[k]`, as SBC names them (was `rate[k]`). The
  `"se"` kernel alias and the rename hints for the old selection modes are gone.
- `Gibbs` and `MeanFieldVB` raise `ValueError` (was `TypeError`) for a model they cannot
  run, with the reason `Engine.unsupported(model)` gives.
- `PopulationModel.stack()` is removed; `likelihoods.stack()` is the one stacking.
- Engines declare `Engine.markov` (draws are Markov chains, checked by R-hat and ESS)
  and `Engine.unsupported(model)`; `PopulationResult(kind=)` defaults to the model's
  kind; `diagnostics.importance_ess(weights)` is Kish's effective sample size.
- The chunked kernels work through 4096 dense rows at a time (was 65,536): Gibbs drew
  labels as fast at either size (N = 2·10⁴ to 2·10⁵, K = 20 to 80) with 16 times less
  memory per block. Sparse rows are sampled all at once, since slicing CSR rows per chunk
  cost time (0.030 s per label draw, from 0.035 s, at N = 2·10⁵, K = 200, 2% filled).
- Faster: Laplace's importance step evaluates its proposals in blocks (5.3 s, from
  28.4 s, for 8000 proposals at N = 10⁵, K = 50, one thread); EM computes `L·ρ` once per
  point; leave-one-out smooths all objects' weights together (3.1 s, from 7.4 s, at
  N = 2·10⁴ with 2000 draws); Gibbs draws labels in place (35 ms, from 47 ms, per draw
  at N = 10⁵, K = 50). `method="auto"` keeps the MAP it computed as `result.best` when that raised no
  warning.
- Plot colours follow the active matplotlib style instead of fixed hex colours; new
  `sample_color` argument on `plot_population` and `plot_individual`; `color` defaults to
  the style and is added to `plot_individual` and `plot_sbc`; 2-D maps and the corner
  plot's 2-D histograms are drawn on ramps to the population colour (relative uncertainty
  to the text colour) instead of viridis, magma and Blues; `POSTERIOR_COLOR`, `STACK_COLOR` and `INK` are removed.

### Fixed (phase-wide review)

- PSIS compared weights with the cut as `log(exp(cut))`, so a weight tied with the cut
  could enter the tail with no exceedance; with duplicated draws (resampled Laplace) the
  fit divided by zero. The cut is compared in logs, as ArviZ does; k-hat changes slightly
  where the old comparison included the cut weight.
- A degenerate generalised-Pareto fit gives k-hat = inf in `diagnostics.psis` (was NaN,
  which compared False with any threshold), so leave-one-out flags such objects.
- Bootstrap and other resampling of a model with an inferred response kept each row's
  measured bin from before rows were dropped.
- `attach` clears everything computed from the previous likelihoods (the total, the
  missing information); a model with another prior than the fit's no longer stands in for
  it in `best` and `binning_bias` (compared by a digest of the prior's arguments, saved
  with the result, since its repr shortens arrays); a saved selection on a basis
  re-attaches after load.
- `save` keeps the unconstrained draws, so `prior_information_ratio` is the same after
  `load`.
- `PopulationResult.from_samples` refuses a model that infers its selection or response
  unless their draws are given.
- Diagnostics of a jointly inferred response are named `*_response` (were `*_selection`).
- Posterior-sample mocks on unequal bins weighted every grid step alike, whatever its
  width (with bins [0, 0.1, 3], half the samples fell in the narrow bin). Their grid is
  now 50 equal steps per bin in the bins' measure (in log z on log bins; was in z).
- `Selection.draw` floors its Beta draws as Gibbs does, so none underflows to zero.
- Pickling likelihoods whose within-bin data cannot be pickled warns (they were dropped
  silently).

### Added

- `benchmarks/matrix.py`: the benchmark matrix of scenario × prior × engine against
  long reference runs, with its results in `benchmarks/results/`.

- **Results**: `result.selected` (the selected
  population, also `which="selected"`), `realized_counts(full=True)` (with the missed
  objects), `data_support` and the `support` summary column, `prior_information_ratio()`
  (Efron's ratio), `check(support_min=)`, and `PointEstimate.check()` for the KKT gap.
- **Binning-bias check**: `result.binning_bias()` refits the
  data on linear B-spline components over the same bins (Gaussian inputs rebuild
  themselves; others take `build`) and reports each bin's shift in posterior s.d. and,
  for flagged bins, `n_crit`, the number of objects at which the binning bias equals
  the statistical error.
- **Best fit and leave-one-out**: every result has
  `result.best` (the MAP estimate, computed on first use) and per-object outputs at it
  with a `_best` suffix; `assignment_probabilities_loo` gives leave-one-out posteriors
  by Pareto-smoothed importance sampling, with the Pareto k-hat per object.
- **Value draws from within-bin information**: `sample_values`
  resamples the samples of `from_samples` inside the drawn bin (`keep_samples=`),
  evaluates `from_density` densities inside 1-D bins (interim divided out for
  posteriors), and draws correlated Gaussians jointly by Gibbs sweeps (`n_sweeps=`);
  tables, binned inputs, compressed rows, densities on grids or without a pdf, samples
  not kept and density functions that could not be pickled fall back to flat within
  the bin, with a warning naming the source.

- **Measures**: each grid axis has a `Measure`, the coordinate
  `u = T(z)` in which the population is flat within a bin: linear, log, a transform with
  its inverse, or the cumulative integral of a positive density. `Bins.log` uses the log
  measure; bin centres are midpoints in `u`; `Bins.from_centers` takes midpoints in `u`
  (positive edges for log-spaced centres, I-16). Geometric hand-built edges warn.
- **Selection from injections**: `Selection.from_injections`
  gives a completeness with Beta posteriors per bin, inferred jointly with the population
  by Gibbs, NUTS, Laplace and `MAP(space="unconstrained")` (logit S as extra
  coordinates); `smooth=τ` ties neighbouring bins by a random walk on logit S;
  `at_mean()` fixes it for VB and the bootstrap.
- **Responses inferred with the population**: a response tallied
  from events (`Response.from_counts`, `from_simulation`, `from_labelled`) has a
  Dirichlet posterior per column (`prior=β`, default 1 per cell) and is inferred jointly
  by Gibbs, NUTS, Laplace and unconstrained MAP; `response.fixed()` takes it as known.
- **Totals and rates from fractions**: every fractions result has
  `total` (draws of Λ ~ Gamma(W, S·ρ), exact under a scale-free prior) and `rates`
  (Λρ), with an absolute or no selection; point estimates give W/(S·ρ̂).
- **Mixture basis and internal resolution**:
  `binference.Basis(bins, order=0|1, resolution=r)` fits components on the bins refined
  `r` times, flat (`order=0`) or linear B-spline hats (`order=1`), and reports in the
  bins. Every constructor takes a `Basis` for `bins` (`Likelihoods.on_basis` for
  per-bin likelihoods); callable selections are averaged with each component as the
  weight. Results report per bin and keep `component_draws`; save and load keep the
  basis. `Bins.refine` is the counterpart of `coarsen`. `sample_values` on non-linear
  measures now draws flat in `u` within a bin (it ignored `T'(z)`).
- `from_samples` stores likelihoods sparsely below 10% non-zero entries (was 30%).
- **Masked grids**: `mask=` on `Bins`, `Bins.grid`, `Bins.linear` and
  `Bins.log`; masked cells leave the support and the model. Saved and loaded with
  results.
- `Bins` with edges `-0.0` and `0.0` now hash alike (I-12); `Bins.index` refuses
  multi-column values for 1-D bins (I-22).
- **`Density`**: one fixed density over the bin coordinate for
  interim priors, the Dirichlet base measure and simulation truth. `Density.flat()` is
  flat in the measure coordinate. An array interim prior on unequal bins needs
  `interim_representation`, so every constructor reads it the same way (I-1); a
  callable on categories receives the labels (I-17).
- `Likelihoods.stack()` uses the interim prior the likelihoods were built with
  (I-5) and raises on rows with no support under it (I-6).
- **Constructors by input form**: `Likelihoods.from_table` and
  `Likelihoods.from_density` replace `from_posteriors`, `from_function` and
  `from_distributions`; each needs `kind="likelihood"` or `interim=`. `interim_prior`
  is now `interim` (also in `stack()` and `simulate.mock_1d`).
- **Adaptive quadrature**: bin averages of functions and distributions
  are refined to `tol` (default 1e-3), in the measure coordinate for likelihoods; `from_gaussian` averages in u on non-linear axes. Far tails no longer
  underflow to empty rows (I-2, I-3).
- **Samples**: one estimator for every interim prior (per-sample
  weights T'(z)/π(z); a per-bin prior is flat in u within bins), an opt-in Dirichlet
  `floor` in pseudo-samples,
  `n_eff` inside the bins with a warning on the share of noisy objects (I-23),
  `Likelihoods.sample_variance` and a Monte Carlo variance check in `result.check()`;
  `Measure.log_jacobian`.
- **Binned measurements**: `binference.Response` holds
  P(measured bin, observed | true bin) with real truth bins, from a matrix, tallies
  (`Response.from_counts`), simulated events (`Response.from_simulation`) or a labelled
  validation set (`Response.from_labelled`: open-ended score bins or a confusion matrix).
  `Likelihoods.from_binned` replaces `from_response`; `from_classifier` is removed
  (route A is `from_table` on categories, route C is `from_binned` with a labelled
  response). `summary()` names the inputs, and `result.check()` runs a posterior
  predictive test of the measured counts (`Response.predictive_pvalue`) that flags label
  shift. `check()` raises `InputWarning` for problems with the inputs (this test and the
  samples' Monte Carlo variance) and `ConvergenceWarning` for sampler problems. `simulate.mock_categorical` gains `temperature` and a labelled `mock.response`;
  `simulate.mock_unfolding` returns `mock.response`.
- **Selection**: the modes are
  `"on_truth"` and `"on_measurement"` (the old names raise an error naming the new
  ones); `Selection.combined(on_truth, on_measurement)` selects on both; a selection may
  be a function of the true value, averaged over each bin (`Selection.from_function`,
  or a callable `selection=`); values are detection probabilities in (0, 1] unless
  declared `scale="exposure"` or `"relative"`, and rates refuse a relative selection
  (audit G-4); an array that changes by more than a factor of two between neighbouring
  bins raises an `InputWarning` (a `Selection` object or a callable is taken as averaged
  within bins).
  `simulate.mock_1d(measured_range=...)` simulates selection on the measurement, alone
  or with `selection=` (combined). `sbc`/`coverage` name rates `lambda[k]` (G-10).
- **Model**: `PopulationModel.log_prob` and
  `log_prob_and_grad` are renamed `log_density_unconstrained` and
  `log_density_unconstrained_and_grad`; the theory page explains the two densities
  and their modes. `kkt_gap` is infinite where an object is impossible and, for rates,
  relative to the exposure (audit M-3, M-4); `missing_information` is correct with a
  selection function and on the support, with `include_prior=` for the MAP EM (M-2).
  Bootstrap replicates drop zero-weight objects in every engine, so EM no longer
  resets to the prior's mode or collapses rates to zero (M-1); EM stops with a warning
  where an object becomes impossible. The rates starting point is W ρ / S (M-7).
  `LogisticGP()` works on two bins (M-14). `PopulationResult.log_prob` is now
  `log_density_unconstrained` (old files still load); `MAP.run` no longer takes
  `weights=`.
- **MAP**: `tol` (default 1e-8, was 1e-10 on the objective) is the
  tolerance on the KKT gap for every simplex algorithm and on the largest gradient over
  `max(1, W)` for L-BFGS (mix-SQP bounds the gap of the problem it solves), and EM is
  converged only when the gap is below it, or below `1e3 * tol` once the objective
  stops improving (M-6, M-9).
  `GammaRates` with shapes of at least one use mix-SQP, with the total rate profiled
  out (M-6). `PointEstimate.log_posterior` is `model.log_posterior(value)` for every
  algorithm (NaN for priors without a closed-form density); L-BFGS adds
  `info["log_density_unconstrained"]` (M-8). mix-SQP drops its SVD and starts after
  10 EM steps instead of 50, 2–3 times faster on large dense inputs (M-5); a failed
  quadratic step hands over to EM instead of raising (M-13); Dirichlet priors
  mixing concentrations of one and above one no longer warn (T-4). The rates KKT gap
  is one-sided, so a rate EM drives towards zero no longer reads as a violation.
- **Priors**: `LogisticRandomWalk(trend_scale=3.0)` is the standard
  deviation of each unpenalised mode's root-mean-square size across the grid, so its
  meaning no longer depends on K (was 10 in basis units, which tightened as 1/√K; P-6).
  Arguments are checked when a prior is made: `DirichletTree` per-node alphas of
  different lengths now work (P-4), `LogisticNormal(cov=)` must be a positive variance
  or a matrix positive semi-definite on zero-sum vectors (P-8), `LogisticGP` `n_basis`
  and `boundary_factor` are checked and `length_scale` takes an array (P-10, P-11),
  `PolyaTree` needs two bins and a non-negative power (P-9), `GeneralizedDirichlet`
  `a`, `b` and `order` are checked (P-22). Reprs show every argument (P-14); tree VB
  draws have no exact zeros (P-12); `GammaRates` has a `prior_transform` (P-15).
  `GeneralizedDirichlet` and `PolyaTree` are no longer subclasses of `DirichletTree`
  (P-13). `LogisticRandomWalk(2).bind` at K = 1000 and 20,000 GP draws are about twice
  as fast (P-20).
- **GP prior**: `LogisticGP` is stationary in the measure coordinate
  (log z on `Bins.log`; length scales are in that coordinate; P-1). Defaults:
  `boundary_factor=3` (was 1.5; P-3), `n_basis` one per bin up to 1024 modes in all
  (was at most 48 per axis), and a
  length-scale prior whose lower 1% quantile is the shortest length scale that basis
  resolves (P-2), or `LogNormal(log(range/2), 1)` on coarse grids (P-7). Partially
  centred GP densities and gradients agree where scales underflow (P-5).
- **NUTS and Laplace**: NUTS results (`x`, `log_density_unconstrained`,
  `model`) and the positions `sample()` yields are in the model's own coordinates, not
  the partly centred sampling ones (`info["centring"]` holds the weights; N-1, N-2).
  `diagnostics.psis_threshold(n)` gives the k̂ threshold for n importance weights. Parallel workers receive the model
  once, reuse one pool per run, never raise a BLAS thread limit already set, respect
  CPU affinity, and errors from the model are raised instead of triggering a serial
  rerun (N-3, N-7, N-13). `max_depth`, `n_warmup`, `jitter` and `init` are checked
  (N-5); Laplace returns every draw asked for without importance resampling (N-4) and
  judges k̂ against min(1 − 1/log10 S, 0.7) (N-6). Dense momentum draws use a
  precomputed triangular factor (N-9).
- **Validation, Gibbs, bootstrap**: SBC p-values and bands use the
  expected count per rank bin (G-1) and `sbc` warns when the ESS is below `n_draws` (G-9);
  `exact_posterior` integrates by stick-breaking Gauss–Legendre quadrature that resolves
  the simplex boundary, also for concentrations below one, adds `log_density`, and bounds
  its memory, with a default `n_grid` of 200 (G-3, G-5); weighted Gibbs on sparse
  likelihoods draws counts over each row's nonzeros only (G-2); Gibbs checks `chunk`,
  `store_labels` and `init`, and chains start apart, from counts that include missed
  objects (G-6, G-7); SBC rejects `n_bins` above `n_draws + 1`; `Bootstrap(kind=)` is now
  `scheme=`, non-converged replicates are counted and the base fit warns (G-8).
- **Correlated Gaussians**: `from_gaussian(..., cov=)` takes full
  per-object covariances, averaged over N-D bins by adaptive cubature in log space.
- `weights` are integer multiplicities; non-integer weights need `fractional_weights=True`
  and warn that the result is a pseudo-likelihood. Stored arrays
  are read-only (I-14); `compress()` keeps sparse storage and is fast on it (I-7, I-19);
  warnings point at the caller's line; clearer errors for bad `rtol`, Gaussian shapes,
  zero objects and 2-D arrays passed to `Bins.grid` (I-8, I-10, I-11, I-20); broad
  Gaussians keep full precision near the mean (I-15).
- Each object's likelihood mass outside the bins is recorded in
  `meta["outside_fraction"]`, with a warning when many objects lose most of it.

- **Inputs.** `Bins` (linear, log, from centres, N-D grids, categories) and
  `Likelihoods` with constructors for:
  - likelihood matrices (density or mass representation, dense or sparse,
    log-space input);
  - Gaussian measurements;
  - per-object posteriors with their interim prior;
  - posterior samples;
  - SciPy distributions and arbitrary likelihood functions;
  - response matrices (unfolding) and classifier scores or labels (quantification),
    as binned measurements with a `Response`.
- **Selection.** `Selection` functions with explicit `"on_measurement"` and
  `"on_truth"` modes, attached to the data or passed to the model.
- **Model.** `PopulationModel` for bin fractions or Poisson rates, with
  log-density and gradient in unconstrained coordinates, expected counts,
  responsibilities, an optimality (KKT) gap and the missing-information fraction.
- **Priors.**
  - Simplex: `Dirichlet` (with reference, count-based and base-measure
    constructors), `GeneralizedDirichlet`, `DirichletTree`, `PolyaTree`.
  - Smoothness: `LogisticNormal`, `LogisticRandomWalk` and `LogisticGP`
    (Hilbert-space GP) on N-D grids, with inferred hyperparameters.
  - Rates: `GammaRates`.
- **Engines.**
  - `NUTS`: multinomial NUTS with windowed metric adaptation.
    Smoothness priors get partial non-centring (where the data constrain
    their scales) and a curvature-based dense metric. Chains are resumable
    and can run in parallel processes, which share the cores' BLAS threads
    (via `threadpoolctl`) rather than oversubscribing them.
  - Data-augmentation `Gibbs` for conjugate priors, with selection augmentation.
  - `MAP` with an optimality certificate: mix-SQP for Dirichlet priors with
    concentrations of at least one (under a fixed selection only `Dirichlet(1)`, which
    gives maximum likelihood) and for `GammaRates` with shapes of at least one;
    SQUAREM-accelerated EM for the other conjugate priors whose density is bounded;
    L-BFGS otherwise.
  - `Laplace` with PSIS importance correction.
  - `MeanFieldVB` (flagged as approximate) and `Bootstrap` (nonparametric or
    Bayesian, over objects, with variable totals for rate models).
- **Interfaces.** `binference.infer` and the scikit-learn-style `PopulationEstimator`.
- **Results.** `PopulationResult` and `PointEstimate` provide:
  - summaries, intervals (central and HDI), functionals and rebinning;
  - per-object assignment probabilities, values and realised counts;
  - posterior predictive scores, diagnostics and checks with hints;
  - plots, save/load and xarray `DataTree` export.
- **Diagnostics.** Rank-normalised split R-hat, bulk and tail ESS,
  autocorrelation time, MCSE and PSIS.
- **Validation.** Exact quadrature posteriors (K <= 3), simulation-based
  calibration and coverage studies. Mock catalogues in 1-D and 2-D, plus
  categorical, unfolding and prior-predictive mocks.
- **Design documents, tests, docs.**
  - Specification, test plan, implementation plan, and a survey of related
    software, including an audit of the `frankenz` samplers.
  - A fast and a slow (statistical) test suite.
  - Sphinx documentation with tutorials.
