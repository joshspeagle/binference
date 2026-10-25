# API/UX design exemplars, packaging and documentation norms, and validation/testing practice for scientific Python inference packages (for `pastry`)

## Scope and evidence

- **Scope:** design, packaging and validation only. Domain-specific population-inference tools are covered by other researchers.
- **Research date:** 2026-10-22.
- **Versions and release dates:** taken from the PyPI JSON API, queried that day.
- **Installed-source facts:** read from package files in a scratch virtualenv, without executing them:
  - arviz 1.3.0 (arviz_base 1.3.1, arviz_stats 1.3.3, arviz_plots 1.3.2)
  - pymc 6.3.2
  - numpyro 0.22.0
  - numpy 2.5.4
  - scipy 1.18.1
  - xarray 2026.9.0
- **GitHub access:**
  - The GitHub REST API and github.com pages could not be reached from the sandbox shell.
  - GitHub issue pages were read with a server-side web fetch.
  - Raw files were downloaded from raw.githubusercontent.com.

## Q1. What makes the exemplar packages praised or criticised for usability, and which concrete patterns should pastry copy?

### Takeaway
The packages users praise have eight traits in common:
1. They take plain inputs: a black-box log-probability function, plain NumPy arrays, or named parameters.
2. They have few core objects, and those objects use the same method names.
3. One call with sensible defaults covers the common case. A generator or object-level route gives more control.
4. One keyword makes a run crash-safe and resumable.
5. Runs report progress and explain why they stopped. Result objects offer `summary()` and plotting.
6. Their outputs work with other tools, mainly ArviZ/xarray `DataTree` and pandas.
7. Priors are declared as objects.
8. Seeding is explicit, through an `rng` keyword (SPEC 7).

Documented complaints cluster in a few areas:
- emcee's most-discussed issues are about walker initialisation, convergence/autocorrelation and parallelisation;
- constructors with too many options, which was the motivation for dynesty 3.0's redesign;
- hidden global random-number state;
- compile delays (PyMC's PyTensor backend);
- confusing defaults, such as corner's 2-D "sigma" contour levels;
- breaking redesigns, such as ArviZ 1.0 removing `InferenceData`.

### Cited Findings

#### emcee (PyPI 3.1.6, last release 2024-04-19; none since, so mature and low-activity)
- **Why users chose it:**
  - emcee "was specifically designed to work with 'black box' models instead of structured graphical models. This has been a popular interface for applications in astrophysics because it is often non-trivial to implement realistic physics within the modeling frameworks required by other libraries" ([Foreman-Mackey et al. 2019, emcee v3, JOSS 4(43) 1864, arXiv:1911.07688](https://arxiv.org/abs/1911.07688)).
  - It "requires hand-tuning of only 1 or 2 parameters compared to ∼N² for a traditional algorithm" and "permits any user to take advantage of multiple CPU cores without extra effort" ([Foreman-Mackey et al. 2013, arXiv:1202.3665](https://arxiv.org/abs/1202.3665)).
- **Sampler object around a plain callable** ([emcee "The Ensemble Sampler"](https://emcee.readthedocs.io/en/stable/user/sampler/)):
  - Signature: `emcee.EnsembleSampler(nwalkers, ndim, log_prob_fn, pool=None, moves=None, args=None, kwargs=None, backend=None, vectorize=False, blobs_dtype=None, parameter_names=None)`.
  - `pool` is "an object with a map method".
  - `vectorize=True` passes all walkers at once.
  - `parameter_names` makes the callable receive a dict.
- **Two-level run API** ([emcee sampler API](https://emcee.readthedocs.io/en/stable/user/sampler/)):
  - `run_mcmc(initial_state, nsteps)` will "Iterate sample() for nsteps iterations"; passing `initial_state=None` resumes.
  - `sample(...)` will "Advance the chain as a generator".
  - `iterations=None` gives an infinite stream (requires `store=False`).
  - `progress=True` shows a progress bar.
- **User-written convergence loop over the generator** ([emcee "Saving & monitoring progress"](https://emcee.readthedocs.io/en/stable/tutorials/monitor/)):
  ```python
  for sample in sampler.sample(coords, iterations=max_n, progress=True):
      if sampler.iteration % 100: continue
      tau = sampler.get_autocorr_time(tol=0)
      converged = np.all(tau * 100 < sampler.iteration) & np.all(np.abs(old_tau - tau) / tau < 0.01)
  ```
- **Resumable backend** ([emcee monitor tutorial](https://emcee.readthedocs.io/en/stable/tutorials/monitor/); [emcee Backends](https://emcee.readthedocs.io/en/stable/user/backends/)):
  ```python
  backend = emcee.backends.HDFBackend("tutorial.h5"); backend.reset(nwalkers, ndim)
  sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, backend=backend)
  # resume: omit reset() and call
  sampler.run_mcmc(None, 100)
  # read without a sampler
  reader = emcee.backends.HDFBackend(fn, read_only=True)
  reader.get_chain(discard=burnin, flat=True, thin=thin)
  ```
  Several runs can share one file through `name=`.
- **Why backends exist:** "even if the sampler crashes, the current state of the chain will always be saved". The in-memory `Backend` and `HDFBackend` share the same accessors: `get_chain`, `get_log_prob`, `get_blobs`, `get_autocorr_time`, `get_last_sample` ([emcee Backends](https://emcee.readthedocs.io/en/stable/user/backends/)).
- **Lessons from version 3** ([emcee "Upgrading From Pre-3.0"](https://emcee.readthedocs.io/en/stable/user/upgrade/)):
  - The tqdm progress bar was "long requested".
  - The `threads` argument was removed because of "memory consumption and hanging processes".
  - MPIPool moved to schwimmbad and PTSampler to ptemcee because "there aren't enough resources to maintain" them. Scope was cut to keep the core maintainable.
- **Input checks at the API boundary** ([emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py)):
  - `sample()` raises `ValueError(f"incompatible input dimensions {state_shape}")`.
  - It also raises "Initial state has a large condition number. Make sure that your walkers are linearly independent for the best performance", unless `skip_initial_state_check=True`.
- **Global, legacy random state** ([emcee ensemble.py](https://github.com/dfm/emcee/blob/main/src/emcee/ensemble.py); [emcee test_proposal.py](https://github.com/dfm/emcee/blob/main/src/emcee/tests/integration/test_proposal.py)):
  - The sampler builds `np.random.mtrand.RandomState()` from `np.random.get_state()`.
  - emcee's own tests seed with `np.random.seed(1234)`.
  - This is exactly the pattern SPEC 7 asks libraries to deprecate.
- **Pain points among the most-commented emcee issues** ([emcee issues sorted by comments](https://github.com/dfm/emcee/issues?q=is%3Aissue+sort%3Acomments-desc)):
  - Initialisation: #55 "FAIL if initialized with ANY walkers at -Inf"; #306 "Add check that `initial_state` allows full exploration of parameter space".
  - Autocorrelation: #195 "AutocorrError, 'The chain is too short to reliably estimate the autocorrelation time'"; #171; #209.
  - Parallelism: MPI (#14, #112) and multiprocessing (#148).
  - Seeding: #299 "ValueError: algorithm must be 'MT19937'".
  - Backends: #336 "Problem using 'backends'".
  - ArviZ support: #279.
  - By contrast, the FAQ's advice on initialisation is one sentence: "start in a small ball around the a priori preferred position" ([emcee FAQ](https://emcee.readthedocs.io/en/stable/user/faq/)).

#### scikit-learn (PyPI 1.9.1, 2026-09-10; needs Python ≥3.11)
- **Founding design principles** ([Buitinck et al. 2013, arXiv:1309.0238](https://arxiv.org/abs/1309.0238)):
  - "Consistency. All objects (basic or composite) share a consistent interface composed of a limited set of methods."
  - "Inspection. Constructor parameters and parameter values determined by learning algorithms are stored and exposed as public attributes."
  - "Non-proliferation of classes. Learning algorithms are the only objects to be represented using custom classes. Datasets are represented as NumPy arrays ... Hyper-parameter names and values are represented as standard Python strings or numbers whenever possible."
  - "Composition."
  - "Sensible defaults ... The default value should cause the operation to be performed in a sensible way (giving a baseline solution for the task at hand)."
- **Rules for estimator developers** ([scikit-learn "Developing scikit-learn estimators"](https://scikit-learn.org/stable/developers/develop.html)):
  - `__init__` takes only keyword hyperparameters with defaults, and stores them unmodified.
  - "There should be no logic, not even input validation" in `__init__`, because validation would then have to be repeated in `set_params`, which `GridSearchCV` uses.
  - Data go to `fit`, which returns `self`.
  - Learned attributes end in an underscore, e.g. `coef_`.
  - Calling `fit` twice is the same as calling it once on the new data, unless `warm_start` is set.
  - `get_params`, `set_params` and `clone` come from `BaseEstimator`.
  - Capabilities are declared in `__sklearn_tags__()`.
  - `check_is_fitted` looks for trailing-underscore attributes or a `__sklearn_is_fitted__` method.
  - An experimental HTML display in notebooks includes a "?" link to the documentation.
- **A conformance test suite you can run on your own estimator** ([parametrize_with_checks](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html)):
  - Call `check_estimator(Est())`, or decorate tests with `parametrize_with_checks(estimators, *, legacy=True, expected_failed_checks=None, xfail_strict=None)`.
  - `expected_failed_checks` (since 1.6) maps check names to reasons; those checks are marked as expected failures.
  - Single checks can be selected, e.g. `pytest -k check_estimators_fit_returns_self`.
- **Randomness:**
  - `random_state` still accepts None, an int or a `RandomState`, normalised by `check_random_state` ([Glossary](https://scikit-learn.org/stable/glossary.html)).
  - A whole docs section covers pitfalls: an int versus an instance gives "subtle but significant differences", and clones sharing a `RandomState` "influence each other" ([Common pitfalls §12.3](https://scikit-learn.org/stable/common_pitfalls.html)).
  - The recommendation is to create one `rng` at the top of the program and pass it down.
  - There is no enhancement proposal (SLEP) on numpy `Generator` or SPEC 7 support ([SLEP index](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/)).
- **SLEP023 "Callback API" (accepted)** ([SLEP023](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep023/proposal.html)):
  - Motivation: "progress bars, structured logging, metric monitoring, snapshots" are use cases "requested many times on the scikit-learn issue tracker".
  - API: `clf.set_callbacks(ProgressBar())` and `gs.set_callbacks(ProgressBar(max_propagation_depth=1))`.
  - `EarlyStopping` is set on the inner estimator, while auto-propagated callbacks such as `ProgressBar` pass down from the outer one.
- **Array API support** is experimental and opt-in: set `array_api_dispatch=True` and `SCIPY_ARRAY_API=1`. It is tested on PyTorch (CPU, CUDA, MPS, XPU), CuPy and dpnp ([scikit-learn Array API support](https://scikit-learn.org/stable/modules/array_api.html)).

#### dynesty (PyPI 3.1.0, 2026-07-17; actively maintained)
- **One-liner and generator** ([dynesty quickstart](https://dynesty.readthedocs.io/en/stable/quickstart.html)):
  - One call: `sampler.run_nested(dlogz=0.5, maxiter=..., maxcall=...)`.
  - Generator: `for it, res in enumerate(sampler.sample(dlogz=0.5)): ...`, then `sampler.add_live_points()`.
  - The docs say: "Similar to emcee, sampler objects in dynesty can also be run externally as a generator".
- **Checkpointing** ([dynesty quickstart](https://dynesty.readthedocs.io/en/stable/quickstart.html)):
  ```python
  sampler.run_nested(checkpoint_file='dynesty.save')
  sampler = NestedSampler.restore('dynesty.save', pool=mypool)
  sampler.run_nested(resume=True)
  ```
  The docs warn: "you should NOT use checkpointing for persistence ... checkpoint files are not guaranteed to be compatible between dynesty versions (even minor ones)". Save samples, weights or the results object instead.
- **Results and helpers** ([dynesty quickstart](https://dynesty.readthedocs.io/en/stable/quickstart.html)):
  - `sampler.results` is a "(modified) dictionary".
  - `results.summary()` prints nlive, niter, ncall, efficiency and logz ± error.
  - `dyfunc.merge_runs(rlist)` merges independent runs.
  - Version 3.0 added `save_evaluation_history=True, history_filename=...` for debugging.
  - dynesty's own `Pool(10, loglike, ptform, logl_args=...)` avoids re-pickling large arguments.
- **dynesty 3.0 redesign, released 2025-10-04** ([dynesty issue #495](https://github.com/joshspeagle/dynesty/issues/495)):
  - The sampler constructor's long list of tuning options was the pain point.
  - Maintainer segasai proposed to "just remove the walks=, slices=, facc= etc options from Sampler" and pass configured objects instead, e.g. `DynamicNestedSampler(sample=RSliceSampler(slices=10))` and `DynamicNestedSampler(bound=MyBound(4))`.
  - Custom samplers subclass `InternalSampler`; custom bounds subclass `bounding.Bound`.
  - The Bilby maintainer (ColmTalbot) said the new interface let him port his extensions with fewer "overheads/workarounds".
  - He asked for a general way to accumulate sampler statistics and a class-level `citation` attribute.

#### nautilus (PyPI nautilus-sampler 1.0.6, 2025-12-29)
- **Priors are declared as objects** ([nautilus Priors](https://nautilus-sampler.readthedocs.io/en/latest/guides/priors.html)):
  ```python
  prior = Prior()
  prior.add_parameter('a', dist=(-5, +5))
  prior.add_parameter('c', dist=norm(loc=0, scale=2.0))
  ```
  - `dist` can be a `(lo, hi)` tuple, any frozen `scipy.stats` distribution, or a constant.
  - A constant fixes the parameter, which gives nested models without changing the likelihood.
  - The likelihood receives a `param_dict`.
- **Three-line run** ([nautilus Quickstart](https://nautilus-sampler.readthedocs.io/en/latest/guides/quickstart.html)):
  ```python
  sampler = Sampler(prior, likelihood, n_live=1000)
  sampler.run(verbose=True)
  points, log_w, log_l = sampler.posterior()
  ```
  - `sampler.log_z` gives the evidence.
  - Verbose output is a single status table: Status | Bounds | Ellipses | Networks | Calls | f_live | N_eff | log Z.
- **One-keyword checkpointing that resumes by default** ([nautilus Checkpointing](https://nautilus-sampler.readthedocs.io/en/latest/guides/checkpointing.html)):
  - `Sampler(prior, likelihood, filepath='checkpoint.hdf5')`: "By default, nautilus will resume a previous run if it finds a file under this path".
  - `resume=False` overwrites the file.
  - The docs warn against reusing the checkpoint "of a different likelihood problem".
- **Other options** ([nautilus high-level API](https://nautilus-sampler.readthedocs.io/en/latest/api_high.html)):
  - `pool` accepts None, an int (creates a `multiprocessing.Pool`), a Pool or dask Client, or a tuple (one pool for likelihood calls, one for sampler work).
  - `seed` is None or an int.
  - `run(f_live=0.01, n_shell=1, n_eff=10000, n_like_max=inf, discard_exploration=False, timeout=inf, verbose=False)`.
  - The API docs are split into "High-Level Interface" and "Full Documentation" pages.
- **Method paper:** reports sampling efficiency "often by more than an order of magnitude" higher than emcee, dynesty, UltraNest and pocoMC ([Lange 2023, arXiv:2306.16923](https://arxiv.org/abs/2306.16923)).

#### UltraNest (PyPI 4.6.3, 2026-10-13; active)
- **Usage** ([UltraNest tutorial](https://johannesbuchner.github.io/UltraNest/example-sine-line.html); [UltraNest performance page](https://johannesbuchner.github.io/UltraNest/performance.html)):
  ```python
  sampler = ReactiveNestedSampler(param_names, loglike, transform,
                                  wrapped_params=[...], log_dir='my_gauss',
                                  resume=True, vectorized=True)
  result = sampler.run(min_num_live_points=400, dKL=np.inf, min_ess=100)
  sampler.print_results()
  sampler.plot_corner(); sampler.plot_trace(); sampler.plot_run()
  ```
- **Output that explains itself** ([UltraNest tutorial](https://johannesbuchner.github.io/UltraNest/example-sine-line.html)):
  - The run log names each stopping criterion and whether it is met, e.g. "Effective samples strategy satisfied (ESS = 2197.7, need >100)".
  - It prints a breakdown of the logZ error.
  - It runs a built-in self-check (the "insert order U test").
  - `print_results()` draws small text histograms of each parameter in the terminal.
- **Advertised features** ([UltraNest README](https://johannesbuchner.github.io/UltraNest/readme.html)):
  - "Sanity checks with meaningful errors".
  - "Can control the run programmatically and check status".
  - "Reasonable defaults, but customizable".
  - "Checkpointing and resuming, even with different number of live points".
- **Stated priorities:** "a focus on correctness and speed (in that order)" ([Buchner 2021, arXiv:2101.09604](https://arxiv.org/abs/2101.09604)).

#### statsmodels (PyPI 0.15.0, 2026-08-27; still classed "4 - Beta")
- **Formula interface and array interface over the same model/results split** ([statsmodels formulas](https://www.statsmodels.org/stable/example_formulas.html); [Getting started](https://www.statsmodels.org/stable/gettingstarted.html)):
  - Formula: `res = smf.ols(formula='Lottery ~ Literacy + Wealth + Region', data=df).fit(); print(res.summary())`. The formula engine handles categorical variables and the intercept (`C(Region)`, `- 1`).
  - Arrays: `y, X = dmatrices(...)`, then `mod = sm.OLS(y, X)`.

#### lmfit (PyPI 1.3.4, 2025-07-19)
- **Why named `Parameter` objects instead of positional arrays** ([lmfit Getting started](https://lmfit.github.io/lmfit-py/intro.html)):
  - With raw `scipy.optimize`, "The user has to keep track of the order of the variables".
  - Fixing a variable means rewriting the residual function.
  - Bounds "require bounds to be set for all variables with separate arrays".
  - With lmfit, users can "forget about the order of variables and refer to Parameters by meaningful names".
  - Bounds become attributes.
  - Parameters can be fixed "without having to rewrite the objective function".
  - Algebraic constraints can be added.
- **What the result object offers** ([lmfit Model](https://lmfit.github.io/lmfit-py/model.html)):
  - Reports: `ModelResult.fit_report()`, `.summary()`, `.conf_interval()`, `.ci_report()`.
  - Uncertainty: `.eval_uncertainty()`.
  - Plots: `.plot()`, `.plot_fit()`, `.plot_residuals()`.
  - Results can be saved and loaded.

#### astropy.modeling and astropy.units (astropy 8.0.2, 2026-10-15)
- **Models and fitters are separate, and fitting returns a new fitted model** ([astropy.modeling](https://docs.astropy.org/en/stable/modeling/index.html)):
  - `fit = fitting.LinearLSQFitter(); fitted_line = fit(models.Linear1D(), x, y)`.
  - The example seeds its fake data with `rng = np.random.default_rng(10)`.
  - "Different fitting algorithms can be used with any model".
- **Units are checked where values enter the API:**
  - `@u.quantity_input(myangle=u.arcsec)`, or annotations such as `def f(myangle: u.arcsec)`. Mismatched units raise `UnitsError` ([astropy quantity_input](https://docs.astropy.org/en/stable/api/astropy.units.quantity_input.html)).
  - Models accept parameters with units ([modeling with units](https://docs.astropy.org/en/stable/modeling/units.html)).

#### scipy.stats (SciPy 1.18.1, 2026-08-21; needs Python ≥3.12)
- **New random-variable API (SciPy ≥1.15)** ([SciPy random-variable transition guide](https://docs.scipy.org/doc/scipy/tutorial/stats/rv_infrastructure.html)):
  - It was motivated by design problems in the old `rv_continuous` system: all 125 distributions are created at import, parameters are re-checked on every call, and per-method docs cannot show shape parameters.
  - New calls: `X = stats.Normal(mu=mu, sigma=sigma)`, `X.sample(shape=(3, 4), rng=rng)`, `X.plot()`, `X.moment(order=1, kind='raw')`.
  - Old distributions convert with `Weibull = stats.make_distribution(stats.weibull_min)`.
  - The old system "is not deprecated at this time", and the new one "may not yet be as convenient ... especially fitting distribution parameters to data".
- **SciPy 1.15 endorsed SPEC 7** ([SciPy 1.15.0 release notes](https://docs.scipy.org/doc/scipy/release/1.15.0-notes.html)):
  - It added `rng` to many functions, including `differential_evolution`, `bootstrap`, `permutation_test` and the quasi-Monte Carlo (qmc) engines.
  - Advice: "users can avoid future disruption by proactively passing an instance of np.random.Generator by keyword rng".
  - Array API support is opt-in via `SCIPY_ARRAY_API=1`, with PyTorch, JAX, CuPy and ndonnx arrays.

#### pyhf (PyPI 0.7.6, 2024-01-05; no release since, so activity has slowed)
- **Swappable backend and one-call inference** ([pyhf docs](https://pyhf.readthedocs.io/en/stable/)):
  - `pyhf.set_backend("numpy")` switches the array library (numpy, jax, pytorch or tensorflow).
  - "All backends can be used in combination with all optimizers" (SciPy or MINUIT).
  - Models are built from declarative JSON: `pyhf.Workspace(spec).model()`.
  - One call runs the test: `CLs_obs, CLs_exp = pyhf.infer.hypotest(test_mu, data, model, test_stat="qtilde", return_expected=True)`.
  - Results are "validated against output computed from HistFactory workspaces".

#### corner (PyPI 2.3.0, 2026-07-05)
- **One user-facing function** ([corner API](https://corner.readthedocs.io/en/latest/api/)):
  - `corner.corner(data, bins=20, *, weights=None, labels=None, truths=None, quantiles=None, show_titles=False, ...)`.
  - `data` can be "Any object that can be converted to an arviz.InferenceData or xarrayDataTree object", selected with `group='posterior'`, `var_names`, `filter_vars` and `coords`.
- **A recurring confusion answered with a docs page** ([corner "A note about sigmas"](https://corner.readthedocs.io/en/latest/pages/sigmas/)):
  - "We are regularly asked about the default 'sigma' confidence levels in the 2D histograms".
  - The defaults (0.118, 0.393, 0.675, 0.864) are the (0.5, 1, 1.5, 2)-sigma regions of a 2-D Gaussian.
  - Users expect the 1-D Gaussian values (0.382, 0.683, 0.866, 0.955).

#### ChainConsumer (PyPI 1.3.0, 2025-12-12) and getdist (1.7.7, 2026-05-08)
- **Version 1 workflow** ([ChainConsumer usage](https://samreay.github.io/ChainConsumer/usage/)):
  - Put samples in a pandas DataFrame, then wrap them: `Chain(samples=df, name=...)`.
  - `c = ChainConsumer(); c.add_chain(chain)`.
  - `c.add_truth(Truth(...))`, `c.set_plot_config(PlotConfig(...))`, `c.add_override(ChainConfig(...))`.
- **Why the maintainer moved options into typed config objects** ([ChainConsumer "Why all these classes and not just kwargs?"](https://samreay.github.io/ChainConsumer/usage/)):
  - The old version "just took tons of kwargs and passed them around, which also caused a huge ton of duplicated docstring and functions".
  - With config objects, "You don't need to remember to update five other functions, and their docstring".
  - "I don't like the extra verbosity, but it's a price I'm willing to pay for more explicit code and better type hinting".

#### tinygp and celerite2 (tinygp 0.3.1, 2026-03-15; celerite2 0.3.3, 2026-07-12)
- **tinygp is a library, not a framework** ([Why tinygp?](https://tinygp.readthedocs.io/en/latest/motivation.html)):
  - "A fundamental design decision is that tinygp does not offer implementations of any inference routines". It plugs into numpyro, flax and blackjax instead.
  - It replaced george, whose "homebrewed modeling framework with limited and awkward support for named parameters, differentiation" was hard to use.
  - george's kernels were written in C++ with a YAML spec, so custom kernels meant recompiling.
- **celerite2: one algorithm, several front-ends.** It has "interfaces in Python and C++, with full support for PyMC (v3 and v5+) and JAX" ([celerite2 docs](https://celerite2.readthedocs.io/en/latest/)).

#### Bambi (PyPI 0.21.0, 2026-09-10)
- **A formula layer built on PyMC and ArviZ:**
  - `model = bmb.Model('Reaction ~ Days', data); results = model.fit(draws=1000); az.summary(results)` ([Bambi docs](https://bambinos.github.io/bambi/)).
  - Bambi "makes it easy to specify complex generalized linear hierarchical models using a formula notation similar to those found in R" ([Capretto et al., arXiv:2012.10754](https://arxiv.org/abs/2012.10754)).

#### sbi (PyPI 0.27.0, 2026-08-04; classed "3 - Alpha")
- **Easy path and full-control path:**
  - The docs promise: "Whether you need fine-grained control or an easy-to-use interface, sbi has you covered" ([sbi docs](https://sbi.readthedocs.io/en/latest/)).
  - Easy path: `inference = NPE(); inference.append_simulations(θ, x).train(); posterior = inference.build_posterior(); posterior.sample((1000,), x=x_o)`.
  - The toolkit paper describes "both high-level interfaces, extensive documentation and tutorials for practitioners, as well as low-level interfaces for experienced users and SBI researchers (giving full control over simulations, the training loop, and the sampling procedure)" ([Boelts et al., "sbi reloaded", arXiv:2411.17337](https://arxiv.org/abs/2411.17337)).
  - It "provides pre-configured defaults for all inference methods, but also allows full customization of every step" (same source).
- **Built-in diagnostics** ([sbi docs](https://sbi.readthedocs.io/en/latest/)):
  - `run_sbc` / `check_sbc` (simulation-based calibration).
  - `run_tarp` / `check_tarp` (coverage tests).
  - L-C2ST (local classifier test) and misspecification checks.
  - `sbc_rank_plot`, plus a "How to choose a diagnostic tool" guide.

#### NumPyro (PyPI 0.22.0, 2026-09-18) and PyMC (PyPI 6.3.2, 2026-09-08)
- **NumPyro separates the algorithm (kernel) from the driver, and uses explicit random keys** ([NumPyro getting started](https://num.pyro.ai/en/stable/getting_started.html)):
  ```python
  mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=1000)
  mcmc.run(random.key(0), J, sigma, y=y, extra_fields=('potential_energy',))
  mcmc.print_summary()
  Predictive(model, mcmc.get_samples())(random.key(1))
  ```
- **PyMC sampling call** ([pymc.sample](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html)):
  - Signature: `pymc.sample(draws=1000, *, tune=None, chains=None, cores=None, random_seed=None, progressbar=True, quiet=False, nuts_sampler=None, return_inferencedata=True, compile_kwargs=None, ...)`.
  - `random_seed` accepts an int, an array of ints or a Generator. "A TypeError will be raised if a legacy RandomState object is passed", because RandomState's seeding "does not allow easy spawning of new independent random streams".
  - `nuts_sampler` is one of "pymc", "nutpie", "blackjax", "numpyro"; "If None (default), 'nutpie' is used if installed".
  - Draws can be streamed to a `ZarrTrace`.
  - `cores` defaults to the number of CPUs, "but at most 4".
- **Compile delays:**
  - The PyTensor FAQ admits "function compilation can be time consuming" and offers `mode=FAST_COMPILE` ([PyTensor FAQ](https://pytensor.readthedocs.io/en/latest/faq.html)).
  - A PyTensor developer reports that compile times above 10 s are common, and one very large model took 30 min ([Numba discourse](https://numba.discourse.group/t/compilation-pipeline-compile-time-and-vectorization/1716)).

#### ArviZ 1.x (arviz 1.3.0, 2026-08-11; JOSS paper March 2026)
- **Why it was redesigned** ([Martin et al. 2026, JOSS 11(119) 9889](https://joss.theoj.org/papers/10.21105/joss.09889)):
  - Updating ess, rhat and loo "needed significant development effort ... because it wasn't possible to change a part of ArviZ without also adapting everything that interacted with it".
  - Statistics now come in two interfaces. One is "A low-level array interface with only numpy and scipy as dependencies, intended for advanced users and developers of third-party libraries".
  - The other is "A higher-level xarray interface designed for end users".
- **Migration changes** ([ArviZ 1.0 migration guide](https://python.arviz.org/en/latest/user_guide/migration_guide.html)):
  - The package is split into arviz-base, arviz-stats and arviz-plots.
  - "the arviz.InferenceData object doesn't exist anymore". It is replaced by `xarray.DataTree`, so "any format supported by xarray is automatically available".
  - The default credible-interval probability (`ci_prob`) changed from 0.94 to 0.89.
  - New xarray accessors: `dt.azstats.ess()` and `dt.azstats.hdi(dim=[...])`.
  - The compute engine is pluggable through `rcParams["stats.module"]` ("base", "numba" or a custom module).
  - Plotting supports matplotlib, bokeh and plotly "with (mostly) feature parity".
- **Installed source (v1.3.x)** ([arviz-base on PyPI](https://pypi.org/project/arviz-base/); [arviz-stats on PyPI](https://pypi.org/project/arviz-stats/); [arviz-stats API](https://arviz-stats.readthedocs.io/en/latest/api/index.html)):
  - Converters: `from_dict(data, *, name=None, sample_dims=None, coords=None, dims=None, ...)`, `from_emcee`, `from_numpyro`, `from_cmdstanpy`, `from_pystan`.
  - `arviz_stats.base.array_stats.ess(ary, chain_axis=-2, draw_axis=-1, method="bulk")`.
  - `array_stats.rhat(ary, chain_axis=-2, draw_axis=-1, method="rank")`, plus `rhat_nested` and `mcse`.
  - `array_stats.uniformity_test(ary, axis=-1, method="pot_c")` returns a p-value and per-point contributions for probability-integral-transform (PIT) values.
  - The top-level API also has `diagnose()` ("Run comprehensive diagnostic checks for MCMC sampling"), `psense_summary` (sensitivity to prior and likelihood scaling), `text_ppc_pit` and `summary`.

#### Patterns that cut across packages: seeding, backends, array libraries, configuration
- **SPEC 7** ([SPEC 7](https://scientific-python.org/specs/spec-0007/)):
  - Use one `rng` keyword, normalised with `np.random.default_rng(rng)`.
  - Deprecate `random_state` and `seed` arguments, and stop honouring `np.random.seed`.
  - Endorsed by ipython, networkx, numpy, pysal, scikit-image and scipy.
  - Suggested signature: `def f(*, rng: RNGLike | SeedLike | None = None)`, with `SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence` and `RNGLike = np.random.Generator | np.random.BitGenerator`.
  - It also supplies a docstring template and a `_transition_to_rng` decorator for the deprecation.
- **Independent random streams for parallel work:** `default_rng(12345).spawn(10)`, or `SeedSequence(12345).spawn(10)` ([NumPy parallel random generation](https://numpy.org/doc/stable/reference/random/parallel.html)).
- **Array API helper libraries:**
  - `xp = array_api_compat.array_namespace(x, y)` returns a standard-conforming namespace. It covers NumPy, CuPy, PyTorch, Dask, JAX, ndonnx, sparse and dpnp ([array-api-compat](https://data-apis.org/array-api-compat/)).
  - array-api-extra adds functions for "array-consuming" libraries ([array-api-extra](https://data-apis.org/array-api-extra/)).
- **BayesFlow 2** has "Multi-backend support via Keras3": PyTorch, TensorFlow or JAX, chosen by `KERAS_BACKEND` ([BayesFlow README](https://github.com/bayesflow-org/bayesflow/blob/main/README.md)).
- **Keras API design guidelines** ([Keras API design guidelines](https://github.com/keras-team/governance/blob/master/keras_api_design_guidelines.md)):
  - "Best-practices should come baked into your API ... all arguments that can be given a default value should be given a default value, and that default should match the most common use case".
  - "Plain Python types are preferable to custom types ... When using enums, make sure that their values are strings".
  - "Explicit, single-level configuration arguments are preferable to nested, hidden configuration arguments".
  - "If you find yourself proposing a signature with more than 6-7 arguments, consider whether all of these arguments are useful".
  - "Things that create state or side-effects should be classes. Functions should be stateless".
  - "Objects that do interchangeable things should have identical or very close APIs".
  - "Catch user errors early and anticipate common mistakes".
  - Error messages should say "What happened, in what context? What did the software expect? How can the user fix it?".
  - "All docstrings should include code examples".

### Inferences
Fit of each exemplar for pastry (my judgement, not sourced):
- **emcee: adopt, with changes.**
  - Copy: the plain-function mindset; `run()` plus a `sample()` generator; a backend whose in-memory and HDF5 versions have the same accessors and an explicit `reset`; the `vectorize` and `pool` hooks.
  - Avoid: global random state, a one-line FAQ on initialisation, and leaving users to write their own convergence loops.
  - Pastry should have a default stopping rule based on effective sample size (ESS) and the convergence statistic R̂, and should print why it stopped, as UltraNest does.
- **scikit-learn: adopt the conventions, not full compliance.**
  - Copy: the `__init__`/`fit`/trailing-underscore contract, `get_params`/`set_params`, the HTML display, and `score()` returning the log marginal likelihood.
  - The data model does not fit. Pastry's input is an N×K likelihood matrix, or posterior samples with an interim prior, not a feature matrix.
  - `check_estimator` would therefore need many `expected_failed_checks`. A scikit-learn-*style* estimator is the right level, optionally inheriting `BaseEstimator` for `clone` and `GridSearchCV`.
  - SLEP023's callbacks (ProgressBar, EarlyStopping, logging) are a good template for progress reporting and monitoring.
- **dynesty 3.0 and nautilus: adopt.**
  - Pass configured strategy objects (`sampler=...`, `bound=...`) instead of piling up keyword arguments.
  - Copy nautilus's declarative `Prior` object and its one-keyword, auto-resuming checkpoint (`filepath=`).
  - Heed dynesty's warning that checkpoints are not an archive format. Save final results in a documented format such as netCDF or Zarr via `DataTree`.
- **UltraNest: adopt its run output.** Explain the stopping decision, run a self-check, show text histograms in the terminal, and copy `print_results()` and `log_dir`/`resume`.
- **ArviZ 1.x: adopt as the output and diagnostics layer.**
  - Return an `xarray.DataTree` whose posterior group has dimensions `chain, draw, bin`. corner, ArviZ plots and netCDF/Zarr input/output then work with no extra code.
  - Internally, depend only on `arviz_stats.base.array_stats` (NumPy and SciPy only) for R̂, ESS and uniformity tests.
  - ArviZ 1.0 is new and broke `InferenceData`, so set minimum versions. Also keep a plain NumPy result (`result.samples`) so users are not forced into xarray.
- **lmfit and nautilus: adopt named parameters and bins.** Keep bin edges and labels on the result, and let users fix or tie parameters without rewriting likelihood code.
- **SciPy, SPEC 7 and PyMC: adopt `rng=` everywhere.**
  - Accept an int, a Generator or a SeedSequence. Reject `RandomState`, or warn about it, as PyMC does.
  - Spawn a separate stream per chain.
  - With a JAX backend, keep `rng` as the user-facing argument and derive JAX keys internally.
- **pyhf, BayesFlow and scikit-learn backend switching: defer.**
  - A swappable array backend is attractive for the planned normalising-flow and flow-matching extensions.
  - The core Dirichlet–multinomial model is cheap in NumPy: N×K matrix–vector products and log-sum-exp.
  - Start NumPy-first, with kernels written so array-api-compat can run them. Add JAX or PyTorch for the flow models later rather than in the core.
- **tinygp: adopt the composability lesson.** Expose the log-density as plain functions (`log_marginal(rho)` and its gradient) so experts can plug pastry's model into NumPyro, BlackJAX or emcee.
- **astropy.units: optional support only.** Accept `Quantity` bin edges, since colour–magnitude axes and distances have units even though redshift does not. Do not make astropy a required dependency.
- **Bambi and statsmodels formula front-ends: do not adopt for the core.** They target regression, and pastry has no formula language. The equivalent entry point is one function, `pastry.fit(likelihoods, ...)`.
- **corner and ChainConsumer: use them, do not rebuild them.**
  - Provide `result.plot()` built on corner or ArviZ.
  - If pastry draws 2-D contours, document which sigma levels it uses, given corner's experience.

### Gaps
- GitHub star and open-issue counts could not be gathered because the API was blocked. Maintenance status is inferred from PyPI release dates only.
- No user-survey data comparing the usability of these packages was found. The evidence is maintainers' statements, documentation and issue titles; the emcee pain points come from titles, not issue bodies.
- No scikit-learn proposal or roadmap item on numpy `Generator` support was found, so its status is unknown.
- PyMC compile-time complaints were not found on PyMC's own Discourse forum. Only the PyTensor documentation and a Numba-forum thread were found.

## Q2. Packaging and documentation norms in 2026

### Takeaway
Norms for a new scientific Python package in late 2026:
- `pyproject.toml` with hatchling (or `uv_build`) and an SPDX licence identifier.
- Minimum Python and NumPy versions set by SPEC 0, which drops Python 3.12 in 2026-Q4.
- ruff for linting and formatting, run through pre-commit or its faster Rust rewrite, prek.
- A type checker: mypy, pyrefly, or ty (still beta).
- Strict pytest settings, including `filterwarnings = ["error"]`.
- nox as the task runner.
- uv-based GitHub Actions test matrices and Trusted Publishing to PyPI.
- Sphinx with MyST and numpydoc, executing the tutorial notebooks during each docs build. The MkDocs ecosystem is in flux, and Jupyter Book 2 is built on MyST-MD.
- asv, pytest-benchmark or CodSpeed for benchmarks.

Since 2026, JOSS also requires at least six months of public, step-by-step development and evidence of research use before you submit.

### Cited Findings

#### Support windows, versioning and deprecation
- **SPEC 0** ([SPEC 0](https://scientific-python.org/specs/spec-0000/)):
  - Drop support for a Python version 3 years after its release, and for a core package version 2 years after its release.
  - Drops due in 2026-Q4: Python 3.12 (released Oct 2023), numpy 2.2.0, scikit-learn 1.6.0, matplotlib 3.10.0, networkx 3.4, xarray 2024.10–2024.11.
  - Due in 2027-Q1: scipy 1.15.0.
- **Status of the SPECs in October 2026** ([SPEC index](https://scientific-python.org/specs/)):
  - Endorsed: SPEC 0, 1 (lazy loading), 4 (nightly wheels), 6 (keys to the castle), 7 (seeding random numbers), 8 (securing the release process).
  - Drafts: SPEC 2 (API dispatch), 3 (accessibility), 5 (CI best practices), 9 (governance).
- **Minimum Python versions actually required on PyPI** ([PyPI JSON API, e.g. scipy](https://pypi.org/pypi/scipy/json)):
  - Python ≥3.12: scipy 1.18.1, numpy 2.5.4, arviz 1.3, pymc 6.3.2, bambi 0.21.
  - Python ≥3.11: scikit-learn 1.9.1.
  - Python ≥3.13: sphinx 9.2.0.
- **NumPy NEP 23** ([NEP 23](https://numpy.org/neps/nep-0023-backwards-compatibility.html)):
  - Deprecations use `DeprecationWarning` and "shall not be introduced in micro (bug fix) releases".
  - Removal comes "after at least 2 releases ... there shall be at least 1 year between deprecation and removal".
  - Changes in behaviour use `FutureWarning`.
  - Reason given: "most users ... do not notice deprecation warnings, and sometimes wait more than one or two years before upgrading".
- **scikit-learn** keeps renamed public API working "for two releases", with warnings. Default changes go through a `"warn"` placeholder value and a `FutureWarning` ([scikit-learn contributing](https://scikit-learn.org/stable/developers/contributing.html)).
- **Version numbering in practice:** several tools use calendar versions (xarray 2026.9.0, nox 2026.10.20, furo 2025.12.19). numpy, scipy and scikit-learn use major.minor.patch with deprecation cycles ([PyPI](https://pypi.org/project/xarray/)).

#### Peer review: JOSS (2026 rules) and pyOpenSci
- **JOSS rejects without review unless all four criteria are met** ([JOSS "Submitting a paper"](https://joss.readthedocs.io/en/latest/submitting.html)):
  1. More than six months of public history: "We run automated checks on commit distribution — a repo dump is not a history".
  2. Evidence of research use: "Aspirational statements about future use are not sufficient".
  3. Good open-source practice. For single-author projects this means a commit history, tagged releases or a changelog, tests and CI, documentation, a CONTRIBUTING file, and a statement on support or governance.
  4. Development in steps over time, not a single burst.

  "Minor utility" packages and single-function packages are out of scope.
- **JOSS AI policy:** a mandatory "AI usage disclosure" listing tools and versions, what the AI did, and confirmation that humans reviewed the output and made the core design decisions ([JOSS AI Usage Policy](https://joss.readthedocs.io/en/latest/submitting.html); background in the [JOSS blog, Jan 2026](https://blog.joss.theoj.org/2026/01/preparing-joss-for-a-generative-ai-future)).
- **JOSS paper sections and reviewer checks** ([JOSS review checklist](https://joss.readthedocs.io/en/latest/review_checklist.html)):
  - Required paper sections: Summary; Statement of need; State of the field (including why build rather than contribute to an existing tool); Software design (trade-offs); Research impact statement; AI usage disclosure.
  - Reviewers check installation, functional claims, performance claims, example usage, API documentation, automated tests and community guidelines.
- **pyOpenSci reviewer checklist** ([pyOpenSci review templates](https://www.pyopensci.org/software-peer-review/appendices/templates.html)):
  - A statement of need and install instructions in the README.
  - "Short quickstart tutorials demonstrating significant functionality that successfully runs locally".
  - Documentation for every user-facing function.
  - Community guidelines and package metadata in `pyproject.toml`.
  - README badges for CI and test coverage.
  - Tests that "cover essential functions of the package and a reasonable range of inputs and conditions".
  - Continuous integration, conformance with the pyOpenSci packaging guide, and passing PEP 8 lint checks.

#### Scientific Python Development Guide (SPDG) recommendations
- **Packaging** ([SPDG simple packaging](https://learn.scientific-python.org/development/guides/packaging-simple/)):
  - "hatchling is an excellent choice" as the build backend. Alternatives: `uv_build`, flit-core, pdm-backend, setuptools.
  - Declare the licence with an SPDX identifier, e.g. `license = "BSD-3-Clause"`, plus `license-files`.
- **pytest settings** ([SPDG pytest](https://learn.scientific-python.org/development/guides/pytest/)):
  - `minversion = "9.0"`, `addopts = ["-ra", "--showlocals"]`, `strict = true`.
  - For older pytest: `--strict-markers`, `--strict-config`, `xfail_strict = true`.
  - `filterwarnings = ["error"]` and `log_level = "INFO"`.
  - Prefer `array1 == approx(array2)` over `numpy.testing`, because the failure reports are clearer.
- **Style tooling** ([SPDG style](https://learn.scientific-python.org/development/guides/style/)):
  - pre-commit, or prek ("a faster Rust rewrite").
  - ruff-format; since version 0.16 it also formats code blocks in Markdown.
  - ruff-check with `lint.extend-select = [ARG, B, C4, EM, EXE, FURB, G, ICN, ISC, LOG, NPY, PD, PERF, PGH, PL, PT, PTH, PYI, Q, RET, RSE, RUF, SIM, SLOT, T10, T20, TC, TRY, UP]`.
  - Type-checking hooks for mypy, pyrefly or ty.
  - pre-commit.ci to apply fixes automatically.
- **Tasks and CI** ([SPDG tasks](https://learn.scientific-python.org/development/guides/tasks/); [SPDG GitHub Actions](https://learn.scientific-python.org/development/guides/gha-basic/)):
  - nox is the recommended task runner ("very explicit"), run in CI with the `wntrblm/nox` action.
  - Test matrix with `fail-fast: false` and `allow-prereleases: true`.
  - Use `astral-sh/setup-uv`, dependabot to keep the CI actions updated, and Trusted Publishing to PyPI.
- **Documentation** ([SPDG docs](https://learn.scientific-python.org/development/guides/docs/)):
  - Sphinx with the MyST Markdown parser; astropy and corner are examples.
  - Recommended extensions: myst_parser, autodoc, intersphinx, mathjax, napoleon, sphinx_autodoc_typehints, sphinx_copybutton.
  - Theme: furo or pydata-sphinx-theme.
  - Alternatives: ProperDocs ("a maintained fork of MkDocs"), Zensical ("still in early development"), and MyST / Jupyter Book.
  - "MkDocs's future is uncertain, and mkdocs-material will be minimally maintained until late 2026".
- **Testing principles** ([SPDG testing principles](https://learn.scientific-python.org/development/principles/testing/)):
  - Unit tests should run "in just a few seconds at most".
  - Use markers such as `slow` to separate test suites.
  - Hypothesis fuzz tests are slow and belong in a separately run suite.

#### Documentation tools (October 2026)
- **numpydoc 1.11.0** (2026-09-15) has a `numpydoc-validation` pre-commit hook and a `numpydoc lint` command. Configure it under `[tool.numpydoc_validation]`, e.g. `checks = ["all", "EX01", "SA01", ...]` ([numpydoc validation](https://numpydoc.readthedocs.io/en/latest/validation.html)).
- **MyST-NB runs notebooks while the docs build** ([MyST-NB execution](https://myst-nb.readthedocs.io/en/latest/computation/execute.html)):
  - `nb_execution_mode` can be off, force, auto, cache (via jupyter-cache) or inline.
  - `nb_execution_timeout` defaults to 30 s.
  - Also available: `nb_execution_allow_errors` and `nb_execution_excludepatterns`.
- **Jupyter Book 2** is a rewrite on the MyST-MD engine (alpha Nov 2024, beta 2025). Version 2.1.7 on PyPI (2026-09-23) is still classed "3 - Alpha" ([Jupyter Book 2 alpha announcement](https://jupyterbook.org/blog/posts/2024-11-15-jupyter-book-2-alpha); [PyPI jupyter-book](https://pypi.org/project/jupyter-book/)).
- **MkDocs ecosystem:**
  - MkDocs 1.6.1 has had no release since 2024-08-30 ([PyPI mkdocs](https://pypi.org/project/mkdocs/)).
  - Material for MkDocs is in maintenance mode: 9.7.0 was the last feature release, with fixes until about November 2026 ([pydevtools handbook](https://pydevtools.com/handbook/reference/mkdocs-material/)).
  - Its successor, Zensical, is at version 0.0.71 (2026-10-16), classed "3 - Alpha" ([PyPI zensical](https://pypi.org/project/zensical/)).
- **Current tool versions on PyPI, 2026-10-22:**
  - uv 0.13.5: "An extremely fast Python package and project manager", "A single tool to replace pip, pip-tools, pipx, poetry, pyenv, twine, virtualenv" ([PyPI uv](https://pypi.org/project/uv/); [uv docs](https://docs.astral.sh/uv/)).
  - ruff 0.17.1.
  - ty 0.0.90 (beta): "An extremely fast Python type checker and language server" ([ty docs](https://docs.astral.sh/ty/)).
  - mypy 2.4.0, pre-commit 4.7.0, prek 0.5.7, nox 2026.10.20, hatchling 1.32.4.
  - pytest 9.1.1, pytest-benchmark 5.3.0, pytest-codspeed 5.0.3, hypothesis 6.170.5, pydata-sphinx-theme 0.23.0.
  - asv 0.6.6: "a tool for benchmarking Python packages over their lifetime. Runtime, memory consumption and even custom-computed values may be tracked" ([asv docs](https://asv.readthedocs.io/en/latest/)).
  - pixi, for conda plus PyPI environments: "a fast, modern, and reproducible package management tool" ([pixi docs](https://pixi.sh/latest/)).

### Inferences
- **Minimum versions.** A first pastry release in late 2026 or early 2027 should require:
  - Python ≥3.12, or ≥3.13 if released after 2026-Q4 (SPEC 0);
  - numpy ≥2.3;
  - scipy ≥1.15 or ≥1.16.
  - Make the heavier dependencies optional extras, e.g. `pastry[plot]`, `pastry[io]`, `pastry[jax]`, `pastry[flows]`: ArviZ 1.x, h5py/zarr, matplotlib/corner, astropy, JAX and PyTorch.
- **Starting template.** Begin from scientific-python/cookie, which sets up hatchling, nox, ruff, pytest and GitHub Actions ([cookie README](https://github.com/scientific-python/cookie)). Add:
  - uv for development environments;
  - prek or pre-commit running ruff, numpydoc-validation and a type checker (mypy is the stable choice today; ty is still beta).
- **Documentation.**
  - Use Sphinx, MyST-NB and pydata-sphinx-theme. Execute the tutorials on every build (`nb_execution_mode="cache"` locally, `"force"` in CI), so each tutorial also works as a quick test.
  - Avoid MkDocs and Material, given their maintenance status.
  - Jupyter Book 2 is workable but adds a Node.js-based MyST-MD toolchain and is still classed alpha.
- **Open the repository from day one.**
  - JOSS requires six months of public history and evidence of research use, so pastry should be public early, with tagged releases, an issue tracker and a CONTRIBUTING file.
  - A paper applying it (e.g. photo-z or colour–magnitude-diagram population inference) should exist before submitting to JOSS.
  - pyOpenSci review is a complementary venue.
- **Deprecation policy.** Follow NEP 23 and scikit-learn: keep deprecated APIs for at least two minor releases, use `FutureWarning` for changes in behaviour, and only ever add a seeding argument named `rng` (SPEC 7).

### Gaps
- No authoritative 2026 survey of how scientific packages split between Sphinx, Jupyter Book 2 and Zensical was found.
- SPEC 5 (CI best practices) is a draft and was not read in detail.
- The current pixi version could not be confirmed: the PyPI name "pixi" belongs to an unrelated package, and the GitHub API was blocked.

## Q3. Validation and testing of statistical/Bayesian software, and reusable tooling

### Takeaway
A credible validation stack for a hierarchical-inference package has six parts:
1. Unit tests of the log-density and its gradients against brute-force and exact calculations.
2. Exact reference cases where the answer is known in closed form (conjugate priors).
3. Seeded integration tests whose tolerances scale with the run's own Monte Carlo error (dynesty's approach).
4. Geweke "getting it right" tests for any Gibbs-style sampler.
5. Simulation-based calibration (SBC) that includes data-dependent test quantities, especially the joint log-likelihood (Modrák et al. 2023), plus coverage tests (TARP or expected coverage) for neural density estimators.
6. Rank-normalised R̂ and ESS checks on every run.

Reusable building blocks exist:
- ArviZ's NumPy-only `array_stats` (R̂, ESS, uniformity tests) and `plot_ecdf_pit`;
- the diagnostics suites in sbi and BayesFlow;
- simuk, which only works with PyMC, Bambi or NumPyro models;
- the SBC R package;
- posteriordb reference posteriors.

Tests that fail at random ("flaky" tests) are known to be common in probabilistic code. They are kept in check with:
- fixed seeds;
- `slow` markers;
- an environment variable to loop over seeds;
- sequential statistical tests;
- rerun plugins, used sparingly.

### Cited Findings

#### Methods (primary sources)
- **SBC, the basic method** ([Talts et al. 2018, arXiv:1804.06788](https://arxiv.org/abs/1804.06788)):
  - Draw θ̃ from the prior, simulate data ỹ, fit, and rank θ̃ among the posterior draws.
  - If the computation is correct, the ranks are uniformly distributed.
  - SBC "identifies inaccurate computation and inconsistencies in model implementations but also provides graphical summaries that can indicate the nature of the problems that arise".
- **Which quantities to rank matters** ([Modrák et al., Bayesian Analysis 2023, DOI 10.1214/23-BA1404, arXiv:2211.02383](https://arxiv.org/abs/2211.02383)):
  - Ranking parameters alone "could never detect large classes of problems including when the posterior is equal to the prior".
  - Adding quantities that depend on the data fixes this. "the joint likelihood of the data is an especially useful test quantity".
  - The paper also distinguishes SBC from checks based on the data-averaged posterior.
- **Graphical uniformity test:** simultaneous confidence bands for the empirical CDF of ranks or PIT values ([Säilynoja, Bürkner & Vehtari 2022, arXiv:2103.10522](https://arxiv.org/abs/2103.10522)).
- **Newer uniformity tests** for dependent PIT values (pot_c, piet_c, prit_c), now ArviZ's defaults ([Tesso & Vehtari 2026, arXiv:2603.02928](https://arxiv.org/abs/2603.02928)).
- **Posterior SBC** checks inference conditional on the observed data, by refitting on the original data plus posterior-predictive data ([Säilynoja et al. 2025, arXiv:2502.03279](https://arxiv.org/abs/2502.03279)).
- **Stan User's Guide, SBC chapter** ([Stan User's Guide: SBC](https://mc-stan.org/docs/stan-users-guide/simulation-based-calibration.html)):
  - The rank r = Σ_m I[θ^(m) < θ̃] should be uniform on 0..M.
  - "thin the posterior draws down to the point where the effective sample size is roughly the same as the number of thinned draws", because autocorrelation creates artefacts.
  - A χ² test on binned ranks (at least 5 expected per bin) is "simple, though not very highly powered".
- **SBC as two independent implementations** ([SBC R package vignette](https://hyunjimoon.github.io/SBC/articles/SBC.html)):
  - The data generator and the fitted model are separate codes. If data are simulated exactly as the model assumes, "any problem has to be a bug".
  - The R package also caches fits and draws rank and ECDF plots.
- **Geweke's "Getting it right" test** ([Geweke 2004, JASA 99(467):799–804](https://ideas.repec.org/r/bes/jnlasa/v99y2004p799-804.html); [LIPS blog, "Testing MCMC code, part 2"](https://lips.cs.princeton.edu/testing-mcmc-code-part-2-integration-tests/)):
  - Compare forward samples (θ from the prior, then x from the likelihood) with a chain that alternates the MCMC update and fresh x ~ p(x|θ).
  - The blog calls it "the gold standard for testing MCMC algorithms".
  - Small errors compound: "Pretty soon, the noise variance will explode".
  - Compare several statistics with P-P plots, and only run it once unit tests pass.
  - It cannot locate a bug, and it does not show that the sampler converges on real problems.
- **Testing MCMC code:** keep "conditional probability calculations ... separate from the logic of the sampler", unit-test them, and integration-test the sampler separately ([Grosse & Duvenaud 2014, arXiv:1412.5218](https://arxiv.org/abs/1412.5218)).
- **Exact tests for unit-testing samplers** ([Gandy & Scott 2020, arXiv:2001.06465](https://arxiv.org/abs/2001.06465)):
  - Exact tests that an MCMC step leaves the right distribution unchanged.
  - Embedded in a sequential procedure with "low expected effort if the simulation shows the desired behavior and high power if it does not".
  - "the false rejection probability can be kept arbitrarily low".
  - Implemented only in the R package MCUnit.
- **Convergence diagnostics:** rank-normalised, folded split-R̂ with bulk and tail ESS. The classic R̂ "will fail to correctly diagnose convergence failures when the chain has a heavy tail or when the variance varies across the chains" ([Vehtari et al. 2021, arXiv:1903.08008](https://arxiv.org/abs/1903.08008)).
- **Neural (amortised) posteriors:**
  - All benchmarked SBI algorithms "can yield overconfident posterior approximations"; ensembles of estimators help ([Hermans et al. 2021, arXiv:2110.06581](https://arxiv.org/abs/2110.06581)).
  - TARP tests coverage using samples only, and is "necessary and sufficient to show that a posterior estimator is accurate" ([Lemos et al. 2023, arXiv:2302.03026](https://arxiv.org/abs/2302.03026)).
  - L-C2ST gives diagnostics for each individual observation ([Linhart et al. 2023, arXiv:2306.03580](https://arxiv.org/abs/2306.03580)).
- **Reference posteriors:** posteriordb is "a database of models and data sets defining target densities along with reference Monte Carlo draws" (120 models) ([Magnusson et al. 2024, arXiv:2407.04967](https://arxiv.org/abs/2407.04967)).
- **Bayesian workflow:** SBC, prior and posterior predictive checks, and diagnostics as routine steps ([Gelman et al. 2020, arXiv:2011.01808](https://arxiv.org/abs/2011.01808)).
- **Flaky tests in probabilistic and machine-learning code:**
  - Dutta et al. ran the first large study, of projects built on Pyro, PyMC3, TensorFlow Probability and PyTorch. Per the abstract, they classified 75 bug reports and commits about flaky tests by cause and fix ([Dutta et al., ISSTA 2020](https://conf.researchr.org/details/issta-2020/issta-2020-papers/41/Detecting-Flaky-Tests-in-Probabilistic-and-Machine-Learning-Applications); [NSF PAR record](https://par.nsf.gov/biblio/10176758-detecting-flaky-tests-probabilistic-machine-learning-applications)).
  - The author's page lists follow-ups: FLEX (tuning assertion bounds) and TERA (optimising stochastic tests) ([Saikat Dutta homepage](https://www.cs.cornell.edu/~saikatd)).

#### Reusable tools
- **ArviZ's array layer (needs only NumPy and SciPy)** ([arviz-stats on PyPI](https://pypi.org/project/arviz-stats/); [Martin et al. 2026](https://joss.theoj.org/papers/10.21105/joss.09889)):
  - `array_stats.rhat(ary, chain_axis=-2, draw_axis=-1, method="rank")` and `array_stats.ess(..., method="bulk")`.
  - Also `mcse` and `rhat_nested`.
  - `array_stats.uniformity_test(pit, axis=-1, method="pot_c")` returns a p-value and each point's contribution.
- **`arviz_plots.plot_ecdf_pit`** ([plot_ecdf_pit docs](https://arviz-plots.readthedocs.io/en/latest/api/generated/arviz_plots.plot_ecdf_pit.html)):
  - Signature: `plot_ecdf_pit(dt, *, group='prior_sbc', method='pot_c', coverage=False, backend=None, ...)`.
  - Plots the difference between the empirical CDF of SBC values and uniform, with envelopes.
  - "pot_c: Good default choice due to its good power against diverse type of local departures from the null".
  - `coverage=True` shows the coverage of central credible intervals.
- **simuk 0.3.0** (2026-07-11, classed "3 - Alpha", from the ArviZ developers) ([simuk docs](https://simuk.readthedocs.io/en/latest/); [simuk sbc.py](https://github.com/arviz-devs/simuk/blob/main/simuk/sbc.py)):
  - `simuk.SBC(model, num_simulations=100, sample_kwargs={...}, seed=..., simulator=..., transform=..., method="prior"|"posterior")`, then `sbc.run_simulations()`.
  - `model` must be a PyMC model, a Bambi model, or a NumPyro MCMC kernel.
  - `transform` defines scalar test quantities.
- **sbi diagnostics** ([sbi SBC tutorial](https://sbi.readthedocs.io/en/latest/advanced_tutorials/11_diagnostics_simulation_based_calibration.html)):
  - `ranks, dap_samples = run_sbc(thetas, xs, posterior, num_posterior_samples=1000, num_workers=...)`.
  - `check_sbc(...)` returns a Kolmogorov–Smirnov p-value per dimension and classifier two-sample-test (C2ST) accuracies, where about 0.5 means uniform ranks.
  - Also `run_tarp` / `check_tarp`, L-C2ST and `sbc_rank_plot`.
- **BayesFlow 2.0.14 diagnostics** ([bayesflow/diagnostics/__init__.py](https://github.com/bayesflow-org/bayesflow/blob/main/bayesflow/diagnostics/__init__.py)):
  - Plots: `calibration_ecdf`, `calibration_histogram`, `coverage`, `recovery`, `z_score_contraction`, `mc_calibration`, `mmd_hypothesis_test`.
  - Metrics: `calibration_error`, `expected_calibration_error`, `posterior_contraction`, `posterior_z_score`, `root_mean_squared_error`, `classifier_two_sample_test`, `calibration_log_gamma`.
- **tarp 0.1.1** (last release 2023-11-21) implements the TARP coverage test ([tarp README](https://github.com/Ciela-Institute/tarp)).
- **SBC (R)** separates the data generator from the fitting backend, caches fits, supports derived quantities, and plots ranks and ECDFs ([SBC R package](https://hyunjimoon.github.io/SBC/)).
- **Reference-posterior clients are stale:** the posteriordb Python client on PyPI is at 0.2.0 from 2020 ([PyPI posteriordb](https://pypi.org/project/posteriordb/)), and inference-gym is at 0.0.5 (2025-01-24) ([PyPI inference-gym](https://pypi.org/project/inference-gym/)).

#### How mature samplers test themselves in CI
- **emcee** ([emcee test_proposal.py](https://github.com/dfm/emcee/blob/main/src/emcee/tests/integration/test_proposal.py)):
  - Fixed seed (`np.random.seed(1234)`), 32 walkers × 2000 steps on a unit Gaussian.
  - Asserts: acceptance rate between 0.1 and 0.9, `|mean| < 0.08`, `|std − 1| < 0.05`, and a 1-D Kolmogorov–Smirnov statistic below 0.05.
- **dynesty** ([dynesty tests/test_gau.py](https://github.com/joshspeagle/dynesty/blob/master/tests/test_gau.py); [dynesty tests/utils.py](https://github.com/joshspeagle/dynesty/blob/master/tests/utils.py)):
  - Tolerances scale with the run's own reported error: `assert np.abs(logz - g.logz_truth) < sig * results.logzerr[-1]` with `sig = 4`.
  - Tolerances for the posterior mean and covariance come from bootstrap resampling of the run.
  - The true values are exact: a Gaussian likelihood inside a uniform box.
  - `get_rstate()` returns `np.random.default_rng(seed)` with default seed 56432. The environment variable `DYNESTY_TEST_RANDOMSEED` overrides it "to run long tests by looping over seed value to catch potentially rare behaviour".
- **pytest-randomly 5.0.0** ([pytest-randomly README](https://github.com/pytest-dev/pytest-randomly)):
  - Shuffles test order and prints a base seed, reusable with `--randomly-seed`.
  - Resets the Python, NumPy and Faker random states before each test, so "each test gets a different but repeatable random seed".
- **pytest-rerunfailures 16.7:** `@pytest.mark.flaky(reruns=5)` and `pytest --reruns 5 --only-rerun AssertionError` ([pytest-rerunfailures README](https://github.com/pytest-dev/pytest-rerunfailures)).
- **Hypothesis (random test-input generation):** `hypothesis.extra.numpy.arrays(dtype, shape, elements=...)`, `array_shapes`, and `hypothesis.extra.array_api.make_strategies_namespace(xp)` for Array-API libraries ([Hypothesis strategies](https://hypothesis.readthedocs.io/en/latest/reference/strategies.html)).
- **pyhf** validates its results against the reference ROOT/HistFactory implementation ([pyhf docs](https://pyhf.readthedocs.io/en/stable/)).
- **scikit-learn** ships a conformance test suite, `parametrize_with_checks`, that other projects run in their own CI ([parametrize_with_checks](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.parametrize_with_checks.html)).

### Inferences
- **Exact reference cases (fast and deterministic, run on every commit):**
  - With "one-hot" likelihoods (each object certainly in one bin, L_ik = δ_{k,z_i}), the posterior is exactly Dir(α + counts).
  - With K = 2 bins, the marginal likelihood is a 1-D integral, so numerical quadrature gives the true evidence and posterior quantiles.
  - For an EM or maximum-a-posteriori solver, check the fixed-point equation ρ_k ∝ Σ_i L_ik ρ_k / Σ_j L_ij ρ_j (plus the α−1 prior terms) against brute-force optimisation.
  - Check gradients against finite differences.
  - Check that rescaling one object's likelihood row by a constant leaves the posterior unchanged. Hypothesis can generate the random L matrices.
- **Gibbs sampler.** If pastry alternates between sampling the z_i and ρ, add a Geweke joint-distribution test. With small N and K it is cheap, and it catches the classic off-by-(α−1) bugs in Dirichlet updates.
- **Ship SBC as a user-facing tool, not just an internal test:**
  - Provide `pastry.validate.sbc(model, n_sims=200, rng=...)`.
  - Draw ρ from the prior, z from Cat(ρ), then per-object likelihoods L from a noise model the user supplies. This reuses the realistic, object-specific noise generator users already have.
  - Rank each ρ_k, and also the joint log-likelihood log p(d|ρ) as a test quantity (Modrák).
  - Report `array_stats.uniformity_test` p-values, with an optional `plot_ecdf_pit`.
  - Building this into pastry avoids simuk's requirement for a PyMC, Bambi or NumPyro model.
- **Interim-prior and posterior-sample inputs.** Add SBC and coverage tests where the per-object inputs are posterior samples drawn under a deliberately wrong interim prior. Mishandling the interim-prior correction should then show up as non-uniform ranks.
- **Normalising flows and flow matching.** Use expected-coverage or TARP tests and classifier two-sample tests, as sbi and BayesFlow do. Ensembles can reduce the overconfidence Hermans et al. documented.
- **CI design:**
  - Unit and exact-answer tests run on every push with fixed `rng` seeds.
  - Statistical integration tests use dynesty-style tolerances: about 4 × the reported Monte Carlo error.
  - SBC and coverage suites are marked `slow` and run nightly or weekly, looping over seeds via an environment variable.
  - Prefer sequential tests (Gandy & Scott) or calibrated tolerances over blanket `--reruns`, which can hide real bugs.
- **Diagnostics on every run.** Every MCMC result should report rank-normalised R̂ and bulk and tail ESS, with warnings (via `arviz_stats.base.array_stats`). Like UltraNest, it should state which convergence criteria were met.

### Gaps
- Cook, Gelman & Rubin (2006, posterior quantiles) could not be retrieved and checked, so it is not cited.
- The "75 flaky tests" figure for Dutta et al. comes from the indexed abstract, not the paper itself.
- No Python package implementing Geweke tests or sequential MCMC unit tests was found (MCUnit is R-only), so pastry would need its own.
- No public SBC study of Dirichlet–multinomial deconvolution with object-specific likelihoods was found in this pass. The domain researchers may cover it.

## Q4. One-line high-level interfaces layered over flexible low-level ones (progressive disclosure)

### Takeaway
Successful packages offer three or four layers:
1. A one-call function or method whose defaults follow best practice.
2. An object API that takes configured strategy and settings objects.
3. A generator or step-by-step API for custom loops.
4. Base classes for plug-ins, and an array-level layer for experts and other libraries.

The docs follow the same layers, e.g. "High-Level Interface" versus "Full Documentation", or tutorials versus how-to guides versus API reference. Advanced options sit in small composable objects rather than in ever-longer keyword lists.

### Cited Findings
- **emcee:** `run_mcmc(p0, nsteps, progress=True)` for the common case. The `sample()` generator lets users write their own convergence logic, and `backend=` adds persistence ([emcee sampler API](https://emcee.readthedocs.io/en/stable/user/sampler/); [emcee monitor tutorial](https://emcee.readthedocs.io/en/stable/tutorials/monitor/)).
- **dynesty:**
  - `run_nested()` for the common case, or the `sample()` generator followed by `add_live_points()` ([dynesty quickstart](https://dynesty.readthedocs.io/en/stable/quickstart.html)).
  - Since 3.0, advanced control comes from configured objects (`sample=RSliceSampler(slices=10)`, `bound=MyBound(4)`) and subclassing (`InternalSampler`, `bounding.Bound`), not `walks=`/`slices=`/`facc=` keywords ([dynesty #495](https://github.com/joshspeagle/dynesty/issues/495)).
- **nautilus:** `Sampler(prior, likelihood).run(verbose=True)` with defaults `n_live=2000`, `n_eff=10000` and `f_live=0.01`. The API docs are split into "High-Level Interface" and "Full Documentation" ([nautilus docs](https://nautilus-sampler.readthedocs.io/en/latest/); [nautilus high-level API](https://nautilus-sampler.readthedocs.io/en/latest/api_high.html)).
- **UltraNest:** `ReactiveNestedSampler(...).run()` with "Reasonable defaults, but customizable"; users "Can control the run programmatically and check status" ([UltraNest README](https://johannesbuchner.github.io/UltraNest/readme.html)).
- **sbi:**
  - High level: `NPE(); append_simulations(θ, x).train(); build_posterior()`.
  - Low level: an interface "giving full control over simulations, the training loop, and the sampling procedure" ([Boelts et al., arXiv:2411.17337](https://arxiv.org/abs/2411.17337); [sbi docs](https://sbi.readthedocs.io/en/latest/)).
  - A separate tutorial covers the training interface ([sbi training interface](https://sbi.readthedocs.io/en/latest/advanced_tutorials/18_training_interface.html)).
- **ArviZ 1.x:**
  - Three entry points: top-level functions (`az.summary`, `az.ess(dt)`); xarray accessors (`dt.azstats.ess()`); and a NumPy array interface "intended for advanced users and developers of third-party libraries" ([Martin et al. 2026](https://joss.theoj.org/papers/10.21105/joss.09889); [migration guide](https://python.arviz.org/en/latest/user_guide/migration_guide.html)).
  - Plots are layered the same way: "batteries-included" plots at the top, and lower-level plotting building blocks such as the `plot_collection` argument ([Martin et al. 2026](https://joss.theoj.org/papers/10.21105/joss.09889); [plot_ecdf_pit docs](https://arviz-plots.readthedocs.io/en/latest/api/generated/arviz_plots.plot_ecdf_pit.html)).
- **Bambi on PyMC:** `bmb.Model('y ~ x', df).fit()` is "built on top of the PyMC probabilistic programming framework and the ArviZ package" ([Capretto et al., arXiv:2012.10754](https://arxiv.org/abs/2012.10754); [Bambi docs](https://bambinos.github.io/bambi/)).
- **PyMC:** `pm.sample()` chooses the sampler automatically ("If None (default), 'nutpie' is used if installed"). Experts pass `nuts_sampler=`, `step=`, step-specific options such as `nuts={'target_accept': 0.9}`, or `compile_kwargs` ([pymc.sample](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sample.html)).
- **NumPyro:** `MCMC(NUTS(model), num_warmup=..., num_samples=...)` keeps the driver separate from the algorithm ([NumPyro getting started](https://num.pyro.ai/en/stable/getting_started.html)).
- **statsmodels:** `smf.ols("y ~ x", df).fit().summary()` through the formula layer, or `sm.OLS(y, X)` through the array layer ([statsmodels formulas](https://www.statsmodels.org/stable/example_formulas.html); [Getting started](https://www.statsmodels.org/stable/gettingstarted.html)).
- **lmfit:** `Model(func).fit(data, params, x=x)` with `fit_report()`, or the lower-level `minimize(residual, params)` ([lmfit Model](https://lmfit.github.io/lmfit-py/model.html); [lmfit intro](https://lmfit.github.io/lmfit-py/intro.html)).
- **pyhf:** one call, `pyhf.infer.hypotest(...)`, on top of a swappable array backend and optimizer ([pyhf docs](https://pyhf.readthedocs.io/en/stable/)).
- **corner:** `corner.corner(...)` is "the main user-facing function"; "the lower level functions corner.hist2d() and corner.quantile() are also documented" ([corner API](https://corner.readthedocs.io/en/latest/api/)).
- **astropy:** `fitter(model, x, y)`, where any fitter works with any model ([astropy.modeling](https://docs.astropy.org/en/stable/modeling/index.html)).
- **SciPy's new distributions:** `stats.Normal(mu=..., sigma=...)` for common cases, and `make_distribution` to wrap any old distribution ([SciPy random-variable guide](https://docs.scipy.org/doc/scipy/tutorial/stats/rv_infrastructure.html)).
- **scikit-learn:**
  - "Sensible defaults ... (giving a baseline solution for the task at hand)", with composition through Pipelines and meta-estimators ([Buitinck et al. 2013](https://arxiv.org/abs/1309.0238)).
  - Callbacks are opt-in objects: `set_callbacks(ProgressBar())` ([SLEP023](https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep023/proposal.html)).
- **Keras API guidelines** ([Keras API design guidelines](https://github.com/keras-team/governance/blob/master/keras_api_design_guidelines.md)):
  - "Simple use cases should be simple, advanced use cases should be possible: Don't increase the cognitive load of common use cases for the sake of niche use cases, even minimally ... It is ok for advanced use cases not to be directly supported in the built-in API options".
  - "Complex objects should be achievable by composing simple objects with few arguments, that do one thing reliably".
  - "Design end-to-end workflows, not individual functions and classes".
  - "Deliberately design the user onboarding process for your feature".
- **ChainConsumer** moved advanced options into typed settings objects (`ChainConfig`, `PlotConfig`) to avoid duplicating keyword arguments and docstrings ([ChainConsumer usage](https://samreay.github.io/ChainConsumer/usage/)).

### Inferences
A possible pastry layering, consistent with these exemplars (a design sketch, not sourced):
- **Layer 0: one call.**
  - `res = pastry.fit(L, bins=edges, rng=0)`, where `L` is the N×K likelihood matrix.
  - Other inputs: `pastry.fit(samples=..., interim_prior=...)`, or a list of per-object posterior-sample arrays.
  - Documented defaults: a Dirichlet prior with α = 1, automatic sampler choice, stopping once converged, and a progress bar.
  - Returns a `PopulationResult` with:
    - `.summary()`: bin fractions with credible intervals, R̂ and ESS, and the reason the run stopped;
    - `.plot()`: via corner or ArviZ;
    - `.to_datatree()`;
    - `.samples`: a NumPy array;
    - `.log_evidence`: where available.
- **Layer 1: estimator objects.**
  - Model classes: `DirichletMultinomial(prior=Dirichlet(alpha=1.0), sampler="nuts"|"gibbs"|"em", rng=None)`, plus `SmoothDirichlet(...)`, `GPDirichlet(...)`, `GeneralizedDirichlet(...)` and `FlowDensity(...)`.
  - scikit-learn style: `fit(L)` returns self; learned attributes `fractions_` and `posterior_`; `score(L)` returns the log marginal likelihood; `get_params`/`set_params`; `rng=` per SPEC 7.
  - Priors, selection functions and samplers are passed as configured objects (as in dynesty 3.0, nautilus's `Prior` and ChainConsumer's configs), not as keyword arguments.
- **Layer 2: run control.**
  - `sampler = model.sampler(L, backend=pastry.backends.HDF5("run.h5"))`, then `for state in sampler.sample(...)`.
  - `sampler.run(checkpoint="run.h5")` resumes automatically, as nautilus does, after checking that the file belongs to the same problem.
  - Callbacks follow SLEP023.
- **Layer 3: experts and other libraries.**
  - Plain functions `pastry.logdensity(rho, L)`, `pastry.grad_logdensity` and `pastry.em_step`, usable from NumPyro, BlackJAX or emcee (the tinygp lesson).
  - Base classes `Prior`, `SelectionFunction` and `Sampler` for plug-ins.
  - Kernels written so they can later run on JAX or PyTorch through the Array API.
- **Documentation mirrors the layers.**
  - A "Quickstart / High-level" page, task-oriented how-to guides, and a full API reference.
  - An example in every docstring (Keras).
  - A "Common pitfalls" page, in the spirit of corner's sigma note and scikit-learn's randomness pitfalls.

### Gaps
- No user studies were found that measure how much layered ("progressive disclosure") APIs help scientific Python users. The evidence is maintainers' design statements and the Keras guidelines.
- How Bambi exposes the underlying PyMC model for expert customisation (e.g. a `model.backend` attribute) was not checked in this pass.
