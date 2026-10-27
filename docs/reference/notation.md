# Notation

Each symbol has one meaning across the documentation and the docstrings. The tables
give the symbol, its meaning, the name the code uses, and the page that defines it.
Indices are $i$ for objects, $k$ for bins, $j$ for measured bins and $s$ for draws;
other indices are declared where they are used. The {doc}`glossary` defines the terms.

## Objects, bins and likelihoods

| Symbol | Meaning | Code name | Where defined |
|---|---|---|---|
| $N$ | number of objects | `lk.N`, `model.N` | [model](../how-it-works/model.md#objects-bins-and-likelihoods) |
| $d_i$ | data of object $i$ | the inputs of `Likelihoods.from_*` | [model](../how-it-works/model.md#objects-bins-and-likelihoods) |
| $D$ | the data of all objects | | [model](../how-it-works/model.md#the-population-model) |
| $z$ | true value of an object | `values` in `result.sample_values()` | [model](../how-it-works/model.md#objects-bins-and-likelihoods) |
| $K$ | number of bins | `bins.K` | [model](../how-it-works/model.md#objects-bins-and-likelihoods) |
| $B_k$ | bin $k$ | `bins.edges`, `bins.labels` | [model](../how-it-works/model.md#objects-bins-and-likelihoods) |
| $v_k$ | volume of bin $k$ | `bins.volumes` | [model](../how-it-works/model.md#objects-bins-and-likelihoods) |
| $u=T(z)$ | measure coordinate, in which the population is flat within a bin | `Measure`, `bins.u_edges` | [inputs](../guides/inputs/advanced.md#measures-flat-in-which-coordinate) |
| $\ell_i(z)$ | likelihood function of object $i$, $p(d_i\mid z)$ | `density=` of `from_density` | [model](../how-it-works/model.md#objects-bins-and-likelihoods) |
| $L_{ik}$ | bin-averaged likelihood | `Likelihoods` | [model](../how-it-works/model.md#objects-bins-and-likelihoods) |
| $\bar L_{ik}$ | $L_{ik}$ normalised to a row maximum of one | `lk.values` | [inputs](../guides/inputs/tables.md#conventions) |
| $w_i$ | weight (multiplicity) of object $i$ | `lk.weights` | [model](../how-it-works/model.md#objects-bins-and-likelihoods) |
| $W$ | total weight, $\sum_iw_i$ | `lk.W` | [model](../how-it-works/model.md#objects-bins-and-likelihoods) |
| $\pi(z)$ | interim prior | `interim=` | [model](../how-it-works/model.md#likelihoods-versus-posteriors) |
| $P_{ik}$ | mass of object $i$'s posterior in bin $k$ | the table of `from_table(..., interim=)` | [model](../how-it-works/model.md#likelihoods-versus-posteriors) |
| $\Pi_k$ | mass of the interim prior in bin $k$ | `Density.masses` | [model](../how-it-works/model.md#likelihoods-versus-posteriors) |
| $\omega_s$ | importance weight of sample or draw $s$ | `sample_weights=` | [inputs](../guides/inputs/samples.md) |
| $n_{\rm draw}$ | number of draws or importance weights | `n_draws`, `n_proposals` | [engines](../how-it-works/engines.md#laplace-with-importance-sampling) |

## The population

| Symbol | Meaning | Code name | Where defined |
|---|---|---|---|
| $\rho_k$ | population fraction of bin $k$ | `result.samples`, `result.mean`, `est.rho` | [model](../how-it-works/model.md#the-population-model) |
| $\Delta^{K-1}$ | the simplex: $K$ non-negative numbers that sum to one | | [model](../how-it-works/model.md#the-population-model) |
| $p(z\mid\rho)$ | population density: flat in the measure coordinate within each bin; its average over bin $k$ is $\rho_k/v_k$ | `result.density()` (the bin averages) | [model](../how-it-works/model.md#the-population-model) |
| $\lambda_k$ | rate: expected number of objects in bin $k$ | `result.rates`, `GammaRates` | [model](../how-it-works/model.md#selection-effects) |
| $\Lambda$ | total expected number of objects | `result.total` | [model](../how-it-works/model.md#selection-effects) |
| $k_i$ | bin (label) of object $i$ | `result.sample_assignments()` | [model](../how-it-works/model.md#the-augmented-form) |
| $r_{ik}(\rho)$ | responsibility: object $i$'s posterior over bins given $\rho$ | `model.responsibilities` | [model](../how-it-works/model.md#the-augmented-form) |
| $P(k_i=k\mid D)$ | hierarchical posterior of object $i$ | `result.assignment_probabilities()` | [model](../how-it-works/model.md#individual-objects) |
| $n_k$ | realised histogram: observed objects in bin $k$ | `result.realized_counts()` | [model](../how-it-works/model.md#population-versus-realised-sample) |
| $\hat n_k(\rho)$ | expected counts, $\sum_iw_ir_{ik}(\rho)$ | `model.expected_counts` | [model](../how-it-works/model.md#the-population-model) |
| $g_k$ | KKT ratio $W^{-1}\sum_iw_iL_{ik}/(L_i\cdot\hat\rho)$; with a flat prior and no selection the KKT gap is $\max_kg_k-1$ | `kkt_gap` | [point estimates](../how-it-works/point-estimates.md#the-certificate) |
| $f$ | fraction of missing information | `model.missing_information` | [point estimates](../how-it-works/point-estimates.md#missing-information) |
| $I_{\rm obs}$, $I_c$ | observed and complete-data information | | [point estimates](../how-it-works/point-estimates.md#missing-information) |

## Selection and responses

| Symbol | Meaning | Code name | Where defined |
|---|---|---|---|
| $S_k$ | selection: probability that an object in bin $k$ is observed, or its exposure | `Selection.values` | [model](../how-it-works/model.md#selection-effects) |
| $\tilde L_{ik}$ | detected-object likelihood: $S_kL_{ik}$ on the truth, $L_{ik}$ on the measurement | | [model](../how-it-works/model.md#selection-effects) |
| $S^{\rm truth}(z)$, $S^{\rm meas}(d)$ | the two parts of a combined selection | `Selection.combined(on_truth=, on_measurement=)` | [selection](../guides/selection.md#three-ways-selection-can-act) |
| $\phi_k$ | selected population: $S_k\rho_k/S\cdot\rho$ | `result.selected` | [point estimates](../how-it-works/point-estimates.md#mix-sqp) |
| $\chi_k$ | composition of the rates in mix-SQP | | [point estimates](../how-it-works/point-estimates.md#mix-sqp) |
| $M$ | number of objects the selection missed | | [model](../how-it-works/model.md#selection-effects) |
| $m_k$ | missed objects in bin $k$ | `realized_counts(full=True)` | [model](../how-it-works/model.md#selection-effects) |
| $n^{\rm inj}_k$, $n^{\rm rec}_k$ | injected and recovered objects in bin $k$ | `selection.injections` | [selection](../guides/selection.md#selection-measured-by-injections) |
| $\tau_S$ | step scale of the random walk on logit $S$ | `smooth=` of `from_injections` | [selection](../guides/selection.md#selection-measured-by-injections) |
| $J$ | number of measured bins | `response.measured_bins.K` | [inputs](../guides/inputs/binned.md) |
| $R_{jk}$ | response: probability of measured bin $j$ and observation, given true bin $k$ | `response.matrix` | [inputs](../guides/inputs/binned.md) |
| $\epsilon_k$ | efficiency: probability that an object in true bin $k$ is observed | `response.efficiency` | [inputs](../guides/inputs/binned.md) |

## Priors

| Symbol | Meaning | Code name | Where defined |
|---|---|---|---|
| $\alpha_k$ | Dirichlet concentration, also of a tallied response's columns | `Dirichlet(alpha)`, `prior=` of `Response.from_counts` | [priors](../how-it-works/priors.md#the-dirichlet-family-as-trees) |
| $n^{\rm ref}_k$ | counts of a reference sample | `counts` of `Dirichlet.from_counts` | [priors](../how-it-works/priors.md#the-dirichlet-family-as-trees) |
| $\kappa$ | weight of the reference counts in $\alpha_k$ | `strength=` | [priors](../how-it-works/priors.md#the-dirichlet-family-as-trees) |
| $G_0$ | base measure of a prior | `base=` | [priors](../how-it-works/priors.md#the-dirichlet-family-as-trees) |
| $\nu_k$ | stick-breaking proportions of a generalised Dirichlet | | [priors](../how-it-works/priors.md#the-dirichlet-family-as-trees) |
| $c_{\rm PT}$, $p_{\rm PT}$, $h$ | Pólya-tree concentration, its growth with depth, and the depth | `PolyaTree(c, power)` | [priors](../how-it-works/priors.md#the-dirichlet-family-as-trees) |
| $a_k$, $b_k$ | Gamma shape and rate of the rate $\lambda_k$ | `GammaRates(shape, rate)` | [model](../how-it-works/model.md#selection-effects) |
| $\eta$ | log-fractions, $\rho=\mathrm{softmax}(\eta)$ | | [priors](../how-it-works/priors.md#the-logistic-family) |
| $\mu$ | prior mean of $\eta$ | `mean=` | [priors](../how-it-works/priors.md#the-logistic-family) |
| $E$ | basis of a logistic prior, one column per mode | | [priors](../how-it-works/priors.md#the-logistic-family) |
| $\theta$ | hyperparameters: amplitude, scale, length scale | `amplitude=`, `scale=`, `length_scale=` | [priors](../how-it-works/priors.md#the-logistic-family) |
| $s_j(\theta)$ | prior standard deviation of mode $j$ | | [priors](../how-it-works/priors.md#the-logistic-family) |
| $\xi_j$ | whitened coefficient of mode $j$ | | [priors](../how-it-works/priors.md#the-logistic-family) |
| $c_j$ | coefficient of mode $j$, $s_j\xi_j$ | | [priors](../how-it-works/priors.md#the-logistic-family) |
| $\zeta_j$ | partially centred coefficient of mode $j$ | | [engines](../how-it-works/engines.md#partial-centring) |
| $\gamma_j$ | partial-centring weight of mode $j$ | `info["centring"]` | [engines](../how-it-works/engines.md#partial-centring) |
| $\tau$ | standard deviation of the random walk's differences | `scale=` of `LogisticRandomWalk` | [priors](../how-it-works/priors.md#the-logistic-family) |
| $r$ | order of the random walk's differences | `order=` | [priors](../how-it-works/priors.md#the-logistic-family) |
| $e_j$ | eigenvalue $j$ of the random walk's difference precision | | [priors](../how-it-works/priors.md#the-logistic-family) |
| $\Delta$ | half the range of a grid axis, in the measure coordinate | | [priors](../how-it-works/priors.md#the-logistic-family) |
| $c_{\rm box}$ | size of the Gaussian-process box relative to $\Delta$ | `boundary_factor=` | [priors](../how-it-works/priors.md#the-logistic-family) |
| $n_{\rm basis}$ | number of sines per axis | `n_basis=` | [priors](../how-it-works/priors.md#the-logistic-family) |
| $f_{\rm ker}$ | resolution factor of a kernel | | [defaults](defaults.md) |
| $\psi_k(z)$ | component $k$ of a `Basis` | `Basis` | [inputs](../guides/inputs/advanced.md#shape-within-bins-basis) |
| $\beta_k$ | fraction of the population in component $k$ | `result.component_draws` | [inputs](../guides/inputs/advanced.md#shape-within-bins-basis) |
| $A_{jk}$ | mass of component $k$ in bin or cell $j$ | `basis.matrix` | [inputs](../guides/inputs/advanced.md#shape-within-bins-basis) |

## Coordinates, engines and diagnostics

| Symbol | Meaning | Code name | Where defined |
|---|---|---|---|
| $x$ | unconstrained coordinates | `result.x` | [priors](../how-it-works/priors.md#coordinates) |
| $Q$ | orthonormal Helmert basis of the isometric log-ratio | | [priors](../how-it-works/priors.md#coordinates) |
| $H$ | Hessian of the log posterior | | [engines](../how-it-works/engines.md#nuts) |
| $\hat R$ | split R-hat | `diagnostics.rhat`, `result.diagnostics["rhat"]` | [validation](../how-it-works/validation.md#diagnostics-of-one-run) |
| ESS | effective sample size, bulk or tail | `diagnostics.ess`, `result.diagnostics["ess_bulk"]` | [validation](../how-it-works/validation.md#diagnostics-of-one-run) |
| $\hat k$ | Pareto shape of importance weights | `diagnostics.psis`, `result.info["k_hat"]` | [engines](../how-it-works/engines.md#laplace-with-importance-sampling) |
| $\tau_{\rm int}$ | integrated autocorrelation time of a chain | `diagnostics.autocorr_time` | [engines](../how-it-works/engines.md#mixing-and-missing-information) |
| $V_i(\rho)$ | relative Monte Carlo variance of a likelihood from samples | `lk.sample_variance` | [inputs](../guides/inputs/samples.md) |
| $n_{\rm sim}$ | number of simulated data sets | `n_simulations=` | [validation](../how-it-works/validation.md#simulation-based-calibration) |

## Symbols on older pages

Some guides, tutorials and docstrings were written before this table and still use a
symbol with a second meaning. Read them with this mapping; the section rewrites of the
documentation adopt the symbols above and remove it.

| Older symbol | Meaning there | Symbol here |
|---|---|---|
| $S$ | number of importance proposals; number of simulations in SBC | $n_{\rm draw}$; $n_{\rm sim}$ |
| $S$ | half the range of a grid axis | $\Delta$ |
| $\lambda$ | strength of a reference sample | $\kappa$ |
| $\lambda_j$ | partial-centring weight | $\gamma_j$ |
| $\ell$ | log-likelihood of the catalogue | $\log p(D\mid\rho)$ |
| $\ell_{ik}$ | likelihood normalised over the bins | $L_{ik}/\sum_kL_{ik}$ |
| $w_k$ | weight of a `Basis` component | $\beta_k$ |
| $w_s$ | importance weight of a sample | $\omega_s$ |
| $\phi_k(z)$ | component of a `Basis` | $\psi_k(z)$ |
| $f(z)$ | population density | $p(z\mid\rho)$ |
| $\psi$ | composition of the rates in mix-SQP | $\chi$ |
| $B$ | basis of a logistic prior | $E$ |
| $c$, $m$, $p$ | Pólya-tree concentration, depth and growth | $c_{\rm PT}$, $h$, $p_{\rm PT}$ |
| $f$ | resolution factor of a kernel | $f_{\rm ker}$ |
| $m$ | reference counts | $n^{\rm ref}_k$ |
| $M$ | draws per SBC fit; component masses on a grid | $n_{\rm draw}$; $A$ |
| $n_k$, $d_k$ | injected and recovered objects | $n^{\rm inj}_k$, $n^{\rm rec}_k$ |
| $z_i$ | bin of object $i$ | $k_i$ |
| $x_i$ | data of object $i$ | $d_i$ |
| $s_i$ | $L_i\cdot\rho$ | $L_i\cdot\rho$ |
| $V$ | Helmert basis | $Q$ |
| $V_k$ | stick-breaking proportion | $\nu_k$ |
| $C(z)$, $T(d)$ | parts of a combined selection | $S^{\rm truth}(z)$, $S^{\rm meas}(d)$ |
| $\tau$ | step scale of a smooth injection selection | $\tau_S$ |
| $\beta$ | concentration of a tallied response | $\alpha$ |
| $G$ | gradient of the log posterior in $\lambda$ | $\partial\log p/\partial\lambda$ |

The field pages quote each field's own formulas. On the gravitational-wave page,
$\Lambda$ denotes hyperparameters, $\theta$ an event's parameters, $S_i$ the number of
samples of event $i$, $R$ the merger rate and $N_\text{draw}$ the number of injections.
On the empirical-Bayes page, $w$ and $g$ denote $\rho$.
