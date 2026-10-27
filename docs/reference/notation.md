# Notation

Each symbol has one meaning across the documentation and the docstrings. The tables
give the symbol, its meaning, the name the code uses, and the page that defines it.
Indices are $i$ for objects, $k$ for bins, $j$ for measured bins and $s$ for draws;
other indices are declared where they are used. The {doc}`glossary` defines the terms.

## Objects, bins and likelihoods

| Symbol | Meaning | Code name | Where defined |
|---|---|---|---|
| $N$ | number of objects | `lk.N`, `model.N` | [theory](../how-it-works/theory.md#objects-bins-and-likelihoods) |
| $d_i$ | data of object $i$ | the inputs of `Likelihoods.from_*` | [theory](../how-it-works/theory.md#objects-bins-and-likelihoods) |
| $D$ | the data of all objects | | [theory](../how-it-works/theory.md#the-population-model) |
| $z$ | true value of an object | `values` in `result.sample_values()` | [theory](../how-it-works/theory.md#objects-bins-and-likelihoods) |
| $K$ | number of bins | `bins.K` | [theory](../how-it-works/theory.md#objects-bins-and-likelihoods) |
| $B_k$ | bin $k$ | `bins.edges`, `bins.labels` | [theory](../how-it-works/theory.md#objects-bins-and-likelihoods) |
| $v_k$ | volume of bin $k$ | `bins.volumes` | [theory](../how-it-works/theory.md#objects-bins-and-likelihoods) |
| $u=T(z)$ | measure coordinate, in which the population is flat within a bin | `Measure`, `bins.u_edges` | [inputs](../guides/inputs.md#measures-flat-in-which-coordinate) |
| $\ell_i(z)$ | likelihood function of object $i$, $p(d_i\mid z)$ | `density=` of `from_density` | [theory](../how-it-works/theory.md#objects-bins-and-likelihoods) |
| $L_{ik}$ | bin-averaged likelihood | `Likelihoods` | [theory](../how-it-works/theory.md#objects-bins-and-likelihoods) |
| $\bar L_{ik}$ | $L_{ik}$ normalised to a row maximum of one | `lk.values` | [inputs](../guides/inputs.md#likelihood-matrices) |
| $w_i$ | weight (multiplicity) of object $i$ | `lk.weights` | [theory](../how-it-works/theory.md#objects-bins-and-likelihoods) |
| $W$ | total weight, $\sum_iw_i$ | `lk.W` | [theory](../how-it-works/theory.md#objects-bins-and-likelihoods) |
| $\pi(z)$ | interim prior | `interim=` | [theory](../how-it-works/theory.md#likelihoods-versus-posteriors) |
| $P_{ik}$ | mass of object $i$'s posterior in bin $k$ | the table of `from_table(..., interim=)` | [theory](../how-it-works/theory.md#likelihoods-versus-posteriors) |
| $\Pi_k$ | mass of the interim prior in bin $k$ | `Density.masses` | [theory](../how-it-works/theory.md#likelihoods-versus-posteriors) |
| $\omega_s$ | importance weight of sample or draw $s$ | `sample_weights=` | [inputs](../guides/inputs.md#posterior-samples) |
| $n_{\rm draw}$ | number of draws or importance weights | `n_draws`, `n_proposals` | [engines](../guides/engines.md#approximations) |

## The population

| Symbol | Meaning | Code name | Where defined |
|---|---|---|---|
| $\rho_k$ | population fraction of bin $k$ | `result.samples`, `result.mean`, `est.rho` | [theory](../how-it-works/theory.md#the-population-model) |
| $\Delta^{K-1}$ | the simplex: $K$ non-negative numbers that sum to one | | [theory](../how-it-works/theory.md#the-population-model) |
| $p(z\mid\rho)$ | population density, $\rho_k/v_k$ on bin $k$ | `result.density()` | [theory](../how-it-works/theory.md#the-population-model) |
| $\lambda_k$ | rate: expected number of objects in bin $k$ | `result.rates`, `GammaRates` | [theory](../how-it-works/theory.md#selection-effects) |
| $\Lambda$ | total expected number of objects | `result.total` | [theory](../how-it-works/theory.md#selection-effects) |
| $k_i$ | bin (label) of object $i$ | `result.sample_assignments()` | [theory](../how-it-works/theory.md#the-augmented-hierarchical-form) |
| $r_{ik}(\rho)$ | responsibility: object $i$'s posterior over bins given $\rho$ | `model.responsibilities` | [theory](../how-it-works/theory.md#the-augmented-hierarchical-form) |
| $P(k_i=k\mid D)$ | hierarchical posterior of object $i$ | `result.assignment_probabilities()` | [theory](../how-it-works/theory.md#individual-objects) |
| $n_k$ | realised histogram: observed objects in bin $k$ | `result.realized_counts()` | [theory](../how-it-works/theory.md#population-versus-realised-sample) |
| $\hat n_k(\rho)$ | expected counts, $\sum_iw_ir_{ik}(\rho)$ | `model.expected_counts` | [theory](../how-it-works/theory.md#the-population-model) |
| $g_k$ | KKT ratio; the gap is $\max_kg_k-1$ | `kkt_gap` | [theory](../how-it-works/theory.md#point-estimates-em-is-iterated-stacking) |
| $f$ | fraction of missing information | `model.missing_information` | [theory](../how-it-works/theory.md#point-estimates-em-is-iterated-stacking) |
| $I_{\rm obs}$, $I_c$ | observed and complete-data information | | [theory](../how-it-works/theory.md#point-estimates-em-is-iterated-stacking) |

## Selection and responses

| Symbol | Meaning | Code name | Where defined |
|---|---|---|---|
| $S_k$ | selection: probability that an object in bin $k$ is observed, or its exposure | `Selection.values` | [theory](../how-it-works/theory.md#selection-effects) |
| $S^{\rm truth}(z)$, $S^{\rm meas}(d)$ | the two parts of a combined selection | `Selection.combined(on_truth=, on_measurement=)` | [selection](../guides/selection.md#three-ways-selection-can-act) |
| $\phi_k$ | selected population: $S_k\rho_k/S\cdot\rho$ | `result.selected` | [theory](../how-it-works/theory.md#point-estimates-em-is-iterated-stacking) |
| $\chi_k$ | composition of the rates in mix-SQP | | [theory](../how-it-works/theory.md#point-estimates-em-is-iterated-stacking) |
| $M$ | number of objects the selection missed | | [theory](../how-it-works/theory.md#selection-effects) |
| $m_k$ | missed objects in bin $k$ | `realized_counts(full=True)` | [theory](../how-it-works/theory.md#selection-effects) |
| $n^{\rm inj}_k$, $n^{\rm rec}_k$ | injected and recovered objects in bin $k$ | `selection.injections` | [selection](../guides/selection.md#selection-measured-by-injections) |
| $\tau_S$ | step scale of the random walk on logit $S$ | `smooth=` of `from_injections` | [selection](../guides/selection.md#selection-measured-by-injections) |
| $J$ | number of measured bins | `response.measured_bins.K` | [inputs](../guides/inputs.md#binned-measurements-and-classifiers) |
| $R_{jk}$ | response: probability of measured bin $j$ and observation, given true bin $k$ | `response.matrix` | [inputs](../guides/inputs.md#binned-measurements-and-classifiers) |
| $\epsilon_k$ | efficiency: probability that an object in true bin $k$ is observed | `response.efficiency` | [inputs](../guides/inputs.md#binned-measurements-and-classifiers) |

## Priors

| Symbol | Meaning | Code name | Where defined |
|---|---|---|---|
| $\alpha_k$ | Dirichlet concentration, also of a tallied response's columns | `Dirichlet(alpha)`, `prior=` of `Response.from_counts` | [theory](../how-it-works/theory.md#the-dirichlet-family) |
| $n^{\rm ref}_k$ | counts of a reference sample | `counts` of `Dirichlet.from_counts` | [theory](../how-it-works/theory.md#the-dirichlet-family) |
| $\kappa$ | weight of the reference counts in $\alpha_k$ | `strength=` | [theory](../how-it-works/theory.md#the-dirichlet-family) |
| $G_0$ | base measure of a prior | `base=` | [theory](../how-it-works/theory.md#the-dirichlet-family) |
| $\nu_k$ | stick-breaking proportions of a generalised Dirichlet | | {class}`~binference.priors.GeneralizedDirichlet` |
| $c_{\rm PT}$, $p_{\rm PT}$, $h$ | Pólya-tree concentration, its growth with depth, and the depth | `PolyaTree(c, power)` | [theory](../how-it-works/theory.md#the-dirichlet-family) |
| $a_k$, $b_k$ | Gamma shape and rate of the rate $\lambda_k$ | `GammaRates(shape, rate)` | [theory](../how-it-works/theory.md#selection-effects) |
| $\eta$ | log-fractions, $\rho=\mathrm{softmax}(\eta)$ | | [theory](../how-it-works/theory.md#smoothness-the-logistic-family) |
| $\mu$ | prior mean of $\eta$ | `mean=` | [theory](../how-it-works/theory.md#smoothness-the-logistic-family) |
| $E$ | basis of a logistic prior, one column per mode | | [theory](../how-it-works/theory.md#smoothness-the-logistic-family) |
| $\theta$ | hyperparameters: amplitude, scale, length scale | `amplitude=`, `scale=`, `length_scale=` | [theory](../how-it-works/theory.md#smoothness-the-logistic-family) |
| $s_j(\theta)$ | prior standard deviation of mode $j$ | | [theory](../how-it-works/theory.md#smoothness-the-logistic-family) |
| $\xi_j$ | whitened coefficient of mode $j$ | | [theory](../how-it-works/theory.md#smoothness-the-logistic-family) |
| $c_j$ | coefficient of mode $j$, $s_j\xi_j$ | | [theory](../how-it-works/theory.md#smoothness-the-logistic-family) |
| $\zeta_j$ | partially centred coefficient of mode $j$ | | [theory](../how-it-works/theory.md#smoothness-the-logistic-family) |
| $\gamma_j$ | partial-centring weight of mode $j$ | `info["centring"]` | [theory](../how-it-works/theory.md#smoothness-the-logistic-family) |
| $\tau$ | standard deviation of the random walk's differences | `scale=` of `LogisticRandomWalk` | [theory](../how-it-works/theory.md#smoothness-the-logistic-family) |
| $r$ | order of the random walk's differences | `order=` | [theory](../how-it-works/theory.md#smoothness-the-logistic-family) |
| $e_j$ | eigenvalue $j$ of the random walk's difference precision | | [theory](../how-it-works/theory.md#smoothness-the-logistic-family) |
| $\Delta$ | half the range of a grid axis, in the measure coordinate | | [theory](../how-it-works/theory.md#smoothness-the-logistic-family) |
| $c_{\rm box}$ | size of the Gaussian-process box relative to $\Delta$ | `boundary_factor=` | [theory](../how-it-works/theory.md#smoothness-the-logistic-family) |
| $n_{\rm basis}$ | number of sines per axis | `n_basis=` | [theory](../how-it-works/theory.md#smoothness-the-logistic-family) |
| $f_{\rm ker}$ | resolution factor of a kernel | | [defaults](defaults.md) |
| $\psi_k(z)$ | component $k$ of a `Basis` | `Basis` | [inputs](../guides/inputs.md#within-bin-shape-and-internal-resolution-basis) |
| $\beta_k$ | fraction of the population in component $k$ | `result.component_draws` | [inputs](../guides/inputs.md#within-bin-shape-and-internal-resolution-basis) |
| $A_{jk}$ | mass of component $k$ in bin or cell $j$ | `basis.matrix` | [inputs](../guides/inputs.md#within-bin-shape-and-internal-resolution-basis) |

## Coordinates, engines and diagnostics

| Symbol | Meaning | Code name | Where defined |
|---|---|---|---|
| $x$ | unconstrained coordinates | `result.x` | [theory](../how-it-works/theory.md#unconstrained-coordinates) |
| $Q$ | orthonormal Helmert basis of the isometric log-ratio | | [theory](../how-it-works/theory.md#unconstrained-coordinates) |
| $H$ | Hessian of the log posterior | | [theory](../how-it-works/theory.md#the-engines) |
| $\hat R$ | split R-hat | `diagnostics.rhat`, `result.diagnostics["rhat"]` | [theory](../how-it-works/theory.md#diagnostics-and-validation) |
| ESS | effective sample size, bulk or tail | `diagnostics.ess`, `result.diagnostics["ess_bulk"]` | [theory](../how-it-works/theory.md#diagnostics-and-validation) |
| $\hat k$ | Pareto shape of importance weights | `diagnostics.psis`, `result.info["k_hat"]` | [theory](../how-it-works/theory.md#the-engines) |
| $V_i(\rho)$ | relative Monte Carlo variance of a likelihood from samples | `lk.sample_variance` | [inputs](../guides/inputs.md#posterior-samples) |
| $n_{\rm sim}$ | number of simulated data sets | `n_simulations=` | [validation](../guides/validation.md#simulation-based-calibration) |

## Symbols on older pages

Some pages written before this table use a symbol with a second meaning. Read them with
this mapping until they are rewritten.

| Older symbol | Meaning there | Symbol here |
|---|---|---|
| $S$ | number of importance proposals | $n_{\rm draw}$ |
| $S$ | half the range of a grid axis | $\Delta$ |
| $S$ | number of simulations in SBC | $n_{\rm sim}$ |
| $\lambda$ | strength of a reference sample | $\kappa$ |
| $\lambda_j$ | eigenvalue of the random walk's precision | $e_j$ |
| $\lambda_j$ | partial-centring weight | $\gamma_j$ |
| $\lambda_{\min}$ | smallest generalised eigenvalue of $(I_{\rm obs},I_c)$ | $\min\operatorname{eig}$ |
| $\ell(\rho)$ | log-likelihood of the catalogue | $\log p(D\mid\rho)$ |
| $\ell_{ik}$ | likelihood normalised over the bins | $L_{ik}/\sum_kL_{ik}$ |
| $w_k$ | weight of a `Basis` component | $\beta_k$ |
| $w_s$ | importance weight of a sample | $\omega_s$ |
| $\phi_k(z)$ | component of a `Basis` | $\psi_k(z)$ |
| $\phi_j(x)$ | sine of the Gaussian-process basis | column $j$ of $E$ |
| $\psi_k$ | composition of the rates in mix-SQP | $\chi_k$ |
| $B$ | basis of a logistic prior | $E$ |
| $L$ | half-width of the Gaussian-process box | $c_{\rm box}\Delta$ |
| $c$ | box factor; Pólya-tree concentration | $c_{\rm box}$; $c_{\rm PT}$ |
| $m$ | reference counts; number of sines; Pólya-tree depth | $n^{\rm ref}_k$; $n_{\rm basis}$; $h$ |
| $M$ | draws per SBC fit; component masses on a grid | $n_{\rm draw}$; $A$ |
| $f$ | resolution factor of a kernel | $f_{\rm ker}$ |
| $n(z\mid\rho)$, $f(z)$ | population density | $p(z\mid\rho)$ |
| $n_k$, $d_k$ | injected and recovered objects | $n^{\rm inj}_k$, $n^{\rm rec}_k$ |
| $z_i$ | bin of object $i$ | $k_i$ |
| $x_i$ | data of object $i$ | $d_i$ |
| $s_i$ | $L_i\cdot\rho$ | $L_i\cdot\rho$ |
| $V$ | Helmert basis | $Q$ |
| $V_k$ | stick-breaking proportion | $\nu_k$ |
| $C(z)$, $T(d)$ | parts of a combined selection | $S^{\rm truth}(z)$, $S^{\rm meas}(d)$ |
| $\tau$ | step scale of a smooth injection selection | $\tau_S$ |
| $\beta$ | concentration of a tallied response | $\alpha$ |
| $p$ | growth of the Pólya-tree concentration | $p_{\rm PT}$ |
| $G$ | gradient of the log posterior in $\lambda$ | $\partial\log p/\partial\lambda$ |

The field pages quote each field's own formulas. On the gravitational-wave page,
$\Lambda$ denotes hyperparameters $\theta$, and $S_i$ the number of samples of event $i$.
On the empirical-Bayes page, $w$ and $g$ denote $\rho$.
