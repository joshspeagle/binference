# Open questions and parking lot

Carried over from the v0.1 summary (spec §13 open questions):

- [ ] Distribution name `pastry-inference` (import `pastry`)? (D5)
- [ ] Default prior Dirichlet(α = 1) or the reference prior α = 1/K? (D3)
- [ ] Roadmap priority: flows / continuous densities, JAX backend, hierarchical models? (spec §12)
- [ ] Python ≥ 3.11 floor? (D6) Is the `threadpoolctl` dependency acceptable? (D1, D14)
- [ ] When to make the repository public (JOSS needs ~6 months of public history)?

Known weaknesses worth discussing when we reach them:

- [ ] NUTS with Dirichlet priors and many nearly empty bins (α < 1, Pólya trees) is slow. (Phase 4)
- [ ] Laplace is usually flagged approximate for K ≳ 10. (Phase 4)
- [ ] Partial-centring rule: threshold 0.28 sits in a band where small problems (K ≈ 6–10) overlap; occasional divergences remain. (Phase 4.2)
- [ ] Binning bias: the population is assumed flat within bins. (Phase 2.2)
- [ ] HSGP basis size default `min(K, 48)` is slightly below the Riutort-Mayol guideline for the shortest length scales. (Phase 3.2)
- [ ] Single-chain runs: the curvature at one warm-up point is a noisy basis for adaptation. (Phase 4.2)

New items:
- [ ] The eight architecture decisions of briefing 1.3 §8 (Prior/BoundPrior split; public PopulationModel; kind set by prior; selection in two places; `infer` kwargs passthrough; results hold the model; flat layout and `samplers` name; closed method registry). Josh: revisit after the code tours of Phases 2–4, with the code in view. (1.3)
- [ ] `Bins.neighbors()` is unused by any prior (the random walk builds per-axis eigenbases instead). Keep as a utility, use it, or remove? (1.3 / 3.2)
- [ ] Pólya trees are 1-D only; N-D grids have no multiscale conjugate prior (a k-d / quadtree Dirichlet tree would be natural). (3.1)
- [x] **Basis-function generalisation of "bins"** → DECISIONS #5, #6. (from 2.1 discussion): the model only needs p(d_i|ρ) linear in ρ, so any fixed non-negative mixture basis φ_k (histogram = order-0 B-splines; linear B-splines; higher order) works with every engine unchanged: L_ik = ∫ p(d_i|z) φ_k(z) dz. Experiment (`experiments/2.1_hat_basis.py`, 6 mocks, N = 5e4, K = 10, exponential population, σ = 0.15–0.3): systematic bin-mass bias up to 14 s.e. with histograms, ≤ 3 s.e. with linear B-splines. Smooth priors could also evaluate their continuous field inside bins. Decide scope and API. (2.1 / Phase 7 / roadmap)
- [x] **Correlated N-D errors** → DECISIONS #2.: `from_gaussian` assumes independent per-axis errors (diagonal covariance). `from_function`, `from_samples` and raw matrices handle correlated errors; add full-covariance Gaussian support (rectangle probabilities via quadrature)? (2.3)
- [x] **Object weights** → DECISIONS #1.: Josh questions why non-integer weights are allowed at all. Options: integer multiplicities only in the public API (non-integer kept internal for the bootstrap), or an explicit opt-in for weighted pseudo-likelihoods. (2.1 §2)
- [x] **Autodiff** → DECISIONS #3.: gradients are hand-derived NumPy (checked by finite-difference tests). Revisit with the JAX-backend roadmap item; cost: every new prior needs hand-written gradients. (Phase 9)
- [x] **Model resolution vs reporting bins** → DECISIONS #4. (Josh: why not evaluate the continuous field inside bins?). Nothing prevents it for priors with a continuous field. Equivalent formulation: run the likelihood on a finer internal grid (Q sub-cells, or quadrature nodes) with p(d_i | z_q), put the smooth prior on that grid, and report bin fractions as sums (a fixed linear map, invisible to users). For the HSGP the parameter count is n_basis, independent of Q, so refinement costs only likelihood passes (×m per axis in memory/time). Limits: needs pointwise likelihoods (available for Gaussian/function/distribution inputs; not for binned posteriors, response matrices or classifiers, which carry no within-bin information); random-walk priors gain one parameter per sub-cell; Dirichlet priors have no within-bin shape (B-spline mixtures are their analogue). Josh is positive on B-splines if the mappings stay invisible. Decide the general design (reporting bins + internal resolution + basis) in Phase 3/7. (2.1)
