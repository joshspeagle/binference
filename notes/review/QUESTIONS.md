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
