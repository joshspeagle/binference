# Decisions made during the walkthrough

Outcomes: **keep** (reviewed and agreed), **change now** (done in the step;
cite the commit), **change later** (collected for Phase 7 or the roadmap),
**revisit** (needs more evidence first). Earlier autonomous decisions D1–D16
live in `docs/design/spec.md` §13; when one is reviewed, record the outcome
here with its D-number.

| # | Date | Step | Topic | Outcome | Rationale | Commit / follow-up |
|---|---|---|---|---|---|---|
| 1 | 2026-10-22 | 2.1 | Object weights | change later | Keep integer multiplicities (compress, unfolding counts) as the public meaning of weights. Bootstrap reweighting stays internal (`MAP.run(weights=)`, used only by `Bootstrap`; Bayesian bootstrap is the only real-valued use). Non-integer user weights only via an explicit opt-in with a clear warning that the result is a pseudo-likelihood, not a proper posterior. | Phase 7 (API); interacts with audit M-1 (zero-weight rows) |
| 2 | 2026-10-22 | 2.1 | Correlated Gaussian errors | change later | Add a correlated-Gaussian constructor for N-D grids with a per-object covariance (rectangle probabilities by quadrature or by a Cholesky-rotated sub-grid). | Step 2.3 |
| 3 | 2026-10-22 | 2.1 | Gradients | keep (for now) | Hand-derived NumPy gradients with finite-difference tests are fine for v0.x; JAX or PyTorch backend stays on the roadmap. | Phase 9 |
| 4 | 2026-10-22 | 2.1 | Model resolution vs reporting bins | change later | Separate the bins users report in from the model's internal resolution. Priors with a continuous field (GP) evaluate it on a finer internal grid of sub-cells or quadrature nodes, with pointwise likelihoods p(d_i \| z_q); bin fractions are sums over sub-points (a fixed linear map, invisible to users). Inputs without within-bin information (binned posteriors, response matrices, classifiers) keep resolution = bins. | Design in Phase 3 / 7 |
| 5 | 2026-10-22 | 2.1 | B-spline mixture basis | change later | Offer non-negative B-spline mixture components (linear first) as an alternative to flat-in-bin histograms for the Dirichlet family: L_ik = ∫ p(d_i\|z) φ_k(z) dz, engines unchanged; all user-facing outputs mapped back to bin space. Experiment: bias ≤ 3 s.e. vs up to 14 s.e. for histograms. | Design with #4; experiment `experiments/2.1_hat_basis.py` |
| 6 | 2026-10-22 | 2.1 | Binning-bias check | change later | Provide a diagnostic that refits on refined bins (or compares histogram vs spline basis) and flags results whose bin masses move by more than their posterior s.d. | Phase 5 (diagnostics) |
