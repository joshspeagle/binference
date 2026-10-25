# Post-v1 open issues

Work deliberately deferred until the full package (v1) is built. Each becomes a
GitHub issue when the repository goes public. v1 assumes its modelling
assumptions hold (calibrated per-object likelihoods, known interim prior,
independent objects from one population) and documents them.

| Item | Why deferred | Needs | Source |
|---|---|---|---|
| Calibration of per-object PDFs (error scaling, likelihood tempering, generalized Bayes) | Not identifiable without ground truth: in 1-D a wrong σ is absorbed by a compensating ρ, so predictive checks of the data cannot detect it | truth labels or labels with known noise | 1.2 §5.2, DECISIONS #38 |
| Implicit interim prior estimated from held-out posteriors | Needs held-out data supplied with the PDFs | held-out set; `Density` constructor (#26) | 1.2 §5.3, #38 |
| Recalibration helpers for classifiers | Same as calibration | labelled validation data | #29 |
| Overdispersion / sample variance (per-field layers, 3sDir-like, simulation covariance) | Proposed post-v1 (awaiting Josh): one field gives one realisation, so overdispersion is unidentifiable without several fields or a simulation covariance | multiple fields/groups or simulations | 1.2 §5.1 |
| Field integrations and exports (RAIL plug-in, qp/CosmoSIS/nz_prior, popsummary, HEP formats) | Adoption work after the core exists | – | 1.2 §4, #39 |
