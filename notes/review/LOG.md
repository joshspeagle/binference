# Session log

Append-only. One entry per working session (or per step within a long session).

## 2026-10-22: plan agreed in outline

- Context: the v0.1 first pass is complete on `claude/pastry-v0.1` (last code
  commit before the walkthrough: see `git log`). Josh wants to go through the
  whole repository slowly, at varying depth, with copious notes that survive
  context resets.
- Created `notes/review/` (this directory) and a pointer in `CLAUDE.md`.
- Next: Josh chooses where to start and the level for the first step.

## 2026-10-22: start of Phase 1

- Josh agreed to go through the phases in the proposed order, starting with
  step 1.3 (architecture) at L2, then 1.1, 1.2, 1.4.
- Wrote `briefings/1.3-architecture.md`: end-to-end call flow, the five core
  objects, the Prior/BoundPrior and Engine contracts, the central equations,
  cross-cutting conventions, dependency graph, size map, and eight
  architecture decisions flagged for discussion.

## 2026-10-22: step 1.3 discussion, pull 2.1 forward

- Josh asked whether N-D grids lose their structure when flattened. Answer
  (checked in code): flattening is storage only (C order). `Bins` keeps
  `shape`, `edges`, `unflatten`, `neighbors`. Smoothness priors are built
  per axis: the random walk uses a Kronecker sum of per-axis RW precisions
  (eigenbasis via `_KronBasis`), the GP a tensor-product Hilbert-space basis
  with a joint D-dimensional Matérn/RBF spectral density and one length scale
  per axis (ARD). Dirichlet priors are exchangeable (no geometry by design);
  Pólya trees reject N-D grids.
- Josh wants: the details of every `from_*` conversion (Phase 2.3); a longer
  discussion of the core math; code reading alongside the briefings before
  judging the eight architecture decisions (deferred, parked in QUESTIONS).
- Pulled step 2.1 forward; wrote `briefings/2.1-core-math.md` (math + exact
  code locations).

## 2026-10-22: background audit launched

- At Josh's request, launched seven independent background auditors (review +
  simplify lenses), one per slice of the codebase. They write only to
  `notes/review/audit/<area>.md`; source is untouched. Index: `audit/README.md`.
- Findings will be triaged when the walkthrough reaches each area; outcomes
  go to DECISIONS.md. The walkthrough continues meanwhile with step 2.1.

## 2026-10-22: step 2.1 §1–2 (generative model, binning assumption, marginal posterior)

- Walked through §1–2 with code: `Likelihoods.from_gaussian` (CDF differences
  divided by bin width = the flat-within-bin average), `_normalise_linear`
  (rows scaled to max 1, log-scales kept), `PopulationModel.__init__`
  (`_ll_const`), `_kernels.loglike_and_counts`, `log_prob_and_grad`.
- Demo of the flat-within-bin approximation: population ∝ exp(−1.5 z),
  bins of width 0.5, object at d = 0.5 (on a bin edge): ratio of pastry's
  L_ik to the exact p(d | z ∈ B_k) is 1.32 / 0.79 / 0.72 for σ = 0.1 and
  1.05 … 0.78 for σ = 0.4. Errors of 20–30% per bin when the population
  changes by ×2 across a bin.
- Raised for discussion: flat in which coordinate (log bins are flat in z,
  not log z); correction options; non-integer weights.

## 2026-10-22: briefings become interactive pages

- Josh asked for briefings as interactive HTML so equations, text, diagrams and
  figures can be explored together. Convention from now on: each step gets
  `briefings/<step>.html` (published as a private artifact, links in
  `ARTIFACTS.md`) plus the `.md` record.
- Built `2.1-core-math.html`: MathJax equations, collapsible code excerpts with
  file:line, a mermaid data-flow diagram, and two explorers: (1) flat-within-bin
  vs exact bin likelihood (sliders: population slope, K, σ, d; shows per-bin
  ratios and the effect on the object's bin posterior), (2) stacking vs EM on a
  2000-object mock (iteration slider, KKT gap curve, distance to truth).

## 2026-10-22: Josh's questions on §1–2

- Flat-within-bin applies to all priors, including smooth ones (they put a
  smooth prior on per-bin log-fractions, but the likelihood is still built
  from flat-in-bin L). Key observation: the engines only need p(d_i|ρ) to be
  linear in ρ, so any fixed mixture basis works unchanged; ran an experiment
  showing linear B-splines cut the systematic bias from up to 14 s.e. to ≤ 3
  s.e. (parked in QUESTIONS).
- N-D independence: only `from_gaussian` assumes per-axis independent errors;
  other constructors accept correlated likelihoods (parked).
- Weights: Josh questions the rationale (parked with options).
- Gradients: hand-derived NumPy, verified by finite-difference tests
  (`tests/test_model.py::test_gradients_match_finite_differences`,
  `tests/test_priors.py`), no autodiff (D1).
- Audit reports received and committed: gibbs-vb-validation (12 findings),
  inputs (23, one major), outputs-api (21, one major).

## 2026-10-22: phone-friendly briefing pages

- Josh is reading on mobile. Reworked `2.1-core-math.html` (version 2): sticky
  swipeable section chips, split equations, selection modes as stacked cards,
  thinned chart labels, ratios listed as text, two sliders per row, EM
  re-simulation on slider release. Checked at 375 px in a headless-Chromium
  iframe. Requirements recorded in README.md.
- Audit report received and committed: model-map (14 findings, one major:
  zero-weight rows make EM reset or collapse under bootstrap weights).

- Audit report received and committed: nuts-laplace (15 findings, no bug-major; NUTS matches Stan; main item: results of logistic-prior fits expose the internal centred coordinates in `x`, `log_prob`, `model`).
- Audit report received and committed: priors (22 findings; core math correct; LogisticGP defaults misbehave on log bins (P-1), K > 48 (P-2), long length scales (P-3); DirichletTree per-node alpha crash (P-4)). Note: this OpenBLAS is pathologically slow multi-threaded for eigh; use OPENBLAS_NUM_THREADS=1 for experiments.

## 2026-10-22: decisions on weights, correlated errors, gradients; continuous field

- Recorded DECISIONS #1 (weights: multiplicities public, bootstrap internal,
  non-integer opt-in with warning), #2 (correlated Gaussian constructor with
  per-object covariances), #3 (NumPy gradients for now; autodiff backend on
  roadmap).
- Josh asked why the continuous field cannot be evaluated inside bins.
  Answer: it can; it amounts to separating the model's internal resolution
  from the reporting bins. Analysis parked in QUESTIONS for the Phase 3/7
  design discussion.
- Audit report received and committed: tests-docs (20 findings; slow statistical tests too weak: pass an ~18%-too-narrow posterior at the shipped seed (T-1); SBC never covers hyperparameters, GP, selection, rates, trees, Laplace (T-2); mix-SQP RuntimeWarning with mixed alpha=1/alpha>1 (T-4); several docs-vs-code contradictions). All seven audit reports are in (135 findings in total).

## 2026-10-22: step 2.1 §3–4 (selection, rates) and design decisions

- Recorded DECISIONS #4 (separate reporting bins from internal model
  resolution; continuous field evaluated inside bins), #5 (B-spline mixture
  basis), #6 (binning-bias diagnostic), at Josh's request.
- Added a selection explorer to the 2.1 page (version 3): 6000 objects,
  selection by true bin (logistic detection curve) or by a cut on the measured
  value; MLE fits with the correct mode, the other mode, and selection ignored;
  readout of distances to truth and the rates-view population size
  N_obs / (S·ρ̂). Checked with pastry that the correct-mode MLE error
  (total variation 0.07–0.14 over 8 mocks; posterior s.d. up to 0.03 per bin)
  is statistical noise, and said so on the page.

## 2026-10-22: Josh on §3–4

- #5 clarified: B-splines are for every prior family; the goal is no
  flat-within-bin assumption anywhere (#5 updated).
- Selection: decided names on_measurement / on_truth (#7) and combined
  selection with a worked example (#8). Input form (arrays vs callables)
  parked with a proposal.
- Rates: currently opt-in via the `GammaRates` prior (separate "kind").
  Derived that fractions-with-selection equals the rates model with the
  total marginalised under 1/Λ, so rates can be drawn exactly after any
  fractions fit; options A/B/C parked for the API discussion.

## 2026-10-22: selection from samples; rates afterburner

- Josh proposed selection inputs as arrays / callables / samples, with pastry fitting selection functions from injection samples. Agreed design recorded as DECISIONS #10 (incl. joint vs cut marginalisation); rates option B + C as #9. Open: default marginalisation.

## 2026-10-22: step 2.1 §5–6 (gradient, EM, stacking, KKT)

- Josh confirmed joint marginalisation as the default (DECISIONS #10).
- Page version 4: §3–4 marked discussed; §5 gains the softmax gradient derivation and cost; §6 gains why self-consistency is not optimality, the KKT code, SQUAREM / mix-SQP pointers, and the relevant audit findings (M-1–M-4, M-6). EM explorer now reports EM's measured asymptotic rate (≈ missing information) and the implied Gibbs autocorrelation time (0.998 → ~1000 iterations at the defaults).

## 2026-10-23: KKT explained; process decision on audits

- Explained the KKT condition (derivation, meaning of g_k as the rate of improvement from moving mass to bin k, sufficiency by concavity, gap as Frank–Wolfe bound ℓ* − ℓ ≤ W·gap). Demo: EM from a prior excluding bin 7 gets stuck at ρ_7 = 0 with g_7 = 1.65 (flagged); optimum has ρ_7 = 0.11, 57 nats better; bin 10 is a legitimate zero (g ≈ 0.2). Added to the page (version 5).
- Decisions #11 (stacking explicit in docs/API), #12 (KKT gap checked by check()), #13 (audit findings handled after the walkthrough).

## 2026-10-23: step 2.1 §7 (coordinates)

- Page version 6: why ILR (zero-sum subspace, orthonormal, Aitchison distance, no reference bin), Helmert code, the other families (logistic: linear in ξ, log θ Jacobian; rates: log λ; conjugate engines work on ρ directly), and a K = 3 explorer: Dir(α + n) on the simplex vs in ILR coordinates, and the tail of log ρ for an empty bin (98% span ≈ ln 99 / α), explaining NUTS tree-depth problems for small α.

## 2026-10-23: small concentrations, adaptive defaults, Jacobian convention

- Josh wants to understand the Dirichlet family (GD, Pólya trees) before
  deciding on small concentrations; prefers dynesty-style adaptive defaults.
  Recorded #14 (adaptive defaults, evidence-based, benchmark matrix in Phase
  4) and #15 (Jacobian convention explicit everywhere). Small-concentration
  decision deferred to after 3.1 (deep dive with explorers) and Phase 4.
- Benchmark `experiments/2.1_small_concentration.py` (1-D K=30 with ~1/3
  near-empty bins, N=1000; 2-D 12×12): NUTS vs Gibbs ESS/s — Dir(1) 66 vs 59;
  Dir(0.3) 8 vs 13; Dir(1/K) 0.5 vs 3.0 (NUTS 633 s, both poor); GD(1) 52 vs
  24; Pólya(c=1) 88 vs 83; Pólya(c=0.1) 22 vs 28; 2-D Dir(1) 34 vs 36. No
  NUTS tree-depth saturation except Dir(1/K) (3%). Problem is concentrated
  at very small concentrations; trees and N-D grids are not special.

- Josh clarified his question: structural assumptions of each prior vs the bin structure (edges, unequal widths, N-D). Answered from the code; parked as a design question.
- Benchmark: the 2-D Dirichlet(1/K) NUTS run did not finish within the 30-minute limit (killed); recorded in the .out file.

## 2026-10-23: step 2.1 §8 and wrap-up

- Josh: explore alternate coordinates and sampling strategies for complex
  priors (parked, Phase 3.1/4). Asked about the NUTS implementation: it is
  pastry's own NumPy implementation of Stan's algorithm (not a wrapper);
  the audit confirmed it matches Stan's base_nuts line by line. Exposed vs
  hard-coded options parked as a question for 4.2.
- Page version 7: §8 (population vs realised sample, with a small in-page
  Gibbs sampler showing both intervals; individual-object posteriors) and §9
  outcomes (decisions #1–#15, open threads). Step 2.1 marked done.

## 2026-10-23: step 2.2 started

- DECISIONS #16: expose everything with sensible defaults. Agreed order: 2.2 → 2.3 → Phase 1.
- Published `2.2-bins.html`: Bins anatomy, where volumes enter, flat-in-which-coordinate, exact large-N binning-bias calculator (N_crit) comparing flat-in-z, flat-in-log-z and linear B-spline bin shapes; numbers checked with a Node harness.
- Josh: bins must stay arbitrary user edges for continuous problems with any prior; internal resolution and B-spline knots should adapt to bin sizes/scope. Recorded #17 (1-D and rectilinear N-D arbitrary edges already supported).
- Josh on 2.2 §6: measure on Bins to discuss (proposal written, parked), wants warnings for outside mass (#18) and masked grids everywhere (#19); asked whether bias is used internally (no) and whether bins adapt (no): proposal #20 pending.
- Josh accepted #20 (bias/adaptivity strategy) and the measure design (#21), with densities allowed via numerical transforms and dynesty-style docs. Step 2.2 done; next 2.3.

## 2026-10-23: step 2.3 page

- Published `2.3-inputs.html` (one section per constructor, audit cross-references, three explorers: interim-prior mistakes, samples per object, quadrature error). Fixed during checking: the posterior explorer's population sampler (now exact), the samples explorer readout (compare the two fits), the quadrature reference (composite Gauss–Legendre instead of an erf approximation); text numbers verified in Node (`experiments/2.3_inputs_check.js`).

## 2026-10-24: step 2.3 discussion

- Josh on the constructors: interim-prior options should pass everywhere; samples need a stated sampling distribution, possibly a rename, and a density-fit route; function/distributions convention unclear; asked what response and classifier inputs are.
- Settled: #22 general bins everywhere, #23 adaptive quadrature, #24 one treatment of likelihood/posterior/interim across inputs, #25 correlated Gaussians (full covariance; exact 1-D/2-D; adaptive cubature vs QMC to test, theory then practice).
- Parked: samples design, function/distributions merge and name, response explanation, classifier discussion (QUESTIONS).

## 2026-10-24: step 2.3 follow-up page (2.3b)

- Josh: PopulationPrior/InterimPrior objects for all interim-prior issues; harder look at defaults/arguments; samples need a likelihood path and are a realisation of the PDF (no KDE); merge function/distributions; clarify response; work classifiers through statistically and as API.
- Experiments: `2.3_samples_mc.py` (log-likelihood MC error ∝ N/S), `2.3_classifier_routes.py` (routes A/B/C under miscalibration, wrong π_train, label-shift violation; coverage of response-uncertainty treatments; Gibbs-over-R comparison run inline and recorded in the briefing).
- Finding: averaging the likelihood over response draws undercovers badly with many response entries; Gibbs with R as parameters is calibrated. Same risk in DECISIONS #10's selection-draw default (flagged in QUESTIONS).
- Published `2.3b-redesign.html`; started `api-ledger.md`. Page checked at 375 px (no overflow, no console errors; explorer reacts as described).

## 2026-10-25: step 2.3b decisions

- Josh accepted: shared density class + PopulationPrior (#26), Response + from_binned (#27), response inferred jointly by default with a fixed option (#28), classifier route set by the inputs and reported, label-shift check (#29), selection treated the same way, revising #10 (#30).
- Pending: samples (Josh asked whether the Monte Carlo integral and the multinomial view are the same; answered: identical for histogram bins with an interim flat inside bins, MC form generalises), constructor naming proposal (from_table / from_density / from_samples / from_gaussian / from_binned).
- Josh confirmed the samples estimator (#31) and the constructor names (#32). Step 2.3 done. Next: Phase 1 step 1.1.

## 2026-10-25: step 1.1 page

- Published `1.1-scope.html`: pitch, the six principles checked against code and decisions (filterable cards), four proposed new principles, use cases today vs after the redesign, scope moves, docs-drift list, questions. Found: principle 6 drift (MAP returns `PointEstimate` without summaries/intervals/functionals). Checked at 375 px.
- Josh: vision = sklearn-easy for novices, all knobs for experts, blazing fast, the right thing so easy that not doing it feels criminal. Decisions #33–#37 (rewrite value statement, adopt principles + 'fast', one result object with draws + best fit by default, scope moves, field-neutral). `vision.md` written. Step 1.1 done; next 1.2.

## 2026-10-25: step 1.2 page

- Published `1.2-landscape.html`: one model / ten names, field-by-field bars to beat, six error-bar failures vs pastry, filterable scorecard of the survey's recommendations (done / changed / not done), items not yet discussed (overdispersion, PDF calibration, implicit interim priors, scale, run control, exports), frankenz recap. Found: `requires-python >=3.11` vs SPEC 0 (≥3.13 after 2026-Q4); no golden-value tests against R/Julia solvers.
- Josh on 1.2: calibration/tempering/implicit interim priors need ground truth → post-v1 open issues (#38, `post-v1.md`); no field priority yet, diverse robust benchmarks first (#39); scorecard items adopted. Proposed back: overdispersion post-v1 too (unidentifiable from one field), and a v1 / post-v1 split of the scorecard items (awaiting confirmation).
- Josh agreed: overdispersion post-v1 (#40), v1/post-v1 split (#41), benchmark matrix (#42). Step 1.2 done; next 1.4.

## 2026-10-25: step 1.4 page

- Published `1.4-decision-log.html`: D1–D16 sorted into settled (6), at their step (8), decide now (D5, D8, going public early).
- Josh on 1.4: D8 → convert notes/survey into thorough docs (#43), raw material as a bundle; no JOSS, journal or arXiv instead (#44); D5 name reopened: brainstorm in `naming.md` (shortlist quincunx, propto, popover, populace, polyurn; PyPI checked).
- Naming round 2: Josh likes popover/populace directions and binference-style twists; round-2 shortlist in naming.md.
- Josh chose the name **binference** (#45). Step 1.4 done; Phase 1 done except 1.3's deferred decisions. Next: 2.4.

## 2026-10-25: step 2.4 page

- Published `2.4-selection.html`: Selection object, model and per-engine handling (all correct; M-2 and VB caveats), proof that only ratios of S matter for ρ under the 1/N total prior but totals/rates need absolute S, new within-bin selection bias explorer (11% bias = 1 s.d. at N ≈ 6,600 for a sharp edge), gaps vs #7–#10, #30, #41.
- Josh agreed all four: absolute/relative S (#46), within-bin selection bias handling (#47), mocks for truth, measurement and combined selection (#48), selected population returned by default (#49). Step 2.4 done; next 2.5.

## 2026-10-25: step 2.5 page

- Published `2.5-individuals.html`: realised counts, assignment probabilities, value draws, predictive methods; explorer of one object's value posterior (histogram vs B-spline, numbers verified in Node; corrected a claim that broad errors remove the effect).
- Josh: value draws from within-bin info (#50), `_best` suffix for plug-in per-object outputs (#51), leave-one-out on the to-do list (#52), full-sample realised counts (#53). Step 2.5 done; next 2.6.

## 2026-10-25: step 2.6 page

- Published `2.6-model-core.html`: model attributes and methods, one evaluation, kernels with a speed benchmark (`experiments/2.6_kernel_speed.py`; memory-bound, dtype-mixing trap), simplex helpers, kkt_gap and missing_information with audit fixes, M-1…M-14 table.
- Josh: speed plan (#54), float32 option (#55), KKT fixes (#56), each concept once (#57); confirmed the audit findings (review + simplify) all go into the Phase 7.3 change list (audit README and PLAN updated). Phase 2 done; next 3.1.

## 2026-10-25: landing plan

- Josh renamed the repository to binference (default branch main, docs via Actions) and asked for a plan to bring the work to main through issues and reviewed, stacked PRs (telescopic as model). Drafted `landing-plan.md` and `briefings/landing-plan.html`: phases A (history, 4 PRs), B (process/CI, rename), C (14 PRs: decisions + all 127 audit findings), D (deferred issues); five questions open.
- Josh: land history (#58), Claude merges on green CI (#58), implement every decision and finding now (#59), minimal tooling, review summaries in PRs (#60). Plan revised; starting Phase A.

## 2026-10-25: landing, phases A and B started

- Tracking issue #4 with phases #5–#8; Phase A PRs #13–#16 (history), #13 merged; #14 needed fix-forward commits for CI (ruff pin, Markdown excluded from ruff, docs step skipped before docs existed). B1 #19 (process/CI) and B2 #20 (rename) opened; B1 went through /simplify (two reviewers) and /code-review (medium); fixes applied (DECISIONS #61).

## 2026-10-25: B merged; C1 reviewed

- #19 (process/CI) and #20 (rename) merged; Phase A and B issues closed by their PRs.
- C1 #22 (measures and masked grids): /simplify with four reviewers (one masked-state pair and a cached cell lookup; `cell_index`; measures compare by an identity key; unused `derivative`/`is_linear` dropped; `from_gaussian` gathers active cells only) and /code-review high (ten findings, all fixed: masked 1-D in the cdf constructor, plots and mock samples; mask shape; partly masked coarsen blocks refused; `from_centers` clamping; warning stacklevel; transform measures reload as tables, documented; DECISIONS #62 corrected).

## 2026-10-25: C1 merged; C2 opened

- #22 (measures and masks) merged after its reviews; phase B closed (#6).
- C2 issue #23 and PR #24 (`Density`): one fixed density for interim priors, base measures, logistic means and mocks; flat in the measure coordinate by default (DECISIONS #63); array interim priors need a representation on unequal bins (I-1); `PopulationPrior`; `stack()` honours the recorded interim (I-5, I-6). /simplify (four reviewers) and /code-review high done, all findings addressed or answered in the PR.

## 2026-10-25: C2 and C3 merged; C3b reviewed

- #24 (Density) and #27 (constructors by input form, adaptive quadrature in the measure coordinate, outside mass; DECISIONS #64) merged. In #27 the efficiency review caught a 25 s regression for Gaussians on log bins (refinement for every object at once); per-object refinement brought it to 0.03 s. Code review caught a false non-convergence warning (fixed reference scale) and N-D shape bugs.
- C3b #28 (#26): weights policy (DECISIONS #1, #65), read-only storage, sparse compress, near-median Gaussian precision, warnings attributed to the caller via skip_file_prefixes, validation and docs drift. /simplify (two reviewers, two angles each) and /code-review high done; two regressions caught and fixed (uneven grid edges in one list, scalar Gaussian mean).

## 2026-10-25: C3b, C3c and C4 merged; C5 opened

- #28 (C3b), #30 (C3c, correlated Gaussians; DECISIONS #66) and #32 (C4, one Monte Carlo estimator for samples; DECISIONS #67) merged. In #32 /simplify unified the per-bin and pointwise estimators (a per-bin prior is flat in u within bins) and built the sums with one sort; /code-review caught the variance using summed weights as the sample count, an infinite variance under the floor, tail underflow from exp-then-log and cdf-only distributions crashing, all fixed with tests.
- C5 issue #33: `Response` (matrix, simulated events, labelled validation set) and `from_binned` replace `from_response`/`from_classifier`; classifier route A is `from_table`, route C `from_binned` with a labelled response; `summary()` names the inputs; a label-shift check (DECISIONS #68). A plug-in χ² was rejected after calibration runs (false alarms on ill-posed unfolding and on equal-size validation sets); the posterior predictive version had no false alarms in 70 null replicates.

## 2026-10-25: C5 merged; C6 opened

- #34 (C5, binned measurements and classifier routes; DECISIONS #68) merged. Its reviews moved the label-shift check onto `Response.predictive_pvalue` with a Woodbury form (0.025 s at J = 100) and found the missing normalisation term of a response with efficiency below one, open-ended and weighted tallies, and the warning class.
- C6 issue #35: selection modes renamed, combined selection, callables, absolute/relative scale, steep-S warning, mocks for selection on the truth, the measurement and both (DECISIONS #69). Calibration of the measurement-cut mock exposed an inverted open upper limit in `measured_range` (fixed with a test); with it, NUTS z-scores are calibrated, and S_k matches empirical pass rates.

## 2026-10-25: C6 merged; C7 opened

- #36 (C6, selection; DECISIONS #69) merged. Its reviews removed an `averaged` flag in favour of checking only raw arrays, unified the bin averaging, and found far-tail cancellation, NaN on empty bins and too coarse a quadrature tolerance for small S_k; macOS CI exposed a mix-SQP test held tighter than the solver's stated tolerance.
- C7 issue #37: model and engines (DECISIONS #70). Checking the corrected missing information against the numerical EM Jacobian showed that the formula holds on the support only: at a fraction of zero the EM rate is that of an active constraint (it reached 0.994 for a bin near the boundary), so the function now works on the support.

## 2026-10-25: C7 merged; C8 opened

- #38 (C7, model; DECISIONS #70) merged after its reviews.
- C8 (MAP, EM, mix-SQP; DECISIONS #71): one tolerance on the KKT gap, mix-SQP for `GammaRates`, one reported log posterior, no SVD. Timing showed the audit's uniform start for mix-SQP is wrong for this solver (105 s against 4.9 s after 10 EM steps at N = 2e5, K = 200), so the start keeps a few EM steps. Requiring the gap for EM convergence exposed two faults: SQUAREM accepted slightly worse extrapolations (a fake stall), and the C7 rates gap read a rate EM drives towards zero as a violation of order one; both fixed.

## 2026-10-25: C8 merged; C9 split

- #40 (C8, MAP; DECISIONS #71) merged. Its code review found that the C7 rates KKT gap and the prior gradient at zero both misread active constraints (−b dropped for a zero rate; α < 1 never converging on the simplex); both now evaluate the prior gradient at the boundary point the MAP objective uses.
- C9 split in two: C9a (mechanical prior fixes, DECISIONS #72) and C9b (the GP defaults P-1–P-3, P-5, P-7, which need measurements and choices).

## 2026-10-25: C9b (GP defaults)

- Calibrated the HSGP basis against exact kernels (field SD and covariance within 5%): the basis needed is m ≈ f·c·S/ℓ with f = 1.27, 1.73, 2.13, 8.3 for RBF and Matérn-5/2, -3/2, -1/2, constant in ℓ; the bin-to-bin step SD of rough kernels needs a basis set by the grid instead. Default: one sine per bin and the length-scale prior raised to what that resolves (DECISIONS #73); the larger-basis alternative is in QUESTIONS.
- The centred form's underflow fix needed log-space arithmetic, not only log scales: a centred coefficient underflows with its scale, and 0·inf is NaN.

## 2026-10-25: C9a merged; C10 opened

- #42 (C9a, priors) merged; #44 (C9b, GP defaults) in review, retargeted to `main`. Its code review removed an ellipsoidal mode cut added in /simplify (it dropped each axis's top mode and lost field variance), so the default basis is bounded at 1024 modes instead.
- C10 (NUTS, Laplace; DECISIONS #74), stacked on C9b. Reordering the Laplace eigenvectors alone moved k̂ from 0.67 to 0.80 on a selection test with the same seed, so that test had passed by luck at the threshold; it now ignores the approximation warning, since it asserts z-scores.

## 2026-10-25: C11 (Gibbs, VB, bootstrap, validation)

- C11 stacked on C10 (DECISIONS #75). The new SBC low-ESS warning fired in two existing tests with short single chains, as intended; they now expect or silence it. The boundary quadrature is exact to 0.01 SD in means down to α = 0.2, but at α = 0.2 the tail mass below a 6e-9 quantile is 17% off at 400 nodes and 22% at the default 200, so the docstring states that limit instead of a stronger substitution.
- #46 (C10, NUTS and Laplace; DECISIONS #74) merged after CI on macOS and the minimum-dependency job: its thread test first assumed a visible BLAS pool, and now asserts the limit only where threadpoolctl sees one.
- C11's /simplify (three reviewers: reuse with simplification, efficiency, altitude) and /code-review high replaced the row-repeating sparse label draw with conditional binomials over each row's nonzeros (a flat row no longer densifies a chunk), lowered the exact posterior's default grid to 200 nodes (tests reading the mode off the nodes keep 500–600), started Gibbs chains from counts that include missed objects, stopped silencing the bootstrap's base-fit warning, and rejected SBC `n_bins` above n_draws + 1.
- The slow SBC tests met the new low-ESS warning (Gibbs: 46 of 200 fits below ESS 99); `test_sbc` filters it, since over-dispersed ranks only make its uniformity test stricter.

## 2026-10-25: C11 merged; C7b (basis and internal resolution)

- #48 (C11) merged after CI.
- The landing plan's revision (DECISIONS #59) placed C7b (basis and internal resolution), C7c (speed) and C11b (joint inference) after C7 and C11; they were passed over in sequence, and C12 depends on C7b and C11b, so they come next: C7b, C7c, C11b, then C12.
- C7b issue #49 (DECISIONS #76). The bias benchmark separates two effects: in the noise-free limit, plain bins and hats at resolution 1 differ by 2×, while finer resolution without a smoothing prior rings (maximum likelihood is not identified component by component); with a random-walk prior, resolution 4 cuts the error from 7.4% to 0.1–0.3%.
- C7c (DECISIONS #77): measured before building. With BLAS threads the NumPy kernel beat a parallel numba kernel up to N = 2×10⁵ and lost by 16% at 10⁶, and float32 changed little while its expected counts drifted, so #54 and #55 are revised to one NumPy kernel in float64; the dense/sparse threshold is 10% (was 30%), the largest fill at which sparse won in every case. The code review caught statements that quoted only the rows that fit and hand-assembled outputs; the experiments were rerun as scripts with separate errors for the log-likelihood and the counts, and the notes now quote every setup.
- C11b split in three PRs: derived rates (#9), selection from injections inferred jointly (#10, #30), response inferred jointly (#28). C11b-1 (DECISIONS #78): fractions results carry exact totals and rates; the rates kind stays as option C (question parked).
- C11b-2a (DECISIONS #79): injections give an uncertain selection on the truth, inferred jointly by Gibbs; SBC calibrated. Selection on the measurement cannot be a free parameter (the likelihoods' cut fixes it; a first version ran S to zero), so it is a plug-in only.

## 2026-10-25: C11b-1 and C11b-2a merged; C11b-2b (joint selection in every gradient engine)

- #54 (C11b-1) and #56 (C11b-2a) merged after CI and the slow suite.
- C11b-2b (#57, DECISIONS #80): logit S joins the unconstrained coordinates, so NUTS, Laplace and unconstrained MAP infer a selection from injections; NUTS and joint Gibbs agree within Monte Carlo error. The smooth injection model (random walk on logit S) lowers the error of S by 20–45% for smooth completeness from 5 injections per bin, and does not help at a sharp step; no default τ.
- The smooth model keeps the Beta(a, b) prior instead of the issue's flat level: with a flat level a bin whose injections were all recovered leaves the posterior improper, as the catalogue fixes only ratios of S.
- The first logistic SBC failed on an overflow warning in the NUTS energy, which the tree already handles as a divergence; it is now silent there. A few fits take 20–40 s with tree depth 6–7, about as often with S fixed (2 of 25) as inferred (3 of 25), so they come from the logistic prior on small problems; the SBC was cut from 100 to 60 simulations (452 s → 170 s).

## 2026-10-25: C11b-2b merged; C11b-3 (response inferred jointly)

- #58 (C11b-2b) merged after both reviews; /simplify kept partial centring on with an inferred selection (acting on the prior's block), /code-review added input checks and the density of S in MAP's log posterior.
- C11b-3 (#59, DECISIONS #81): a tallied response is inferred with the population by every engine but VB and the bootstrap. The plug-in response's 68% intervals covered the truth 23% and 34% of the time with 30 and 100 validation objects per class; the joint fit's 58–62%.
- The prior per cell was first a total of one per column (β = 1/(J+1)), which had the lowest error, but cells without events then have α < 1 and NUTS took 251 s at tree depth 7.3 on tutorial 6's route C; β = 1 took 40 s, so it is the default.
- The SBC found a measured bin with no tallied events but observed objects: `from_binned` rejected them (zero row). For an uncertain response the rows are now the posterior means.

## 2026-10-25: C11b-3 merged; C12 (results)

- #60 (C11b-3) merged. /code-review caught that `fixed()` dropped the tallies, so the plug-in's label-shift check lost the covariance calibrated in DECISIONS #68, and that responses without lost events got a spurious random efficiency; both fixed before the merge. The single interface for jointly inferred blocks is deferred (#61).

## 2026-10-25: C12a merged; C12b

- #63 (C12a, audit O-1–O-21) merged after CI and the slow suite. Checking the audit against `main` found all but O-16 open and a new defect (attach redrew the total without the selection); /code-review caught that the new bins check refused a reloaded transform axis and that R-hat was checked for independent draws.
- C12b (#64): selected population, full-sample counts, KKT in point-estimate checks, data support and the prior-vs-data ratio (DECISIONS #12, #41, #49, #53).

## 2026-10-25: C12b merged; C12c

- #65 (C12b) merged. /simplify moved the ratio to diagonal Hessians (same traces) and shared the check reporting; /code-review caught an infinite KKT gap passing, exposure selections and attached rate results accepted by full-sample counts, data support ignoring a selection on the truth, and truth overlays on the selected population not reweighted; all fixed with tests before the merge.
- #67 (C12c) merged. /simplify rebuilt the dispatch as one draw function per input and vectorised the sample draws (20,000 objects: 0.04 s, from 0.36 s) and the correlated Gaussians (80,000 draws: 0.39 s, from 1.17 s); /code-review caught NaN cumulative weights from zero sample weights, draws collapsing where the interim prior vanishes, cdf-only distributions crashing and density data lost on every pickle; all fixed with tests before the merge.

## 2026-10-25: C12d (best fit, `_best` outputs, leave-one-out)

- C12d (#68, DECISIONS #85): `result.best`, the `_best` outputs and PSIS leave-one-out posteriors. Leave-one-out matches an exact refit within Monte Carlo error where the in-sample posterior is off by more than 8σ. Two choices are parked in QUESTIONS (one result type; the count downdate).
- #69 (C12d) merged. /simplify moved leave-one-out onto the two-product average (2000 draws: 0.56 s, from 5.4 s); /code-review caught the total missing from rate-model weights (0.530 against a refit's 0.504 with unequal exposures; fixed: 0.5035), `best` after `load` using the default prior, bootstrap draws accepted, and `psis` giving k̂ = ∞ for equal weights; all fixed with tests before the merge.

## 2026-10-25: C12e (binning-bias check)

- C12e (#70, DECISIONS #86): `result.binning_bias(build)` with `n_crit` as the planning number. The default threshold is 2 s.d., not 1: two models of the same flat population differed by up to 1.0 s.d. One question parked (exact large-N calculation, suggested spacing).
- #71 (C12e) merged; C12 is complete. /simplify made `build` optional for Gaussian inputs and reported `n_crit` only for flagged bins; /code-review caught an unvalidated `build`, selections from injections and the flattened fit selection, method names, bins without spread and tables printing inf as nan; CI then caught a test whose `build` returned other data, so its warning depended on the platform's numerics.

## 2026-10-25: C13 (tests)

- Checked the test audit against `main`: of 33 items in T-1–T-3 and T-9–T-13, 4 were fixed, 9 partly and 20 open. Split in two: C13-1 (#72) the fast-test gaps (T-3, T-9, T-10, T-11, T-13), C13-2 (#73) the statistical power and the new SBC cases (T-1, T-2).
- C13-1: three agents wrote the tests in separate worktrees (T-3 and T-10; T-9; T-11 and T-13), each check confirmed by a temporary mutation that makes it fail. The bootstrap KKT test needed a selection and a non-flat Dirichlet: without them MAP runs mix-SQP, which revives zero components, so the old start passed too. The 2-D GP covariance is compared with k(r) on the scaled distance (the Matérn here is not separable). The ESS, R-hat and MCSE now match ArviZ 0.23.4 to 1e-10 on a fixed array. The fast suite: 609 tests in 88 s, none over 4 s (the centring test went from 11 s to 0.2 s). Found on the way: an RBF GP with inferred hyperparameters and partial centring on a 2-D grid has gradients near 3e11, too ill-conditioned for finite differences (not tested; a follow-up).
- #74 (C13-1) merged. /code-review caught a one-sided bootstrap bound (an s.d. 1/√2 too small passed) and a shortened warm-up that moved the centring decision from iteration 25 to 9; both fixed before the merge.
- Josh asked how the plan stands and whether the landing-plan page is kept current. It is not (last updated in Phase B); he chose to have it rewritten once, as the landing summary, at the end of C15.
- C13-2 (#73, DECISIONS #87): two agents in separate worktrees, one on power (an exact ECDF test, 1000-simulation Gibbs SBC, a negative control, coverage and recovery), one on new cases (selection and rates in `prior_predictive`, six SBC cases, three Geweke cases). No case showed a calibration problem in the package. The GP SBC is left as a follow-up: its NUTS fits need long warm-ups.
- C13-2 reviews. /simplify shared one SBC harness between the two slow files (the new cases now also require the ECDF test, and pass), built the measurement-side selection with `Selection.from_function`, and returned the Bonferroni bound for ECDF p-values below 1e-12 (badly skewed ranks: 0.1 s, from 26 s). /code-review found no statistical error; it caught an unbounded redraw loop for rate priors expecting almost no objects (now an error after 1000 empty draws), the hard-coded 1e6 object limit (now `max_objects`) and ESS asserts that could never fail. One question parked (which test `passed` runs by default). The slow suite took 19.6 min on one thread (13.8 before); a second run under load from four review agents took 25.3 min.
- #75 (C13-2) merged; C13 is complete. Next: C13b (benchmark matrix and adaptive defaults, #14, #39, #42; golden values #41).

## 2026-10-26: C13b (benchmark matrix, adaptive defaults)

- C13b (#76, DECISIONS #88): `benchmarks/matrix.py` runs 18 mock catalogues × the priors that apply (43 combinations) × six engines against 20,000-draw NUTS references; the full run took 90 min with four single-thread workers. NUTS was within 0.08 s.d. on every mean; Gibbs beat NUTS wherever the missing information was at most 0.976 and lost in 18 of 21 combinations above 0.994; Laplace passed its check only with the Pólya tree (there within 0.05 s.d.) and was up to 8 s.d. off elsewhere. From this, `infer` now defaults to `method="auto"`: Gibbs (run to a bulk ESS of about 400) below 0.98 missing information, then Laplace if its check passes with an importance ESS of 400, then NUTS.
- Golden values against mixsqp, REBayes and Empirikos (DECISIONS #41) need R or Julia; installing them was refused in this environment, so they are deferred to #77.
- A slow regression test checks NUTS, Gibbs, Laplace and auto against stored references of four matrix cells. Its first version flagged the random-walk cell: the reference had 42 divergences (cell replaced; references must have none), and an s.d.'s Monte Carlo error needs the ESS of the squared deviations, about half the bulk ESS under NUTS.
- #78 (C13b) merged. /simplify turned auto into one engine (`Auto`) with its thresholds as options and filtered every candidate by the options given (`infer(lk, df=3.0)` had crashed in the NUTS fallback); a hopeless Laplace trial now stops after 1000 proposals (0.2 s, from 1.6 s at N = 10⁴) and Laplace finds its mode once. /code-review caught that the new default crashed on basis likelihoods (missing information taken on bin fractions instead of component weights); fixed with a test, with five smaller fixes.

## 2026-10-26: C14 (docs)

- Split in three: C14-1 (#79) the docs audit findings and the spec and test plan refreshed; C14-2 (#80) vision, principles and a field-neutral front page; C14-3 (#81) the survey as background pages and a tutorial per field. Three agents in separate worktrees (audit, background pages, tutorials); C14-2 written directly.
- C14-1: of 11 findings, 3 were already fixed, 7 partly (finished here) and T-12 fixed here: the test plan rewritten against `tests/`, with planned tests that do not exist marked "Not implemented"; spec §3–§11 and the implementation plan matched to the code, citing the decisions that changed them. Hidden assert cells now check the numbers tutorial prose relies on; two prose claims were false (a divergence in tutorial 06; "smaller" realised-count uncertainty in 01).
- #84 (PR #85, DECISIONS #89) merged: writing tutorials 10 and 11 showed that auto picked Gibbs for jointly inferred responses and selections and that run failed its own check (R-hat 1.09 and 1.54); auto now extends Gibbs until it mixes, or gives way. #82 (C14-1) merged after its reviews: /simplify chunked `log_predictive` instead of documenting its memory, and gave the auto and MAP rules one home each (their docstrings); /code-review caught a full N×K copy of new-object rows under a selection on the truth (fixed with a test) and seven documentation slips.
- C14-3: the eight background pages and tutorials 09 (empirical Bayes), 10 (mixed stocks) and 11 (injections), each with hidden assert cells; 26–43 s each on one thread. Tutorial 11 showed that totals derived under the default flat Dirichlet come out low when some bins are poorly determined (simulations of its setup: mean z 1.5–1.9): documented in the selection guide and parked in QUESTIONS. /simplify (four reviewers): repeated explanations in the background pages cut to links (classifier routes, unfolding responses, the injection selection, the auto rule, the binning-bias figure, the survey date), assert cells reuse the visible cells' values, `sel.injections` instead of re-tallying, shorter NUTS runs in tutorial 11 (49 s to 23 s), and links between each tutorial and its field page. Kept: the cross-field comparison table in software.md beside the per-field tables (it is that page's purpose), and the "Added" paragraphs (each says what is new for its field). A public accessor for inferred selection draws (tutorial 11 reads `info["selection_draws"]`) is deferred to #86.
- C14-3 /code-review high: ten findings, all fixed. Background pages overstated binference in four places (GW: a selection on the measurement is only a plug-in; mix-SQP only for concentrations ≥ 1; Laplace before NUTS in auto; `sample_variance` also returns effective sample sizes) and made one unverified claim about PyMC (removed). Tutorials: a reversed colour description (09), a "fraction below 10 Msun" that was below 10.15 (11), and three prose claims without a test or with a tolerance not from Monte Carlo error (now asserted: A and B intervals widen, A–B the most confused pair, totals compared in standard deviations; stock E's interval also widens, so the prose names C and D only).
- #87 (C14-3) merged.
- C14-2 (#80): `docs/vision.md` (what binference is for, and its design principles), a field-neutral README and front page, and spec §1–§2 rewritten. /simplify (two reviewers): the principles now have one home (vision.md; the spec lists their titles and decisions), and the specifics that would drift (input lists, the auto rule, dependencies, speed features) are links. The accuracy reviewer and /code-review medium found the principles promising more than the code does. Automatic checks run only after Gibbs and NUTS fits. A bare array, and the plain `Likelihoods(values, bins)`, are read as likelihood densities. Model approximations are measured on request. Order-0 bases do not relax the flat-within-bin assumption. Nothing chooses a basis automatically. The page now states what the code does, and the code-side gaps are deferred to #88.
- The same review found that auto kept Laplace draws whose importance ESS fell short of `target_ess` (on very large inputs, or as the last candidate) without flagging them approximate, contrary to its docstring: fixed in #89 (PR #90), which flags them, warns once, and gives the reason.
- #90 (the auto flag fix) and #91 (C14-2) merged; C14 is complete. The slow suite passed on the final Phase C code (36 passed, 1 expected failure: VB's over-confidence; 28.8 min on one thread, with other jobs running).

## 2026-10-26: C15 (phase-wide review)

- C15 (#92): `/simplify` with four reviewers and `/code-review high` on Phase C's whole diff (`7edf994..main`, about 10,000 lines of `src/`). The code review found eight bugs between separately reviewed PRs: `attach` after load refused a valid result with a basis and a selection, and accepted a model with another prior for `best` and `binning_bias`; `save` dropped the unconstrained draws (so `prior_information_ratio` changed after load); `from_samples` silently fixed an inferred selection or response at its mean; a jointly inferred response's diagnostics were named the selection's; dropping rows from a response model left its measured-bin index stale; Gibbs `labels` on a basis are component indices (documented). The cleanup ran as three agents in separate worktrees (results and validation; engines and model; inputs and utilities), every line read before it was taken. It found three more bugs (PSIS compared the cut as `log(exp(cut))`, so ties entered the tail and leave-one-out crashed on resampled Laplace draws; mock posterior samples ignored unequal step widths; `Selection.draw` could return exact zeros) and made Laplace's importance step 5.3 s from 28.4 s (N = 10⁵, K = 50), leave-one-out 3.1 s from 7.4 s, EM one `L·x` per point, Gibbs labels in place. Names unified and unreleased shims removed (DECISIONS #90). Deferred: plain bins as an identity basis (#93).
- C15's own reviews (1,628 lines: four simplify reviewers, `/code-review high`): the batched density became private (`_log_density_many`), SBC's log-likelihoods are batched, a smaller dense chunk had slowed sparse sampling (up to 57%), so sparse rows are now sampled at once (0.030 s per draw, from 0.035 s); `attach` compared priors by repr, which shortens arrays, so it compares a digest of the arguments (saved with the result); `attach` set its state before validating; Auto's cached MAP hid MAP's warnings (now cached only when it raised none); a degenerate PSIS fit gives k-hat = inf (was NaN); DECISIONS #90 matched to what shipped.
- #94 (C15) merged after the slow suite passed on its branch (36 passed, 1 expected failure; 29.3 min on one thread). Phase C (#7) is complete. The landing-plan page is rewritten as the landing summary (same link); its source is `briefings/landing-plan.html`.
- CI on #96 (notes only) failed once on Windows with Python 3.14: `test_mixsqp_with_selection_solves_for_observed_fractions` asserted a KKT gap in ρ below 1e-7, which mix-SQP does not guarantee with a selection (it bounds the observed fractions' gap by `tol`, or 1e3 × `tol` once round-off stops it; the gap in ρ is larger by up to max S / (S·ρ)). The run stopped on round-off at 4.1e-7. A second Windows job failed the same way (`test_mixsqp_on_tall_dense_inputs_and_fallbacks`, 1.03e-7 against 1e-7). Every MAP test now asserts the solver's guarantee (`tol`, or 1e3 × `tol` once round-off stops it; scaled by max S / (S·ρ) for ρ under a selection) instead of the gap a run usually reaches.

## 2026-10-26: Phase 3 resumes, step 3.1 (Dirichlet family)

- Briefing `briefings/3.1-dirichlet-family.html` published (record `.md` beside it): the four priors as Dirichlet trees, from first principles, with three explorers (Dirichlet on the 3-simplex; any tree on K = 8 with its correlation matrix and the package's unbounded-density test; Pólya draws on 16 bins) and a three-level code tour of `priors/dirichlet.py` and `priors/base.py`. Every figure measured on `main`.
- Findings: symmetric Dirichlet correlations are −1/(K−1) whatever α is (−0.14 at K = 8 for α = 1 and 0.1); trees give positive within-group correlation (+0.10, two groups of four, α = 2); the default Pólya tree +0.72 for neighbours; GD(1, 1) halves the mean per bin; most tree priors, including the default Pólya tree, have unbounded densities on the simplex (bounded in ILR).
- Five questions for Josh: small concentrations under NUTS, tree-matched coordinates, GD's mean, Pólya splits, GD order on N-D grids. Three new items parked in QUESTIONS with defaults; the other two already were.

## 2026-10-27: walkthrough paused for the documentation site

- Josh paused the walkthrough at step 3.1 (briefing published, his questions open) to build the full documentation site first, following the process used for dorothy, brutus and GSPICE and the design language of joshspeagle.com and the AstroStat U of T site: a design sheet, a logo and design principles agreed with him, then the pages written autonomously through tracking issues and pull requests that Claude merges. Reading the docs is meant to give him the background for resuming at 3.1. The 3.1 page is to be updated against the code once the docs land.

## 2026-10-27: docs phase (#99), audit to design lock

- Research on dorothy, brutus, GSPICE and Josh's two websites; one process for all: audit, content plan, design rounds, a committed design document, then the build. Decisions #91 (process, Sphinx with a local theme, an independent identity, students first) and #92 (an explicit MAP algorithm runs or raises).
- Audit (three agents): 36 wrong or stale statements and 3 code bugs; a flat menu of 11 pages and no concepts layer; 26 notebook figures without alt text. Fixed on the current site: #100 (PR #101, code bugs) and #102 (PR #104, the statements; tutorial 08's E-BFMI warning fixed with 10 sines per axis). Deferred: #103 (`n_samples` under auto, the 2-D GP basis, `auto_reason`).
- Content plan (seven sections, reader paths, 20 visuals V-1 to V-20) and logo rounds: seven directions, then three combining Josh's favourites; Josh chose the Overlay b (#93) and locked the round-2 design sheet with every default (#94). The design lives in `docs/design/brand.md` and `docs/_scripts/brand.py`.
- #105 (PR #106) merged: the locked design document, `brand.py` as the one home of its numbers (checked in CI) and the brand assets and fonts (DECISIONS #95).

## 2026-10-27: docs phase, the theme (#107)

- A local Sphinx theme on `basic` replaces the PyData theme, with the text unchanged: top bar with the sections, sidebar, outline, pager, footer, both colour schemes, and a plot style that `brand.py` writes and the tutorials pick up through `MATPLOTLIBRC`. Checked at 390 and 1440 px in both themes (no horizontal overflow, no console errors).
- Reviews: `/simplify` with four reviewers (stylesheet order and `!important`, type sizes and layout constants into tokens and variables, sections computed once per build, the "/" key left to `doctools.js`) and `/code-review high` (search snippets need `role="main"`; phones could not open the search box; plots had no light panel in the dark theme; the root doctree cached in the pickled environment went stale; toctree URLs crashed the section list; no navigation on phones without JavaScript; focus rings clipped in scrolling boxes; the local notebook cache ignored the plot style). Decision #96 (plot ground, dpi, build settings) replaces #95's transparent plots. Deferred to the structure PR: whether the seven sections fit the top bar between 900 and 1400 px.
- #107 (PR #108) merged after CI passed.

## 2026-10-27: docs phase, the structure (#109)

- Pages moved into six captioned sections with the text unchanged (DECISIONS #97); links, README, CLAUDE.md, docstrings and CI paths follow. Found and fixed while checking: the phone menu stopped at its content's height; long inline code and the general index scrolled sideways on phones. Reviews: two `/simplify` reviewers (one footer hub for contributing and the design notes, dead code and options dropped) and `/code-review medium` (design pages had become unreachable, captions disagreed with page titles, a missing-section warning). Left for the final pass: one long inline formula on `design/spec` scrolls at 390 px; four viewcode anchors on `Auto` point nowhere.
- #109 (PR #110) merged.

## 2026-10-27: docs phase, foundations (#111) and plot colours (#112)

- #112 (PR #113) merged: plots take their colours from the matplotlib style, with a `sample_color` argument (DECISIONS #98).
- #111: one bibliography (127 checked entries; about 200 citations converted), the API page grouped by task with a coverage test (8 names were missing), a weekly link check with every DOI checked at the resolver, a writing-rules page, and the glossary, notation, defaults and warnings pages (DECISIONS #99; the notation renames are in QUESTIONS). The reviews reshaped the defaults page: run-time thresholds that were literals inside functions became named constants with `#:` comments, the page lists those, and numpydoc's PR01 check in the docs build keeps every public parameter documented (about 280 parameter entries added to docstrings). Behaviour is unchanged; fast and slow suites pass.
- #111 (PR #114) merged after #121 (PR #122): CI runs only the jobs a change touches (a `changes` job with path filters, one `ci-ok` gate, the notebook cache keyed on the package and the docs style; DECISIONS #100). A docs-only PR now runs lint and the docs build only. A parallel docs build (`-j auto`) was measured from a cold cache at 13.6 min against about 9 min serial, so it was not adopted.

## 2026-10-27/28: docs phase, the six section rewrites

- Six writers rewrote the sections in parallel worktrees, each from the content plan and the writing rules, each with its own issue, four `/simplify` reviewers and `/code-review high`. Every line was read before committing. Merged in link order: Fields (#120, PR #123), Start (#115, PR #125), Concepts (#116, PR #126), How it works (#119, PR #127), Guides (#118, PR #128) and Tutorials (#117, PR #130). Each later PR brought `main` in and repointed the links into pages the earlier ones renamed; the strict build caught the rest.
- **Start:** what binference does (one executed figure), is it right for my data, the quick start, eleven principles; the README and home page share one example, regenerated with its figure.
- **Concepts (new):** six pages in words and figures (V-1 to V-7, V-10, V-13), no code for the reader to run.
- **How it works:** the theory page split into the model (now written in the measure coordinate), point estimates (the KKT gap in its general form), priors, engines, auto and validation. The auto page computes its benchmark table and findings from `benchmarks/results/matrix.json` at build time, and CI rebuilds the docs when `benchmarks/` changes.
- **Guides:** one task per page (inputs split by type; new pages on bins, checking and troubleshooting; pitfalls folded in). Thresholds link the defaults page; numbers no committed code produces became qualitative or come from a dated benchmark run; the snippet runner leaves notebook cells to the docs build.
- **Fields:** rewritten, with ecology added; tool facts live only in the software table.
- **Tutorials:** thirteen, named by topic and grouped by input, with a new posterior-samples tutorial; execution down from about 690 s to about 516 s, every page under about 60 s; hidden asserts with tolerances from Monte Carlo error.
- Code and docstring bugs found and fixed on the way: the `from_injections` docstring now says only a selection on the truth, for fractions, is inferred jointly; the `Auto` and `infer` docstrings lost their DECISIONS citations and point to the auto page; the `MAP` docstring names `grad_norm` and the NaN `kkt_gap`; a CSS change that broke wrapping site-wide was scoped to table cells.
- Statistical findings from the reviews: the 2-D tutorial's poor coverage was binning bias (a ridge narrower than a cell), and the validation tutorial's under-coverage came from unconverged Gibbs runs; neither is a bug. Open: whether `LogisticGP` is calibrated on grids (#129).
- Process slip: PR #128 sat green overnight because the session waited for a CI notification that never came. Every open PR now has a timed check-in.
