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
- C7c (DECISIONS #77): measured before building. The fused numba kernel and float32 gave no speed on four cores (NumPy with BLAS threads is as fast), so #54 and #55 are revised to what the measurements support: one NumPy kernel, float64, and a measured dense/sparse threshold (10%, down from 30%). The float32-for-memory option is parked in QUESTIONS.
