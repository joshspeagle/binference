# pastry: vision and principles (agreed in step 1.1, 2026-10-25)

Draft replacement for spec §1 and the README/docs opening. Field-neutral (#37).

## Vision (Josh, 2026-10-25)

pastry should be as easy to pick up as scikit-learn: a novice gets a correct,
calibrated answer without understanding the internals, while an expert can
tune every knob. It should be blazing fast. It should make doing the right
thing so easy that *not* doing it feels almost criminal.

## What pastry does (draft)

Many objects, each measured noisily: per-object likelihoods, posteriors under
some prior, samples, binned measurements with a response, or classifier
scores. pastry infers the distribution of the population they come from,
with honest uncertainty, and each object's posterior under that population.
The same problem appears as population / n(z) inference in astronomy,
unfolding in particle physics, g-modelling (NPMLE) in statistics,
quantification under label shift in machine learning, and mixed-stock
analysis in ecology; pastry handles all of them through one model and one
API.

Value (rewritten, #33): correct input conventions, calibrated uncertainty,
built-in validation and speed, plus what existing implementations leave out:
inputs whose own uncertainty is propagated (selection, responses, calibration
sets), bases that remove the flat-within-bin approximation, input estimators
with error control, and defaults chosen automatically from evidence.

## Principles (#34)

1. Correct by construction: users state what they pass (likelihood or
   posterior, which interim prior, mass or density, which measure); the API
   refuses ambiguity. (#15, #21, #24, #26, #32)
2. Easy things one line, hard things possible: `infer` → model + engines →
   raw log densities; modular pieces usable on their own.
3. Exact by default; every approximation labelled, including approximations
   of the model (flat within bins, Monte Carlo inputs), not only of the
   sampler.
4. Self-validating: checks of the sampler, the inputs and the approximations
   run automatically (#6, #12, #18, #29, #30, #31).
5. Lightweight core: NumPy, SciPy, threadpoolctl; extras optional; JAX later.
6. One result object (#35): a fit returns draws and a best fit together by
   default; either can be switched off.
7. Expose everything, sensible defaults (#16).
8. Adaptive, evidence-based defaults, reported to the user (#14).
9. The user's bins define the problem; internals adapt (#17, #20, #21).
10. Every uncertain input is modelled jointly by default; fixing it is an
    explicit choice (#10, #28, #30).
11. Fast: performance is a feature (benchmarks in 4.6; speed budgets in
    tests).
