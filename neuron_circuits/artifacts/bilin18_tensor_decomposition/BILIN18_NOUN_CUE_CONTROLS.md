## H1002 — Frozen noun contrast follows actions more than background tools, with limited novel-verb transfer

Previous turn was progress: H1001 found compact noun-contrast transfer and exposed fragment vocabulary dependence. This turn tests the frozen nounrank1 direction with crossed background tools and actions. Eighty unique prompts,72paired comparisons,80native forwards plus1008intervention forwards=1088total. No new fitting, prompt replacement, or revised answer groups.

Four subjects cross background knife/key with cut/unlocked, sliced/opened, ate/entered, cooked/locked. Background appears in a separate sentence: A knife lay nearby. The pilot cooked the. Action pairs hold background fixed; nuisance pairs swap background while action stays fixed. Two additional coupled novel-tool/action contrasts use fork/ate versus card/entered and spoon/tasted versus code/accessed. These latter probes are not factorial. All pairs share the final determiner. Seven arms: identity, full donor, frozen nounrank1, oldtask128, their129dimensional union, and paired +/- residual noise norm-matched to the rank1 edit.

The native gate remains four pairs, >=75%both endpoints category-correct, >=50%intended full-vocabulary top1. Five of ten action cohort/condition cells pass. Failed cells are diagnostic only; no individual correctness filtering.

| Action cohort/background | Native gate | Frozen rank1 effect/full | Rank1 relative full-patch KL |
|---|---|---:|---:|
| cut/unlocked,knife |pass|0.686|0.278|
| cut/unlocked,key |fail|0.702|0.417|
| sliced/opened,knife |pass|0.943|0.511|
| sliced/opened,key |pass|1.053|0.555|
| cooked/locked,knife |pass|0.359|0.539|
| cooked/locked,key |pass|0.410|0.685|
| ate/entered,both backgrounds |fail|0.292–0.296|0.757–0.789|
| fork/ate versus card/entered |fail|0.268|0.854|
| spoon/tasted versus code/accessed |fail|0.277|0.955|

Effect/full is a ratio of mean signed answer-margin changes, not an additive explanation fraction. Relative KL compares full-donor-patched output to component-patched output and normalizes by identity. Overshoot above1 on an answer contrast is not whole-output equivalence. Strong answer preservation still leaves substantial distribution mismatch.

Frozen rank1 mean absolute margin changes from background swaps are0.048,0.060,0.027,0.062 across the four verb cohorts, versus action-change signed means roughly0.4–0.8. The direction is more sensitive to action contrasts than these background substitutions, but is not nuisance-invariant. Matched random-noise absolute effects are~0.005–0.017; observed nuisance sensitivity is meaningful enough to retain as a limitation. Full-write background swaps themselves change margins by0.052–0.120, so background is not causally irrelevant to the model.

Native failures cannot be equated with absence of the intended behavior. Inspection of the first subject shows sensible omitted top predictions: ate food, entered cockpit, tasted sauce, accessed website, among others. Some outputs also repeat the nearby tool, e.g. cooked knife. The fixed food/access token lists are incomplete and the category framing does not cover all actions cleanly. Do not broaden lists after seeing these results and then call this a passed held-out test. A future category redesign needs fresh independent examples.

Interpretation: the existing direction captures a reusable but narrow action-conditioned food/access contrast and retains lexical/background effects. Novel cooking/locking transfer of36–41% weakens a generic noun-expectation interpretation. This control round supports keeping it as a provisional contrast anchor, not a universal semantic module. Fragmentrank4 remains source-specific and should not become a general morphology label.

Stopping criterion for this side investigation has been reached: we have a fixed direction, demonstrated transfer and explicit confounds sufficient to define a narrow supervision target. Next return to joint representation evaluation: compare old task span plus this frozen direction with an equal-size random residual extension, and evaluate old-task retention, competent new contrasts and natural-text fidelity. Use existing discovery-fitted geometry; do not train on these control/evaluation prompts. This gives the featurizer loop a concrete deliverable without prolonging the single-task mechanism detour. Keep whole-state reconstruction distinct from donor-difference fidelity and retain old task holdouts.

Audits:1088/1088forwards,1008unique edits, identity logit error2.62e-5. Saved-logit independent margin and referenceKL replay errors0, frozen-coordinate replay error0. All10native gates independently replayed; no exact prompt overlap with H999. Source/basis/manifest hashes unchanged; no fitting or reserved-test access. Natural-document pool remains26,112. H961 context-bridge evaluation pending; broad goal active.
