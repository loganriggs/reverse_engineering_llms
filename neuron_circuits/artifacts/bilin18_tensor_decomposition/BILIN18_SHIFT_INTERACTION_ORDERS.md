## H644 — Low-order composition suffices for small shifts; broad-format higher orders are partly an amplitude effect

H643 was progress: the reusable context-shift effect was distributed across later layers and could not be reproduced closely by adding isolated layer effects. H644 resolves which interaction orders account for that difference.

At fixed native operating states, scaling each learned MLP-context shift by a shared scalar s makes each layer's tangent map affine: A_l+s K_l. A_l includes native residual scaling, attention and normalized MLP derivative; K_l is the learned bilinear-context change after the same native preceding operations. Multiplying eight such maps yields a polynomial of degree at most eight in s. This is an exact algebraic expansion of the local tangent model, not a Taylor expansion of finite activation changes or a fitted approximation.

Propagate coefficients recursively: at each layer the order-k coefficient receives A_l times its previous value plus K_l times the previous order-(k−1) coefficient. Final readout is fixed per chosen receiver/donor condition and linear in the tangent. Order1 sums all single-layer effects; order2 adds every ordered two-layer interaction; higher orders collect paths involving that many distinct shifted layers, with native operations between them. It does not identify an individual semantic circuit or selected pair.

The full coefficient sum recovers the eight-layer predictor; constant and first-order terms recover native and additive controls. Maximum relative residual discrepancy across these controls is4.59e-7. Direct computations at s=0.5 and s=−1 on the original receiver-input condition agree with polynomial evaluation within1.23e-7. These numerical checks support the algebraic interpretation.

**Pairwise terms approximate matched instruction changes well, but not broad format changes at their native shift amplitudes.** Errors below are measured relative to the full predicted shift effect, under donor incoming/readout conditions:

| Condition | Order1 | Through order2 | Through order3 | Through order4 | Through order5 |
|---|---:|---:|---:|---:|---:|
|Copy matched|0.2756|0.0479|0.0066|0.0008|0.0001|
|Copy broad|0.6457|0.3174|0.1154|0.0330|0.0069|
|Fact matched|0.3257|0.0740|0.0128|0.0016|0.0002|
|Fact broad|0.6257|0.2721|0.0869|0.0222|0.0039|

Matched pairwise errors average0.048/0.074, with individual maxima0.079/0.113 for copy/facts. Broad pairwise errors average0.317/0.272, with maxima0.467/0.588. Broad order3 still leaves means0.115/0.087; through order4 these fall to0.033/0.022, with maxima0.054/0.047. Higher orders are small but not literally absent. There was no predetermined acceptance threshold, so these values should not be converted into a categorical statement that order2 or4 is universally “sufficient.”

The pattern persists with receiver-only incoming/readout: matched pairwise errors0.048/0.076 and broad errors0.309/0.217. Broad order4 errors are0.033/0.025 in that setting. All template cells,64 prompts and four probes remain included; no order was selected for future prediction based on these observations.

**Approximating the predictor is distinct from improving its accuracy against exact donor-context replacement.** For matched copying/facts, pairwise truncation already gives exact-context effect errors0.417/0.393, close to the full predictor's0.416/0.388. The remaining error therefore mostly reflects the frozen mean-shift approximation, not omitted interaction orders. For broad copying/facts, pairwise errors against the exact intervention are0.534/0.328 versus full-predictor0.454/0.202; orders3–4 recover most of the lost fidelity. Occasional lower error than the full predictor is possible through cancellation of predictor errors and is not evidence of a more faithful context model.

**A post-result norm-matching diagnostic attributes much of the matched/broad difference to shift amplitude.** The eight-layer stacked shift Frobenius norms are21.07 matched versus47.50 broad for copying, and25.02 versus41.28 for facts. These norms depend only on frozen discovery-fitted vectors. Rescale each broad shift chain down to its family's matched norm, and each matched chain up to its broad norm, keeping its direction and per-layer relative sizes unchanged. Evaluate the already-computed full polynomial at that amplitude and compare truncations with that same scaled full effect.

| Rescaled intervention | Pairwise error at original amplitude | Pairwise error after norm exchange | Order3 error after norm exchange |
|---|---:|---:|---:|
|Copy matched, scaled up|0.0479|0.2231|0.0683|
|Copy broad, scaled down|0.3174|0.0632|0.0103|
|Fact matched, scaled up|0.0740|0.1964|0.0546|
|Fact broad, scaled down|0.2721|0.1152|0.0220|

Downscaling broad contrasts reduces pairwise error to0.063 for copying and0.115 for facts; upscaling matched contrasts raises it to0.223/0.196. Thus high-order remainder is not a categorical property of broad wording. Shift magnitude accounts for a substantial part of the regime difference, with remaining differences attributable to direction, distribution across layers, native operator geometry and endpoint conditioning. Equal Euclidean context norm is not equal functional operator norm, so this diagnostic does not isolate amplitude perfectly or assign an explained fraction.

The rescaled interventions are artificial counterfactuals, not observed alternative prompts. No new model parameters are fitted and no actual donor target exists for the scaled amplitude; their role is to test the mathematical mechanism behind interaction order. The original s=1 result remains the prospective predictor evaluated in H642.

The working explanation is a sequence of reusable context-conditioned operator changes whose products generate higher-order effects. Small changes admit useful low-order approximations; larger changes require more orders. This is consistent with ordinary composition of varying linear maps and does not require postulating an indivisible eight-layer semantic feature. Neither the polynomial nor its coefficients are a whole-model canonical form or a proof of computational minimality.

Next investigate the dominant remaining source of predictive error: whether held-out context changes differ from the learned mean mostly in magnitude or also in direction. Compare fixed mean shifts with oracle per-layer scalar gains along those same shifts, using held-out donor contexts only for this diagnostic. If gain correction removes most functional error, pursue a receiver-based gain predictor; if substantial error remains, identify item-dependent shift directions. Label oracle coefficients clearly and retain the already validated frozen predictor as the actual predictive baseline.

Audits:512 cached attention-stage reconstructions;1824 batched JVP calls; no new whole-model forwards; constant/additive/full and off-unit-amplitude controls verified; all source/runner/frozen-model hashes unchanged; finite metrics. Norm-exchange audit is separately hashed and uses the existing polynomial without new forwards. No new natural-text documents; FineWeb remains26,112. Broad goal active and incomplete.
