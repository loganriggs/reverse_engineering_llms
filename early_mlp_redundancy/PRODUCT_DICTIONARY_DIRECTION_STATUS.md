# Fixed-direction diagnostic: training complete; held-out derivative accounting unresolved

M97 follows the frozen coefficient direction from each KL-only fit to its joint-supervised fit. It measures whether this correction has opposing local effects across training and held-out examples, or whether downstream curvature changes its finite effect. No model is trained or selected.

The training-only implementation preflight passed its declared criteria: four endpoint directions, twelve finite-difference steps, independent NumPy affine tangents, and exact saved endpoint output/loss replay (196 native forwards). Tangent discrepancies were below 6.5e-8 relative. Float32 endpoint secants differ from the ideal tangent by roughly 0.014–0.016%, so values are anchored to their actual saved endpoints while derivatives use the affine tangent.

The finite-difference acceptance criterion retains the earlier absolute-plus-relative tolerance. For these small slopes (around −0.00017), observed finite-difference discrepancies ranged from 4e-6 to 1e-4; the absolute tolerance dominates. Passing this check does **not** certify 1e-7 derivative accuracy.

Full training-direction scoring completed all 528 endpoint records and 3,684 forwards, exactly replaying all 4,740 prior response vectors. The audit passed case/model/schedule inventory, every recorded K/B/response value, identity zeros, independent NumPy tangent-energy checks, and weighted derivative accounting. Maximum tangent-energy discrepancy was 5.21e-8 relative. Training comparisons are saved, but no cross-population interpretation is made yet.

The source-group run completed and checkpointed seed 71 (56 endpoint records, 392 forwards, 504 response replays). It then failed the derivative-accounting assertion for seed 72, first capital-recall pair, endpoint 1, at the KL-only anchor. A targeted reproduction found:

| Quantity | Value |
|---|---:|
| Derivative of balanced loss B directly | −0.004382308573 |
| Weighted sum of separate branch derivatives | −0.004382470435 |
| Absolute difference | 1.6186e-7 |
| Frozen allowed difference | 1.4382e-7 |

The discrepancy is about 0.0037% of the derivative. Endpoint outputs still replayed exactly. Float32 accumulation is a plausible explanation, but it has not yet been established; the old criterion and failed evidence remain unchanged. The completed source prefix is preserved, and uncommitted work is not counted as zero.

Next: one case-specific precision diagnostic to separate scalar tangent reduction from accumulation through the native model's reverse-mode derivatives. Preserve the original failure and compare implementations before deciding whether a numerical repair is justified. Do not restart successful training scoring, treat a tolerance change as a scientific result, or interpret the incomplete held-out cohort. The broader M97 stopping rule remains one fixed-path synthesis before returning to a concrete causal-coverage or compatibility experiment.

Evidence: [preflight](validation/product_dictionary_direction_preflight.json), [training audit](validation/product_dictionary_direction_training_audit.json), [training comparison](validation/product_dictionary_direction_training_comparison.json), [source failure receipt](validation/product_dictionary_direction_source_failure.json).
