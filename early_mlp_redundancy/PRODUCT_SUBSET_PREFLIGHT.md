# Executable product-subset replacements pass training-only preflight

M92 implements and verifies the frozen 119-candidate causal comparison. It does not yet report held-out behavioral or causal results.

## What is executable now

The predictor takes the normalized MLP1 input and computes only the retained native bilinear products. It maps their sum either to the native output space or through one of the 16 learned output dictionaries. The native Down_bias remains a separate addition by the caller; projected affine offsets remain fixed.

All 17 spaces have a full-product reference and six half-product variants: three fixed rankings (energy and two random permutations), each raw or multiplied by its frozen M90 training scalar. That gives 119 candidates. No candidate is selected using transfer performance.

The interface's support/code tensors describe retained **native products**, not semantic features. Half retention computes 2304 products; the projected output dictionary has 128 directions. These are different dimensions and will be reported separately.

## Verification performed

- All 119 predictors agree with independent float64 input/product algebra on training document 0; maximum relative write discrepancy is 6.12e-7 against the frozen tolerance 1e-5.
- All 17 full references agree with the corresponding native or existing projected-write calculation; maximum relative discrepancy is 2.80e-7.
- All 119 predictors run while native Left, Right, and Down module access is guarded, and an unusable object is passed as the optional native-write argument. They do not consume a precomputed native write.
- The native full-product control replaces the MLP forward itself, with native module access blocked. Its output KL against the original model is 0, meeting the fixed 1e-8 criterion.
- All 2142 training-pair constructions pass: 119 candidates × two endpoints × source/final/identity × three background/change corners. Donor-minus-recipient values and unchanged positions are checked independently.
- Four prespecified controls receive actual native-forward smoke tests: native full, native energy-half raw, native energy-half scaled, and dense seed71 KL-only full projection. All 72 conditions are finite, and all 24 identity conditions replay exactly.
- A separate structural audit verifies all 119 manifest entries, exact training-derived subsets/scalars, source state references, and coverage.

There were 84 native forwards. No fitting or evaluation-set scoring occurred. The native-forward smoke tests cover four controls, not all 119 candidates; the full held-out evaluation and response-vector audits remain required.

## Why this matters conceptually

The substitution is now an actual input-to-output computation rather than an analysis that requires the original MLP write first. It can test whether the constructive distribution and scalar-compensated substitution observed geometrically preserve downstream behavior. Correct implementation is a prerequisite, not evidence that those causal properties hold.

The next frozen evaluation retains all 119 candidates on 32 natural-text documents, 56 whole-task pairs and 14 source-group pairs. It separates reconstructed backgrounds from reconstructed task changes at source and final positions, retains identity controls and weak/ineligible native examples, and reports natural retention, task prediction, local reconstruction and causal fidelity separately. Whole-task training exclusions remain unchanged. No further local subset search or scalar tuning precedes it.

Evidence: [candidate manifest](validation/product_subset_models.json), [numerical and native preflight](validation/product_subset_preflight.json), [structural audit](validation/product_subset_preflight_audit.json), [evaluation execution receipt](validation/product_subset_causal_execution.json), [predictor implementation](product_subset_predictor.py).
