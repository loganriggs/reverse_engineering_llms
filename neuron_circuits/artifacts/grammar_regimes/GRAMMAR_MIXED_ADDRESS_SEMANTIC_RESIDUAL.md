# Semantic-prototype versus residual address interventions

**Correction:** the first execution covered only relations 0–3 in the within-cell swap, leaving half the panel unchanged and producing a spurious median near 1. This rerun covers all eight relations, asserts full coverage, and retains the original locked hypotheses.

Each state is executed through the original bilinear tensor. `prototype_only` uses an oracle relation-answer mean learned on a disjoint discovery panel. Residual swaps therefore test precise state construction, not whether the answer can be guessed without the prompt.

## Locked hypotheses

- **D0_native_tensor_is_exact: supported.**
- **D1_semantic_prototype_is_an_answer_actuator_not_full_mechanism: not supported.**
- **D2_residual_alone_does_not_specify_the_answer: not supported.**
- **D3_within_cell_residual_identity_is_not_required_for_answer: supported.**
- **D4_residual_identity_is_required_for_precise_effect: supported.**
- **D5_cross_cell_residuals_are_more_disruptive: supported.**
- **D6_half_residual_preserves_the_answer: not supported.**
- **D7_wrong_answer_prototype_flips_control: not supported.**
- **D8_rule_and_arbitrary_facts_use_the_same_decomposition: not supported.**
- **D9_template_transfer_is_stable: not supported.**
- **D10_signal_gate_passes: supported.**

| Seed | Status | proto fid/acc | residual fid/acc | within-swap fid/acc | global-swap fid/acc | half fid/acc | wrong-proto fid/acc |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | unselected | 0.690/0.982 | 0.674/0.859 | 0.662/0.966 | 0.158/0.597 | 0.887/0.985 | 0.392/0.626 |
| 0 | selected | 0.246/0.537 | 0.887/0.962 | 0.420/0.925 | -0.516/0.224 | 0.668/0.768 | 0.819/0.890 |
| 101 | unselected | 0.717/0.977 | 0.671/0.842 | 0.651/0.957 | 0.161/0.564 | 0.884/0.981 | 0.418/0.589 |
| 101 | selected | 0.219/0.544 | 0.889/0.932 | 0.426/0.931 | -0.530/0.217 | 0.667/0.786 | 0.843/0.885 |
| 104 | unselected | 0.750/0.982 | 0.548/0.825 | 0.626/0.971 | 0.251/0.740 | 0.907/0.990 | 0.155/0.583 |
| 104 | selected | 0.397/0.711 | 0.848/0.898 | 0.427/0.926 | -0.421/0.332 | 0.726/0.836 | 0.781/0.822 |

Accuracy asks whether the final answer survives; effect fidelity asks whether the full eight-answer causal correction is reproduced. Their divergence is itself mechanistic evidence: an answer actuator can be behaviorally sufficient while omitting much of the native computation.
