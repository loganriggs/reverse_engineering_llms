# Layer-0 QK-key versus value-identity red team

For each raw digit position, its Layer-0 key and value are replaced independently before attention. The mean control uses the exact average projection of all eight digit tokens at that physical position. The cyclic control substitutes the next digit's projection, preserving a natural projection norm distribution while making identity wrong. Only rule-irrelevant digits are altered unless the method begins with `all`. Candidate attention states are decoded into a bilinear gate, patched into the native Layer-0 execution, and followed by the exact suffix; the direct attention residual remains native so this isolates how QK/value information controls the MLP gate.

## Locked hypotheses

- **Q0_native_projected_attention_recomposes: supported.**
- **Q1_pure_rules_ignore_irrelevant_digit_identity: supported.**
- **Q2_mixed_selected_needs_irrelevant_digit_identity_mean_control: supported.**
- **Q3_mixed_selected_needs_irrelevant_digit_identity_roll_control: supported.**
- **Q4_value_identity_dominates_key_identity: supported.**
- **Q5_key_identity_is_also_material: not supported.**
- **Q6_mixed_unselected_routes_are_globally_changed: supported.**
- **Q7_mean_and_roll_controls_agree: not supported.**
- **Q8_identity_route_conclusion_transfers_templates: not supported.**
- **Q9_signal_gate_passes: supported.**

| Regime | Seed | Status | irrelevant mean K/V/both | irrelevant roll K/V/both | all mean K/V/both |
|---|---:|---|---:|---:|---:|
| pure | 0 | all | 1.000/1.000/1.000 | 1.000/1.000/1.000 | 1.000/0.999/0.999 |
| pure | 101 | all | 1.000/1.000/1.000 | 1.000/1.000/1.000 | 1.000/0.950/0.950 |
| pure | 103 | all | 1.000/0.999/0.999 | 1.000/1.000/1.000 | 1.000/0.994/0.994 |
| mixed | 0 | unselected | 0.985/0.675/0.676 | 0.991/0.311/0.312 | 0.964/0.274/0.275 |
| mixed | 0 | selected | 0.980/0.286/0.297 | 0.989/-0.064/-0.082 | 0.957/-0.070/-0.106 |
| mixed | 101 | unselected | 0.980/0.752/0.742 | 0.970/0.541/0.437 | 0.963/0.458/0.425 |
| mixed | 101 | selected | 0.971/0.473/0.494 | 0.939/0.124/0.145 | 0.941/-0.069/-0.068 |
| mixed | 104 | unselected | 0.975/0.687/0.701 | 0.974/0.359/0.249 | 0.949/0.405/0.416 |
| mixed | 104 | selected | 0.949/-0.129/-0.074 | 0.946/-0.203/-0.070 | 0.901/-0.251/-0.212 |

Key replacement changes the attention pattern and therefore every source coefficient; value replacement changes transported content while retaining native routing. Neither intervention removes the same digit's influence through later layers or non-final token states. The result is specific to the Layer-0 final-position MLP path, not a claim about the model's total use of a token.
