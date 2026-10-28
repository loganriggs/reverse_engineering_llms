# Is the compact final gate code an answer quotient?

The answer-centroid span is learned on a disjoint discovery panel and has rank at most seven. All gate candidates use the native down projection and exact suffix. Oracle prototypes are actuator diagnostics, not explanations of how the prompt retrieves its answer.

## Locked hypotheses

- **E0_native_gate_is_exact: supported.**
- **E1_answer_centroid_span_is_a_causal_quotient: supported.**
- **E2_answer_span_explains_most_of_rank8_pca: supported.**
- **E3_answer_prototype_is_behaviorally_sufficient: supported.**
- **E4_answer_prototype_omits_precise_calibration: supported.**
- **E5_relation_conditioning_adds_little_behavioral_control: supported.**
- **E6_relation_conditioning_adds_precise_effect_information: not supported.**
- **E7_shuffled_answer_span_is_not_causal: not supported.**
- **E8_within_answer_residual_is_not_an_answer_code: supported.**
- **E9_rule_and_arbitrary_facts_share_answer_quotient: supported.**
- **E10_template_transfer_is_stable: supported.**
- **E11_signal_gate_passes: supported.**

| Seed | Status | answer variance | PCA8 fid | answer-span fid/acc | answer proto fid/acc | relation+answer proto fid/acc | residual fid/acc | shuffled span |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | unselected | 0.090 | 0.946 | 0.963/0.978 | 0.830/1.000 | 0.839/1.000 | 0.090/0.341 | 0.363 |
| 0 | selected | 0.097 | 0.939 | 0.972/0.976 | 0.672/1.000 | 0.674/1.000 | 0.160/0.137 | 0.339 |
| 101 | unselected | 0.099 | 0.945 | 0.956/0.984 | 0.840/1.000 | 0.838/0.999 | -0.046/0.295 | 0.310 |
| 101 | selected | 0.103 | 0.933 | 0.962/0.984 | 0.645/1.000 | 0.644/1.000 | 0.120/0.103 | 0.221 |
| 104 | unselected | 0.104 | 0.897 | 0.914/0.982 | 0.823/1.000 | 0.813/1.000 | -0.013/0.432 | 0.288 |
| 104 | selected | 0.091 | 0.914 | 0.964/0.968 | 0.718/1.000 | 0.717/1.000 | 0.155/0.184 | 0.202 |

A successful answer quotient explains what the dense gate code controls. It does not explain the upstream lookup that chooses a point in that quotient, and prototype success cannot be used as a non-oracle retrieval result.
