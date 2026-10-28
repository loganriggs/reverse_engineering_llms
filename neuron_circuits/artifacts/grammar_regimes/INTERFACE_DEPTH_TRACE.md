# Depth trace from rule-specific computation to answer interface

At every MLP layer, paired semantic-rule deltas and unrelated answer-class prototype deltas are patched into the same held-out recipients.

## Locked hypotheses

- **D0_pure_rule_direction_is_specific_upstream: not supported.**
- **D1_pure_rule_converges_to_output_interface: supported.**
- **D2_generic_answer_interface_strengthens_with_depth: supported.**
- **D3_mixed_arbitrary_answer_interface_emerges_earlier: not supported.**
- **D4_native_axis_privilege_is_regime_and_depth_specific: supported.**

| rho | Layer | Generic full/native/rotated acc | Semantic full/native/rotated acc | Matched answer acc | Effect cosine/fidelity | Support overlap |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0 | 0.001/0.000/0.000 | 0.000/0.000/0.000 | 0.002 | -0.281/-28224178950.857 | 0.341 |
| 0.0 | 1 | 0.216/0.090/0.038 | 0.722/0.486/0.187 | 0.288 | 0.868/-0.322 | 0.450 |
| 0.0 | 2 | 0.384/0.286/0.086 | 0.601/0.365/0.103 | 0.437 | 0.924/-0.816 | 0.576 |
| 0.0 | 3 | 0.492/0.372/0.084 | 0.843/0.741/0.077 | 0.415 | 0.992/0.899 | 0.676 |
| 0.5 | 0 | 0.005/0.003/0.001 | 0.133/0.097/0.085 | 0.019 | 0.417/-0.279 | 0.303 |
| 0.5 | 1 | 0.007/0.003/0.003 | 0.115/0.072/0.052 | 0.035 | 0.645/0.160 | 0.256 |
| 0.5 | 2 | 0.034/0.010/0.011 | 0.160/0.092/0.086 | 0.109 | 0.932/0.565 | 0.295 |
| 0.5 | 3 | 0.783/0.323/0.373 | 0.503/0.291/0.265 | 0.800 | 0.992/0.495 | 0.354 |
| 1.0 | 0 | 0.005/0.002/0.001 | n/a | n/a | n/a | n/a |
| 1.0 | 1 | 0.006/0.003/0.003 | n/a | n/a | n/a | n/a |
| 1.0 | 2 | 0.030/0.011/0.010 | n/a | n/a | n/a | n/a |
| 1.0 | 3 | 0.819/0.443/0.472 | n/a | n/a | n/a | n/a |

Effect cosine asks whether the two patches implement the same downstream verb; accuracy and magnitude determine whether that verb is driven strongly enough. A final shared direction does not locate the upstream rule computation.
