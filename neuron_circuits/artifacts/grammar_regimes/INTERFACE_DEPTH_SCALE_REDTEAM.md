# Calibrated depth boundary of the grammar answer interface

At Layers 1–3, generic answer deltas are norm matched or discovery-margin calibrated before held-out semantic evaluation.

## Locked hypotheses

- **B0_layer1_contains_rule_specific_effect: supported.**
- **B1_layer2_is_already_a_scaled_answer_interface: not supported.**
- **B2_layer3_is_a_scaled_answer_interface: supported.**
- **B3_interface_equivalence_increases_through_depth: supported.**
- **B4_mixed_models_lack_the_upstream_rule_effect: supported.**

| rho | Layer | Semantic acc | Norm/calibrated acc | Norm/calibrated cosine | Norm/calibrated fidelity | Semantic effect energy | Norm/calibrated scale |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 1 | 0.722 | 0.811/0.694 | 0.914/0.923 | -3.827/-1.211 | 2.49e-02 | 2.10/1.90 |
| 0.0 | 2 | 0.601 | 0.863/0.616 | 0.943/0.942 | -5.287/-0.711 | 2.35e-02 | 1.74/1.24 |
| 0.0 | 3 | 0.843 | 0.937/0.776 | 0.994/0.994 | 0.870/0.948 | 2.86e-02 | 1.36/1.20 |
| 0.5 | 1 | 0.115 | 0.162/0.121 | 0.709/0.690 | -0.687/0.021 | 1.47e-01 | 3.30/2.52 |
| 0.5 | 2 | 0.160 | 0.400/0.172 | 0.927/0.932 | -0.180/0.575 | 2.23e-01 | 2.60/1.40 |
| 0.5 | 3 | 0.503 | 0.902/0.532 | 0.992/0.992 | 0.023/0.926 | 6.80e-01 | 1.25/0.67 |

Layer 0 is omitted from relative fidelity claims because its semantic patch effect is behaviorally negligible. The boundary identifies effect equivalence, not the complete rule circuit upstream of the patched activation.
