# Full-effect calibration of the grammar answer verb

One scalar per transition/layer is chosen on discovery identities to maximize complete softmax-Fisher fidelity, then frozen for held-out identities and template 6.

## Locked hypotheses

- **F0_layer1_full_effect_is_a_scaled_answer_verb: not supported.**
- **F1_layer2_full_effect_is_a_scaled_answer_verb: supported.**
- **F2_layer3_full_effect_is_a_scaled_answer_verb: not supported.**
- **F3_fisher_scale_generalizes: supported.**
- **F4_rule_specific_effect_remains_upstream_of_boundary: not supported.**

| Layer | Semantic/answer accuracy | Discovery cosine/fidelity | Held-out cosine/fidelity | Scale |
|---:|---:|---:|---:|---:|
| 1 | 0.722/0.588 | 0.949/0.902 | 0.925/0.854 | 1.88 |
| 2 | 0.601/0.570 | 0.924/0.876 | 0.940/0.829 | 1.16 |
| 3 | 0.843/0.734 | 0.993/0.968 | 0.994/0.973 | 1.18 |

Full-effect calibration is stricter than target-class calibration but still only tests a one-dimensional effect ray. Failure can reflect state-dependent scale or a genuinely different direction; the discovery/held-out gap distinguishes them partially.
