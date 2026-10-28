# Final-layer grammar rule versus generic answer interface

Class-prototype activation deltas learned from unrelated facts are compared with paired semantic-rule deltas on disjoint identities and an unseen template.

## Locked hypotheses

- **A0_generic_answer_interface_exists_in_all_regimes: not supported.**
- **A1_answer_interface_is_native_axis_privileged: not supported.**
- **A2_pure_semantic_patch_contains_rule_specific_computation: not supported.**
- **A3_pure_semantic_headline_is_explained_by_output_interface: not supported.**
- **A4_mixed_models_keep_output_interface_after_semantic_transfer_fails: supported.**

| rho | Runs | Answer full/native/rotated/random accuracy | Semantic full/native | Matched answer full/native | Effect cosine | Support overlap |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 3 | 0.492/0.339/0.085/0.008 | 0.843/0.741 | 0.415/0.339 | 0.992 | 0.676 |
| 0.5 | 2 | 0.783/0.427/0.373/0.025 | 0.503/0.291 | 0.800/0.427 | 0.992 | 0.354 |
| 1.0 | 2 | 0.819/0.443/0.483/0.043 | n/a | n/a | n/a | n/a |

A generic final answer interface is an interpretable and useful mechanism, but it is downstream of the computation that determined the answer. If it explains the semantic patch, earlier layers—not the final patched channels—must carry the rule-specific evidence.
