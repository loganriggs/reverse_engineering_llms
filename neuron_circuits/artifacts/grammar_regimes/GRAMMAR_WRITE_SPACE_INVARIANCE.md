# Symmetry-aware cross-initialization MLP-write invariance

The alignment and transferred semantic edits are expressed in each MLP's residual write space, quotienting out bilinear channel permutation, scaling, and null directions.

## Locked hypotheses

- **W0_write_alignment_is_adjudicative: supported.**
- **W1_semantic_write_is_initialization_invariant: not supported.**
- **W2_same_rule_write_beats_wrong_rule: not supported.**
- **W3_write_space_is_more_canonical_than_channels: not supported.**
- **W4_final_write_matches_target_answer_interface: supported.**
- **W5_activation_inverse_is_exact: supported.**

Maximum write realization relative error: `2.20e-06`.

| Layer | Write R2 mean/min | Raw-channel R2 mean/min | Own/mapped/wrong accuracy | Mapped/wrong fidelity | Target-answer fidelity | Adjudicative |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | -2.187/-19.092 | -3.609/-19.153 | 0.537/0.147/0.043 | -149.968/-66.770 | -2.135 | no |
| 2 | 0.093/-0.440 | -0.642/-1.229 | 0.651/0.354/0.072 | -1.460/-0.303 | -0.446 | no |
| 3 | 0.624/0.423 | 0.482/0.100 | 0.683/0.460/0.028 | 0.883/0.213 | 0.898 | yes |

A write-space success would locate the reusable object after the channel basis but before residual-stream transport. A failure with an exact inverse cannot be blamed on MLP channel null directions.
