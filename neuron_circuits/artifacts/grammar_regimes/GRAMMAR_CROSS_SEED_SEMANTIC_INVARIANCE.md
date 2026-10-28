# Cross-seed causal invariance of grammar computations

A global ridge map is fitted on unrelated shared prompts, frozen, and used to translate semantic activation edits between independently trained pure-rule models with different answer permutations.

## Locked hypotheses

- **X0_alignment_is_adjudicative: supported.**
- **X1_semantic_computation_is_cross_seed_reusable: supported.**
- **X2_same_rule_beats_wrong_rule: supported.**
- **X3_semantics_adds_more_than_source_answer_identity: supported.**
- **X4_final_layer_is_a_shared_output_interface: not supported.**

| Layer | Alignment R2 mean/min | Own/mapped/wrong/answer accuracy | Mapped/wrong/answer effect fidelity | Mapped/wrong/answer cosine | Adjudicative |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0.665/0.601 | 0.718/0.652/0.016/0.013 | 0.780/-7.063/0.087 | 0.995/0.423/0.532 | yes |
| 2 | 0.654/0.621 | 0.603/0.542/0.003/0.007 | 0.964/0.086/0.030 | 0.994/0.590/0.518 | yes |
| 3 | 0.611/0.514 | 0.843/0.604/0.002/0.003 | 0.963/0.125/0.202 | 0.996/0.387/0.534 | yes |

A failed causal transfer is treated as evidence against global-linear invariance only where held-out state alignment passes. Wrong-rule vectors are norm matched after mapping; this prevents simple edit magnitude from winning the control.
