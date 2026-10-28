# Leave-one-relation-out cross-seed causal invariance

Each source-to-target activation map is fit without examples from the relation whose semantic edit is later translated and patched.

## Locked hypotheses

- **L0_unseen_relation_alignment_is_adjudicative: not supported.**
- **L1_rule_edit_generalizes_beyond_alignment_relations: not supported.**
- **L2_same_rule_beats_norm_matched_wrong_rule: not supported.**
- **L3_final_effect_matches_target_answer_interface: not supported.**
- **L4_semantic_correspondence_exceeds_source_label_identity: not supported.**

| Layer | Alignment R2 mean/min | Own/mapped/wrong accuracy | Mapped/wrong fidelity | Source/target-answer fidelity | Mapped/target-answer cosine | Adjudicative |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | -2.407/-4.295 | 0.718/0.044/0.043 | -0.261/-2.597 | -0.059/-0.152 | 0.615/0.885 | no |
| 2 | -2.001/-3.898 | 0.603/0.021/0.050 | -0.835/-0.715 | -0.098/-0.974 | 0.376/0.921 | no |
| 3 | -0.923/-2.060 | 0.843/0.006/0.017 | 0.174/0.200 | 0.085/0.906 | 0.427/0.992 | no |

The alignment gate is a minimum over every ordered seed pair and omitted relation. Passing therefore cannot be driven by one easy pair. The target-answer control, unlike mapped source label identity, uses the target model's own output codebook.
