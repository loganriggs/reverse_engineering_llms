# Same-codebook initialization invariance

The grammar, data manifest, and random answer permutation are fixed; only initialization and training order differ. Alignment excludes the tested relation.

## Locked hypotheses

- **S0_same_codebook_alignment_is_adjudicative: not supported.**
- **S1_rule_edit_is_initialization_invariant: not supported.**
- **S2_same_rule_beats_wrong_rule: not supported.**
- **S3_final_effect_matches_target_answer_interface: not supported.**
- **S4_random_codebooks_break_linear_invariance: not supported.**

| Layer | Alignment R2 mean/min | Own/mapped/wrong accuracy | Mapped/wrong fidelity | Target-answer fidelity | Different-codebook R2/fidelity | Adjudicative |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | -3.609/-19.153 | 0.537/0.108/0.082 | -87.216/-124.927 | -2.135 | -2.407/-0.261 | no |
| 2 | -0.642/-1.229 | 0.651/0.349/0.079 | -1.296/-0.050 | -0.446 | -2.001/-0.835 | no |
| 3 | 0.482/0.100 | 0.683/0.404/0.027 | 0.846/0.194 | 0.898 | -0.923/0.174 | no |

The comparison isolates task-codebook changes from initialization changes. Different training durations remain a nuisance variable, but all included models independently pass the same behavioral recall and template-transfer gate.
