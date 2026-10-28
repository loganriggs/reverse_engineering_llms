# MLP-write codebook factorial

The symmetry-aware leave-one-relation-out assay is repeated across models with independently randomized answer permutations.

## Locked hypotheses

- **C0_different_codebooks_break_unseen_relation_write_alignment: supported.**
- **C1_late_invariance_is_codebook_relative: supported.**
- **C2_target_internal_answer_interface_survives: supported.**
- **C3_wrong_rule_control_remains_noncausal: supported.**

| Layer | Different-codebook R2 mean/min | Same-codebook R2 mean/min | Different/same mapped fidelity | Mapped/wrong accuracy | Target-answer fidelity | Adjudicative |
|---:|---:|---:|---:|---:|---:|:---:|
| 1 | -1.568/-3.964 | -2.187/-19.092 | -0.774/-149.968 | 0.043/0.039 | -0.152 | no |
| 2 | -1.507/-3.232 | 0.093/-0.440 | -0.157/-1.460 | 0.028/0.030 | -0.974 | no |
| 3 | -0.490/-1.029 | 0.624/0.423 | 0.103/0.883 | 0.016/0.009 | 0.906 | no |

The random output convention is part of the implemented task. A codebook-relative result means the final interface is reusable within that convention, not that it lacks structure.
