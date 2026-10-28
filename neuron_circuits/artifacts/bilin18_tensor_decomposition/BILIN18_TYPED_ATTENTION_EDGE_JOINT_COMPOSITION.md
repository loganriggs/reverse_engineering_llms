# Joint composition of typed attention-edge programs

Discovery-frozen typed-edge programs are executed at every token position in progressively deeper prefixes of Layers 0--5. Each later layer sees the states produced by the earlier approximate programs. The remaining suffix is native. KL recovery is normalized against deleting all attention writes in the same prefix.

Maximum native-branch logit error: `5.58e-05`.

## Locked hypotheses

- **J0_native_branch_matches_checkpoint: supported.**
- **J1_fine128_composes_through_six_layers: supported.**
- **J2_fine32_retains_nontrivial_six_layer_behavior: supported.**
- **J3_larger_fine_program_dominates: supported.**
- **J4_broad_closure_improves_joint_program: supported.**
- **J5_programs_do_not_depend_on_native_intermediate_states: supported.**

| Approximate layers | Fine32 | Fine128 | Drop category | Drop distance | Drop head | Fine128 CE / top1 |
|---|---:|---:|---:|---:|---:|---:|
| 0--0 | 0.915 | 0.985 | 0.962 | 0.954 | 0.966 | +0.0128 / 0.957 |
| 0--1 | 0.847 | 0.991 | 0.976 | 0.967 | 0.952 | +0.0285 / 0.923 |
| 0--2 | 0.740 | 0.985 | 0.971 | 0.913 | 0.866 | +0.0570 / 0.900 |
| 0--3 | 0.676 | 0.989 | 0.986 | 0.905 | 0.839 | +0.1097 / 0.867 |
| 0--4 | 0.619 | 0.983 | 0.976 | 0.863 | 0.801 | +0.1852 / 0.816 |
| 0--5 | 0.583 | 0.978 | 0.956 | 0.823 | 0.789 | +0.1839 / 0.824 |

## Guardrails

This is closed-loop composition: QK scores, values, RMSNorm, and bilinear MLPs are recomputed from each approximation's own states. It is not a polynomial expansion. The no-attention denominator changes with prefix depth, so recovery values compare approximation quality within a prefix, not absolute causal importance across prefixes. The token-category vocabulary remains human-declared, and every approximate layer alters all destinations, making this stricter than the prior one-destination suffix test.
