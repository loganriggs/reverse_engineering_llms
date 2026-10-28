# Pure-rule versus mixed-memory finite gate profiles

Every cell uses a literal gate patch and exact suffix recomputation. Pure and mixed checkpoints share the data manifest but are independently trained; paired rows are comparison slots, not neuron alignments.

## Locked hypotheses

- **R0_native_patches_are_exact: supported.**
- **R1_arbitrary_load_globally_broadens_middle_computation: supported.**
- **R2_pure_semantics_emerge_earlier: supported.**
- **R3_both_regimes_share_a_final_answer_quotient: supported.**
- **R4_pure_middle_program_has_lower_tested_k90: supported.**
- **R5_mixed_selected_and_unselected_share_late_capacity: supported.**
- **R6_template_transfer_is_stable: supported.**
- **R7_signal_gate_passes: supported.**

| Regime | Seed | Status | Layer | PCA32/64 | answer span | K90 |
|---|---:|---|---:|---:|---:|---:|
| pure | 0 | all | 0 | 1.000/1.000 | 0.717 | 8 |
| pure | 0 | all | 1 | 1.000/1.000 | 0.979 | 8 |
| pure | 0 | all | 2 | 1.000/1.000 | 0.996 | 8 |
| pure | 0 | all | 3 | 1.000/1.000 | 0.994 | 8 |
| pure | 101 | all | 0 | 0.999/1.000 | 0.866 | 8 |
| pure | 101 | all | 1 | 0.999/1.000 | 0.953 | 8 |
| pure | 101 | all | 2 | 0.999/1.000 | 0.990 | 8 |
| pure | 101 | all | 3 | 0.999/1.000 | 0.987 | 8 |
| pure | 103 | all | 0 | 1.000/1.000 | 0.841 | 8 |
| pure | 103 | all | 1 | 1.000/1.000 | 0.990 | 8 |
| pure | 103 | all | 2 | 0.999/1.000 | 0.994 | 8 |
| pure | 103 | all | 3 | 1.000/1.000 | 0.990 | 8 |
| mixed | 0 | unselected | 0 | 0.965/0.997 | 0.447 | 32 |
| mixed | 0 | unselected | 1 | 0.874/0.952 | 0.388 | 64 |
| mixed | 0 | unselected | 2 | 0.885/0.948 | 0.565 | 64 |
| mixed | 0 | unselected | 3 | 0.983/0.993 | 0.961 | 8 |
| mixed | 0 | selected | 0 | 0.953/0.996 | 0.237 | 32 |
| mixed | 0 | selected | 1 | 0.853/0.941 | 0.249 | 64 |
| mixed | 0 | selected | 2 | 0.862/0.940 | 0.437 | 64 |
| mixed | 0 | selected | 3 | 0.981/0.991 | 0.970 | 8 |
| mixed | 101 | unselected | 0 | 0.965/0.998 | 0.319 | 32 |
| mixed | 101 | unselected | 1 | 0.869/0.961 | 0.446 | 64 |
| mixed | 101 | unselected | 2 | 0.863/0.940 | 0.515 | 64 |
| mixed | 101 | unselected | 3 | 0.980/0.992 | 0.956 | 8 |
| mixed | 101 | selected | 0 | 0.958/0.997 | 0.294 | 32 |
| mixed | 101 | selected | 1 | 0.856/0.959 | 0.249 | 64 |
| mixed | 101 | selected | 2 | 0.846/0.935 | 0.394 | 64 |
| mixed | 101 | selected | 3 | 0.975/0.989 | 0.959 | 8 |
| mixed | 104 | unselected | 0 | 0.924/0.989 | 0.344 | 32 |
| mixed | 104 | unselected | 1 | 0.883/0.960 | 0.547 | 64 |
| mixed | 104 | unselected | 2 | 0.910/0.967 | 0.656 | 32 |
| mixed | 104 | unselected | 3 | 0.969/0.987 | 0.919 | 16 |
| mixed | 104 | selected | 0 | 0.885/0.986 | 0.164 | 64 |
| mixed | 104 | selected | 1 | 0.818/0.935 | 0.248 | 64 |
| mixed | 104 | selected | 2 | 0.859/0.946 | 0.526 | 64 |
| mixed | 104 | selected | 3 | 0.971/0.990 | 0.958 | 8 |

A global mixed-regime difference means arbitrary load changes the shared implementation used even by unselected rule facts. It does not show that any individual unselected fact is memorized or that native neurons align across checkpoints.
