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
- **R7_signal_gate_passes: not supported.**

| Regime | Seed | Status | Layer | PCA32/64 | answer span | K90 |
|---|---:|---|---:|---:|---:|---:|
| pure | 0 | all | 0 | 1.000/1.000 | 0.840 | 8 |
| pure | 0 | all | 1 | 1.000/1.000 | 0.976 | 8 |
| pure | 0 | all | 2 | 1.000/1.000 | 0.996 | 8 |
| pure | 0 | all | 3 | 1.000/1.000 | 0.995 | 8 |
| pure | 101 | all | 0 | 0.999/1.000 | 0.790 | 8 |
| pure | 101 | all | 1 | 0.999/1.000 | 0.949 | 8 |
| pure | 101 | all | 2 | 0.999/1.000 | 0.989 | 8 |
| pure | 101 | all | 3 | 0.999/1.000 | 0.987 | 8 |
| pure | 103 | all | 0 | 1.000/1.000 | 0.817 | 8 |
| pure | 103 | all | 1 | 1.000/1.000 | 0.990 | 8 |
| pure | 103 | all | 2 | 1.000/1.000 | 0.995 | 8 |
| pure | 103 | all | 3 | 1.000/1.000 | 0.989 | 8 |
| mixed | 0 | unselected | 0 | 0.972/0.997 | 0.446 | 32 |
| mixed | 0 | unselected | 1 | 0.852/0.953 | 0.420 | 64 |
| mixed | 0 | unselected | 2 | 0.860/0.947 | 0.558 | 64 |
| mixed | 0 | unselected | 3 | 0.980/0.991 | 0.955 | 8 |
| mixed | 0 | selected | 0 | 0.958/0.996 | 0.268 | 32 |
| mixed | 0 | selected | 1 | 0.806/0.935 | 0.190 | 64 |
| mixed | 0 | selected | 2 | 0.850/0.943 | 0.439 | 64 |
| mixed | 0 | selected | 3 | 0.979/0.991 | 0.974 | 8 |
| mixed | 101 | unselected | 0 | 0.970/0.997 | 0.387 | 32 |
| mixed | 101 | unselected | 1 | 0.873/0.962 | 0.477 | 64 |
| mixed | 101 | unselected | 2 | 0.847/0.946 | 0.555 | 64 |
| mixed | 101 | unselected | 3 | 0.975/0.991 | 0.952 | 8 |
| mixed | 101 | selected | 0 | 0.957/0.997 | 0.269 | 32 |
| mixed | 101 | selected | 1 | 0.848/0.959 | 0.218 | 64 |
| mixed | 101 | selected | 2 | 0.838/0.934 | 0.333 | 64 |
| mixed | 101 | selected | 3 | 0.976/0.988 | 0.962 | 8 |
| mixed | 104 | unselected | 0 | 0.934/0.991 | 0.307 | 32 |
| mixed | 104 | unselected | 1 | 0.903/0.961 | 0.557 | 32 |
| mixed | 104 | unselected | 2 | 0.906/0.960 | 0.689 | 32 |
| mixed | 104 | unselected | 3 | 0.965/0.989 | 0.910 | 16 |
| mixed | 104 | selected | 0 | 0.866/0.983 | 0.049 | 64 |
| mixed | 104 | selected | 1 | 0.825/0.946 | 0.280 | 64 |
| mixed | 104 | selected | 2 | 0.867/0.953 | 0.579 | 64 |
| mixed | 104 | selected | 3 | 0.971/0.989 | 0.963 | 8 |

A global mixed-regime difference means arbitrary load changes the shared implementation used even by unselected rule facts. It does not show that any individual unselected fact is memorized or that native neurons align across checkpoints.
