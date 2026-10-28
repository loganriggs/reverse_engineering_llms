# Thirty-two-point convergence red team for identity tensor paths

The retained eight-point run failed its frozen exactness gate on rare large wrong-codeword interventions. This rerun changes only Gauss--Legendre order from 8 to 32. It compares the resulting finite path metrics against the original artifact rather than replacing it.

## Locked hypotheses

- **R0_thirty_two_point_paths_are_numerically_exact: supported.**
- **R1_top_path_fidelities_are_stable: supported.**
- **R2_bilinear_projection_is_stable: supported.**
- **R3_one_hundred_twenty_eight_paths_are_sufficient: supported.**
- **R4_mean_rank_transfers_to_wrong_digit_substitution: supported.**
- **R5_ranked_paths_beat_random_banks: supported.**
- **R6_path_program_transfers_templates: supported.**
- **R7_signal_gate_passes: supported.**

| Seed | max state/logit error | selected mean top32/128/256 | selected roll top32/128/256 | bilinear projection mean/roll |
|---:|---:|---:|---:|---:|
| 0 | 6.61e-06/8.31e-06 | 0.890/0.989/0.999 | 0.868/0.976/0.998 | 0.960/1.009 |
| 101 | 1.68e-06/3.03e-06 | 0.824/0.985/0.999 | 0.777/0.983/0.997 | 0.980/0.944 |
| 104 | 7.26e-07/1.83e-06 | 0.840/0.984/0.998 | 0.643/0.960/0.995 | 0.947/0.949 |

Passing this red team establishes numerical completeness for the declared finite path expansion. It does not make individual paths unique semantic objects: legal regroupings and cancellations remain, and rankings are specific to the identity-correction intervention family.
