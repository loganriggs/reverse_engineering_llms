# Red team: coordinate binding in the joint typed-edge program

Fine-128 is compared on a third held-out FineWeb split with controls that preserve two coordinates while randomizing the third in every layer, plus a cyclic shift of each layer's selected program to another layer.

Maximum native-branch error: `7.01e-05`.

## Locked hypotheses

- **B1_category_binding_matters: supported.**
- **B2_distance_binding_matters: supported.**
- **B3_head_binding_matters: supported.**
- **B4_layer_assignment_matters: supported.**
- **B5_selected_program_prediction_fidelity_replicates: supported.**
- **native_branch_matches_checkpoint: supported.**

| Program | KL recovery | CE delta | Top-1 agreement |
|---|---:|---:|---:|
| fine128 | 0.9711 | +0.2075 | 0.7915 |
| category_0 | 0.3809 | +4.3514 | 0.1323 |
| category_1 | 0.3786 | +4.3658 | 0.1274 |
| category_2 | 0.1653 | +5.8683 | 0.1030 |
| distance_0 | 0.6900 | +2.1935 | 0.3247 |
| distance_1 | 0.7217 | +1.9519 | 0.3569 |
| distance_2 | 0.6853 | +2.2173 | 0.3242 |
| head_0 | 0.7043 | +2.0845 | 0.3364 |
| head_1 | 0.7194 | +1.9685 | 0.3438 |
| head_2 | 0.8395 | +1.1250 | 0.4761 |
| layer_shift | 0.5471 | +3.1911 | 0.2041 |

## Guardrails

Only three randomized programs are used per coordinate, so their empirical p95 is a coarse control. Randomization preserves the other-coordinate group counts but may create duplicate types, as in the earlier local test. A layer shift preserves every selected triple globally but changes where it executes. These tests establish coordinate and layer specificity of the executable program, not that the human category partition is uniquely optimal.
