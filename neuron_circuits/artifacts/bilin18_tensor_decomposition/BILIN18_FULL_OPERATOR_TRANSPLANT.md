# Finite transplant of a complete bus-aware value operator

The pair L2H5/L3H8 maximizes the product of absolute full-operator cosine and replicated natural-response cosine. Each complete current-state/value-bus map is transplanted under the target head's native QK routing and rescaled to the target operator norm. Controls combine the target with one other learned operator to match both operator cosine and norm exactly.

## Locked hypotheses

- **W1_factored_native_operator_matches_head_write: supported.**
- **W2_controls_match_operator_angle_and_norm: supported.**
- **W3_partner_operator_is_privileged: not supported.**
- **W4_partner_operator_beats_control_tail: supported.**
- **W5_partner_operator_beats_deletion: supported.**
- **W6_operator_advantage_is_split_stable: supported.**
- **native_branch_matches_checkpoint: supported.**

Maximum factorization error `3.66e-07`; native checkpoint error `3.93e-05`; matched-geometry error `1.11e-16`.

| Transplant / split | Operator cosine | Partner KL | Control median [p10] | Delete KL | Partner/control |
|---|---:|---:|---:|---:|---:|
| L2H5_to_L3H8 / a | 0.210 | 1.246e-02 | 1.199e-02 [1.044e-02] | 2.091e-02 | 1.040 |
| L2H5_to_L3H8 / b | 0.210 | 1.187e-02 | 1.077e-02 [9.324e-03] | 2.264e-02 | 1.102 |
| L3H8_to_L2H5 / a | 0.210 | 2.092e-03 | 3.488e-03 [3.114e-03] | 5.121e-03 | 0.600 |
| L3H8_to_L2H5 / b | 0.210 | 1.278e-03 | 1.751e-03 [1.644e-03] | 2.265e-03 | 0.730 |

## Guardrails

This tests the value-side operator under the target's native QK routing; it does not transplant the router or destination layer. Controls are exact in the raw Frobenius operator metric but downstream curvature can distinguish directions with the same angle. The pair was selected from prior weight and natural-response panels, while the finite evaluation panel is fresh.
