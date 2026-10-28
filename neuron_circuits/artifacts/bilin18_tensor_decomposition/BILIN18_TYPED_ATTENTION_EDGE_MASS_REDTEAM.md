# Red team: operator vocabulary versus frequency and routing mass

The six-layer closed-loop program is selected four ways on the same discovery data: destination-operator effect, raw occurrence frequency, absolute bilinear QK-pattern mass, or pattern mass times source-value norm. Evaluation uses a fourth untouched FineWeb split.

Maximum native-branch error: `4.58e-05`.

## Locked hypotheses

- **M1_operator_beats_frequency: supported.**
- **M2_operator_beats_routing_mass: not supported.**
- **M3_operator_beats_transport_mass: not supported.**
- **M4_operator_program_prediction_fidelity_replicates: supported.**
- **M5_operator_program_is_instance_sparse: supported.**
- **native_branch_matches_checkpoint: supported.**

| Program | KL recovery | CE delta | Top-1 agreement |
|---|---:|---:|---:|
| operator128 | 0.9749 | +0.1850 | 0.7905 |
| frequency128 | 0.4815 | +3.6891 | 0.1694 |
| routing_mass128 | 0.8900 | +0.7609 | 0.6064 |
| transport_mass128 | 0.9127 | +0.6091 | 0.6475 |

| Layer | Operator-128 edge fraction | Routing-mass fraction | Transport-mass fraction |
|---:|---:|---:|---:|
| 0 | 0.174 | 0.397 | 0.500 |
| 1 | 0.171 | 0.536 | 0.618 |
| 2 | 0.213 | 0.523 | 0.591 |
| 3 | 0.238 | 0.495 | 0.553 |
| 4 | 0.123 | 0.427 | 0.482 |
| 5 | 0.378 | 0.467 | 0.676 |

## Guardrails

Coverage is measured on discovery at the same 16 destination positions used for ranking, while causal metrics use a disjoint split and execute all destinations. Frequency counts architectural causal edge instances, not FLOPs after an optimized grouped implementation. QK pattern is signed in this model; magnitude controls use its absolute value. Value norm remains an incomplete proxy for destination effect because it omits output projection, RMSNorm geometry, and the following bilinear tensor.
