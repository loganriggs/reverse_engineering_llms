# Lexical generalization of the boundary-style circuit

Eleven single-token US/UK spelling pairs that were absent from circuit discovery are used in both directions: as new cue swaps with the original targets, and as new output targets with the original cues. The native model, finite L8H2 cue-edge intervention, L9H8 boundary-edge freeze, upstream/downstream controls, and random signed contrasts are all reevaluated.

## Locked results

- **G0_native_model_generalizes_both_directions: supported.**
- **G1_L8H2_cue_edge_generalizes_both_directions: supported.**
- **G2_L9H8_mediation_generalizes: supported.**
- **G3_route_specificity_generalizes: supported.**
- **G4_declared_contrasts_beat_random: supported.**
- **G5_typed_circuit_generalizes_new_lexemes: supported.**

| Case/split | N | model win/mean | L8 edge win/mean | L9-mediated fraction | random percentiles |
|---|---:|---:|---:|---:|---:|
| new_cues_old_targets/0 | 80 | 0.875/+0.5552 | 0.775/+0.03261 | 0.541 | 1.000/1.000 |
| new_cues_old_targets/1 | 80 | 0.950/+0.7483 | 0.913/+0.05648 | 0.556 | 1.000/1.000 |
| old_cues_new_targets/0 | 80 | 0.925/+0.9248 | 0.875/+0.05912 | 0.517 | 1.000/1.000 |
| old_cues_new_targets/1 | 80 | 0.975/+0.8246 | 0.850/+0.06178 | 0.505 | 1.000/1.000 |

## Interpretation boundary

This is lexical transfer within the same manually declared US/UK spelling family, not transfer to an unrelated style variable. The new list was fixed after checking only single-token tokenization and corpus availability, before inspecting model or circuit effects. Passing supports a reusable typed lexical-style circuit rather than a twelve-pair lookup-table explanation.
