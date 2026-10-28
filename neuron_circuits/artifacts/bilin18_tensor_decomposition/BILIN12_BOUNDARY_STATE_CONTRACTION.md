# State contraction of the natural newline write ray

A frozen Layer-11 newline write direction is given either its per-example oracle coefficient or a coefficient predicted linearly from the paired upstream state change. The predictor is trained on FineWeb and compared with a shuffled-target fit.

## Locked hypotheses

- **T0_one_effect_ray_is_oracle_sufficient: not supported.**
- **T1_wikitext_failure_is_mostly_scale: not supported.**
- **T2_state_change_predicts_contraction: supported.**
- **T3_predicted_contraction_is_causally_sufficient: not supported.**
- **T4_state_contraction_repairs_wikitext: not supported.**
- **T5_real_state_mapping_beats_shuffled_control: not supported.**

Selected relative ridge penalty: `1`; calibration scalar MSE: `0.01154`.

| Corpus | Coefficient Pearson/R2 | Ray write-energy fraction | Fixed fidelity | Oracle-ray fidelity | Predicted fidelity | Shuffled fidelity |
|---|---:|---:|---:|---:|---:|---:|
| fineweb | 0.891/0.770 | 0.897 | 0.714 | 0.710 | 0.706 | 0.712 |
| wikitext | 0.930/0.814 | 0.706 | 0.376 | 0.464 | 0.441 | 0.334 |
| tinystories | 0.895/0.759 | 0.891 | 0.711 | 0.717 | 0.685 | 0.701 |

Oracle scaling tests whether the direction is sufficient; predicted scaling tests whether a simple reusable state-contraction rule can recover it. These are deliberately separate claims.
