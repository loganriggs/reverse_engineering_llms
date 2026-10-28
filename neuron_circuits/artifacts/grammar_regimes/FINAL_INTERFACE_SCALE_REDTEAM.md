# Direction versus amplitude in the final grammar interface

Generic answer deltas are activation-norm matched or scalar-calibrated on discovery examples, then compared with semantic-rule patches on held-out identities and template 6.

## Locked hypotheses

- **S0_pure_semantic_and_answer_effect_directions_agree: supported.**
- **S1_discovery_calibrated_answer_delta_recovers_semantic_patch: not supported.**
- **S2_norm_matching_is_sufficient: supported.**
- **S3_final_semantic_patch_is_output_verb_plus_amplitude: not supported.**
- **S4_mixed_semantic_failure_is_not_fixed_by_scalar_calibration: not supported.**

| rho | Runs | Semantic acc | Answer unscaled/norm/calibrated acc | Effect cosine unscaled/calibrated | Fidelity unscaled/norm/calibrated | Norm/calibrated scale |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 3 | 0.843 | 0.415/0.937/0.776 | 0.992/0.994 | 0.899/0.870/0.948 | 1.36/1.20 |
| 0.5 | 2 | 0.503 | 0.800/0.902/0.532 | 0.992/0.992 | 0.495/0.023/0.926 | 1.25/0.67 |

Calibration uses only discovery identities/template. If it succeeds, the final-layer rule patch is a generic effect direction with rule-conditioned amplitude, not the location of the upstream rule computation.
