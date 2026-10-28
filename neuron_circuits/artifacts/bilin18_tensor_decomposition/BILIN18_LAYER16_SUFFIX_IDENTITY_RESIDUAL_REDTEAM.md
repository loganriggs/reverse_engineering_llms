# Identity-residual red-team for the output-conditioned suffix atlas

The failed zero-centered suffix regression is replaced by a fitted shared scaled identity plus a low-rank ridge correction. This preserves the strong full-rank baseline and tests ranks through 768. Rank and regularization are again selected only on disjoint FineWeb validation.

## Locked hypotheses

- **I0_saved_factor_recomposition_is_exact: supported.**
- **I1_identity_residual_improves_validation_write: supported.**
- **I2_identity_residual_predicts_behavior: not supported.**
- **I3_identity_residual_explains_model_margin: not supported.**
- **I4_shared_residual_matches_component_specific_ceiling: not supported.**
- **I5_identity_residual_is_rank512_or_smaller: not supported.**
- **I6_identity_residual_transfers_ood: not supported.**
- **I7_identity_residual_avoids_behavioral_sign_failure: not supported.**

Scaled-identity validation write fidelity: `0.437`. Selected residual rank `768`, ridge multiplier `1`, validation write fidelity `0.587`.

| Corpus | Component | Shared write | Component write | Identity write | Shared behavior | Component behavior | Identity behavior |
|---|---|---:|---:|---:|---:|---:|---:|
| fineweb | ce | 0.669 | 0.680 | 0.649 | 0.236 | 0.389 | 0.181 |
| fineweb | prediction | 0.537 | 0.528 | 0.269 | 0.450 | 0.443 | -0.063 |
| fineweb | target | 0.590 | 0.571 | 0.443 | 0.369 | 0.329 | -0.032 |
| fineweb | margin | 0.579 | 0.606 | 0.578 | 0.254 | 0.427 | 0.355 |
| wikitext | ce | 0.643 | 0.700 | 0.681 | 0.238 | 0.512 | 0.390 |
| wikitext | prediction | 0.563 | 0.549 | 0.283 | 0.423 | 0.448 | -0.068 |
| wikitext | target | 0.602 | 0.578 | 0.494 | 0.380 | 0.331 | 0.024 |
| wikitext | margin | 0.642 | 0.685 | 0.664 | 0.190 | 0.529 | 0.464 |
| tinystories | ce | 0.793 | 0.834 | 0.828 | -0.177 | 0.540 | 0.143 |
| tinystories | prediction | 0.763 | 0.766 | 0.222 | 0.539 | 0.494 | -0.375 |
| tinystories | target | 0.787 | 0.758 | 0.509 | 0.373 | 0.301 | -0.570 |
| tinystories | margin | 0.777 | 0.810 | 0.775 | 0.034 | 0.569 | 0.377 |

If this residual model remains weak, the zero-centered failure cannot be explained by discarding the identity path or by a rank-256 ceiling. The remaining alternatives are context-conditioned/routed suffix maps, a different nonlinear shared parameterization, or insufficient discovery coverage—not one small static linear pullback.
