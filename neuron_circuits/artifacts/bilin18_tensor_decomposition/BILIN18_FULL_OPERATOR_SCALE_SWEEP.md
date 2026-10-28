# Scale sweep for directional full-operator compatibility

The complete 52-control census is repeated at small and intermediate interpolation scales. This distinguishes a local destination-metric relation from a large finite-edit accident. Scale 1.0 is copied from the complete census on the same panel; smaller scales use its first 64 rows.

## Locked hypotheses

- **S1_forward_compatibility_is_local: supported.**
- **S2_forward_percentile_is_scale_stable: supported.**
- **S3_reverse_direction_never_enters_top_decile: supported.**
- **S4_directional_gap_persists_locally: not supported.**
- **S5_small_scale_effect_is_approximately_quadratic: supported.**
- **factorization_invariant: supported.**

| Target ← partner / half | Scale | Partner percentile | Partner/control | Partner KL |
|---|---:|---:|---:|---:|
| L2H5_to_L3H8 / a | 0.05 | 0.442 | 0.985 | 2.224e-05 |
| L2H5_to_L3H8 / a | 0.20 | 0.327 | 0.965 | 3.317e-04 |
| L2H5_to_L3H8 / a | 0.50 | 0.404 | 0.967 | 2.202e-03 |
| L2H5_to_L3H8 / a | 1.00 | 0.462 | 0.995 | 1.246e-02 |
| L2H5_to_L3H8 / b | 0.05 | 0.173 | 0.923 | 1.247e-05 |
| L2H5_to_L3H8 / b | 0.20 | 0.154 | 0.912 | 1.900e-04 |
| L2H5_to_L3H8 / b | 0.50 | 0.192 | 0.933 | 1.218e-03 |
| L2H5_to_L3H8 / b | 1.00 | 0.596 | 1.061 | 1.187e-02 |
| L3H8_to_L2H5 / a | 0.05 | 0.000 | 0.746 | 5.919e-06 |
| L3H8_to_L2H5 / a | 0.20 | 0.000 | 0.749 | 9.562e-05 |
| L3H8_to_L2H5 / a | 0.50 | 0.000 | 0.705 | 6.110e-04 |
| L3H8_to_L2H5 / a | 1.00 | 0.000 | 0.600 | 2.092e-03 |
| L3H8_to_L2H5 / b | 0.05 | 0.000 | 0.656 | 2.703e-06 |
| L3H8_to_L2H5 / b | 0.20 | 0.000 | 0.667 | 4.319e-05 |
| L3H8_to_L2H5 / b | 0.50 | 0.000 | 0.684 | 2.722e-04 |
| L3H8_to_L2H5 / b | 1.00 | 0.000 | 0.671 | 1.278e-03 |

## Guardrails

The smaller-scale panels are subsets of the scale-1 census rather than new data. At very small KL, float precision and logit saturation can affect ranks; approximate quadratic scaling is therefore an explicit invariant. Persistence across scale identifies local metric compatibility, not semantic equivalence or a transferable QK router.
