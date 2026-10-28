# Nonlinear nearest-state router for the Layer-8 suffix atlas

A cosine kernel averages complete causal mask-response surfaces from neighboring FineWeb states. Hyperparameters are selected by query-grouped FineWeb validation and frozen for WikiText.

## Locked hypotheses

- **K0_local_state_atlas_validates: not supported.**
- **K1_kernel_beats_linear_tail: not supported.**
- **K2_raw_state_neighborhood_is_insufficient: supported.**

Selected `8` neighbors and temperature `1.0`.

| Split | Median fidelity | 10th percentile | Minimum |
|---|---:|---:|---:|
| fineweb_cv | 0.990 | 0.609 | -2.859 |
| wikitext | 0.984 | 0.812 | -2.636 |

Failure means raw Layer-8 cosine neighborhoods do not provide a reliable atlas chart selector at this data scale. It does not rule out learned nonlinear metrics or downstream-sensitivity features.
