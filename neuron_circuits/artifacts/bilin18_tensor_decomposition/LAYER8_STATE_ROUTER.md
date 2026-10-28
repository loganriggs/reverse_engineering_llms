# State-conditioned router for the Layer-8 suffix atlas

A bilinear ridge model combines pairwise mask features with either Layer-8 state PCs, per-layer derivative-diagnostic PCs, or both. It is trained on one FineWeb site per tensor query and frozen for another FineWeb site and WikiText.

## Locked hypotheses

- **R0_state_routes_mask: not supported.**
- **R1_diagnostics_route_mask: not supported.**
- **R2_combined_router: not supported.**
- **R3_linear_address_is_insufficient: not supported.**

| Router | Corpus | Median fidelity | 10th percentile | Minimum | Ridge alpha |
|---|---|---:|---:|---:|---:|
| state | train | 0.992 | 0.904 | 0.762 | 0.001 |
| state | fineweb | 0.990 | 0.648 | -0.988 | 0.001 |
| state | wikitext | 0.983 | 0.735 | -2.636 | 0.001 |
| diagnostics | train | 0.991 | 0.917 | 0.762 | 0.001 |
| diagnostics | fineweb | 0.961 | -1.317 | -2.556 | 0.001 |
| diagnostics | wikitext | 0.974 | 0.162 | -4.039 | 0.001 |
| combined | train | 0.996 | 0.919 | 0.762 | 0.001 |
| combined | fineweb | 0.948 | -0.741 | -0.970 | 0.001 |
| combined | wikitext | 0.963 | 0.238 | -4.133 | 0.001 |

The router is trained from finite causal labels. Passing would show that the atlas address is predictable from a low-rank state description; it would not make the explanation purely weight-derived. Failure only rejects this bilinear low-data router.
