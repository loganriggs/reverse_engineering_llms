# Cross-initialization linear conjugacy

An affine ridge map and an orthogonal Procrustes map are fit across all token-position residual states from trained-template discovery facts. They are evaluated first on disjoint facts in the same template and then on those facts in a held-out template. Causal conjugacy asks whether the same ridge state map also aligns the models' exact centered-answer Jacobians. A prompt-shuffled map controls for dimensionality and marginal state statistics.

## Locked hypotheses

- **Q0_residual_codes_are_linearly_equivalent: not supported.**
- **Q1_residual_codes_are_orthogonally_equivalent: not supported.**
- **Q2_linear_map_conjugates_causal_tangents: not supported.**
- **Q3_alignment_beats_shuffled_correspondence: not supported.**
- **Q4_residual_coordinates_are_more_stable_than_native_gates: not supported.**
- **Q5_causal_conjugacy_tracks_interface_fidelity: not supported.**

| Seed | Gate | Same-template ridge R2 0..4 | Transfer ridge R2 0..4 | Orthogonal transfer R2 0..4 | Causal cos 0..4 | Gate transfer R2 0..3 | Shuffled R2 0..4 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 101 | pass | 1.00/0.88/0.78/0.78/0.88 | 0.78/-0.27/-0.14/-1.59/-0.76 | -0.81/-8.85/-16.28/-65.47/-751.79 | 0.00/0.02/0.02/0.04/0.24 | -1.42/0.22/0.48/0.50 | 0.57/0.07/0.01/-0.55/-0.42 |
| 102 | fail | 1.00/0.91/0.69/0.63/0.92 | 0.70/-0.17/-0.17/-4.59/-1.08 | 0.26/-0.23/-1.25/-6.14/-23.26 | 0.01/0.01/0.01/0.03/0.22 | -0.92/-0.09/0.31/0.56 | 0.50/0.25/0.06/-0.81/-0.56 |
| 103 | pass | 1.00/0.92/0.74/0.78/0.92 | 0.73/-0.35/-0.21/-1.75/-2.30 | -0.69/-11.03/-16.54/-36.68/-286.89 | 0.01/0.01/0.01/0.04/0.24 | 0.67/0.36/0.60/0.72 | 0.49/0.06/-0.11/-0.80/-2.62 |

Boundary 0 is the embedding-plus-position state before block 0; boundary 4 is after the final block. State maps are fit over all positions, avoiding the degenerate constant final marker. High state R2 alone can reflect a shared codebook. Causal conjugacy is the stronger requirement that the learned coordinate map also transports local computation.
