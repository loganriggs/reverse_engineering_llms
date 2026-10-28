# Cross-initialization linear conjugacy

An affine ridge map and an orthogonal Procrustes map are fit across all token-position residual states from trained-template discovery facts. They are evaluated first on disjoint facts in the same template and then on those facts in a held-out template. Causal conjugacy asks whether the same ridge state map also aligns the models' exact centered-answer Jacobians. A prompt-shuffled map controls for dimensionality and marginal state statistics.

## Locked hypotheses

- **L0_residual_codes_are_linearly_equivalent: not supported.**
- **L1_residual_codes_are_orthogonally_equivalent: not supported.**
- **L2_linear_map_conjugates_causal_tangents: not supported.**
- **L3_alignment_beats_shuffled_correspondence: not supported.**
- **L4_residual_coordinates_are_more_stable_than_native_gates: supported.**
- **L5_causal_conjugacy_tracks_interface_fidelity: not supported.**

| Seed | Gate | Same-template ridge R2 0..4 | Transfer ridge R2 0..4 | Orthogonal transfer R2 0..4 | Causal cos 0..4 | Gate transfer R2 0..3 | Shuffled R2 0..4 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 101 | pass | 1.00/0.74/0.67/0.67/0.66 | 0.78/0.34/0.26/0.03/-0.16 | 0.68/-0.16/-0.27/-0.73/-0.86 | 0.01/0.00/-0.00/0.01/0.10 | -0.24/-2.65/-2.82/-2.64 | 0.56/0.14/0.13/-0.07/-0.23 |
| 102 | fail | 1.00/0.77/0.68/0.66/0.72 | 0.85/0.43/0.37/0.13/-0.19 | 0.66/0.31/0.26/0.11/-0.08 | 0.00/-0.00/0.00/0.01/0.08 | -0.17/-2.73/-2.90/-2.78 | 0.61/0.21/0.22/0.07/-0.08 |
| 103 | fail | 1.00/0.80/0.74/0.74/0.80 | 0.83/0.49/0.49/0.25/-0.62 | 0.67/0.32/0.32/0.12/-0.23 | 0.03/-0.00/0.00/0.01/0.10 | 0.17/-2.75/-3.05/-2.69 | 0.66/0.41/0.43/0.22/-0.50 |
| 104 | pass | 1.00/0.73/0.64/0.64/0.67 | 0.77/0.34/0.24/-0.04/-0.17 | 0.54/-1.75/-2.29/-3.59/-3.36 | 0.02/0.00/0.00/0.01/0.06 | -0.01/-2.72/-2.96/-2.56 | 0.54/0.18/0.13/-0.03/-0.11 |
| 105 | fail | 1.00/0.73/0.66/0.64/0.72 | 0.84/0.28/0.26/0.11/-0.41 | 0.65/0.13/0.10/-0.09/-0.46 | 0.02/0.00/0.00/0.01/0.10 | -0.31/-2.75/-2.81/-2.91 | 0.61/0.12/0.14/0.07/-0.41 |
| 106 | fail | 1.00/0.78/0.70/0.69/0.77 | 0.85/0.38/0.33/0.11/-0.32 | 0.70/0.29/0.24/0.11/-0.07 | 0.03/-0.00/0.01/0.02/0.10 | 0.12/-2.76/-2.91/-2.73 | 0.65/0.18/0.18/0.03/-0.17 |

Boundary 0 is the embedding-plus-position state before block 0; boundary 4 is after the final block. State maps are fit over all positions, avoiding the degenerate constant final marker. High state R2 alone can reflect a shared codebook. Causal conjugacy is the stronger requirement that the learned coordinate map also transports local computation.
