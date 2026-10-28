# Native-versus-rotated frontier for conditional final-query readouts

For each prospective edit, channels are selected anew from the realized final-position bilinear product change. Native activation magnitude and exact local-write contribution are compared with local contribution after an exact dense orthogonal product-coordinate rotation. Every selected delta is propagated through the complete suffix.

## Locked hypotheses

- **C0_native_local_reaches_half_by_256: supported.**
- **C1_high_fidelity_is_not_sparse: supported.**
- **C2_native_basis_beats_rotation: supported.**
- **C3_activation_ranking_nearly_matches_local: supported.**
- **C4_full_frontier_reconstructs: supported.**
- **C5_native_advantage_transfers_ood: supported.**

| Layer | Corpus | Native activation S50/S90 | Native local S50/S90 | Rotated local S50/S90 | Rank-256 native/rotated advantage |
|---:|---|---:|---:|---:|---:|
| 0 | fineweb | 256/2048 | 128/2048 | 256/4096 | 0.661/0.570 (+0.091) |
| 0 | tinystories | 128/1024 | 128/2048 | 128/2048 | 0.764/0.695 (+0.069) |
| 1 | fineweb | 128/1024 | 128/2048 | 256/4096 | 0.727/0.547 (+0.180) |
| 1 | tinystories | 128/1024 | 64/2048 | 256/4096 | 0.783/0.631 (+0.152) |

Maximum full-coordinate rotation reconstruction error: **4.86e-07**.

Native advantage here means the architectural product coordinates are unusually useful for sparse **instance-conditioned** execution. It does not imply that the same neurons recur across examples or possess a fixed semantic label.
