# Finite gate-bottleneck depth telescope

Every layer uses the same assay: project its native post-product gate in a discovery-frozen affine subspace, patch that gate, and recompute the entire suffix. Effects are relative to literally zeroing that layer's gate. Final answer-coordinate R2 uses the frozen Layer-3 answer quotient.

## Locked hypotheses

- **D0_native_gate_patches_are_exact: supported.**
- **D1_semantic_answer_quotient_emerges_only_finally: supported.**
- **D2_rank64_dense_bottleneck_is_sufficient_through_depth: supported.**
- **D3_dense_axes_beat_native_coordinates_through_depth: not supported.**
- **D4_causal_dimension_contracts_with_depth: not supported.**
- **D5_arbitrary_complexity_is_visible_early: not supported.**
- **D6_rank64_pca_beats_random_subspaces: supported.**
- **D7_template_transfer_is_stable: supported.**
- **D8_signal_gate_passes: supported.**

| Seed | Status | Layer | PCA r16/r32/r64 | coord64 | answer span | K90 | random64 p95 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | unselected | 0 | 0.867/0.972/0.997 | 0.900 | 0.434 | 32 | 0.650 |
| 0 | unselected | 1 | 0.771/0.891/0.963 | 0.768 | 0.407 | 64 | 0.689 |
| 0 | unselected | 2 | 0.770/0.872/0.952 | 0.742 | 0.546 | 64 | 0.641 |
| 0 | unselected | 3 | 0.966/0.982/0.992 | 0.784 | 0.961 | 8 | 0.656 |
| 0 | selected | 0 | 0.814/0.956/0.996 | 0.861 | 0.250 | 32 | 0.654 |
| 0 | selected | 1 | 0.631/0.801/0.937 | 0.692 | 0.183 | 64 | 0.617 |
| 0 | selected | 2 | 0.724/0.842/0.944 | 0.670 | 0.462 | 64 | 0.636 |
| 0 | selected | 3 | 0.960/0.979/0.992 | 0.758 | 0.971 | 8 | 0.633 |
| 101 | unselected | 0 | 0.838/0.966/0.997 | 0.901 | 0.409 | 32 | 0.631 |
| 101 | unselected | 1 | 0.712/0.866/0.960 | 0.757 | 0.389 | 64 | 0.649 |
| 101 | unselected | 2 | 0.776/0.879/0.950 | 0.739 | 0.567 | 64 | 0.635 |
| 101 | unselected | 3 | 0.962/0.977/0.991 | 0.804 | 0.950 | 8 | 0.663 |
| 101 | selected | 0 | 0.776/0.957/0.997 | 0.894 | 0.208 | 32 | 0.631 |
| 101 | selected | 1 | 0.686/0.835/0.953 | 0.754 | 0.255 | 64 | 0.644 |
| 101 | selected | 2 | 0.682/0.845/0.944 | 0.700 | 0.339 | 64 | 0.627 |
| 101 | selected | 3 | 0.961/0.976/0.989 | 0.765 | 0.964 | 8 | 0.633 |
| 104 | unselected | 0 | 0.742/0.932/0.991 | 0.821 | 0.250 | 32 | 0.642 |
| 104 | unselected | 1 | 0.756/0.885/0.953 | 0.738 | 0.525 | 64 | 0.642 |
| 104 | unselected | 2 | 0.816/0.906/0.959 | 0.774 | 0.695 | 32 | 0.662 |
| 104 | unselected | 3 | 0.939/0.973/0.990 | 0.778 | 0.923 | 16 | 0.651 |
| 104 | selected | 0 | 0.594/0.874/0.984 | 0.690 | 0.054 | 64 | 0.502 |
| 104 | selected | 1 | 0.673/0.831/0.947 | 0.715 | 0.256 | 64 | 0.617 |
| 104 | selected | 2 | 0.743/0.874/0.950 | 0.711 | 0.522 | 64 | 0.605 |
| 104 | selected | 3 | 0.946/0.969/0.990 | 0.744 | 0.960 | 16 | 0.620 |

This telescope distinguishes causal dimension from semantic organization. A layer can be compact without yet aligning to the answer quotient, or semantically aligned while using many native neurons.
