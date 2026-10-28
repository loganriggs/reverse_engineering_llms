# Finite bottleneck inside the Block-2 MLP hub

Block-2 post-product gates are projected in a discovery-frozen affine subspace, patched at the real model site, and Blocks 3 plus the complete output suffix are recomputed. The target effect is the native Block-2 MLP contribution relative to a literal zero-gate intervention.

## Locked hypotheses

- **B0_native_gate_patch_is_exact: supported.**
- **B1_rank64_block2_gate_is_sufficient: supported.**
- **B2_rank16_block2_gate_is_insufficient: supported.**
- **B3_dense_block2_axes_beat_native_coordinates: supported.**
- **B4_block2_pca_beats_random_subspaces: supported.**
- **B5_arbitrary_block2_code_is_broader: not supported.**
- **B6_block2_answer_centroids_are_not_a_sufficient_controller: supported.**
- **B7_block2_subspace_predicts_final_answer_coordinates: supported.**
- **B8_template_transfer_is_stable: supported.**
- **B9_signal_gate_passes: supported.**

| Seed | Status | PCA r0/r16/r32/r64/r96 | coordinate64 | answer-span | random64 mean/p95 | final-coordinate R2@64 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | unselected | 0.045/0.774/0.872/0.949/0.978 | 0.723 | 0.616 | 0.616/0.653 | 0.927 |
| 0 | selected | 0.022/0.721/0.854/0.947/0.978 | 0.712 | 0.438 | 0.609/0.653 | 0.929 |
| 101 | unselected | -0.012/0.762/0.840/0.941/0.974 | 0.702 | 0.529 | 0.608/0.655 | 0.925 |
| 101 | selected | 0.029/0.685/0.847/0.940/0.973 | 0.697 | 0.377 | 0.608/0.627 | 0.922 |
| 104 | unselected | 0.050/0.800/0.903/0.961/0.981 | 0.744 | 0.676 | 0.632/0.673 | 0.955 |
| 104 | selected | 0.002/0.726/0.858/0.946/0.978 | 0.696 | 0.544 | 0.573/0.601 | 0.939 |

This is a literal finite replacement, unlike the affine final-pair decomposition. A compact dense Block-2 gate would identify an earlier causal bottleneck; a broad or status-specific result would localize arbitrary-address complexity inside this hub.
