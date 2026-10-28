# Low-rank final-gate red team

All candidate post-product gates are passed through the original down projection and exact suffix. Rank zero is the discovery mean. Opposite-status bases use the target-status mean, so only orientation transfers.

## Locked hypotheses

- **M0_native_gate_is_exact: supported.**
- **M1_mean_gate_is_insufficient: supported.**
- **M2_eight_dense_gate_directions_are_usable: supported.**
- **M3_rank8_pca_beats_random_subspaces: supported.**
- **M4_rule_and_arbitrary_gate_bases_are_shared: supported.**
- **M5_dense_axes_beat_native_coordinates: supported.**
- **M6_lowrank_gate_reconstructs_residual_write: supported.**
- **M7_bottom_gate_directions_are_not_sufficient: supported.**
- **M8_template_transfer_is_stable: supported.**
- **M9_signal_gate_passes: supported.**

| Seed | Status | mean | PCA r1/r2/r4/r8/r16 | opposite8 | coord8 | bottom8 | random8 mean/p95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | unselected | -0.018 | 0.109/0.388/0.748/0.944/0.968 | 0.944 | 0.136 | -0.016 | 0.081/0.113 |
| 0 | selected | -0.002 | 0.103/0.305/0.676/0.923/0.957 | 0.912 | 0.131 | 0.001 | 0.089/0.110 |
| 101 | unselected | -0.038 | 0.054/0.328/0.702/0.946/0.963 | 0.954 | 0.120 | -0.037 | 0.077/0.111 |
| 101 | selected | 0.008 | 0.128/0.341/0.635/0.929/0.963 | 0.905 | 0.160 | 0.007 | 0.096/0.119 |
| 104 | unselected | -0.047 | 0.089/0.324/0.617/0.882/0.935 | 0.919 | 0.137 | -0.042 | 0.052/0.089 |
| 104 | selected | -0.008 | 0.095/0.293/0.589/0.897/0.939 | 0.824 | 0.150 | -0.007 | 0.076/0.097 |

A low-dimensional dense gate code is evidence against treating the 192 native channels as 192 independent computational primitives. It does not make the PCA directions semantic or identify the upstream fact lookup.
