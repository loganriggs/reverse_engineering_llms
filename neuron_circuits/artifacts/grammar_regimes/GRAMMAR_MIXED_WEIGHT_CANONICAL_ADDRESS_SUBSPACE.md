# Weight-canonical causal subspace for the mixed address code

PCA is fit only to discovery-template normalized pre-MLP states. Held-out states are projected into each frozen subspace and executed through the original left/right/down bilinear weights—no write regression is fit. Equal-rank random subspaces use the same state mean and native tensor.

## Locked hypotheses

- **P0_full_weight_tensor_is_exact: supported.**
- **P1_rank48_pca_address_is_sufficient: supported.**
- **P2_rank32_pca_address_is_sufficient: supported.**
- **P3_rank48_is_near_saturated: supported.**
- **P4_pca_address_beats_random_subspaces: supported.**
- **P5_fact_specific_residual_is_captured: supported.**
- **P6_arbitrary_and_rule_addresses_have_similar_compactness: supported.**
- **P7_address_subspace_transfers_templates: supported.**
- **P8_signal_gate_passes: supported.**

| Seed | Status | Template | r8 | r16 | r24 | r32 | r48 | r64 | r96 | random48 mean/p95 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | unselected | 0 | 0.273 | 0.596 | 0.789 | 0.903 | 0.976 | 0.995 | 1.000 | 0.534/0.566 |
| 0 | unselected | 6 | 0.260 | 0.590 | 0.780 | 0.894 | 0.974 | 0.994 | 1.000 | 0.526/0.564 |
| 0 | selected | 0 | 0.152 | 0.497 | 0.720 | 0.843 | 0.963 | 0.993 | 1.000 | 0.487/0.504 |
| 0 | selected | 6 | 0.143 | 0.511 | 0.733 | 0.846 | 0.958 | 0.992 | 1.000 | 0.480/0.513 |
| 101 | unselected | 0 | 0.372 | 0.666 | 0.833 | 0.908 | 0.977 | 0.995 | 1.000 | 0.557/0.584 |
| 101 | unselected | 6 | 0.347 | 0.631 | 0.822 | 0.893 | 0.972 | 0.994 | 1.000 | 0.552/0.592 |
| 101 | selected | 0 | 0.182 | 0.553 | 0.731 | 0.868 | 0.962 | 0.993 | 1.000 | 0.494/0.516 |
| 101 | selected | 6 | 0.168 | 0.530 | 0.735 | 0.852 | 0.957 | 0.992 | 1.000 | 0.482/0.518 |
| 104 | unselected | 0 | 0.463 | 0.693 | 0.856 | 0.926 | 0.983 | 0.997 | 1.000 | 0.536/0.572 |
| 104 | unselected | 6 | 0.405 | 0.688 | 0.855 | 0.916 | 0.980 | 0.996 | 1.000 | 0.537/0.564 |
| 104 | selected | 0 | 0.187 | 0.586 | 0.780 | 0.899 | 0.976 | 0.996 | 1.000 | 0.504/0.541 |
| 104 | selected | 6 | 0.152 | 0.548 | 0.764 | 0.887 | 0.974 | 0.995 | 1.000 | 0.490/0.521 |

This is a causal compression test of the fixed tensor program. Success identifies a reusable state subspace; it does not make the PCA axes individually semantic or explain how upstream layers place each fact in that subspace.
