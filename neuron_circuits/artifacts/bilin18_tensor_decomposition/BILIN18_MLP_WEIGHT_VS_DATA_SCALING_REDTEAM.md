# Data-scaling red team for MLP write geometry

The earlier task-blind finite test used 256 discovery sites and found leading static Down-output modes competitive with or better than data PCA before the final layer. This red team fits centered and through-origin PCA on 64, 256, and 1,024 new discovery sites, then evaluates all bases on a separate 256-site FineWeb panel.

## Locked hypotheses

- **G0_native_replay_is_exact: supported.** Observed: `{'maximum_logit_error': 4.1484832763671875e-05}`
- **G1_data_pca_improves_with_sample_size: supported.** Observed: `{'passing': 3}`
- **G2_large_sample_data_geometry_beats_static_weights: not supported.** Observed: `{'passing': 2}`
- **G3_affine_mean_does_not_explain_weight_advantage: supported.** Observed: `{'passing': 4}`
- **G4_early_mid_write_is_moderately_compressible: not supported.** Observed: `{'passing': 1}`
- **G5_late_compactness_replicates: supported.** Observed: `{'fidelity': 0.9860454959810973}`
- **G6_large_sample_data_pca_beats_random: not supported.** Observed: `{'passing': 3}`

| Layer | Centered64 by data size 64/256/1024 | Origin64 64/256/1024 | Centered1024 rank16/64/128 | Weight rank16/64/128 | Random64 p95 |
|---:|---:|---:|---:|---:|---:|
| 1 | -0.188/0.043/0.370 | -0.171/0.050/0.361 | -0.134/0.370/0.680 | 0.002/0.323/0.493 | 0.376 |
| 8 | 0.131/0.204/0.246 | 0.149/0.215/0.252 | 0.087/0.246/0.374 | 0.143/0.296/0.427 | 0.079 |
| 12 | 0.097/0.146/0.198 | 0.127/0.152/0.203 | 0.073/0.198/0.297 | -0.028/0.109/0.234 | 0.054 |
| 17 | 0.986/0.989/0.991 | 0.986/0.989/0.991 | 0.986/0.991/0.992 | 0.930/0.940/0.947 | 0.229 |

## Guardrail

Improvement with sample size diagnoses estimation error, not semantic clarity. Static Down modes describe the layer's available output gain; data PCA describes its natural write distribution. Both can be causally useful without being a unique feature basis. All tests preserve the model's nonlinear suffix and are sufficiency relative to deleting the audited write.
