# Unseeded route-coalition discreteness red team

Stable k=4 centroids can be either discrete operators or a reproducible quantization of continuous anisotropic route geometry. This audit sweeps k, measures assignment margins and center separation, and compares frozen centroid reconstruction with a parameter-matched rank-4 linear subspace on replication routes.

## Locked hypotheses

- **R0_k4_is_a_privileged_discrete_vocabulary: not supported.** Observed: `{'layers': []}`
- **R1_k4_centers_are_well_separated: supported.** Observed: `{'layers': [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17]}`
- **R2_high_margin_candidates_survive: supported.** Observed: `{'candidates': ['L16C0', 'L16C1', 'L1C2', 'L6C0', 'L16C3', 'L17C0', 'L1C0', 'L6C2', 'L17C3', 'L0C0', 'L4C0']}`
- **R3_clusters_beat_parameter_matched_continuous_subspaces: not supported.** Observed: `{'layers': [], 'mean_cluster_minus_subspace': -0.15792157426969713}`
- **R4_continuous_geometry_is_depth_dependent: supported.** Observed: `{'high_margin_layers': [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17], 'continuous_or_ambiguous_layers': [5, 15]}`

| Layer | k=3/4/6 stability | k4 margin D/R | max center cosine | centroid fidelity | rank-4 subspace |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.985/0.990/0.965 | 0.208/0.194 | 0.792 | 0.702 | 0.880 |
| 1 | 0.999/0.996/0.992 | 0.202/0.210 | 0.767 | 0.778 | 0.875 |
| 2 | 0.999/0.987/0.997 | 0.373/0.352 | 0.521 | 0.696 | 0.817 |
| 3 | 0.994/0.997/0.948 | 0.403/0.390 | 0.475 | 0.666 | 0.821 |
| 4 | 0.998/0.995/0.991 | 0.262/0.226 | 0.714 | 0.732 | 0.862 |
| 5 | 0.998/0.999/0.999 | 0.008/0.008 | 0.991 | 0.989 | 0.997 |
| 6 | 0.997/0.994/0.976 | 0.179/0.169 | 0.861 | 0.786 | 0.902 |
| 7 | 0.857/0.904/0.918 | 0.227/0.183 | 0.790 | 0.681 | 0.846 |
| 8 | 0.484/0.936/0.956 | 0.491/0.461 | 0.345 | 0.551 | 0.826 |
| 9 | 0.990/0.925/0.984 | 0.159/0.171 | 0.772 | 0.648 | 0.811 |
| 10 | 0.991/0.951/0.990 | 0.162/0.157 | 0.779 | 0.672 | 0.837 |
| 11 | 0.982/0.573/0.957 | 0.425/0.428 | 0.385 | 0.513 | 0.803 |
| 12 | 0.964/0.782/0.980 | 0.334/0.332 | 0.623 | 0.414 | 0.740 |
| 13 | 0.986/0.721/0.721 | 0.373/0.553 | 0.689 | 0.579 | 0.764 |
| 14 | 0.931/0.684/0.960 | 0.873/0.294 | 0.250 | 0.680 | 0.850 |
| 15 | 0.975/0.867/0.917 | 0.167/0.361 | 0.936 | 0.747 | 0.880 |
| 16 | 0.996/0.995/0.993 | 0.148/0.158 | 0.869 | 0.856 | 0.921 |
| 17 | 0.954/0.987/0.969 | 0.258/0.259 | 0.862 | 0.779 | 0.879 |

## Interpretation

Stable centers establish reproducible route density, not automatically a discrete operator vocabulary. High-margin, well-separated layers support coalition-like states; low-margin layers are better described as continuous cones or manifolds even when any fixed k replicates. The rank-4 comparison is parameter-matched for basis storage but not runtime code: centroids use a discrete index, whereas the subspace uses continuous coordinates.
