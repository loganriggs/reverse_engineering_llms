# Unseeded natural route-coalition geometry

High-magnitude natural attention edges are represented only by their normalized signed nine-head route vectors. Four-cluster layer-local partitions are fit independently on two disjoint FineWeb panels and matched without decoding token strings. Per-edge head permutations provide a destroyed-binding null.

## Locked hypotheses

- **U0_two_disjoint_panels_are_used: supported.** Observed: `{'discovery_range': [0, 255], 'replication_range': [512, 767]}`
- **U1_discrete_route_coalitions_replicate: supported.** Observed: `{'stable_layers': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17], 'mean_cosines': {'0': 0.9896061420440674, '1': 0.9963517785072327, '2': 0.9867435693740845, '3': 0.9966874122619629, '4': 0.9953221678733826, '5': 0.99919193983078, '6': 0.9935919642448425, '7': 0.9037869572639465, '8': 0.9364131689071655, '9': 0.9280986189842224, '10': 0.9658204913139343, '11': 0.5728024840354919, '12': 0.7879886627197266, '13': 0.7204036116600037, '14': 0.6859868764877319, '15': 0.9011446237564087, '16': 0.9945657849311829, '17': 0.9872371554374695}}`
- **U2_recurrence_beats_destroyed_head_binding: supported.** Observed: `{'real_mean': 0.9078746338685354, 'null_mean': 0.30227539123136, 'null_p95': 0.32709276435539947}`
- **U3_stable_candidates_exist: supported.** Observed: `{'eligible': 59, 'selected': 16}`
- **U4_stable_candidates_are_not_only_single_heads: supported.** Observed: `{'multihead': 13, 'selected': 16}`

| Layer | Mean matched cosine | Cluster cosines |
|---:|---:|---|
| 0 | 0.990 | 0.999, 0.983, 0.992, 0.985 |
| 1 | 0.996 | 0.999, 0.994, 1.000, 0.993 |
| 2 | 0.987 | 0.999, 0.952, 0.997, 0.999 |
| 3 | 0.997 | 0.994, 0.999, 0.996, 0.999 |
| 4 | 0.995 | 0.995, 0.996, 0.998, 0.992 |
| 5 | 0.999 | 0.999, 1.000, 1.000, 0.998 |
| 6 | 0.994 | 0.994, 0.996, 0.998, 0.987 |
| 7 | 0.904 | 0.993, 0.994, 0.978, 0.650 |
| 8 | 0.936 | 0.983, 0.997, 0.983, 0.783 |
| 9 | 0.928 | 0.906, 0.844, 0.973, 0.989 |
| 10 | 0.966 | 0.978, 0.980, 0.969, 0.937 |
| 11 | 0.573 | 0.992, 0.987, -0.620, 0.933 |
| 12 | 0.788 | 0.978, 0.745, 0.516, 0.912 |
| 13 | 0.720 | 0.981, 0.950, -0.008, 0.957 |
| 14 | 0.686 | 0.969, 0.991, -0.205, 0.989 |
| 15 | 0.901 | 0.736, 0.971, 0.978, 0.920 |
| 16 | 0.995 | 0.999, 0.999, 0.987, 0.994 |
| 17 | 0.987 | 0.992, 0.999, 0.973, 0.985 |

## Frozen candidates for semantic discovery

| Rank | Layer/cluster | Center cosine | D/R size | D/R coherence | Effective heads |
|---:|---|---:|---:|---:|---:|
| 1 | L5C2 | 1.000 | 366/368 | 0.997/0.997 | 1.99 |
| 2 | L5C1 | 1.000 | 310/346 | 0.994/0.995 | 2.44 |
| 3 | L5C0 | 0.999 | 208/158 | 0.995/0.993 | 1.71 |
| 4 | L5C3 | 0.998 | 140/152 | 0.988/0.991 | 2.51 |
| 5 | L16C0 | 0.999 | 583/594 | 0.945/0.944 | 5.68 |
| 6 | L16C1 | 0.999 | 123/132 | 0.942/0.945 | 2.09 |
| 7 | L1C2 | 1.000 | 303/270 | 0.918/0.927 | 3.17 |
| 8 | L15C2 | 0.978 | 401/696 | 0.953/0.933 | 2.49 |
| 9 | L6C0 | 0.994 | 382/376 | 0.909/0.909 | 4.89 |
| 10 | L16C3 | 0.994 | 280/244 | 0.903/0.914 | 5.16 |
| 11 | L17C0 | 0.992 | 427/320 | 0.905/0.906 | 5.99 |
| 12 | L1C0 | 0.999 | 417/395 | 0.894/0.902 | 1.93 |
| 13 | L6C2 | 0.998 | 204/193 | 0.891/0.921 | 2.01 |
| 14 | L17C3 | 0.985 | 261/349 | 0.897/0.895 | 6.33 |
| 15 | L0C0 | 0.999 | 410/384 | 0.883/0.882 | 2.20 |
| 16 | L4C0 | 0.995 | 331/309 | 0.901/0.885 | 3.78 |

## Guardrail

These are input-conditioned route-coordinate clusters, not semantic features. Candidate selection used no decoded tokens, but the high-magnitude edge sampler and fixed k=4 are part of the ontology. If clusters fail recurrence or the destroyed-binding null, a continuous routing-field description should be preferred.
