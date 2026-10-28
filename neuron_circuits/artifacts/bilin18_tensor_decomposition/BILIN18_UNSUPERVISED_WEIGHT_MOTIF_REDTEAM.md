# Unsupervised weight-motif red team

The frozen native motifs were recomputed after removing the persistent layer-0 value bus, and from address/write halves separately.

## Results

| representation | edges | pairwise r | neighbor Jaccard | same head index | same layer | adjacent layer | native-edge Jaccard |
|---|---:|---:|---:|---:|---:|---:|---:|
| native_joint | 64 | 0.812 | 0.500 | 0.328 | 0.062 | 0.375 | — |
| address_only | 14 | 0.418 | 0.000 | 0.071 | 0.143 | 0.500 | 0.08333333333333333 |
| native_write_only | 79 | 0.861 | 0.500 | 0.392 | 0.051 | 0.354 | 0.6067415730337079 |
| current_joint | 62 | 0.724 | 0.200 | 0.097 | 0.097 | 0.500 | 0.4482758620689655 |
| current_write_only | 78 | 0.792 | 0.500 | 0.154 | 0.051 | 0.487 | — |
| bus_write_only | 79 | 0.921 | 0.500 | 0.962 | 0.013 | 0.165 | — |

Random-pair structural baselines: same_head_index_fraction=0.106, same_layer_fraction=0.050, adjacent_layer_fraction=0.106.

## Locked decisions

- **R1_native_edges_are_not_bus_index_artifacts: failed.**
- **R2_native_edges_survive_bus_removal: failed.**
- **R3_joint_motifs_are_not_address_only: passed.**
- **R4_joint_motifs_are_not_write_only: passed.**
