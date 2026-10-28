# Frozen attention-head transfer at Layers 0 and 1

Top-three heads are frozen from the original semantic substitutions in two FineWeb contexts, then evaluated on seventeen new lexical pairs and both FineWeb and TinyStories. Global and family-calibrated rankings are compared with a state-free tensor-capacity ranking and a per-example target oracle.

## Locked hypotheses

- **H0_layer0_has_a_fixed_transport_module: not supported.**
- **H1_layer1_has_a_fixed_transport_module: supported.**
- **H2_family_specific_heads_improve_global: not supported.**
- **H3_calibrated_heads_beat_static_capacity: supported.**
- **H4_frozen_heads_transfer_ood: supported.**
- **H5_discovery_head_sets_are_stable: supported.**
- **H6_family_head_sets_are_distinct: not supported.**

| Layer | Corpus | Frozen global | Frozen family | Static capacity | Per-state | Global head IDs |
|---:|---|---:|---:|---:|---:|---:|
| 0 | fineweb | 0.752 | 0.696 | 0.123 | 0.916 | [7, 4, 2] |
| 0 | tinystories | 0.692 | 0.675 | 0.165 | 0.926 | [7, 4, 2] |
| 1 | fineweb | 0.634 | 0.694 | 0.184 | 0.883 | [8, 0, 2] |
| 1 | tinystories | 0.656 | 0.675 | 0.179 | 0.832 | [8, 0, 2] |

| Layer | Context-bank Jaccard | Same-family overlap | Different-family overlap |
|---:|---:|---:|---:|
| 0 | 0.500 | 0.500 | 0.350 |
| 1 | 0.500 | 0.500 | 0.500 |

## Post-run exhaustive subset red-team

All 84 native three-head subsets are evaluated on the same prospective examples. This diagnostic was added after observing the frozen result and is not a locked hypothesis.

| Layer | Corpus | Median subset | Frozen global (percentile) | Best fixed heads / score | Gap to best |
|---:|---|---:|---:|---:|---:|
| 0 | fineweb | 0.153 | 0.752 (0.988) | [0, 4, 7] / 0.774 | 0.022 |
| 0 | tinystories | 0.130 | 0.692 (0.774) | [1, 3, 7] / 0.768 | 0.077 |
| 1 | fineweb | 0.197 | 0.634 (0.845) | [3, 4, 8] / 0.720 | 0.086 |
| 1 | tinystories | 0.229 | 0.656 (0.857) | [1, 4, 8] / 0.695 | 0.039 |

A strong frozen-global result identifies a reusable transport module after minimal calibration; a family advantage would additionally support semantic specialization. Failure of the raw capacity ranking only shows that operator magnitude is not an execution map.
