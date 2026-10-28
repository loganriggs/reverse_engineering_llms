# Invariant late-MLP path replication in the 546M FineWeb model

The checkpoint has **545,902,902 parameters**. Natural FineWeb PCA supplies separate 32D input and output charts around the Layer-12→17 suffix. With attention responses frozen, Layers 13–16 are expanded into the same 16 symmetry-invariant `I/B` layer-subset paths used in the independent 190M model. Ordering is learned on discovery FineWeb and frozen for disjoint FineWeb, WikiText, and TinyStories.

## Locked hypotheses

- **X0_checkpoint_is_over_500m: supported.**
- **X1_grouped_path_sum_is_exact: supported.**
- **X2_frozen_four_group_program_is_compact: supported.**
- **X3_frozen_eight_group_program_is_high_fidelity: supported.**
- **X4_degree_one_truncation_is_sufficient: supported.**
- **X5_group_program_predicts_finite_effects: supported.**
- **X6_group_program_transfers_ood: supported.**
- **X7_native_suffix_is_low_order: supported.**
- **X8_context_specific_selection_is_small: supported.**

Maximum independent-JVP recomposition error: `8.53e-07`.

Frozen group order: `IIII`, `IIIB`, `IIBB`, `IIBI`, `BIII`, `IBIB`, `IBII`, `BIIB`, `BIBB`, `IBBB`, `BIBI`, `BBII`, `IBBI`, `BBBI`, `BBIB`, `BBBB`.

| Corpus | K | Frozen | Oracle | Random p95 | Finite effect |
|---|---:|---:|---:|---:|---:|
| fineweb | 1 | 0.749 | 0.749 | 0.749 | 0.749 |
| fineweb | 2 | 0.908 | 0.908 | 0.751 | 0.908 |
| fineweb | 4 | 0.961 | 0.963 | 0.829 | 0.961 |
| fineweb | 5 | 0.973 | 0.975 | 0.924 | 0.973 |
| fineweb | 8 | 0.990 | 0.991 | 0.971 | 0.990 |
| fineweb | 12 | 0.998 | 0.998 | 0.993 | 0.998 |
| fineweb | 16 | 1.000 | 1.000 | 1.000 | 1.000 |
| fineweb | degree≤1 (5), K90=2 | 0.954 | — | — | — |
| wikitext | 1 | 0.724 | 0.724 | 0.724 | 0.724 |
| wikitext | 2 | 0.905 | 0.905 | 0.747 | 0.905 |
| wikitext | 4 | 0.964 | 0.966 | 0.865 | 0.964 |
| wikitext | 5 | 0.974 | 0.977 | 0.873 | 0.974 |
| wikitext | 8 | 0.990 | 0.992 | 0.972 | 0.990 |
| wikitext | 12 | 0.998 | 0.998 | 0.993 | 0.998 |
| wikitext | 16 | 1.000 | 1.000 | 1.000 | 1.000 |
| wikitext | degree≤1 (5), K90=2 | 0.945 | — | — | — |
| tinystories | 1 | 0.747 | 0.747 | 0.259 | 0.747 |
| tinystories | 2 | 0.919 | 0.919 | 0.777 | 0.919 |
| tinystories | 4 | 0.971 | 0.972 | 0.840 | 0.971 |
| tinystories | 5 | 0.979 | 0.983 | 0.931 | 0.979 |
| tinystories | 8 | 0.995 | 0.995 | 0.978 | 0.995 |
| tinystories | 12 | 0.999 | 0.999 | 0.995 | 0.999 |
| tinystories | 16 | 1.000 | 1.000 | 1.000 | 1.000 |
| tinystories | degree≤1 (5), K90=2 | 0.954 | — | — | — |

This is an exact local account of the declared attention-frozen suffix route, not an explanation of how text selects the input perturbation or how attention carries it. Replication would establish that low interaction order is not peculiar to the smaller checkpoint; failure would make the regime architecture- or scale-dependent.
