# High-count invariant late-MLP path replication in the 546M FineWeb model

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

Maximum independent-JVP recomposition error: `9.26e-07`.

Frozen group order: `IIII`, `IIIB`, `IIBB`, `IIBI`, `BIIB`, `IBII`, `BIII`, `IBIB`, `BIBB`, `IBBB`, `BIBI`, `IBBI`, `BBII`, `BBBI`, `BBIB`, `BBBB`.

| Corpus | K | Frozen | Oracle | Random p95 | Finite effect |
|---|---:|---:|---:|---:|---:|
| fineweb | 1 | 0.662 | 0.662 | 0.662 | 0.662 |
| fineweb | 2 | 0.891 | 0.891 | 0.664 | 0.891 |
| fineweb | 4 | 0.958 | 0.959 | 0.786 | 0.958 |
| fineweb | 5 | 0.971 | 0.973 | 0.904 | 0.971 |
| fineweb | 8 | 0.989 | 0.991 | 0.964 | 0.989 |
| fineweb | 12 | 0.998 | 0.998 | 0.994 | 0.998 |
| fineweb | 16 | 1.000 | 1.000 | 1.000 | 1.000 |
| fineweb | degree≤1 (5), K90=3 | 0.938 | — | — | — |
| wikitext | 1 | 0.656 | 0.656 | 0.656 | 0.656 |
| wikitext | 2 | 0.890 | 0.890 | 0.660 | 0.890 |
| wikitext | 4 | 0.958 | 0.958 | 0.851 | 0.958 |
| wikitext | 5 | 0.971 | 0.973 | 0.853 | 0.971 |
| wikitext | 8 | 0.990 | 0.991 | 0.964 | 0.990 |
| wikitext | 12 | 0.998 | 0.998 | 0.993 | 0.998 |
| wikitext | 16 | 1.000 | 1.000 | 1.000 | 1.000 |
| wikitext | degree≤1 (5), K90=3 | 0.938 | — | — | — |
| tinystories | 1 | 0.672 | 0.672 | 0.549 | 0.672 |
| tinystories | 2 | 0.894 | 0.894 | 0.730 | 0.894 |
| tinystories | 4 | 0.963 | 0.964 | 0.793 | 0.963 |
| tinystories | 5 | 0.976 | 0.978 | 0.910 | 0.976 |
| tinystories | 8 | 0.993 | 0.994 | 0.950 | 0.993 |
| tinystories | 12 | 0.999 | 0.999 | 0.994 | 0.999 |
| tinystories | 16 | 1.000 | 1.000 | 1.000 | 1.000 |
| tinystories | degree≤1 (5), K90=3 | 0.941 | — | — | — |

This is an exact local account of the declared attention-frozen suffix route, not an explanation of how text selects the input perturbation or how attention carries it. Replication would establish that low interaction order is not peculiar to the smaller checkpoint; failure would make the regime architecture- or scale-dependent.
