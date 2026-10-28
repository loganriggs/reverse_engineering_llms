# Symmetry red-team of the late-MLP path program

`Left` and `Right` are exchangeable factors of each bilinear neuron, so an ontology that treats them as distinct is not canonical. Summing `L+R` within every active layer quotients that exact symmetry and reduces 81 paths to 16 invariant layer-subset paths: residual `I` or full bilinear derivative `B` at each of Layers 7–10.

## Locked hypotheses

- **R0_symmetry_quotient_is_exact: supported.**
- **R1_frozen_four_group_program_is_compact: supported.**
- **R2_frozen_eight_group_program_is_high_fidelity: not supported.**
- **R3_group_program_beats_uniform_random: supported.**
- **R4_group_program_beats_degree_matched_random: not supported.**
- **R5_degree_one_truncation_explains_compactness: supported.**
- **R6_context_specific_group_selection_is_small: supported.**
- **R7_group_program_transfers_ood: supported.**
- **R8_group_program_predicts_finite_effects: supported.**

Maximum quotient recomposition error: `2.05e-07`.

Frozen group order: `IIII`, `IIIB`, `BIII`, `IIBI`, `BBII`, `BIIB`, `IBBI`, `IBII`, `IBIB`, `BBIB`, `BIBI`, `BBBI`, `IIBB`, `IBBB`, `BIBB`, `BBBB`.

| Corpus | K | Frozen | Oracle | Uniform p95 | Degree-matched p95 | Finite effect |
|---|---:|---:|---:|---:|---:|---:|
| fineweb | 1 | 0.714 | 0.714 | 0.714 | 0.714 | 0.714 |
| fineweb | 2 | 0.797 | 0.799 | 0.726 | 0.797 | 0.797 |
| fineweb | 4 | 0.878 | 0.884 | 0.817 | 0.878 | 0.878 |
| fineweb | 5 | 0.902 | 0.908 | 0.834 | 0.899 | 0.902 |
| fineweb | 8 | 0.949 | 0.953 | 0.916 | 0.949 | 0.949 |
| fineweb | 12 | 0.984 | 0.986 | 0.972 | 0.983 | 0.984 |
| fineweb | 16 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| fineweb | degree≤1 (5) | 0.880 | — | — | — | — |
| wikitext | 1 | 0.729 | 0.729 | 0.154 | 0.729 | 0.729 |
| wikitext | 2 | 0.815 | 0.817 | 0.751 | 0.815 | 0.815 |
| wikitext | 4 | 0.901 | 0.903 | 0.828 | 0.901 | 0.901 |
| wikitext | 5 | 0.920 | 0.925 | 0.867 | 0.918 | 0.920 |
| wikitext | 8 | 0.961 | 0.965 | 0.940 | 0.963 | 0.961 |
| wikitext | 12 | 0.989 | 0.991 | 0.979 | 0.989 | 0.989 |
| wikitext | 16 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| wikitext | degree≤1 (5) | 0.912 | — | — | — | — |
| tinystories | 1 | 0.743 | 0.743 | 0.743 | 0.743 | 0.743 |
| tinystories | 2 | 0.827 | 0.828 | 0.781 | 0.827 | 0.827 |
| tinystories | 4 | 0.904 | 0.906 | 0.841 | 0.904 | 0.904 |
| tinystories | 5 | 0.922 | 0.927 | 0.878 | 0.921 | 0.922 |
| tinystories | 8 | 0.961 | 0.966 | 0.939 | 0.963 | 0.961 |
| tinystories | 12 | 0.989 | 0.991 | 0.981 | 0.989 | 0.989 |
| tinystories | 16 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| tinystories | degree≤1 (5) | 0.911 | — | — | — | — |

If the discovery-free degree≤1 program matches the frozen program, compactness mostly reflects a low-order residual expansion. If particular layer subsets beat degree-matched controls, the ordering contains additional learned structure. Either way, I/B groups are more canonical than individual L/R arms.
