# Natural causal reconstruction by the sparse router graph

At 9,600 natural sites per corpus, the exact pooled rank-32 component of the native Layer-11 MLP write is removed and replaced by the complete restricted quadratic program, its diagonal-plus-32-edge approximation, a diagonal-only program, or a matched random-edge graph. Orthogonal native write components and final RMSNorm are preserved.

## Locked hypotheses

- **C0_full_restricted_program_recovers_natural_effect: supported.**
- **C1_selected_graph_recovers_natural_effect: supported.**
- **C2_selected_graph_matches_full_restricted: supported.**
- **C3_selected_graph_beats_random: supported.**
- **C4_selected_graph_preserves_predictions: not supported.**
- **C5_boundary_sites_are_not_the_only_use: supported.**

| Corpus | Full/selected/diagonal/random KL recovery | Selected top-1 / CE delta | Newline/non-newline selected recovery | Sites |
|---|---:|---:|---:|---:|
| fineweb | 0.953/0.940/0.687/0.718 | 0.887 / 0.0457 | 0.975/0.938 | 9600 |
| wikitext | 0.939/0.920/0.594/0.615 | 0.876 / 0.0897 | 0.982/0.918 | 9600 |
| tinystories | 0.961/0.938/0.654/0.678 | 0.897 / 0.0073 | 0.985/0.934 | 9600 |

KL recovery is normalized against deleting the exact pooled component: 1 means native-equivalent, 0 means no better than deletion, and negative values are worse. This is a replacement test of a declared output component, not a claim that the graph is independently necessary when all other final-layer directions may compensate.
