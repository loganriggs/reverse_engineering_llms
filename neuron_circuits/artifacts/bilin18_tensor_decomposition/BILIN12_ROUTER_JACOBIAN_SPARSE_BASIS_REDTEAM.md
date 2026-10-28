# Sparse-basis red-team of Jacobian semantics

Only 23 predicates with adequate support on discovery FineWeb and every evaluation corpus are eligible. Top-16 PCA scores are whitened before ICA, varimax, and 20 matched random rotations. ICA start selection uses excess kurtosis only, never semantic labels.

## Locked hypotheses

- **R0_pca_result_survives_coverage_control: supported.**
- **R1_ica_improves_individual_semantics: not supported.**
- **R2_ica_is_privileged_over_random_rotations: not supported.**
- **R3_varimax_improves_individual_semantics: not supported.**
- **R4_ica_coordinates_are_reproducible: not supported.**
- **R5_sparse_basis_labels_are_diverse: supported.**

| Corpus | PCA | ICA | Varimax | Random p95 |
|---|---:|---:|---:|---:|
| fineweb | 0.782 | 0.690 | 0.665 | 0.812 |
| wikitext | 0.750 | 0.675 | 0.702 | 0.839 |
| tinystories | 0.741 | 0.736 | 0.732 | 0.872 |

ICA cross-start greedy overlaps: `[0.721, 0.736, 0.742, 0.735]`. Better sparse basis by mean held-out median: **ica**; distinct selected predicates: **10**.

A sparse rotation that transfers and beats random rotations would supply a more canonical verb basis than PCA. If sparse rotations improve but do not beat random controls or reproduce across starts, they are useful coordinate choices rather than uniquely identified model objects.
