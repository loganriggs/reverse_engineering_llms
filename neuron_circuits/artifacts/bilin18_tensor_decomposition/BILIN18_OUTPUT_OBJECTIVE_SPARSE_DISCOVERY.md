# Label-free discovery of output objectives

ICA and varimax rotate the stable rank-16 output tensor core using only its weight-derived vocabulary loadings. Human-readable token predicates are consulted only after the bases are frozen. Native HOSVD axes and matched random rotations of the identical physical subspace are controls.

## Locked hypotheses

- **O0_basis_fit_is_label_free: supported.**
- **O1_sparse_objective_improves_kurtosis: supported.**
- **O2_sparse_objective_improves_weight_semantics: not supported.**
- **O3_sparse_objective_is_semantically_privileged: not supported.**
- **O4_sparse_coordinates_are_reproducible: supported.**
- **O5_discovered_objectives_are_diverse: supported.**
- **O6_natural_semantics_transfer: supported.**

Selected label-free sparse basis: **varimax**.

| Basis | Weight semantic AUROC | Weight kurtosis | Split stability | Natural target/top1 AUROC | Distinct labels |
|---|---:|---:|---:|---:|---:|
| native | 0.828 | 0.777 | — | 0.896/0.913 | 9 |
| ica | 0.834 | 0.550 | 0.614 | 0.932/0.953 | 9 |
| varimax | 0.803 | 2.960 | 0.992 | 0.905/0.923 | 11 |
| random rotation p95 | 0.842 | 0.717 | — | 0.946/0.960 | — |

## Discovered weight-side labels

| Axis | ICA | Varimax |
|---:|---|---|
| 0 | `coarse:preposition` (0.83) | `coarse:punctuation` (0.68) |
| 1 | `flag:suffix_ed` (0.74) | `coarse:pronoun` (0.80) |
| 2 | `flag:suffix_ing` (0.90) | `flag:code_operator` (0.76) |
| 3 | `coarse:conjunction` (0.89) | `coarse:auxiliary` (0.86) |
| 4 | `flag:suffix_ing` (0.89) | `coarse:subword` (0.80) |
| 5 | `flag:code_operator` (0.70) | `coarse:subword` (0.88) |
| 6 | `coarse:conjunction` (0.84) | `generic:capitalized` (0.81) |
| 7 | `coarse:content_word` (0.89) | `generic:digit` (0.76) |
| 8 | `coarse:auxiliary` (0.75) | `coarse:auxiliary` (0.74) |
| 9 | `flag:suffix_ing` (0.85) | `coarse:pronoun` (0.82) |
| 10 | `coarse:auxiliary` (0.81) | `flag:suffix_ed` (0.90) |
| 11 | `coarse:auxiliary` (0.82) | `flag:suffix_ing` (0.96) |
| 12 | `flag:suffix_ing` (0.86) | `coarse:preposition` (0.88) |
| 13 | `coarse:determiner` (0.89) | `coarse:auxiliary` (0.69) |
| 14 | `coarse:auxiliary` (0.68) | `flag:code_operator` (0.85) |
| 15 | `flag:all_caps` (0.79) | `coarse:conjunction` (0.67) |

## Guardrail

A sparse rotation is a candidate coordinate system inside an already discovered physical subspace. High semantic AUROC is not enough: it must beat matched rotations and reproduce from disjoint weight samples before individual axes count as model-preferred objectives. If only the whole subspace passes, the correct ontology remains a semantic output subspace with non-identifiable internal coordinates.
