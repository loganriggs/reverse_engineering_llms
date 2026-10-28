# Deterministic optimizer red-team for the Layer-16 behavior subspace

Rank-64 projectors start from the forward-PCA span and receive exact full-training-set gradients of the `J P h` reconstruction error. The gradient is projected onto the Grassmann tangent space and each update is retracted by QR. Learning rate and checkpoint are selected only on the disjoint FineWeb validation panel.

## Locked hypotheses

- **R0_factor_collection_is_exact: supported.**
- **R1_full_batch_optimization_improves_validation: not supported.**
- **R2_optimized_rank64_is_sufficient: not supported.**
- **R3_optimized_rank64_beats_forward_pca: not supported.**
- **R4_optimized_subspace_transfers_ood: not supported.**
- **R5_training_improvement_is_not_pure_overfit: supported.**
- **R6_selected_basis_generalizes_to_fineweb: supported.**

Autograd error: `6.27e-07`; maximum contraction error: `7.78e-07`.

| Learning rate | Initial train | Best-step train | Initial validation | Best validation | Step |
|---:|---:|---:|---:|---:|---:|
| 0.0001 | 0.758 | 0.760 | 0.758 | 0.758 | 100 |
| 0.0003 | 0.758 | 0.764 | 0.758 | 0.758 | 100 |
| 0.001 | 0.758 | 0.777 | 0.758 | 0.758 | 100 |
| 0.003 | 0.758 | 0.804 | 0.758 | 0.759 | 100 |

Selected learning rate: `0.003`.

| Corpus | Forward PCA | Optimized | Difference |
|---|---:|---:|---:|
| fineweb | 0.804 | 0.806 | +0.002 |
| wikitext | 0.704 | 0.704 | +0.000 |
| tinystories | 0.847 | 0.848 | +0.000 |

This red-team can distinguish stochastic-optimizer failure from a stable shared-subspace limitation. It still cannot prove global optimality: the rank-constrained projector objective is non-convex, and every tested coordinate is dense and nonsemantic.
