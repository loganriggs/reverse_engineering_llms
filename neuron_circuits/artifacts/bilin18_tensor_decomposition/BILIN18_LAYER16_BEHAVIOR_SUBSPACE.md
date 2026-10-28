# Behavior-optimized shared residual subspace for the Layer-16 loss path

An orthonormal projector is optimized directly inside the exact `J P h` contraction, starting from the forward-PCA span. Checkpoints are selected only by disjoint FineWeb validation. This tests whether PCA's variance objective—not shared-subspace breadth—caused the earlier low fidelity.

## Locked hypotheses

- **C0_factor_collection_is_exact: supported.**
- **C1_behavior_rank64_is_sufficient: not supported.**
- **C2_behavior_rank64_beats_forward_pca: not supported.**
- **C3_behavior_rank128_is_high_fidelity: not supported.**
- **C4_behavior_subspace_transfers_ood: not supported.**
- **C5_behavior_optimization_improves_validation: not supported.**
- **C6_behavior_subspace_generalizes_from_validation: supported.**
- **C7_rank32_behavior_subspace_is_useful: not supported.**

Autograd error: `6.27e-07`; maximum contraction error: `7.78e-07`.

| Rank | Initial validation | Best validation | Step |
|---:|---:|---:|---:|
| 32 | 0.622 | 0.622 | 0 |
| 64 | 0.758 | 0.758 | 0 |
| 128 | 0.798 | 0.798 | 0 |

| Corpus | Rank | Forward PCA | Behavior optimized |
|---|---:|---:|---:|
| fineweb | 32 | 0.717 | 0.717 |
| fineweb | 64 | 0.804 | 0.804 |
| fineweb | 128 | 0.825 | 0.825 |
| wikitext | 32 | 0.628 | 0.628 |
| wikitext | 64 | 0.704 | 0.704 |
| wikitext | 128 | 0.750 | 0.750 |
| tinystories | 32 | 0.792 | 0.792 |
| tinystories | 64 | 0.847 | 0.847 |
| tinystories | 128 | 0.867 | 0.867 |

A successful optimized subspace is a shared causal interface for this loss-path family. It is not automatically semantic or native-sparse: every coordinate remains a dense residual direction, and the site-specific forward tangent and effective adjoint are still required to execute the contraction.
