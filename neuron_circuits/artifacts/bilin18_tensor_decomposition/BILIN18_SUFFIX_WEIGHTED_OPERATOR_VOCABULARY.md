# Suffix-weighted early operator vocabulary

The exact bilinear operator tensor is weighted by real incoming adjoints obtained by transporting the stable residual-output subspace through the complete native suffix. Random tangent probes provide the second tensor leg. Bases are fit on disjoint FineWeb and evaluated on new residual objectives plus held-out human semantic objectives. Frobenius-operator modes and state PCA use the same state covariance and rank budgets.

## Locked hypotheses

- **T0_independent_q_sketches_recover_task_subspace: not supported.**
- **T1_task_rank64_reconstructs_residual_objectives: not supported.**
- **T2_task_rank64_beats_frobenius: not supported.**
- **T3_task_rank64_beats_state_pca: not supported.**
- **T4_task_rank64_transfers_to_semantics: not supported.**
- **T5_task_rank256_is_high_fidelity: not supported.**
- **T6_adjoint_magnitude_changes_the_basis: supported.**

![Suffix-weighted vocabulary](bilin18_suffix_weighted_operator_vocabulary.png)

| Block | Unit stability | Raw/unit overlap | Residual unit64 / Frobenius / PCA | Residual unit256 | Semantic unit64 / Frobenius / PCA | Semantic unit256 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.743 | 0.543 | 0.747/0.752/0.751 | 0.888 | 0.707/0.705/0.713 | 0.867 |
| 2 | 0.726 | 0.549 | 0.689/0.702/0.691 | 0.868 | 0.618/0.621/0.612 | 0.836 |
| 3 | 0.721 | 0.552 | 0.702/0.718/0.708 | 0.862 | 0.643/0.650/0.650 | 0.843 |
| 4 | 0.699 | 0.542 | 0.674/0.681/0.670 | 0.851 | 0.641/0.653/0.639 | 0.830 |
| 5 | 0.730 | 0.553 | 0.682/0.696/0.681 | 0.844 | 0.672/0.679/0.676 | 0.830 |

## Guardrails

This is a declared residual-output metric, not a universal operator ontology. The suffix-weighted basis uses native forward states and exact suffix adjoints on discovery data but no semantic labels. Transfer to the semantic panel tests reuse beyond its fitting objectives. Unit normalization asks for recurring directions; the raw variant allows high-gain contexts to dominate. Local response reconstruction is not yet a simultaneous multi-block causal replacement.
