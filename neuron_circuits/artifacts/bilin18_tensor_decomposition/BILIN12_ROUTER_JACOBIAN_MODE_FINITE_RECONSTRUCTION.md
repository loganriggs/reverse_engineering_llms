# Finite causal sufficiency of Jacobian spectral modes

A FineWeb-only PCA dictionary of weight-derived Layer-6 transport Jacobians predicts one-sided interventions of scale 0.5. Predictions are scored against the finite attention-frozen MLP path, the complete native path, changes in the selected sparse edge products, and changes in all 32 restricted quadratic output coefficients. This tests predictive causal sufficiency; operator modes are not uniquely removable physical components.

## Locked hypotheses

- **B0_rank16_is_finitely_sufficient: supported.**
- **B1_rank32_is_high_fidelity: supported.**
- **B2_rank32_recovers_sparse_grammar_changes: supported.**
- **B3_outliers_beat_equal_size_bulk: supported.**
- **B4_causal_sufficiency_is_rotation_invariant: supported.**
- **B5_rank32_predicts_native_transport: supported.**
- **B6_leading_modes_are_necessary_for_compact_reconstruction: supported.**

| Corpus | Mean / r16 / r32 / r128 coordinate | r32 edge / coefficient | r32 native | Bulk16 | r128 minus top16 |
|---|---:|---:|---:|---:|---:|
| fineweb | 0.792 / 0.871 / 0.893 / 0.929 | 0.909 / 0.912 | 0.868 | 0.814 | 0.850 |
| wikitext | 0.807 / 0.886 / 0.905 / 0.940 | 0.914 / 0.915 | 0.877 | 0.827 | 0.861 |
| tinystories | 0.814 / 0.891 / 0.910 / 0.942 | 0.922 / 0.924 | 0.892 | 0.834 | 0.866 |

Randomly rotating the top-16 operator subspace changes its reconstructed response by at most `1.19e-07`. Thus causal sufficiency belongs to the subspace, whereas individual semantic mode labels can vary under an equally sufficient basis.
