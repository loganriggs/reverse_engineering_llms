# Recursive feature-ontology red team

The strongest residual examples from the primitive fifty-feature model suggested conjunctions rather than wholly unrelated concepts. Before viewing a third panel, nine mutually exclusive current-token classes and nine next-token classes were frozen; their 81 pairwise interactions were added to the original current/next feature and position design.

## Locked hypotheses

- **R0_third_panel_is_fresh: supported.** Observed: `{'range': [15512, 15767]}`
- **R1_recursive_interactions_improve_write_explanation: not supported.** Observed: `{'passing': 0}`
- **R2_recursive_features_explain_prior_residual_axes: supported.** Observed: `{'passing': 3, 'axes': [{'layer': 1, 'axis': 0, 'best_interaction': {'feature': 'byte_fragment->continuation', 'auroc': 0.9905386709510421, 'positive_pole': True}}, {'layer': 1, 'axis': 1, 'best_interaction': {'feature': 'function->byte_fragment', 'auroc': 0.8869213942422839, 'positive_pole': False}}, {'layer': 12, 'axis': 0, 'best_interaction': {'feature': 'function->byte_fragment', 'auroc': 0.841197102513745, 'positive_pole': False}}, {'layer': 17, 'axis': 0, 'best_interaction': {'feature': 'function->byte_fragment', 'auroc': 0.779085301271509, 'positive_pole': True}}, {'layer': 17, 'axis': 1, 'best_interaction': {'feature': 'function->byte_fragment', 'auroc': 0.8029081309510504, 'positive_pole': True}}, {'layer': 17, 'axis': 2, 'best_interaction': {'feature': 'byte_fragment->continuation', 'auroc': 0.8166951175971795, 'positive_pole': True}}, {'layer': 17, 'axis': 3, 'best_interaction': {'feature': 'function->continuation', 'auroc': 0.9052965394458663, 'positive_pole': False}}, {'layer': 17, 'axis': 4, 'best_interaction': {'feature': 'byte_fragment->byte_fragment', 'auroc': 0.9241257302723608, 'positive_pole': False}}]}`
- **R3_residual_geometry_survives_recursive_expansion: supported.** Observed: `{'passing': 3}`
- **R4_recursive_expansion_reduces_late_residual_concentration: not supported.** Observed: `{'change': -0.006775435095905058}`
- **R5_late_compact_residual_code_survives: supported.** Observed: `{'r2': 0.966063157337776}`
- **R6_stable_unexplained_axes_remain: supported.** Observed: `{'count': 22}`

| Layer | Base/extended write R2 | Gain | Base/extended residual overlap | Extended known+rank16 R2 | Top-five residual share base/extended |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.191/0.217 | +0.026 | 0.892/0.904 | 0.372 | 0.119/0.105 |
| 8 | 0.043/0.041 | -0.001 | 0.796/0.796 | 0.165 | 0.076/0.074 |
| 12 | 0.069/0.069 | +0.000 | 0.711/0.697 | 0.169 | 0.077/0.071 |
| 17 | 0.489/0.517 | +0.028 | 0.952/0.951 | 0.966 | 0.893/0.886 |

## Prior frozen axes on the third panel

| Axis | Best recursive interaction | AUROC |
|---|---|---:|
| L1A0 | byte_fragment->continuation | 0.991 |
| L1A1 | function->byte_fragment | 0.887 |
| L12A0 | function->byte_fragment | 0.841 |
| L17A0 | function->byte_fragment | 0.779 |
| L17A1 | function->byte_fragment | 0.803 |
| L17A2 | byte_fragment->continuation | 0.817 |
| L17A3 | function->continuation | 0.905 |
| L17A4 | byte_fragment->byte_fragment | 0.924 |

## Guardrail

The interaction vocabulary is a prospective test of one recursive generalization prompted by earlier examples; it is not a license to add arbitrary labels until every axis is named. An interaction can explain axis membership without proving that the axis is causally dedicated to it. Stable residual geometry after expansion is evidence that the ontology remains incomplete.
