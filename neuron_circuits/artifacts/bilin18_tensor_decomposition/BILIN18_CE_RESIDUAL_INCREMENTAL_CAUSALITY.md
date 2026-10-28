# Incremental finite causality of novel CE-residual route axes

On fresh FineWeb, the known semantic modes plus a generic rank-four route projection form the baseline. The newly discovered rank-two generic-complement axes, equal-rank random complement controls, and a CE-selected residual rank-four reference are added before recomputing the target-layer MLP and complete suffix.

## Locked hypotheses

- **I0_native_path_is_exact: supported.** Observed: `{'maximum_logit_error': 4.673004150390625e-05}`
- **I1_novel_axes_add_finite_logit_coverage: supported.** Observed: `{'by_layer': {1: 0.01819601751685873, 2: 0.07192700457386525, 4: 0.08968109134385838, 5: 0.007651018794237796}, 'mean': 0.04686378305720504}`
- **I2_novel_axes_add_prediction_relevant_coverage: supported.** Observed: `{'by_layer': {1: 0.016323657469485053, 2: 0.0900027261048918, 4: 0.12844929950280004, 5: 0.006339046238574286}, 'mean': 0.060278682328937794}`
- **I3_novel_axes_beat_equal_rank_random_complements: supported.** Observed: `{'observed': 0.04686378305720504, 'null_means': [0.042990346531930484, 0.03886181374879946, 0.031557853211680004, 0.040432092143739745, 0.03266971511159927, 0.034849267997472705, 0.03253038528584859, 0.03796759888896062]}`
- **I4_ce_selected_rank4_is_a_strong_compact_reference: supported.** Observed: `{'mean': 0.9236212856263512, 'by_layer': {'1': 0.9838521623036351, '2': 0.8684158445840453, '4': 0.8506002244637503, '5': 0.9916169111539735}}`

| L | Base logit/CE-effect fidelity | +novel2 logit/CE-effect | Improvement logit/CE | CE-rank4 logit/CE | Best random logit improvement |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.977/0.981 | 0.996/0.998 | +0.018/+0.016 | 0.984/0.995 | +0.019 |
| 2 | 0.878/0.869 | 0.950/0.959 | +0.072/+0.090 | 0.868/0.869 | +0.074 |
| 4 | 0.845/0.799 | 0.935/0.928 | +0.090/+0.128 | 0.851/0.829 | +0.091 |
| 5 | 0.983/0.979 | 0.991/0.985 | +0.008/+0.006 | 0.992/0.992 | +0.009 |

## Guardrail

The new axes were chosen after an earlier novelty screen, while this text panel is fresh. Improvement shows incremental causal utility for this frozen coordinate code; it does not turn each PCA pole into a discrete semantic circuit. Random controls are orthogonal to the generic basis but are not optimized on discovery CE, so the comparison tests selection value rather than parameter-count-only reconstruction.
