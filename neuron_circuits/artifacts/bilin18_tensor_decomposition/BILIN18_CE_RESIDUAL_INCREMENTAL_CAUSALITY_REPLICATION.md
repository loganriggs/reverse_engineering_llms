# Incremental finite causality of novel CE-residual route axes

On fresh FineWeb, the known semantic modes plus a generic rank-four route projection form the baseline. The newly discovered rank-two generic-complement axes, equal-rank random complement controls, and a CE-selected residual rank-four reference are added before recomputing the target-layer MLP and complete suffix.

## Locked hypotheses

- **I0_native_path_is_exact: supported.** Observed: `{'maximum_logit_error': 4.863739013671875e-05}`
- **I1_novel_axes_add_finite_logit_coverage: supported.** Observed: `{'by_layer': {1: 0.020058817299962284, 2: 0.0690328678703509, 4: 0.11193523143317274, 5: 0.007540943853306836}, 'mean': 0.05214196511419819}`
- **I2_novel_axes_add_prediction_relevant_coverage: supported.** Observed: `{'by_layer': {1: 0.039279794484535224, 2: 0.1352782183894814, 4: 0.09843532867004745, 5: 0.002004684466636153}, 'mean': 0.06874950650267506}`
- **I3_novel_axes_beat_equal_rank_random_complements: supported.** Observed: `{'observed': 0.05214196511419819, 'null_means': [0.03911565359886707, 0.05189903282202726, 0.03789021231637871, 0.03347988825042453, 0.03821494738150702, 0.03728983963973723, 0.0379640029697024, 0.04145701496072782, 0.047205915705088614, 0.04564567269491518, 0.031353471942452915, 0.037596189706681576, 0.04093630863420156, 0.04361747650232978, 0.04482825073590249, 0.03340346781158157, 0.03861875992488942, 0.04066572497165513, 0.040484248231695164, 0.047685118005587, 0.030496789607502073, 0.042677048607041884, 0.034877102153277356, 0.03731734949746385, 0.04142957350554227, 0.03437131101085486, 0.027568447236371862, 0.039727897088686015, 0.03440047215395178, 0.03860192819568778, 0.048290633380826065, 0.04200995568622268]}`
- **I4_ce_selected_rank4_is_a_strong_compact_reference: supported.** Observed: `{'mean': 0.932572254372563, 'by_layer': {'1': 0.9743310365967673, '2': 0.8836856868331286, '4': 0.8829783620085697, '5': 0.9892939320517857}}`

| L | Base logit/CE-effect fidelity | +novel2 logit/CE-effect | Improvement logit/CE | CE-rank4 logit/CE | Best random logit improvement |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.976/0.958 | 0.996/0.997 | +0.020/+0.039 | 0.974/0.909 | +0.019 |
| 2 | 0.890/0.783 | 0.959/0.919 | +0.069/+0.135 | 0.884/0.589 | +0.079 |
| 4 | 0.839/0.875 | 0.951/0.974 | +0.112/+0.098 | 0.883/0.938 | +0.113 |
| 5 | 0.981/0.993 | 0.989/0.995 | +0.008/+0.002 | 0.989/0.996 | +0.012 |

## Guardrail

The new axes were chosen after an earlier novelty screen, while this text panel is fresh. Improvement shows incremental causal utility for this frozen coordinate code; it does not turn each PCA pole into a discrete semantic circuit. Random controls are orthogonal to the generic basis but are not optimized on discovery CE, so the comparison tests selection value rather than parameter-count-only reconstruction.
