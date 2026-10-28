# Incremental finite causality of novel CE-residual route axes

On fresh FineWeb, the known semantic modes plus a generic rank-four route projection form the baseline. The newly discovered rank-two generic-complement axes, equal-rank random complement controls, and a CE-selected residual rank-four reference are added before recomputing the target-layer MLP and complete suffix.

## Locked hypotheses

- **I0_native_path_is_exact: supported.** Observed: `{'maximum_logit_error': 6.341934204101562e-05}`
- **I1_novel_axes_add_finite_logit_coverage: supported.** Observed: `{'by_layer': {1: 0.01819602591304892, 2: 0.07192670793950207, 4: 0.08968039305202391, 5: 0.007651038703564206}, 'mean': 0.046863541402034775}`
- **I2_novel_axes_add_prediction_relevant_coverage: supported.** Observed: `{'by_layer': {1: 0.016323568594346582, 2: 0.0900042069458813, 4: 0.12844706499113567, 5: 0.006339135059311496}, 'mean': 0.06027849389766876}`
- **I3_novel_axes_beat_equal_rank_random_complements: supported.** Observed: `{'observed': 0.046863541402034775, 'null_means': [0.03181561795043272, 0.03649815324782277, 0.045866708843872284, 0.043472772703888274, 0.030075504531400232, 0.04088004353050223, 0.029678877469573567, 0.03323720347014303, 0.0434558536235308, 0.03179447623274512, 0.0413171191504694, 0.03639543200200396, 0.0352625235770459, 0.038983281711533424, 0.03698684988083778, 0.04028125327697987, 0.043348365057264415, 0.03685482687918007, 0.04636351709922876, 0.04053207390510197, 0.033404882758773846, 0.033189589180276535, 0.03454374549848696, 0.03703480161534917, 0.04629708444634045, 0.03434125203357019, 0.042042213071391915, 0.03643440014855165, 0.03630408751886505, 0.03995281915818796, 0.03845588990508944, 0.0405335804341366]}`
- **I4_ce_selected_rank4_is_a_strong_compact_reference: supported.** Observed: `{'mean': 0.9236212149808698, 'by_layer': {'1': 0.983852015791596, '2': 0.868416179440595, '4': 0.8505996996200808, '5': 0.9916169650712072}}`

| L | Base logit/CE-effect fidelity | +novel2 logit/CE-effect | Improvement logit/CE | CE-rank4 logit/CE | Best random logit improvement |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.977/0.981 | 0.996/0.998 | +0.018/+0.016 | 0.984/0.995 | +0.019 |
| 2 | 0.878/0.869 | 0.950/0.959 | +0.072/+0.090 | 0.868/0.869 | +0.081 |
| 4 | 0.845/0.799 | 0.935/0.928 | +0.090/+0.128 | 0.851/0.829 | +0.096 |
| 5 | 0.983/0.979 | 0.991/0.985 | +0.008/+0.006 | 0.992/0.992 | +0.010 |

## Guardrail

The new axes were chosen after an earlier novelty screen, while this text panel is fresh. Improvement shows incremental causal utility for this frozen coordinate code; it does not turn each PCA pole into a discrete semantic circuit. Random controls are orthogonal to the generic basis but are not optimized on discovery CE, so the comparison tests selection value rather than parameter-count-only reconstruction.
