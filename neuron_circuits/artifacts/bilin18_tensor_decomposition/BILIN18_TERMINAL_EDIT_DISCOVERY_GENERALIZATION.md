# Prospective terminal edit discovery beyond spelling

Natural one-token swaps are propagated through all 18 layers. Layer-17 post-boundary head edits are ranked either by routed weight-mode occupancy alone or by their analytic alignment with the model's own changed next-token distribution. The latter uses the original-output Fisher metric and is validated with finite interventions through the final bilinear MLP and normalization.

## Locked results

- **T0_panels_are_balanced: supported.**
- **T1_tensor_quotient_predicts_finite_head_effects: supported.**
- **T2_context_only_selection_transfers_within_family: supported.**
- **T3_output_conditioned_selection_transfers_within_family: supported.**
- **T4_H2_is_a_universal_terminal_edit_compressor: not supported.**
- **T5_H2_weight_mode_beats_random_rank1_controls: not supported.**
- **T6_H2_postboundary_relation_is_privileged: not supported.**

| Family/split | N | endpoint Fisher energy | winner | H2 ranks spectral/occupancy/routed/quotient/finite | direct→finite r (without winner) |
|---|---:|---:|---:|---:|---:|
| spelling/0 | 80 | 0.01037 | H2 | 2/1/1/1/1 | 1.000 (1.000) |
| spelling/1 | 80 | 0.009996 | H2 | 2/1/1/1/1 | 1.000 (1.000) |
| unrelated/0 | 80 | 0.003574 | H6 | 2/3/2/5/5 | 1.000 (1.000) |
| unrelated/1 | 80 | 0.003828 | H6 | 2/3/2/8/8 | 1.000 (1.000) |

## Frozen selections

| Family | context selection/rank | quotient selection/rank | H2 held-out rank |
|---|---:|---:|---:|
| spelling | H2 / 1 | H2 / 1 | 1 |
| unrelated | H6 / 1 | H6 / 1 | 8 |

## Interpretation boundary

The endpoint is the model's own output change, not a human-labeled concept. Success therefore establishes behavioral mechanism discovery, not semantic interpretation. Routed write magnitude can identify an occupied transport operation without saying whether it helps the relevant output change; the Fisher quotient supplies that missing destination type. Failure of universal H2 reuse would not contradict the earlier generic-axis result: many edits can share one H2 value direction without H2 being their dominant terminal consumer.
