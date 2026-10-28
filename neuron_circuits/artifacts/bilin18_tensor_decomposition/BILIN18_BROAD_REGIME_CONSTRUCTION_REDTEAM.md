# Broad-regime construction red team

This audit separates whether an attention branch is behaviorally material from whether that branch is internally compressible. It also compares each frozen discovery-selected top-three program against every one of the 84 possible three-head subsets on held-out pairs.

## Locked results

- **R0_exactness_survives: supported.** Observed: `{'construction_errors': {'state_relative': 1.7640331861912273e-06, 'shapley_relative': 4.451313202480378e-07, 'base_logit_abs': 4.00543212890625e-05}, 'maximum_head_error': 0.00067901611328125}`
- **R1_crossing_depth_generalizes: supported.** Observed: `{'differences': {'agreement_local': 1, 'agreement_attractor': 2, 'ioi': 0, 'capital_recall': 0, 'sentiment_classification': 0, 'negated_sentiment': 2, 'repeated_sequence': 2}}`
- **R2_material_attention_is_not_universal: supported.** Observed: `{'ratios': {'agreement_local': 0.0016240368630707313, 'agreement_attractor': 0.7028036180747446, 'ioi': 0.5922174133721045, 'capital_recall': 0.6057208498740594, 'sentiment_classification': 0.10415515471940877, 'negated_sentiment': 0.3669294155022117, 'repeated_sequence': 0.3575993922961609}}`
- **R3_material_attention_remains_compact: supported.** Observed: `{'families': ['agreement_attractor', 'ioi', 'capital_recall', 'sentiment_classification', 'negated_sentiment', 'repeated_sequence']}`
- **R4_discovery_rankings_beat_equal_size_subset_nulls: supported.** Observed: `{'families': ['agreement_attractor', 'ioi', 'capital_recall', 'sentiment_classification', 'negated_sentiment', 'repeated_sequence'], 'percentiles': {'agreement_attractor': 0.9880952380952381, 'ioi': 0.9880952380952381, 'capital_recall': 0.9880952380952381, 'sentiment_classification': 0.9880952380952381, 'negated_sentiment': 1.0, 'repeated_sequence': 1.0}}`
- **R5_small_family_panels_are_disclosed: supported.** Observed: `{'exploratory': ['capital_recall', 'sentiment_classification', 'negated_sentiment']}`

| Family | Cross D/H | Heldout n | Attention/native RMS | Material | Frozen top-3 fidelity | All-subset median / p90 | Percentile |
|---|---:|---:|---:|---|---:|---:|---:|
| agreement_local | 1/2 | 4 | 0.002 | no | 0.847 | -0.116 / 0.662 | 0.952 |
| agreement_attractor | 11/9 | 6 | 0.703 | yes | 1.000 | 0.023 / 1.000 | 0.988 |
| ioi | 14/14 | 6 | 0.592 | yes | 0.984 | 0.328 / 0.947 | 0.988 |
| capital_recall | 13/13 | 3 | 0.606 | yes | 0.998 | 0.105 / 0.996 | 0.988 |
| sentiment_classification | 9/9 | 3 | 0.104 | yes | 0.953 | 0.397 / 0.899 | 0.988 |
| negated_sentiment | 5/7 | 3 | 0.367 | yes | 0.896 | 0.453 / 0.780 | 1.000 |
| repeated_sequence | 8/10 | 4 | 0.358 | yes | 0.999 | 0.478 / 0.874 | 1.000 |

## Interpretation

A high subset percentile says the frozen head identity matters beyond merely choosing three of nine heads. It still describes one family-conditioned invocation at one layer. Families with fewer than four held-out pairs are exploratory, and a compact but immaterial branch is not counted as a task mechanism.
