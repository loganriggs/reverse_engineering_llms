# Can broad circuits be found without task labels?

The construction layer and semantic program were already frozen. This prospective red team asks whether the selected heads could instead have been found from an older label-blind weight atlas, static head scale, or the paired input computation with all answer labels hidden.

## Locked hypotheses

- **P0_exact_recomposition: supported.** Observed: `{'maximum_head_sum_abs': 0.00038909912109375}`
- **P1_frozen_weight_motifs_are_enriched: supported.** Observed: `{'selected_heads': 18, 'motif_members': 9, 'motif_member_fraction': 0.5, 'matched_member_mean': 5.33076, 'member_one_sided_p': 0.02045979540204598, 'component_pairs': 3, 'matched_component_pair_mean': 0.74942, 'component_pair_one_sided_p': 0.03915960840391596}`
- **P2_static_weight_scale_predicts_programs: supported.** Observed: `{'families_above_median': ['agreement_attractor', 'ioi', 'capital_recall', 'sentiment_classification', 'repeated_sequence'], 'mean': {'energy_fidelity': 0.8452693428236842, 'subset_percentile': 0.8154761904761904}}`
- **P3_input_conditioned_state_scale_predicts_programs: supported.** Observed: `{'passing_families': ['agreement_attractor', 'ioi', 'capital_recall', 'negated_sentiment', 'repeated_sequence'], 'mean': {'energy_fidelity': 0.952556679202058, 'subset_percentile': 0.9384920634920636}}`
- **P4_label_blind_output_scale_predicts_programs: supported.** Observed: `{'passing_families': ['agreement_attractor', 'ioi', 'capital_recall', 'negated_sentiment', 'repeated_sequence'], 'mean': {'energy_fidelity': 0.952792698017345, 'subset_percentile': 0.9404761904761904}}`
- **P5_output_scale_improves_on_state_scale: not supported.** Observed: `{'mean_fidelity_difference': 0.00023601881528700908}`
- **P6_semantic_selection_adds_information: not supported.** Observed: `{'mean_fidelity_advantage': 0.018904326934451765}`

| Family | Site | Static heads / fidelity / pct | State heads / fidelity / pct | Unlabeled-output heads / fidelity / pct | Semantic heads / fidelity / pct |
|---|---:|---|---|---|---|
| agreement_attractor | L11 | H2,H3,H6 / 1.000 / 0.905 | H2,H3,H6 / 1.000 / 0.905 | H2,H3,H6 / 1.000 / 0.905 | H0,H2,H3 / 1.000 / 0.988 |
| ioi | L14 | H3,H4,H6 / 0.956 / 0.952 | H4,H6,H7 / 0.984 / 0.988 | H1,H4,H6 / 0.986 / 1.000 | H4,H6,H7 / 0.984 / 0.988 |
| capital_recall | L13 | H0,H3,H5 / 0.998 / 0.976 | H0,H3,H6 / 0.999 / 1.000 | H0,H3,H6 / 0.999 / 1.000 | H0,H5,H6 / 0.998 / 0.988 |
| sentiment_classification | L9 | H1,H6,H7 / 0.919 / 0.940 | H2,H7,H8 / 0.842 / 0.750 | H2,H7,H8 / 0.842 / 0.750 | H1,H3,H7 / 0.953 / 0.988 |
| negated_sentiment | L5 | H3,H5,H8 / 0.226 / 0.179 | H1,H4,H7 / 0.892 / 0.988 | H1,H4,H7 / 0.892 / 0.988 | H4,H7,H8 / 0.896 / 1.000 |
| repeated_sequence | L8 | H1,H3,H4 / 0.973 / 0.940 | H3,H4,H6 / 0.999 / 1.000 | H3,H4,H6 / 0.999 / 1.000 | H3,H4,H6 / 0.999 / 1.000 |

## Meaning

Static weight scale asks which heads are generically large. Input-state scale sees which heads respond to the counterfactual pair but not what answer is desired. Unlabeled-output scale additionally sees which changed head writes survive the suffix into any vocabulary logits, but still receives no semantic contrast. Semantic-causal selection is the task-informed ceiling, not an unsupervised method.

A failed static predictor does not falsify weight-based operator discovery: invocation is input-conditioned. A successful unlabeled predictor finds behaviorally active heads, but semantic labeling and signed causal validation remain separate problems.
