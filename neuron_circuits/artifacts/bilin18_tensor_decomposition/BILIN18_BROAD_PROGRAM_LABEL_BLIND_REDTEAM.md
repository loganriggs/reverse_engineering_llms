# Label-blind broad-program discovery: generic-magnitude red team

The positive label-blind result still receives a task-defined counterfactual pair and a semantic-discovery-selected layer. This audit asks whether generic active heads, mismatched sources from the same family, or arbitrary tokens at the same edit positions yield the same program.

## Locked hypotheses

- **R0_exact_recomposition: supported.** Observed: `{'maximum_head_sum_abs': 0.00060272216796875}`
- **R1_pair_delta_beats_native_activity: supported.** Observed: `{'true_mean_fidelity': 0.952556679202058, 'native_mean_fidelity': 0.8539859432386133, 'difference': 0.09857073596344468}`
- **R2_family_level_pairing_is_sufficient: supported.** Observed: `{'passing_families': ['agreement_attractor', 'ioi', 'capital_recall', 'repeated_sequence']}`
- **R3_arbitrary_matched_edits_do_not_explain_the_result: not supported.** Observed: `{'true_mean_fidelity': 0.952556679202058, 'random_mean_fidelity': 0.947690412113935, 'difference': 0.004866267088122989, 'random_passing_families': ['agreement_attractor', 'ioi', 'capital_recall', 'sentiment_classification', 'negated_sentiment', 'repeated_sequence']}`
- **R4_suffix_magnitude_adds_distinct_ranking_information: not supported.** Observed: `{'mean_top3_jaccard': 0.9166666666666666}`
- **R5_static_motif_enrichment_survives_two-test_correction: supported.** Observed: `{'bonferroni_p': 0.04091959080409196}`

| Family | True pair fidelity | Native activity fidelity | Cyclic sources fidelity / Jaccard | Random edits mean fidelity / pct / pass fraction |
|---|---:|---:|---:|---:|
| agreement_attractor | 1.000 | 1.000 | 1.000 / 1.000 | 1.000 / 0.905 / 1.000 |
| ioi | 0.984 | 0.984 | 0.984 / 1.000 | 0.984 / 0.988 / 1.000 |
| capital_recall | 0.999 | 0.999 | 0.999 / 1.000 | 0.999 / 1.000 / 1.000 |
| sentiment_classification | 0.842 | 0.348 | 0.779 / 0.500 | 0.893 / 0.871 / 0.875 |
| negated_sentiment | 0.892 | 0.794 | 0.876 / 0.500 | 0.876 / 0.976 / 1.000 |
| repeated_sequence | 0.999 | 0.999 | 0.999 / 1.000 | 0.934 / 0.929 / 1.000 |

## Guardrail

This remains a conditional circuit finder: the behavior family, counterfactual construction, and construction layer are supplied. It tests whether answer labels are needed to select heads, not whether the complete circuit can be discovered from raw text with no behavioral hypothesis. High performance from native or random-edit controls would weaken the claim by showing that generic head magnitude is enough.
