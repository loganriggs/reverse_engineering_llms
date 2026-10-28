# Broad-regime construction and compact-head assay

Seven operationally screened families are traced with disjoint discovery and held-out minimal pairs. At each layer, the changed destination state is exactly decomposed into carried state, fresh attention, and realized MLP write; all eight coalitions pass through the untouched suffix. Crossing layers and head rankings are frozen on discovery pairs.

## Locked results

- **B0_broad_endpoint_coverage: supported.** Observed: `{'families': ['agreement_local', 'agreement_attractor', 'ioi', 'capital_recall', 'sentiment_classification', 'negated_sentiment', 'repeated_sequence']}`
- **B1_destination_variables_emerge: supported.** Observed: `{'families': ['agreement_local', 'agreement_attractor', 'ioi', 'capital_recall', 'sentiment_classification', 'negated_sentiment', 'repeated_sequence']}`
- **B2_construction_branches_are_not_universal: supported.** Observed: `{'dominant': {'agreement_local': 'mlp', 'agreement_attractor': 'attention', 'ioi': 'attention', 'capital_recall': 'attention', 'sentiment_classification': 'carried', 'negated_sentiment': 'attention', 'repeated_sequence': 'attention'}}`
- **B3_attention_pulses_often_have_compact_head_programs: supported.** Observed: `{'top3_fidelity': {'agreement_local': 0.8471798633037761, 'agreement_attractor': 0.9998372478353225, 'ioi': 0.9837747215240525, 'capital_recall': 0.9981365621749247, 'sentiment_classification': 0.9529640382849962, 'negated_sentiment': 0.8964049187528244, 'repeated_sequence': 0.9985649240746165}}`
- **B4_no_single_universal_head: supported.** Observed: `{'top_heads': {'agreement_local': 1, 'agreement_attractor': 3, 'ioi': 4, 'capital_recall': 0, 'sentiment_classification': 7, 'negated_sentiment': 7, 'repeated_sequence': 3}}`
- **B5_negation_is_not_assumed_to_be_plain_sentiment: supported.** Observed: `{'plain_site': 9, 'negated_site': 5, 'top3_overlap': 1}`

| Family | Regime | Pairs D/H | Site | Held destination | Dominant branch | Top head | Top-3 head fidelity | k90 |
|---|---|---:|---:|---:|---|---:|---:|---:|
| agreement_local | lexical | 4/4 | 1 | 0.4837219802993321 | mlp | H1 | 0.8471798633037761 | 2 |
| agreement_attractor | binding | 6/6 | 11 | 1.0581463405100617 | attention | H3 | 0.9998372478353225 | 1 |
| ioi | binding | 6/6 | 14 | 0.9360304698442352 | attention | H4 | 0.9837747215240525 | 2 |
| capital_recall | factual_memory | 2/3 | 13 | 0.9495473746992149 | attention | H0 | 0.9981365621749247 | 1 |
| sentiment_classification | classification | 3/3 | 9 | 0.6329902113979192 | carried | H7 | 0.9529640382849962 | 2 |
| negated_sentiment | composition | 3/3 | 5 | 0.36848134851118786 | attention | H7 | 0.8964049187528244 | 4 |
| repeated_sequence | retrieval | 4/4 | 8 | 0.45539457462845134 | attention | H3 | 0.9985649240746165 | 2 |

## Guardrail

A compact head program reconstructs the finite attention branch at one discovery-selected site. It does not show that the head has one global semantic meaning, and it does not imply attention dominates the complete task. Branch dominance, destination sufficiency, and head compactness are reported separately.
