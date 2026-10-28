# Broad task competence: semantic-output red team

The original screen asked whether the intended answer beat a declared candidate set. This red team additionally requires the intended answer—or a preregistered semantic synonym—to rank near the top of the complete vocabulary. Candidate-only success is called contrast decodability, not operational competence.

## Locked results

- **R0_candidate_choice_overstates_operational_competence: supported.**
- **R1_semantic_output_sets_rescue_valid_classifications: supported.**
- **R2_exact_retrieval_requires_exact_output: supported.**
- **R3_breadth_survives_output_redteam: supported.**
- **R4_competent_results_beat_label_permutation: supported.**

| Family | Regime | Candidate gate | Semantic choice | Literal top-1 | Semantic top-1 | Semantic rank | Permutation p | Operational |
|---|---|---|---:|---:|---:|---:|---:|---|
| agreement_attractor | binding | yes | 1.000 | 0.545 | 0.589 | 1.0 | 0.0005 | yes |
| agreement_local | lexical | yes | 1.000 | 0.812 | 0.812 | 1.0 | 0.0005 | yes |
| arithmetic_add | algorithmic | no | 0.182 | 0.182 | 0.182 | 9.0 | 0.0055 | no |
| boolean_negation | formal | no | 0.250 | 0.000 | 0.000 | 105.0 | 1.0000 | no |
| bracket_instruction | formal | no | 0.611 | 0.000 | 0.000 | 132.0 | 0.0035 | no |
| capital_recall | factual_memory | yes | 1.000 | 0.900 | 0.900 | 1.0 | 0.0005 | yes |
| currency_recall | — | — | — | — | — | — | — | no |
| in_context_key_value | retrieval | yes | 0.875 | 0.188 | 0.188 | 3.5 | 0.0075 | no |
| ioi | binding | yes | 1.000 | 1.000 | 1.000 | 1.0 | 0.0005 | yes |
| list_maximum | algorithmic | no | 0.000 | 0.000 | 0.000 | 124.0 | 1.0000 | no |
| list_minimum | algorithmic | no | 0.250 | 0.000 | 0.000 | 65.0 | 0.0885 | no |
| month_successor | structured_recall | no | 0.083 | 0.000 | 0.000 | 116.0 | 0.6192 | no |
| negated_sentiment | composition | yes | 0.917 | 0.000 | 0.583 | 1.0 | 0.0070 | yes |
| repeated_bigram | retrieval | no | 0.250 | 0.125 | 0.125 | 6.0 | 0.2659 | no |
| repeated_sequence | retrieval | yes | 1.000 | 0.750 | 0.750 | 1.0 | 0.0005 | yes |
| sentiment_classification | classification | yes | 1.000 | 0.000 | 0.667 | 1.0 | 0.0015 | yes |
| weekday_successor | structured_recall | no | 0.286 | 0.000 | 0.000 | 36.0 | 0.2699 | no |

## Consequence

The next circuit phase may use operational families directly. Candidate-only families remain useful for studying latent contrasts, but cannot be described as tasks the model actually executes without an explicit output quotient.
