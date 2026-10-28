# Broad task competence screen for Bilin18

This is a behavioral eligibility screen for the FineWeb-only checkpoint. A family passes only if it has at least eight exactly tokenized examples, achieves at least halfway from candidate-set chance to perfect accuracy, and has positive median correct-versus-best-alternative margin. Failure is not mechanistic evidence.

## Locked gates

- **C0_tokenization_coverage: supported.**
- **C1_broad_competence: supported.**
- **C2_nonlexical_competence: supported.**
- **C3_intervention_ready: supported.**
- **C4_instruction_bias_is_not_assumed: supported.**

## Family results

| Family | Regime | Usable | Choices | Accuracy | Normalized | Median margin | Median vocab rank | Top-1 | CF pairs | Eligible |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| agreement_attractor | binding | 112 | 2 | 1.000 | 1.000 | 3.584 | 1.0 | 0.545 | 3136 | yes |
| agreement_local | lexical | 16 | 2 | 1.000 | 1.000 | 6.182 | 1.0 | 0.812 | 64 | yes |
| arithmetic_add | algorithmic | 55 | 10 | 0.182 | 0.091 | -3.242 | 9.0 | 0.182 | 1320 | no |
| boolean_negation | formal | 4 | 2 | 0.250 | -0.500 | -0.225 | 103.5 | 0.000 | 2 | no |
| bracket_instruction | formal | 18 | 3 | 0.611 | 0.417 | 1.987 | 132.0 | 0.000 | 34 | no |
| capital_recall | factual_memory | 10 | 10 | 1.000 | 1.000 | 5.772 | 1.0 | 0.900 | 45 | yes |
| currency_recall | factual_memory | 0 | 0 | 0.000 | 0.000 | — | — | — | 0 | no |
| in_context_key_value | retrieval | 16 | 2 | 0.875 | 0.750 | 1.273 | 3.5 | 0.188 | 8 | yes |
| ioi | binding | 56 | 2 | 1.000 | 1.000 | 3.047 | 1.0 | 1.000 | 28 | yes |
| list_maximum | algorithmic | 12 | 10 | 0.000 | -0.111 | -1.300 | 124.0 | 0.000 | 44 | no |
| list_minimum | algorithmic | 12 | 10 | 0.250 | 0.167 | -0.394 | 65.0 | 0.000 | 51 | no |
| month_successor | structured_recall | 12 | 12 | 0.083 | 0.000 | -0.741 | 116.0 | 0.000 | 66 | no |
| negated_sentiment | composition | 12 | 2 | 1.000 | 1.000 | 1.030 | 337.0 | 0.000 | 36 | yes |
| repeated_bigram | retrieval | 8 | 8 | 0.250 | 0.143 | -0.054 | 6.0 | 0.125 | 0 | no |
| repeated_sequence | retrieval | 8 | 8 | 1.000 | 1.000 | 1.645 | 1.0 | 0.750 | 0 | yes |
| sentiment_classification | classification | 12 | 2 | 1.000 | 1.000 | 1.619 | 196.5 | 0.000 | 36 | yes |
| weekday_successor | structured_recall | 7 | 7 | 0.286 | 0.167 | -0.149 | 36.0 | 0.000 | 21 | no |

## Interpretation rule

Only competent families are eligible for later circuit discovery. Candidate-set success without full-vocabulary top-1 is retained as graded evidence, but both quantities are reported. Instruction-like failures are prompt failures unless a naturalistic version of the same operation also fails.
