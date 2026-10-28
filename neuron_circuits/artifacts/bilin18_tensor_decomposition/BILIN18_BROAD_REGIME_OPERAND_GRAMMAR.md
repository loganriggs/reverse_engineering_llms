# Broad-regime operand grammar

Each frozen top-three head program is expanded into exact source-position tensor edges. Five disjoint source roles and a separate QK-routing/value factorial pass through the nonlinear suffix with Shapley allocation.

## Locked results

- **O0_tensor_edge_factorials_recompose: not supported.** Observed: `{'observed': {'state_max_abs': 0.0001220703125, 'role_shapley_relative': 1.0646226655808277e-07, 'qkv_shapley_relative': 6.686158826596511e-08}}`
- **O1_operand_profiles_transfer: supported.** Observed: `{'cosines': {'agreement_local': 0.5576254162883849, 'agreement_attractor': 0.9990991424118926, 'ioi': 0.9993771950905566, 'capital_recall': 0.9999995997392342, 'sentiment_classification': 0.9988643902183056, 'negated_sentiment': 0.9912309321016085, 'repeated_sequence': 0.9998798283850389}}`
- **O2_programs_use_relationally_relevant_sources: supported.** Observed: `{'shares': {'agreement_attractor': 0.6228958365329681, 'ioi': 0.9570103833893425, 'capital_recall': 1.0012398372511284, 'sentiment_classification': 0.10931143531777793, 'negated_sentiment': 0.36867169301930475, 'repeated_sequence': 0.9950338496859252}}`
- **O3_retrieval_uses_repetition_structure: supported.** Observed: `{'share': 0.9950338496859252}`
- **O4_binding_is_not_one_operand_type: supported.** Observed: `{'cosine': 0.8890140363537719, 'dominant_roles': {'ioi': 'edited_non_destination', 'agreement_attractor': 'edited_non_destination'}}`
- **O5_value_change_is_common_but_not_assumed_universal: supported.** Observed: `{'shares': {'agreement_attractor': 1.0123575379371257, 'ioi': 0.9415451819925912, 'capital_recall': 0.9984331747894969, 'sentiment_classification': 1.0260734779132354, 'negated_sentiment': 1.023925347938673, 'repeated_sequence': 0.980456261771741}}`

| Family | Site/program | Edited | Answer mention | Query repeat | Other | Destination | Routing | Value | D/H cosine |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| agreement_local | L1 H1,H4,H8 | 0.000 | 0.000 | 0.000 | 0.717 | 0.283 | 0.650 | 0.350 | 0.558 |
| agreement_attractor | L11 H3,H2,H0 | 0.623 | 0.000 | 0.000 | 0.357 | 0.020 | -0.012 | 1.012 | 0.999 |
| ioi | L14 H4,H6,H7 | 0.940 | 0.017 | 0.000 | 0.043 | 0.000 | 0.058 | 0.942 | 0.999 |
| capital_recall | L13 H0,H6,H5 | 1.001 | 0.000 | 0.000 | -0.001 | -0.000 | 0.002 | 0.998 | 1.000 |
| sentiment_classification | L9 H7,H1,H3 | 0.111 | 0.000 | -0.002 | 0.976 | -0.085 | -0.026 | 1.026 | 0.999 |
| negated_sentiment | L5 H7,H8,H4 | 0.359 | 0.000 | 0.009 | 0.638 | -0.007 | -0.024 | 1.024 | 0.991 |
| repeated_sequence | L8 H3,H4,H6 | 0.995 | 0.000 | -0.000 | 0.004 | 0.001 | 0.020 | 0.980 | 1.000 |

## Guardrail

These roles are relational partitions of the selected counterfactual prompts, not global semantic labels for a head. A large `other` or destination share can still be a coherent relay. Source-role profiles identify where the invocation reads; they do not by themselves prove the transported state has the proposed semantic meaning.
