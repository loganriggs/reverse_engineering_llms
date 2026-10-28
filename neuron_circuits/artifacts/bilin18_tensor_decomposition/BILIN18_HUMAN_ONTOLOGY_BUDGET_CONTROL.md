# Human ontology route-budget control

This post-result red team gives the original 21 human source categories the same 192/256 route budgets used by the larger readable grammars. It separates a better variable vocabulary from simply retaining more typed edges.

Maximum native-branch error: `5.03e-05`. Readable-42/256 recovery: `0.9539`.

## Locked hypotheses

- **R1_budget_alone_explains_scaled_grammar_gain: supported.**
- **R2_more_human_routes_improve_recovery: supported.**
- **R3_readable_42_adds_behavior_beyond_human_categories: not supported.**
- **R4_budget_matched_human_reaches_high_prediction_fidelity: supported.**
- **native_branch_matches_checkpoint: supported.**

| Human variables / routes | KL recovery | CE delta | Top-1 |
|---|---:|---:|---:|
| 21 / 128 | 0.9670 | +0.2176 | 0.774 |
| 21 / 192 | 0.9861 | +0.0844 | 0.846 |
| 21 / 256 | 0.9923 | +0.0445 | 0.893 |

## Guardrails

This is a post-result red team, not an independent preregistered discovery. Both ontologies use the same discovery/evaluation panels and route selector. A human win would show that the scaled grammar result does not establish novel variables; a readable-grammar win would still require targeted intervention examples before naming those extra distinctions as model features.
