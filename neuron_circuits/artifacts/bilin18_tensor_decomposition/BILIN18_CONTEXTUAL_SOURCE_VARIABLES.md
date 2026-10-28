# Context-conditioned source variables

Static token classes cannot represent a token changing role with its causal prefix. This experiment crosses the 21 human token categories with three readable contextual predicates and compares them with static human categories and a matched random previous-token role. Typed routes are discovered once, then evaluated on two disjoint FineWeb panels.

Maximum native-branch error: `5.44e-05`.

## Locked hypotheses

- **C1_previous_role_beats_random_context: supported.**
- **C2_context_beats_static_human: not supported.**
- **C3_context_advantage_is_not_only_more_types: not supported.**
- **C4_context_program_is_prediction_useful: supported.**
- **native_branch_matches_checkpoint: supported.**

| Scheme | Categories | Recovery 128 A/B | Recovery 256 A/B | Top-1 256 A/B |
|---|---:|---:|---:|---:|
| human | 21 | 0.9685/0.9650 | 0.9931/0.9919 | 0.897/0.887 |
| previous_role | 84 | 0.9197/0.9108 | 0.9647/0.9620 | 0.771/0.754 |
| line_phase | 63 | 0.9477/0.9505 | 0.9811/0.9790 | 0.826/0.820 |
| recent_repeat | 42 | 0.9586/0.9515 | 0.9913/0.9885 | 0.872/0.872 |
| random_previous_role | 84 | 0.7356/0.7360 | 0.9196/0.9172 | 0.636/0.633 |

## Guardrails

The contextual predicates are hand-written probes, not a complete grammar. Crossing variables increases the type inventory and can fragment route evidence, so failure does not show that contextual state is absent. Conversely, a gain would show that the predicate predicts useful routing, not that it is a unique or causally atomic feature. The random control is matched in category count but not exactly in empirical group sizes.
