# Readable grammar with proportional route budgets

Variable count and invocation budget are increased together: 21/128, 32/192, and 42/256. This tests whether larger readable grammars failed because they fragmented useful routes across more types.

Maximum native-branch error: `4.29e-05`. Human-21/128 recovery: `0.9674`.

## Locked hypotheses

- **B1_scaled_budget_closes_human_gap: supported.**
- **B2_scaled_budget_restores_prediction_fidelity: supported.**
- **B3_more_variables_help_at_proportional_budget: supported.**
- **B4_grammar_remains_stable: supported.**
- **B5_extra_variables_are_not_free: supported.**
- **native_branch_matches_checkpoint: supported.**

| Variables / predicates | Best scaled recovery | Best top-1 | Fixed-128 recovery | Assignment NMI |
|---|---:|---:|---:|---:|
| 21 / 128 | 0.9308 | 0.653 | 0.9384 | 0.989 |
| 32 / 192 | 0.9362 | 0.662 | 0.8083 | 0.983 |
| 42 / 256 | 0.9539 | 0.720 | 0.7189 | 0.986 |

## Guardrails

Budgets are approximately proportional rather than MDL-optimal. More selected predicates increase both support-description and execution cost. The trees still optimize profile variance, not final KL. A positive result would establish a variable/invocation tradeoff, not that larger grammars are universally preferable.
