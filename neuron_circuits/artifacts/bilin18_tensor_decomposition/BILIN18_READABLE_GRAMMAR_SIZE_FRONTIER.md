# Readable grammar size frontier

Readable routing grammars with 21, 32, and 42 leaf variables are fit independently on two profile panels. Every program still receives only 128 selected `(head, distance, variable)` predicates per layer, so increased variable vocabulary competes for the same execution budget.

Maximum native-branch error: `4.29e-05`. Human-21 recovery reference: `0.9674`.

## Locked hypotheses

- **L1_larger_readable_grammar_matches_human: not supported.**
- **L2_more_readable_variables_improve_behavior: not supported.**
- **L3_larger_readable_grammar_is_prediction_useful: not supported.**
- **L4_grammar_stability_persists: supported.**
- **L5_frontier_has_diminishing_returns: not supported.**
- **L6_readable_grammar_does_not_artificially_surpass_human: supported.**
- **native_branch_matches_checkpoint: supported.**

| Leaves | Split A recovery/top1 | Split B recovery/top1 | Assignment NMI |
|---:|---:|---:|---:|
| 21 | 0.9384/0.665 | 0.9130/0.607 | 0.989 |
| 32 | 0.7770/0.405 | 0.8083/0.438 | 0.983 |
| 42 | 0.7189/0.339 | 0.7064/0.331 | 0.986 |

## Guardrails

The route budget is fixed, but selecting among more possible types increases support-description cost. Trees optimize routing-profile variance rather than output KL, and split B can choose different predicates. This frontier tests whether readable variable vocabulary size was the main limitation; it does not optimize a full end-to-end MDL objective.
