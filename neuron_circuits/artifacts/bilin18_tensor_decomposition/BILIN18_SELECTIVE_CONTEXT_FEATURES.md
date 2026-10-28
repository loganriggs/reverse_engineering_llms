# Selective context-feature discovery

A fully crossed context vocabulary was informative but inefficient. Here, a split is nominated only when separating a human source category by a readable context increases the summed squared operator-effect signal, indicating cancellation in the unsplit route. Splits are ranked on discovery half A, checked on discovery half B, then frozen for causal evaluation.

Maximum native-branch error: `5.44e-05`.

## Locked hypotheses

- **S1_context_cancellation_is_concentrated: not supported.**
- **S2_selected_splits_are_stable: supported.**
- **S3_selective_context_beats_static_human: not supported.**
- **S4_selective_context_is_more_efficient_than_full_crossing: supported.**
- **native_branch_matches_checkpoint: supported.**

Top split candidates from discovery A:

- `other` by `line_phase`: gain 6.413e+01 (replicated-positive)
- `quote` by `recent_repeat`: gain 3.071e+01 (not replicated)
- `closing_delimiter` by `line_phase`: gain 4.172e-07 (not replicated)
- `tab` by `previous_role`: gain 0.000e+00 (not replicated)
- `question_exclamation` by `recent_repeat`: gain -1.118e-07 (replicated-positive)
- `slash` by `line_phase`: gain -1.907e-06 (not replicated)
- `colon_semicolon` by `line_phase`: gain -3.439e-05 (replicated-positive)
- `programming_keyword` by `line_phase`: gain -4.358e+01 (not replicated)
- `dash` by `line_phase`: gain -1.491e+02 (not replicated)
- `opening_delimiter` by `recent_repeat`: gain -4.132e+02 (not replicated)

| Selective grammar | Categories | Recovery 128 A/B | Recovery 256 A/B |
|---|---:|---:|---:|
| 4 splits | 26 | 0.9680/0.9644 | 0.9925/0.9906 |
| 8 splits | 26 | 0.9680/0.9644 | 0.9925/0.9906 |

## Guardrails

The split score is a native-state local operator sketch, while evaluation is closed-loop final-logit behavior; disagreement is expected. Selecting on one discovery half and validating ranking signs on another limits but does not eliminate researcher degrees of freedom. The predicate library is small, and an unsuccessful selective grammar would not falsify context-dependent latent variables outside this library.
