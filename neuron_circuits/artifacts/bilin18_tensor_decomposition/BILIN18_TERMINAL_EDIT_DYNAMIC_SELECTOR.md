# Dynamic terminal-head selector

Natural edits are analyzed example by example. A static family head, the largest routed rank-1 write, and the output-conditioned tensor score are compared with the actual finite per-example head winner. Complete-head effects measure whether the terminal coalition itself is sparse; rank-1 versus complete-head agreement tests whether weight outliers preserve the native head ordering.

## Locked results

- **D0_analytic_selector_matches_finite_rank1: supported.**
- **D1_terminal_positive_support_is_sparse: not supported.**
- **D2_one_family_head_explains_most_examples: not supported.**
- **D3_context_only_dynamic_selection_is_useful: not supported.**
- **D4_output_conditioned_dynamic_selection_is_near_oracle: supported.**
- **D5_weight_rank1_preserves_complete_head_order: not supported.**

| Family/split | static H | winner accuracy static/context/quotient | effect fraction static/context/quotient | full support/top share | rank1↔full r/winner |
|---|---:|---:|---:|---:|---:|
| spelling/0 | H2 | 0.863/0.850/0.988 | 0.997/0.998/1.000 | 1.35/0.857 | 0.988/0.887 |
| spelling/1 | H2 | 0.812/0.787/1.000 | 0.996/0.995/1.000 | 1.60/0.781 | 0.983/0.900 |
| unrelated/0 | H2 | 0.125/0.362/0.975 | 0.684/0.819/1.000 | 3.09/0.481 | 0.517/0.300 |
| unrelated/1 | H2 | 0.113/0.362/0.988 | 0.288/0.517/1.000 | 2.99/0.458 | 0.638/0.450 |

## Interpretation boundary

A quotient-conditioned selector is an executable explanation for a declared output change, not an autonomous semantic feature detector. Context-only success would indicate that occupancy and routing identify the invoked operation before looking downstream. Sparse complete-head support would favor a small conditional coalition; diffuse support would mean that the failure of static heads reflects genuine multi-head computation rather than merely changing winners.
