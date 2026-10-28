# Broad-program payload subspaces

Head identity failed the arbitrary-edit specificity test. This assay freezes each semantic-causal three-head backbone, learns rank-one and rank-two bases of its discovery payload vectors, and asks whether those directions transfer to held-out examples. Equal-rank bases from arbitrary tokens at the same positions and Gaussian bases are controls.

## Locked hypotheses

- **S0_exact_head_recomposition: supported.** Observed: `{'maximum_head_sum_abs': 0.00067901611328125}`
- **S1_rank2_true_payload_transfers: not supported.** Observed: `{'passing_families': ['agreement_attractor', 'sentiment_classification', 'negated_sentiment'], 'mean_program_output_fidelity': 0.630146337852595}`
- **S2_true_payload_beats_matched_arbitrary_edits: supported.** Observed: `{'true_mean': 0.630146337852595, 'random_edit_mean': 0.4243274960966163, 'difference': 0.2058188417559787}`
- **S3_true_payload_beats_gaussian_subspaces: supported.** Observed: `{'true_mean': 0.630146337852595, 'gaussian_mean': 0.00327571579730526, 'difference': 0.6268706220552898}`
- **S4_rank1_often_suffices: not supported.** Observed: `{'passing_families': ['agreement_attractor', 'negated_sentiment']}`
- **S5_payload_is_more_output_compact_than_state_compact: not supported.** Observed: `{'families': ['agreement_attractor', 'sentiment_classification']}`

| Family | Site/heads | Rank-1 output | Rank-2 state / output / all-attn | Random-edit output | Gaussian output | Best other-family output |
|---|---|---:|---:|---:|---:|---:|
| agreement_attractor | L11 H0,H2,H3 | 0.979 | 0.627 / 0.993 / 0.992 | 0.946 | 0.003 | 0.002 |
| ioi | L14 H4,H6,H7 | 0.041 | 0.230 / 0.412 / 0.364 | 0.119 | 0.005 | 0.009 |
| capital_recall | L13 H0,H5,H6 | 0.088 | 0.080 / 0.136 / 0.132 | 0.167 | 0.002 | 0.007 |
| sentiment_classification | L9 H1,H3,H7 | 0.052 | 0.731 / 0.968 / 0.880 | 0.397 | 0.003 | 0.435 |
| negated_sentiment | L5 H4,H7,H8 | 0.946 | 0.825 / 0.989 / 0.855 | 0.645 | 0.003 | 0.293 |
| repeated_sequence | L8 H3,H4,H6 | 0.105 | 0.119 / 0.283 / 0.283 | 0.272 | 0.004 | 0.025 |

## Interpretation boundary

A transferable payload span is an invocation coordinate system, not automatically a named concept. Its semantic reading additionally depends on the task-defined source relation and output quotient. Cross-family bases come from different depths and therefore test family-depth specificity rather than pure semantic identity.
