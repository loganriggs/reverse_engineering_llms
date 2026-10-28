# Per-context native support and cancellation on natural upstream tangents

This asks whether the broad fixed channel bank is merely the union of sparse context-specific coalitions. Effective support is computed from contribution-vector norms and from absolute target projections. Fixed subsets keep the model's native coefficient of one. OMP is reported separately and may freely reweight selected channels, so it measures descriptive span rather than native execution.

## Locked hypotheses

- **S0_context_channel_sum_is_exact: supported.**
- **S1_contexts_have_sparse_norm_participation: not supported.**
- **S2_contexts_have_sparse_target_projection: not supported.**
- **S3_fixed_coefficient_context512_is_sufficient: not supported.**
- **S4_most_contexts_have_small_fixed_subsets: not supported.**
- **S5_contexts_use_substantial_cancellation: supported.**
- **S6_reweighted_span_is_compact: supported.**
- **S7_participation_support_is_ood_stable: supported.**

Maximum channel-sum error: `6.28e-07`.

| Corpus | Norm support p10 / med / p90 | Projection support med | Abs projection mass med | Fixed512 agg | Fixed K90 med | OMP K90 / K99 med |
|---|---:|---:|---:|---:|---:|---:|
| fineweb | 1228 / 1869 / 2136 | 866 | 4.84 | -1.033 | 4608 | 5 / 9 |
| wikitext | 1218 / 1894 / 2160 | 902 | 4.99 | -1.312 | 4608 | 5 / 9 |
| tinystories | 1093 / 1560 / 2092 | 634 | 4.55 | -0.771 | 4608 | 5 / 9 |

A small reweighted OMP span does not imply sparse native execution: in a 32D target space, arbitrary rescaling can synthesize a target from at most 32 independent vectors. The fixed-coefficient and participation statistics answer whether the model's actual channel contributions are sparse within a context.
