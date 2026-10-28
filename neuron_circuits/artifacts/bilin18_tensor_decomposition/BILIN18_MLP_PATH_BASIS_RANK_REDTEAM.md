# Endpoint-basis and rank red-team in the 546M model

The exact 16-path suffix program is repeated in natural PCA and Haar-random input/output charts at ranks 32 and 64. Each chart gets its own FineWeb discovery ordering; all evaluation remains disjoint and OOD corpora are untouched during selection.

## Locked hypotheses

- **B0_pca_rank64_remains_compact: supported.**
- **B1_random_rank32_remains_compact: supported.**
- **B2_random_rank64_remains_compact: supported.**
- **B3_every_chart_is_low_order: supported.**
- **B4_group_order_is_chart_stable: not supported.**
- **B5_natural_chart_is_privileged: supported.**
- **B6_endpoint_rank_is_stable: supported.**

| Chart | Top-4 overlap | FineWeb K4/K8/K90 | WikiText | TinyStories |
|---|---:|---:|---:|---:|
| pca32 | 4/4 | 0.961/0.990/2 | 0.964/0.990/2 | 0.971/0.995/2 |
| random32 | 2/4 | 0.883/0.971/5 | 0.883/0.970/5 | 0.926/0.987/4 |
| pca64 | 3/4 | 0.954/0.989/2 | 0.955/0.990/2 | 0.966/0.995/2 |
| random64 | 2/4 | 0.876/0.969/5 | 0.873/0.968/5 | 0.920/0.985/4 |

If random charts are equally compact, low interaction order is a property of the local residual operator chain rather than a semantic endpoint basis. If only PCA charts are compact, natural activity selects a privileged contractive slice. Rank stability tests whether the 32D endpoint hid broad off-chart propagation.
