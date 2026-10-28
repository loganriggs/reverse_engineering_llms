# Context-conditioned tangent-state charts

This diagnostic asks whether the failed static vocabulary should be replaced by a rotating chart. For each held-out context independently, PCA is fit to that context's 53 candidate tangents across all token positions. Rank-64 and rank-128 projections are then propagated through the exact suffix. This is an oracle existence test, not a prospective feature extractor: the same context's tangent bundle defines its chart.

## Locked hypotheses

- **D1_per_context_rank64_is_locally_compact: not supported.**
- **D2_per_context_rank64_is_causally_useful: not supported.**
- **D3_dynamic_chart_beats_static_vocabulary: supported.**
- **D4_per_context_rank128_is_causally_sufficient: not supported.**
- **D5_context_charts_rotate: supported.**
- **attention_jvp_forward_invariant: supported.**

## Dynamic-chart frontier

| Target | Boundary | Rank90 | Chart overlap | r64 energy / Fisher | r128 energy / Fisher | Static r64 Fisher |
|---|---:|---:|---:|---:|---:|---:|
| L0H0 | 6 | 90 | 0.131 | 0.858 / 0.918 | 0.931 / 0.966 | 0.402 |
| L0H0 | 10 | 88 | 0.120 | 0.857 / 0.929 | 0.934 / 0.974 | 0.411 |
| L0H0 | 14 | 86 | 0.104 | 0.858 / 0.919 | 0.937 / 0.971 | 0.346 |
| L0H0 | 17 | 53 | 0.149 | 0.917 / 0.888 | 0.965 / 0.964 | 0.306 |
| L2H2 | 6 | 253 | 0.134 | 0.720 / 0.825 | 0.837 / 0.919 | 0.125 |
| L2H2 | 10 | 232 | 0.121 | 0.723 / 0.844 | 0.843 / 0.932 | 0.380 |
| L2H2 | 14 | 220 | 0.111 | 0.727 / 0.842 | 0.848 / 0.926 | 0.328 |
| L2H2 | 17 | 98 | 0.167 | 0.860 / 0.787 | 0.923 / 0.900 | 0.100 |
| L5H3 | 6 | 358 | 0.234 | 0.696 / 0.773 | 0.778 / 0.865 | 0.318 |
| L5H3 | 10 | 397 | 0.158 | 0.622 / 0.784 | 0.738 / 0.887 | 0.430 |
| L5H3 | 14 | 368 | 0.133 | 0.638 / 0.768 | 0.755 / 0.874 | 0.423 |
| L5H3 | 17 | 201 | 0.189 | 0.805 / 0.711 | 0.871 / 0.829 | 0.295 |

## Guardrails

A same-context PCA is an optimistic upper bound and has a large implicit description cost. It does not show that the model or an interpreter can cheaply generate the chart. Euclidean tangent variance is still not behaviorally optimized; failure at rank 128 would motivate a destination-Fisher chart rather than prove irreducibility. Low chart overlap can reflect near-degenerate spectra, so it is interpreted jointly with held-out/static performance.
