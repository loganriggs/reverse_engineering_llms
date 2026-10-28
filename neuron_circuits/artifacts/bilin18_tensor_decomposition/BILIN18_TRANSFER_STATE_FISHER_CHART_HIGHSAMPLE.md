# Prospective suffix-Fisher tangent charts

For each native held-out context, score-function adjoints are propagated backward from the output distribution to four residual boundaries. The chart covariance sums probability-weighted gradient outer products across token positions. It uses only the native state, output distribution, and exact tensor VJPs—never candidate replacement tangents. Independent conditional-tail samples red-team sketch stability. Frozen charts are then used to project all 53 candidate tangents before exact forward propagation.

## Locked hypotheses

- **F1_fisher_chart_sketch_is_stable: not supported.**
- **F2_prospective_fisher_rank64_is_causally_useful: not supported.**
- **F3_fisher_chart_beats_static_vocabulary: supported.**
- **F4_prospective_fisher_rank128_is_sufficient: not supported.**
- **F5_fisher_chart_approaches_oracle_dynamic_chart: not supported.**
- **attention_jvp_forward_invariant: supported.**

## Causal chart frontier

| Target | Boundary | Sketch overlap | Fisher rank90 | Fisher r64/r128 | Static r64 | Oracle tangent r64 |
|---|---:|---:|---:|---:|---:|---:|
| L0H0 | 6 | 0.603 | 100 | 0.853 / 0.928 | 0.402 | 0.918 |
| L0H0 | 10 | 0.473 | 76 | 0.821 / 0.894 | 0.411 | 0.929 |
| L0H0 | 14 | 0.442 | 67 | 0.757 / 0.844 | 0.346 | 0.919 |
| L0H0 | 17 | 0.468 | 60 | 0.711 / 0.822 | 0.306 | 0.888 |
| L2H2 | 6 | 0.603 | 100 | 0.819 / 0.902 | 0.125 | 0.825 |
| L2H2 | 10 | 0.473 | 76 | 0.790 / 0.868 | 0.380 | 0.844 |
| L2H2 | 14 | 0.442 | 67 | 0.738 / 0.820 | 0.328 | 0.842 |
| L2H2 | 17 | 0.468 | 60 | 0.645 / 0.775 | 0.100 | 0.787 |
| L5H3 | 6 | 0.603 | 100 | 0.779 / 0.869 | 0.318 | 0.773 |
| L5H3 | 10 | 0.473 | 76 | 0.731 / 0.836 | 0.430 | 0.784 |
| L5H3 | 14 | 0.442 | 67 | 0.694 / 0.791 | 0.423 | 0.768 |
| L5H3 | 17 | 0.468 | 60 | 0.629 / 0.736 | 0.295 | 0.711 |

## Guardrails

This chart is prospective with respect to candidate operators but is conditioned on the complete native suffix and declared next-token Fisher metric. It is therefore appropriate for output-sensitive causal compression, not a universal semantic ontology. The covariance sums residual-axis gradient blocks and discards cross-position blocks; failure can mean that position-specific structure is essential rather than that Fisher weighting is unhelpful. Tail-sketch instability invalidates fine-grained chart interpretation even if aggregate causal performance is good.
